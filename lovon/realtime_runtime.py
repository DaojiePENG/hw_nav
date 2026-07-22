"""Latest-frame asynchronous runtime with a fail-safe fixed-rate interface."""

from __future__ import annotations

import copy
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


LOGGER = logging.getLogger(__name__)


@dataclass
class RuntimeSnapshot:
    sequence: int = 0
    processed_frames: int = 0
    dropped_frames: int = 0
    frame_timestamp: Optional[float] = None
    completed_at: Optional[float] = None
    inference_ms: float = 0.0
    state: Optional[Dict[str, Any]] = None
    motion: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    error: Optional[str] = None
    annotated: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class ControlTick:
    timestamp: float
    motion: List[float]
    reason: str
    snapshot: RuntimeSnapshot


class AsyncLovonRuntime:
    """Run perception in one worker while control reads the newest safe result.

    Only one pending frame is retained.  If perception is temporarily slower
    than the camera, old frames are replaced instead of building latency.  The
    control side returns zero whenever the source frame is too old, perception
    failed, or no target is currently tracked.
    """

    def __init__(
        self,
        agent: Any,
        max_target_age_sec: float = 0.40,
        annotate: bool = False,
        copy_frame: bool = False,
        clock: Callable[[], float] = time.monotonic,
        autostart: bool = True,
    ) -> None:
        if max_target_age_sec <= 0.0:
            raise ValueError("max_target_age_sec 必须大于 0")
        self.agent = agent
        self.max_target_age_sec = float(max_target_age_sec)
        self.annotate_enabled = bool(annotate)
        self.copy_frame = bool(copy_frame)
        self.clock = clock
        self._condition = threading.Condition()
        self._pending: Optional[Tuple[int, np.ndarray, str, float, int]] = None
        self._snapshot = RuntimeSnapshot()
        self._next_sequence = 1
        self._instruction_generation = 0
        self._latest_instruction: Optional[str] = None
        self._running = False
        self._worker: Optional[threading.Thread] = None
        if autostart:
            self.start()

    def start(self) -> None:
        with self._condition:
            if self._running:
                return
            self._running = True
            self._worker = threading.Thread(target=self._worker_loop, name="lovon-perception", daemon=True)
            self._worker.start()

    def submit_frame(
        self,
        frame: np.ndarray,
        instruction: str,
        timestamp: Optional[float] = None,
    ) -> int:
        if frame is None or not isinstance(frame, np.ndarray) or frame.ndim < 2:
            raise ValueError("frame 必须是 OpenCV numpy 图像")
        if not instruction or not instruction.strip():
            raise ValueError("人物语言指令不能为空")
        captured_at = self.clock() if timestamp is None else float(timestamp)
        frame_for_worker = frame.copy() if self.copy_frame else frame
        with self._condition:
            if not self._running:
                raise RuntimeError("AsyncLovonRuntime 尚未启动或已关闭")
            sequence = self._next_sequence
            self._next_sequence += 1
            normalized_instruction = " ".join(instruction.strip().split())
            if normalized_instruction != self._latest_instruction:
                self._latest_instruction = normalized_instruction
                self._instruction_generation += 1
                # A command for the previous person must not remain active while
                # the new language target is being acquired.
                self._snapshot.state = None
                self._snapshot.motion = [0.0, 0.0, 0.0]
                self._snapshot.frame_timestamp = None
                self._snapshot.error = None
            if self._pending is not None:
                self._snapshot.dropped_frames += 1
            self._pending = (
                sequence,
                frame_for_worker,
                normalized_instruction,
                captured_at,
                self._instruction_generation,
            )
            self._condition.notify()
            return sequence

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                while self._running and self._pending is None:
                    self._condition.wait()
                if not self._running and self._pending is None:
                    return
                pending = self._pending
                self._pending = None
            if pending is None:
                continue
            sequence, frame, instruction, captured_at, instruction_generation = pending
            started = self.clock()
            state: Optional[Dict[str, Any]] = None
            motion = [0.0, 0.0, 0.0]
            annotated = None
            error = None
            try:
                state, result = self.agent.run(frame, user_instruction=instruction)
                motion = [float(item) for item in result]
                if self.annotate_enabled:
                    annotated = self.agent.annotate(frame)
            except Exception as exc:  # fail closed; the caller receives STOP
                error = f"{type(exc).__name__}: {exc}"
                LOGGER.exception("Lovon perception failed; control output is forced to STOP")
            completed_at = self.clock()
            with self._condition:
                if instruction_generation != self._instruction_generation:
                    # The result belongs to the old target.  Never expose it to
                    # the fixed-rate controller, even briefly.
                    self._snapshot.processed_frames += 1
                    self._condition.notify_all()
                    continue
                self._snapshot.sequence = sequence
                self._snapshot.processed_frames += 1
                self._snapshot.frame_timestamp = captured_at
                self._snapshot.completed_at = completed_at
                self._snapshot.inference_ms = max(0.0, completed_at - started) * 1000.0
                self._snapshot.state = copy.deepcopy(state)
                self._snapshot.motion = motion
                self._snapshot.error = error
                self._snapshot.annotated = annotated
                self._condition.notify_all()

    def get_snapshot(self, copy_annotated: bool = False) -> RuntimeSnapshot:
        with self._condition:
            source = self._snapshot
            annotated = source.annotated
            if copy_annotated and annotated is not None:
                annotated = annotated.copy()
            return RuntimeSnapshot(
                sequence=source.sequence,
                processed_frames=source.processed_frames,
                dropped_frames=source.dropped_frames,
                frame_timestamp=source.frame_timestamp,
                completed_at=source.completed_at,
                inference_ms=source.inference_ms,
                state=copy.deepcopy(source.state),
                motion=list(source.motion),
                error=source.error,
                annotated=annotated,
            )

    def get_control_command(self, now: Optional[float] = None) -> Tuple[List[float], str, RuntimeSnapshot]:
        current_time = self.clock() if now is None else float(now)
        snapshot = self.get_snapshot(copy_annotated=False)
        if snapshot.error is not None:
            return [0.0, 0.0, 0.0], "perception_error", snapshot
        if snapshot.state is None or snapshot.frame_timestamp is None:
            return [0.0, 0.0, 0.0], "not_ready", snapshot
        age = current_time - snapshot.frame_timestamp
        if age < 0.0 or age > self.max_target_age_sec:
            return [0.0, 0.0, 0.0], "stale_target", snapshot
        if snapshot.state.get("predicted_object") != "person":
            return [0.0, 0.0, 0.0], "target_not_visible", snapshot
        if snapshot.state.get("search_state_in") != "tracking":
            return [0.0, 0.0, 0.0], "target_not_tracking", snapshot
        return list(snapshot.motion), "tracking", snapshot

    def wait_until_processed(self, sequence: int, timeout: float = 5.0) -> RuntimeSnapshot:
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._snapshot.sequence < sequence and self._running:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(f"等待 perception sequence={sequence} 超时")
                self._condition.wait(timeout=remaining)
        return self.get_snapshot()

    def close(self, timeout: float = 5.0) -> None:
        with self._condition:
            self._running = False
            self._pending = None
            self._condition.notify_all()
        worker = self._worker
        worker_stopped = True
        if worker is not None and worker.is_alive():
            worker.join(timeout=timeout)
            if worker.is_alive():
                worker_stopped = False
                LOGGER.warning("Perception worker did not stop within %.1fs", timeout)
        if worker_stopped:
            for component_name in ("detector", "matcher"):
                component = getattr(self.agent, component_name, None)
                if component is not None and hasattr(component, "release"):
                    component.release()

    def __enter__(self) -> "AsyncLovonRuntime":
        self.start()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()


class FixedRateControlPublisher:
    """Publish the runtime's safe command independently from camera reads."""

    def __init__(
        self,
        runtime: AsyncLovonRuntime,
        command: Callable[[List[float]], None],
        frequency_hz: float = 5.0,
        on_tick: Optional[Callable[[ControlTick], None]] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if frequency_hz <= 0.0:
            raise ValueError("frequency_hz 必须大于 0")
        self.runtime = runtime
        self.command = command
        self.frequency_hz = float(frequency_hz)
        self.period_sec = 1.0 / self.frequency_hz
        self.on_tick = on_tick
        self.clock = clock
        self.tick_count = 0
        self.error: Optional[BaseException] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="lovon-control", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        next_tick = self.clock()
        while not self._stop.is_set():
            now = self.clock()
            remaining = next_tick - now
            if remaining > 0.0:
                if self._stop.wait(remaining):
                    return
                now = self.clock()
            try:
                motion, reason, snapshot = self.runtime.get_control_command(now)
                self.command(motion)
                self.tick_count += 1
                if self.on_tick is not None:
                    self.on_tick(ControlTick(now, motion, reason, snapshot))
            except BaseException as exc:  # surfaced to the main thread
                self.error = exc
                LOGGER.exception("Fixed-rate control publisher failed")
                self._stop.set()
                return
            next_tick += self.period_sec
            if next_tick <= now:
                # Do not send a burst of stale catch-up commands after a stall.
                next_tick = now + self.period_sec

    def raise_if_failed(self) -> None:
        if self.error is not None:
            raise RuntimeError("固定频率控制线程失败") from self.error

    def close(self, timeout: float = 2.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
            if thread.is_alive():
                LOGGER.warning("Control publisher did not stop within %.1fs", timeout)


__all__ = ["AsyncLovonRuntime", "ControlTick", "FixedRateControlPublisher", "RuntimeSnapshot"]
