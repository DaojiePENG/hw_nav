#!/usr/bin/env python3
"""Benchmark acquisition, locked inference, and 5 Hz control on a real RK3588."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lovon.lovon_agent_pro import LovonAgentPro  # noqa: E402
from lovon.realtime_runtime import AsyncLovonRuntime, FixedRateControlPublisher  # noqa: E402


def percentile(values: List[float], quantile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def summary(values: List[float]) -> Dict[str, float]:
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values),
        "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
        "max_ms": max(values),
        "effective_fps": 1000.0 / statistics.fmean(values),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "lovon_agent_pro_rk3588.yaml")
    parser.add_argument("--image", type=Path, default=ROOT / "artifacts" / "multi-people.jpg")
    parser.add_argument("--instruction", default="跟随最右边穿橙色碎花连衣裙的金发女人")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--required-control-hz", type=float, default=5.0)
    parser.add_argument("--control-duration", type=float, default=5.0, help="异步控制节拍实测时长（秒）")
    parser.add_argument("--camera-hz", type=float, default=20.0, help="静态图片提交频率")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "rk3588_benchmark.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if platform.machine() != "aarch64":
        raise RuntimeError("该命令必须在 RK3588 aarch64 板端执行")
    if (
        args.iterations < 5
        or args.required_control_hz <= 0.0
        or args.control_duration < 2.0
        or args.camera_hz <= args.required_control_hz
    ):
        raise ValueError("iterations 至少为 5，控制测试至少 2 秒，camera-hz 必须高于 required-control-hz")
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.image)
    agent = LovonAgentPro.from_config_file(args.config)
    if agent.config["detector"].get("backend") != "rknn_yolo11":
        raise ValueError("benchmark 必须使用 detector.backend=rknn_yolo11")
    try:
        # Cold acquisition includes text encoding and every candidate crop.
        started = time.perf_counter()
        acquired_state, _motion = agent.run(image, user_instruction=args.instruction)
        acquisition_ms = (time.perf_counter() - started) * 1000.0
        if acquired_state["predicted_object"] != "person":
            raise RuntimeError(f"目标获取失败：{acquired_state['selector_reason']}")

        # Keep CLIP out of the locked path.  This is the production scheduling
        # invariant that makes a 5 Hz controller independent of candidate count.
        configured_refresh_interval = agent.refresh_interval_sec
        agent.refresh_interval_sec = float("inf")
        locked_latencies = []
        matcher_calls = 0
        for _ in range(args.iterations):
            started = time.perf_counter()
            state, _motion = agent.run(image)
            locked_latencies.append((time.perf_counter() - started) * 1000.0)
            matcher_calls += int(state["matcher_ran"])
            if state["predicted_object"] != "person":
                raise RuntimeError(f"锁定路径丢失目标：{state['selector_reason']}")
        locked = summary(locked_latencies)
        budget_ms = 1000.0 / args.required_control_hz
        # The asynchronous phase restores the production refresh interval so
        # it also exercises periodic one-crop CLIP work during control output.
        agent.refresh_interval_sec = configured_refresh_interval

        tick_times: List[float] = []
        tick_reasons: List[str] = []
        runtime = AsyncLovonRuntime(agent, max_target_age_sec=agent.config["runtime"]["max_target_age_sec"])
        publisher = FixedRateControlPublisher(
            runtime,
            command=lambda _motion: None,
            frequency_hz=args.required_control_hz,
            on_tick=lambda tick: (tick_times.append(tick.timestamp), tick_reasons.append(tick.reason)),
        )
        try:
            publisher.start()
            deadline = time.monotonic() + args.control_duration
            next_frame = time.monotonic()
            while time.monotonic() < deadline:
                now = time.monotonic()
                if now >= next_frame:
                    runtime.submit_frame(image, args.instruction, timestamp=now)
                    next_frame += 1.0 / args.camera_hz
                    if next_frame <= now:
                        next_frame = now + 1.0 / args.camera_hz
                time.sleep(0.001)
            publisher.raise_if_failed()
        finally:
            publisher.close()
            runtime_snapshot = runtime.get_snapshot()
            runtime.close()
        intervals_ms = [value * 1000.0 for value in np.diff(tick_times).tolist()]
        cadence = summary(intervals_ms) if intervals_ms else {}
        first_tracking = next((index for index, reason in enumerate(tick_reasons) if reason == "tracking"), len(tick_reasons))
        steady_reasons = tick_reasons[first_tracking:]
        tracking_ticks = sum(reason == "tracking" for reason in steady_reasons)
        tracking_ratio = tracking_ticks / len(steady_reasons) if steady_reasons else 0.0
        measured_hz = 1000.0 / cadence["mean_ms"] if cadence else 0.0
        cadence_pass = (
            bool(cadence)
            and cadence["p95_ms"] <= budget_ms * 1.25
            and measured_hz >= args.required_control_hz * 0.95
            and tracking_ratio >= 0.95
        )
        passed = locked["p95_ms"] <= budget_ms and matcher_calls == 0 and cadence_pass
        report = {
            "platform": platform.platform(),
            "config": str(args.config.resolve()),
            "image": str(args.image.resolve()),
            "instruction": args.instruction,
            "candidate_count": acquired_state["candidate_count"],
            "acquisition_ms": acquisition_ms,
            "locked_path": locked,
            "locked_matcher_calls": matcher_calls,
            "configured_refresh_interval_sec": configured_refresh_interval,
            "required_control_hz": args.required_control_hz,
            "budget_ms": budget_ms,
            "async_control": {
                "duration_sec": args.control_duration,
                "camera_hz": args.camera_hz,
                "tick_count": len(tick_times),
                "interval": cadence,
                "measured_hz": measured_hz,
                "tracking_ratio_after_first_lock": tracking_ratio,
                "reason_counts": {reason: tick_reasons.count(reason) for reason in sorted(set(tick_reasons))},
                "processed_frames": runtime_snapshot.processed_frames,
                "dropped_frames": runtime_snapshot.dropped_frames,
                "pass": cadence_pass,
            },
            "pass": passed,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not passed:
            print(
                f"FAIL: locked p95 <= {budget_ms:.1f} ms, zero locked CLIP calls, and async cadence gate are required",
                file=sys.stderr,
            )
            return 1
        print(
            f"PASS: locked inference and asynchronous control sustain "
            f"{args.required_control_hz:.1f} Hz"
        )
        return 0
    finally:
        for component_name in ("detector", "matcher"):
            component = getattr(agent, component_name, None)
            if component is not None and hasattr(component, "release"):
                component.release()


if __name__ == "__main__":
    raise SystemExit(main())
