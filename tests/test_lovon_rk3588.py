from __future__ import annotations

import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest

from lovon.lovon_agent_pro import LovonAgentPro, PersonDetection, load_pro_config
from lovon.realtime_runtime import AsyncLovonRuntime, FixedRateControlPublisher
from lovon.rknn_backend import (
    GreedyIoUTracker,
    RknnClipPersonMatcher,
    RknnLiteModel,
    RknnYolo11PersonDetector,
)


class FakeClock:
    def __init__(self, value=0.0):
        self.value = float(value)

    def __call__(self):
        return self.value


def test_repository_rk3588_config_builds_lazy_rknn_backends_without_npu():
    root = Path(__file__).resolve().parents[1]
    config = load_pro_config(root / "configs" / "lovon_agent_pro_rk3588.yaml")
    agent = LovonAgentPro(config=config)
    assert isinstance(agent.detector, RknnYolo11PersonDetector)
    assert isinstance(agent.matcher, RknnClipPersonMatcher)
    assert agent.matcher_policy == "event_driven"
    assert agent.config["runtime"]["async_perception"] is True


def test_rknn_lite_auto_uses_runtime_auto_core(monkeypatch, tmp_path):
    calls = {}

    class FakeRKNNLite:
        NPU_CORE_AUTO = 0
        NPU_CORE_0 = 1
        NPU_CORE_1 = 2
        NPU_CORE_2 = 4
        NPU_CORE_0_1_2 = 7
        NPU_CORE_ALL = 0xFFFF

        def __init__(self, verbose):
            calls["verbose"] = verbose

        def load_rknn(self, path):
            calls["path"] = path
            return 0

        def init_runtime(self, **kwargs):
            calls["init"] = kwargs
            return 0

        def inference(self, inputs):
            return [np.asarray(inputs[0])]

        def release(self):
            calls["released"] = True

    package = types.ModuleType("rknnlite")
    api = types.ModuleType("rknnlite.api")
    api.RKNNLite = FakeRKNNLite
    monkeypatch.setitem(sys.modules, "rknnlite", package)
    monkeypatch.setitem(sys.modules, "rknnlite.api", api)
    model_path = tmp_path / "model.rknn"
    model_path.write_bytes(b"test")

    model = RknnLiteModel(str(model_path), core_mask="auto")
    assert model.infer([np.ones((1,), dtype=np.uint8)])[0].tolist() == [1]
    assert calls["init"] == {"core_mask": FakeRKNNLite.NPU_CORE_AUTO}
    model.release()
    assert calls["released"] is True


class SamePeopleDetector:
    def detect(self, _image):
        return [
            PersonDetection((10.0, 5.0, 50.0, 95.0), 0.9, track_id=1),
            PersonDetection((120.0, 5.0, 180.0, 95.0), 0.8, track_id=2),
        ]


class CountingMatcher:
    def __init__(self):
        self.calls = 0
        self.candidate_counts = []

    def score(self, _image, detections, _prompt):
        self.calls += 1
        self.candidate_counts.append(len(detections))
        scores = [0.2 if item.track_id == 1 else 0.9 for item in detections]
        embeddings = [
            np.array([0.0, 1.0], dtype=np.float32) if item.track_id == 1 else np.array([1.0, 0.0], dtype=np.float32)
            for item in detections
        ]
        return scores, embeddings


def test_event_driven_matcher_is_removed_from_locked_control_path():
    clock = FakeClock()
    matcher = CountingMatcher()
    agent = LovonAgentPro(
        config={
            "scheduling": {
                "matcher_policy": "event_driven",
                "search_interval_sec": 0.5,
                "refresh_interval_sec": 2.0,
            },
            "selector": {"acquire_margin": 0.0},
        },
        detector=SamePeopleDetector(),
        matcher=matcher,
        clock=clock,
    )
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    first, _ = agent.run(image, user_instruction="跟随穿红衣服的人")
    assert first["target_track_id"] == 2
    assert first["matcher_ran"] is True
    assert matcher.calls == 1
    assert matcher.candidate_counts == [2]

    # Ten 5 Hz control periods fit here, yet only YOLO/tracking runs.
    for step in range(1, 10):
        clock.value = step * 0.2
        state, _ = agent.run(image)
        assert state["target_track_id"] == 2
        assert state["matcher_ran"] is False
        assert state["matcher_reason"] == "tracking_cache"
    assert matcher.calls == 1

    clock.value = 2.01
    refreshed, _ = agent.run(image)
    assert refreshed["matcher_ran"] is True
    assert refreshed["matcher_reason"] == "periodic_refresh"
    assert refreshed["matcher_candidate_count"] == 1
    assert matcher.calls == 2
    assert matcher.candidate_counts == [2, 1]


def test_instruction_change_forces_new_match_even_during_throttle_window():
    clock = FakeClock()
    matcher = CountingMatcher()
    agent = LovonAgentPro(
        config={"scheduling": {"matcher_policy": "event_driven", "search_interval_sec": 10.0}},
        detector=SamePeopleDetector(),
        matcher=matcher,
        clock=clock,
    )
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    agent.run(image, user_instruction="跟随红衣人")
    clock.value = 0.1
    state, _ = agent.run(image, user_instruction="follow the person in blue")
    assert state["matcher_ran"] is True
    assert matcher.calls == 2


class RejectingRefreshMatcher(CountingMatcher):
    def score(self, image, detections, prompt):
        scores, embeddings = super().score(image, detections, prompt)
        if self.calls == 2:
            return [0.0 for _ in scores], embeddings
        return scores, embeddings


def test_failed_periodic_verification_stops_and_forces_full_reacquisition():
    clock = FakeClock()
    matcher = RejectingRefreshMatcher()
    agent = LovonAgentPro(
        config={
            "scheduling": {
                "matcher_policy": "event_driven",
                "search_interval_sec": 0.5,
                "refresh_interval_sec": 2.0,
            },
            "selector": {"acquire_margin": 0.0},
        },
        detector=SamePeopleDetector(),
        matcher=matcher,
        clock=clock,
    )
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    acquired, _ = agent.run(image, user_instruction="follow the person in red")
    assert acquired["target_track_id"] == 2

    clock.value = 2.01
    rejected, motion = agent.run(image)
    assert rejected["predicted_object"] == "NULL"
    assert rejected["matcher_reason"] == "periodic_rejected_semantic"
    assert rejected["selector_reason"] == "periodic_verification_failed"
    assert motion == [0.0, 0.0, 0.0]

    # Rejection resets the matcher deadline, so reacquisition is immediate and
    # evaluates every person rather than trusting the rejected Track ID.
    clock.value = 2.02
    reacquired, _ = agent.run(image)
    assert reacquired["predicted_object"] == "person"
    assert reacquired["matcher_reason"] == "target_id_missing"
    assert matcher.candidate_counts == [2, 1, 2]


def test_greedy_tracker_preserves_ids_and_does_not_reuse_removed_track():
    tracker = GreedyIoUTracker(iou_threshold=0.2, max_missed=0)
    first = [PersonDetection((10, 10, 30, 80), 0.9), PersonDetection((100, 10, 130, 80), 0.8)]
    tracker.update(first)
    assert [item.track_id for item in first] == [1, 2]
    moved = [PersonDetection((13, 10, 33, 80), 0.9), PersonDetection((103, 10, 133, 80), 0.8)]
    tracker.update(moved)
    assert [item.track_id for item in moved] == [1, 2]
    tracker.update([])
    newcomer = [PersonDetection((13, 10, 33, 80), 0.9)]
    tracker.update(newcomer)
    assert newcomer[0].track_id == 3


class StaticRknnRuntime:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = 0

    def inference(self, inputs):
        assert inputs
        self.calls += 1
        return self.outputs

    def release(self):
        pass


def yolo_outputs_with_one_person():
    outputs = []
    for branch in range(3):
        position = np.full((1, 64, 1, 1), -10.0, dtype=np.float32)
        for side in range(4):
            position[0, side * 16 + 1, 0, 0] = 10.0
        classes = np.zeros((1, 80, 1, 1), dtype=np.float32)
        if branch == 0:
            classes[0, 0, 0, 0] = 0.9
        outputs.extend((position, classes))
    return outputs


def test_rknn_yolo11_decodes_official_six_output_layout_and_tracks():
    runtime = StaticRknnRuntime(yolo_outputs_with_one_person())
    detector = RknnYolo11PersonDetector(
        "unused.rknn",
        confidence=0.3,
        image_size=640,
        runtime=runtime,
    )
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    first = detector.detect(image)
    second = detector.detect(image)
    assert len(first) == 1
    assert first[0].track_id == second[0].track_id == 1
    assert first[0].confidence == pytest.approx(0.9)
    assert first[0].xyxy[0] == pytest.approx(0.0)
    assert first[0].xyxy[2] == pytest.approx(639.0)

    nhwc_runtime = StaticRknnRuntime(
        [tensor.transpose(0, 2, 3, 1) for tensor in yolo_outputs_with_one_person()]
    )
    nhwc_detector = RknnYolo11PersonDetector("unused.rknn", runtime=nhwc_runtime)
    assert len(nhwc_detector.detect(image)) == 1


class FakeTokenizer:
    def __init__(self):
        self.calls = 0

    def __call__(self, prompts, padding, return_tensors):
        assert padding is True and return_tensors == "np"
        self.calls += 1
        return {"input_ids": np.array([[49406, 10, 49407] for _ in prompts], dtype=np.int64)}


class SequenceEmbeddingRuntime:
    def __init__(self, embeddings):
        self.embeddings = [np.asarray(item, dtype=np.float32) for item in embeddings]
        self.calls = 0

    def inference(self, inputs):
        output = self.embeddings[self.calls % len(self.embeddings)]
        self.calls += 1
        return [output.reshape(1, -1)]

    def release(self):
        pass


def test_rknn_clip_uses_english_variant_and_caches_text_embedding():
    tokenizer = FakeTokenizer()
    text_runtime = SequenceEmbeddingRuntime([[1.0, 0.0]])
    image_runtime = SequenceEmbeddingRuntime([[1.0, 0.0], [0.0, 1.0]])
    matcher = RknnClipPersonMatcher(
        "image.rknn",
        "text.rknn",
        "tokenizer",
        image_runtime=image_runtime,
        text_runtime=text_runtime,
        tokenizer=tokenizer,
    )
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = [
        PersonDetection((0, 0, 40, 90), 0.9, track_id=1),
        PersonDetection((50, 0, 90, 90), 0.9, track_id=2),
    ]
    prompts = ["一张穿红衣服的人的全身照片", "a full-body photo of a person wearing red"]
    scores, embeddings = matcher.score(image, detections, prompts)
    matcher.score(image, detections, prompts)
    assert scores == pytest.approx([1.0, 0.5])
    assert len(embeddings) == 2
    assert tokenizer.calls == 1
    assert text_runtime.calls == 1
    assert image_runtime.calls == 4


class ImmediateAgent:
    def run(self, _frame, user_instruction):
        assert user_instruction
        return {
            "predicted_object": "person",
            "search_state_in": "tracking",
        }, [0.2, 0.0, -0.1]


def test_async_runtime_stops_on_stale_bbox_and_allows_fresh_5hz_ticks():
    runtime = AsyncLovonRuntime(ImmediateAgent(), max_target_age_sec=0.4)
    try:
        captured = time.monotonic()
        sequence = runtime.submit_frame(np.zeros((10, 10, 3), dtype=np.uint8), "follow", timestamp=captured)
        runtime.wait_until_processed(sequence)
        for tick in (0.0, 0.2, 0.39):
            motion, reason, _ = runtime.get_control_command(now=captured + tick)
            assert reason == "tracking"
            assert motion == [0.2, 0.0, -0.1]
        motion, reason, _ = runtime.get_control_command(now=captured + 0.401)
        assert reason == "stale_target"
        assert motion == [0.0, 0.0, 0.0]
    finally:
        runtime.close()


def test_fixed_rate_publisher_runs_independently_of_camera_reads():
    runtime = AsyncLovonRuntime(ImmediateAgent(), max_target_age_sec=1.0)
    commands = []
    ticks = []
    publisher = None
    try:
        sequence = runtime.submit_frame(np.zeros((10, 10, 3), dtype=np.uint8), "follow")
        runtime.wait_until_processed(sequence)
        publisher = FixedRateControlPublisher(
            runtime,
            commands.append,
            frequency_hz=10.0,
            on_tick=lambda tick: ticks.append(tick.timestamp),
        )
        publisher.start()
        time.sleep(0.24)
        publisher.raise_if_failed()
        assert len(commands) >= 3
        assert all(command == [0.2, 0.0, -0.1] for command in commands)
        intervals = np.diff(ticks[:3])
        assert intervals == pytest.approx([0.1, 0.1], abs=0.035)
    finally:
        if publisher is not None:
            publisher.close()
        runtime.close()


class BlockingAgent(ImmediateAgent):
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def run(self, frame, user_instruction):
        self.started.set()
        self.release.wait(timeout=2.0)
        return super().run(frame, user_instruction)


class InstructionRaceAgent:
    def __init__(self):
        self.old_started = threading.Event()
        self.release_old = threading.Event()
        self.new_started = threading.Event()
        self.release_new = threading.Event()

    def run(self, _frame, user_instruction):
        if user_instruction == "old target":
            self.old_started.set()
            self.release_old.wait(timeout=2.0)
        else:
            self.new_started.set()
            self.release_new.wait(timeout=2.0)
        return {
            "predicted_object": "person",
            "search_state_in": "tracking",
            "target_instruction": user_instruction,
        }, [0.2, 0.0, -0.1]


def test_instruction_change_immediately_invalidates_inflight_old_motion():
    agent = InstructionRaceAgent()
    runtime = AsyncLovonRuntime(agent)
    try:
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        runtime.submit_frame(frame, "old target")
        assert agent.old_started.wait(timeout=1.0)
        newest = runtime.submit_frame(frame, "new target")
        motion, reason, _ = runtime.get_control_command()
        assert reason == "not_ready"
        assert motion == [0.0, 0.0, 0.0]

        agent.release_old.set()
        assert agent.new_started.wait(timeout=1.0)
        # Completion of the old inference is discarded while the new target is
        # still being acquired.
        motion, reason, _ = runtime.get_control_command()
        assert reason == "not_ready"
        assert motion == [0.0, 0.0, 0.0]

        agent.release_new.set()
        snapshot = runtime.wait_until_processed(newest)
        assert snapshot.state["target_instruction"] == "new target"
    finally:
        agent.release_old.set()
        agent.release_new.set()
        runtime.close()


def test_async_runtime_drops_queued_old_frames_instead_of_accumulating_latency():
    agent = BlockingAgent()
    runtime = AsyncLovonRuntime(agent)
    try:
        runtime.submit_frame(np.zeros((10, 10, 3), dtype=np.uint8), "follow")
        assert agent.started.wait(timeout=1.0)
        runtime.submit_frame(np.ones((10, 10, 3), dtype=np.uint8), "follow")
        newest = runtime.submit_frame(np.full((10, 10, 3), 2, dtype=np.uint8), "follow")
        agent.release.set()
        snapshot = runtime.wait_until_processed(newest)
        assert snapshot.sequence == newest
        assert snapshot.processed_frames == 2
        assert snapshot.dropped_frames == 1
    finally:
        agent.release.set()
        runtime.close()
