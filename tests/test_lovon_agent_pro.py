from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lovon.lovon_agent_pro import (
    BBoxMotionController,
    apply_spatial_hint,
    build_prompt_variants,
    L2MMMotionController,
    LovonAgentPro,
    PersonDetection,
    StableTargetSelector,
    load_pro_config,
    normalize_target_description,
    parse_requested_speed,
    translate_common_zh_attributes,
)


def candidate(
    xyxy=(0.0, 0.0, 20.0, 80.0),
    confidence=0.8,
    track_id=None,
    semantic=0.2,
    appearance=(1.0, 0.0),
):
    return PersonDetection(
        xyxy=tuple(float(value) for value in xyxy),
        confidence=confidence,
        track_id=track_id,
        semantic_score=semantic,
        appearance=np.asarray(appearance, dtype=np.float32),
    )


def test_normalizes_chinese_and_english_instructions():
    description, prompt = normalize_target_description("请跟随那个穿红色上衣、背黑色双肩包的人")
    assert description == "穿红色上衣、背黑色双肩包的人"
    assert prompt == "一张穿红色上衣、背黑色双肩包的人的全身照片"

    description, prompt = normalize_target_description("Please follow the person wearing a blue hat")
    assert description == "wearing a blue hat"
    assert prompt == "a full-body photo of wearing a blue hat"

    description, _ = normalize_target_description("跟随穿红色上衣的人，速度为 0.2 米每秒")
    assert description == "穿红色上衣的人"

    description, _ = normalize_target_description("follow the person in red at speed of 0.2 m/s")
    assert description == "in red"


def test_builds_auxiliary_english_prompt_for_common_chinese_attributes():
    translated = translate_common_zh_attributes("最左边穿黑色T恤和蓝色牛仔裤的男人")
    assert translated == "leftmost man wearing black T-shirt and blue jeans"
    prompts = build_prompt_variants("最左边穿黑色T恤和蓝色牛仔裤的男人")
    assert prompts[0].startswith("一张")
    assert prompts[1] == "a full-body photo of leftmost man wearing black T-shirt and blue jeans"
    assert prompts[2] == "a full-body photo of man wearing black T-shirt and blue jeans"


def test_applies_explicit_spatial_hint_without_using_crop_content():
    people = [
        candidate(xyxy=(10, 0, 30, 80)),
        candidate(xyxy=(80, 0, 120, 80)),
        candidate(xyxy=(170, 0, 190, 80)),
    ]
    assert apply_spatial_hint(people, (100, 200, 3), "最左边的人") == "left"
    assert [item.spatial_score for item in people] == pytest.approx([1.0, 0.5, 0.0])

    assert apply_spatial_hint(people, (100, 200, 3), "the person in the center") == "center"
    assert people[1].spatial_score == pytest.approx(1.0)


def test_speed_parser_defaults_and_caps_requested_speed():
    assert parse_requested_speed("跟随穿红衣服的人", 0.25, 0.4) == pytest.approx(0.25)
    assert parse_requested_speed("以速度 0.3 米每秒跟随他", 0.25, 0.4) == pytest.approx(0.3)
    assert parse_requested_speed("follow him at speed of 1.2 m/s", 0.25, 0.4) == pytest.approx(0.4)


def test_bbox_controller_steers_slows_and_stops():
    controller = BBoxMotionController(
        default_speed=0.3,
        max_linear_speed=0.4,
        stop_bbox_width=0.6,
        slow_down_bbox_width=0.4,
        center_deadband=0.05,
        turn_in_place_error=0.3,
        yaw_gain=2.0,
        max_yaw_speed=1.0,
        angular_sign=-1.0,
    )
    state = {"object_xyn": [0.7, 0.5], "object_whn": [0.2, 0.8]}
    status, motion = controller.predict(state, True, "跟随目标，速度为 0.3 米每秒")
    assert status == "running"
    assert 0.0 < motion[0] < 0.3
    assert motion[1] == 0.0
    assert motion[2] < 0.0

    state = {"object_xyn": [0.5, 0.5], "object_whn": [0.5, 0.9]}
    _, slowed = controller.predict(state, True, "follow at speed of 0.3 m/s")
    assert slowed[0] == pytest.approx(0.15)
    assert slowed[2] == 0.0

    state["object_whn"][0] = 0.6
    status, stopped = controller.predict(state, True, "follow")
    assert status == "success"
    assert stopped == [0.0, 0.0, 0.0]

    status, lost = controller.predict(state, False, "follow")
    assert status == "searching"
    assert lost == [0.0, 0.0, 0.0]


def test_selector_uses_semantics_not_detector_confidence_then_preserves_identity():
    selector = StableTargetSelector(acquire_score_threshold=0.1, acquire_margin=0.01)
    high_detector_wrong = candidate(
        xyxy=(0, 0, 30, 100), confidence=0.99, track_id=10, semantic=0.12, appearance=(0, 1)
    )
    described_person = candidate(
        xyxy=(60, 0, 100, 100), confidence=0.55, track_id=20, semantic=0.80, appearance=(1, 0)
    )
    selected = selector.select([high_detector_wrong, described_person])
    assert selected is described_person
    assert selector.target_track_id == 20
    assert selector.reason == "semantic_acquired"

    # The wrong person now has a better text score, but tracker ID and appearance
    # keep the original identity locked.
    wrong_next = candidate(
        xyxy=(25, 0, 58, 100), confidence=0.99, track_id=10, semantic=0.95, appearance=(0, 1)
    )
    target_next = candidate(
        xyxy=(55, 0, 96, 100), confidence=0.50, track_id=20, semantic=0.20, appearance=(0.99, 0.01)
    )
    selected = selector.select([wrong_next, target_next])
    assert selected is target_next
    assert selector.reason == "identity_associated"


def test_selector_refuses_ambiguous_people_and_marks_occlusion():
    selector = StableTargetSelector(acquire_score_threshold=0.1, acquire_margin=0.05, max_missed_frames=1)
    assert selector.select([
        candidate(semantic=0.50, confidence=0.8),
        candidate(xyxy=(40, 0, 70, 80), semantic=0.48, confidence=0.8, appearance=(0, 1)),
    ]) is None
    assert selector.reason == "ambiguous_candidates"
    assert selector.status == "searching"

    selector = StableTargetSelector(acquire_score_threshold=0.1, acquire_margin=0.0, max_missed_frames=1)
    assert selector.select([candidate(track_id=3, semantic=0.8)]) is not None
    assert selector.select([]) is None
    assert selector.status == "lost"
    assert selector.select([]) is None
    assert selector.status == "searching"


def test_selector_does_not_switch_to_different_appearance_at_same_location():
    selector = StableTargetSelector(acquire_score_threshold=0.1, acquire_margin=0.0)
    assert selector.select([candidate(track_id=7, semantic=0.8, appearance=(1, 0))]) is not None
    intruder = candidate(
        xyxy=(0, 0, 20, 80),
        track_id=8,
        semantic=0.95,
        appearance=(0, 1),
    )
    assert selector.select([intruder]) is None
    assert selector.status == "lost"
    assert selector.reason == "association_gate_failed"


class SequenceDetector:
    def __init__(self, frames):
        self.frames = list(frames)
        self.index = 0

    def detect(self, _image):
        frame = self.frames[min(self.index, len(self.frames) - 1)]
        self.index += 1
        return [
            PersonDetection(xyxy=item["xyxy"], confidence=item["confidence"], track_id=item["track_id"])
            for item in frame
        ]


class PositionMatcher:
    def score(self, _image, detections, prompt):
        prompts = [prompt] if isinstance(prompt, str) else prompt
        target_right = any("红" in item or "red" in item for item in prompts)
        scores = []
        embeddings = []
        for detection in detections:
            is_right = detection.xyxy[0] >= 50
            matched = is_right == target_right
            scores.append(0.85 if matched else 0.15)
            embeddings.append(np.array([1.0, 0.0]) if is_right else np.array([0.0, 1.0]))
        return scores, embeddings


def test_agent_end_to_end_selects_described_bbox_and_stops_when_lost():
    detector = SequenceDetector(
        [
            [
                {"xyxy": (5, 5, 45, 95), "confidence": 0.99, "track_id": 1},
                {"xyxy": (120, 5, 180, 95), "confidence": 0.55, "track_id": 2},
            ],
            [],
        ]
    )
    config = {
        "selector": {"acquire_margin": 0.01},
        "controller": {"stop_bbox_width": 0.7, "slow_down_bbox_width": 0.5},
    }
    agent = LovonAgentPro(config=config, detector=detector, matcher=PositionMatcher())
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    state, motion = agent.run(image, user_instruction="跟随穿红色衣服的人")
    assert state["predicted_object"] == "person"
    assert state["target_track_id"] == 2
    assert state["confidence"] == [0.55]
    assert state["object_xyn"][0] == pytest.approx(0.75)
    assert motion[2] < 0.0

    state, motion = agent.run(image)
    assert state["predicted_object"] == "NULL"
    assert state["search_state_in"] == "lost"
    assert motion == [0.0, 0.0, 0.0]


def test_instruction_change_resets_lock_and_can_select_a_new_person():
    detections = [
        {"xyxy": (5, 5, 45, 95), "confidence": 0.8, "track_id": 1},
        {"xyxy": (120, 5, 180, 95), "confidence": 0.8, "track_id": 2},
    ]
    detector = SequenceDetector([detections, detections])
    agent = LovonAgentPro(detector=detector, matcher=PositionMatcher())
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    first, _ = agent.run(image, user_instruction="跟随穿红衣服的人")
    second, _ = agent.run(image, user_instruction="follow the person in blue")
    assert first["target_track_id"] == 2
    assert second["target_track_id"] == 1
    assert second["selector_reason"] == "semantic_acquired"


class FakePredictor:
    def __init__(self):
        self.last_input = None

    def predict(self, data):
        self.last_input = data
        return {"predicted_state": "running", "motion_vector": [0.2, 0.0, -0.1]}


def test_l2mm_adapter_receives_only_selected_person_geometry_and_stops_if_absent():
    predictor = FakePredictor()
    controller = L2MMMotionController("unused", "unused", velocity_scale=2.0, predictor=predictor)
    state = {
        "predicted_object": "person",
        "confidence": [0.7],
        "object_xyn": [0.7, 0.5],
        "object_whn": [0.2, 0.8],
        "mission_state_in": "running",
        "search_state_in": "tracking",
    }
    status, vector = controller.predict(state, True, "跟随红衣人，速度为 0.3 米每秒")
    assert status == "running"
    assert vector == pytest.approx([0.4, 0.0, -0.2])
    assert "person" in predictor.last_input["mission_instruction_1"]
    assert "0.30" in predictor.last_input["mission_instruction_1"]

    predictor.last_input = None
    status, vector = controller.predict(state, False, "follow")
    assert status == "searching"
    assert vector == [0.0, 0.0, 0.0]
    assert predictor.last_input is None


def test_repository_config_loads_and_keeps_safe_defaults():
    root = Path(__file__).resolve().parents[1]
    config = load_pro_config(root / "configs" / "lovon_agent_pro.yaml")
    assert config["controller"]["backend"] == "bbox"
    assert config["controller"]["search_angular_speed"] == 0.0
    assert config["detector"]["use_bytetrack"] is True
