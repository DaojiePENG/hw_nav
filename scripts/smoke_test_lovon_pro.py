#!/usr/bin/env python3
"""One-frame LovonAgentPro smoke test with either mocks or real local models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lovon.lovon_agent_pro import LovonAgentPro, PersonDetection  # noqa: E402


class MockDetector:
    def detect(self, image):
        height, width = image.shape[:2]
        return [
            PersonDetection(
                xyxy=(width * 0.30, height * 0.05, width * 0.70, height * 0.95),
                confidence=0.90,
                track_id=1,
            )
        ]


class MockMatcher:
    def score(self, _image, detections, _prompt):
        return [0.9] * len(detections), [np.array([1.0, 0.0], dtype=np.float32)] * len(detections)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true", help="不用下载模型，只验证管线")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "lovon_agent_pro.yaml")
    parser.add_argument("--image", type=Path, default=ROOT / "lovon" / "person.png")
    parser.add_argument("--instruction", default="跟随戴眼镜、穿白色衬衫、打黑色领带的男人")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "lovon_pro_smoke.jpg")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.image)
    dependencies = {"detector": MockDetector(), "matcher": MockMatcher()} if args.mock else {}
    agent = LovonAgentPro.from_config_file(args.config, **dependencies)
    state, motion = agent.run(image, user_instruction=args.instruction)
    annotated = agent.annotate(image)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), annotated):
        raise RuntimeError(f"无法写入 {args.output}")
    print(json.dumps({"state": state, "motion_vector": motion, "output": str(args.output)}, ensure_ascii=False, indent=2))
    if args.mock and state["predicted_object"] != "person":
        raise RuntimeError("mock smoke test did not select the person")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
