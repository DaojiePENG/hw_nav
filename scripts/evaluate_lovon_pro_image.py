#!/usr/bin/env python3
"""Evaluate multiple language instructions on one multi-person image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lovon.lovon_agent_pro import LovonAgentPro  # noqa: E402


DEFAULT_INSTRUCTIONS = [
    "跟随最右边穿橙色碎花连衣裙的金发女人",
    "跟随最左边穿黑色T恤和蓝色牛仔裤的男人",
    "跟随中间穿蓝白花纹上衣和黑色裤子的黑人男性",
]
DEFAULT_CENTER_X_RANGES = [(0.70, 1.00), (0.00, 0.30), (0.45, 0.70)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=ROOT / "artifacts" / "multi-people.jpg")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "lovon_agent_pro_offline.yaml")
    parser.add_argument(
        "--instruction",
        action="append",
        dest="instructions",
        help="可重复传入；未传时使用脚本内置的三条验收指令",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "lovon_pro_evaluation")
    parser.add_argument("--no-assert", action="store_true", help="不校验内置三条指令的预期横向位置")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    using_default_instructions = not args.instructions
    instructions = args.instructions or DEFAULT_INSTRUCTIONS
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.image)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    agent = LovonAgentPro.from_config_file(args.config)
    report = []
    for index, instruction in enumerate(instructions, start=1):
        state, motion = agent.run(image, user_instruction=instruction)
        output_path = args.output_dir / f"instruction_{index}.jpg"
        if not cv2.imwrite(str(output_path), agent.annotate(image)):
            raise RuntimeError(f"无法保存 {output_path}")
        candidates = []
        for candidate in agent.last_candidates:
            candidates.append(
                {
                    "track_id": candidate.track_id,
                    "bbox_xyxy": [round(value, 2) for value in candidate.xyxy],
                    "detector_confidence": round(candidate.confidence, 4),
                    "text_match_score": round(candidate.semantic_score, 4),
                    "spatial_score": None if candidate.spatial_score is None else round(candidate.spatial_score, 4),
                    "association_score": round(candidate.association_score, 4),
                    "selected": candidate is agent.last_selected,
                }
            )
        report.append(
            {
                "instruction": instruction,
                "state": state,
                "motion_vector": motion,
                "candidates": candidates,
                "annotated_image": str(output_path),
            }
        )
    report_path = args.output_dir / "report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if using_default_instructions and not args.no_assert:
        failures = []
        for index, (item, expected_range) in enumerate(zip(report, DEFAULT_CENTER_X_RANGES), start=1):
            state = item["state"]
            center_x = float(state["object_xyn"][0])
            if state["predicted_object"] != "person" or not expected_range[0] <= center_x <= expected_range[1]:
                failures.append(
                    f"instruction {index}: predicted={state['predicted_object']} center_x={center_x:.3f}, "
                    f"expected {expected_range}"
                )
        if failures:
            raise RuntimeError("多人图验收失败：" + "; ".join(failures))
        print("PASS: all three default instructions selected the expected person region")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
