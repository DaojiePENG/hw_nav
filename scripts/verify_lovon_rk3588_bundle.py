#!/usr/bin/env python3
"""Verify transferred RK3588 model files against their conversion report."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lovon.lovon_agent_pro import load_pro_config  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "lovon_agent_pro_rk3588.yaml")
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "models" / "lovon_pro_rk3588" / "conversion_report.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_pro_config(args.config)
    report_path = args.report.expanduser().resolve()
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("target") != "rk3588":
        raise RuntimeError(f"转换目标不是 rk3588：{report.get('target')!r}")
    manifest_path = report_path.parent / "model_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if report.get("pinned_source") != manifest.get("source"):
        raise RuntimeError("conversion_report 的 RKNN 源版本与 model_manifest 不一致")
    toolchain = report.get("toolchain", {})
    if toolchain.get("rknn-toolkit2") != "2.3.2" or toolchain.get("onnx") != "1.16.1":
        raise RuntimeError(f"转换工具链版本不符合固定基线：{toolchain}")
    expected_inputs = {item["name"].removesuffix(".onnx"): item["sha256"] for item in manifest["onnx_models"]}
    report_inputs = report.get("inputs", {})
    input_mapping = {
        "yolo11n": "yolo",
        "clip_images": "clip-image",
        "clip_text": "clip-text",
    }
    for manifest_name, report_name in input_mapping.items():
        if report_inputs.get(report_name) != expected_inputs[manifest_name]:
            raise RuntimeError(f"{report_name} ONNX 输入哈希不符合固定清单")
    if not report.get("calibration"):
        raise RuntimeError("conversion_report 没有记录 YOLO INT8 量化图片")
    configured: Dict[str, Path] = {
        "yolo": Path(config["detector"]["model"]),
        "clip-image": Path(config["matcher"]["image_model"]),
        "clip-text": Path(config["matcher"]["text_model"]),
    }
    for name, path in configured.items():
        resolved = path if path.is_absolute() else ROOT / path
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        output_report = report["outputs"][name]
        if output_report.get("load_rknn") != "ok":
            raise RuntimeError(f"{name} 转换后没有通过 load_rknn 复核")
        expected = output_report["sha256"]
        actual = sha256(resolved)
        if actual != expected:
            raise RuntimeError(f"{name} SHA256 不匹配：期望 {expected}，实际 {actual}")
        print(f"OK {name}: {resolved} ({actual})")
    tokenizer = Path(config["matcher"]["tokenizer"])
    tokenizer = tokenizer if tokenizer.is_absolute() else ROOT / tokenizer
    tokenizer_files = manifest["tokenizer"]["files"]
    missing = [name for name in tokenizer_files if not (tokenizer / name).is_file()]
    if missing:
        raise FileNotFoundError(f"CLIP tokenizer 缺少文件：{missing}")
    for name, expected in tokenizer_files.items():
        path = tokenizer / name
        if path.stat().st_size != int(expected["size_bytes"]) or sha256(path) != expected["sha256"]:
            raise RuntimeError(f"CLIP tokenizer 校验失败：{path}")
    print(f"OK tokenizer: {tokenizer}")
    print(f"PASS: RK3588 bundle matches {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
