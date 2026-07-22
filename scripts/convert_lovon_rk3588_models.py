#!/usr/bin/env python3
"""Convert the pinned Rockchip YOLO11/CLIP ONNX files to RK3588 RKNN."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "models" / "lovon_pro_rk3588"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def checked(operation: str, return_code: Any) -> None:
    if return_code not in (None, 0):
        raise RuntimeError(f"RKNN {operation} 失败，错误码：{return_code}")


def with_rknn(action: Callable[[Any], None]) -> None:
    try:
        from rknn.api import RKNN
    except ImportError as exc:
        raise RuntimeError(
            "转换只能在 x86_64 的 rknn-toolkit2 环境执行；请先运行 scripts/create_lovon_rk3588_env.sh convert"
        ) from exc
    rknn = RKNN(verbose=False)
    try:
        action(rknn)
    finally:
        rknn.release()


def validate_rknn(path: Path) -> None:
    try:
        from rknn.api import RKNN
    except ImportError as exc:
        raise RuntimeError("无法导入 rknn-toolkit2 做产物复核") from exc
    rknn = RKNN(verbose=False)
    try:
        checked(f"load_rknn({path.name})", rknn.load_rknn(str(path)))
    finally:
        rknn.release()


def convert_yolo(source: Path, output: Path, dataset: Path, target: str) -> None:
    def action(rknn: Any) -> None:
        checked(
            "config(yolo11)",
            rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform=target),
        )
        checked("load_onnx(yolo11)", rknn.load_onnx(model=str(source)))
        checked("build(yolo11 INT8)", rknn.build(do_quantization=True, dataset=str(dataset)))
        checked("export_rknn(yolo11)", rknn.export_rknn(str(output)))

    with_rknn(action)


def convert_clip_image(source: Path, output: Path, target: str) -> None:
    def action(rknn: Any) -> None:
        checked(
            "config(clip image)",
            rknn.config(
                target_platform=target,
                mean_values=[[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255]],
                std_values=[[0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255]],
            ),
        )
        checked(
            "load_onnx(clip image)",
            rknn.load_onnx(
                model=str(source),
                inputs=["pixel_values"],
                input_size_list=[[1, 3, 224, 224]],
            ),
        )
        checked("build(clip image FP16)", rknn.build(do_quantization=False))
        checked("export_rknn(clip image)", rknn.export_rknn(str(output)))

    with_rknn(action)


def convert_clip_text(source: Path, output: Path, target: str) -> None:
    def action(rknn: Any) -> None:
        checked("config(clip text)", rknn.config(target_platform=target))
        checked(
            "load_onnx(clip text)",
            rknn.load_onnx(
                model=str(source),
                inputs=["input_ids"],
                input_size_list=[[1, 20]],
            ),
        )
        checked("build(clip text FP16)", rknn.build(do_quantization=False))
        checked("export_rknn(clip text)", rknn.export_rknn(str(output)))

    with_rknn(action)


def write_calibration_list(model_dir: Path, images: List[Path]) -> Path:
    if not images:
        raise ValueError("YOLO11 INT8 转换至少需要一张 --calibration-image")
    resolved = []
    for image in images:
        path = image.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        resolved.append(str(path))
    dataset = model_dir / "yolo_calibration.txt"
    dataset.write_text("\n".join(resolved) + "\n", encoding="utf-8")
    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--target", default="rk3588")
    parser.add_argument("--only", choices=("all", "yolo", "clip-image", "clip-text"), default="all")
    parser.add_argument(
        "--calibration-image",
        type=Path,
        action="append",
        help="可重复传入；缺省用仓库多人测试图。量产应换成至少 100 张现场图片。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    sources: Dict[str, Path] = {
        "yolo": model_dir / "yolo11n.onnx",
        "clip-image": model_dir / "clip_images.onnx",
        "clip-text": model_dir / "clip_text.onnx",
    }
    requested = list(sources) if args.only == "all" else [args.only]
    for name in requested:
        if not sources[name].is_file():
            raise FileNotFoundError(f"缺少 {sources[name]}；请先运行下载脚本")
    outputs = {
        "yolo": model_dir / "yolo11n_i8.rknn",
        "clip-image": model_dir / "clip_images_fp16.rknn",
        "clip-text": model_dir / "clip_text_fp16.rknn",
    }
    calibration_images: List[Path] = []
    if "yolo" in requested:
        calibration_images = args.calibration_image or [ROOT / "artifacts" / "multi-people.jpg"]
        dataset = write_calibration_list(model_dir, calibration_images)
        convert_yolo(sources["yolo"], outputs["yolo"], dataset, args.target)
    if "clip-image" in requested:
        convert_clip_image(sources["clip-image"], outputs["clip-image"], args.target)
    if "clip-text" in requested:
        convert_clip_text(sources["clip-text"], outputs["clip-text"], args.target)

    existing_outputs = {name: path for name, path in outputs.items() if path.is_file()}
    for output in existing_outputs.values():
        validate_rknn(output)

    manifest_path = model_dir / "model_manifest.json"
    pinned_source: Dict[str, Any] = {}
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as handle:
            pinned_source = json.load(handle).get("source", {})

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": args.target,
        "pinned_source": pinned_source,
        "toolchain": {
            "rknn-toolkit2": package_version("rknn-toolkit2"),
            "onnx": package_version("onnx"),
            "numpy": package_version("numpy"),
            "setuptools": package_version("setuptools"),
        },
        "inputs": {name: sha256(path) for name, path in sources.items() if path.is_file()},
        "calibration": [
            {"path": str(path.expanduser().resolve()), "sha256": sha256(path.expanduser().resolve())}
            for path in calibration_images
        ],
        "outputs": {
            name: {"sha256": sha256(path), "size_bytes": path.stat().st_size, "load_rknn": "ok"}
            for name, path in existing_outputs.items()
        },
    }
    report_path = model_dir / "conversion_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Conversion report: {report_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
