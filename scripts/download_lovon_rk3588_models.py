#!/usr/bin/env python3
"""Download and verify the exact Rockchip ONNX sources and CLIP tokenizer."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "models" / "lovon_pro_rk3588"
DEFAULT_MANIFEST = DEFAULT_OUTPUT / "model_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def download_file(info: Dict[str, Any], output_dir: Path, verify_only: bool) -> None:
    destination = output_dir / info["name"]
    if not destination.exists():
        if verify_only:
            raise FileNotFoundError(destination)
        partial = output_dir / f"{info['name']}.part"
        print(f"Downloading {info['url']} -> {destination}")
        with urllib.request.urlopen(info["url"]) as response, partial.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        partial.replace(destination)
    actual_size = destination.stat().st_size
    if actual_size != int(info["size_bytes"]):
        raise RuntimeError(f"文件大小不匹配：{destination}，期望 {info['size_bytes']}，实际 {actual_size}")
    actual_hash = sha256(destination)
    if actual_hash != info["sha256"]:
        raise RuntimeError(f"SHA256 不匹配：{destination}，期望 {info['sha256']}，实际 {actual_hash}")
    print(f"OK {destination.name}: {actual_hash}")


def download_tokenizer(info: Dict[str, Any], output_dir: Path, verify_only: bool) -> None:
    destination = output_dir / info["local_dir"]
    file_manifest = info["files"]
    required = tuple(file_manifest)
    if verify_only:
        missing = [name for name in required if not (destination / name).is_file()]
        if missing:
            raise FileNotFoundError(f"CLIP tokenizer 缺少文件：{missing}")
    else:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError("下载 tokenizer 需要 huggingface-hub") from exc
        snapshot_download(
            repo_id=info["repo_id"],
            revision=info["revision"],
            local_dir=destination,
            allow_patterns=[*required, "special_tokens_map.json"],
        )
    for name, expected in file_manifest.items():
        path = destination / name
        if path.stat().st_size != int(expected["size_bytes"]):
            raise RuntimeError(f"tokenizer 文件大小不匹配：{path}")
        actual = sha256(path)
        if actual != expected["sha256"]:
            raise RuntimeError(f"tokenizer SHA256 不匹配：{path}，期望 {expected['sha256']}，实际 {actual}")
    print(f"OK tokenizer: {destination} @ {info['revision']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--onnx-only", action="store_true")
    parser.add_argument("--tokenizer-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.onnx_only and args.tokenizer_only:
        raise ValueError("--onnx-only 与 --tokenizer-only 不能同时使用")
    manifest = load_manifest(args.manifest.resolve())
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.tokenizer_only:
        for info in manifest["onnx_models"]:
            download_file(info, output_dir, args.verify_only)
    if not args.onnx_only:
        download_tokenizer(manifest["tokenizer"], output_dir, args.verify_only)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
