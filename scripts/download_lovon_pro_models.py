#!/usr/bin/env python3
"""Download the exact detector and matcher revisions used by LovonAgentPro."""

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
DEFAULT_OUTPUT = ROOT / "models" / "lovon_pro"
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


def download_detector(manifest: Dict[str, Any], output_dir: Path, verify_only: bool) -> Path:
    info = manifest["detector"]
    destination = output_dir / info["name"]
    expected = info["sha256"]
    if not destination.exists():
        if verify_only:
            raise FileNotFoundError(destination)
        partial = destination.with_suffix(destination.suffix + ".part")
        print(f"Downloading {info['url']} -> {destination}")
        with urllib.request.urlopen(info["url"]) as response, partial.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        partial.replace(destination)
    actual = sha256(destination)
    if actual != expected:
        raise RuntimeError(f"YOLO SHA256 mismatch: expected {expected}, got {actual}")
    if destination.stat().st_size != int(info["size_bytes"]):
        raise RuntimeError(f"YOLO file size mismatch: {destination}")
    print(f"OK detector: {destination} ({actual})")
    return destination


def download_matcher(manifest: Dict[str, Any], output_dir: Path, verify_only: bool) -> Path:
    info = manifest["matcher"]
    destination = output_dir / info["local_dir"]
    required = ("config.json", "model.safetensors", "preprocessor_config.json", "tokenizer.json")
    if verify_only:
        missing = [name for name in required if not (destination / name).is_file()]
        if missing:
            raise FileNotFoundError(f"SigLIP2 缺少文件：{missing}")
    else:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError("请先安装 requirements-lovon-pro.txt") from exc
        print(f"Downloading {info['repo_id']}@{info['revision']} -> {destination}")
        snapshot_download(
            repo_id=info["repo_id"],
            revision=info["revision"],
            local_dir=destination,
        )
    print(f"OK matcher: {destination} @ {info['revision']}")
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--detector-only", action="store_true")
    parser.add_argument("--matcher-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.detector_only and args.matcher_only:
        raise ValueError("--detector-only 与 --matcher-only 不能同时使用")
    manifest = load_manifest(args.manifest.resolve())
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.matcher_only:
        download_detector(manifest, output_dir, args.verify_only)
    if not args.detector_only:
        download_matcher(manifest, output_dir, args.verify_only)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
