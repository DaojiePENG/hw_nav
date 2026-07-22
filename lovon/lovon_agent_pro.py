"""Language-conditioned person following for LOVON.

This module is intentionally independent from :mod:`lovon.lovon_agent`.  The
original agent imports private l2mm assets at module import time; keeping this
implementation separate means a clean clone can run the deterministic bbox
controller even when those assets are not available.

The public API mirrors ``LovonAgent.run`` and returns ``(state, motion_vector)``.
Heavy dependencies (Ultralytics and Transformers) are imported lazily so unit
tests and configuration validation do not download or initialize any models.
"""

from __future__ import annotations

import copy
import importlib
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np


LOGGER = logging.getLogger(__name__)
BBox = Tuple[float, float, float, float]


DEFAULT_CONFIG: Dict[str, Any] = {
    "runtime": {
        "backend": "torch",
        "device": "auto",
        "half_precision": True,
        "async_perception": False,
        "control_hz": 5.0,
        "max_target_age_sec": 0.40,
    },
    "detector": {
        "backend": "ultralytics",
        "model": "yolo11n.pt",
        "confidence": 0.30,
        "iou": 0.55,
        "image_size": 640,
        "use_bytetrack": True,
        "tracker": "bytetrack.yaml",
    },
    "matcher": {
        "backend": "siglip",
        "model": "google/siglip2-base-patch16-224",
        "revision": "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
        "local_files_only": False,
        "crop_padding": 0.05,
        "prompt_template_zh": "一张{description}的全身照片",
        "prompt_template_other": "a full-body photo of {description}",
    },
    "scheduling": {
        # ``always`` preserves the original x86 behavior.  RK3588 uses
        # ``event_driven`` so the expensive vision-language model is not on
        # the control loop's critical path.
        "matcher_policy": "always",
        "search_interval_sec": 0.0,
        "refresh_interval_sec": 0.0,
    },
    "selector": {
        "acquire_score_threshold": 0.10,
        "acquire_margin": 0.015,
        "acquire_semantic_weight": 0.75,
        "acquire_spatial_weight": 0.15,
        "acquire_detector_weight": 0.10,
        "max_missed_frames": 12,
        "min_iou": 0.08,
        "min_appearance_similarity": 0.72,
        "min_association_score": 0.30,
        "track_id_weight": 0.30,
        "iou_weight": 0.25,
        "appearance_weight": 0.35,
        "semantic_weight": 0.10,
        "embedding_momentum": 0.85,
    },
    "controller": {
        "backend": "bbox",
        "default_speed": 0.25,
        "max_linear_speed": 0.40,
        "stop_bbox_width": 0.62,
        "slow_down_bbox_width": 0.38,
        "center_deadband": 0.06,
        "turn_in_place_error": 0.30,
        "yaw_gain": 1.8,
        "max_yaw_speed": 1.0,
        "angular_sign": -1.0,
        "search_angular_speed": 0.0,
        "wh_scale_factor": 1.0,
        "velocity_scale": 1.0,
        "l2mm_tokenizer_path": None,
        "l2mm_model_path": None,
    },
}


def _deep_update(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            _deep_update(base[key], value)  # type: ignore[index]
        else:
            base[key] = value
    return base


def load_pro_config(path: Optional[os.PathLike[str] | str] = None) -> Dict[str, Any]:
    """Load a YAML config and merge it over safe defaults.

    ``PyYAML`` is imported only when a path is supplied.  Unknown top-level
    sections are retained, while required sections keep their default values.
    """

    config = copy.deepcopy(DEFAULT_CONFIG)
    if path is None:
        return config
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - actionable environment error
        raise RuntimeError("读取 YAML 配置需要 PyYAML：pip install pyyaml") from exc
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, Mapping):
        raise ValueError(f"配置文件顶层必须是映射：{config_path}")
    _deep_update(config, loaded)
    return config


def normalize_target_description(instruction: str) -> Tuple[str, str]:
    """Turn a follow command into a compact target description and VLM prompt.

    SigLIP2 is multilingual, so this function does not translate or reduce the
    instruction to a fixed attribute vocabulary.  It only removes common motion
    verbs that describe the robot rather than the person.
    """

    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("人物语言指令不能为空")
    text = " ".join(instruction.strip().split())
    is_zh = bool(re.search(r"[\u3400-\u9fff]", text))
    if is_zh:
        description = re.sub(r"^(请|请你|麻烦|机器人|小车|帮我)+", "", text).strip(" ，,。")
        description = re.sub(
            r"^(去)?(跟随|跟着|追踪|追随|锁定|寻找|找到|找出)+",
            "",
            description,
        ).strip(" ，,。的")
        description = re.sub(r"^(那个|那位|这个|这位)", "", description).strip()
        description = re.sub(
            r"[，,]?\s*(?:以)?速度(?:为|是)?\s*\d+(?:\.\d+)?\s*(?:m/s|米每秒)?.*$",
            "",
            description,
            flags=re.IGNORECASE,
        ).strip(" ，,。")
        description = description or "目标人物"
        prompt = f"一张{description}的全身照片"
    else:
        description = re.sub(
            r"^(please\s+)?(robot\s+)?(follow|track|find|locate|look for|chase)\s+",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip(" ,.")
        description = re.sub(r"^(that|the)\s+(person|man|woman)\s+(who\s+is\s+)?", "", description, flags=re.IGNORECASE)
        description = re.sub(
            r"\s+(?:at\s+)?speed(?:\s+of)?\s+\d+(?:\.\d+)?\s*m/s.*$",
            "",
            description,
            flags=re.IGNORECASE,
        ).strip(" ,.")
        description = description or "the target person"
        prompt = f"a full-body photo of {description}"
    return description, prompt


def translate_common_zh_attributes(description: str) -> Optional[str]:
    """Translate common person attributes into an auxiliary English prompt.

    This deliberately small, deterministic lexicon covers the PP-Human-style
    clothing/accessory vocabulary plus position, hair and gender.  It is not a
    general translator: SigLIP2 still receives the original Chinese prompt, and
    this variant only improves a few Chinese phrases that its tokenizer scores
    too conservatively (notably ``T恤``).
    """

    if not re.search(r"[\u3400-\u9fff]", description):
        return None
    text = description
    position = ""
    for source, target in (("最左边", "leftmost"), ("最右边", "rightmost"), ("中间", "center")):
        if source in text:
            position = target
            text = text.replace(source, "")
            break

    subject = "person"
    subjects = (
        (r"金发(女人|女性)$", "blonde woman"),
        (r"黑人(男人|男性)$", "Black man"),
        (r"黑人(女人|女性)$", "Black woman"),
        (r"亚洲(男人|男性)$", "Asian man"),
        (r"亚洲(女人|女性)$", "Asian woman"),
        (r"(男人|男性|男生|男孩|男士)$", "man"),
        (r"(女人|女性|女生|女孩|女士)$", "woman"),
        (r"人$", "person"),
    )
    for pattern, translated in subjects:
        match = re.search(pattern, text)
        if match:
            subject = translated
            text = text[: match.start()].rstrip("的 ")
            break

    # Longer phrases must be replaced before their component words.
    phrases = (
        ("橙色碎花连衣裙", "orange floral dress"),
        ("蓝白花纹上衣", "blue and white patterned shirt"),
        ("蓝色牛仔裤", "blue jeans"),
        ("黑色双肩包", "black backpack"),
        ("白色衬衫", "white shirt"),
        ("黑色领带", "black tie"),
        ("双肩包", "backpack"),
        ("手提包", "handbag"),
        ("背包", "backpack"),
        ("太阳镜", "sunglasses"),
        ("眼镜", "glasses"),
        ("棒球帽", "baseball cap"),
        ("帽子", "hat"),
        ("连衣裙", "dress"),
        ("牛仔裤", "jeans"),
        ("短裤", "shorts"),
        ("裤子", "pants"),
        ("短裙", "skirt"),
        ("T恤", "T-shirt"),
        ("t恤", "T-shirt"),
        ("衬衫", "shirt"),
        ("上衣", "shirt"),
        ("外套", "jacket"),
        ("碎花", "floral"),
        ("花纹", "patterned"),
        ("条纹", "striped"),
        ("长头发", "long hair"),
        ("短头发", "short hair"),
        ("长发", "long hair"),
        ("短发", "short hair"),
        ("金发", "blonde hair"),
        ("黑色", "black"),
        ("白色", "white"),
        ("红色", "red"),
        ("蓝色", "blue"),
        ("蓝白", "blue and white"),
        ("黄色", "yellow"),
        ("灰色", "gray"),
        ("绿色", "green"),
        ("橙色", "orange"),
        ("紫色", "purple"),
        ("粉色", "pink"),
        ("棕褐色", "brown"),
        ("咖啡色", "brown"),
        ("褐色", "brown"),
        ("棕色", "brown"),
        ("穿着", "wearing"),
        ("穿", "wearing"),
        ("背着", "carrying"),
        ("背", "carrying"),
        ("戴着", "wearing"),
        ("戴", "wearing"),
        ("拿着", "holding"),
        ("拿", "holding"),
        ("和", "and"),
    )
    for source, target in phrases:
        text = text.replace(source, f" {target} ")
    text = re.sub(r"[的，、,。]+", " ", text)
    text = " ".join(text.split())
    components = [part for part in (position, subject, text) if part]
    return " ".join(components)


def build_prompt_variants(
    description: str,
    zh_template: str = "一张{description}的全身照片",
    other_template: str = "a full-body photo of {description}",
) -> List[str]:
    """Build one or two prompts for multilingual score fusion."""

    if re.search(r"[\u3400-\u9fff]", description):
        prompts = [zh_template.format(description=description)]
        translated = translate_common_zh_attributes(description)
        if translated:
            prompts.append(other_template.format(description=translated))
            position_free = re.sub(r"^(leftmost|rightmost|center)\s+", "", translated)
            if position_free != translated:
                prompts.append(other_template.format(description=position_free))
        return prompts
    return [other_template.format(description=description)]


def apply_spatial_hint(
    detections: Sequence["PersonDetection"],
    image_shape: Sequence[int],
    description: str,
) -> Optional[str]:
    """Attach a geometric prior for explicit left/right/center descriptions."""

    if not detections:
        return None
    lowered = description.lower()
    if re.search(r"最左边|左侧|左边|leftmost|on the left", lowered):
        hint = "left"
    elif re.search(r"最右边|右侧|右边|rightmost|on the right", lowered):
        hint = "right"
    elif re.search(r"中间|中央|正中|middle|center", lowered):
        hint = "center"
    else:
        return None
    width = float(image_shape[1])
    centers = np.asarray([(item.xyxy[0] + item.xyxy[2]) / (2.0 * width) for item in detections], dtype=np.float32)
    if hint == "center":
        scores = 1.0 - np.clip(np.abs(centers - 0.5) * 2.0, 0.0, 1.0)
    else:
        span = float(centers.max() - centers.min())
        if span <= 1e-9:
            scores = np.ones_like(centers)
        elif hint == "left":
            scores = (centers.max() - centers) / span
        else:
            scores = (centers - centers.min()) / span
    for detection, score in zip(detections, scores.tolist()):
        detection.spatial_score = float(score)
    return hint


def _scalar(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _xyxy_iou(first: BBox, second: BBox) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _cosine_similarity(first: Optional[np.ndarray], second: Optional[np.ndarray]) -> float:
    if first is None or second is None:
        return 0.0
    first_vector = np.asarray(first, dtype=np.float32).reshape(-1)
    second_vector = np.asarray(second, dtype=np.float32).reshape(-1)
    denominator = float(np.linalg.norm(first_vector) * np.linalg.norm(second_vector))
    if denominator <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(first_vector, second_vector) / denominator, -1.0, 1.0))


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    result = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(result))
    return result / norm if norm > 1e-12 else result


@dataclass
class PersonDetection:
    """A detector result enriched with semantic and appearance information."""

    xyxy: BBox
    confidence: float
    track_id: Optional[int] = None
    semantic_score: float = 0.0
    spatial_score: Optional[float] = None
    appearance: Optional[np.ndarray] = field(default=None, repr=False)
    association_score: float = 0.0

    def normalized_geometry(self, image_shape: Sequence[int]) -> Tuple[List[float], List[float]]:
        height, width = int(image_shape[0]), int(image_shape[1])
        if width <= 0 or height <= 0:
            raise ValueError(f"非法图像尺寸：{tuple(image_shape)}")
        x1, y1, x2, y2 = self.xyxy
        center = [((x1 + x2) / 2.0) / width, ((y1 + y2) / 2.0) / height]
        size = [max(0.0, x2 - x1) / width, max(0.0, y2 - y1) / height]
        return [float(np.clip(v, 0.0, 1.0)) for v in center], [
            float(np.clip(v, 0.0, 1.0)) for v in size
        ]


class YoloPersonDetector:
    """Ultralytics person detector with optional persistent ByteTrack IDs."""

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        confidence: float = 0.30,
        iou: float = 0.55,
        image_size: int = 640,
        use_bytetrack: bool = True,
        tracker: str = "bytetrack.yaml",
        device: str = "auto",
        model: Any = None,
    ) -> None:
        self.model_path = str(model_path)
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.image_size = int(image_size)
        self.use_bytetrack = bool(use_bytetrack)
        self.tracker = str(tracker)
        self.device = device
        self._model = model

    def _ensure_model(self) -> Any:
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError("缺少 Ultralytics：请按 docs/LOVON_AGENT_PRO.md 创建环境") from exc
            LOGGER.info("Loading person detector: %s", self.model_path)
            self._model = YOLO(self.model_path)
        return self._model

    def detect(self, image: np.ndarray) -> List[PersonDetection]:
        if image is None or not isinstance(image, np.ndarray) or image.ndim < 2:
            raise ValueError("image 必须是 OpenCV BGR numpy 图像")
        model = self._ensure_model()
        kwargs: Dict[str, Any] = {
            "source": image,
            "classes": [0],
            "conf": self.confidence,
            "iou": self.iou,
            "imgsz": self.image_size,
            "verbose": False,
        }
        if self.device != "auto":
            kwargs["device"] = self.device
        if self.use_bytetrack:
            results = model.track(persist=True, tracker=self.tracker, **kwargs)
        else:
            results = model.predict(**kwargs)

        detections: List[PersonDetection] = []
        for result in results or []:
            names = getattr(result, "names", {})
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                class_id = int(_scalar(box.cls))
                class_name = names.get(class_id, str(class_id)) if isinstance(names, Mapping) else names[class_id]
                if str(class_name).lower() != "person":
                    continue
                coordinates = box.xyxy[0].tolist()
                track_id = None
                if getattr(box, "id", None) is not None:
                    track_id = int(_scalar(box.id))
                detections.append(
                    PersonDetection(
                        xyxy=tuple(float(v) for v in coordinates[:4]),  # type: ignore[arg-type]
                        confidence=_scalar(box.conf),
                        track_id=track_id,
                    )
                )
        return detections


class SiglipPersonMatcher:
    """Score each person crop against a free-form multilingual description."""

    def __init__(
        self,
        model_name_or_path: str = "google/siglip2-base-patch16-224",
        revision: Optional[str] = None,
        device: str = "auto",
        half_precision: bool = True,
        local_files_only: bool = False,
        crop_padding: float = 0.05,
        model: Any = None,
        processor: Any = None,
    ) -> None:
        self.model_name_or_path = str(model_name_or_path)
        self.revision = revision
        self.device_name = device
        self.half_precision = bool(half_precision)
        self.local_files_only = bool(local_files_only)
        self.crop_padding = float(crop_padding)
        self._model = model
        self._processor = processor
        self._torch: Any = None
        self._device: Any = None

    def _ensure_model(self) -> Tuple[Any, Any, Any]:
        if self._model is not None and self._processor is not None:
            if self._torch is None:
                import torch

                self._torch = torch
                self._device = torch.device(
                    "cuda" if self.device_name == "auto" and torch.cuda.is_available() else
                    ("cpu" if self.device_name == "auto" else self.device_name)
                )
                if hasattr(self._model, "to"):
                    self._model.to(self._device)
                if hasattr(self._model, "eval"):
                    self._model.eval()
            return self._model, self._processor, self._torch

        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("缺少 torch/transformers：请按 docs/LOVON_AGENT_PRO.md 创建环境") from exc

        self._torch = torch
        if self.device_name == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(self.device_name)
        dtype = torch.float16 if self._device.type == "cuda" and self.half_precision else torch.float32
        load_kwargs: Dict[str, Any] = {
            "local_files_only": self.local_files_only,
        }
        if self.revision and not Path(self.model_name_or_path).exists():
            load_kwargs["revision"] = self.revision
        LOGGER.info("Loading multilingual person matcher: %s on %s", self.model_name_or_path, self._device)
        self._processor = AutoProcessor.from_pretrained(self.model_name_or_path, use_fast=False, **load_kwargs)
        self._model = AutoModel.from_pretrained(
            self.model_name_or_path,
            torch_dtype=dtype,
            **load_kwargs,
        ).to(self._device).eval()
        return self._model, self._processor, torch

    def _crop_people(self, image: np.ndarray, detections: Sequence[PersonDetection]) -> List[Any]:
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - dependency is pinned
            raise RuntimeError("缺少 Pillow") from exc
        image_height, image_width = image.shape[:2]
        crops: List[Any] = []
        for detection in detections:
            x1, y1, x2, y2 = detection.xyxy
            pad_x = (x2 - x1) * self.crop_padding
            pad_y = (y2 - y1) * self.crop_padding
            left = max(0, int(math.floor(x1 - pad_x)))
            top = max(0, int(math.floor(y1 - pad_y)))
            right = min(image_width, int(math.ceil(x2 + pad_x)))
            bottom = min(image_height, int(math.ceil(y2 + pad_y)))
            if right <= left or bottom <= top:
                raise ValueError(f"检测框没有有效面积：{detection.xyxy}")
            crop_bgr = image[top:bottom, left:right]
            crop_rgb = crop_bgr[..., :3][:, :, ::-1]
            crops.append(Image.fromarray(np.ascontiguousarray(crop_rgb)))
        return crops

    def score(
        self,
        image: np.ndarray,
        detections: Sequence[PersonDetection],
        prompt: str | Sequence[str],
    ) -> Tuple[List[float], List[np.ndarray]]:
        if not detections:
            return [], []
        model, processor, torch = self._ensure_model()
        crops = self._crop_people(image, detections)
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        if not prompts:
            raise ValueError("matcher prompt 不能为空")
        inputs = processor(
            text=prompts,
            images=crops,
            padding="max_length",
            max_length=64,
            return_tensors="pt",
        )
        inputs = {key: value.to(self._device) if hasattr(value, "to") else value for key, value in inputs.items()}
        model_dtype = next(model.parameters()).dtype if hasattr(model, "parameters") else None
        if model_dtype is not None and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model_dtype)
        with torch.inference_mode():
            outputs = model(**inputs)
            scores_tensor = torch.sigmoid(outputs.logits_per_image).amax(dim=1)
            embeddings_tensor = outputs.image_embeds
        scores = [float(value) for value in scores_tensor.detach().float().cpu().tolist()]
        embeddings = [
            _unit_vector(np.asarray(row, dtype=np.float32))
            for row in embeddings_tensor.detach().float().cpu().numpy()
        ]
        return scores, embeddings


class StableTargetSelector:
    """Lock one identity using tracker, location, appearance and semantics."""

    def __init__(self, **config: Any) -> None:
        self.acquire_score_threshold = float(config.get("acquire_score_threshold", 0.10))
        self.acquire_margin = float(config.get("acquire_margin", 0.015))
        self.acquire_semantic_weight = float(config.get("acquire_semantic_weight", 0.75))
        self.acquire_spatial_weight = float(config.get("acquire_spatial_weight", 0.15))
        self.acquire_detector_weight = float(config.get("acquire_detector_weight", 0.10))
        self.max_missed_frames = int(config.get("max_missed_frames", 12))
        self.min_iou = float(config.get("min_iou", 0.08))
        self.min_appearance_similarity = float(config.get("min_appearance_similarity", 0.72))
        self.min_association_score = float(config.get("min_association_score", 0.30))
        self.track_id_weight = float(config.get("track_id_weight", 0.30))
        self.iou_weight = float(config.get("iou_weight", 0.25))
        self.appearance_weight = float(config.get("appearance_weight", 0.35))
        self.semantic_weight = float(config.get("semantic_weight", 0.10))
        self.embedding_momentum = float(config.get("embedding_momentum", 0.85))
        acquire_weight_sum = (
            self.acquire_semantic_weight + self.acquire_spatial_weight + self.acquire_detector_weight
        )
        if not math.isclose(acquire_weight_sum, 1.0, abs_tol=1e-6):
            raise ValueError("selector 的首次锁定权重之和必须等于 1")
        weight_sum = self.track_id_weight + self.iou_weight + self.appearance_weight + self.semantic_weight
        if weight_sum <= 0.0:
            raise ValueError("selector 的关联权重之和必须大于 0")
        self.reset()

    def reset(self) -> None:
        self.target_track_id: Optional[int] = None
        self.target_bbox: Optional[BBox] = None
        self.target_embedding: Optional[np.ndarray] = None
        self.missed_frames = 0
        self.ever_acquired = False
        self.status = "searching"
        self.reason = "instruction_changed"

    def _acquire(self, detections: Sequence[PersonDetection], use_appearance: bool) -> Optional[PersonDetection]:
        ranked: List[Tuple[float, PersonDetection]] = []
        for detection in detections:
            appearance = max(0.0, _cosine_similarity(self.target_embedding, detection.appearance))
            if use_appearance and self.target_embedding is not None:
                rank_score = 0.65 * detection.semantic_score + 0.35 * appearance
            else:
                if detection.spatial_score is None:
                    semantic_weight = self.acquire_semantic_weight + self.acquire_spatial_weight
                    spatial_term = 0.0
                else:
                    semantic_weight = self.acquire_semantic_weight
                    spatial_term = self.acquire_spatial_weight * detection.spatial_score
                rank_score = (
                    semantic_weight * detection.semantic_score
                    + spatial_term
                    + self.acquire_detector_weight * detection.confidence
                )
            ranked.append((rank_score, detection))
        ranked.sort(key=lambda item: item[0], reverse=True)
        if not ranked:
            self.reason = "no_person"
            return None
        best_rank, best = ranked[0]
        if best.semantic_score < self.acquire_score_threshold:
            self.reason = "semantic_score_below_threshold"
            return None
        if len(ranked) > 1 and (best_rank - ranked[1][0]) < self.acquire_margin:
            self.reason = "ambiguous_candidates"
            return None
        best.association_score = best_rank
        self.reason = "reacquired" if self.ever_acquired else "semantic_acquired"
        return best

    def _associate(self, detections: Sequence[PersonDetection]) -> Optional[PersonDetection]:
        if self.target_bbox is None:
            return None
        ranked: List[Tuple[float, PersonDetection]] = []
        for detection in detections:
            same_id = float(
                self.target_track_id is not None
                and detection.track_id is not None
                and self.target_track_id == detection.track_id
            )
            overlap = _xyxy_iou(self.target_bbox, detection.xyxy)
            appearance = max(0.0, _cosine_similarity(self.target_embedding, detection.appearance))
            # IoU is a motion cue, not proof of identity: a different person can
            # walk into the previous bbox.  A changed track ID therefore needs
            # appearance support before location/semantics may affect ranking.
            valid = same_id > 0.0 or appearance >= self.min_appearance_similarity
            if not valid:
                continue
            score = (
                self.track_id_weight * same_id
                + self.iou_weight * overlap
                + self.appearance_weight * appearance
                + self.semantic_weight * detection.semantic_score
            )
            detection.association_score = score
            ranked.append((score, detection))
        if not ranked:
            self.reason = "association_gate_failed"
            return None
        ranked.sort(key=lambda item: item[0], reverse=True)
        if ranked[0][0] < self.min_association_score:
            self.reason = "association_score_below_threshold"
            return None
        self.reason = "identity_associated"
        return ranked[0][1]

    def _commit(self, selected: PersonDetection) -> PersonDetection:
        self.target_bbox = selected.xyxy
        if selected.track_id is not None:
            self.target_track_id = selected.track_id
        if selected.appearance is not None:
            new_embedding = _unit_vector(selected.appearance)
            if self.target_embedding is None:
                self.target_embedding = new_embedding
            else:
                mixed = self.embedding_momentum * self.target_embedding + (1.0 - self.embedding_momentum) * new_embedding
                self.target_embedding = _unit_vector(mixed)
        self.missed_frames = 0
        self.ever_acquired = True
        self.status = "tracking"
        return selected

    def select(self, detections: Sequence[PersonDetection]) -> Optional[PersonDetection]:
        if not detections:
            if self.ever_acquired:
                self.missed_frames += 1
                self.status = "lost" if self.missed_frames <= self.max_missed_frames else "searching"
                self.reason = "target_not_visible"
            else:
                self.status = "searching"
                self.reason = "no_person"
            return None

        if not self.ever_acquired:
            selected = self._acquire(detections, use_appearance=False)
        elif self.missed_frames <= self.max_missed_frames:
            selected = self._associate(detections)
        else:
            selected = self._acquire(detections, use_appearance=True)

        if selected is None:
            if self.ever_acquired:
                self.missed_frames += 1
                self.status = "lost" if self.missed_frames <= self.max_missed_frames else "searching"
            else:
                self.status = "searching"
            return None
        return self._commit(selected)


def parse_requested_speed(instruction: str, default_speed: float, max_speed: float) -> float:
    """Extract an optional m/s value, capped by the configured safety limit."""

    patterns = (
        r"(?:speed(?:\s+of)?|速度(?:为|是)?)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:m/s|米每秒)?",
        r"(\d+(?:\.\d+)?)\s*(?:m/s|米每秒)",
    )
    for pattern in patterns:
        match = re.search(pattern, instruction, flags=re.IGNORECASE)
        if match:
            return float(np.clip(float(match.group(1)), 0.0, max_speed))
    return float(np.clip(default_speed, 0.0, max_speed))


class BBoxMotionController:
    """Transparent replacement for the geometric behavior learned by l2mm."""

    def __init__(self, **config: Any) -> None:
        self.default_speed = float(config.get("default_speed", 0.25))
        self.max_linear_speed = float(config.get("max_linear_speed", 0.40))
        self.stop_bbox_width = float(config.get("stop_bbox_width", 0.62))
        self.slow_down_bbox_width = float(config.get("slow_down_bbox_width", 0.38))
        self.center_deadband = float(config.get("center_deadband", 0.06))
        self.turn_in_place_error = float(config.get("turn_in_place_error", 0.30))
        self.yaw_gain = float(config.get("yaw_gain", 1.8))
        self.max_yaw_speed = float(config.get("max_yaw_speed", 1.0))
        self.angular_sign = float(config.get("angular_sign", -1.0))
        self.search_angular_speed = float(config.get("search_angular_speed", 0.0))
        self.velocity_scale = float(config.get("velocity_scale", 1.0))
        if not 0.0 <= self.slow_down_bbox_width < self.stop_bbox_width <= 1.0:
            raise ValueError("需要满足 0 <= slow_down_bbox_width < stop_bbox_width <= 1")

    def predict(
        self,
        state: Mapping[str, Any],
        target_visible: bool,
        instruction: str,
    ) -> Tuple[str, List[float]]:
        if not target_visible:
            return "searching", [0.0, 0.0, self.search_angular_speed * self.velocity_scale]
        center_x = float(state["object_xyn"][0])
        bbox_width = float(state["object_whn"][0])
        if bbox_width >= self.stop_bbox_width:
            return "success", [0.0, 0.0, 0.0]

        error = center_x - 0.5
        angular = 0.0 if abs(error) <= self.center_deadband else self.angular_sign * self.yaw_gain * error
        angular = float(np.clip(angular, -self.max_yaw_speed, self.max_yaw_speed))
        requested_speed = parse_requested_speed(instruction, self.default_speed, self.max_linear_speed)
        if bbox_width <= self.slow_down_bbox_width:
            distance_factor = 1.0
        else:
            distance_factor = (self.stop_bbox_width - bbox_width) / (
                self.stop_bbox_width - self.slow_down_bbox_width
            )
        alignment_factor = float(np.clip(1.0 - abs(error) / self.turn_in_place_error, 0.0, 1.0))
        linear = requested_speed * distance_factor * alignment_factor
        motion = [linear * self.velocity_scale, 0.0, angular * self.velocity_scale]
        return "running", [float(value) for value in motion]


class L2MMMotionController:
    """Adapter for the original private ``MotionPredictor`` checkpoint."""

    def __init__(
        self,
        tokenizer_path: str,
        model_path: str,
        velocity_scale: float = 1.0,
        predictor: Any = None,
    ) -> None:
        self.velocity_scale = float(velocity_scale)
        if predictor is None:
            try:
                module = importlib.import_module("lovon.models_cxn_025.api_language2motion_transformer")
                predictor = module.MotionPredictor(model_path=model_path, tokenizer_path=tokenizer_path)
            except (ImportError, AttributeError, FileNotFoundError) as exc:
                raise RuntimeError(
                    "l2mm 私有代码/权重不可用；请配置 controller.backend=bbox，"
                    "或按文档放置原始 tokenizer 和 checkpoint"
                ) from exc
        self.predictor = predictor

    def predict(
        self,
        state: Mapping[str, Any],
        target_visible: bool,
        instruction: str,
    ) -> Tuple[str, List[float]]:
        if not target_visible:
            return "searching", [0.0, 0.0, 0.0]
        speed = parse_requested_speed(instruction, default_speed=0.25, max_speed=0.50)
        control_instruction = f"run to the person at speed of {speed:.2f} m/s"
        input_data = {
            "mission_instruction_0": control_instruction,
            "mission_instruction_1": control_instruction,
            **dict(state),
        }
        prediction = self.predictor.predict(input_data)
        vector = prediction.get("motion_vector") or [0.0, 0.0, 0.0]
        if len(vector) != 3:
            raise RuntimeError(f"l2mm 返回了非法 motion_vector：{vector!r}")
        motion = [float(value) * self.velocity_scale for value in vector]
        return str(prediction.get("predicted_state", "running")), motion


class LovonAgentPro:
    """Select and follow the person described by a natural-language command."""

    def __init__(
        self,
        config: Optional[Mapping[str, Any]] = None,
        detector: Any = None,
        matcher: Any = None,
        selector: Optional[StableTargetSelector] = None,
        controller: Any = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = copy.deepcopy(DEFAULT_CONFIG)
        if config:
            _deep_update(self.config, config)
        runtime = self.config["runtime"]
        detector_config = self.config["detector"]
        matcher_config = self.config["matcher"]
        scheduling_config = self.config["scheduling"]
        selector_config = self.config["selector"]
        controller_config = self.config["controller"]

        self.detector = detector or self._build_detector(detector_config, runtime)
        self.matcher = matcher or self._build_matcher(matcher_config, runtime)
        self.selector = selector or StableTargetSelector(**selector_config)
        self.controller = controller or self._build_controller(controller_config)
        self.clock = clock
        self.matcher_policy = str(scheduling_config.get("matcher_policy", "always")).lower()
        if self.matcher_policy not in {"always", "event_driven"}:
            raise ValueError("scheduling.matcher_policy 只能是 always 或 event_driven")
        self.search_interval_sec = float(scheduling_config.get("search_interval_sec", 0.0))
        self.refresh_interval_sec = float(scheduling_config.get("refresh_interval_sec", 0.0))
        if self.search_interval_sec < 0.0 or self.refresh_interval_sec < 0.0:
            raise ValueError("scheduling 时间间隔不能为负数")
        self.last_match_time = -math.inf
        self.matcher_ran = False
        self.matcher_reason = "not_started"
        self.matcher_candidate_count = 0
        self._feature_cache: Dict[int, Tuple[float, np.ndarray, float]] = {}
        self.wh_scale_factor = float(controller_config.get("wh_scale_factor", 1.0))
        self.target_instruction: Optional[str] = None
        self.target_description = ""
        self.target_prompt: List[str] = []
        self.last_candidates: List[PersonDetection] = []
        self.last_selected: Optional[PersonDetection] = None
        self.motion_vector = [0.0, 0.0, 0.0]
        self.state: Dict[str, Any] = self._empty_state()

    @classmethod
    def from_config_file(cls, path: os.PathLike[str] | str, **dependencies: Any) -> "LovonAgentPro":
        return cls(config=load_pro_config(path), **dependencies)

    @staticmethod
    def _build_detector(config: Mapping[str, Any], runtime: Mapping[str, Any]) -> Any:
        backend = str(config.get("backend", "ultralytics")).lower()
        if backend == "ultralytics":
            return YoloPersonDetector(
                model_path=config["model"],
                confidence=config["confidence"],
                iou=config["iou"],
                image_size=config["image_size"],
                use_bytetrack=config["use_bytetrack"],
                tracker=config["tracker"],
                device=runtime["device"],
            )
        if backend == "rknn_yolo11":
            from lovon.rknn_backend import RknnYolo11PersonDetector

            return RknnYolo11PersonDetector(
                model_path=str(config["model"]),
                confidence=float(config["confidence"]),
                iou=float(config["iou"]),
                image_size=int(config["image_size"]),
                core_mask=str(config.get("core_mask", "auto")),
                tracker_iou_threshold=float(config.get("tracker_iou_threshold", 0.20)),
                tracker_max_missed=int(config.get("tracker_max_missed", 15)),
            )
        raise ValueError(f"未知 detector.backend：{backend!r}")

    @staticmethod
    def _build_matcher(config: Mapping[str, Any], runtime: Mapping[str, Any]) -> Any:
        backend = str(config.get("backend", "siglip")).lower()
        if backend == "siglip":
            return SiglipPersonMatcher(
                model_name_or_path=config["model"],
                revision=config.get("revision"),
                device=runtime["device"],
                half_precision=runtime["half_precision"],
                local_files_only=config["local_files_only"],
                crop_padding=config["crop_padding"],
            )
        if backend == "rknn_clip":
            from lovon.rknn_backend import RknnClipPersonMatcher

            return RknnClipPersonMatcher(
                image_model_path=str(config["image_model"]),
                text_model_path=str(config["text_model"]),
                tokenizer_path=str(config["tokenizer"]),
                crop_padding=float(config.get("crop_padding", 0.05)),
                core_mask=str(config.get("core_mask", "auto")),
                sequence_length=int(config.get("sequence_length", 20)),
            )
        raise ValueError(f"未知 matcher.backend：{backend!r}")

    @staticmethod
    def _build_controller(config: Mapping[str, Any]) -> Any:
        backend = str(config.get("backend", "bbox")).lower()
        if backend == "bbox":
            return BBoxMotionController(**config)
        if backend == "l2mm":
            tokenizer_path = config.get("l2mm_tokenizer_path")
            model_path = config.get("l2mm_model_path")
            if not tokenizer_path or not model_path:
                raise ValueError("l2mm 后端需要 l2mm_tokenizer_path 和 l2mm_model_path")
            return L2MMMotionController(
                tokenizer_path=str(tokenizer_path),
                model_path=str(model_path),
                velocity_scale=float(config.get("velocity_scale", 1.0)),
            )
        raise ValueError(f"未知 controller.backend：{backend!r}")

    def _empty_state(self) -> Dict[str, Any]:
        return {
            "predicted_object": "NULL",
            "confidence": [0.0],
            "object_xyn": [0.0, 0.0],
            "object_whn": [0.0, 0.0],
            "mission_state_in": "searching",
            "search_state_in": "searching",
            "target_instruction": self.target_instruction,
            "target_description": self.target_description,
            "target_match_score": 0.0,
            "target_track_id": None,
            "candidate_count": 0,
            "selector_reason": self.selector.reason if hasattr(self, "selector") else "not_initialized",
            "missed_frames": self.selector.missed_frames if hasattr(self, "selector") else 0,
            "matcher_ran": self.matcher_ran if hasattr(self, "matcher_ran") else False,
            "matcher_reason": self.matcher_reason if hasattr(self, "matcher_reason") else "not_initialized",
            "matcher_candidate_count": (
                self.matcher_candidate_count if hasattr(self, "matcher_candidate_count") else 0
            ),
            "matcher_age_sec": None,
        }

    def set_target_instruction(self, instruction: str) -> None:
        normalized = " ".join(instruction.strip().split())
        description, _default_prompt = normalize_target_description(normalized)
        if normalized == self.target_instruction:
            return
        self.target_instruction = normalized
        self.target_description = description
        matcher_config = self.config["matcher"]
        self.target_prompt = build_prompt_variants(
            description,
            zh_template=str(matcher_config.get("prompt_template_zh", DEFAULT_CONFIG["matcher"]["prompt_template_zh"])),
            other_template=str(
                matcher_config.get("prompt_template_other", DEFAULT_CONFIG["matcher"]["prompt_template_other"])
            ),
        )
        self.selector.reset()
        self._feature_cache.clear()
        self.last_match_time = -math.inf
        self.matcher_reason = "instruction_changed"
        self.matcher_candidate_count = 0
        self.state = self._empty_state()

    def _should_run_matcher(
        self,
        detections: Sequence[PersonDetection],
        now: float,
        instruction_changed: bool,
    ) -> Tuple[bool, str]:
        if not detections:
            return False, "no_candidates"
        if self.matcher_policy == "always":
            return True, "always"
        if instruction_changed or not self.selector.ever_acquired:
            due = now - self.last_match_time >= self.search_interval_sec
            return due, "acquire_due" if due else "search_throttled"
        target_id = self.selector.target_track_id
        target_id_visible = target_id is not None and any(item.track_id == target_id for item in detections)
        if not target_id_visible:
            due = now - self.last_match_time >= self.search_interval_sec
            return due, "target_id_missing" if due else "search_throttled"
        due = self.refresh_interval_sec > 0.0 and now - self.last_match_time >= self.refresh_interval_sec
        return due, "periodic_refresh" if due else "tracking_cache"

    def _restore_cached_features(self, detections: Sequence[PersonDetection]) -> None:
        for detection in detections:
            if detection.track_id is None:
                continue
            cached = self._feature_cache.get(detection.track_id)
            if cached is None:
                continue
            detection.semantic_score = cached[0]
            detection.appearance = cached[1].copy()

    def _update_feature_cache(self, detections: Sequence[PersonDetection], now: float) -> None:
        for detection in detections:
            if detection.track_id is None or detection.appearance is None:
                continue
            self._feature_cache[detection.track_id] = (
                float(detection.semantic_score),
                _unit_vector(detection.appearance),
                now,
            )
        if len(self._feature_cache) > 256:
            newest = sorted(self._feature_cache.items(), key=lambda item: item[1][2], reverse=True)[:128]
            self._feature_cache = dict(newest)

    def _matcher_age(self) -> Optional[float]:
        if not math.isfinite(self.last_match_time):
            return None
        return max(0.0, self.clock() - self.last_match_time)

    def _update_state(self, image: np.ndarray, selected: Optional[PersonDetection], count: int) -> None:
        if selected is None:
            self.state.update(self._empty_state())
            self.state.update(
                {
                    "target_instruction": self.target_instruction,
                    "target_description": self.target_description,
                    "candidate_count": count,
                    "search_state_in": self.selector.status,
                    "selector_reason": self.selector.reason,
                    "missed_frames": self.selector.missed_frames,
                    "matcher_ran": self.matcher_ran,
                    "matcher_reason": self.matcher_reason,
                    "matcher_candidate_count": self.matcher_candidate_count,
                    "matcher_age_sec": self._matcher_age(),
                }
            )
            return
        center, size = selected.normalized_geometry(image.shape)
        scaled_size = [float(np.clip(value * self.wh_scale_factor, 0.0, 1.0)) for value in size]
        self.state.update(
            {
                "predicted_object": "person",
                "confidence": [selected.confidence],
                "object_xyn": center,
                "object_whn": scaled_size,
                "search_state_in": self.selector.status,
                "target_instruction": self.target_instruction,
                "target_description": self.target_description,
                "target_match_score": selected.semantic_score,
                "target_track_id": selected.track_id,
                "candidate_count": count,
                "selector_reason": self.selector.reason,
                "missed_frames": self.selector.missed_frames,
                "matcher_ran": self.matcher_ran,
                "matcher_reason": self.matcher_reason,
                "matcher_candidate_count": self.matcher_candidate_count,
                "matcher_age_sec": self._matcher_age(),
            }
        )

    def run(
        self,
        image: np.ndarray,
        mission_instruction_0: Optional[str] = None,
        mission_instruction_1: Optional[str] = None,
        user_instruction: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], List[float]]:
        """Run one frame.

        ``user_instruction`` is preferred.  ``mission_instruction_1`` is accepted
        as a drop-in bridge from the original loop.  ``mission_instruction_0`` is
        ignored for visual selection, but remains in the signature for API
        compatibility.
        """

        del mission_instruction_0
        instruction = user_instruction or mission_instruction_1 or self.target_instruction
        if instruction is None:
            raise ValueError("首次 run 必须提供 user_instruction（例如：跟随穿红色上衣、背黑色包的人）")
        previous_instruction = self.target_instruction
        self.set_target_instruction(instruction)
        instruction_changed = previous_instruction != self.target_instruction
        now = self.clock()
        detections = self.detector.detect(image)
        self._restore_cached_features(detections)
        self.matcher_ran, self.matcher_reason = self._should_run_matcher(
            detections,
            now,
            instruction_changed,
        )
        self.matcher_candidate_count = 0
        periodic_rejected = False
        if self.matcher_ran:
            match_candidates = list(detections)
            if self.matcher_reason == "periodic_refresh" and self.selector.target_track_id is not None:
                match_candidates = [
                    item for item in detections if item.track_id == self.selector.target_track_id
                ]
            self.matcher_candidate_count = len(match_candidates)
            scores, embeddings = self.matcher.score(image, match_candidates, self.target_prompt)
            if len(scores) != len(match_candidates) or len(embeddings) != len(match_candidates):
                raise RuntimeError("matcher 输出数量与人物检测数量不一致")
            self.last_match_time = now
            normalized_embeddings = [
                _unit_vector(np.asarray(embedding, dtype=np.float32)) for embedding in embeddings
            ]
            if self.matcher_reason == "periodic_refresh" and match_candidates:
                semantic_ok = float(scores[0]) >= self.selector.acquire_score_threshold
                appearance_similarity = _cosine_similarity(
                    self.selector.target_embedding,
                    normalized_embeddings[0],
                )
                appearance_ok = (
                    self.selector.target_embedding is None
                    or appearance_similarity >= self.selector.min_appearance_similarity
                )
                if not semantic_ok or not appearance_ok:
                    # Do not let a possibly switched tracker ID update the
                    # identity template.  Stop this frame and force an
                    # all-candidate semantic reacquisition on the next frame.
                    periodic_rejected = True
                    self.matcher_reason = (
                        "periodic_rejected_semantic" if not semantic_ok else "periodic_rejected_appearance"
                    )
                    self.selector.target_track_id = None
                    self.selector.missed_frames = self.selector.max_missed_frames + 1
                    self.selector.status = "searching"
                    self.selector.reason = "periodic_verification_failed"
                    self.last_match_time = -math.inf
            if not periodic_rejected:
                for detection, score, embedding in zip(match_candidates, scores, normalized_embeddings):
                    detection.semantic_score = float(score)
                    detection.appearance = embedding
                self._update_feature_cache(match_candidates, now)
        apply_spatial_hint(detections, image.shape, self.target_description)
        selected = None if periodic_rejected else self.selector.select(detections)
        self.last_candidates = list(detections)
        self.last_selected = selected
        self._update_state(image, selected, len(detections))
        mission_state, motion = self.controller.predict(
            self.state,
            target_visible=selected is not None,
            instruction=self.target_instruction,
        )
        if motion is None or len(motion) != 3 or not all(np.isfinite(value) for value in motion):
            raise RuntimeError(f"控制器返回了非法 motion_vector：{motion!r}")
        self.motion_vector = [float(value) for value in motion]
        self.state["mission_state_in"] = mission_state
        return copy.deepcopy(self.state), list(self.motion_vector)

    def annotate(self, image: np.ndarray) -> np.ndarray:
        """Draw candidates and the locked target without modifying ``image``."""

        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - environment error
            raise RuntimeError("可视化需要 opencv-python") from exc
        output = image.copy()
        panel = output.copy()
        cv2.rectangle(panel, (0, 0), (output.shape[1], 62), (0, 0, 0), -1)
        output = cv2.addWeighted(panel, 0.72, output, 0.28, 0.0)
        for detection in self.last_candidates:
            is_selected = detection is self.last_selected
            color = (0, 220, 0) if is_selected else (0, 180, 255)
            thickness = 3 if is_selected else 1
            x1, y1, x2, y2 = (int(round(value)) for value in detection.xyxy)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
            identity = "-" if detection.track_id is None else str(detection.track_id)
            label = f"id={identity} d={detection.confidence:.2f} t={detection.semantic_score:.2f}"
            label_y = min(output.shape[0] - 5, max(82, y1 + 20))
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
            cv2.rectangle(
                output,
                (x1, label_y - label_height - 4),
                (min(output.shape[1] - 1, x1 + label_width + 4), label_y + 3),
                (0, 0, 0),
                -1,
            )
            cv2.putText(output, label, (x1 + 2, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)
        status = f"{self.selector.status}: {self.selector.reason}"
        cv2.putText(output, status, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            output,
            f"motion=({self.motion_vector[0]:+.2f}, {self.motion_vector[1]:+.2f}, {self.motion_vector[2]:+.2f})",
            (12, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return output


__all__ = [
    "BBoxMotionController",
    "apply_spatial_hint",
    "build_prompt_variants",
    "DEFAULT_CONFIG",
    "L2MMMotionController",
    "LovonAgentPro",
    "PersonDetection",
    "SiglipPersonMatcher",
    "StableTargetSelector",
    "YoloPersonDetector",
    "load_pro_config",
    "normalize_target_description",
    "parse_requested_speed",
    "translate_common_zh_attributes",
]
