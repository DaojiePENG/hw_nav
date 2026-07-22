"""Rockchip RK3588 inference backends for :mod:`lovon.lovon_agent_pro`.

The implementations follow the tensor layout of Rockchip's optimized YOLO11
and CLIP examples.  They intentionally import ``rknnlite`` and Transformers
lazily: x86 development and unit tests therefore do not need an RK3588 wheel,
and a board deployment never silently falls back to the much slower PyTorch
models.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from lovon.lovon_agent_pro import BBox, PersonDetection, _xyxy_iou


LOGGER = logging.getLogger(__name__)


class RknnLiteModel:
    """Lazy, checked wrapper around one ``RKNNLite`` model."""

    def __init__(self, model_path: str, core_mask: str = "auto", runtime: Any = None) -> None:
        self.model_path = str(model_path)
        self.core_mask = str(core_mask).lower()
        self._runtime = runtime

    @staticmethod
    def _checked(operation: str, result: Any) -> None:
        if result not in (None, 0):
            raise RuntimeError(f"RKNN {operation} 失败，错误码：{result}")

    def _ensure_runtime(self) -> Any:
        if self._runtime is not None:
            return self._runtime
        path = Path(self.model_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"RKNN 模型不存在：{path}")
        try:
            from rknnlite.api import RKNNLite
        except ImportError as exc:
            raise RuntimeError(
                "缺少 rknn-toolkit-lite2；请按 docs/LOVON_AGENT_PRO_RK3588.md 安装与板端 RKNPU 驱动匹配的 wheel"
            ) from exc

        runtime = RKNNLite(verbose=False)
        self._checked("load_rknn", runtime.load_rknn(str(path)))
        init_kwargs: Dict[str, Any] = {}
        mask_names = {
            "core0": "NPU_CORE_0",
            "core1": "NPU_CORE_1",
            "core2": "NPU_CORE_2",
            "core0_1_2": "NPU_CORE_0_1_2",
            "all": "NPU_CORE_ALL",
            "auto": "NPU_CORE_AUTO",
        }
        constant_name = mask_names.get(self.core_mask)
        if constant_name is None:
            raise ValueError(f"未知 RKNN core_mask：{self.core_mask!r}")
        core_value = getattr(RKNNLite, constant_name, None)
        if core_value is not None:
            init_kwargs["core_mask"] = core_value
        self._checked("init_runtime", runtime.init_runtime(**init_kwargs))
        self._runtime = runtime
        LOGGER.info("Loaded RKNN model %s (core_mask=%s)", path, self.core_mask)
        return runtime

    def infer(self, inputs: Sequence[np.ndarray]) -> List[np.ndarray]:
        runtime = self._ensure_runtime()
        outputs = runtime.inference(inputs=list(inputs))
        if outputs is None:
            raise RuntimeError(f"RKNN inference 未返回结果：{self.model_path}")
        return [np.asarray(item) for item in outputs]

    def release(self) -> None:
        if self._runtime is not None and hasattr(self._runtime, "release"):
            self._runtime.release()
        self._runtime = None


@dataclass
class _Track:
    track_id: int
    bbox: BBox
    missed: int = 0


class GreedyIoUTracker:
    """Small CPU tracker used to keep RKNN detections on the fast path.

    It is deliberately conservative: unmatched boxes receive new IDs instead
    of guessing across a large jump.  Appearance-based reacquisition remains
    the selector's responsibility.
    """

    def __init__(self, iou_threshold: float = 0.20, max_missed: int = 15) -> None:
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("tracker_iou_threshold 必须在 [0, 1]")
        if max_missed < 0:
            raise ValueError("tracker_max_missed 不能为负数")
        self.iou_threshold = float(iou_threshold)
        self.max_missed = int(max_missed)
        self.tracks: Dict[int, _Track] = {}
        self.next_id = 1

    def update(self, detections: Sequence[PersonDetection]) -> None:
        pairs: List[Tuple[float, int, int]] = []
        for track_id, track in self.tracks.items():
            for detection_index, detection in enumerate(detections):
                overlap = _xyxy_iou(track.bbox, detection.xyxy)
                if overlap >= self.iou_threshold:
                    pairs.append((overlap, track_id, detection_index))
        pairs.sort(reverse=True)
        assigned_tracks = set()
        assigned_detections = set()
        for _overlap, track_id, detection_index in pairs:
            if track_id in assigned_tracks or detection_index in assigned_detections:
                continue
            detection = detections[detection_index]
            detection.track_id = track_id
            self.tracks[track_id].bbox = detection.xyxy
            self.tracks[track_id].missed = 0
            assigned_tracks.add(track_id)
            assigned_detections.add(detection_index)

        for track_id, track in list(self.tracks.items()):
            if track_id not in assigned_tracks:
                track.missed += 1
                if track.missed > self.max_missed:
                    del self.tracks[track_id]

        for detection_index, detection in enumerate(detections):
            if detection_index in assigned_detections:
                continue
            track_id = self.next_id
            self.next_id += 1
            detection.track_id = track_id
            self.tracks[track_id] = _Track(track_id=track_id, bbox=detection.xyxy)


def _softmax(values: np.ndarray, axis: int) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.maximum(np.sum(exponent, axis=axis, keepdims=True), 1e-12)


def _as_nchw(tensor: np.ndarray, expected_channels: Optional[int] = None) -> np.ndarray:
    result = np.asarray(tensor)
    if result.ndim != 4:
        raise RuntimeError(f"RKNN YOLO11 输出应为四维，实际为 {result.shape}")
    if expected_channels is not None and result.shape[1] != expected_channels and result.shape[-1] == expected_channels:
        result = result.transpose(0, 3, 1, 2)
    return result.astype(np.float32, copy=False)


def _dfl(position: np.ndarray) -> np.ndarray:
    batch, channels, height, width = position.shape
    if batch != 1 or channels % 4 != 0:
        raise RuntimeError(f"非法 YOLO11 DFL 输出：{position.shape}")
    bins = channels // 4
    distribution = _softmax(position.reshape(batch, 4, bins, height, width), axis=2)
    weights = np.arange(bins, dtype=np.float32).reshape(1, 1, bins, 1, 1)
    return np.sum(distribution * weights, axis=2)


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> List[int]:
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[current], x1[rest])
        yy1 = np.maximum(y1[current], y1[rest])
        xx2 = np.minimum(x2[current], x2[rest])
        yy2 = np.minimum(y2[current], y2[rest])
        intersection = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[current] + areas[rest] - intersection
        overlap = intersection / np.maximum(union, 1e-12)
        order = rest[overlap <= threshold]
    return keep


class RknnYolo11PersonDetector:
    """Person-only post-processing for Rockchip's optimized YOLO11 RKNN."""

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.30,
        iou: float = 0.55,
        image_size: int = 640,
        core_mask: str = "auto",
        tracker_iou_threshold: float = 0.20,
        tracker_max_missed: int = 15,
        runtime: Any = None,
    ) -> None:
        self.image_size = int(image_size)
        if self.image_size <= 0:
            raise ValueError("detector.image_size 必须大于 0")
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.model = RknnLiteModel(model_path, core_mask=core_mask, runtime=runtime)
        self.tracker = GreedyIoUTracker(tracker_iou_threshold, tracker_max_missed)

    def _letterbox(self, image: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        import cv2

        height, width = image.shape[:2]
        scale = min(self.image_size / width, self.image_size / height)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        pad_x = (self.image_size - resized_width) // 2
        pad_y = (self.image_size - resized_height) // 2
        canvas = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized[:, :, :3]
        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), scale, pad_x, pad_y

    def _decode(self, outputs: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        if len(outputs) not in {6, 9}:
            raise RuntimeError(
                "YOLO11 RKNN 需要 Rockchip 优化版的 6 或 9 个输出；"
                f"当前得到 {len(outputs)} 个，请勿直接使用 Ultralytics 原始 ONNX"
            )
        branch_stride = len(outputs) // 3
        all_boxes: List[np.ndarray] = []
        all_scores: List[np.ndarray] = []
        for branch in range(3):
            position = _as_nchw(outputs[branch * branch_stride], expected_channels=64)
            class_scores = _as_nchw(outputs[branch * branch_stride + 1], expected_channels=80)
            distances = _dfl(position)
            grid_height, grid_width = distances.shape[2:]
            columns, rows = np.meshgrid(
                np.arange(grid_width, dtype=np.float32),
                np.arange(grid_height, dtype=np.float32),
            )
            stride_x = self.image_size / grid_width
            stride_y = self.image_size / grid_height
            x1 = (columns + 0.5 - distances[0, 0]) * stride_x
            y1 = (rows + 0.5 - distances[0, 1]) * stride_y
            x2 = (columns + 0.5 + distances[0, 2]) * stride_x
            y2 = (rows + 0.5 + distances[0, 3]) * stride_y
            boxes = np.stack((x1, y1, x2, y2), axis=-1).reshape(-1, 4)
            person_scores = class_scores[0, 0].reshape(-1)
            mask = person_scores >= self.confidence
            all_boxes.append(boxes[mask])
            all_scores.append(person_scores[mask])
        boxes = np.concatenate(all_boxes, axis=0) if all_boxes else np.empty((0, 4), dtype=np.float32)
        scores = np.concatenate(all_scores, axis=0) if all_scores else np.empty((0,), dtype=np.float32)
        keep = _nms(boxes, scores, self.iou)
        return boxes[keep], scores[keep]

    def detect(self, image: np.ndarray) -> List[PersonDetection]:
        if image is None or not isinstance(image, np.ndarray) or image.ndim != 3:
            raise ValueError("image 必须是 OpenCV BGR numpy 图像")
        model_input, scale, pad_x, pad_y = self._letterbox(image)
        boxes, scores = self._decode(self.model.infer([model_input]))
        height, width = image.shape[:2]
        detections: List[PersonDetection] = []
        for box, score in zip(boxes, scores):
            x1 = float(np.clip((box[0] - pad_x) / scale, 0.0, width - 1.0))
            y1 = float(np.clip((box[1] - pad_y) / scale, 0.0, height - 1.0))
            x2 = float(np.clip((box[2] - pad_x) / scale, 0.0, width - 1.0))
            y2 = float(np.clip((box[3] - pad_y) / scale, 0.0, height - 1.0))
            if x2 > x1 and y2 > y1:
                detections.append(PersonDetection((x1, y1, x2, y2), float(score)))
        self.tracker.update(detections)
        return detections

    def release(self) -> None:
        self.model.release()


class RknnClipPersonMatcher:
    """OpenAI CLIP ViT-B/32 matcher using separate RKNN text/image models."""

    PAD_TOKEN_ID = 49407

    def __init__(
        self,
        image_model_path: str,
        text_model_path: str,
        tokenizer_path: str,
        crop_padding: float = 0.05,
        core_mask: str = "auto",
        sequence_length: int = 20,
        image_runtime: Any = None,
        text_runtime: Any = None,
        tokenizer: Any = None,
    ) -> None:
        self.image_model = RknnLiteModel(image_model_path, core_mask, image_runtime)
        self.text_model = RknnLiteModel(text_model_path, core_mask, text_runtime)
        self.tokenizer_path = str(tokenizer_path)
        self.crop_padding = float(crop_padding)
        self.sequence_length = int(sequence_length)
        if self.sequence_length <= 0:
            raise ValueError("matcher.sequence_length 必须大于 0")
        self._tokenizer = tokenizer
        self._text_cache: Dict[Tuple[str, ...], np.ndarray] = {}

    def _ensure_tokenizer(self) -> Any:
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise RuntimeError("RKNN CLIP 需要 transformers/tokenizers 读取离线 tokenizer") from exc
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, local_files_only=True)
        return self._tokenizer

    @staticmethod
    def _english_prompts(prompt: str | Sequence[str]) -> Tuple[str, ...]:
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        # The Rockchip model is OpenAI CLIP, whose text encoder is English-only.
        english = tuple(item for item in prompts if item.strip() and not re.search(r"[\u3400-\u9fff]", item))
        if not english:
            raise ValueError(
                "RKNN CLIP 没有可用的英文提示。请使用已支持的中文衣着属性，"
                "或直接输入英文；任意中文需要另接翻译/属性解析器。"
            )
        return english

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        result = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(result))
        return result / norm if norm > 1e-12 else result

    def _encode_text(self, prompts: Tuple[str, ...]) -> np.ndarray:
        cached = self._text_cache.get(prompts)
        if cached is not None:
            return cached
        tokenizer = self._ensure_tokenizer()
        tokenized = tokenizer(list(prompts), padding=True, return_tensors="np")
        input_ids = np.asarray(tokenized["input_ids"], dtype=np.int64)
        embeddings = []
        for row in input_ids:
            fixed = np.full((1, self.sequence_length), self.PAD_TOKEN_ID, dtype=np.int64)
            length = min(row.size, self.sequence_length)
            fixed[0, :length] = row[:length]
            output = self.text_model.infer([fixed])[0]
            embeddings.append(self._normalize(output))
        matrix = np.stack(embeddings, axis=0)
        if len(self._text_cache) >= 64:
            self._text_cache.pop(next(iter(self._text_cache)))
        self._text_cache[prompts] = matrix
        return matrix

    def _crop(self, image: np.ndarray, detection: PersonDetection) -> np.ndarray:
        import cv2

        height, width = image.shape[:2]
        x1, y1, x2, y2 = detection.xyxy
        pad_x = (x2 - x1) * self.crop_padding
        pad_y = (y2 - y1) * self.crop_padding
        left = max(0, int(math.floor(x1 - pad_x)))
        top = max(0, int(math.floor(y1 - pad_y)))
        right = min(width, int(math.ceil(x2 + pad_x)))
        bottom = min(height, int(math.ceil(y2 + pad_y)))
        if right <= left or bottom <= top:
            raise ValueError(f"检测框没有有效面积：{detection.xyxy}")
        crop = image[top:bottom, left:right, :3]
        crop_height, crop_width = crop.shape[:2]
        side = max(crop_height, crop_width)
        square = np.zeros((side, side, 3), dtype=np.uint8)
        offset_y = (side - crop_height) // 2
        offset_x = (side - crop_width) // 2
        square[offset_y : offset_y + crop_height, offset_x : offset_x + crop_width] = crop
        resized = cv2.resize(square, (224, 224), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return np.expand_dims(rgb, axis=0)

    def score(
        self,
        image: np.ndarray,
        detections: Sequence[PersonDetection],
        prompt: str | Sequence[str],
    ) -> Tuple[List[float], List[np.ndarray]]:
        if not detections:
            return [], []
        prompts = self._english_prompts(prompt)
        text_embeddings = self._encode_text(prompts)
        scores: List[float] = []
        appearances: List[np.ndarray] = []
        for detection in detections:
            output = self.image_model.infer([self._crop(image, detection)])[0]
            image_embedding = self._normalize(output)
            cosine = float(np.max(text_embeddings @ image_embedding))
            scores.append(float(np.clip((cosine + 1.0) * 0.5, 0.0, 1.0)))
            appearances.append(image_embedding)
        return scores, appearances

    def release(self) -> None:
        self.image_model.release()
        self.text_model.release()


__all__ = [
    "GreedyIoUTracker",
    "RknnClipPersonMatcher",
    "RknnLiteModel",
    "RknnYolo11PersonDetector",
]
