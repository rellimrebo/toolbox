from __future__ import annotations

import math
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from PIL import Image

from frame_labeler.domain import Box, BoxOrigin
from frame_labeler.media import MediaFrame


class InferenceError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    frame_index: int
    timestamp_seconds: float
    image: Image.Image
    class_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Detection:
    class_id: int
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float | None = None

    def __post_init__(self) -> None:
        coordinates = (self.x_min, self.y_min, self.x_max, self.y_max)
        if self.class_id < 0:
            raise ValueError("Detection class ID cannot be negative")
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError("Detection coordinates must be finite")
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("Detection width and height must be greater than zero")
        if self.confidence is not None and (
            not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError("Detection confidence must be between zero and one")


class InferenceProvider(Protocol):
    provider_id: str

    def predict(self, request: InferenceRequest) -> Sequence[Detection]: ...


def suggest_boxes(
    provider: InferenceProvider,
    frame: MediaFrame,
    class_names: Sequence[str],
) -> tuple[Box, ...]:
    provider_id = provider.provider_id.strip()
    if not provider_id or "\n" in provider_id or "\r" in provider_id:
        raise InferenceError("Inference provider must have a non-empty, single-line identifier")
    classes = tuple(class_names)
    request = InferenceRequest(
        frame_index=frame.index,
        timestamp_seconds=frame.timestamp_seconds,
        image=frame.image,
        class_names=classes,
    )
    try:
        detections = tuple(provider.predict(request))
    except Exception as error:
        raise InferenceError(f"Inference provider {provider_id!r} failed") from error

    boxes: list[Box] = []
    for index, detection in enumerate(detections):
        if detection.class_id >= len(classes):
            raise InferenceError(
                f"Detection {index} returned unknown class ID {detection.class_id}"
            )
        x_min = min(max(detection.x_min, 0.0), float(frame.width))
        y_min = min(max(detection.y_min, 0.0), float(frame.height))
        x_max = min(max(detection.x_max, 0.0), float(frame.width))
        y_max = min(max(detection.y_max, 0.0), float(frame.height))
        if x_max <= x_min or y_max <= y_min:
            raise InferenceError(f"Detection {index} falls outside the source frame")
        boxes.append(
            Box(
                id=str(uuid.uuid4()),
                class_id=detection.class_id,
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
                origin=BoxOrigin.INFERRED,
                inference_provider=provider_id,
                confidence=detection.confidence,
            )
        )
    return tuple(boxes)
