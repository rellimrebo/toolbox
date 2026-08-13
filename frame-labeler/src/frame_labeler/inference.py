from __future__ import annotations

import math
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from PIL import Image

from frame_labeler.domain import AnnotationOrigin, Box
from frame_labeler.media import MediaFrame


class InferenceError(RuntimeError):
    pass


PredictionT_co = TypeVar("PredictionT_co", covariant=True)
OBJECT_DETECTION_OUTPUT = "object-detection/v1"


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


@dataclass(frozen=True, slots=True)
class InferenceResult(Generic[PredictionT_co]):
    provider_id: str
    output_type: str
    frame_index: int
    timestamp_seconds: float
    frame_size: tuple[int, int]
    class_names: tuple[str, ...]
    predictions: tuple[PredictionT_co, ...]


class InferenceProvider(Protocol[PredictionT_co]):
    provider_id: str
    output_type: str

    def predict(self, request: InferenceRequest) -> Sequence[PredictionT_co]: ...


def _validated_identifier(value: str, label: str) -> str:
    cleaned = value.strip()
    if not cleaned or "\n" in cleaned or "\r" in cleaned:
        raise InferenceError(f"{label} must be a non-empty, single-line identifier")
    return cleaned


def run_inference(
    provider: InferenceProvider[PredictionT_co],
    frame: MediaFrame,
    class_names: Sequence[str],
) -> InferenceResult[PredictionT_co]:
    provider_id = _validated_identifier(provider.provider_id, "Inference provider")
    output_type = _validated_identifier(provider.output_type, "Inference output type")
    classes = tuple(class_names)
    request = InferenceRequest(
        frame_index=frame.index,
        timestamp_seconds=frame.timestamp_seconds,
        image=frame.image,
        class_names=classes,
    )
    try:
        predictions = tuple(provider.predict(request))
    except Exception as error:
        raise InferenceError(f"Inference provider {provider_id!r} failed") from error

    return InferenceResult(
        provider_id=provider_id,
        output_type=output_type,
        frame_index=frame.index,
        timestamp_seconds=frame.timestamp_seconds,
        frame_size=(frame.width, frame.height),
        class_names=classes,
        predictions=predictions,
    )


def detections_to_boxes(
    result: InferenceResult[Detection],
    frame: MediaFrame,
    class_names: Sequence[str],
) -> tuple[Box, ...]:
    if result.output_type != OBJECT_DETECTION_OUTPUT:
        raise InferenceError(
            f"Inference output {result.output_type!r} cannot be materialized as detection boxes"
        )
    if result.frame_index != frame.index or result.frame_size != (frame.width, frame.height):
        raise InferenceError("Inference result does not match the source frame")
    classes = tuple(class_names)
    if result.class_names != classes:
        raise InferenceError("Inference result does not match the project class catalog")

    boxes: list[Box] = []
    for index, detection in enumerate(result.predictions):
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
                origin=AnnotationOrigin.INFERRED,
                inference_provider=result.provider_id,
                confidence=detection.confidence,
            )
        )
    return tuple(boxes)
