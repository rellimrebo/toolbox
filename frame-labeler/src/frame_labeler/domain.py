from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from enum import StrEnum


class AnnotationOrigin(StrEnum):
    MANUAL = "manual"
    COPIED = "copied"
    INFERRED = "inferred"


class ReviewState(StrEnum):
    UNREVIEWED = "unreviewed"
    DRAFT = "draft"
    REVIEWED = "reviewed"


@dataclass(frozen=True, slots=True)
class Box:
    id: str
    class_id: int
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    origin: AnnotationOrigin
    source_box_id: str | None = None
    inference_provider: str | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Box ID cannot be empty")
        if self.class_id < 0:
            raise ValueError("Class ID cannot be negative")
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("Box width and height must be greater than zero")
        if self.inference_provider is not None and (
            not self.inference_provider.strip()
            or "\n" in self.inference_provider
            or "\r" in self.inference_provider
        ):
            raise ValueError("Inference provider must be a non-empty, single-line identifier")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Inference confidence must be between zero and one")
        if self.origin is AnnotationOrigin.INFERRED and self.inference_provider is None:
            raise ValueError("Inferred boxes must identify their inference provider")
        if self.origin is AnnotationOrigin.MANUAL and (
            self.inference_provider is not None or self.confidence is not None
        ):
            raise ValueError("Manual boxes cannot contain inference provenance")

    @property
    def coordinates(self) -> tuple[float, float, float, float]:
        return self.x_min, self.y_min, self.x_max, self.y_max

    def clipped(self, width: int, height: int) -> Box:
        if width <= 0 or height <= 0:
            raise ValueError("Image dimensions must be greater than zero")
        x_min = min(max(self.x_min, 0.0), float(width))
        y_min = min(max(self.y_min, 0.0), float(height))
        x_max = min(max(self.x_max, 0.0), float(width))
        y_max = min(max(self.y_max, 0.0), float(height))
        return self.with_coordinates(x_min, y_min, x_max, y_max)

    def with_coordinates(self, x_min: float, y_min: float, x_max: float, y_max: float) -> Box:
        return replace(self, x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)

    def with_class(self, class_id: int) -> Box:
        return replace(self, class_id=class_id)


@dataclass(frozen=True, slots=True)
class FrameRecord:
    index: int
    timestamp_seconds: float
    width: int
    height: int
    state: ReviewState
    reviewed_at: str | None = None


def sampled_frame_indices(frame_count: int, stride: int) -> Iterator[int]:
    if frame_count <= 0:
        raise ValueError("Frame count must be greater than zero")
    if stride <= 0:
        raise ValueError("Stride must be greater than zero")
    return iter(range(0, frame_count, stride))
