from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from enum import StrEnum


class BoxOrigin(StrEnum):
    MANUAL = "manual"
    COPIED = "copied"


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
    origin: BoxOrigin
    source_box_id: str | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Box ID cannot be empty")
        if self.class_id < 0:
            raise ValueError("Class ID cannot be negative")
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("Box width and height must be greater than zero")

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

    def to_yolo(
        self, image_width: int, image_height: int
    ) -> tuple[int, float, float, float, float]:
        if image_width <= 0 or image_height <= 0:
            raise ValueError("Image dimensions must be greater than zero")
        width = self.x_max - self.x_min
        height = self.y_max - self.y_min
        return (
            self.class_id,
            (self.x_min + width / 2.0) / image_width,
            (self.y_min + height / 2.0) / image_height,
            width / image_width,
            height / image_height,
        )

    @classmethod
    def from_yolo(
        cls,
        box_id: str,
        class_id: int,
        center_x: float,
        center_y: float,
        width: float,
        height: float,
        image_width: int,
        image_height: int,
        origin: BoxOrigin = BoxOrigin.MANUAL,
    ) -> Box:
        pixel_width = width * image_width
        pixel_height = height * image_height
        pixel_center_x = center_x * image_width
        pixel_center_y = center_y * image_height
        return cls(
            box_id,
            class_id,
            pixel_center_x - pixel_width / 2.0,
            pixel_center_y - pixel_height / 2.0,
            pixel_center_x + pixel_width / 2.0,
            pixel_center_y + pixel_height / 2.0,
            origin,
        )


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
