from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from frame_labeler.domain import Box, BoxOrigin, sampled_frame_indices


def test_sampled_frame_indices_stop_before_frame_count() -> None:
    assert list(sampled_frame_indices(frame_count=12, stride=3)) == [0, 3, 6, 9]


@pytest.mark.parametrize("frame_count,stride", [(0, 1), (1, 0), (1, -1)])
def test_sampled_frame_indices_reject_invalid_values(frame_count: int, stride: int) -> None:
    with pytest.raises(ValueError):
        list(sampled_frame_indices(frame_count=frame_count, stride=stride))


def test_box_clips_to_source_bounds() -> None:
    box = Box("box-1", 0, -2.0, 4.0, 102.0, 52.0, BoxOrigin.MANUAL)

    assert box.clipped(width=100, height=50) == Box(
        "box-1", 0, 0.0, 4.0, 100.0, 50.0, BoxOrigin.MANUAL
    )


def test_box_converts_to_yolo_center_coordinates() -> None:
    box = Box("box-1", 2, 10.0, 20.0, 50.0, 60.0, BoxOrigin.MANUAL)

    assert box.to_yolo(image_width=100, image_height=80) == pytest.approx((2, 0.3, 0.5, 0.4, 0.5))


@given(
    image_width=st.integers(min_value=1, max_value=8192),
    image_height=st.integers(min_value=1, max_value=8192),
    left=st.floats(min_value=0, max_value=0.9, allow_nan=False, allow_infinity=False),
    top=st.floats(min_value=0, max_value=0.9, allow_nan=False, allow_infinity=False),
    width=st.floats(min_value=0.0001, max_value=0.1, allow_nan=False, allow_infinity=False),
    height=st.floats(min_value=0.0001, max_value=0.1, allow_nan=False, allow_infinity=False),
)
def test_yolo_conversion_round_trips(
    image_width: int,
    image_height: int,
    left: float,
    top: float,
    width: float,
    height: float,
) -> None:
    right = min(1.0, left + width)
    bottom = min(1.0, top + height)
    box = Box(
        "box-1",
        0,
        left * image_width,
        top * image_height,
        right * image_width,
        bottom * image_height,
        BoxOrigin.MANUAL,
    )

    _, center_x, center_y, normalized_width, normalized_height = box.to_yolo(
        image_width, image_height
    )
    reconstructed = Box.from_yolo(
        "box-1",
        0,
        center_x,
        center_y,
        normalized_width,
        normalized_height,
        image_width,
        image_height,
    )

    assert math.isclose(reconstructed.x_min, box.x_min, abs_tol=1e-9)
    assert math.isclose(reconstructed.y_min, box.y_min, abs_tol=1e-9)
    assert math.isclose(reconstructed.x_max, box.x_max, abs_tol=1e-9)
    assert math.isclose(reconstructed.y_max, box.y_max, abs_tol=1e-9)
