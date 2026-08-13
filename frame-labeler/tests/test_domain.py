from __future__ import annotations

import pytest

from frame_labeler.domain import AnnotationOrigin, Box, sampled_frame_indices


def test_sampled_frame_indices_stop_before_frame_count() -> None:
    assert list(sampled_frame_indices(frame_count=12, stride=3)) == [0, 3, 6, 9]


@pytest.mark.parametrize("frame_count,stride", [(0, 1), (1, 0), (1, -1)])
def test_sampled_frame_indices_reject_invalid_values(frame_count: int, stride: int) -> None:
    with pytest.raises(ValueError):
        list(sampled_frame_indices(frame_count=frame_count, stride=stride))


def test_box_clips_to_source_bounds() -> None:
    box = Box("box-1", 0, -2.0, 4.0, 102.0, 52.0, AnnotationOrigin.MANUAL)

    assert box.clipped(width=100, height=50) == Box(
        "box-1", 0, 0.0, 4.0, 100.0, 50.0, AnnotationOrigin.MANUAL
    )
