from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from PIL import Image

from frame_labeler.domain import AnnotationOrigin, Box
from frame_labeler.export import (
    available_export_formats,
    box_from_yolo,
    box_to_yolo,
    export_dataset,
    export_yolo,
    get_exporter,
)
from frame_labeler.media import open_media
from frame_labeler.project import AnnotationProject


def test_box_converts_to_yolo_center_coordinates() -> None:
    box = Box("box-1", 2, 10.0, 20.0, 50.0, 60.0, AnnotationOrigin.MANUAL)

    assert box_to_yolo(box, image_width=100, image_height=80) == pytest.approx(
        (2, 0.3, 0.5, 0.4, 0.5)
    )


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
        AnnotationOrigin.MANUAL,
    )

    _, center_x, center_y, normalized_width, normalized_height = box_to_yolo(
        box, image_width, image_height
    )
    reconstructed = box_from_yolo(
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


def test_yolo_export_includes_only_reviewed_frames(tmp_path: Path) -> None:
    source_path = tmp_path / "image.png"
    Image.new("RGB", (100, 80), "white").save(source_path)
    project = AnnotationProject.create(
        tmp_path / "labels.sqlite3",
        source_path,
        class_names=["person", "vehicle"],
        split="train",
    )
    project.ensure_frame(0, timestamp_seconds=0.0, width=100, height=80)
    project.replace_boxes(0, [Box("box-1", 1, 10.0, 20.0, 50.0, 60.0, AnnotationOrigin.MANUAL)])
    project.mark_reviewed(0)
    output = tmp_path / "export"

    summary = export_dataset("yolo", project, open_media(source_path), output)

    assert summary.exported == 1
    image_paths = list((output / "images" / "train").glob("*.png"))
    label_paths = list((output / "labels" / "train").glob("*.txt"))
    assert len(image_paths) == 1
    assert len(label_paths) == 1
    assert label_paths[0].read_text(encoding="utf-8") == (
        "1 0.30000000 0.50000000 0.40000000 0.50000000\n"
    )
    assert '0: "person"' in (output / "dataset.yaml").read_text(encoding="utf-8")
    manifest = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
    assert manifest[0]["source_frame_index"] == 0
    project.close()


def test_export_dispatch_lists_formats_and_rejects_unknown_format() -> None:
    assert available_export_formats() == ("yolo",)
    assert get_exporter("yolo") is export_yolo

    with pytest.raises(ValueError, match="Unsupported export format"):
        get_exporter("unknown")


def test_reviewed_empty_frame_exports_image_without_label(tmp_path: Path) -> None:
    source_path = tmp_path / "image.png"
    Image.new("RGB", (20, 10), "white").save(source_path)
    project = AnnotationProject.create(
        tmp_path / "labels.sqlite3", source_path, class_names=["object"]
    )
    project.ensure_frame(0, timestamp_seconds=0.0, width=20, height=10)
    project.mark_reviewed(0)
    output = tmp_path / "export"

    export_yolo(project, open_media(source_path), output)

    assert len(list((output / "images" / "train").glob("*.png"))) == 1
    assert list((output / "labels" / "train").glob("*.txt")) == []
    project.close()
