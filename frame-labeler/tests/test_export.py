from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from frame_labeler.domain import Box, BoxOrigin
from frame_labeler.export import export_yolo
from frame_labeler.media import open_media
from frame_labeler.project import AnnotationProject


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
    project.replace_boxes(0, [Box("box-1", 1, 10.0, 20.0, 50.0, 60.0, BoxOrigin.MANUAL)])
    project.mark_reviewed(0)
    output = tmp_path / "export"

    summary = export_yolo(project, open_media(source_path), output)

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
