from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from frame_labeler.domain import AnnotationOrigin, Box
from frame_labeler.media import MediaSource
from frame_labeler.project import AnnotationProject, ProjectError


@dataclass(frozen=True, slots=True)
class ExportSummary:
    exported: int
    failed: int = 0


ExportFunction = Callable[[AnnotationProject, MediaSource, str | Path], ExportSummary]


def box_to_yolo(
    box: Box, image_width: int, image_height: int
) -> tuple[int, float, float, float, float]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be greater than zero")
    width = box.x_max - box.x_min
    height = box.y_max - box.y_min
    return (
        box.class_id,
        (box.x_min + width / 2.0) / image_width,
        (box.y_min + height / 2.0) / image_height,
        width / image_width,
        height / image_height,
    )


def box_from_yolo(
    box_id: str,
    class_id: int,
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
    origin: AnnotationOrigin = AnnotationOrigin.MANUAL,
) -> Box:
    pixel_width = width * image_width
    pixel_height = height * image_height
    pixel_center_x = center_x * image_width
    pixel_center_y = center_y * image_height
    return Box(
        box_id,
        class_id,
        pixel_center_x - pixel_width / 2.0,
        pixel_center_y - pixel_height / 2.0,
        pixel_center_x + pixel_width / 2.0,
        pixel_center_y + pixel_height / 2.0,
        origin,
    )


def _safe_name(value: str) -> str:
    cleaned = "".join(character if character.isalnum() else "-" for character in value)
    return cleaned.strip("-") or "source"


def _dataset_yaml(class_names: tuple[str, ...]) -> str:
    names = "\n".join(
        f"  {class_id}: {json.dumps(name, ensure_ascii=False)}"
        for class_id, name in enumerate(class_names)
    )
    return f"path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n"


def _remove_previous_files(output: Path, marker: dict[str, Any]) -> None:
    for relative in marker.get("generated_files", []):
        candidate = (output / str(relative)).resolve()
        if output.resolve() not in candidate.parents:
            raise ProjectError("Export marker contains an unsafe path")
        if candidate.is_file():
            candidate.unlink()


def export_yolo(
    project: AnnotationProject, media: MediaSource, output_path: str | Path
) -> ExportSummary:
    output = Path(output_path).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    marker_path = output / ".frame-labeler-export.json"
    if marker_path.exists():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("project_id") != project.project_id:
            raise ProjectError("Export directory belongs to another Frame Labeler project")
        _remove_previous_files(output, marker)
    elif any(output.iterdir()):
        raise ProjectError("Export directory is not empty and has no Frame Labeler marker")

    image_dir = output / "images" / project.split
    label_dir = output / "labels" / project.split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    source_name = _safe_name(project.source_path.stem)
    source_digest = project.source_identity["sample_sha256"][:10]
    prefix = f"{source_name}-{source_digest}"
    generated: list[str] = []
    manifest: list[dict[str, Any]] = []

    for frame_record in project.iter_reviewed_frames():
        frame = media.read_frame(frame_record.index)
        filename = f"{prefix}-f{frame_record.index:09d}.png"
        image_path = image_dir / filename
        try:
            frame.image.save(image_path, format="PNG", optimize=False)
        finally:
            frame.image.close()
        generated.append(str(image_path.relative_to(output)))

        boxes = project.get_boxes(frame_record.index)
        if boxes:
            label_path = label_dir / f"{Path(filename).stem}.txt"
            lines = []
            for box in boxes:
                class_id, center_x, center_y, width, height = box_to_yolo(
                    box, frame_record.width, frame_record.height
                )
                lines.append(f"{class_id} {center_x:.8f} {center_y:.8f} {width:.8f} {height:.8f}")
            label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            generated.append(str(label_path.relative_to(output)))

        manifest.append(
            {
                "project_id": project.project_id,
                "source": str(project.source_path),
                "source_frame_index": frame_record.index,
                "timestamp_seconds": frame_record.timestamp_seconds,
                "width": frame_record.width,
                "height": frame_record.height,
                "split": project.split,
                "reviewed_at": frame_record.reviewed_at,
                "image": str(image_path.relative_to(output)),
            }
        )

    dataset_path = output / "dataset.yaml"
    dataset_path.write_text(_dataset_yaml(project.class_names), encoding="utf-8")
    generated.append(str(dataset_path.relative_to(output)))
    manifest_path = output / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in manifest),
        encoding="utf-8",
    )
    generated.append(str(manifest_path.relative_to(output)))
    marker_path.write_text(
        json.dumps(
            {"project_id": project.project_id, "generated_files": sorted(generated)},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return ExportSummary(exported=len(manifest))


_EXPORTERS: dict[str, ExportFunction] = {"yolo": export_yolo}


def available_export_formats() -> tuple[str, ...]:
    return tuple(sorted(_EXPORTERS))


def get_exporter(format_name: str) -> ExportFunction:
    try:
        return _EXPORTERS[format_name]
    except KeyError as error:
        raise ValueError(f"Unsupported export format: {format_name}") from error


def export_dataset(
    format_name: str,
    project: AnnotationProject,
    media: MediaSource,
    output_path: str | Path,
) -> ExportSummary:
    return get_exporter(format_name)(project, media, output_path)
