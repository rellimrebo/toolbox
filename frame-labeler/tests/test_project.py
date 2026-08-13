from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from frame_labeler.domain import AnnotationOrigin, Box, ReviewState
from frame_labeler.project import AnnotationProject, SourceMismatchError


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "source.bin"
    path.write_bytes(b"source-media")
    return path


def test_project_persists_classes_frame_state_and_boxes(tmp_path: Path, source: Path) -> None:
    project_path = tmp_path / "labels.sqlite3"
    project = AnnotationProject.create(
        project_path,
        source,
        class_names=["person", "vehicle"],
        stride=5,
        split="train",
    )
    frame = project.ensure_frame(0, timestamp_seconds=0.0, width=1920, height=1080)
    box = Box("box-1", 1, 10.0, 20.0, 110.0, 220.0, AnnotationOrigin.MANUAL)
    project.replace_boxes(frame.index, [box])
    project.mark_reviewed(frame.index)
    project.last_frame_index = frame.index
    project.close()

    reopened = AnnotationProject.open(project_path, source)

    assert reopened.class_names == ("person", "vehicle")
    assert reopened.stride == 5
    assert reopened.split == "train"
    assert reopened.last_frame_index == 0
    assert reopened.get_frame(0).state is ReviewState.REVIEWED
    assert reopened.get_boxes(0) == (box,)
    reopened.close()


def test_forward_navigation_copies_reviewed_boxes_as_independent_drafts(
    tmp_path: Path, source: Path
) -> None:
    project = AnnotationProject.create(
        tmp_path / "labels.sqlite3", source, class_names=["person"], stride=3
    )
    project.ensure_frame(0, timestamp_seconds=0.0, width=100, height=80)
    original = Box("box-1", 0, 10.0, 12.0, 30.0, 42.0, AnnotationOrigin.MANUAL)
    project.replace_boxes(0, [original])
    project.mark_reviewed(0)

    project.ensure_frame(3, timestamp_seconds=0.1, width=100, height=80)
    assert project.carry_forward_boxes(0, 3) is True
    copied_frame = project.get_frame(3)
    copied_box = project.get_boxes(3)[0]

    assert copied_frame.state is ReviewState.DRAFT
    assert copied_box.id != original.id
    assert copied_box.origin is AnnotationOrigin.COPIED
    assert copied_box.source_box_id == original.id
    assert copied_box.coordinates == original.coordinates

    edited = copied_box.with_coordinates(11.0, 12.0, 31.0, 42.0)
    project.replace_boxes(3, [edited])
    assert project.get_boxes(0) == (original,)
    project.close()


def test_forward_navigation_copies_draft_boxes(tmp_path: Path, source: Path) -> None:
    project = AnnotationProject.create(
        tmp_path / "labels.sqlite3", source, class_names=["person"], stride=3
    )
    project.ensure_frame(0, timestamp_seconds=0.0, width=100, height=80)
    original = Box("box-1", 0, 10.0, 12.0, 30.0, 42.0, AnnotationOrigin.MANUAL)
    project.replace_boxes(0, [original])

    project.ensure_frame(3, timestamp_seconds=0.1, width=100, height=80)
    assert project.carry_forward_boxes(0, 3) is True
    copied_frame = project.get_frame(3)

    assert project.get_frame(0).state is ReviewState.DRAFT
    assert copied_frame.state is ReviewState.DRAFT
    copied_box = project.get_boxes(3)[0]
    assert copied_box.origin is AnnotationOrigin.COPIED
    assert copied_box.source_box_id == original.id
    assert copied_box.coordinates == original.coordinates
    project.close()


def test_revisiting_empty_unreviewed_frame_carries_new_previous_boxes(
    tmp_path: Path, source: Path
) -> None:
    project = AnnotationProject.create(tmp_path / "labels.sqlite3", source, class_names=["person"])
    project.ensure_frame(0, timestamp_seconds=0.0, width=100, height=80)
    project.ensure_frame(1, timestamp_seconds=0.1, width=100, height=80)
    original = Box("box-1", 0, 10.0, 12.0, 30.0, 42.0, AnnotationOrigin.MANUAL)
    project.replace_boxes(0, [original])

    project.ensure_frame(1, timestamp_seconds=0.1, width=100, height=80)
    assert project.carry_forward_boxes(0, 1) is True
    revisited = project.get_frame(1)

    assert revisited.state is ReviewState.DRAFT
    copied_box = project.get_boxes(1)[0]
    assert copied_box.origin is AnnotationOrigin.COPIED
    assert copied_box.source_box_id == original.id
    assert copied_box.coordinates == original.coordinates
    project.close()


def test_revisiting_intentionally_emptied_draft_does_not_restore_boxes(
    tmp_path: Path, source: Path
) -> None:
    project = AnnotationProject.create(tmp_path / "labels.sqlite3", source, class_names=["person"])
    project.ensure_frame(0, timestamp_seconds=0.0, width=100, height=80)
    project.ensure_frame(1, timestamp_seconds=0.1, width=100, height=80)
    project.replace_boxes(1, [])
    project.replace_boxes(
        0,
        [Box("box-1", 0, 10.0, 12.0, 30.0, 42.0, AnnotationOrigin.MANUAL)],
    )

    assert project.carry_forward_boxes(0, 1) is False
    revisited = project.get_frame(1)

    assert revisited.state is ReviewState.DRAFT
    assert project.get_boxes(1) == ()
    project.close()


def test_revisiting_reviewed_empty_frame_does_not_restore_boxes(
    tmp_path: Path, source: Path
) -> None:
    project = AnnotationProject.create(tmp_path / "labels.sqlite3", source, class_names=["person"])
    project.ensure_frame(0, timestamp_seconds=0.0, width=100, height=80)
    project.ensure_frame(1, timestamp_seconds=0.1, width=100, height=80)
    project.mark_reviewed(1)
    project.replace_boxes(
        0,
        [Box("box-1", 0, 10.0, 12.0, 30.0, 42.0, AnnotationOrigin.MANUAL)],
    )

    assert project.carry_forward_boxes(0, 1) is False
    revisited = project.get_frame(1)

    assert revisited.state is ReviewState.REVIEWED
    assert project.get_boxes(1) == ()
    project.close()


def test_first_frame_starts_empty_without_a_preceding_frame(tmp_path: Path, source: Path) -> None:
    project = AnnotationProject.create(tmp_path / "labels.sqlite3", source, class_names=["person"])

    frame = project.ensure_frame(0, timestamp_seconds=0.0, width=100, height=80)

    assert frame.state is ReviewState.UNREVIEWED
    assert project.get_boxes(0) == ()
    project.close()


def test_frame_creation_does_not_apply_box_propagation_policy(tmp_path: Path, source: Path) -> None:
    project = AnnotationProject.create(tmp_path / "labels.sqlite3", source, class_names=["person"])
    project.ensure_frame(0, timestamp_seconds=0.0, width=100, height=80)
    project.replace_boxes(
        0,
        [Box("box-1", 0, 10.0, 12.0, 30.0, 42.0, AnnotationOrigin.MANUAL)],
    )

    frame = project.ensure_frame(1, timestamp_seconds=0.1, width=100, height=80)

    assert frame.state is ReviewState.UNREVIEWED
    assert project.get_boxes(1) == ()
    project.close()


def test_box_carry_forward_requires_consecutive_sampled_frames(
    tmp_path: Path, source: Path
) -> None:
    project = AnnotationProject.create(
        tmp_path / "labels.sqlite3", source, class_names=["person"], stride=2
    )
    project.ensure_frame(0, timestamp_seconds=0.0, width=100, height=80)
    project.ensure_frame(4, timestamp_seconds=0.2, width=100, height=80)

    with pytest.raises(ValueError, match="consecutive sampled frames"):
        project.carry_forward_boxes(0, 4)

    project.close()


def test_carry_forward_from_another_connection_does_not_overwrite_started_work(
    tmp_path: Path, source: Path
) -> None:
    project_path = tmp_path / "labels.sqlite3"
    editor = AnnotationProject.create(project_path, source, class_names=["person"])
    editor.ensure_frame(0, timestamp_seconds=0.0, width=100, height=80)
    editor.replace_boxes(
        0,
        [Box("source-1", 0, 10.0, 12.0, 30.0, 42.0, AnnotationOrigin.MANUAL)],
    )
    editor.ensure_frame(1, timestamp_seconds=0.1, width=100, height=80)
    worker = AnnotationProject.open(project_path, source)
    editor.replace_boxes(
        1,
        [Box("manual-1", 0, 1.0, 2.0, 10.0, 12.0, AnnotationOrigin.MANUAL)],
    )

    assert worker.carry_forward_boxes(0, 1) is False
    assert worker.get_boxes(1)[0].id == "manual-1"
    worker.close()
    editor.close()


def test_inference_seeds_only_empty_unreviewed_frame(tmp_path: Path, source: Path) -> None:
    project = AnnotationProject.create(tmp_path / "labels.sqlite3", source, class_names=["person"])
    project.ensure_frame(0, timestamp_seconds=0.0, width=100, height=80)
    inferred = Box(
        "inferred-1",
        0,
        10.0,
        12.0,
        30.0,
        42.0,
        AnnotationOrigin.INFERRED,
        inference_provider="stub-detector/v1",
        confidence=0.8,
    )

    assert project.seed_inferred_boxes(0, [inferred]) is True
    assert project.seed_inferred_boxes(0, []) is False
    assert project.get_frame(0).state is ReviewState.DRAFT
    assert project.get_boxes(0) == (inferred,)
    project.close()


def test_inference_does_not_overwrite_intentionally_empty_frame(
    tmp_path: Path, source: Path
) -> None:
    project = AnnotationProject.create(tmp_path / "labels.sqlite3", source, class_names=["person"])
    project.ensure_frame(0, timestamp_seconds=0.0, width=100, height=80)
    project.mark_reviewed(0)
    inferred = Box(
        "inferred-1",
        0,
        10.0,
        12.0,
        30.0,
        42.0,
        AnnotationOrigin.INFERRED,
        inference_provider="stub-detector/v1",
        confidence=0.8,
    )

    assert project.seed_inferred_boxes(0, [inferred]) is False
    assert project.get_boxes(0) == ()
    assert project.get_frame(0).state is ReviewState.REVIEWED
    project.close()


def test_inference_from_another_connection_does_not_overwrite_started_work(
    tmp_path: Path, source: Path
) -> None:
    project_path = tmp_path / "labels.sqlite3"
    editor = AnnotationProject.create(project_path, source, class_names=["person"])
    editor.ensure_frame(0, timestamp_seconds=0.0, width=100, height=80)
    worker = AnnotationProject.open(project_path, source)
    editor.replace_boxes(
        0,
        [Box("manual-1", 0, 1.0, 2.0, 10.0, 12.0, AnnotationOrigin.MANUAL)],
    )
    inferred = Box(
        "inferred-1",
        0,
        10.0,
        12.0,
        30.0,
        42.0,
        AnnotationOrigin.INFERRED,
        inference_provider="stub-detector/v1",
        confidence=0.8,
    )

    assert worker.seed_inferred_boxes(0, [inferred]) is False
    assert worker.get_boxes(0)[0].id == "manual-1"
    worker.close()
    editor.close()


def test_project_persists_inference_provenance(tmp_path: Path, source: Path) -> None:
    project_path = tmp_path / "labels.sqlite3"
    project = AnnotationProject.create(project_path, source, class_names=["person"])
    project.ensure_frame(0, timestamp_seconds=0.0, width=100, height=80)
    inferred = Box(
        "inferred-1",
        0,
        10.0,
        12.0,
        30.0,
        42.0,
        AnnotationOrigin.INFERRED,
        inference_provider="stub-detector/v1",
        confidence=0.8,
    )
    project.seed_inferred_boxes(0, [inferred])
    project.close()

    reopened = AnnotationProject.open(project_path, source)

    assert reopened.get_boxes(0) == (inferred,)
    reopened.close()


def test_carry_forward_preserves_inference_provenance(tmp_path: Path, source: Path) -> None:
    project = AnnotationProject.create(tmp_path / "labels.sqlite3", source, class_names=["person"])
    project.ensure_frame(0, timestamp_seconds=0.0, width=100, height=80)
    inferred = Box(
        "inferred-1",
        0,
        10.0,
        12.0,
        30.0,
        42.0,
        AnnotationOrigin.INFERRED,
        inference_provider="stub-detector/v1",
        confidence=0.8,
    )
    project.seed_inferred_boxes(0, [inferred])

    project.ensure_frame(1, 0.1, 100, 80)
    project.carry_forward_boxes(0, 1)

    copied = project.get_boxes(1)[0]
    assert copied.origin is AnnotationOrigin.COPIED
    assert copied.source_box_id == inferred.id
    assert copied.inference_provider == inferred.inference_provider
    assert copied.confidence == inferred.confidence
    project.close()


def test_open_migrates_v1_boxes_without_losing_annotations(tmp_path: Path, source: Path) -> None:
    project_path = tmp_path / "labels.sqlite3"
    project = AnnotationProject.create(project_path, source, class_names=["person"])
    project.ensure_frame(0, timestamp_seconds=0.0, width=100, height=80)
    project.replace_boxes(
        0,
        [Box("manual-1", 0, 10.0, 12.0, 30.0, 42.0, AnnotationOrigin.MANUAL)],
    )
    project.close()
    connection = sqlite3.connect(project_path)
    with connection:
        connection.execute("UPDATE metadata SET value = '1' WHERE key = 'schema_version'")
        connection.execute("ALTER TABLE boxes RENAME TO boxes_v2")
        connection.executescript(
            """
            CREATE TABLE boxes (
                id TEXT PRIMARY KEY,
                frame_index INTEGER NOT NULL REFERENCES frames(source_index) ON DELETE CASCADE,
                class_id INTEGER NOT NULL REFERENCES classes(id),
                x_min REAL NOT NULL,
                y_min REAL NOT NULL,
                x_max REAL NOT NULL,
                y_max REAL NOT NULL,
                origin TEXT NOT NULL CHECK (origin IN ('manual', 'copied')),
                source_box_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                CHECK (x_max > x_min),
                CHECK (y_max > y_min)
            );
            INSERT INTO boxes(
                id, frame_index, class_id, x_min, y_min, x_max, y_max,
                origin, source_box_id, created_at, updated_at
            )
            SELECT
                id, frame_index, class_id, x_min, y_min, x_max, y_max,
                origin, source_box_id, created_at, updated_at
            FROM boxes_v2;
            DROP TABLE boxes_v2;
            CREATE INDEX boxes_frame_index ON boxes(frame_index);
            """
        )
    connection.close()

    migrated = AnnotationProject.open(project_path, source)

    assert migrated.get_boxes(0) == (
        Box("manual-1", 0, 10.0, 12.0, 30.0, 42.0, AnnotationOrigin.MANUAL),
    )
    migrated.close()
    connection = sqlite3.connect(project_path)
    assert connection.execute(
        "SELECT value FROM metadata WHERE key = 'schema_version'"
    ).fetchone() == ("2",)
    assert {row[1] for row in connection.execute("PRAGMA table_info(boxes)")} >= {
        "inference_provider",
        "confidence",
    }
    connection.close()


def test_forward_navigation_from_reviewed_empty_frame_starts_empty(
    tmp_path: Path, source: Path
) -> None:
    project = AnnotationProject.create(tmp_path / "labels.sqlite3", source, class_names=["person"])
    project.ensure_frame(0, timestamp_seconds=0.0, width=100, height=80)
    project.mark_reviewed(0)

    frame = project.ensure_frame(1, timestamp_seconds=0.1, width=100, height=80)
    assert project.carry_forward_boxes(0, 1) is False

    assert frame.state is ReviewState.UNREVIEWED
    assert project.get_boxes(1) == ()
    project.close()


def test_mutating_a_reviewed_frame_returns_it_to_draft(tmp_path: Path, source: Path) -> None:
    project = AnnotationProject.create(tmp_path / "labels.sqlite3", source, class_names=["person"])
    project.ensure_frame(0, timestamp_seconds=0.0, width=100, height=80)
    project.mark_reviewed(0)

    project.replace_boxes(0, [Box("box-1", 0, 1.0, 2.0, 10.0, 12.0, AnnotationOrigin.MANUAL)])

    assert project.get_frame(0).state is ReviewState.DRAFT
    project.close()


def test_replacing_boxes_preserves_user_visible_order(tmp_path: Path, source: Path) -> None:
    project = AnnotationProject.create(tmp_path / "labels.sqlite3", source, class_names=["person"])
    project.ensure_frame(0, timestamp_seconds=0.0, width=100, height=80)
    boxes = (
        Box("z-box", 0, 1.0, 2.0, 10.0, 12.0, AnnotationOrigin.MANUAL),
        Box("a-box", 0, 20.0, 22.0, 30.0, 32.0, AnnotationOrigin.MANUAL),
    )

    project.replace_boxes(0, boxes)

    assert project.get_boxes(0) == boxes
    project.close()


def test_reviewed_empty_frame_is_distinct_from_unreviewed(tmp_path: Path, source: Path) -> None:
    project = AnnotationProject.create(tmp_path / "labels.sqlite3", source, class_names=["person"])
    project.ensure_frame(0, timestamp_seconds=0.0, width=100, height=80)
    project.ensure_frame(1, timestamp_seconds=0.1, width=100, height=80)
    project.mark_reviewed(0)

    assert project.get_frame(0).state is ReviewState.REVIEWED
    assert project.get_boxes(0) == ()
    assert project.get_frame(1).state is ReviewState.UNREVIEWED
    project.close()


def test_project_rejects_changed_source(tmp_path: Path, source: Path) -> None:
    project_path = tmp_path / "labels.sqlite3"
    AnnotationProject.create(project_path, source, class_names=["person"]).close()
    source.write_bytes(b"different-media")

    with pytest.raises(SourceMismatchError):
        AnnotationProject.open(project_path, source)


@pytest.mark.parametrize(
    "class_names",
    [[], [""], ["person", " person "], ["person\nvehicle"]],
)
def test_project_rejects_invalid_class_catalog(
    tmp_path: Path, source: Path, class_names: list[str]
) -> None:
    with pytest.raises(ValueError):
        AnnotationProject.create(tmp_path / "labels.sqlite3", source, class_names)
