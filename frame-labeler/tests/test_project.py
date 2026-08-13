from __future__ import annotations

from pathlib import Path

import pytest

from frame_labeler.domain import Box, BoxOrigin, ReviewState
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
    box = Box("box-1", 1, 10.0, 20.0, 110.0, 220.0, BoxOrigin.MANUAL)
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
    original = Box("box-1", 0, 10.0, 12.0, 30.0, 42.0, BoxOrigin.MANUAL)
    project.replace_boxes(0, [original])
    project.mark_reviewed(0)

    copied_frame = project.ensure_frame(
        3,
        timestamp_seconds=0.1,
        width=100,
        height=80,
        previous_index=0,
    )
    copied_box = project.get_boxes(3)[0]

    assert copied_frame.state is ReviewState.DRAFT
    assert copied_box.id != original.id
    assert copied_box.origin is BoxOrigin.COPIED
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
    original = Box("box-1", 0, 10.0, 12.0, 30.0, 42.0, BoxOrigin.MANUAL)
    project.replace_boxes(0, [original])

    copied_frame = project.ensure_frame(
        3,
        timestamp_seconds=0.1,
        width=100,
        height=80,
        previous_index=0,
    )

    assert project.get_frame(0).state is ReviewState.DRAFT
    assert copied_frame.state is ReviewState.DRAFT
    copied_box = project.get_boxes(3)[0]
    assert copied_box.origin is BoxOrigin.COPIED
    assert copied_box.source_box_id == original.id
    assert copied_box.coordinates == original.coordinates
    project.close()


def test_revisiting_empty_unreviewed_frame_carries_new_previous_boxes(
    tmp_path: Path, source: Path
) -> None:
    project = AnnotationProject.create(tmp_path / "labels.sqlite3", source, class_names=["person"])
    project.ensure_frame(0, timestamp_seconds=0.0, width=100, height=80)
    project.ensure_frame(1, timestamp_seconds=0.1, width=100, height=80)
    original = Box("box-1", 0, 10.0, 12.0, 30.0, 42.0, BoxOrigin.MANUAL)
    project.replace_boxes(0, [original])

    revisited = project.ensure_frame(
        1,
        timestamp_seconds=0.1,
        width=100,
        height=80,
        previous_index=0,
    )

    assert revisited.state is ReviewState.DRAFT
    copied_box = project.get_boxes(1)[0]
    assert copied_box.origin is BoxOrigin.COPIED
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
        [Box("box-1", 0, 10.0, 12.0, 30.0, 42.0, BoxOrigin.MANUAL)],
    )

    revisited = project.ensure_frame(
        1,
        timestamp_seconds=0.1,
        width=100,
        height=80,
        previous_index=0,
    )

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
        [Box("box-1", 0, 10.0, 12.0, 30.0, 42.0, BoxOrigin.MANUAL)],
    )

    revisited = project.ensure_frame(
        1,
        timestamp_seconds=0.1,
        width=100,
        height=80,
        previous_index=0,
    )

    assert revisited.state is ReviewState.REVIEWED
    assert project.get_boxes(1) == ()
    project.close()


def test_first_frame_starts_empty_without_a_preceding_frame(tmp_path: Path, source: Path) -> None:
    project = AnnotationProject.create(tmp_path / "labels.sqlite3", source, class_names=["person"])

    frame = project.ensure_frame(0, timestamp_seconds=0.0, width=100, height=80)

    assert frame.state is ReviewState.UNREVIEWED
    assert project.get_boxes(0) == ()
    project.close()


def test_forward_navigation_from_reviewed_empty_frame_starts_empty(
    tmp_path: Path, source: Path
) -> None:
    project = AnnotationProject.create(tmp_path / "labels.sqlite3", source, class_names=["person"])
    project.ensure_frame(0, timestamp_seconds=0.0, width=100, height=80)
    project.mark_reviewed(0)

    frame = project.ensure_frame(
        1,
        timestamp_seconds=0.1,
        width=100,
        height=80,
        previous_index=0,
    )

    assert frame.state is ReviewState.UNREVIEWED
    assert project.get_boxes(1) == ()
    project.close()


def test_mutating_a_reviewed_frame_returns_it_to_draft(tmp_path: Path, source: Path) -> None:
    project = AnnotationProject.create(tmp_path / "labels.sqlite3", source, class_names=["person"])
    project.ensure_frame(0, timestamp_seconds=0.0, width=100, height=80)
    project.mark_reviewed(0)

    project.replace_boxes(0, [Box("box-1", 0, 1.0, 2.0, 10.0, 12.0, BoxOrigin.MANUAL)])

    assert project.get_frame(0).state is ReviewState.DRAFT
    project.close()


def test_replacing_boxes_preserves_user_visible_order(tmp_path: Path, source: Path) -> None:
    project = AnnotationProject.create(tmp_path / "labels.sqlite3", source, class_names=["person"])
    project.ensure_frame(0, timestamp_seconds=0.0, width=100, height=80)
    boxes = (
        Box("z-box", 0, 1.0, 2.0, 10.0, 12.0, BoxOrigin.MANUAL),
        Box("a-box", 0, 20.0, 22.0, 30.0, 32.0, BoxOrigin.MANUAL),
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
