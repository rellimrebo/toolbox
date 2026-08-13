from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QImage, QWheelEvent
from PySide6.QtWidgets import QToolBar

from frame_labeler.domain import Box, BoxOrigin, ReviewState
from frame_labeler.gui import AnnotationCanvas, LabelerWindow
from frame_labeler.media import MediaFrame, open_media
from frame_labeler.project import AnnotationProject


class _TwoFrameMedia:
    frame_count = 2

    def __init__(self, path: Path) -> None:
        self.path = path

    def read_frame(self, index: int) -> MediaFrame:
        if index not in {0, 1}:
            raise IndexError(index)
        return MediaFrame(index, index / 10.0, Image.new("RGB", (100, 80), "white"))

    def close(self) -> None:
        return


def _create_window_with_boxes(
    qtbot, tmp_path: Path, boxes: list[Box]
) -> tuple[LabelerWindow, AnnotationProject]:  # type: ignore[no-untyped-def]
    source_path = tmp_path / "image.png"
    Image.new("RGB", (100, 80), "white").save(source_path)
    project = AnnotationProject.create(
        tmp_path / "labels.sqlite3", source_path, class_names=["dog", "person"]
    )
    project.ensure_frame(0, timestamp_seconds=0.0, width=100, height=80)
    project.replace_boxes(0, boxes)
    window = LabelerWindow(project, open_media(source_path))
    qtbot.addWidget(window)
    window.show()
    return window, project


def test_canvas_draw_emits_source_pixel_coordinates(qtbot) -> None:  # type: ignore[no-untyped-def]
    canvas = AnnotationCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(500, 400)
    canvas.show()
    canvas.set_frame(QImage(100, 80, QImage.Format.Format_RGB888), (), ("object",))
    canvas.fit_to_window()
    created: list[tuple[float, float, float, float]] = []
    canvas.box_created.connect(created.append)
    start = canvas.mapFromScene(QPointF(10.0, 12.0))
    end = canvas.mapFromScene(QPointF(50.0, 60.0))

    qtbot.mousePress(canvas.viewport(), Qt.MouseButton.LeftButton, pos=start)
    qtbot.mouseMove(canvas.viewport(), pos=end)
    qtbot.mouseRelease(canvas.viewport(), Qt.MouseButton.LeftButton, pos=end)

    assert len(created) == 1
    assert created[0] == pytest.approx((10.0, 12.0, 50.0, 60.0), abs=0.5)


def test_selected_box_handles_stay_inside_repaint_bounds(qtbot) -> None:  # type: ignore[no-untyped-def]
    canvas = AnnotationCanvas()
    qtbot.addWidget(canvas)
    canvas.set_frame(
        QImage(100, 80, QImage.Format.Format_RGB888),
        [Box("box-1", 0, 10.0, 12.0, 50.0, 60.0, BoxOrigin.MANUAL)],
        ("object",),
    )

    item = canvas.annotation_items[0]
    handles = item.handle_rects()

    assert set(handles) == {
        "top_left",
        "top",
        "top_right",
        "right",
        "bottom_right",
        "bottom",
        "bottom_left",
        "left",
    }
    assert all(item.rect().contains(handle) for handle in handles.values())


def test_box_label_is_included_in_item_repaint_bounds(qtbot) -> None:  # type: ignore[no-untyped-def]
    canvas = AnnotationCanvas()
    qtbot.addWidget(canvas)
    canvas.set_frame(
        QImage(100, 80, QImage.Format.Format_RGB888),
        [Box("box-1", 0, 10.0, 12.0, 14.0, 60.0, BoxOrigin.MANUAL)],
        ("long-class-name",),
    )

    item = canvas.annotation_items[0]

    assert item.boundingRect().contains(item.label_rect())
    assert item.label_rect().right() > item.rect().right()


def test_labeler_window_loads_image_project(qtbot, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    source_path = tmp_path / "image.png"
    Image.new("RGB", (100, 80), "white").save(source_path)
    project = AnnotationProject.create(
        tmp_path / "labels.sqlite3", source_path, class_names=["object"]
    )
    media = open_media(source_path)
    window = LabelerWindow(project, media)
    qtbot.addWidget(window)

    assert window.current_frame_index == 0
    assert window.windowTitle() == "Frame Labeler - image.png"
    assert window.frame_status_text.startswith("Frame 0")
    assert window.canvas.boxes() == ()
    assert window.box_list.count() == 0

    window.close()


def test_next_frame_carries_draft_boxes_without_review(qtbot, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    source_path = tmp_path / "video-source.png"
    Image.new("RGB", (100, 80), "white").save(source_path)
    project = AnnotationProject.create(
        tmp_path / "labels.sqlite3", source_path, class_names=["object"]
    )
    window = LabelerWindow(project, _TwoFrameMedia(source_path))
    qtbot.addWidget(window)
    window.canvas.box_created.emit((10.0, 12.0, 30.0, 42.0))

    window.next_frame()

    assert window.current_frame_index == 1
    copied_box = window.canvas.boxes()[0]
    assert copied_box.coordinates == (10.0, 12.0, 30.0, 42.0)
    assert copied_box.origin is BoxOrigin.COPIED
    assert project.get_frame(1).state is ReviewState.DRAFT
    window.close()


def test_next_frame_seeds_previously_visited_unreviewed_frame(qtbot, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    source_path = tmp_path / "video-source.png"
    Image.new("RGB", (100, 80), "white").save(source_path)
    project = AnnotationProject.create(
        tmp_path / "labels.sqlite3", source_path, class_names=["object"]
    )
    window = LabelerWindow(project, _TwoFrameMedia(source_path))
    qtbot.addWidget(window)
    window.next_frame()
    window.previous_frame()
    window.canvas.box_created.emit((10.0, 12.0, 30.0, 42.0))

    window.next_frame()

    assert window.current_frame_index == 1
    assert window.canvas.boxes()[0].coordinates == (10.0, 12.0, 30.0, 42.0)
    assert project.get_frame(1).state is ReviewState.DRAFT
    window.close()


def test_initial_fit_uses_visible_canvas_size(qtbot, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    window, _project = _create_window_with_boxes(qtbot, tmp_path, [])
    qtbot.waitUntil(lambda: window.canvas.viewport().width() > 500)

    image_left = window.canvas.mapFromScene(QPointF(0.0, 0.0)).x()
    image_right = window.canvas.mapFromScene(QPointF(100.0, 0.0)).x()

    assert image_right - image_left > window.canvas.viewport().width() * 0.5
    window.close()


def test_selecting_reviewed_box_does_not_change_review_state(qtbot, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    source_path = tmp_path / "image.png"
    Image.new("RGB", (100, 80), "white").save(source_path)
    project = AnnotationProject.create(
        tmp_path / "labels.sqlite3", source_path, class_names=["object"]
    )
    project.ensure_frame(0, timestamp_seconds=0.0, width=100, height=80)
    project.replace_boxes(0, [Box("box-1", 0, 10.0, 10.0, 40.0, 40.0, BoxOrigin.MANUAL)])
    project.mark_reviewed(0)
    media = open_media(source_path)
    window = LabelerWindow(project, media)
    qtbot.addWidget(window)
    window.show()
    point = window.canvas.mapFromScene(QPointF(25.0, 25.0))

    qtbot.mouseClick(window.canvas.viewport(), Qt.MouseButton.LeftButton, pos=point)

    assert project.get_frame(0).state is ReviewState.REVIEWED
    window.close()


def test_box_list_selects_overlapping_box_by_stable_order(qtbot, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    boxes = [
        Box("z-box", 0, 10.0, 10.0, 60.0, 60.0, BoxOrigin.MANUAL),
        Box("a-box", 1, 20.0, 20.0, 70.0, 70.0, BoxOrigin.MANUAL),
    ]
    window, _project = _create_window_with_boxes(qtbot, tmp_path, boxes)

    assert window.box_list.count() == 2
    assert window.box_list.item(0).data(Qt.ItemDataRole.UserRole) == "z-box"

    window.box_list.setCurrentRow(1)

    assert window.canvas.selected_box_id() == "a-box"
    window.close()


def test_box_selection_cycles_and_wraps(qtbot, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    boxes = [
        Box("box-1", 0, 10.0, 10.0, 30.0, 30.0, BoxOrigin.MANUAL),
        Box("box-2", 1, 20.0, 20.0, 40.0, 40.0, BoxOrigin.MANUAL),
    ]
    window, _project = _create_window_with_boxes(qtbot, tmp_path, boxes)

    window.box_list.setCurrentRow(0)
    window.select_previous_box()
    assert window.canvas.selected_box_id() == "box-2"

    window.select_next_box()
    assert window.canvas.selected_box_id() == "box-1"

    window.canvas.scene().clearSelection()
    window.select_previous_box()
    assert window.canvas.selected_box_id() == "box-2"
    window.close()


def test_nudging_selected_box_updates_persistent_source_coordinates(qtbot, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    box = Box("box-1", 0, 10.0, 12.0, 30.0, 42.0, BoxOrigin.MANUAL)
    window, project = _create_window_with_boxes(qtbot, tmp_path, [box])
    window.box_list.setCurrentRow(0)

    window.nudge_selected_box(-1.0, 0.0)

    assert project.get_boxes(0)[0].coordinates == (9.0, 12.0, 29.0, 42.0)
    assert project.get_frame(0).state is ReviewState.DRAFT
    window.close()


def test_vim_style_shortcuts_nudge_by_one_and_ten_pixels(qtbot, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    box = Box("box-1", 0, 20.0, 20.0, 40.0, 40.0, BoxOrigin.MANUAL)
    window, project = _create_window_with_boxes(qtbot, tmp_path, [box])
    window.box_list.setCurrentRow(0)

    window.canvas.setFocus()
    qtbot.keyClick(window.canvas, Qt.Key.Key_H)
    qtbot.keyClick(window.canvas, Qt.Key.Key_L, Qt.KeyboardModifier.ShiftModifier)

    assert project.get_boxes(0)[0].coordinates == (29.0, 20.0, 49.0, 40.0)
    window.close()


def test_right_drag_pans_without_changing_boxes(qtbot) -> None:  # type: ignore[no-untyped-def]
    box = Box("box-1", 0, 100.0, 100.0, 200.0, 200.0, BoxOrigin.MANUAL)
    canvas = AnnotationCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(320, 240)
    canvas.set_frame(QImage(1000, 800, QImage.Format.Format_RGB888), [box], ("object",))
    canvas.fit_to_window()
    canvas.scale(4.0, 4.0)
    canvas.show()
    before_scroll = (canvas.horizontalScrollBar().value(), canvas.verticalScrollBar().value())
    movement = QPointF(40.0, 30.0).toPoint()

    qtbot.mousePress(canvas.viewport(), Qt.MouseButton.RightButton, pos=canvas.rect().center())
    qtbot.mouseMove(canvas.viewport(), pos=canvas.rect().center() - movement)
    qtbot.mouseRelease(
        canvas.viewport(),
        Qt.MouseButton.RightButton,
        pos=canvas.rect().center() - movement,
    )

    after_scroll = (canvas.horizontalScrollBar().value(), canvas.verticalScrollBar().value())
    assert after_scroll != before_scroll
    assert canvas.boxes() == (box,)


def test_left_drag_moves_box_in_source_pixels_while_zoomed(qtbot) -> None:  # type: ignore[no-untyped-def]
    box = Box("box-1", 0, 400.0, 300.0, 500.0, 400.0, BoxOrigin.MANUAL)
    canvas = AnnotationCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(500, 400)
    canvas.set_frame(QImage(1000, 800, QImage.Format.Format_RGB888), [box], ("object",))
    canvas.fit_to_window()
    canvas.scale(2.0, 2.0)
    canvas.show()
    start = canvas.mapFromScene(QPointF(450.0, 350.0))
    end = canvas.mapFromScene(QPointF(460.0, 365.0))

    qtbot.mousePress(canvas.viewport(), Qt.MouseButton.LeftButton, pos=start)
    qtbot.mouseMove(canvas.viewport(), pos=end)
    qtbot.mouseRelease(canvas.viewport(), Qt.MouseButton.LeftButton, pos=end)

    assert canvas.boxes()[0].coordinates == pytest.approx((410.0, 315.0, 510.0, 415.0), abs=1.0)


def test_edge_handle_resizes_box_at_zoom(qtbot) -> None:  # type: ignore[no-untyped-def]
    box = Box("box-1", 0, 400.0, 300.0, 500.0, 400.0, BoxOrigin.MANUAL)
    canvas = AnnotationCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(500, 400)
    canvas.set_frame(QImage(1000, 800, QImage.Format.Format_RGB888), [box], ("object",))
    canvas.fit_to_window()
    canvas.scale(2.0, 2.0)
    canvas.show()
    canvas.select_box("box-1")
    start = canvas.mapFromScene(canvas.annotation_items[0].handle_rects()["right"].center())
    end = canvas.mapFromScene(QPointF(540.0, 350.0))

    qtbot.mousePress(canvas.viewport(), Qt.MouseButton.LeftButton, pos=start)
    qtbot.mouseMove(canvas.viewport(), pos=end)
    qtbot.mouseRelease(canvas.viewport(), Qt.MouseButton.LeftButton, pos=end)

    assert canvas.boxes()[0].coordinates == pytest.approx((400.0, 300.0, 540.0, 400.0), abs=1.0)


def test_high_resolution_scroll_event_zooms_around_pointer(qtbot) -> None:  # type: ignore[no-untyped-def]
    canvas = AnnotationCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(500, 400)
    canvas.set_frame(QImage(1000, 800, QImage.Format.Format_RGB888), (), ("object",))
    canvas.fit_to_window()
    canvas.show()
    position = QPointF(220.0, 180.0)
    before_scale = canvas.transform().m11()
    before_scene_position = canvas.mapToScene(position.toPoint())
    event = QWheelEvent(
        position,
        QPointF(canvas.viewport().mapToGlobal(position.toPoint())),
        QPoint(0, 10),
        QPoint(),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.ScrollUpdate,
        False,
    )

    canvas.wheelEvent(event)

    after_scene_position = canvas.mapToScene(position.toPoint())
    assert canvas.transform().m11() > before_scale
    assert after_scene_position.x() == pytest.approx(before_scene_position.x(), abs=1.0)
    assert after_scene_position.y() == pytest.approx(before_scene_position.y(), abs=1.0)


def test_toolbar_omits_one_hundred_percent_zoom(qtbot, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    window, _project = _create_window_with_boxes(qtbot, tmp_path, [])

    toolbar = window.findChild(QToolBar, "Labeling")

    assert toolbar is not None
    assert "100%" not in [action.text() for action in toolbar.actions()]
    window.close()
