from __future__ import annotations

import sys
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path

from PIL import Image
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QAction,
    QCloseEvent,
    QColor,
    QFontMetricsF,
    QImage,
    QKeyEvent,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QShowEvent,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSceneHoverEvent,
    QGraphicsSceneMouseEvent,
    QGraphicsView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QStyle,
    QStyleOptionGraphicsItem,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from frame_labeler.domain import Box, BoxOrigin, ReviewState
from frame_labeler.export import export_yolo
from frame_labeler.media import MediaFrame, MediaSource
from frame_labeler.project import AnnotationProject

BoxEditStarted = Callable[[], None]
BoxChanged = Callable[[], None]


def pil_to_qimage(image: Image.Image) -> QImage:
    rgb = image.convert("RGB")
    data = rgb.tobytes("raw", "RGB")
    return QImage(data, rgb.width, rgb.height, rgb.width * 3, QImage.Format.Format_RGB888).copy()


def _class_color(class_id: int) -> QColor:
    hue = (class_id * 137 + 18) % 360
    return QColor.fromHsv(hue, 210, 245)


class AnnotationItem(QGraphicsRectItem):
    _handle_size = 8.0

    def __init__(
        self,
        box: Box,
        class_name: str,
        is_draft: bool,
        edit_started: BoxEditStarted,
        changed: BoxChanged,
    ) -> None:
        super().__init__(QRectF(box.x_min, box.y_min, box.x_max - box.x_min, box.y_max - box.y_min))
        self.box = box
        self.class_name = class_name
        self.is_draft = is_draft
        self._edit_started = edit_started
        self._changed = changed
        self._resize_handle: str | None = None
        self._moving = False
        self._edit_announced = False
        self._press_scene_rect: QRectF | None = None
        self.setPen(QPen(Qt.PenStyle.NoPen))
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemIsMovable
        )
        self.setAcceptHoverEvents(True)
        self.setZValue(2.0)

    def _handle_radius(self) -> float:
        scene = self.scene()
        if scene is None or not scene.views():
            return self._handle_size
        scale = scene.views()[0].transform().m11()
        return self._handle_size / max(scale, 0.001)

    def handle_rects(self) -> dict[str, QRectF]:
        rect = self.rect()
        size = min(self._handle_radius(), rect.width() / 3.0, rect.height() / 3.0)
        half = size / 2.0
        return {
            "top_left": QRectF(rect.left(), rect.top(), size, size),
            "top": QRectF(rect.center().x() - half, rect.top(), size, size),
            "top_right": QRectF(rect.right() - size, rect.top(), size, size),
            "right": QRectF(rect.right() - size, rect.center().y() - half, size, size),
            "bottom_right": QRectF(rect.right() - size, rect.bottom() - size, size, size),
            "bottom": QRectF(rect.center().x() - half, rect.bottom() - size, size, size),
            "bottom_left": QRectF(rect.left(), rect.bottom() - size, size, size),
            "left": QRectF(rect.left(), rect.center().y() - half, size, size),
        }

    def _handle_at(self, position: QPointF) -> str | None:
        for name, rect in self.handle_rects().items():
            if rect.contains(position):
                return name
        return None

    def label_rect(self) -> QRectF:
        label_rect = QFontMetricsF(QApplication.font()).boundingRect(self.class_name)
        label_rect = label_rect.adjusted(-4.0, -2.0, 4.0, 2.0)
        label_rect.moveTopLeft(self.rect().topLeft())
        return label_rect

    def boundingRect(self) -> QRectF:
        return super().boundingRect().united(self.label_rect())

    def _handle_cursor(self, handle: str | None) -> Qt.CursorShape:
        if handle in {"top_left", "bottom_right"}:
            return Qt.CursorShape.SizeFDiagCursor
        if handle in {"top_right", "bottom_left"}:
            return Qt.CursorShape.SizeBDiagCursor
        if handle in {"left", "right"}:
            return Qt.CursorShape.SizeHorCursor
        if handle in {"top", "bottom"}:
            return Qt.CursorShape.SizeVerCursor
        return Qt.CursorShape.SizeAllCursor

    def hoverMoveEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        self.setCursor(self._handle_cursor(self._handle_at(event.pos())))
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        self.unsetCursor()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._resize_handle = self._handle_at(event.pos())
            self._moving = self._resize_handle is None
            self._edit_announced = False
            self._press_scene_rect = self.mapRectToScene(self.rect())
            if self._resize_handle is not None:
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if not self._edit_announced:
            self._edit_started()
            self._edit_announced = True
        if self._resize_handle is None:
            super().mouseMoveEvent(event)
            return
        rect = self.rect()
        position = event.pos()
        scene_rect = self.scene().sceneRect()
        position.setX(min(max(position.x(), scene_rect.left()), scene_rect.right()))
        position.setY(min(max(position.y(), scene_rect.top()), scene_rect.bottom()))
        candidate = QRectF(rect)
        if "left" in self._resize_handle:
            candidate.setLeft(min(position.x(), rect.right() - 1.0))
        if "right" in self._resize_handle:
            candidate.setRight(max(position.x(), rect.left() + 1.0))
        if "top" in self._resize_handle:
            candidate.setTop(min(position.y(), rect.bottom() - 1.0))
        if "bottom" in self._resize_handle:
            candidate.setBottom(max(position.y(), rect.top() + 1.0))
        self.setRect(candidate.intersected(self.scene().sceneRect()))
        event.accept()

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        resizing = self._resize_handle is not None
        if not resizing:
            super().mouseReleaseEvent(event)
        self._resize_handle = None
        current_rect = self.mapRectToScene(self.rect())
        changed = self._press_scene_rect is not None and current_rect != self._press_scene_rect
        self._press_scene_rect = None
        self._moving = False
        if changed:
            if not self._edit_announced:
                self._edit_started()
            self._commit_geometry()
            self._changed()
        self._edit_announced = False
        event.accept()

    def _commit_geometry(self) -> None:
        scene_bounds = self.scene().sceneRect()
        rect = self.mapRectToScene(self.rect()).normalized()
        width = min(rect.width(), scene_bounds.width())
        height = min(rect.height(), scene_bounds.height())
        left = min(max(rect.left(), scene_bounds.left()), scene_bounds.right() - width)
        top = min(max(rect.top(), scene_bounds.top()), scene_bounds.bottom() - height)
        rect = QRectF(left, top, width, height)
        self.setPos(0.0, 0.0)
        self.setRect(rect)
        self.box = self.box.with_coordinates(rect.left(), rect.top(), rect.right(), rect.bottom())

    def can_move_by(self, dx: float, dy: float) -> bool:
        return self._translated_rect(dx, dy) != self.mapRectToScene(self.rect())

    def move_by(self, dx: float, dy: float) -> None:
        rect = self._translated_rect(dx, dy)
        self.setPos(0.0, 0.0)
        self.setRect(rect)
        self.box = self.box.with_coordinates(rect.left(), rect.top(), rect.right(), rect.bottom())

    def _translated_rect(self, dx: float, dy: float) -> QRectF:
        scene_bounds = self.scene().sceneRect()
        rect = self.mapRectToScene(self.rect())
        left = min(max(rect.left() + dx, scene_bounds.left()), scene_bounds.right() - rect.width())
        top = min(max(rect.top() + dy, scene_bounds.top()), scene_bounds.bottom() - rect.height())
        return QRectF(left, top, rect.width(), rect.height())

    def set_class(self, class_id: int, class_name: str) -> None:
        self.prepareGeometryChange()
        self.box = self.box.with_class(class_id)
        self.class_name = class_name
        self.update()

    def set_review_state(self, state: ReviewState) -> None:
        self.is_draft = state is ReviewState.DRAFT
        self.update()

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        del widget
        color = _class_color(self.box.class_id)
        pen = QPen(color, 2.0)
        pen.setCosmetic(True)
        if self.is_draft and self.box.origin is BoxOrigin.COPIED:
            pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(self.rect())

        label_rect = self.label_rect()
        painter.fillRect(label_rect, color)
        painter.setPen(QColor("#111827"))
        painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, self.class_name)

        if option.state & QStyle.StateFlag.State_Selected:
            painter.setPen(QPen(QColor("white"), 1.0))
            painter.setBrush(color)
            for handle in self.handle_rects().values():
                painter.drawRect(handle)


class AnnotationCanvas(QGraphicsView):
    box_created = Signal(tuple)
    box_edit_started = Signal()
    boxes_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._draft_item: QGraphicsRectItem | None = None
        self._annotation_items: list[AnnotationItem] = []
        self._draw_start: QPointF | None = None
        self._panning = False
        self._space_pressed = False
        self._pan_position = QPoint()
        self._class_names: tuple[str, ...] = ()
        self._review_state = ReviewState.UNREVIEWED
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setBackgroundBrush(QColor("#111827"))
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setToolTip("Wheel: zoom | Space-drag or right-drag: pan | Left-drag empty space: draw")

    def set_frame(
        self,
        image: QImage,
        boxes: Sequence[Box],
        class_names: Sequence[str],
        review_state: ReviewState = ReviewState.UNREVIEWED,
    ) -> None:
        self.clear_frame()
        self._class_names = tuple(class_names)
        self._review_state = review_state
        self._pixmap_item = self._scene.addPixmap(QPixmap.fromImage(image))
        self._pixmap_item.setZValue(0.0)
        self._scene.setSceneRect(0.0, 0.0, float(image.width()), float(image.height()))
        self.set_boxes(boxes)

    def set_boxes(self, boxes: Sequence[Box]) -> None:
        for item in self._annotation_items:
            self._scene.removeItem(item)
        self._annotation_items.clear()
        for box in boxes:
            self._add_box_item(box)

    def _add_box_item(self, box: Box) -> None:
        item = AnnotationItem(
            box,
            self._class_names[box.class_id],
            self._review_state is ReviewState.DRAFT,
            self.box_edit_started.emit,
            self.boxes_changed.emit,
        )
        self._scene.addItem(item)
        self._annotation_items.append(item)

    def boxes(self) -> tuple[Box, ...]:
        return tuple(item.box for item in self._annotation_items)

    @property
    def annotation_items(self) -> tuple[AnnotationItem, ...]:
        return tuple(self._annotation_items)

    def selected_items(self) -> list[AnnotationItem]:
        return [item for item in self._annotation_items if item.isSelected()]

    def selected_box_id(self) -> str | None:
        selected = self.selected_items()
        return selected[0].box.id if selected else None

    def select_box(self, box_id: str, *, ensure_visible: bool = False) -> bool:
        selected: AnnotationItem | None = None
        self._scene.clearSelection()
        for item in self._annotation_items:
            if item.box.id == box_id:
                item.setSelected(True)
                selected = item
                break
        if selected is not None and ensure_visible:
            self.ensureVisible(selected, 40, 40)
        return selected is not None

    def nudge_selected(self, dx: float, dy: float) -> bool:
        selected = self.selected_items()
        if not selected or not any(item.can_move_by(dx, dy) for item in selected):
            return False
        self.box_edit_started.emit()
        for item in selected:
            item.move_by(dx, dy)
        self.boxes_changed.emit()
        return True

    def delete_selected(self) -> bool:
        selected = self.selected_items()
        for item in selected:
            self._scene.removeItem(item)
            self._annotation_items.remove(item)
        if selected:
            self.boxes_changed.emit()
        return bool(selected)

    def clear_frame(self) -> None:
        for item in self._annotation_items:
            self._scene.removeItem(item)
        self._annotation_items.clear()
        if self._draft_item is not None:
            self._scene.removeItem(self._draft_item)
            self._draft_item = None
        if self._pixmap_item is not None:
            self._scene.removeItem(self._pixmap_item)
            self._pixmap_item = None
        self._draw_start = None

    def reclassify_selected(self, class_id: int) -> bool:
        selected = self.selected_items()
        for item in selected:
            item.set_class(class_id, self._class_names[class_id])
        if selected:
            self.boxes_changed.emit()
        return bool(selected)

    def set_review_state(self, state: ReviewState) -> None:
        self._review_state = state
        for item in self._annotation_items:
            item.set_review_state(state)

    def fit_to_window(self) -> None:
        if not self._scene.sceneRect().isEmpty():
            self.resetTransform()
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._scene.sceneRect().isEmpty():
            return
        before = self.mapToScene(event.position().toPoint())
        current = self.transform().m11()
        angle_delta = event.angleDelta().y()
        pixel_delta = event.pixelDelta().y()
        if angle_delta:
            steps = angle_delta / 120.0
        elif pixel_delta:
            steps = pixel_delta / 40.0
        else:
            event.accept()
            return
        factor = 1.2**steps
        target = min(max(current * factor, 0.05), 32.0)
        self.scale(target / current, target / current)
        after = self.mapToScene(event.position().toPoint())
        delta = after - before
        self.translate(delta.x(), delta.y())
        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pressed = True
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return
        if event.modifiers() in {
            Qt.KeyboardModifier.NoModifier,
            Qt.KeyboardModifier.ShiftModifier,
        }:
            direction = {
                Qt.Key.Key_H: (-1.0, 0.0),
                Qt.Key.Key_J: (0.0, 1.0),
                Qt.Key.Key_K: (0.0, -1.0),
                Qt.Key.Key_L: (1.0, 0.0),
            }.get(Qt.Key(event.key()))
            if direction is not None:
                step = 10.0 if event.modifiers() == Qt.KeyboardModifier.ShiftModifier else 1.0
                self.nudge_selected(direction[0] * step, direction[1] * step)
                event.accept()
                return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pressed = False
            self.unsetCursor()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() in {Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton} or (
            event.button() == Qt.MouseButton.LeftButton and self._space_pressed
        ):
            self._panning = True
            self._pan_position = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            item = self.itemAt(event.position().toPoint())
            if not isinstance(item, AnnotationItem):
                start = self.mapToScene(event.position().toPoint())
                if self._scene.sceneRect().contains(start):
                    self._scene.clearSelection()
                    self._draw_start = start
                    self._draft_item = self._scene.addRect(
                        QRectF(start, start), QPen(QColor("white"), 1.0, Qt.PenStyle.DashLine)
                    )
                    self._draft_item.setZValue(3.0)
                    event.accept()
                    return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._panning:
            current = event.position().toPoint()
            delta = current - self._pan_position
            self._pan_position = current
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        if self._draw_start is not None and self._draft_item is not None:
            end = self.mapToScene(event.position().toPoint())
            end.setX(min(max(end.x(), 0.0), self._scene.sceneRect().width()))
            end.setY(min(max(end.y(), 0.0), self._scene.sceneRect().height()))
            self._draft_item.setRect(QRectF(self._draw_start, end).normalized())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._panning:
            self._panning = False
            self.setCursor(
                Qt.CursorShape.OpenHandCursor if self._space_pressed else Qt.CursorShape.ArrowCursor
            )
            event.accept()
            return
        if self._draw_start is not None and self._draft_item is not None:
            rect = self._draft_item.rect().normalized()
            self._scene.removeItem(self._draft_item)
            self._draft_item = None
            self._draw_start = None
            if rect.width() >= 1.0 and rect.height() >= 1.0:
                self.box_created.emit((rect.left(), rect.top(), rect.right(), rect.bottom()))
            event.accept()
            return
        super().mouseReleaseEvent(event)


class LabelerWindow(QMainWindow):
    def __init__(self, project: AnnotationProject, media: MediaSource) -> None:
        super().__init__()
        self.project = project
        self.media = media
        self.canvas = AnnotationCanvas(self)
        self.setCentralWidget(self.canvas)
        self.setWindowTitle(f"Frame Labeler - {media.path.name}")
        self.resize(1280, 840)
        self._closed = False
        self._initial_fit_pending = True
        self._current_media_frame: MediaFrame | None = None
        self._undo: list[tuple[Box, ...]] = []
        self._redo: list[tuple[Box, ...]] = []
        self._suppress_class_change = False
        self._suppress_box_list_change = False
        saved = project.last_frame_index
        saved_is_valid = 0 <= saved < media.frame_count and saved % project.stride == 0
        self.current_frame_index = saved if saved_is_valid else 0
        self._sampled_frame_count = (media.frame_count + project.stride - 1) // project.stride

        self._status_label = QLabel()
        self._class_combo = QComboBox()
        self._class_combo.addItems(project.class_names)
        self._box_list = QListWidget()
        self._build_toolbar()
        self._build_box_panel()
        self._build_canvas_shortcuts()
        self.canvas.box_created.connect(self._create_box)
        self.canvas.box_edit_started.connect(self._push_undo)
        self.canvas.boxes_changed.connect(self._persist_canvas_boxes)
        self.canvas.scene().selectionChanged.connect(self._selection_changed)
        self._load_frame(self.current_frame_index)

    @property
    def frame_status_text(self) -> str:
        return self._status_label.text()

    @property
    def box_list(self) -> QListWidget:
        return self._box_list

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Labeling", self)
        toolbar.setObjectName("Labeling")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        self._previous_action = self._action("Previous", Qt.Key.Key_Left, self.previous_frame)
        self._next_action = self._action("Next", Qt.Key.Key_Right, self.next_frame)
        self._review_action = self._action("Mark reviewed", Qt.Key.Key_R, self.mark_reviewed)
        self._delete_action = self._action(
            "Delete box", QKeySequence.StandardKey.Delete, self.delete_selected
        )
        self._undo_action = self._action("Undo", QKeySequence.StandardKey.Undo, self.undo)
        self._redo_action = self._action("Redo", QKeySequence.StandardKey.Redo, self.redo)
        fit_action = self._action("Fit", Qt.Key.Key_F, self.canvas.fit_to_window)
        export_action = self._action("Export", None, self.export_dialog)
        for action in (
            self._previous_action,
            self._next_action,
            self._review_action,
            self._delete_action,
            self._undo_action,
            self._redo_action,
            fit_action,
            export_action,
        ):
            toolbar.addAction(action)
        toolbar.addSeparator()
        toolbar.addWidget(QLabel("Class:"))
        toolbar.addWidget(self._class_combo)
        toolbar.addSeparator()
        toolbar.addWidget(self._status_label)
        self._class_combo.currentIndexChanged.connect(self._class_changed)

    def _build_box_panel(self) -> None:
        dock = QDockWidget("Boxes", self)
        dock.setObjectName("Boxes")
        dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        panel = QWidget(dock)
        layout = QVBoxLayout(panel)
        layout.addWidget(QLabel("Current frame"))
        self._box_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._box_list.setMinimumWidth(280)
        layout.addWidget(self._box_list, 1)
        controls = QLabel(
            "Mouse\n"
            "Wheel                  Zoom\n"
            "Space/right drag       Pan\n"
            "Drag empty space       Draw box\n"
            "Drag box/handles       Move/resize\n\n"
            "Keyboard\n"
            "H J K L                Nudge 1 px\n"
            "Shift+H J K L          Nudge 10 px\n"
            "[ / ]                  Previous/next box\n"
            "Left / Right           Previous/next frame\n"
            "R                       Mark reviewed\n"
            "Delete                 Delete box"
        )
        controls.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(controls)
        panel.setLayout(layout)
        dock.setWidget(panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self._box_list.currentRowChanged.connect(self._box_list_changed)

    def _build_canvas_shortcuts(self) -> None:
        actions = (
            self._action("Nudge left", QKeySequence("H"), lambda: self.nudge_selected_box(-1, 0)),
            self._action("Nudge down", QKeySequence("J"), lambda: self.nudge_selected_box(0, 1)),
            self._action("Nudge up", QKeySequence("K"), lambda: self.nudge_selected_box(0, -1)),
            self._action("Nudge right", QKeySequence("L"), lambda: self.nudge_selected_box(1, 0)),
            self._action(
                "Nudge left 10", QKeySequence("Shift+H"), lambda: self.nudge_selected_box(-10, 0)
            ),
            self._action(
                "Nudge down 10", QKeySequence("Shift+J"), lambda: self.nudge_selected_box(0, 10)
            ),
            self._action(
                "Nudge up 10", QKeySequence("Shift+K"), lambda: self.nudge_selected_box(0, -10)
            ),
            self._action(
                "Nudge right 10",
                QKeySequence("Shift+L"),
                lambda: self.nudge_selected_box(10, 0),
            ),
            self._action("Previous box", QKeySequence("["), self.select_previous_box),
            self._action("Next box", QKeySequence("]"), self.select_next_box),
        )
        for action in actions:
            self.addAction(action)

    def _action(
        self,
        text: str,
        shortcut: QKeySequence.StandardKey | QKeySequence | Qt.Key | None,
        callback: Callable[[], None],
    ) -> QAction:
        action = QAction(text, self)
        if shortcut is not None:
            action.setShortcut(shortcut)
        action.triggered.connect(callback)
        return action

    def _load_frame(self, index: int, previous_index: int | None = None) -> None:
        media_frame = self.media.read_frame(index)
        frame = self.project.ensure_frame(
            index,
            media_frame.timestamp_seconds,
            media_frame.width,
            media_frame.height,
            previous_index=previous_index,
        )
        self.current_frame_index = index
        self.project.last_frame_index = index
        if self._current_media_frame is not None:
            self._current_media_frame.image.close()
        self._current_media_frame = media_frame
        self.canvas.set_frame(
            pil_to_qimage(media_frame.image),
            self.project.get_boxes(index),
            self.project.class_names,
            frame.state,
        )
        self.canvas.fit_to_window()
        self._undo.clear()
        self._redo.clear()
        self._refresh_box_list()
        self._update_status(frame.state)

    def _update_status(self, state: ReviewState | None = None) -> None:
        if state is None:
            state = self.project.get_frame(self.current_frame_index).state
        position = self.current_frame_index // self.project.stride + 1
        self._status_label.setText(
            f"Frame {self.current_frame_index} | {position}/{self._sampled_frame_count} | "
            f"{state.value.title()} | Saved"
        )
        self._previous_action.setEnabled(position > 1)
        self._next_action.setEnabled(position < self._sampled_frame_count)
        self._undo_action.setEnabled(bool(self._undo))
        self._redo_action.setEnabled(bool(self._redo))

    def _push_undo(self) -> None:
        self._undo.append(self.project.get_boxes(self.current_frame_index))
        self._redo.clear()
        self._update_status()

    def _persist_canvas_boxes(self) -> None:
        selected_box_id = self.canvas.selected_box_id()
        self.project.replace_boxes(self.current_frame_index, self.canvas.boxes())
        self.canvas.set_review_state(ReviewState.DRAFT)
        self._refresh_box_list(selected_box_id)
        self._update_status(ReviewState.DRAFT)

    def _create_box(self, coordinates: tuple[float, float, float, float]) -> None:
        self._push_undo()
        box = Box(
            str(uuid.uuid4()),
            self._class_combo.currentIndex(),
            *coordinates,
            BoxOrigin.MANUAL,
        )
        boxes = (*self.project.get_boxes(self.current_frame_index), box)
        self.project.replace_boxes(self.current_frame_index, boxes)
        self.canvas.set_boxes(boxes)
        self.canvas.select_box(box.id)
        self.canvas.set_review_state(ReviewState.DRAFT)
        self._refresh_box_list(box.id)
        self._update_status(ReviewState.DRAFT)

    def _selection_changed(self) -> None:
        selected = self.canvas.selected_items()
        self._delete_action.setEnabled(bool(selected))
        if selected:
            self._suppress_class_change = True
            self._class_combo.setCurrentIndex(selected[0].box.class_id)
            self._suppress_class_change = False
        self._select_box_list_item(selected[0].box.id if selected else None)

    def _select_box_list_item(self, box_id: str | None) -> None:
        self._suppress_box_list_change = True
        try:
            row = -1
            if box_id is not None:
                for index in range(self._box_list.count()):
                    if self._box_list.item(index).data(Qt.ItemDataRole.UserRole) == box_id:
                        row = index
                        break
            self._box_list.setCurrentRow(row)
        finally:
            self._suppress_box_list_change = False

    def _refresh_box_list(self, selected_box_id: str | None = None) -> None:
        if selected_box_id is None:
            selected_box_id = self.canvas.selected_box_id()
        self._suppress_box_list_change = True
        try:
            self._box_list.clear()
            selected_row = -1
            for index, box in enumerate(self.canvas.boxes()):
                class_name = self.project.class_names[box.class_id]
                width = box.x_max - box.x_min
                height = box.y_max - box.y_min
                item = QListWidgetItem(
                    f"{index + 1}. {class_name}  "
                    f"x:{box.x_min:.1f} y:{box.y_min:.1f}  {width:.1f} x {height:.1f}"
                )
                item.setData(Qt.ItemDataRole.UserRole, box.id)
                self._box_list.addItem(item)
                if box.id == selected_box_id:
                    selected_row = index
            self._box_list.setCurrentRow(selected_row)
        finally:
            self._suppress_box_list_change = False

    def _box_list_changed(self, row: int) -> None:
        if self._suppress_box_list_change or row < 0:
            return
        item = self._box_list.item(row)
        self.canvas.select_box(str(item.data(Qt.ItemDataRole.UserRole)), ensure_visible=True)

    def select_previous_box(self) -> None:
        count = self._box_list.count()
        if count:
            current = self._box_list.currentRow()
            self._box_list.setCurrentRow(count - 1 if current < 0 else (current - 1) % count)

    def select_next_box(self) -> None:
        count = self._box_list.count()
        if count:
            self._box_list.setCurrentRow((self._box_list.currentRow() + 1) % count)

    def nudge_selected_box(self, dx: float, dy: float) -> None:
        self.canvas.nudge_selected(dx, dy)

    def _class_changed(self, class_id: int) -> None:
        if self._suppress_class_change or not self.canvas.selected_items():
            return
        self._push_undo()
        self.canvas.reclassify_selected(class_id)

    def delete_selected(self) -> None:
        if not self.canvas.selected_items():
            return
        self._push_undo()
        self.canvas.delete_selected()

    def mark_reviewed(self) -> None:
        self.project.mark_reviewed(self.current_frame_index)
        self.canvas.set_review_state(ReviewState.REVIEWED)
        self._update_status(ReviewState.REVIEWED)

    def previous_frame(self) -> None:
        if self.current_frame_index >= self.project.stride:
            self._load_frame(self.current_frame_index - self.project.stride)

    def next_frame(self) -> None:
        next_index = self.current_frame_index + self.project.stride
        if next_index < self.media.frame_count:
            previous = self.current_frame_index
            self._load_frame(next_index, previous_index=previous)

    def undo(self) -> None:
        if not self._undo:
            return
        current = self.project.get_boxes(self.current_frame_index)
        boxes = self._undo.pop()
        self._redo.append(current)
        self.project.replace_boxes(self.current_frame_index, boxes)
        self.canvas.set_boxes(boxes)
        self.canvas.set_review_state(ReviewState.DRAFT)
        self._refresh_box_list()
        self._update_status(ReviewState.DRAFT)

    def redo(self) -> None:
        if not self._redo:
            return
        current = self.project.get_boxes(self.current_frame_index)
        boxes = self._redo.pop()
        self._undo.append(current)
        self.project.replace_boxes(self.current_frame_index, boxes)
        self.canvas.set_boxes(boxes)
        self.canvas.set_review_state(ReviewState.DRAFT)
        self._refresh_box_list()
        self._update_status(ReviewState.DRAFT)

    def export_dialog(self) -> None:
        destination = QFileDialog.getExistingDirectory(self, "Export YOLO dataset")
        if not destination:
            return
        try:
            summary = export_yolo(self.project, self.media, Path(destination))
            QMessageBox.information(
                self, "Export complete", f"Exported {summary.exported} reviewed frame(s)."
            )
            self._load_frame(self.current_frame_index)
        except Exception as error:
            QMessageBox.critical(self, "Export failed", str(error))

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        if self._initial_fit_pending:
            self._initial_fit_pending = False
            self.canvas.fit_to_window()

    def closeEvent(self, event: QCloseEvent) -> None:
        if not self._closed:
            self._closed = True
            self.canvas.scene().selectionChanged.disconnect(self._selection_changed)
            self.canvas.clear_frame()
            if self._current_media_frame is not None:
                self._current_media_frame.image.close()
                self._current_media_frame = None
            self.project.last_frame_index = self.current_frame_index
            self.media.close()
            self.project.close()
        event.accept()


def run_application(project: AnnotationProject, media: MediaSource) -> int:
    application = QApplication.instance()
    owns_application = application is None
    if application is None:
        application = QApplication(sys.argv)
        application.setApplicationName("Frame Labeler")
    window = LabelerWindow(project, media)
    window.show()
    if owns_application:
        return application.exec()
    return 0
