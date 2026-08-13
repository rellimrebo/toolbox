from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from frame_labeler.domain import Box, BoxOrigin, FrameRecord, ReviewState

SCHEMA_VERSION = 2


class ProjectError(RuntimeError):
    pass


class SourceMismatchError(ProjectError):
    pass


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _source_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    stat = resolved.stat()
    digest = hashlib.sha256()
    chunk_size = 1024 * 1024
    with resolved.open("rb") as source:
        digest.update(source.read(chunk_size))
        if stat.st_size > chunk_size:
            source.seek(max(0, stat.st_size - chunk_size))
            digest.update(source.read(chunk_size))
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "modified_ns": stat.st_mtime_ns,
        "sample_sha256": digest.hexdigest(),
    }


def _validate_classes(class_names: Sequence[str]) -> tuple[str, ...]:
    cleaned = tuple(name.strip() for name in class_names)
    if not cleaned or any(not name or "\n" in name or "\r" in name for name in cleaned):
        raise ValueError("Classes must contain at least one non-empty, single-line name")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("Class names must be unique")
    return cleaned


def _create_boxes_table(connection: sqlite3.Connection, table_name: str) -> None:
    if table_name not in {"boxes", "boxes_v2"}:
        raise ValueError(f"Unexpected boxes table name: {table_name}")
    connection.execute(
        f"""
        CREATE TABLE {table_name} (
            id TEXT PRIMARY KEY,
            frame_index INTEGER NOT NULL REFERENCES frames(source_index) ON DELETE CASCADE,
            class_id INTEGER NOT NULL REFERENCES classes(id),
            x_min REAL NOT NULL,
            y_min REAL NOT NULL,
            x_max REAL NOT NULL,
            y_max REAL NOT NULL,
            origin TEXT NOT NULL CHECK (origin IN ('manual', 'copied', 'inferred')),
            source_box_id TEXT,
            inference_provider TEXT,
            confidence REAL CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            CHECK (x_max > x_min),
            CHECK (y_max > y_min),
            CHECK (origin != 'inferred' OR inference_provider IS NOT NULL),
            CHECK (
                origin != 'manual'
                OR (inference_provider IS NULL AND confidence IS NULL)
            )
        )
        """
    )


class AnnotationProject:
    def __init__(self, path: Path, connection: sqlite3.Connection) -> None:
        self.path = path
        self._connection = connection
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")

    @classmethod
    def create(
        cls,
        project_path: str | Path,
        source_path: str | Path,
        class_names: Sequence[str],
        stride: int = 1,
        split: str = "train",
    ) -> AnnotationProject:
        path = Path(project_path).expanduser().resolve()
        source = Path(source_path)
        classes = _validate_classes(class_names)
        if stride <= 0:
            raise ValueError("Stride must be greater than zero")
        if split not in {"train", "val", "test"}:
            raise ValueError("Split must be train, val, or test")
        if path.exists():
            raise FileExistsError(f"Project already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path)
        project = cls(path, connection)
        try:
            project._initialize_schema()
            project_id = str(uuid.uuid4())
            identity = _source_identity(source)
            now = _utc_now()
            with connection:
                metadata = {
                    "schema_version": str(SCHEMA_VERSION),
                    "project_id": project_id,
                    "source_identity": json.dumps(identity, sort_keys=True),
                    "stride": str(stride),
                    "split": split,
                    "last_frame_index": "0",
                    "created_at": now,
                    "updated_at": now,
                }
                connection.executemany(
                    "INSERT INTO metadata(key, value) VALUES (?, ?)", metadata.items()
                )
                connection.executemany(
                    "INSERT INTO classes(id, name) VALUES (?, ?)", enumerate(classes)
                )
            return project
        except Exception:
            project.close()
            if path.exists():
                path.unlink()
            raise

    @classmethod
    def open(
        cls, project_path: str | Path, source_path: str | Path | None = None
    ) -> AnnotationProject:
        path = Path(project_path).expanduser().resolve(strict=True)
        connection = sqlite3.connect(path)
        project = cls(path, connection)
        try:
            version = int(project._metadata("schema_version"))
            if version not in {1, SCHEMA_VERSION}:
                raise ProjectError(f"Unsupported project schema version: {version}")
            recorded = json.loads(project._metadata("source_identity"))
            source = Path(source_path) if source_path is not None else Path(recorded["path"])
            if _source_identity(source) != recorded:
                raise SourceMismatchError(
                    "The project source does not match the current media file"
                )
            if version == 1:
                project._migrate_v1_to_v2()
            return project
        except Exception:
            project.close()
            raise

    def _initialize_schema(self) -> None:
        self._connection.executescript(
            """
            PRAGMA journal_mode = WAL;
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE classes (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE
            );
            CREATE TABLE frames (
                source_index INTEGER PRIMARY KEY CHECK (source_index >= 0),
                timestamp_seconds REAL NOT NULL CHECK (timestamp_seconds >= 0),
                width INTEGER NOT NULL CHECK (width > 0),
                height INTEGER NOT NULL CHECK (height > 0),
                state TEXT NOT NULL CHECK (state IN ('unreviewed', 'draft', 'reviewed')),
                reviewed_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        _create_boxes_table(self._connection, "boxes")
        self._connection.execute("CREATE INDEX boxes_frame_index ON boxes(frame_index)")

    def _migrate_v1_to_v2(self) -> None:
        now = _utc_now()
        with self._connection:
            _create_boxes_table(self._connection, "boxes_v2")
            self._connection.execute(
                """
                INSERT INTO boxes_v2(
                    id, frame_index, class_id, x_min, y_min, x_max, y_max,
                    origin, source_box_id, inference_provider, confidence,
                    created_at, updated_at
                )
                SELECT
                    id, frame_index, class_id, x_min, y_min, x_max, y_max,
                    origin, source_box_id, NULL, NULL, created_at, updated_at
                FROM boxes
                """
            )
            self._connection.execute("DROP TABLE boxes")
            self._connection.execute("ALTER TABLE boxes_v2 RENAME TO boxes")
            self._connection.execute("CREATE INDEX boxes_frame_index ON boxes(frame_index)")
            self._connection.execute(
                "UPDATE metadata SET value = ? WHERE key = 'schema_version'",
                (str(SCHEMA_VERSION),),
            )
            self._connection.execute(
                "UPDATE metadata SET value = ? WHERE key = 'updated_at'", (now,)
            )

    def _metadata(self, key: str) -> str:
        row = self._connection.execute(
            "SELECT value FROM metadata WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            raise ProjectError(f"Project metadata is missing {key!r}")
        return str(row["value"])

    def _set_metadata(self, key: str, value: str) -> None:
        with self._connection:
            self._connection.execute("UPDATE metadata SET value = ? WHERE key = ?", (value, key))
            self._connection.execute(
                "UPDATE metadata SET value = ? WHERE key = 'updated_at'", (_utc_now(),)
            )

    @property
    def project_id(self) -> str:
        return self._metadata("project_id")

    @property
    def source_path(self) -> Path:
        return Path(json.loads(self._metadata("source_identity"))["path"])

    @property
    def source_identity(self) -> dict[str, Any]:
        return dict(json.loads(self._metadata("source_identity")))

    @property
    def stride(self) -> int:
        return int(self._metadata("stride"))

    @property
    def split(self) -> str:
        return self._metadata("split")

    @property
    def class_names(self) -> tuple[str, ...]:
        rows = self._connection.execute("SELECT name FROM classes ORDER BY id").fetchall()
        return tuple(str(row["name"]) for row in rows)

    @property
    def last_frame_index(self) -> int:
        return int(self._metadata("last_frame_index"))

    @last_frame_index.setter
    def last_frame_index(self, value: int) -> None:
        if value < 0:
            raise ValueError("Frame index cannot be negative")
        self._set_metadata("last_frame_index", str(value))

    def ensure_frame(
        self,
        index: int,
        timestamp_seconds: float,
        width: int,
        height: int,
        previous_index: int | None = None,
    ) -> FrameRecord:
        if index < 0 or timestamp_seconds < 0 or width <= 0 or height <= 0:
            raise ValueError("Invalid frame metadata")
        existing = self._connection.execute(
            "SELECT * FROM frames WHERE source_index = ?", (index,)
        ).fetchone()
        if existing is not None:
            frame = self._frame_from_row(existing)
            if (frame.width, frame.height) != (width, height):
                raise ProjectError("Stored frame dimensions do not match decoded media")
            if (
                previous_index is not None
                and frame.state is ReviewState.UNREVIEWED
                and not self.get_boxes(index)
            ):
                boxes_to_carry = self.get_boxes(previous_index)
                if boxes_to_carry:
                    now = _utc_now()
                    with self._connection:
                        self._insert_copied_boxes(index, boxes_to_carry, now)
                        self._connection.execute(
                            "UPDATE frames SET state = ?, updated_at = ? WHERE source_index = ?",
                            (ReviewState.DRAFT.value, now, index),
                        )
                    return self.get_frame(index)
            return frame

        previous_boxes: tuple[Box, ...] = ()
        if previous_index is not None:
            previous_boxes = self.get_boxes(previous_index)

        state = ReviewState.DRAFT if previous_boxes else ReviewState.UNREVIEWED
        now = _utc_now()
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO frames(
                    source_index, timestamp_seconds, width, height, state,
                    reviewed_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, NULL, ?, ?)
                """,
                (index, timestamp_seconds, width, height, state.value, now, now),
            )
            self._insert_copied_boxes(index, previous_boxes, now)
        return self.get_frame(index)

    def _insert_copied_boxes(self, frame_index: int, boxes: Sequence[Box], now: str) -> None:
        for box in boxes:
            copied = Box(
                str(uuid.uuid4()),
                box.class_id,
                *box.coordinates,
                BoxOrigin.COPIED,
                source_box_id=box.id,
                inference_provider=box.inference_provider,
                confidence=box.confidence,
            )
            self._insert_box(frame_index, copied, now)

    def get_frame(self, index: int) -> FrameRecord:
        row = self._connection.execute(
            "SELECT * FROM frames WHERE source_index = ?", (index,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Frame has not been visited: {index}")
        return self._frame_from_row(row)

    def iter_reviewed_frames(self) -> Iterable[FrameRecord]:
        rows = self._connection.execute(
            "SELECT * FROM frames WHERE state = ? ORDER BY source_index",
            (ReviewState.REVIEWED.value,),
        )
        for row in rows:
            yield self._frame_from_row(row)

    @staticmethod
    def _frame_from_row(row: sqlite3.Row) -> FrameRecord:
        return FrameRecord(
            int(row["source_index"]),
            float(row["timestamp_seconds"]),
            int(row["width"]),
            int(row["height"]),
            ReviewState(str(row["state"])),
            str(row["reviewed_at"]) if row["reviewed_at"] is not None else None,
        )

    def get_boxes(self, frame_index: int) -> tuple[Box, ...]:
        rows = self._connection.execute(
            "SELECT * FROM boxes WHERE frame_index = ? ORDER BY rowid",
            (frame_index,),
        ).fetchall()
        return tuple(
            Box(
                id=str(row["id"]),
                class_id=int(row["class_id"]),
                x_min=float(row["x_min"]),
                y_min=float(row["y_min"]),
                x_max=float(row["x_max"]),
                y_max=float(row["y_max"]),
                origin=BoxOrigin(str(row["origin"])),
                source_box_id=(
                    str(row["source_box_id"]) if row["source_box_id"] is not None else None
                ),
                inference_provider=(
                    str(row["inference_provider"])
                    if row["inference_provider"] is not None
                    else None
                ),
                confidence=(float(row["confidence"]) if row["confidence"] is not None else None),
            )
            for row in rows
        )

    def replace_boxes(self, frame_index: int, boxes: Sequence[Box]) -> None:
        frame = self.get_frame(frame_index)
        class_count = len(self.class_names)
        for box in boxes:
            if box.class_id >= class_count:
                raise ValueError(f"Unknown class ID: {box.class_id}")
            box.clipped(frame.width, frame.height)
        now = _utc_now()
        with self._connection:
            self._connection.execute("DELETE FROM boxes WHERE frame_index = ?", (frame_index,))
            for box in boxes:
                self._insert_box(frame_index, box.clipped(frame.width, frame.height), now)
            self._connection.execute(
                """
                UPDATE frames
                SET state = ?, reviewed_at = NULL, updated_at = ?
                WHERE source_index = ?
                """,
                (ReviewState.DRAFT.value, now, frame_index),
            )

    def seed_inferred_boxes(self, frame_index: int, boxes: Sequence[Box]) -> bool:
        if not boxes:
            return False
        if any(box.origin is not BoxOrigin.INFERRED for box in boxes):
            raise ValueError("Inference seeding accepts only inferred boxes")
        frame = self.get_frame(frame_index)
        if frame.state is not ReviewState.UNREVIEWED or self.get_boxes(frame_index):
            return False
        self.replace_boxes(frame_index, boxes)
        return True

    def _insert_box(self, frame_index: int, box: Box, now: str) -> None:
        self._connection.execute(
            """
            INSERT INTO boxes(
                id, frame_index, class_id, x_min, y_min, x_max, y_max,
                origin, source_box_id, inference_provider, confidence,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                box.id,
                frame_index,
                box.class_id,
                *box.coordinates,
                box.origin.value,
                box.source_box_id,
                box.inference_provider,
                box.confidence,
                now,
                now,
            ),
        )

    def mark_reviewed(self, frame_index: int) -> None:
        self.get_frame(frame_index)
        now = _utc_now()
        with self._connection:
            self._connection.execute(
                """
                UPDATE frames SET state = ?, reviewed_at = ?, updated_at = ?
                WHERE source_index = ?
                """,
                (ReviewState.REVIEWED.value, now, now, frame_index),
            )

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> AnnotationProject:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
