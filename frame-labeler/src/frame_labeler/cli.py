from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from frame_labeler import __version__
from frame_labeler.export import available_export_formats, export_dataset
from frame_labeler.media import open_media
from frame_labeler.project import AnnotationProject


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="frame-labeler")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    open_command = commands.add_parser("open", help="Open an image or video for labeling")
    open_command.add_argument("source")
    open_command.add_argument("--project", required=True)
    open_command.add_argument("--classes")
    open_command.add_argument("--stride", type=_positive_int, default=1)
    open_command.add_argument("--split", choices=("train", "val", "test"), default="train")

    export_command = commands.add_parser("export", help="Export reviewed annotations")
    export_command.add_argument("project")
    export_command.add_argument("--format", choices=available_export_formats(), default="yolo")
    export_command.add_argument("--output", required=True)
    return parser


def _read_classes(path: str) -> list[str]:
    return Path(path).expanduser().read_text(encoding="utf-8").splitlines()


def _open_project(args: argparse.Namespace) -> AnnotationProject:
    project_path = Path(args.project).expanduser()
    if project_path.exists():
        return AnnotationProject.open(project_path, args.source)
    if args.classes is None:
        raise ValueError("--classes is required when creating a project")
    return AnnotationProject.create(
        project_path,
        args.source,
        _read_classes(args.classes),
        stride=args.stride,
        split=args.split,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            with AnnotationProject.open(args.project) as project:
                media = open_media(project.source_path)
                try:
                    summary = export_dataset(args.format, project, media, args.output)
                finally:
                    media.close()
            print(f"Exported {summary.exported} reviewed frame(s) to {args.output}")
            return 0

        media = open_media(args.source)
        try:
            project = _open_project(args)
        except Exception:
            media.close()
            raise
        try:
            from frame_labeler.gui import run_application

            return run_application(project, media)
        except Exception:
            media.close()
            project.close()
            raise
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(2, f"frame-labeler: error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
