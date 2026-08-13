from __future__ import annotations

from frame_labeler.cli import build_parser


def test_open_command_parses_required_project_inputs() -> None:
    args = build_parser().parse_args(
        [
            "open",
            "clip.mp4",
            "--project",
            "clip.sqlite3",
            "--classes",
            "classes.txt",
            "--stride",
            "5",
        ]
    )

    assert args.command == "open"
    assert args.source == "clip.mp4"
    assert args.stride == 5
