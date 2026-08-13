# Frame Labeler

Local desktop software for labeling axis-aligned objects in images and stride-sampled video frames.

## Setup

Requires Python 3.11 or newer and `uv`. Dependencies are installed into a project-local virtual environment.

```shell
uv sync --locked
```

This creates or updates `frame-labeler/.venv` from `pyproject.toml` and `uv.lock`. Activation is optional because `uv run` executes commands in that environment directly. To activate it for an interactive shell:

```shell
source .venv/bin/activate
```

Run `deactivate` when finished. This project does not maintain a `requirements.txt`; `pyproject.toml` declares dependencies and `uv.lock` pins the resolved versions.

Create a UTF-8 class file with one class per line, then open an image or video:

```shell
uv run frame-labeler open /path/to/video.mp4 \
  --project /path/to/annotations.sqlite3 \
  --classes /path/to/classes.txt \
  --stride 30 \
  --split train
```

When reopening an existing project, `--classes`, `--stride`, and `--split` are not required:

```shell
uv run frame-labeler open /path/to/video.mp4 \
  --project /path/to/annotations.sqlite3
```

The project database is the canonical annotation state and saves each change automatically.
A new project's first image or sampled frame starts with no boxes. New forward frames receive
boxes from the immediately preceding frame; reopening a project restores its saved
annotations.

## Controls

- Drag empty image space to create a box with the active class.
- Select a box on the canvas or in the Boxes panel to move, resize, reclassify, or delete it.
- Resize from any corner or edge handle.
- Use the mouse wheel or trackpad to zoom around the pointer.
- Hold Space and drag, right-drag, or middle-drag to pan without changing boxes.
- Use H, J, K, and L to nudge a selected box one source pixel; hold Shift to nudge ten pixels.
- Use [ and ] to select the previous or next box, including overlapping boxes.
- Use Left and Right to navigate sampled frames.
- Use R to mark the current frame reviewed.
- Use F to fit the image.
- Use the platform undo and redo shortcuts for box edits.

Boxes copied from the preceding frame use a dashed outline and remain drafts until the current frame is marked reviewed.

## Export

Only reviewed frames are exported:

```shell
uv run frame-labeler export /path/to/annotations.sqlite3 \
  --format yolo \
  --output /path/to/dataset
```

The output contains normalized YOLO detection labels, deterministic PNG frames, `dataset.yaml`, and a JSON Lines provenance manifest.

## Development

```shell
uv sync --locked
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

GitHub Actions runs the same checks and builds the package on Python 3.11 and 3.14 for
pull requests and changes merged to `main`. There is no deployment workflow because the
desktop application does not yet have a release artifact or deployment target.

Keep local manual-test inputs, projects, and exports under `.local/<test-name>/`. The `.local/` directory is excluded from Git history.
