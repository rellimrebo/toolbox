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

An empty unreviewed frame may receive boxes when revisited from its preceding frame. Draft and
reviewed frames are treated as intentional work and are never repopulated automatically.

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

## Inference providers

Model integrations implement the generic `InferenceProvider` protocol in
`frame_labeler.inference`. A provider declares a versioned output type, receives a PIL image,
source-frame metadata, and the project class catalog, then returns its typed predictions.
`run_inference()` does not assume a labeling domain, allowing detection, classification,
segmentation, and other outputs to use the same execution boundary.

Object detection currently provides the first domain adapter. `detections_to_boxes()` validates
and clips source-pixel detections into editable boxes while retaining the provider identifier and
optional confidence. Pass those boxes to `AnnotationProject.seed_inferred_boxes()` to seed an
empty unreviewed frame. This method never overwrites existing boxes or an intentionally empty
draft or reviewed frame. Concrete model runtimes, result caching, and GUI scheduling remain
separate from the core package.

Frame creation and review state are independent of detection policy. Box carry-forward,
inference seeding, and YOLO conversion are explicit detection operations. Additional annotation
domains should add typed domain records and persistence methods instead of expanding `Box` or
storing opaque annotation payloads.

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
