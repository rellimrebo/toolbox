# Label Preview

`label-preview` is a small Go CLI that renders an image's aspect-ratio frame and labeled bounding
boxes from a dataset. It is meant for quick dataset spot checks and novelty
previews, not precise annotation review. The compiled executable is self-contained and does not
require a Go runtime on the target machine.

The frame accounts for terminal cells being roughly twice as tall as they are wide. Its interior
is blank except for annotation boxes and class names. Output contains no pixel approximation or
ANSI sequences.

## Setup

Requires Go 1.25 or newer to build:

```shell
CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" \
  -o bin/label-preview ./cmd/label-preview
```

## Usage

Pass an image from a supported dataset. Its format, annotation file, and class names are inferred
from the standard dataset layout:

```shell
./bin/label-preview /path/to/dataset/images/train/example.png
```

Supported detection layouts:

- YOLO: `images/<split>/name.jpg` with `labels/<split>/name.txt`, or
  `<split>/images/name.jpg` with `<split>/labels/name.txt`; class names come from `dataset.yaml`,
  `data.yaml`, or their `.yml` variants.
- COCO: `<split>/name.jpg` with `annotations/instances_<split>.json`, including the common
  `images/<split>/` variant; Roboflow-style `_annotations.coco.json` beside the image also works.
- Pascal VOC: `JPEGImages/name.jpg` with `Annotations/name.xml`, or `name.xml` beside the image.

The tool deliberately has no annotation-path or format flags. An image outside a recognized
layout is rejected rather than guessed. In a recognized YOLO dataset, a missing label file is a
valid empty sample. Unknown numeric class IDs use the ID as their label.

Useful sizing controls:

```shell
./bin/label-preview /path/to/dataset/images/train/example.jpg \
  --width 64 \
  --max-height 24
```

YOLO detection rows use the usual normalized representation:

```text
class_id center_x center_y width height
```

All four coordinates are normalized to the image. PNG, JPEG, and GIF input are supported.
Polygon, mask, and classification annotations are outside this first version.

## Development

```shell
go test ./...
go vet ./...
test -z "$(gofmt -l .)"
CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" \
  -o bin/label-preview ./cmd/label-preview
```
