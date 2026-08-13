# Label Preview

`label-preview` is a small Go CLI that renders a YOLO-labeled image as a terminal image with
class-colored bounding boxes. It is meant for quick dataset spot checks and novelty previews,
not precise annotation review. The compiled executable is self-contained and does not require a
Go runtime on the target machine.

The renderer uses Unicode half blocks so each terminal row carries two vertical image samples.
This keeps the result close to the source aspect ratio on terminals whose character cells are
roughly twice as tall as they are wide. ANSI truecolor is used when supported, with a monochrome
fallback for redirected output, `NO_COLOR`, or `--color never`.

## Setup

Requires Go 1.25 or newer to build:

```shell
CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" \
  -o bin/label-preview ./cmd/label-preview
```

## Usage

For a standard YOLO dataset, pass an image. The matching `labels/<split>/<name>.txt` and the
nearest `dataset.yaml` are discovered automatically:

```shell
./bin/label-preview /path/to/dataset/images/train/example.png
```

For another directory layout, provide the files explicitly:

```shell
./bin/label-preview /path/to/example.jpg \
  --labels /path/to/example.txt \
  --classes /path/to/classes.txt
```

`--classes` accepts a YOLO `dataset.yaml` or a UTF-8 text file with one class name per line.
Images without a matching label file render normally with no boxes. Unknown class IDs use the
numeric ID as their label.

Useful sizing and output controls:

```shell
./bin/label-preview /path/to/example.jpg \
  --width 64 \
  --max-height 24 \
  --color always
```

The supported annotation rows are YOLO detection boxes:

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
