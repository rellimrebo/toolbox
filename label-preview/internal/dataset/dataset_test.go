package dataset

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeFile(t *testing.T, path string, contents string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(contents), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestLoadsStandardYOLODataset(t *testing.T) {
	root := t.TempDir()
	image := filepath.Join(root, "images", "train", "dog.png")
	writeFile(t, image, "")
	writeFile(t, filepath.Join(root, "labels", "train", "dog.txt"), "1 0.5 0.5 0.4 0.6\n")
	writeFile(t, filepath.Join(root, "dataset.yaml"), "names:\n  0: \"cat\"\n  1: \"dog\"\n")

	boxes, err := LoadAnnotations(image, 100, 80)
	if err != nil {
		t.Fatal(err)
	}
	if len(boxes) != 1 {
		t.Fatalf("box count = %d, want 1", len(boxes))
	}
	box := boxes[0]
	if box.Label != "dog" || !closeEnough(box.XMin, 0.3) || !closeEnough(box.YMin, 0.2) ||
		!closeEnough(box.XMax, 0.7) || !closeEnough(box.YMax, 0.8) {
		t.Fatalf("unexpected box: %#v", box)
	}
}

func TestLoadsSplitFirstYOLODatasetWithDataYAML(t *testing.T) {
	root := t.TempDir()
	image := filepath.Join(root, "train", "images", "dog.jpg")
	writeFile(t, image, "")
	writeFile(t, filepath.Join(root, "train", "labels", "dog.txt"), "0 0.5 0.5 1 1\n")
	writeFile(t, filepath.Join(root, "data.yaml"), "names: [dog]\n")

	boxes, err := LoadAnnotations(image, 100, 80)
	if err != nil {
		t.Fatal(err)
	}
	if len(boxes) != 1 || boxes[0].Label != "dog" {
		t.Fatalf("unexpected boxes: %#v", boxes)
	}
}

func TestLoadsYOLOIndentlessClassSequence(t *testing.T) {
	root := t.TempDir()
	image := filepath.Join(root, "images", "test", "dog.jpg")
	writeFile(t, image, "")
	writeFile(t, filepath.Join(root, "labels", "test", "dog.txt"), "1 0.5 0.5 1 1\n")
	writeFile(t, filepath.Join(root, "data.yaml"), "names:\n- cat\n- dog\n")

	boxes, err := LoadAnnotations(image, 100, 80)
	if err != nil {
		t.Fatal(err)
	}
	if len(boxes) != 1 || boxes[0].Label != "dog" {
		t.Fatalf("unexpected boxes: %#v", boxes)
	}
}

func TestMissingYOLOLabelRepresentsEmptySample(t *testing.T) {
	root := t.TempDir()
	image := filepath.Join(root, "images", "val", "empty.png")
	writeFile(t, image, "")
	writeFile(t, filepath.Join(root, "dataset.yaml"), "names: [cat, dog]\n")

	boxes, err := LoadAnnotations(image, 100, 80)
	if err != nil {
		t.Fatal(err)
	}
	if len(boxes) != 0 {
		t.Fatalf("unexpected boxes: %#v", boxes)
	}
}

func TestLoadsStandardCOCODetectionDataset(t *testing.T) {
	root := t.TempDir()
	image := filepath.Join(root, "train2017", "cat.jpg")
	writeFile(t, image, "")
	writeFile(t, filepath.Join(root, "annotations", "instances_train2017.json"), `{
  "images": [{"id": 7, "file_name": "cat.jpg", "width": 200, "height": 100}],
  "annotations": [{"id": 10, "image_id": 7, "category_id": 3, "bbox": [20, 10, 40, 50]}],
  "categories": [{"id": 3, "name": "cat"}]
}`)

	boxes, err := LoadAnnotations(image, 200, 100)
	if err != nil {
		t.Fatal(err)
	}
	if len(boxes) != 1 {
		t.Fatalf("unexpected boxes: %#v", boxes)
	}
	box := boxes[0]
	if box.Label != "cat" || !closeEnough(box.XMin, 0.1) || !closeEnough(box.YMin, 0.1) ||
		!closeEnough(box.XMax, 0.3) || !closeEnough(box.YMax, 0.6) {
		t.Fatalf("unexpected box: %#v", box)
	}
}

func TestLoadsRoboflowStyleCOCODataset(t *testing.T) {
	root := t.TempDir()
	image := filepath.Join(root, "train", "car.jpg")
	writeFile(t, image, "")
	writeFile(t, filepath.Join(root, "train", "_annotations.coco.json"), `{
  "images": [{"id": 2, "file_name": "car.jpg", "width": 40, "height": 20}],
  "annotations": [{"image_id": 2, "category_id": 1, "bbox": [4, 2, 20, 10]}],
  "categories": [{"id": 1, "name": "car"}]
}`)

	boxes, err := LoadAnnotations(image, 40, 20)
	if err != nil {
		t.Fatal(err)
	}
	if len(boxes) != 1 || boxes[0].Label != "car" {
		t.Fatalf("unexpected boxes: %#v", boxes)
	}
}

func TestLoadsCOCOWhenAnnotationsPrecedeImages(t *testing.T) {
	root := t.TempDir()
	image := filepath.Join(root, "train2017", "cat.jpg")
	writeFile(t, image, "")
	writeFile(t, filepath.Join(root, "annotations", "instances_train2017.json"), `{
  "annotations": [{"image_id": 7, "category_id": 3, "bbox": [20, 10, 40, 50]}],
  "categories": [{"id": 3, "name": "cat"}],
  "images": [{"id": 7, "file_name": "cat.jpg", "width": 200, "height": 100}]
}`)

	boxes, err := LoadAnnotations(image, 200, 100)
	if err != nil {
		t.Fatal(err)
	}
	if len(boxes) != 1 || boxes[0].Label != "cat" {
		t.Fatalf("unexpected boxes: %#v", boxes)
	}
}

func TestLoadsStandardPascalVOCDataset(t *testing.T) {
	root := t.TempDir()
	image := filepath.Join(root, "JPEGImages", "dog.jpg")
	writeFile(t, image, "")
	writeFile(t, filepath.Join(root, "Annotations", "dog.xml"), `<annotation>
  <filename>dog.jpg</filename>
  <size><width>100</width><height>80</height></size>
  <object><name>dog</name><bndbox><xmin>10</xmin><ymin>20</ymin><xmax>60</xmax><ymax>70</ymax></bndbox></object>
</annotation>`)

	boxes, err := LoadAnnotations(image, 100, 80)
	if err != nil {
		t.Fatal(err)
	}
	if len(boxes) != 1 {
		t.Fatalf("unexpected boxes: %#v", boxes)
	}
	box := boxes[0]
	if box.Label != "dog" || !closeEnough(box.XMin, 0.1) || !closeEnough(box.YMin, 0.25) ||
		!closeEnough(box.XMax, 0.6) || !closeEnough(box.YMax, 0.875) {
		t.Fatalf("unexpected box: %#v", box)
	}
}

func TestLoadsPascalVOCXMLBesideImage(t *testing.T) {
	root := t.TempDir()
	image := filepath.Join(root, "dog.jpg")
	writeFile(t, image, "")
	writeFile(t, filepath.Join(root, "dog.xml"), `<annotation>
  <object><name>dog</name><bndbox><xmin>0</xmin><ymin>0</ymin><xmax>10</xmax><ymax>10</ymax></bndbox></object>
</annotation>`)

	boxes, err := LoadAnnotations(image, 10, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(boxes) != 1 {
		t.Fatalf("unexpected boxes: %#v", boxes)
	}
}

func TestRejectsImageOutsideRecognizedDataset(t *testing.T) {
	root := t.TempDir()
	image := filepath.Join(root, "photo.jpg")
	writeFile(t, image, "")
	writeFile(t, filepath.Join(root, "photo.txt"), "0 0.5 0.5 1 1\n")
	writeFile(t, filepath.Join(root, "classes.txt"), "dog\n")

	_, err := LoadAnnotations(image, 100, 80)

	if err == nil || !strings.Contains(err.Error(), "recognized dataset layout") {
		t.Fatalf("error = %v, want layout error", err)
	}
}

func TestRejectsInvalidYOLOBoxes(t *testing.T) {
	root := t.TempDir()
	image := filepath.Join(root, "images", "train", "dog.png")
	writeFile(t, image, "")
	writeFile(t, filepath.Join(root, "labels", "train", "dog.txt"), "0 1.5 0.5 0.2 0.2\n")
	writeFile(t, filepath.Join(root, "dataset.yaml"), "names: [dog]\n")

	_, err := LoadAnnotations(image, 100, 80)

	if err == nil || !strings.Contains(err.Error(), "between 0 and 1") {
		t.Fatalf("error = %v, want coordinate error", err)
	}
}

func TestRejectsTerminalControlCharactersInClassNames(t *testing.T) {
	root := t.TempDir()
	image := filepath.Join(root, "images", "train", "dog.png")
	writeFile(t, image, "")
	writeFile(t, filepath.Join(root, "dataset.yaml"), "names:\n  0: \"dog\\u001b[2J\"\n")

	_, err := LoadAnnotations(image, 100, 80)

	if err == nil || !strings.Contains(err.Error(), "control characters") {
		t.Fatalf("error = %v, want control-character rejection", err)
	}
}

func closeEnough(actual float64, expected float64) bool {
	difference := actual - expected
	return difference > -1e-9 && difference < 1e-9
}
