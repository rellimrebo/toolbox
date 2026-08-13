package dataset

import (
	"os"
	"path/filepath"
	"reflect"
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

func TestDiscoversFrameLabelerYOLOExport(t *testing.T) {
	root := t.TempDir()
	image := filepath.Join(root, "images", "train", "dog.png")
	writeFile(t, image, "")
	label := filepath.Join(root, "labels", "train", "dog.txt")
	writeFile(t, label, "1 0.5 0.5 0.4 0.6\n")
	writeFile(t, filepath.Join(root, "dataset.yaml"), "names:\n  0: \"cat\"\n  1: \"dog\"\n")

	inputs, err := DiscoverInputs(image, "", "")
	if err != nil {
		t.Fatal(err)
	}
	if inputs.ImagePath != image {
		t.Fatalf("image path = %q, want %q", inputs.ImagePath, image)
	}
	if inputs.LabelsPath != label {
		t.Fatalf("label path = %q, want %q", inputs.LabelsPath, label)
	}
	wantClasses := map[int]string{0: "cat", 1: "dog"}
	if !reflect.DeepEqual(inputs.Classes, wantClasses) {
		t.Fatalf("classes = %#v, want %#v", inputs.Classes, wantClasses)
	}
}

func TestMissingImplicitLabelRepresentsEmptySample(t *testing.T) {
	root := t.TempDir()
	image := filepath.Join(root, "images", "val", "empty.png")
	writeFile(t, image, "")
	writeFile(t, filepath.Join(root, "dataset.yaml"), "names: [cat, dog]\n")

	inputs, err := DiscoverInputs(image, "", "")
	if err != nil {
		t.Fatal(err)
	}
	if inputs.LabelsPath != "" {
		t.Fatalf("label path = %q, want empty", inputs.LabelsPath)
	}
	wantClasses := map[int]string{0: "cat", 1: "dog"}
	if !reflect.DeepEqual(inputs.Classes, wantClasses) {
		t.Fatalf("classes = %#v, want %#v", inputs.Classes, wantClasses)
	}
}

func TestExplicitTextClassesAndLabels(t *testing.T) {
	root := t.TempDir()
	image := filepath.Join(root, "photo.jpg")
	writeFile(t, image, "")
	labels := filepath.Join(root, "annotations.txt")
	writeFile(t, labels, "0 0.5 0.5 1 1\n")
	classes := filepath.Join(root, "classes.txt")
	writeFile(t, classes, "dog\ncar\n")

	inputs, err := DiscoverInputs(image, labels, classes)
	if err != nil {
		t.Fatal(err)
	}
	if inputs.LabelsPath != labels {
		t.Fatalf("label path = %q, want %q", inputs.LabelsPath, labels)
	}
	wantClasses := map[int]string{0: "dog", 1: "car"}
	if !reflect.DeepEqual(inputs.Classes, wantClasses) {
		t.Fatalf("classes = %#v, want %#v", inputs.Classes, wantClasses)
	}
}

func TestLoadsAndNamesNormalizedYOLOBoxes(t *testing.T) {
	labels := filepath.Join(t.TempDir(), "labels.txt")
	writeFile(t, labels, "2 0.25 0.75 0.2 0.4\n")

	boxes, err := LoadYOLOBoxes(labels, map[int]string{2: "car"})
	if err != nil {
		t.Fatal(err)
	}
	if len(boxes) != 1 {
		t.Fatalf("box count = %d, want 1", len(boxes))
	}
	box := boxes[0]
	if box.Label != "car" || box.XMin != 0.15 || box.YMin != 0.55 ||
		box.XMax != 0.35 || box.YMax != 0.95 {
		t.Fatalf("unexpected box: %#v", box)
	}
}

func TestRejectsInvalidYOLORows(t *testing.T) {
	tests := []struct {
		name     string
		contents string
		message  string
	}{
		{"class", "dog 0.5 0.5 0.2 0.2\n", "class id"},
		{"fields", "0 0.5 0.5 0.2\n", "five fields"},
		{"range", "0 1.5 0.5 0.2 0.2\n", "between 0 and 1"},
		{"size", "0 0.5 0.5 0 0.2\n", "greater than zero"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			labels := filepath.Join(t.TempDir(), "labels.txt")
			writeFile(t, labels, test.contents)
			_, err := LoadYOLOBoxes(labels, nil)
			if err == nil || !strings.Contains(err.Error(), test.message) {
				t.Fatalf("error = %v, want message containing %q", err, test.message)
			}
		})
	}
}

func TestRejectsTerminalControlCharactersInClassNames(t *testing.T) {
	classes := filepath.Join(t.TempDir(), "dataset.yaml")
	writeFile(t, classes, "names:\n  0: \"dog\\u001b[2J\"\n")

	_, err := LoadClasses(classes)

	if err == nil || !strings.Contains(err.Error(), "control characters") {
		t.Fatalf("error = %v, want control-character rejection", err)
	}
}
