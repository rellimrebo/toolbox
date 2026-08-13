package app

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCLIRendersYOLOFrameAndBoxOnly(t *testing.T) {
	root := t.TempDir()
	imagePath := filepath.Join(root, "images", "train", "dog.png")
	if err := os.MkdirAll(filepath.Dir(imagePath), 0o755); err != nil {
		t.Fatal(err)
	}
	file, err := os.Create(imagePath)
	if err != nil {
		t.Fatal(err)
	}
	source := image.NewRGBA(image.Rect(0, 0, 40, 20))
	for y := range 20 {
		for x := range 40 {
			source.Set(x, y, color.White)
		}
	}
	if err := png.Encode(file, source); err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	labelsPath := filepath.Join(root, "labels", "train", "dog.txt")
	if err := os.MkdirAll(filepath.Dir(labelsPath), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(labelsPath, []byte("0 0.5 0.5 0.5 0.5\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "dataset.yaml"), []byte("names: [dog]\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	exitCode := Run([]string{
		imagePath,
		"--width", "20",
		"--max-height", "10",
	}, &stdout, &stderr)

	if exitCode != 0 {
		t.Fatalf("exit code = %d, stderr = %q", exitCode, stderr.String())
	}
	if !strings.Contains(stdout.String(), "dog") {
		t.Fatalf("output does not contain label: %q", stdout.String())
	}
	if strings.Contains(stdout.String(), "\x1b[") {
		t.Fatalf("output unexpectedly contains ANSI color: %q", stdout.String())
	}
	if strings.ContainsAny(stdout.String(), ".:=+*#%@") {
		t.Fatalf("output unexpectedly contains raster characters: %q", stdout.String())
	}
}

func TestCLIRejectsRemovedColorOption(t *testing.T) {
	var stdout bytes.Buffer
	var stderr bytes.Buffer

	exitCode := Run([]string{"image.jpg", "--color", "always"}, &stdout, &stderr)

	if exitCode != 2 {
		t.Fatalf("exit code = %d, want 2", exitCode)
	}
	if !strings.Contains(stderr.String(), "unknown option: --color") {
		t.Fatalf("stderr = %q", stderr.String())
	}
}

func TestCLIReportsMissingImage(t *testing.T) {
	var stdout bytes.Buffer
	var stderr bytes.Buffer

	exitCode := Run(nil, &stdout, &stderr)

	if exitCode != 2 {
		t.Fatalf("exit code = %d, want 2", exitCode)
	}
	if !strings.Contains(stderr.String(), "an image path is required") {
		t.Fatalf("stderr = %q", stderr.String())
	}
}

func TestCLIRejectsManualAnnotationPaths(t *testing.T) {
	var stdout bytes.Buffer
	var stderr bytes.Buffer

	exitCode := Run([]string{"image.jpg", "--labels", "labels.txt"}, &stdout, &stderr)

	if exitCode != 2 {
		t.Fatalf("exit code = %d, want 2", exitCode)
	}
	if !strings.Contains(stderr.String(), "unknown option: --labels") {
		t.Fatalf("stderr = %q", stderr.String())
	}
}
