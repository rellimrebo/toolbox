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

func TestCLIRendersExplicitYOLOSampleWithoutColor(t *testing.T) {
	root := t.TempDir()
	imagePath := filepath.Join(root, "dog.png")
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
	labelsPath := filepath.Join(root, "dog.txt")
	if err := os.WriteFile(labelsPath, []byte("0 0.5 0.5 0.5 0.5\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	classesPath := filepath.Join(root, "classes.txt")
	if err := os.WriteFile(classesPath, []byte("dog\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	exitCode := Run([]string{
		imagePath,
		"--labels", labelsPath,
		"--classes", classesPath,
		"--width", "20",
		"--max-height", "10",
		"--color", "never",
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
