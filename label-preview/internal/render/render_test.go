package render

import (
	"image"
	"image/color"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/rellimrebo/toolbox/label-preview/internal/dataset"
)

func TestFitAccountsForTallTerminalCells(t *testing.T) {
	width, height, err := FitTerminalSize(400, 200, 40, 40)
	if err != nil {
		t.Fatal(err)
	}
	if width != 40 || height != 10 {
		t.Fatalf("size = (%d, %d), want (40, 10)", width, height)
	}
}

func TestFitRespectsHeightAndImageShape(t *testing.T) {
	width, height, err := FitTerminalSize(400, 800, 80, 10)
	if err != nil {
		t.Fatal(err)
	}
	if width != 10 || height != 10 {
		t.Fatalf("size = (%d, %d), want (10, 10)", width, height)
	}
}

func TestFitRejectsUnreasonableOutputDimensions(t *testing.T) {
	_, _, err := FitTerminalSize(100, 100, maxPreviewDimension+1, 10)
	if err == nil || !strings.Contains(err.Error(), "must not exceed") {
		t.Fatalf("error = %v, want dimension limit", err)
	}
}

func TestFitRejectsUnreasonableOutputArea(t *testing.T) {
	_, _, err := FitTerminalSize(100, 100, maxPreviewDimension, maxPreviewDimension)
	if err == nil || !strings.Contains(err.Error(), "terminal cells") {
		t.Fatalf("error = %v, want cell-count limit", err)
	}
}

func TestMonochromePreviewDrawsNamedBox(t *testing.T) {
	source := image.NewRGBA(image.Rect(0, 0, 16, 8))
	for y := range 8 {
		for x := range 16 {
			source.Set(x, y, color.White)
		}
	}
	box := dataset.Box{ClassID: 0, Label: "dog", XMin: 0.125, YMin: 0.25, XMax: 0.875, YMax: 0.75}

	preview, err := Render(source, []dataset.Box{box}, 16, 8)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(preview, "\n")
	if len(lines) != 6 {
		t.Fatalf("line count = %d, want 6\n%s", len(lines), preview)
	}
	for _, line := range lines {
		if utf8.RuneCountInString(line) != 18 {
			t.Fatalf("line width = %d, want 18: %q", utf8.RuneCountInString(line), line)
		}
	}
	for _, expected := range []string{"┌dog", "┐", "└", "┘"} {
		if !strings.Contains(preview, expected) {
			t.Fatalf("preview does not contain %q:\n%s", expected, preview)
		}
	}
}
