package render

import (
	"fmt"
	"math"
	"strings"

	"github.com/rellimrebo/toolbox/label-preview/internal/dataset"
)

const maxPreviewDimension = 4096
const maxPreviewCells = 1_000_000

func FitTerminalSize(imageWidth int, imageHeight int, maxWidth int, maxHeight int) (int, int, error) {
	if imageWidth <= 0 || imageHeight <= 0 {
		return 0, 0, fmt.Errorf("image dimensions must be greater than zero")
	}
	if maxWidth <= 0 || maxHeight <= 0 {
		return 0, 0, fmt.Errorf("preview dimensions must be greater than zero")
	}
	if maxWidth > maxPreviewDimension || maxHeight > maxPreviewDimension {
		return 0, 0, fmt.Errorf("preview dimensions must not exceed %d", maxPreviewDimension)
	}

	columns := maxWidth
	rows := max(
		1,
		int(math.Round(float64(columns)*float64(imageHeight)/float64(imageWidth)/2)),
	)
	if rows > maxHeight {
		rows = maxHeight
		columns = max(
			1,
			int(math.Round(float64(rows)*2*float64(imageWidth)/float64(imageHeight))),
		)
	}
	if columns > maxPreviewCells/rows {
		return 0, 0, fmt.Errorf("preview must not exceed %d terminal cells", maxPreviewCells)
	}
	return columns, rows, nil
}

func blankCanvas(columns int, rows int) [][]rune {
	canvas := make([][]rune, rows)
	for row := range rows {
		canvas[row] = []rune(strings.Repeat(" ", columns))
	}
	return canvas
}

func boxExtent(start float64, end float64, size int) (int, int) {
	first := min(size-1, max(0, int(math.Floor(start*float64(size)))))
	last := min(size-1, max(first, int(math.Ceil(end*float64(size)))-1))
	return first, last
}

func drawBox(canvas [][]rune, box dataset.Box) {
	rows, columns := len(canvas), len(canvas[0])
	left, right := boxExtent(box.XMin, box.XMax, columns)
	top, bottom := boxExtent(box.YMin, box.YMax, rows)

	if left == right {
		for row := top; row <= bottom; row++ {
			canvas[row][left] = '│'
		}
		return
	}
	if top == bottom {
		for column := left; column <= right; column++ {
			canvas[top][column] = '─'
		}
		return
	}

	for column := left + 1; column < right; column++ {
		canvas[top][column] = '─'
		canvas[bottom][column] = '─'
	}
	for row := top + 1; row < bottom; row++ {
		canvas[row][left] = '│'
		canvas[row][right] = '│'
	}
	canvas[top][left] = '┌'
	canvas[top][right] = '┐'
	canvas[bottom][left] = '└'
	canvas[bottom][right] = '┘'

	label := []rune(box.Label)
	for offset := 1; offset < right-left && offset <= len(label); offset++ {
		canvas[top][left+offset] = label[offset-1]
	}
}

func Render(
	imageWidth int,
	imageHeight int,
	boxes []dataset.Box,
	width int,
	maxHeight int,
) (string, error) {
	columns, rows, err := FitTerminalSize(imageWidth, imageHeight, width, maxHeight)
	if err != nil {
		return "", err
	}
	canvas := blankCanvas(columns, rows)
	for _, box := range boxes {
		drawBox(canvas, box)
	}

	var output strings.Builder
	fmt.Fprintf(&output, "┌%s┐\n", strings.Repeat("─", columns))
	for _, row := range canvas {
		output.WriteRune('│')
		output.WriteString(string(row))
		output.WriteString("│\n")
	}
	fmt.Fprintf(&output, "└%s┘", strings.Repeat("─", columns))
	return output.String(), nil
}
