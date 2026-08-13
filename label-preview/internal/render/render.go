package render

import (
	"fmt"
	"image"
	"image/color"
	"math"
	"strings"

	"github.com/rellimrebo/toolbox/label-preview/internal/dataset"
)

const maxPreviewDimension = 4096
const maxPreviewCells = 1_000_000

var luminanceRamp = []rune(" .:-=+*#%@")

type rgb struct {
	red   uint8
	green uint8
	blue  uint8
}

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

func max(first int, second int) int {
	if first > second {
		return first
	}
	return second
}

func clamp(value float64, minimum float64, maximum float64) float64 {
	return math.Min(maximum, math.Max(minimum, value))
}

func sourceRGB(source image.Image, x int, y int) rgb {
	pixel := color.NRGBAModel.Convert(source.At(x, y)).(color.NRGBA)
	alpha := uint16(pixel.A)
	return rgb{
		red:   uint8(uint16(pixel.R) * alpha / 255),
		green: uint8(uint16(pixel.G) * alpha / 255),
		blue:  uint8(uint16(pixel.B) * alpha / 255),
	}
}

func interpolate(first uint8, second uint8, amount float64) float64 {
	return float64(first)*(1-amount) + float64(second)*amount
}

func bilinear(source image.Image, x float64, y float64) rgb {
	bounds := source.Bounds()
	x = clamp(x, 0, float64(bounds.Dx()-1))
	y = clamp(y, 0, float64(bounds.Dy()-1))
	x0, y0 := int(math.Floor(x)), int(math.Floor(y))
	x1, y1 := min(x0+1, bounds.Dx()-1), min(y0+1, bounds.Dy()-1)
	xAmount, yAmount := x-float64(x0), y-float64(y0)

	topLeft := sourceRGB(source, bounds.Min.X+x0, bounds.Min.Y+y0)
	topRight := sourceRGB(source, bounds.Min.X+x1, bounds.Min.Y+y0)
	bottomLeft := sourceRGB(source, bounds.Min.X+x0, bounds.Min.Y+y1)
	bottomRight := sourceRGB(source, bounds.Min.X+x1, bounds.Min.Y+y1)

	red := interpolate(
		uint8(interpolate(topLeft.red, topRight.red, xAmount)),
		uint8(interpolate(bottomLeft.red, bottomRight.red, xAmount)),
		yAmount,
	)
	green := interpolate(
		uint8(interpolate(topLeft.green, topRight.green, xAmount)),
		uint8(interpolate(bottomLeft.green, bottomRight.green, xAmount)),
		yAmount,
	)
	blue := interpolate(
		uint8(interpolate(topLeft.blue, topRight.blue, xAmount)),
		uint8(interpolate(bottomLeft.blue, bottomRight.blue, xAmount)),
		yAmount,
	)
	return rgb{uint8(math.Round(red)), uint8(math.Round(green)), uint8(math.Round(blue))}
}

func resize(source image.Image, width int, height int) []rgb {
	bounds := source.Bounds()
	pixels := make([]rgb, width*height)
	for y := range height {
		sourceY := (float64(y)+0.5)*float64(bounds.Dy())/float64(height) - 0.5
		for x := range width {
			sourceX := (float64(x)+0.5)*float64(bounds.Dx())/float64(width) - 0.5
			pixels[y*width+x] = bilinear(source, sourceX, sourceY)
		}
	}
	return pixels
}

func monochromeGlyph(upper rgb, lower rgb) rune {
	red := (float64(upper.red) + float64(lower.red)) / 2
	green := (float64(upper.green) + float64(lower.green)) / 2
	blue := (float64(upper.blue) + float64(lower.blue)) / 2
	luminance := 0.2126*red + 0.7152*green + 0.0722*blue
	index := int(math.Round(luminance / 255 * float64(len(luminanceRamp)-1)))
	return luminanceRamp[index]
}

func imageCells(source image.Image, columns int, rows int) [][]rune {
	pixels := resize(source, columns, rows*2)
	canvas := make([][]rune, rows)
	for row := range rows {
		canvas[row] = make([]rune, columns)
		for column := range columns {
			upper := pixels[row*2*columns+column]
			lower := pixels[(row*2+1)*columns+column]
			canvas[row][column] = monochromeGlyph(upper, lower)
		}
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
	source image.Image,
	boxes []dataset.Box,
	width int,
	maxHeight int,
) (string, error) {
	bounds := source.Bounds()
	columns, rows, err := FitTerminalSize(bounds.Dx(), bounds.Dy(), width, maxHeight)
	if err != nil {
		return "", err
	}
	canvas := imageCells(source, columns, rows)
	for _, box := range boxes {
		drawBox(canvas, box)
	}

	var output strings.Builder
	fmt.Fprintf(&output, "┌%s┐\n", strings.Repeat("─", columns))
	for _, row := range canvas {
		output.WriteRune('│')
		for _, current := range row {
			output.WriteRune(current)
		}
		output.WriteString("│\n")
	}
	fmt.Fprintf(&output, "└%s┘", strings.Repeat("─", columns))
	return output.String(), nil
}
