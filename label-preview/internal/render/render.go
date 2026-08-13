package render

import (
	"fmt"
	"image"
	"image/color"
	"math"
	"strings"

	"github.com/rellimrebo/toolbox/label-preview/internal/dataset"
)

const reset = "\x1b[0m"
const maxPreviewDimension = 4096
const maxPreviewCells = 1_000_000

var luminanceRamp = []rune(" .:-=+*#%@")

type rgb struct {
	red   uint8
	green uint8
	blue  uint8
}

type cell struct {
	glyph         rune
	foreground    rgb
	background    rgb
	hasForeground bool
	hasBackground bool
}

var palette = []rgb{
	{255, 99, 132},
	{54, 162, 235},
	{255, 206, 86},
	{75, 192, 192},
	{153, 102, 255},
	{255, 159, 64},
	{136, 216, 176},
	{238, 130, 238},
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

func imageCells(source image.Image, columns int, rows int, useColor bool) [][]cell {
	pixels := resize(source, columns, rows*2)
	canvas := make([][]cell, rows)
	for row := range rows {
		canvas[row] = make([]cell, columns)
		for column := range columns {
			upper := pixels[row*2*columns+column]
			lower := pixels[(row*2+1)*columns+column]
			if useColor {
				canvas[row][column] = cell{
					glyph:         '▀',
					foreground:    upper,
					background:    lower,
					hasForeground: true,
					hasBackground: true,
				}
			} else {
				canvas[row][column] = cell{glyph: monochromeGlyph(upper, lower)}
			}
		}
	}
	return canvas
}

func boxExtent(start float64, end float64, size int) (int, int) {
	first := min(size-1, max(0, int(math.Floor(start*float64(size)))))
	last := min(size-1, max(first, int(math.Ceil(end*float64(size)))-1))
	return first, last
}

func paint(target *cell, glyph rune, foreground rgb, label bool) {
	target.glyph = glyph
	target.foreground = foreground
	target.hasForeground = true
	if label {
		target.background = rgb{}
		target.hasBackground = true
	}
}

func drawBox(canvas [][]cell, box dataset.Box) {
	rows, columns := len(canvas), len(canvas[0])
	left, right := boxExtent(box.XMin, box.XMax, columns)
	top, bottom := boxExtent(box.YMin, box.YMax, rows)
	boxColor := palette[box.ClassID%len(palette)]

	if left == right {
		for row := top; row <= bottom; row++ {
			paint(&canvas[row][left], '│', boxColor, false)
		}
		return
	}
	if top == bottom {
		for column := left; column <= right; column++ {
			paint(&canvas[top][column], '─', boxColor, false)
		}
		return
	}

	for column := left + 1; column < right; column++ {
		paint(&canvas[top][column], '─', boxColor, false)
		paint(&canvas[bottom][column], '─', boxColor, false)
	}
	for row := top + 1; row < bottom; row++ {
		paint(&canvas[row][left], '│', boxColor, false)
		paint(&canvas[row][right], '│', boxColor, false)
	}
	paint(&canvas[top][left], '┌', boxColor, false)
	paint(&canvas[top][right], '┐', boxColor, false)
	paint(&canvas[bottom][left], '└', boxColor, false)
	paint(&canvas[bottom][right], '┘', boxColor, false)

	label := []rune(box.Label)
	for offset := 1; offset < right-left && offset <= len(label); offset++ {
		paint(&canvas[top][left+offset], label[offset-1], boxColor, true)
	}
}

func ansiRow(cells []cell) string {
	var output strings.Builder
	var activeForeground rgb
	var activeBackground rgb
	hasActiveForeground, hasActiveBackground := false, false
	for _, current := range cells {
		if current.hasForeground != hasActiveForeground ||
			(current.hasForeground && current.foreground != activeForeground) {
			if current.hasForeground {
				fmt.Fprintf(
					&output,
					"\x1b[38;2;%d;%d;%dm",
					current.foreground.red,
					current.foreground.green,
					current.foreground.blue,
				)
			} else {
				output.WriteString(reset)
				hasActiveBackground = false
			}
			activeForeground = current.foreground
			hasActiveForeground = current.hasForeground
		}
		if current.hasBackground != hasActiveBackground ||
			(current.hasBackground && current.background != activeBackground) {
			if current.hasBackground {
				fmt.Fprintf(
					&output,
					"\x1b[48;2;%d;%d;%dm",
					current.background.red,
					current.background.green,
					current.background.blue,
				)
			} else {
				output.WriteString("\x1b[49m")
			}
			activeBackground = current.background
			hasActiveBackground = current.hasBackground
		}
		output.WriteRune(current.glyph)
	}
	output.WriteString(reset)
	return output.String()
}

func Render(
	source image.Image,
	boxes []dataset.Box,
	width int,
	maxHeight int,
	useColor bool,
) (string, error) {
	bounds := source.Bounds()
	columns, rows, err := FitTerminalSize(bounds.Dx(), bounds.Dy(), width, maxHeight)
	if err != nil {
		return "", err
	}
	canvas := imageCells(source, columns, rows, useColor)
	for _, box := range boxes {
		drawBox(canvas, box)
	}

	var output strings.Builder
	fmt.Fprintf(&output, "┌%s┐\n", strings.Repeat("─", columns))
	for _, row := range canvas {
		output.WriteRune('│')
		if useColor {
			output.WriteString(ansiRow(row))
		} else {
			for _, current := range row {
				output.WriteRune(current.glyph)
			}
		}
		output.WriteString("│\n")
	}
	fmt.Fprintf(&output, "└%s┘", strings.Repeat("─", columns))
	return output.String(), nil
}
