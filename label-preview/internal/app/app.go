package app

import (
	"errors"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"os"
	"strconv"
	"strings"

	"golang.org/x/term"

	"github.com/rellimrebo/toolbox/label-preview/internal/dataset"
	"github.com/rellimrebo/toolbox/label-preview/internal/render"
)

const usage = `Usage: label-preview IMAGE [options]

Render an image from a recognized labeled dataset as a compact terminal preview.

Dataset formats are detected from their standard layout:
  YOLO detection, COCO detection JSON, and Pascal VOC XML.

Options:
  --width CELLS       maximum image width (default: available terminal width)
  --max-height ROWS   maximum image height (default: available terminal height)
  -h, --help          show this help
`

type options struct {
	imagePath string
	width     int
	maxHeight int
}

func optionValue(arguments []string, index *int, name string) (string, error) {
	argument := arguments[*index]
	if before, after, found := strings.Cut(argument, "="); found {
		if before != name || after == "" {
			return "", fmt.Errorf("%s requires a value", name)
		}
		return after, nil
	}
	if *index+1 >= len(arguments) {
		return "", fmt.Errorf("%s requires a value", name)
	}
	*index++
	return arguments[*index], nil
}

func positiveInteger(value string, name string) (int, error) {
	parsed, err := strconv.Atoi(value)
	if err != nil || parsed <= 0 {
		return 0, fmt.Errorf("%s must be greater than zero", name)
	}
	return parsed, nil
}

func parseArguments(arguments []string) (options, bool, error) {
	result := options{}
	for index := 0; index < len(arguments); index++ {
		argument := arguments[index]
		switch {
		case argument == "-h" || argument == "--help":
			return result, true, nil
		case argument == "--width" || strings.HasPrefix(argument, "--width="):
			value, err := optionValue(arguments, &index, "--width")
			if err != nil {
				return result, false, err
			}
			result.width, err = positiveInteger(value, "--width")
			if err != nil {
				return result, false, err
			}
		case argument == "--max-height" || strings.HasPrefix(argument, "--max-height="):
			value, err := optionValue(arguments, &index, "--max-height")
			if err != nil {
				return result, false, err
			}
			result.maxHeight, err = positiveInteger(value, "--max-height")
			if err != nil {
				return result, false, err
			}
		case strings.HasPrefix(argument, "-"):
			return result, false, fmt.Errorf("unknown option: %s", argument)
		case result.imagePath == "":
			result.imagePath = argument
		default:
			return result, false, fmt.Errorf("unexpected argument: %s", argument)
		}
	}
	if result.imagePath == "" {
		return result, false, errors.New("an image path is required")
	}
	return result, false, nil
}

type fileDescriptor interface {
	Fd() uintptr
}

func terminalSize(output io.Writer) (int, int) {
	if descriptor, ok := output.(fileDescriptor); ok {
		if width, height, err := term.GetSize(int(descriptor.Fd())); err == nil {
			return width, height
		}
	}
	return 80, 24
}

func run(arguments []string, output io.Writer) error {
	options, help, err := parseArguments(arguments)
	if err != nil {
		return err
	}
	if help {
		_, err = io.WriteString(output, usage)
		return err
	}

	terminalWidth, terminalHeight := terminalSize(output)
	if options.width == 0 {
		options.width = max(1, terminalWidth-2)
	}
	if options.maxHeight == 0 {
		options.maxHeight = max(1, terminalHeight-2)
	}

	imageFile, err := os.Open(options.imagePath)
	if err != nil {
		return fmt.Errorf("open image: %w", err)
	}
	defer imageFile.Close()
	decoded, _, err := image.Decode(imageFile)
	if err != nil {
		return fmt.Errorf("decode image: %w", err)
	}
	bounds := decoded.Bounds()
	annotations, err := dataset.LoadAnnotations(options.imagePath, bounds.Dx(), bounds.Dy())
	if err != nil {
		return err
	}
	preview, err := render.Render(
		decoded,
		annotations.Boxes,
		options.width,
		options.maxHeight,
	)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintln(output, preview)
	return err
}

func Run(arguments []string, stdout io.Writer, stderr io.Writer) int {
	if err := run(arguments, stdout); err != nil {
		fmt.Fprintf(stderr, "label-preview: %v\n\n%s", err, usage)
		return 2
	}
	return 0
}
