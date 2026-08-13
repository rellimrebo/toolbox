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

Render a YOLO-labeled PNG, JPEG, or GIF as a compact terminal preview.

Options:
  --labels PATH       YOLO label file; inferred from images/<split>/ by default
  --classes PATH      dataset.yaml or one-name-per-line text file
  --width CELLS       maximum image width (default: available terminal width)
  --max-height ROWS   maximum image height (default: available terminal height)
  --color MODE        auto, always, or never (default: auto)
  -h, --help          show this help
`

type options struct {
	imagePath   string
	labelsPath  string
	classesPath string
	width       int
	maxHeight   int
	color       string
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
	result := options{color: "auto"}
	for index := 0; index < len(arguments); index++ {
		argument := arguments[index]
		switch {
		case argument == "-h" || argument == "--help":
			return result, true, nil
		case argument == "--labels" || strings.HasPrefix(argument, "--labels="):
			value, err := optionValue(arguments, &index, "--labels")
			if err != nil {
				return result, false, err
			}
			result.labelsPath = value
		case argument == "--classes" || strings.HasPrefix(argument, "--classes="):
			value, err := optionValue(arguments, &index, "--classes")
			if err != nil {
				return result, false, err
			}
			result.classesPath = value
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
		case argument == "--color" || strings.HasPrefix(argument, "--color="):
			value, err := optionValue(arguments, &index, "--color")
			if err != nil {
				return result, false, err
			}
			if value != "auto" && value != "always" && value != "never" {
				return result, false, errors.New("--color must be auto, always, or never")
			}
			result.color = value
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

func useColor(mode string, output io.Writer) bool {
	if mode == "always" {
		return true
	}
	_, noColor := os.LookupEnv("NO_COLOR")
	if mode == "never" || noColor || os.Getenv("TERM") == "dumb" {
		return false
	}
	descriptor, ok := output.(fileDescriptor)
	return ok && term.IsTerminal(int(descriptor.Fd()))
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

	inputs, err := dataset.DiscoverInputs(
		options.imagePath,
		options.labelsPath,
		options.classesPath,
	)
	if err != nil {
		return err
	}
	var boxes []dataset.Box
	if inputs.LabelsPath != "" {
		boxes, err = dataset.LoadYOLOBoxes(inputs.LabelsPath, inputs.Classes)
		if err != nil {
			return err
		}

	}
	imageFile, err := os.Open(inputs.ImagePath)
	if err != nil {
		return fmt.Errorf("open image: %w", err)
	}
	defer imageFile.Close()
	decoded, _, err := image.Decode(imageFile)
	if err != nil {
		return fmt.Errorf("decode image: %w", err)
	}
	preview, err := render.Render(
		decoded,
		boxes,
		options.width,
		options.maxHeight,
		useColor(options.color, output),
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
