package dataset

import (
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"unicode"
	"unicode/utf8"
)

type format string

const (
	formatYOLO      format = "YOLO"
	formatCOCO      format = "COCO"
	formatPascalVOC format = "Pascal VOC"
)

type Box struct {
	Label string
	XMin  float64
	YMin  float64
	XMax  float64
	YMax  float64
}

type source struct {
	format         format
	annotationPath string
	metadataPath   string
	datasetRoot    string
}

func existingImage(path string) (string, error) {
	absolute, err := filepath.Abs(path)
	if err != nil {
		return "", fmt.Errorf("resolve image: %w", err)
	}
	info, err := os.Stat(absolute)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return "", fmt.Errorf("image does not exist: %s", path)
		}
		return "", fmt.Errorf("inspect image: %w", err)
	}
	if !info.Mode().IsRegular() {
		return "", fmt.Errorf("image is not a file: %s", path)
	}
	return absolute, nil
}

func firstExisting(paths ...string) string {
	for _, path := range paths {
		if path == "" {
			continue
		}
		if info, err := os.Stat(path); err == nil && info.Mode().IsRegular() {
			absolute, absoluteErr := filepath.Abs(path)
			if absoluteErr == nil {
				return absolute
			}
		}
	}
	return ""
}

func discoverYOLO(imagePath string) (source, bool) {
	for imagesDirectory := filepath.Dir(imagePath); ; imagesDirectory = filepath.Dir(imagesDirectory) {
		if filepath.Base(imagesDirectory) == "images" {
			relativeImage, err := filepath.Rel(imagesDirectory, imagePath)
			if err != nil {
				return source{}, false
			}
			layoutRoot := filepath.Dir(imagesDirectory)
			labelPath := filepath.Join(
				layoutRoot,
				"labels",
				strings.TrimSuffix(relativeImage, filepath.Ext(relativeImage))+".txt",
			)
			for _, directory := range []string{layoutRoot, filepath.Dir(layoutRoot)} {
				configPath := firstExisting(
					filepath.Join(directory, "dataset.yaml"),
					filepath.Join(directory, "dataset.yml"),
					filepath.Join(directory, "data.yaml"),
					filepath.Join(directory, "data.yml"),
				)
				if configPath != "" {
					return source{
						format:         formatYOLO,
						annotationPath: labelPath,
						metadataPath:   configPath,
					}, true
				}
			}
			return source{}, false
		}
		parent := filepath.Dir(imagesDirectory)
		if parent == imagesDirectory {
			return source{}, false
		}
	}
}

func discoverCOCO(imagePath string) (source, bool) {
	imageDirectory := filepath.Dir(imagePath)
	local := firstExisting(
		filepath.Join(imageDirectory, "_annotations.coco.json"),
		filepath.Join(imageDirectory, "annotations.coco.json"),
	)
	if local != "" {
		return source{
			format:         formatCOCO,
			annotationPath: local,
			datasetRoot:    imageDirectory,
		}, true
	}

	split := filepath.Base(imageDirectory)
	parent := filepath.Dir(imageDirectory)
	for _, directory := range []string{parent, filepath.Dir(parent)} {
		candidate := firstExisting(
			filepath.Join(directory, "annotations", "instances_"+split+".json"),
		)
		if candidate != "" {
			return source{
				format:         formatCOCO,
				annotationPath: candidate,
				datasetRoot:    directory,
			}, true
		}
	}
	return source{}, false
}

func discoverPascalVOC(imagePath string) (source, bool) {
	sidecar := strings.TrimSuffix(imagePath, filepath.Ext(imagePath)) + ".xml"
	if path := firstExisting(sidecar); path != "" {
		return source{format: formatPascalVOC, annotationPath: path}, true
	}
	imageDirectory := filepath.Dir(imagePath)
	if filepath.Base(imageDirectory) != "JPEGImages" {
		return source{}, false
	}
	annotation := filepath.Join(
		filepath.Dir(imageDirectory),
		"Annotations",
		strings.TrimSuffix(filepath.Base(imagePath), filepath.Ext(imagePath))+".xml",
	)
	if path := firstExisting(annotation); path != "" {
		return source{format: formatPascalVOC, annotationPath: path}, true
	}
	return source{}, false
}

func LoadAnnotations(imagePath string, imageWidth int, imageHeight int) ([]Box, error) {
	image, err := existingImage(imagePath)
	if err != nil {
		return nil, err
	}
	if imageWidth <= 0 || imageHeight <= 0 {
		return nil, errors.New("image dimensions must be greater than zero")
	}

	discoverers := []func(string) (source, bool){discoverPascalVOC, discoverYOLO, discoverCOCO}
	var sources []source
	for _, discover := range discoverers {
		if candidate, found := discover(image); found {
			sources = append(sources, candidate)
		}
	}
	if len(sources) == 0 {
		return nil, fmt.Errorf(
			"image is not in a recognized dataset layout (supported: YOLO, COCO detection, Pascal VOC)",
		)
	}
	if len(sources) > 1 {
		formats := make([]string, len(sources))
		for index, candidate := range sources {
			formats[index] = string(candidate.format)
		}
		return nil, fmt.Errorf(
			"dataset layout is ambiguous; matched %s",
			strings.Join(formats, " and "),
		)
	}

	candidate := sources[0]
	var boxes []Box
	switch candidate.format {
	case formatYOLO:
		boxes, err = loadYOLO(candidate.annotationPath, candidate.metadataPath)
	case formatCOCO:
		boxes, err = loadCOCO(
			candidate.annotationPath,
			candidate.datasetRoot,
			image,
			imageWidth,
			imageHeight,
		)
	case formatPascalVOC:
		boxes, err = loadPascalVOC(candidate.annotationPath, imageWidth, imageHeight)
	default:
		err = fmt.Errorf("unsupported annotation format: %s", candidate.format)
	}
	if err != nil {
		return nil, err
	}
	return boxes, nil
}

func validateClassName(value string) (string, error) {
	if strings.TrimSpace(value) == "" {
		return "", errors.New("class names must not be empty")
	}
	if !utf8.ValidString(value) {
		return "", errors.New("class names must be valid UTF-8")
	}
	for _, character := range value {
		if unicode.IsControl(character) || unicode.Is(unicode.Cf, character) {
			return "", errors.New("class names must not contain control characters")
		}
	}
	return value, nil
}

func normalizedBox(
	label string,
	xMin float64,
	yMin float64,
	xMax float64,
	yMax float64,
	imageWidth int,
	imageHeight int,
) (Box, error) {
	coordinates := []float64{xMin, yMin, xMax, yMax}
	for _, value := range coordinates {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return Box{}, errors.New("box coordinates must be finite")
		}
	}
	xMin = math.Max(0, math.Min(float64(imageWidth), xMin))
	yMin = math.Max(0, math.Min(float64(imageHeight), yMin))
	xMax = math.Max(0, math.Min(float64(imageWidth), xMax))
	yMax = math.Max(0, math.Min(float64(imageHeight), yMax))
	if xMax <= xMin || yMax <= yMin {
		return Box{}, errors.New("box must have positive area inside the image")
	}
	label, err := validateClassName(label)
	if err != nil {
		return Box{}, err
	}
	return Box{
		Label: label,
		XMin:  xMin / float64(imageWidth),
		YMin:  yMin / float64(imageHeight),
		XMax:  xMax / float64(imageWidth),
		YMax:  yMax / float64(imageHeight),
	}, nil
}
