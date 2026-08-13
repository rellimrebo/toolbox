package dataset

import (
	"bufio"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"unicode"
)

type Box struct {
	ClassID int
	Label   string
	XMin    float64
	YMin    float64
	XMax    float64
	YMax    float64
}

type Inputs struct {
	ImagePath  string
	LabelsPath string
	Classes    map[int]string
}

func existingFile(path string, description string) (string, error) {
	absolute, err := filepath.Abs(path)
	if err != nil {
		return "", fmt.Errorf("resolve %s: %w", description, err)
	}
	info, err := os.Stat(absolute)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return "", fmt.Errorf("%s does not exist: %s", description, path)
		}
		return "", fmt.Errorf("inspect %s: %w", description, err)
	}
	if !info.Mode().IsRegular() {
		return "", fmt.Errorf("%s is not a file: %s", description, path)
	}
	return absolute, nil
}

func yoloRootAndLabel(imagePath string) (string, string, bool) {
	for directory := filepath.Dir(imagePath); ; directory = filepath.Dir(directory) {
		if filepath.Base(directory) == "images" {
			relative, err := filepath.Rel(directory, imagePath)
			if err != nil {
				return "", "", false
			}
			root := filepath.Dir(directory)
			label := filepath.Join(root, "labels", strings.TrimSuffix(relative, filepath.Ext(relative))+".txt")
			return root, label, true
		}
		parent := filepath.Dir(directory)
		if parent == directory {
			break
		}
	}
	return "", "", false
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

func findDatasetFile(imagePath string, yoloRoot string) string {
	if yoloRoot != "" {
		if found := firstExisting(
			filepath.Join(yoloRoot, "dataset.yaml"),
			filepath.Join(yoloRoot, "dataset.yml"),
		); found != "" {
			return found
		}
	}
	for directory := filepath.Dir(imagePath); ; directory = filepath.Dir(directory) {
		if found := firstExisting(
			filepath.Join(directory, "dataset.yaml"),
			filepath.Join(directory, "dataset.yml"),
		); found != "" {
			return found
		}
		parent := filepath.Dir(directory)
		if parent == directory {
			return ""
		}
	}
}

func DiscoverInputs(imagePath string, labelsPath string, classesPath string) (Inputs, error) {
	image, err := existingFile(imagePath, "image")
	if err != nil {
		return Inputs{}, err
	}

	yoloRoot, yoloLabel, foundLayout := yoloRootAndLabel(image)
	labels := ""
	if labelsPath != "" {
		labels, err = existingFile(labelsPath, "label file")
		if err != nil {
			return Inputs{}, err
		}
	} else {
		if !foundLayout {
			yoloLabel = ""
		}
		labels = firstExisting(yoloLabel, strings.TrimSuffix(image, filepath.Ext(image))+".txt")
	}

	classFile := ""
	if classesPath != "" {
		classFile, err = existingFile(classesPath, "class file")
		if err != nil {
			return Inputs{}, err
		}
	} else {
		classFile = findDatasetFile(image, yoloRoot)
	}

	classes := map[int]string{}
	if classFile != "" {
		classes, err = LoadClasses(classFile)
		if err != nil {
			return Inputs{}, err
		}
	}
	return Inputs{ImagePath: image, LabelsPath: labels, Classes: classes}, nil
}

func parseScalar(value string) (string, error) {
	value = strings.TrimSpace(value)
	if value == "" {
		return "", errors.New("class names must not be empty")
	}
	if strings.HasPrefix(value, "\"") {
		parsed, err := strconv.Unquote(value)
		if err != nil {
			return "", fmt.Errorf("invalid quoted class name %q", value)
		}
		return validateClassName(parsed)
	}
	if len(value) >= 2 && value[0] == '\'' && value[len(value)-1] == '\'' {
		value = value[1 : len(value)-1]
	}
	if comment := strings.Index(value, " #"); comment >= 0 {
		value = strings.TrimSpace(value[:comment])
	}
	return validateClassName(value)
}

func validateClassName(value string) (string, error) {
	if strings.TrimSpace(value) == "" {
		return "", errors.New("class names must not be empty")
	}
	for _, character := range value {
		if unicode.IsControl(character) || unicode.Is(unicode.Cf, character) {
			return "", errors.New("class names must not contain control characters")
		}
	}
	return value, nil
}

func splitInline(value string) ([]string, error) {
	var items []string
	var current strings.Builder
	var quote rune
	escaped := false
	for _, character := range value {
		if escaped {
			current.WriteRune(character)
			escaped = false
			continue
		}
		if character == '\\' && quote != 0 {
			current.WriteRune(character)
			escaped = true
			continue
		}
		if character == '\'' || character == '"' {
			if quote == 0 {
				quote = character
			} else if quote == character {
				quote = 0
			}
			current.WriteRune(character)
			continue
		}
		if character == ',' && quote == 0 {
			items = append(items, strings.TrimSpace(current.String()))
			current.Reset()
			continue
		}
		current.WriteRune(character)
	}
	if quote != 0 {
		return nil, errors.New("unterminated quote in inline class names")
	}
	items = append(items, strings.TrimSpace(current.String()))
	return items, nil
}

func parseInlineNames(value string) (map[int]string, error) {
	if len(value) < 2 {
		return nil, errors.New("inline names must be a list or mapping")
	}
	opening, closing := value[0], value[len(value)-1]
	if (opening != '[' || closing != ']') && (opening != '{' || closing != '}') {
		return nil, errors.New("inline names must be a list or mapping")
	}
	items, err := splitInline(value[1 : len(value)-1])
	if err != nil {
		return nil, err
	}
	classes := map[int]string{}
	for index, item := range items {
		if item == "" {
			continue
		}
		classID := index
		nameValue := item
		if opening == '{' {
			rawID, rawName, found := strings.Cut(item, ":")
			if !found {
				return nil, fmt.Errorf("invalid class mapping entry %q", item)
			}
			classID, err = strconv.Atoi(strings.TrimSpace(rawID))
			if err != nil || classID < 0 {
				return nil, fmt.Errorf("invalid class id %q", rawID)
			}
			nameValue = rawName
		}
		name, parseErr := parseScalar(nameValue)
		if parseErr != nil {
			return nil, parseErr
		}
		classes[classID] = name
	}
	if len(classes) == 0 {
		return nil, errors.New("class names must not be empty")
	}
	return classes, nil
}

func indentation(line string) int {
	return len(line) - len(strings.TrimLeft(line, " \t"))
}

func loadYAMLClasses(path string) (map[int]string, error) {
	contents, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read class file: %w", err)
	}
	lines := strings.Split(string(contents), "\n")
	for namesIndex, line := range lines {
		trimmed := strings.TrimSpace(line)
		if !strings.HasPrefix(trimmed, "names:") {
			continue
		}
		inline := strings.TrimSpace(strings.TrimPrefix(trimmed, "names:"))
		if inline != "" {
			classes, parseErr := parseInlineNames(inline)
			if parseErr != nil {
				return nil, fmt.Errorf("parse %s: %w", path, parseErr)
			}
			return classes, nil
		}

		baseIndent := indentation(line)
		classes := map[int]string{}
		sequenceID := 0
		style := ""
		for _, child := range lines[namesIndex+1:] {
			childTrimmed := strings.TrimSpace(child)
			if childTrimmed == "" || strings.HasPrefix(childTrimmed, "#") {
				continue
			}
			if indentation(child) <= baseIndent {
				break
			}
			if strings.HasPrefix(childTrimmed, "-") {
				if style == "mapping" {
					return nil, fmt.Errorf("parse %s: mixed class-name styles", path)
				}
				style = "sequence"
				name, parseErr := parseScalar(strings.TrimSpace(strings.TrimPrefix(childTrimmed, "-")))
				if parseErr != nil {
					return nil, fmt.Errorf("parse %s: %w", path, parseErr)
				}
				classes[sequenceID] = name
				sequenceID++
				continue
			}
			if style == "sequence" {
				return nil, fmt.Errorf("parse %s: mixed class-name styles", path)
			}
			style = "mapping"
			rawID, rawName, found := strings.Cut(childTrimmed, ":")
			if !found {
				return nil, fmt.Errorf("parse %s: invalid class entry %q", path, childTrimmed)
			}
			classID, parseErr := strconv.Atoi(strings.TrimSpace(rawID))
			if parseErr != nil || classID < 0 {
				return nil, fmt.Errorf("parse %s: invalid class id %q", path, rawID)
			}
			name, parseErr := parseScalar(rawName)
			if parseErr != nil {
				return nil, fmt.Errorf("parse %s: %w", path, parseErr)
			}
			classes[classID] = name
		}
		if len(classes) == 0 {
			return nil, fmt.Errorf("parse %s: no class names found under names", path)
		}
		return classes, nil
	}
	return nil, fmt.Errorf("parse %s: no names section found", path)
}

func LoadClasses(path string) (map[int]string, error) {
	extension := strings.ToLower(filepath.Ext(path))
	if extension == ".yaml" || extension == ".yml" {
		return loadYAMLClasses(path)
	}
	contents, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read class file: %w", err)
	}
	lines := strings.Split(strings.TrimSuffix(string(contents), "\n"), "\n")
	classes := map[int]string{}
	for classID, line := range lines {
		name := strings.TrimSpace(line)
		name, err = validateClassName(name)
		if err != nil {
			return nil, fmt.Errorf("invalid class name in %s: %w", path, err)
		}
		classes[classID] = name
	}
	if len(classes) == 0 {
		return nil, fmt.Errorf("class file must contain at least one name: %s", path)
	}
	return classes, nil
}

func LoadYOLOBoxes(path string, classes map[int]string) ([]Box, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open label file: %w", err)
	}
	defer file.Close()

	var boxes []Box
	scanner := bufio.NewScanner(file)
	for lineNumber := 1; scanner.Scan(); lineNumber++ {
		fields := strings.Fields(scanner.Text())
		if len(fields) == 0 {
			continue
		}
		if len(fields) != 5 {
			return nil, fmt.Errorf("%s:%d: expected five fields for a YOLO box", path, lineNumber)
		}
		classID, parseErr := strconv.Atoi(fields[0])
		if parseErr != nil || classID < 0 {
			return nil, fmt.Errorf("%s:%d: class id must be a non-negative integer", path, lineNumber)
		}
		coordinates := make([]float64, 4)
		for index, field := range fields[1:] {
			coordinates[index], parseErr = strconv.ParseFloat(field, 64)
			if parseErr != nil {
				return nil, fmt.Errorf("%s:%d: box coordinates must be numbers", path, lineNumber)
			}
		}
		centerX, centerY, width, height := coordinates[0], coordinates[1], coordinates[2], coordinates[3]
		for _, value := range coordinates {
			if math.IsNaN(value) || math.IsInf(value, 0) || value < 0 || value > 1 {
				return nil, fmt.Errorf("%s:%d: box coordinates must be between 0 and 1", path, lineNumber)
			}
		}
		if width <= 0 || height <= 0 {
			return nil, fmt.Errorf("%s:%d: box width and height must be greater than zero", path, lineNumber)
		}
		label, known := classes[classID]
		if !known {
			label = strconv.Itoa(classID)
		}
		boxes = append(boxes, Box{
			ClassID: classID,
			Label:   label,
			XMin:    math.Max(0, centerX-width/2),
			YMin:    math.Max(0, centerY-height/2),
			XMax:    math.Min(1, centerX+width/2),
			YMax:    math.Min(1, centerY+height/2),
		})
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read label file: %w", err)
	}
	return boxes, nil
}
