package dataset

import (
	"bufio"
	"errors"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"

	"go.yaml.in/yaml/v3"
)

func dereferenceYAML(node *yaml.Node) *yaml.Node {
	for node.Kind == yaml.AliasNode && node.Alias != nil {
		node = node.Alias
	}
	return node
}

func findYAMLMappingValue(mapping *yaml.Node, key string) *yaml.Node {
	mapping = dereferenceYAML(mapping)
	if mapping.Kind != yaml.MappingNode {
		return nil
	}
	for index := 0; index+1 < len(mapping.Content); index += 2 {
		if mapping.Content[index].Value == key {
			return dereferenceYAML(mapping.Content[index+1])
		}
	}
	return nil
}

func loadYAMLClasses(path string) (map[int]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open YOLO dataset config: %w", err)
	}
	defer file.Close()
	decoder := yaml.NewDecoder(file)
	var document yaml.Node
	if err := decoder.Decode(&document); err != nil {
		return nil, fmt.Errorf("parse YOLO dataset config %s: %w", path, err)
	}
	if len(document.Content) != 1 {
		return nil, fmt.Errorf("parse YOLO dataset config %s: expected one document", path)
	}
	names := findYAMLMappingValue(document.Content[0], "names")
	if names == nil {
		return nil, fmt.Errorf("parse YOLO dataset config %s: no names field", path)
	}

	classes := map[int]string{}
	switch names.Kind {
	case yaml.SequenceNode:
		for classID, item := range names.Content {
			item = dereferenceYAML(item)
			name, nameErr := validateClassName(item.Value)
			if nameErr != nil {
				return nil, fmt.Errorf("invalid YOLO class %d: %w", classID, nameErr)
			}
			classes[classID] = name
		}
	case yaml.MappingNode:
		for index := 0; index+1 < len(names.Content); index += 2 {
			keyNode := dereferenceYAML(names.Content[index])
			valueNode := dereferenceYAML(names.Content[index+1])
			var classID int
			if decodeErr := keyNode.Decode(&classID); decodeErr != nil || classID < 0 {
				return nil, fmt.Errorf(
					"parse YOLO dataset config %s: invalid class id %q",
					path,
					keyNode.Value,
				)
			}
			name, nameErr := validateClassName(valueNode.Value)
			if nameErr != nil {
				return nil, fmt.Errorf("invalid YOLO class %d: %w", classID, nameErr)
			}
			if _, duplicate := classes[classID]; duplicate {
				return nil, fmt.Errorf("duplicate YOLO class id: %d", classID)
			}
			classes[classID] = name
		}
	default:
		return nil, fmt.Errorf(
			"parse YOLO dataset config %s: names must be a sequence or mapping",
			path,
		)
	}
	if len(classes) == 0 {
		return nil, fmt.Errorf("parse YOLO dataset config %s: names must not be empty", path)
	}
	return classes, nil
}

func loadYOLO(labelPath string, configPath string) ([]Box, error) {
	classes, err := loadYAMLClasses(configPath)
	if err != nil {
		return nil, err
	}
	file, err := os.Open(labelPath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, fmt.Errorf("open YOLO label file: %w", err)
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
			return nil, fmt.Errorf("%s:%d: expected five fields for a YOLO box", labelPath, lineNumber)
		}
		classID, parseErr := strconv.Atoi(fields[0])
		if parseErr != nil || classID < 0 {
			return nil, fmt.Errorf("%s:%d: class id must be a non-negative integer", labelPath, lineNumber)
		}
		coordinates := make([]float64, 4)
		for index, field := range fields[1:] {
			coordinates[index], parseErr = strconv.ParseFloat(field, 64)
			if parseErr != nil {
				return nil, fmt.Errorf("%s:%d: box coordinates must be numbers", labelPath, lineNumber)
			}
		}
		centerX, centerY, width, height := coordinates[0], coordinates[1], coordinates[2], coordinates[3]
		for _, value := range coordinates {
			if math.IsNaN(value) || math.IsInf(value, 0) || value < 0 || value > 1 {
				return nil, fmt.Errorf("%s:%d: box coordinates must be between 0 and 1", labelPath, lineNumber)
			}
		}
		if width <= 0 || height <= 0 {
			return nil, fmt.Errorf("%s:%d: box width and height must be greater than zero", labelPath, lineNumber)
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
		return nil, fmt.Errorf("read YOLO label file: %w", err)
	}
	return boxes, nil
}
