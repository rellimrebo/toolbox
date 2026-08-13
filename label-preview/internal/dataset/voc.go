package dataset

import (
	"encoding/xml"
	"fmt"
	"os"
)

type vocDocument struct {
	Objects []vocObject `xml:"object"`
}

type vocObject struct {
	Name string `xml:"name"`
	Box  vocBox `xml:"bndbox"`
}

type vocBox struct {
	XMin float64 `xml:"xmin"`
	YMin float64 `xml:"ymin"`
	XMax float64 `xml:"xmax"`
	YMax float64 `xml:"ymax"`
}

func loadPascalVOC(annotationPath string, imageWidth int, imageHeight int) ([]Box, error) {
	file, err := os.Open(annotationPath)
	if err != nil {
		return nil, fmt.Errorf("open Pascal VOC annotation: %w", err)
	}
	defer file.Close()
	decoder := xml.NewDecoder(file)
	var document vocDocument
	if err := decoder.Decode(&document); err != nil {
		return nil, fmt.Errorf("parse Pascal VOC annotation %s: %w", annotationPath, err)
	}

	var boxes []Box
	for _, object := range document.Objects {
		name, nameErr := validateClassName(object.Name)
		if nameErr != nil {
			return nil, fmt.Errorf("invalid Pascal VOC class: %w", nameErr)
		}
		box, boxErr := normalizedBox(
			name,
			object.Box.XMin,
			object.Box.YMin,
			object.Box.XMax,
			object.Box.YMax,
			imageWidth,
			imageHeight,
		)
		if boxErr != nil {
			return nil, fmt.Errorf("invalid Pascal VOC bbox for %s: %w", name, boxErr)
		}
		boxes = append(boxes, box)
	}
	return boxes, nil
}
