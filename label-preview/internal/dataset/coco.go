package dataset

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
)

type cocoImage struct {
	ID       int    `json:"id"`
	FileName string `json:"file_name"`
}

type cocoAnnotation struct {
	ImageID    int       `json:"image_id"`
	CategoryID int       `json:"category_id"`
	BBox       []float64 `json:"bbox"`
}

type cocoCategory struct {
	ID   int    `json:"id"`
	Name string `json:"name"`
}

func skipJSONValue(decoder *json.Decoder) error {
	token, err := decoder.Token()
	if err != nil {
		return err
	}
	delimiter, structured := token.(json.Delim)
	if !structured {
		return nil
	}
	for decoder.More() {
		if delimiter == '{' {
			if _, err := decoder.Token(); err != nil {
				return err
			}
		}
		if err := skipJSONValue(decoder); err != nil {
			return err
		}
	}
	closing, err := decoder.Token()
	if err != nil {
		return err
	}
	expected := json.Delim(']')
	if delimiter == '{' {
		expected = json.Delim('}')
	}
	if closing != expected {
		return errors.New("mismatched JSON delimiter")
	}
	return nil
}

func decodeJSONArray[T any](decoder *json.Decoder, visit func(T) error) error {
	token, err := decoder.Token()
	if err != nil {
		return err
	}
	if token != json.Delim('[') {
		return errors.New("expected an array")
	}
	for decoder.More() {
		var value T
		if err := decoder.Decode(&value); err != nil {
			return err
		}
		if err := visit(value); err != nil {
			return err
		}
	}
	closing, err := decoder.Token()
	if err != nil {
		return err
	}
	if closing != json.Delim(']') {
		return errors.New("expected end of array")
	}
	return nil
}

func visitCOCODocument(
	path string,
	visitors map[string]func(*json.Decoder) error,
) error {
	file, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("open COCO annotations: %w", err)
	}
	defer file.Close()
	decoder := json.NewDecoder(file)
	opening, err := decoder.Token()
	if err != nil {
		return err
	}
	if opening != json.Delim('{') {
		return errors.New("COCO document must be a JSON object")
	}
	for decoder.More() {
		keyToken, tokenErr := decoder.Token()
		if tokenErr != nil {
			return tokenErr
		}
		key, ok := keyToken.(string)
		if !ok {
			return errors.New("COCO document keys must be strings")
		}
		if visit, known := visitors[key]; known {
			if err := visit(decoder); err != nil {
				return fmt.Errorf("parse COCO %s: %w", key, err)
			}
		} else if err := skipJSONValue(decoder); err != nil {
			return err
		}
	}
	closing, err := decoder.Token()
	if err != nil {
		return err
	}
	if closing != json.Delim('}') {
		return errors.New("expected end of COCO document")
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		if err == nil {
			return errors.New("unexpected data after COCO document")
		}
		return err
	}
	return nil
}

func loadCOCO(
	annotationPath string,
	datasetRoot string,
	imagePath string,
	imageWidth int,
	imageHeight int,
) ([]Box, error) {
	relativeImage, err := filepath.Rel(datasetRoot, imagePath)
	if err != nil {
		return nil, fmt.Errorf("resolve COCO image path: %w", err)
	}
	relativeImage = filepath.ToSlash(relativeImage)
	baseName := filepath.Base(imagePath)
	categories := map[int]string{}
	var targetAnnotations []cocoAnnotation
	var imageID int
	imageFound := false
	annotationsDeferred := false

	visitors := map[string]func(*json.Decoder) error{}
	visitors["images"] = func(decoder *json.Decoder) error {
		var exactMatches []cocoImage
		var baseMatches []cocoImage
		decodeErr := decodeJSONArray(decoder, func(image cocoImage) error {
			fileName := filepath.ToSlash(filepath.Clean(image.FileName))
			if fileName == relativeImage {
				exactMatches = append(exactMatches, image)
			}
			if filepath.Base(filepath.FromSlash(fileName)) == baseName {
				baseMatches = append(baseMatches, image)
			}
			return nil
		})
		if decodeErr != nil {
			return decodeErr
		}
		matches := exactMatches
		if len(matches) == 0 {
			matches = baseMatches
		}
		if len(matches) == 0 {
			return fmt.Errorf("annotations do not contain image %s", relativeImage)
		}
		if len(matches) > 1 {
			return fmt.Errorf("annotations contain multiple images named %s", baseName)
		}
		imageID = matches[0].ID
		imageFound = true
		return nil
	}
	visitors["categories"] = func(decoder *json.Decoder) error {
		return decodeJSONArray(decoder, func(category cocoCategory) error {
			if category.ID < 0 {
				return fmt.Errorf("category id must be non-negative: %d", category.ID)
			}
			name, nameErr := validateClassName(category.Name)
			if nameErr != nil {
				return fmt.Errorf("invalid category %d: %w", category.ID, nameErr)
			}
			categories[category.ID] = name
			return nil
		})
	}
	visitors["annotations"] = func(decoder *json.Decoder) error {
		if !imageFound {
			annotationsDeferred = true
			return skipJSONValue(decoder)
		}
		return decodeJSONArray(decoder, func(annotation cocoAnnotation) error {
			if annotation.ImageID == imageID {
				targetAnnotations = append(targetAnnotations, annotation)
			}
			return nil
		})
	}
	if err := visitCOCODocument(annotationPath, visitors); err != nil {
		return nil, fmt.Errorf("parse COCO annotations %s: %w", annotationPath, err)
	}
	if !imageFound {
		return nil, fmt.Errorf("COCO annotations do not identify image %s", imagePath)
	}
	if annotationsDeferred {
		targetAnnotations = nil
		err := visitCOCODocument(
			annotationPath,
			map[string]func(*json.Decoder) error{
				"annotations": func(decoder *json.Decoder) error {
					return decodeJSONArray(decoder, func(annotation cocoAnnotation) error {
						if annotation.ImageID == imageID {
							targetAnnotations = append(targetAnnotations, annotation)
						}
						return nil
					})
				},
			},
		)
		if err != nil {
			return nil, fmt.Errorf("parse COCO annotations %s: %w", annotationPath, err)
		}
	}

	boxes := make([]Box, 0, len(targetAnnotations))
	for _, annotation := range targetAnnotations {
		if annotation.CategoryID < 0 {
			return nil, fmt.Errorf(
				"COCO category id must be non-negative for image %s",
				relativeImage,
			)
		}
		if len(annotation.BBox) != 4 {
			return nil, fmt.Errorf("COCO bbox must contain four values for image %s", relativeImage)
		}
		if annotation.BBox[2] <= 0 || annotation.BBox[3] <= 0 {
			return nil, fmt.Errorf("COCO bbox must have positive width and height for image %s", relativeImage)
		}
		label, known := categories[annotation.CategoryID]
		if !known {
			label = strconv.Itoa(annotation.CategoryID)
		}
		box, boxErr := normalizedBox(
			label,
			annotation.BBox[0],
			annotation.BBox[1],
			annotation.BBox[0]+annotation.BBox[2],
			annotation.BBox[1]+annotation.BBox[3],
			imageWidth,
			imageHeight,
		)
		if boxErr != nil {
			return nil, fmt.Errorf("invalid COCO bbox for image %s: %w", relativeImage, boxErr)
		}
		boxes = append(boxes, box)
	}
	return boxes, nil
}
