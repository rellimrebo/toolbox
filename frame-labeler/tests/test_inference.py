from __future__ import annotations

import pytest
from PIL import Image

from frame_labeler.domain import AnnotationOrigin
from frame_labeler.inference import (
    OBJECT_DETECTION_OUTPUT,
    Detection,
    InferenceError,
    InferenceRequest,
    detections_to_boxes,
    run_inference,
)
from frame_labeler.media import MediaFrame


class _StubProvider:
    provider_id = "stub-detector/v1"
    output_type = OBJECT_DETECTION_OUTPUT

    def predict(self, request: InferenceRequest) -> tuple[Detection, ...]:
        assert request.frame_index == 7
        assert request.timestamp_seconds == 0.25
        assert request.image.size == (100, 80)
        assert request.class_names == ("dog", "person")
        return (
            Detection(0, -5.0, 10.0, 40.0, 60.0, confidence=0.9),
            Detection(1, 50.0, 20.0, 110.0, 70.0, confidence=0.75),
        )


def test_provider_detections_become_clipped_source_pixel_boxes() -> None:
    frame = MediaFrame(7, 0.25, Image.new("RGB", (100, 80), "white"))

    result = run_inference(_StubProvider(), frame, ("dog", "person"))
    boxes = detections_to_boxes(result, frame, ("dog", "person"))

    assert result.frame_index == 7
    assert result.output_type == OBJECT_DETECTION_OUTPUT
    assert [box.class_id for box in boxes] == [0, 1]
    assert [box.coordinates for box in boxes] == [
        (0.0, 10.0, 40.0, 60.0),
        (50.0, 20.0, 100.0, 70.0),
    ]
    assert all(box.origin is AnnotationOrigin.INFERRED for box in boxes)
    assert all(box.inference_provider == "stub-detector/v1" for box in boxes)
    assert [box.confidence for box in boxes] == [0.9, 0.75]
    assert len({box.id for box in boxes}) == 2
    frame.image.close()


def test_provider_cannot_return_an_unknown_project_class() -> None:
    class UnknownClassProvider:
        provider_id = "unknown-class/v1"
        output_type = OBJECT_DETECTION_OUTPUT

        def predict(self, _request: InferenceRequest) -> tuple[Detection, ...]:
            return (Detection(1, 10.0, 10.0, 20.0, 20.0),)

    frame = MediaFrame(0, 0.0, Image.new("RGB", (100, 80), "white"))
    result = run_inference(UnknownClassProvider(), frame, ("dog",))

    with pytest.raises(InferenceError, match="unknown class ID 1"):
        detections_to_boxes(result, frame, ("dog",))

    frame.image.close()


def test_provider_execution_is_not_specific_to_detection() -> None:
    class ClassificationProvider:
        provider_id = "stub-classifier/v1"
        output_type = "image-classification/v1"

        def predict(self, request: InferenceRequest) -> tuple[str, ...]:
            assert request.class_names == ("dog", "cat")
            return ("dog",)

    frame = MediaFrame(3, 0.1, Image.new("RGB", (100, 80), "white"))

    result = run_inference(ClassificationProvider(), frame, ("dog", "cat"))

    assert result.provider_id == "stub-classifier/v1"
    assert result.output_type == "image-classification/v1"
    assert result.frame_index == 3
    assert result.timestamp_seconds == 0.1
    assert result.predictions == ("dog",)
    frame.image.close()


def test_detection_materializer_rejects_another_output_type() -> None:
    class SegmentationProvider:
        provider_id = "stub-segmenter/v1"
        output_type = "instance-segmentation/v1"

        def predict(self, _request: InferenceRequest) -> tuple[Detection, ...]:
            return (Detection(0, 10.0, 10.0, 20.0, 20.0),)

    frame = MediaFrame(0, 0.0, Image.new("RGB", (100, 80), "white"))
    result = run_inference(SegmentationProvider(), frame, ("dog",))

    with pytest.raises(InferenceError, match="cannot be materialized as detection boxes"):
        detections_to_boxes(result, frame, ("dog",))

    frame.image.close()
