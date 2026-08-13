from __future__ import annotations

import pytest
from PIL import Image

from frame_labeler.domain import BoxOrigin
from frame_labeler.inference import Detection, InferenceError, InferenceRequest, suggest_boxes
from frame_labeler.media import MediaFrame


class _StubProvider:
    provider_id = "stub-detector/v1"

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

    boxes = suggest_boxes(_StubProvider(), frame, ("dog", "person"))

    assert [box.class_id for box in boxes] == [0, 1]
    assert [box.coordinates for box in boxes] == [
        (0.0, 10.0, 40.0, 60.0),
        (50.0, 20.0, 100.0, 70.0),
    ]
    assert all(box.origin is BoxOrigin.INFERRED for box in boxes)
    assert all(box.inference_provider == "stub-detector/v1" for box in boxes)
    assert [box.confidence for box in boxes] == [0.9, 0.75]
    assert len({box.id for box in boxes}) == 2
    frame.image.close()


def test_provider_cannot_return_an_unknown_project_class() -> None:
    class UnknownClassProvider:
        provider_id = "unknown-class/v1"

        def predict(self, _request: InferenceRequest) -> tuple[Detection, ...]:
            return (Detection(1, 10.0, 10.0, 20.0, 20.0),)

    frame = MediaFrame(0, 0.0, Image.new("RGB", (100, 80), "white"))

    with pytest.raises(InferenceError, match="unknown class ID 1"):
        suggest_boxes(UnknownClassProvider(), frame, ("dog",))

    frame.image.close()
