from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Protocol

import av
from av.container import InputContainer
from av.video.stream import VideoStream
from PIL import Image, ImageOps


class MediaError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class MediaFrame:
    index: int
    timestamp_seconds: float
    image: Image.Image

    @property
    def width(self) -> int:
        return self.image.width

    @property
    def height(self) -> int:
        return self.image.height


class MediaSource(Protocol):
    path: Path
    frame_count: int

    def read_frame(self, index: int) -> MediaFrame: ...

    def close(self) -> None: ...


class ImageSource:
    frame_count = 1

    def __init__(self, path: Path) -> None:
        self.path = path
        try:
            with Image.open(path) as image:
                self._image = ImageOps.exif_transpose(image).convert("RGB").copy()
        except Exception as error:
            raise MediaError(f"Unable to read image: {path}") from error

    def read_frame(self, index: int) -> MediaFrame:
        if index != 0:
            raise IndexError("An image contains only frame 0")
        return MediaFrame(0, 0.0, self._image.copy())

    def close(self) -> None:
        self._image.close()


class VideoSource:
    def __init__(self, path: Path) -> None:
        self.path = path
        try:
            self._container: InputContainer = av.open(str(path))
            self._stream: VideoStream = self._container.streams.video[0]
        except Exception as error:
            raise MediaError(f"Unable to read video: {path}") from error
        rate = self._stream.average_rate or self._stream.base_rate
        time_base = self._stream.time_base
        if rate is None or rate <= 0 or time_base is None:
            self.close()
            raise MediaError("Video does not provide usable timing metadata")
        self._rate: Fraction = rate
        self._time_base: Fraction = time_base
        self.frame_count = self._calculate_frame_count()
        self._decoder = iter(self._container.decode(self._stream))
        self._next_sequential_index = 0

    def _calculate_frame_count(self) -> int:
        if self._stream.frames:
            return int(self._stream.frames)
        if self._stream.duration is not None:
            seconds = float(self._stream.duration * self._time_base)
        elif self._container.duration is not None:
            seconds = self._container.duration / av.time_base
        else:
            raise MediaError("Video does not provide a usable duration")
        return max(1, round(seconds * float(self._rate)))

    def _seek(self, index: int) -> None:
        seconds = Fraction(index, 1) / self._rate
        offset = int(seconds / self._time_base)
        self._container.seek(max(0, offset), stream=self._stream, backward=True, any_frame=False)
        self._decoder = iter(self._container.decode(self._stream))
        self._next_sequential_index = -1

    def read_frame(self, index: int) -> MediaFrame:
        if index < 0 or index >= self.frame_count:
            raise IndexError(f"Frame index is outside the video: {index}")
        if index < self._next_sequential_index or index - self._next_sequential_index > 120:
            self._seek(index)

        for decoded in self._decoder:
            if self._next_sequential_index < 0:
                if decoded.time is None:
                    self._next_sequential_index = 0
                else:
                    self._next_sequential_index = max(
                        0, round(float(decoded.time) * float(self._rate))
                    )
            decoded_index = self._next_sequential_index
            self._next_sequential_index += 1
            if decoded_index < index:
                continue
            if decoded_index > index:
                raise MediaError(f"Decoder skipped requested frame {index}")
            timestamp = (
                float(decoded.time) if decoded.time is not None else index / float(self._rate)
            )
            image = decoded.to_image().convert("RGB")  # type: ignore[no-untyped-call]
            return MediaFrame(index, timestamp, image)
        raise MediaError(f"Unable to decode frame {index}")

    def close(self) -> None:
        self._container.close()


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def open_media(path: str | Path) -> MediaSource:
    resolved = Path(path).expanduser().resolve(strict=True)
    if resolved.suffix.lower() in _IMAGE_SUFFIXES:
        return ImageSource(resolved)
    return VideoSource(resolved)
