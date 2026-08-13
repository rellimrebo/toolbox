from __future__ import annotations

from pathlib import Path

import av
from PIL import Image

from frame_labeler.media import ImageSource, open_media


def test_image_source_reads_one_rgb_frame(tmp_path: Path) -> None:
    source_path = tmp_path / "image.png"
    Image.new("RGB", (64, 48), "navy").save(source_path)

    source = open_media(source_path)
    frame = source.read_frame(0)

    assert isinstance(source, ImageSource)
    assert source.frame_count == 1
    assert (frame.index, frame.width, frame.height) == (0, 64, 48)
    assert frame.image.mode == "RGB"
    source.close()


def test_video_source_reads_exact_source_frame_indices(tmp_path: Path) -> None:
    source_path = tmp_path / "video.mp4"
    container = av.open(str(source_path), mode="w")
    stream = container.add_stream("libx264", rate=10)
    stream.width = 64
    stream.height = 48
    stream.pix_fmt = "yuv420p"
    for index in range(12):
        image = Image.new("RGB", (64, 48), (index * 20, 0, 0))
        frame = av.VideoFrame.from_image(image)
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()

    source = open_media(source_path)
    frames = [source.read_frame(index) for index in (0, 3, 6, 9, 3)]

    assert source.frame_count == 12
    assert [frame.index for frame in frames] == [0, 3, 6, 9, 3]
    assert all((frame.width, frame.height) == (64, 48) for frame in frames)
    source.close()
