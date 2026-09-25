"""Pruebas de la representación numérica local del Sprint 1."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from client.preprocessing import (
    VideoPreprocessingError,
    VideoPreprocessor,
    VideoReconstructor,
)


def _make_video(
    path: Path,
    *,
    width: int,
    height: int,
    fps: float,
    frames: int,
) -> Path:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    assert writer.isOpened()
    for index in range(frames):
        value = index % 256
        writer.write(np.full((height, width, 3), value, dtype=np.uint8))
    writer.release()
    return path


def test_converts_to_raw_grayscale_and_reduces_fps(tmp_path: Path) -> None:
    source = _make_video(
        tmp_path / "tiny.mp4", width=64, height=48, fps=10, frames=10
    )

    result = VideoPreprocessor().process(source)

    assert result.metadata.frame_count == 5
    assert result.metadata.width == 64
    assert result.metadata.height == 48
    assert result.metadata.channels == 1
    assert result.metadata.fps == pytest.approx(5, abs=0.1)
    assert result.metadata.dtype == "uint8"
    assert result.metadata.version == 2
    assert len(result.payload) == 5 * 64 * 48
    frames = np.frombuffer(result.payload, dtype=np.uint8).reshape(5, 48, 64)
    assert [float(frame.mean()) for frame in frames] == sorted(
        float(frame.mean()) for frame in frames
    )


def test_downscales_but_does_not_upscale(tmp_path: Path) -> None:
    large = _make_video(
        tmp_path / "large.mp4", width=640, height=480, fps=2, frames=2
    )
    small = _make_video(
        tmp_path / "small.mp4", width=64, height=48, fps=2, frames=2
    )

    large_result = VideoPreprocessor().process(large)
    small_result = VideoPreprocessor().process(small)

    assert (large_result.metadata.width, large_result.metadata.height) == (320, 240)
    assert (small_result.metadata.width, small_result.metadata.height) == (64, 48)


def test_rejects_video_longer_than_twenty_seconds(tmp_path: Path) -> None:
    source = _make_video(
        tmp_path / "long.mp4", width=16, height=16, fps=2, frames=42
    )

    with pytest.raises(VideoPreprocessingError, match="duración máxima"):
        VideoPreprocessor().process(source)


@pytest.mark.parametrize("content", [b"", b"not-an-mp4"])
def test_rejects_invalid_mp4(tmp_path: Path, content: bytes) -> None:
    source = tmp_path / "invalid.mp4"
    source.write_bytes(content)
    with pytest.raises(VideoPreprocessingError):
        VideoPreprocessor().process(source)


def test_reconstructs_a_playable_mp4(tmp_path: Path) -> None:
    source = _make_video(
        tmp_path / "source.mp4", width=64, height=48, fps=2, frames=4
    )
    processed = VideoPreprocessor().process(source)

    destination = VideoReconstructor().reconstruct(
        processed.payload,
        processed.metadata,
        "12345678-1234-5678-1234-567812345678",
        tmp_path / "output",
    )

    assert destination.is_file()
    capture = cv2.VideoCapture(str(destination))
    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 4
    assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 64
    assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 48
    capture.release()
