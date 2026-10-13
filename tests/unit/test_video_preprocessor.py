"""Pruebas del perfil configurable y la representación numérica de identidad."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from client.preprocessing import (
    VideoPreprocessingError,
    VideoPreprocessor,
    VideoReconstructionError,
    VideoReconstructor,
)
from client.preprocessing.video_preprocessor import VideoMetadata


@pytest.fixture(autouse=True)
def _clear_profile_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MPC_MAX_WIDTH", raising=False)
    monkeypatch.delenv("MPC_MAX_HEIGHT", raising=False)


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
        tmp_path / "tiny.mp4", width=64, height=48, fps=20, frames=20
    )

    result = VideoPreprocessor().process(source)

    assert result.metadata.frame_count == 10
    assert result.metadata.width == 64
    assert result.metadata.height == 48
    assert result.metadata.channels == 1
    assert result.metadata.fps == pytest.approx(10, abs=0.1)
    assert result.metadata.dtype == "uint8"
    assert result.metadata.version == 2
    assert len(result.payload) == 10 * 64 * 48
    frames = np.frombuffer(result.payload, dtype=np.uint8).reshape(10, 48, 64)
    assert [float(frame.mean()) for frame in frames] == sorted(
        float(frame.mean()) for frame in frames
    )


def test_downscales_but_does_not_upscale(tmp_path: Path) -> None:
    large = _make_video(
        tmp_path / "large.mp4", width=1280, height=960, fps=2, frames=2
    )
    small = _make_video(
        tmp_path / "small.mp4", width=64, height=48, fps=2, frames=2
    )

    large_result = VideoPreprocessor().process(large)
    small_result = VideoPreprocessor().process(small)

    assert (large_result.metadata.width, large_result.metadata.height) == (640, 480)
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
        tmp_path / "source.mp4", width=64, height=48, fps=10, frames=10
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
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 10
    assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 64
    assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 48
    assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(10, abs=0.25)
    capture.release()


def test_previous_resolution_and_fps_remain_configurable(tmp_path: Path) -> None:
    source = _make_video(
        tmp_path / "source.mp4", width=640, height=480, fps=10, frames=10
    )

    result = VideoPreprocessor(
        max_width=320, max_height=240, target_fps=5
    ).process(source)

    assert (result.metadata.width, result.metadata.height) == (320, 240)
    assert result.metadata.fps == pytest.approx(5)
    assert result.metadata.frame_count == 5
    assert len(result.payload) == 5 * 320 * 240


def test_environment_controls_maximum_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MPC_MAX_WIDTH", "480")
    monkeypatch.setenv("MPC_MAX_HEIGHT", "360")

    assert VideoPreprocessor()._scaled_size(1280, 960) == (480, 360)
    assert VideoPreprocessor(max_width=320, max_height=240)._scaled_size(
        1280, 960
    ) == (320, 240)


@pytest.mark.parametrize(
    ("source_size", "target_size"),
    [
        ((1920, 1080), (640, 360)),
        ((1080, 1920), (270, 480)),
        ((1280, 960), (640, 480)),
        ((368, 368), (368, 368)),
        ((64, 48), (64, 48)),
        ((641, 479), (640, 478)),
        ((65, 49), (64, 48)),
        ((1920, 500), (640, 166)),
    ],
)
def test_scaled_size_preserves_aspect_without_upscaling(
    source_size: tuple[int, int], target_size: tuple[int, int]
) -> None:
    result = VideoPreprocessor()._scaled_size(*source_size)

    assert result == target_size
    assert result[0] <= source_size[0] and result[1] <= source_size[1]
    assert result[0] % 2 == result[1] % 2 == 0
    # Las dimensiones pares pueden retirar menos de dos píxeles de cada eje.
    scale = min(1, 640 / source_size[0], 480 / source_size[1])
    assert 0 <= source_size[0] * scale - result[0] < 2
    assert 0 <= source_size[1] * scale - result[1] < 2


@pytest.mark.parametrize("source_size", [(1, 10), (10, 1)])
def test_rejects_frames_too_small_for_mp4(source_size: tuple[int, int]) -> None:
    with pytest.raises(VideoPreprocessingError, match="2x2"):
        VideoPreprocessor()._scaled_size(*source_size)


@pytest.mark.parametrize("value", [0, -1, 10.1, float("nan"), float("inf")])
def test_rejects_invalid_target_fps(value: float) -> None:
    with pytest.raises(ValueError, match="target_fps"):
        VideoPreprocessor(target_fps=value)


@pytest.mark.parametrize("value", [0, -1, 20.1, float("nan"), float("inf")])
def test_rejects_invalid_duration(value: float) -> None:
    with pytest.raises(ValueError, match="max_duration_seconds"):
        VideoPreprocessor(max_duration_seconds=value)


@pytest.mark.parametrize("options", [
    {"max_width": 641},
    {"max_height": 481},
    {"max_width": 1},
    {"max_height": 1},
    {"max_width": 320.5},
])
def test_rejects_invalid_resolution(options: dict[str, int | float]) -> None:
    with pytest.raises(ValueError):
        VideoPreprocessor(**options)


class _ArrayCapture:
    """Entrada sin pérdidas para comprobar el muestreo y el orden exacto."""

    def __init__(self, frames: list[np.ndarray], fps: float) -> None:
        self.frames = iter(frames)
        self.properties = {
            cv2.CAP_PROP_FRAME_COUNT: len(frames),
            cv2.CAP_PROP_FPS: fps,
            cv2.CAP_PROP_FRAME_WIDTH: frames[0].shape[1],
            cv2.CAP_PROP_FRAME_HEIGHT: frames[0].shape[0],
        }
        self.released = False

    def isOpened(self) -> bool:
        return True

    def get(self, property_id: int) -> float:
        return self.properties.get(property_id, 0)

    def read(self) -> tuple[bool, np.ndarray | None]:
        frame = next(self.frames, None)
        return frame is not None, frame

    def release(self) -> None:
        self.released = True


@pytest.mark.parametrize(
    ("source_fps", "frame_count", "expected_values", "expected_fps"),
    [
        (30, 30, list(range(0, 30, 3)), 10),
        (20, 20, list(range(0, 20, 2)), 10),
        (5, 5, list(range(5)), 5),
    ],
)
def test_sampling_preserves_exact_pixel_values_and_temporal_order(
    monkeypatch: pytest.MonkeyPatch,
    source_fps: float,
    frame_count: int,
    expected_values: list[int],
    expected_fps: float,
) -> None:
    capture = _ArrayCapture(
        [np.full((12, 16, 3), index, dtype=np.uint8) for index in range(frame_count)],
        source_fps,
    )
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)

    result = VideoPreprocessor().process("lossless-test.mp4")
    frames = np.frombuffer(result.payload, dtype=np.uint8).reshape(-1, 12, 16)

    assert result.metadata.frame_count == len(expected_values)
    assert result.metadata.fps == expected_fps
    assert [int(frame[0, 0]) for frame in frames] == expected_values
    assert all(np.all(frame == value) for frame, value in zip(frames, expected_values))
    assert capture.released


def test_accepts_exactly_twenty_seconds_and_two_hundred_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = _ArrayCapture(
        [np.full((2, 2, 3), index, dtype=np.uint8) for index in range(200)], 10
    )
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)

    result = VideoPreprocessor().process("twenty-seconds.mp4")

    assert result.metadata.frame_count == 200
    assert result.metadata.fps == 10
    assert len(result.payload) == 200 * 2 * 2


@pytest.mark.parametrize("fps", [0, float("nan"), float("inf")])
def test_rejects_invalid_source_fps(
    monkeypatch: pytest.MonkeyPatch, fps: float
) -> None:
    capture = _ArrayCapture([np.zeros((12, 16, 3), dtype=np.uint8)], fps)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)

    with pytest.raises(VideoPreprocessingError, match="FPS válido"):
        VideoPreprocessor().process("invalid-fps.mp4")
    assert capture.released


def _metadata() -> VideoMetadata:
    return VideoMetadata(
        frame_count=2, height=12, width=16, channels=1, fps=10, dtype="uint8"
    )


@pytest.mark.parametrize(
    "override",
    [
        {cv2.CAP_PROP_FRAME_WIDTH: 14},
        {cv2.CAP_PROP_FRAME_HEIGHT: 10},
        {cv2.CAP_PROP_FRAME_COUNT: 1},
        {cv2.CAP_PROP_FPS: 5},
        {cv2.CAP_PROP_FPS: float("nan")},
    ],
)
def test_reconstruction_validation_rejects_wrong_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, override: dict[int, float]
) -> None:
    output = tmp_path / "output.mp4"
    output.write_bytes(b"non-empty placeholder")
    capture = _ArrayCapture([np.zeros((12, 16, 3), dtype=np.uint8)] * 2, 10)
    capture.properties.update(override)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)

    with pytest.raises(VideoReconstructionError, match="dimensiones/frames/FPS"):
        VideoReconstructor._validate_output(output, _metadata())
    assert capture.released


def test_reconstruction_validation_checks_actual_decoded_frame_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output.mp4"
    output.write_bytes(b"non-empty placeholder")
    capture = _ArrayCapture([np.zeros((12, 16, 3), dtype=np.uint8)], 10)
    capture.properties[cv2.CAP_PROP_FRAME_COUNT] = 2
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)

    with pytest.raises(VideoReconstructionError, match="todos los frames"):
        VideoReconstructor._validate_output(output, _metadata())
    assert capture.released


def test_reconstruction_validation_checks_size_of_each_decoded_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output.mp4"
    output.write_bytes(b"non-empty placeholder")
    capture = _ArrayCapture(
        [np.zeros((12, 16, 3), dtype=np.uint8), np.zeros((10, 16, 3), dtype=np.uint8)],
        10,
    )
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)

    with pytest.raises(VideoReconstructionError, match="cambió de tamaño"):
        VideoReconstructor._validate_output(output, _metadata())
    assert capture.released


def test_reconstruction_rejects_incorrect_payload_size(tmp_path: Path) -> None:
    with pytest.raises(VideoReconstructionError, match="bytes"):
        VideoReconstructor().reconstruct(
            b"too short", _metadata(), "12345678-1234-5678-1234-567812345678", tmp_path
        )
