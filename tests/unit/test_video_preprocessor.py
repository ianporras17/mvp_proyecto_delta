"""Pruebas de la representación numérica local del Sprint 1."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from client.preprocessing import (
    FaceRegion,
    VideoPreprocessingError,
    VideoPreprocessor,
    VideoReconstructor,
)
from client.preprocessing.video_preprocessor import (
    PreprocessedVideo,
    VideoMetadata,
    scale_bounding_box,
)


class FakeYuNet:
    def __init__(self) -> None:
        self.input_sizes: list[tuple[int, int]] = []
        self.frames: list[np.ndarray] = []

    def setInputSize(self, size: list[int]) -> None:
        self.input_sizes.append(tuple(size))

    def infer(self, image: np.ndarray) -> np.ndarray:
        self.frames.append(image.copy())
        return np.empty((0, 15), dtype=np.float32)


@pytest.fixture(autouse=True)
def local_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MPC_MAX_WIDTH", raising=False)
    monkeypatch.delenv("MPC_MAX_HEIGHT", raising=False)


def _make_video(
    path: Path,
    *,
    width: int,
    height: int,
    fps: float,
    frames: int,
    color: bool = False,
) -> Path:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    assert writer.isOpened()
    for index in range(frames):
        if color:
            image = np.empty((height, width, 3), dtype=np.uint8)
            image[:] = (10 + index * 5, 60 + index * 5, 120 + index * 5)
        else:
            image = np.full((height, width, 3), index % 256, dtype=np.uint8)
        writer.write(image)
    writer.release()
    return path


def test_converts_to_raw_grayscale_and_reduces_fps(tmp_path: Path) -> None:
    source = _make_video(
        tmp_path / "tiny.mp4", width=64, height=48, fps=20, frames=20
    )

    result = VideoPreprocessor(target_fps=10).process(source, FakeYuNet())

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


@pytest.mark.parametrize(
    ("source_fps", "source_frames", "target_fps", "expected_fps", "selected_indices"),
    [
        (20, 20, None, 20, list(range(20))),
        (30, 30, None, 30, list(range(30))),
        (60, 60, None, 30, list(range(0, 60, 2))),
        (20, 20, 5, 5, list(range(0, 20, 4))),
        (10, 10, None, 10, list(range(10))),
        (6, 12, None, 6, list(range(12))),
        (7.5, 15, None, 7.5, list(range(15))),
        (2, 4, None, 2, list(range(4))),
    ],
)
def test_fps_sampling_preserves_existing_frames_and_temporal_order(
    tmp_path: Path,
    source_fps: float,
    source_frames: int,
    target_fps: float | None,
    expected_fps: float,
    selected_indices: list[int],
) -> None:
    source = _make_video(
        tmp_path / "sampling.mp4", width=16, height=16,
        fps=source_fps, frames=source_frames,
    )
    capture = cv2.VideoCapture(str(source))
    originals = []
    while True:
        success, frame = capture.read()
        if not success:
            break
        originals.append(frame)
    capture.release()
    options = {} if target_fps is None else {"target_fps": target_fps}
    model = FakeYuNet()
    result = VideoPreprocessor(**options).process(source, model)

    assert result.metadata.fps == pytest.approx(expected_fps, abs=0.01)
    assert result.metadata.frame_count == len(selected_indices)
    assert result.metadata.frame_count <= source_frames
    assert result.metadata.frame_count / result.metadata.fps == pytest.approx(
        source_frames / source_fps, abs=0.01
    )
    selected = [originals[index] for index in selected_indices]
    np.testing.assert_array_equal(np.stack(model.frames), np.stack(selected))
    expected = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in selected])
    assert result.payload == expected.tobytes()


def test_downscales_but_does_not_upscale(tmp_path: Path) -> None:
    large = _make_video(
        tmp_path / "large.mp4", width=640, height=480, fps=2, frames=2
    )
    small = _make_video(
        tmp_path / "small.mp4", width=64, height=48, fps=2, frames=2
    )

    large_result = VideoPreprocessor(max_width=320).process(large, FakeYuNet())
    small_result = VideoPreprocessor(max_width=320).process(small, FakeYuNet())

    assert (large_result.metadata.width, large_result.metadata.height) == (320, 240)
    assert (small_result.metadata.width, small_result.metadata.height) == (64, 48)


def test_default_resolution_box_is_full_hd_without_upscaling() -> None:
    assert VideoPreprocessor()._scaled_size(3840, 2160) == (1920, 1080)
    assert VideoPreprocessor()._scaled_size(1280, 720) == (1280, 720)


def test_accepts_video_longer_than_twenty_seconds_and_optional_local_cap(tmp_path: Path) -> None:
    source = _make_video(
        tmp_path / "long.mp4", width=16, height=16, fps=2, frames=42
    )

    assert VideoPreprocessor().process(source, FakeYuNet()).metadata.frame_count == 42
    with pytest.raises(VideoPreprocessingError, match="duración máxima"):
        VideoPreprocessor(max_duration_seconds=20).process(source, FakeYuNet())


@pytest.mark.parametrize("content", [b"", b"not-an-mp4"])
def test_rejects_invalid_mp4(tmp_path: Path, content: bytes) -> None:
    source = tmp_path / "invalid.mp4"
    source.write_bytes(content)
    with pytest.raises(VideoPreprocessingError):
        VideoPreprocessor().process(source, FakeYuNet())


@pytest.mark.parametrize(
    ("source_size", "max_width", "expected_size", "source_fps", "source_frames",
     "target_fps", "expected_fps", "expected_frames"),
    [
        ((64, 48), 320, (64, 48), 2, 4, None, 2, 4),
        ((642, 482), 480, (478, 360), 2, 4, None, 2, 4),
        ((640, 480), 640, (640, 480), 2, 4, None, 2, 4),
        ((64, 48), 640, (64, 48), 20, 20, None, 20, 20),
        ((64, 48), 640, (64, 48), 20, 20, 5, 5, 5),
    ],
)
def test_reconstructs_a_playable_mp4(
    tmp_path: Path,
    source_size: tuple[int, int],
    max_width: int,
    expected_size: tuple[int, int],
    source_fps: float,
    source_frames: int,
    target_fps: float | None,
    expected_fps: float,
    expected_frames: int,
) -> None:
    source = _make_video(
        tmp_path / "source.mp4", width=source_size[0], height=source_size[1],
        fps=source_fps, frames=source_frames,
    )
    options = {} if target_fps is None else {"target_fps": target_fps}
    processed = VideoPreprocessor(max_width=max_width, **options).process(source, FakeYuNet())
    assert (processed.metadata.width, processed.metadata.height) == expected_size
    assert processed.metadata.frame_count == expected_frames
    assert processed.metadata.fps == pytest.approx(expected_fps, abs=0.01)
    assert len(processed.payload) == expected_frames * expected_size[0] * expected_size[1]

    destination = VideoReconstructor().reconstruct(
        processed.payload,
        processed.metadata,
        "12345678-1234-5678-1234-567812345678",
        tmp_path / "output",
    )

    assert destination.is_file()
    capture = cv2.VideoCapture(str(destination))
    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == expected_frames
    assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == expected_size[0]
    assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == expected_size[1]
    assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(expected_fps, abs=0.1)
    assert capture.get(cv2.CAP_PROP_FRAME_COUNT) / capture.get(cv2.CAP_PROP_FPS) == pytest.approx(
        source_frames / source_fps, abs=0.05
    )
    decoded = []
    while True:
        success, frame = capture.read()
        if not success:
            break
        decoded.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    capture.release()
    expected = np.frombuffer(processed.payload, dtype=np.uint8).reshape(
        expected_frames, expected_size[1], expected_size[0]
    )
    error = np.abs(expected.astype(np.int16) - np.stack(decoded).astype(np.int16))
    assert error.mean() < 5


@pytest.mark.parametrize("max_width", [320, 480, 640])
def test_configured_width_derives_a_four_by_three_height(
    tmp_path: Path, max_width: int
) -> None:
    source = _make_video(
        tmp_path / "large.mp4", width=640, height=480, fps=2, frames=2
    )
    processed = VideoPreprocessor(max_width=max_width).process(source, FakeYuNet())
    assert (processed.metadata.width, processed.metadata.height) == (
        max_width, max_width * 3 // 4
    )


def test_environment_profile_and_explicit_dimensions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _make_video(
        tmp_path / "large.mp4", width=640, height=480, fps=2, frames=2
    )
    monkeypatch.setenv("MPC_MAX_WIDTH", "480")
    from_environment = VideoPreprocessor().process(source, FakeYuNet())
    assert (from_environment.metadata.width, from_environment.metadata.height) == (480, 360)

    monkeypatch.setenv("MPC_MAX_HEIGHT", "240")
    height_limited = VideoPreprocessor().process(source, FakeYuNet())
    assert (height_limited.metadata.width, height_limited.metadata.height) == (320, 240)

    explicit = VideoPreprocessor(max_width=640, max_height=480).process(source, FakeYuNet())
    assert (explicit.metadata.width, explicit.metadata.height) == (640, 480)


def test_widescreen_aspect_ratio_is_preserved_without_upscaling(tmp_path: Path) -> None:
    source = _make_video(
        tmp_path / "wide.mp4", width=640, height=360, fps=2, frames=2
    )
    reduced = VideoPreprocessor(max_width=480).process(source, FakeYuNet())
    assert (reduced.metadata.width, reduced.metadata.height) == (480, 270)
    small = _make_video(
        tmp_path / "small.mp4", width=64, height=48, fps=2, frames=2
    )
    unchanged = VideoPreprocessor(max_width=640).process(small, FakeYuNet())
    assert (unchanged.metadata.width, unchanged.metadata.height) == (64, 48)


@pytest.mark.parametrize(
    "options",
    [
        {"max_width": 0}, {"max_height": 0}, {"max_width": 1},
        {"max_height": 1}, {"max_width": 1921},
        {"max_height": 1081}, {"max_duration_seconds": 0},
        {"max_duration_seconds": float("inf")}, {"target_fps": 0}, {"target_fps": 31},
        {"target_fps": float("nan")}, {"target_fps": float("inf")},
    ],
)
def test_rejects_profiles_outside_the_temporal_and_resolution_limits(options: dict) -> None:
    with pytest.raises(ValueError):
        VideoPreprocessor(**options)


def test_rejects_a_resize_too_small_for_the_mp4_codec() -> None:
    with pytest.raises(VideoPreprocessingError):
        VideoPreprocessor(max_width=2, max_height=2)._scaled_size(32, 2)


@pytest.mark.parametrize("source_size", [(606, 340), (616, 346)])
def test_floating_point_roundoff_does_not_remove_two_extra_pixels(
    source_size: tuple[int, int],
) -> None:
    assert VideoPreprocessor(max_width=320)._scaled_size(*source_size) == (320, 178)


def test_blank_height_environment_uses_the_derived_height(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MPC_MAX_WIDTH", "480")
    monkeypatch.setenv("MPC_MAX_HEIGHT", "")
    assert VideoPreprocessor()._scaled_size(640, 480) == (480, 360)


@pytest.mark.parametrize(
    ("target_fps", "expected_frames", "expected_fps"), [(None, 200, 10), (5, 100, 5)]
)
def test_accepts_twenty_seconds_and_at_most_two_hundred_sampled_frames(
    tmp_path: Path, target_fps: float | None, expected_frames: int, expected_fps: float
) -> None:
    source = _make_video(
        tmp_path / "boundary.mp4", width=16, height=16, fps=10, frames=200
    )
    model = FakeYuNet()
    options = {} if target_fps is None else {"target_fps": target_fps}
    result = VideoPreprocessor(**options).process(source, model)
    assert result.metadata.frame_count == expected_frames
    assert result.metadata.fps == expected_fps
    assert result.metadata.frame_count / result.metadata.fps == 20
    assert len(model.frames) == expected_frames


@pytest.mark.parametrize(
    ("target_fps", "sample_stride", "expected_frames"), [(None, 1, 20), (5, 4, 5)]
)
def test_yunet_uses_original_bgr_before_grayscale_and_resize_in_temporal_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    target_fps: float | None, sample_stride: int, expected_frames: int,
) -> None:
    source = _make_video(
        tmp_path / "color.mp4", width=640, height=480, fps=20, frames=20, color=True
    )
    capture = cv2.VideoCapture(str(source))
    originals = []
    while True:
        success, frame = capture.read()
        if not success:
            break
        originals.append(frame)
    capture.release()
    selected = originals[::sample_stride]
    expected = np.stack([
        cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (480, 360),
                   interpolation=cv2.INTER_AREA)
        for frame in selected
    ])
    events = []

    class RecordingYuNet(FakeYuNet):
        def infer(self, image: np.ndarray) -> np.ndarray:
            events.append("infer")
            return super().infer(image)

    original_grayscale = VideoPreprocessor._to_grayscale
    original_resize = cv2.resize

    def grayscale(frame: np.ndarray) -> np.ndarray:
        events.append("grayscale")
        return original_grayscale(frame)

    def resize(frame: np.ndarray, size: tuple[int, int], **kwargs) -> np.ndarray:
        events.append("resize")
        assert frame.ndim == 2
        return original_resize(frame, size, **kwargs)

    monkeypatch.setattr(VideoPreprocessor, "_to_grayscale", staticmethod(grayscale))
    monkeypatch.setattr(cv2, "resize", resize)
    model = RecordingYuNet()
    options = {} if target_fps is None else {"target_fps": target_fps}
    result = VideoPreprocessor(max_width=480, **options).process(source, model)

    assert result.metadata.frame_count == expected_frames
    assert result.metadata.fps == (20 if target_fps is None else target_fps)
    assert events == ["infer", "grayscale", "resize"] * expected_frames
    assert model.input_sizes == [(640, 480)] * expected_frames
    np.testing.assert_array_equal(np.stack(model.frames), np.stack(selected))
    assert all(frame.shape == (480, 640, 3) for frame in model.frames)
    assert all(not np.array_equal(frame[:, :, 0], frame[:, :, 2]) for frame in model.frames)
    assert result.payload == expected.tobytes()
    assert [frame.mean() for frame in model.frames] == sorted(frame.mean() for frame in model.frames)
    assert set(result.metadata.as_dict()) == {
        "frame_count", "height", "width", "channels", "fps", "dtype", "version"
    }
    assert result.regions == ()


def test_face_regions_are_scaled_to_processed_frames_and_no_debug_images_written(
    tmp_path: Path,
) -> None:
    source = _make_video(
        tmp_path / "faces.mp4", width=640, height=480, fps=20, frames=6, color=True
    )

    class DetectionYuNet(FakeYuNet):
        def infer(self, image: np.ndarray) -> np.ndarray:
            self.frames.append(image.copy())
            return np.array([[20.5, 20.5, 39.0, 39.0, *([0.0] * 10), 0.91]], dtype=np.float32)

    result = VideoPreprocessor(max_width=480, target_fps=5).process(
        source, DetectionYuNet()
    )

    assert result.metadata.frame_count == 2
    assert [(region.frame_id, region.x, region.y, region.width, region.height)
            for region in result.regions] == [(0, 15, 15, 30, 30), (1, 15, 15, 30, 30)]
    assert [region.confidence for region in result.regions] == pytest.approx([0.91, 0.91])
    assert [region.frame_id for region in result.regions] == [0, 1]
    assert len(result.payload) == result.metadata.frame_count * result.metadata.width * result.metadata.height
    assert set(result.metadata.as_dict()) == {
        "frame_count", "height", "width", "channels", "fps", "dtype", "version"
    }
    assert not (Path("results").exists())


def test_face_region_validates_required_geometry() -> None:
    with pytest.raises(ValueError):
        FaceRegion(0, -1, 0, 1, 1, 0.9)
    with pytest.raises(ValueError):
        FaceRegion(0, 0, 0, 0, 1, 0.9)


def test_only_face_detections_above_point_six_become_mask_regions() -> None:
    detections = np.array([
        [0, 0, 2, 2, *([0.0] * 10), 0.59],
        [2, 0, 2, 2, *([0.0] * 10), 0.60],
        [4, 0, 2, 2, *([0.0] * 10), 0.61],
    ], dtype=np.float32)
    regions = VideoPreprocessor._face_regions(detections, 0, (8, 4), (8, 4))
    assert [(region.x, region.confidence) for region in regions] == [(4, pytest.approx(0.61))]


def test_face_threshold_distinguishes_adjacent_fp32_scores() -> None:
    threshold = np.float32(0.6)
    scores = (
        np.nextafter(threshold, np.float32(0)),
        threshold,
        np.nextafter(threshold, np.float32(1)),
    )
    detections = np.array([
        [2 * index, 0, 2, 2, *([0.0] * 10), score]
        for index, score in enumerate(scores)
    ], dtype=np.float32)
    regions = VideoPreprocessor._face_regions(detections, 0, (8, 4), (8, 4))
    assert [(region.x, region.confidence) for region in regions] == [(4, float(scores[2]))]


def test_empty_face_detections_keep_every_sampled_pixel_frame(tmp_path: Path) -> None:
    source = _make_video(tmp_path / "empty.mp4", width=32, height=24, fps=2, frames=2)
    result = VideoPreprocessor().process(source, FakeYuNet())
    assert result.metadata.frame_count == 2
    assert result.regions == ()
    assert len(result.payload) == 2 * 32 * 24


def test_scales_bounding_boxes_with_actual_dimensions_without_clipping() -> None:
    box = (-10.0, 20.0, 100.0, 80.0)
    assert scale_bounding_box(box, (642, 482), (478, 360)) == pytest.approx((
        -10 * 478 / 642, 20 * 360 / 482, 100 * 478 / 642, 80 * 360 / 482
    ))
    assert scale_bounding_box((0, 0, 0, 0), (640, 480), (320, 240)) == (0, 0, 0, 0)


@pytest.mark.parametrize(
    ("box", "source_size", "target_size"),
    [
        ((0, 0, -1, 2), (640, 480), (320, 240)),
        ((0, 0, 2, -1), (640, 480), (320, 240)),
        ((0, 0, 2, 2), (0, 480), (320, 240)),
        ((0, 0, 2, 2), (640, 480), (320, 0)),
        ((float("nan"), 0, 2, 2), (640, 480), (320, 240)),
        ((0, 0, 2, 2), (640, float("inf")), (320, 240)),
    ],
)
def test_bounding_box_scaling_rejects_invalid_geometry(
    box: tuple, source_size: tuple, target_size: tuple
) -> None:
    with pytest.raises(ValueError):
        scale_bounding_box(box, source_size, target_size)


@pytest.mark.parametrize(
    ("source_fps", "target_fps", "selected_indices"),
    [
        (30, 10, [0, 3, 6, 9, 12]),
        (30, 5, [0, 6, 12]),
        (6, 10, list(range(15))),
    ],
)
def test_sample_hook_matches_yunet_originals_and_processed_frame_indices(
    tmp_path: Path,
    source_fps: float,
    target_fps: float,
    selected_indices: list[int],
) -> None:
    source = _make_video(
        tmp_path / "plates.mp4", width=320, height=240,
        fps=source_fps, frames=15, color=True,
    )
    capture = cv2.VideoCapture(str(source))
    originals = []
    while True:
        success, frame = capture.read()
        if not success:
            break
        originals.append(frame)
    capture.release()
    samples = []

    def on_sample(frame: np.ndarray, frame_index: int) -> None:
        assert frame.shape == (240, 320, 3)
        assert frame.dtype == np.uint8
        assert not np.array_equal(frame[:, :, 0], frame[:, :, 2])
        samples.append((frame_index, frame.copy()))

    model = FakeYuNet()
    processed = VideoPreprocessor(max_width=160, target_fps=target_fps).process(
        source, model, on_sample=on_sample,
    )

    selected = [originals[index] for index in selected_indices]
    assert processed.source_size == (320, 240)
    assert (processed.metadata.width, processed.metadata.height) == (160, 120)
    assert processed.metadata.frame_count == len(selected_indices)
    assert processed.metadata.fps == min(source_fps, target_fps)
    assert [index for index, _ in samples] == list(range(len(selected_indices)))
    np.testing.assert_array_equal(np.stack([frame for _, frame in samples]), np.stack(selected))
    np.testing.assert_array_equal(np.stack(model.frames), np.stack(selected))
    expected = np.stack([
        cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 120),
                   interpolation=cv2.INTER_AREA)
        for frame in selected
    ])
    assert processed.payload == expected.tobytes()


def test_sample_hook_cannot_mutate_yunet_or_mpc_pixels(tmp_path: Path) -> None:
    source = _make_video(
        tmp_path / "untouched.mp4", width=32, height=24,
        fps=2, frames=2, color=True,
    )
    baseline_model = FakeYuNet()
    baseline = VideoPreprocessor().process(source, baseline_model)
    model = FakeYuNet()

    def on_sample(frame: np.ndarray, frame_index: int) -> None:
        frame[:] = 0

    observed = VideoPreprocessor().process(source, model, on_sample=on_sample)

    assert observed.payload == baseline.payload
    np.testing.assert_array_equal(np.stack(model.frames), np.stack(baseline_model.frames))
    assert observed.source_size == baseline.source_size == (32, 24)


class _SampleCapture:
    def __init__(self, frames: list[np.ndarray], fps: float = 10) -> None:
        self.frames = frames
        self.fps = fps
        self.index = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def get(self, prop: int) -> float:
        return self.fps if prop == cv2.CAP_PROP_FPS else len(self.frames)

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.index == len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame

    def release(self) -> None:
        self.released = True


def test_sample_hook_failure_aborts_and_releases_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = _SampleCapture([np.zeros((24, 32, 3), dtype=np.uint8)] * 2)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)
    received = []

    def on_sample(frame: np.ndarray, frame_index: int) -> None:
        received.append(frame_index)
        if frame_index == 1:
            raise RuntimeError("fallo técnico de placas")

    with pytest.raises(RuntimeError, match="fallo técnico de placas"):
        VideoPreprocessor().process("unused.mp4", FakeYuNet(), on_sample=on_sample)
    assert received == [0, 1]
    assert capture.released


def test_sample_hook_rejects_changing_source_dimensions_before_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = _SampleCapture([
        np.zeros((24, 32, 3), dtype=np.uint8),
        np.zeros((24, 34, 3), dtype=np.uint8),
    ])
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)
    received = []

    with pytest.raises(VideoPreprocessingError, match="dimensiones originales cambian"):
        VideoPreprocessor().process(
            "unused.mp4", FakeYuNet(),
            on_sample=lambda frame, index: received.append(index),
        )
    assert received == [0]
    assert capture.released


def test_preprocessed_video_source_size_preserves_legacy_construction() -> None:
    metadata = VideoMetadata(1, 2, 2, 1, 1, "uint8")
    assert PreprocessedVideo(b"\0" * 4, metadata).source_size is None
