"""Original-color detector inputs remain aligned with the private MPC payload."""

import cv2
import numpy as np
import pytest

from client.preprocessing import VideoPreprocessingError, VideoPreprocessor


class ArrayCapture:
    def __init__(self, frames, fps):
        self.frames = frames
        self.fps = fps
        self.index = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        return self.fps if prop == cv2.CAP_PROP_FPS else len(self.frames)

    def read(self):
        if self.index == len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame

    def release(self):
        self.released = True


class RecordingYuNet:
    def __init__(self, detections=None):
        self.frames = []
        self.sizes = []
        self.detections = (
            np.empty((0, 15), dtype=np.float32) if detections is None else detections
        )

    def setInputSize(self, size):
        self.sizes.append(tuple(size))

    def infer(self, image):
        self.frames.append(image.copy())
        return self.detections


@pytest.mark.parametrize(
    "fps,target_fps,indices",
    [(30, 10, [0, 3, 6, 9, 12]), (30, 5, [0, 6, 12]), (6, 10, list(range(15)))],
)
def test_both_detectors_observe_original_bgr_and_processed_indices(
    monkeypatch, fps, target_fps, indices,
):
    frames = [np.full((24, 32, 3), (10 + i, 80 + i, 180 + i), dtype=np.uint8)
              for i in range(15)]
    capture = ArrayCapture(frames, fps)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)
    samples = []

    def on_sample(original, index):
        samples.append((index, original.copy()))
        original[:] = 0  # A plate adapter cannot mutate YuNet or private pixels.

    detector = RecordingYuNet()
    processed = VideoPreprocessor(
        max_width=16, max_height=12, target_fps=target_fps,
    ).process("input.mp4", detector, on_sample=on_sample)

    selected = np.stack([frames[index] for index in indices])
    assert processed.source_size == (32, 24)
    assert (processed.metadata.width, processed.metadata.height) == (16, 12)
    assert processed.metadata.frame_count == len(indices)
    assert processed.metadata.fps == min(fps, target_fps)
    assert [index for index, _ in samples] == list(range(len(indices)))
    np.testing.assert_array_equal(np.stack([image for _, image in samples]), selected)
    np.testing.assert_array_equal(np.stack(detector.frames), selected)
    assert detector.sizes == [(32, 24)] * len(indices)
    expected = np.stack([
        cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (16, 12),
                   interpolation=cv2.INTER_AREA) for image in selected
    ])
    assert processed.payload == expected.tobytes()
    assert processed.regions == ()
    assert capture.released


def test_face_score_threshold_uses_adjacent_fp32_values():
    threshold = np.float32(0.6)
    scores = [np.nextafter(threshold, np.float32(0)), threshold,
              np.nextafter(threshold, np.float32(1))]
    detections = np.array([
        [2 * index, 0, 2, 2, *([0.0] * 10), score]
        for index, score in enumerate(scores)
    ], dtype=np.float32)
    regions = VideoPreprocessor._face_regions(detections, 7, (8, 4), (8, 4))
    assert [(region.frame_id, region.x, region.confidence) for region in regions] == [
        (7, 4, float(scores[2]))
    ]


def test_face_boxes_are_scaled_and_clipped_to_actual_mpc_dimensions():
    detections = np.array([
        [-4, -2, 12, 10, *([0.0] * 10), 0.9],
        [31, 23, 10, 10, *([0.0] * 10), 0.9],
    ], dtype=np.float32)
    regions = VideoPreprocessor._face_regions(detections, 2, (32, 24), (16, 12))
    assert [(r.frame_id, r.x, r.y, r.width, r.height) for r in regions] == [
        (2, 0, 0, 4, 4), (2, 15, 11, 1, 1),
    ]


def test_failed_plate_callback_aborts_preparation_and_releases_capture(monkeypatch):
    capture = ArrayCapture([np.ones((24, 32, 3), dtype=np.uint8)] * 2, 10)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)
    detector = RecordingYuNet()

    def on_sample(image, index):
        if index == 1:
            raise RuntimeError("plate detector failed")

    with pytest.raises(RuntimeError, match="plate detector failed"):
        VideoPreprocessor().process("input.mp4", detector, on_sample=on_sample)
    assert len(detector.frames) == 1
    assert capture.released


def test_changing_original_dimensions_aborts_before_detectors(monkeypatch):
    capture = ArrayCapture([
        np.ones((24, 32, 3), dtype=np.uint8),
        np.ones((24, 34, 3), dtype=np.uint8),
    ], 10)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)
    samples = []
    detector = RecordingYuNet()
    with pytest.raises(VideoPreprocessingError, match="dimensiones originales cambian"):
        VideoPreprocessor().process(
            "input.mp4", detector,
            on_sample=lambda image, index: samples.append(index),
        )
    assert samples == [0]
    assert len(detector.frames) == 1
    assert capture.released
