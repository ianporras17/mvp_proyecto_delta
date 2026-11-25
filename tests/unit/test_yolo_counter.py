import hashlib

import numpy as np
import pytest

from client.detection.yolo_counter import (
    CountingError, DEFAULT_YOLO_MODEL, MODEL_SHA256, ObjectDetection,
    VideoCounter, YOLODetector,
)


def detection(kind="person", x=10, y=10):
    return ObjectDetection(kind, (x, y, 40, 80), .9)


def test_one_person_in_one_thousand_frames_is_counted_once():
    counter = VideoCounter(detector=object())
    for frame in range(1000):
        counter.update((detection(x=10 + frame * .1),), frame)
    assert counter.summary()["people"] == 1
    assert counter.summary()["cars"] == 0
    assert counter.summary()["frames_analyzed"] == 1000


def test_two_identical_cars_are_distinct_tracks_not_observation_sums():
    counter = VideoCounter(detector=object())
    for frame in range(30):
        counter.update((detection("car", 10+frame), detection("car", 100+frame)), frame)
    assert counter.summary()["cars"] == 2
    assert counter.summary()["people"] == 0


def test_classes_do_not_share_ids_and_short_occlusion_does_not_recount():
    counter = VideoCounter(detector=object(), max_missed=3)
    for frame in range(3):
        counter.update((detection(), detection("car")), frame)
    counter.update((), 3)
    counter.update((), 4)
    counter.update((detection(), detection("car")), 5)
    assert counter.summary()["people"] == counter.summary()["cars"] == 1


def test_single_frame_false_positive_is_not_counted_and_expired_track_is_removed():
    counter = VideoCounter(detector=object(), max_missed=1)
    counter.update((detection(),), 0)
    counter.update((), 1)
    counter.update((), 2)
    counter.update((), 3)
    assert counter.summary()["people"] == 0
    assert not counter.tracks


def test_long_reentry_is_explicitly_a_new_trajectory_not_biometric_identity():
    counter = VideoCounter(detector=object(), max_missed=0, min_hits=1)
    counter.update((detection(),), 0)
    counter.update((), 1)
    counter.update((detection(),), 2)
    assert counter.summary()["people"] == 2
    assert "reingresos" in counter.summary()["limitation"]


@pytest.mark.parametrize("index", [-1, 1, True])
def test_rejects_nonconsecutive_indices(index):
    with pytest.raises(CountingError):
        VideoCounter(detector=object()).update((), index)


def test_detector_failure_is_not_empty_success():
    class Broken:
        def detect(self, image):
            raise CountingError("fallo técnico")
    counter = VideoCounter(detector=Broken())
    with pytest.raises(CountingError):
        counter.observe(np.zeros((32, 32, 3), np.uint8), 0)
    assert counter.frame_index == -1


def test_yolo_resource_hash_and_real_native_inference():
    assert hashlib.sha256(DEFAULT_YOLO_MODEL.read_bytes()).hexdigest() == MODEL_SHA256
    detector = YOLODetector()
    image = np.zeros((96, 128, 3), np.uint8)
    original = image.copy()
    assert detector.detect(image) == ()
    np.testing.assert_array_equal(image, original)


def test_yolo_decoder_filters_other_classes_and_uses_class_separate_nms():
    detector = YOLODetector()
    output = np.zeros((1, 8400, 85), np.float32)
    for index, kind in ((0, 0), (1, 2), (2, 7)):
        output[0, index, :4] = (5, 5, 1, 1)
        output[0, index, 4] = .95
        output[0, index, 5+kind] = .95
    result = detector._decode(output, 1., 640, 640)
    assert {item.kind for item in result} == {"person", "car"}
    assert all(item.box[2] > 0 for item in result)
    with pytest.raises(CountingError):
        detector._decode(np.zeros((1, 1, 85)), 1., 640, 640)


@pytest.mark.parametrize("image", [np.zeros((20, 20), np.uint8), np.zeros((20, 20, 3), np.float32)])
def test_yolo_rejects_non_bgr_uint8(image):
    with pytest.raises(CountingError):
        YOLODetector().detect(image)
