"""Conteo en el mismo muestreo original del cliente, sin alterar sus píxeles."""
import json
from unittest.mock import Mock
from uuid import uuid4

import cv2
import numpy as np

from client import cli
from client.detection.yolo_counter import VideoCounter, ObjectDetection
from client.networking import PartyEndpoint, VideoRegistration
from client.preprocessing import VideoPreprocessor, VideoReconstructor


def test_counting_and_mpc_keep_same_pixels_and_order(tmp_path, monkeypatch):
    source = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    assert writer.isOpened()
    for index in range(6):
        writer.write(np.full((48, 64, 3), 30+index*10, np.uint8))
    writer.release()
    originals, observed = [], []

    class FaceModel:
        def setInputSize(self, size):
            assert size == [64, 48]
        def infer(self, frame):
            originals.append(frame.copy())
            return np.empty((0, 15), np.float32)

    class Detector:
        def detect(self, frame):
            observed.append(frame.copy())
            return (ObjectDetection("person", (10, 10, 20, 20), .9),)

    counter = VideoCounter(detector=Detector())
    monkeypatch.setattr(cli, "VideoCounter", lambda: counter)
    coordinator, parties, mpc = Mock(), Mock(), Mock()
    video_id = str(uuid4())
    coordinator.register_video.return_value = VideoRegistration(video_id, "REGISTERED", tuple(
        PartyEndpoint(index, f"http://p{index}") for index in range(3)))
    mpc.process_identity.side_effect = lambda payload: payload
    assert cli.upload_video(source, coordinator, parties, VideoPreprocessor(), mpc,
                            VideoReconstructor(), FaceModel(), tmp_path / "out",
                            count_objects=True) == 0
    assert counter.summary()["people"] == 1
    np.testing.assert_array_equal(np.stack(originals), np.stack(observed))
    payload = mpc.process_identity.call_args.args[0]
    expected = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in originals])
    assert payload == expected.tobytes()
    report = json.loads((tmp_path / "out" / video_id / "counts.json").read_text())
    assert report["people"] == 1 and report["frames_analyzed"] == 6
