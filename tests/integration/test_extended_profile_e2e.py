"""Pruebas reales acotadas: 1080p y más de 20 segundos a 30 FPS, por separado."""
import os
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import pytest

from client import cli
from client.networking import CoordinatorClient, PartyClient
from client.preprocessing import VideoPreprocessor, VideoReconstructor
from mpc import MpcExternalClient


pytestmark = [pytest.mark.e2e, pytest.mark.skipif(
    os.getenv("RUN_MPC_E2E") != "1", reason="requires three real MP-SPDZ parties")]


class NoFaces:
    def setInputSize(self, size):
        pass
    def infer(self, frame):
        return np.empty((0, 15), np.float32)


@pytest.mark.parametrize("size,frames", [((1920, 1080), 1), ((16, 16), 630)],
                         ids=["full-hd-single-frame", "21-seconds-at-30fps"])
def test_extended_profile_real_identity(size, frames, tmp_path, monkeypatch):
    source = tmp_path / "profile.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 30, size)
    assert writer.isOpened()
    for index in range(frames):
        writer.write(np.full((size[1], size[0], 3), 30 + index % 170, np.uint8))
    writer.release()
    exchanges = []
    mpc = MpcExternalClient()
    original = mpc.process_identity
    def process(payload):
        start = perf_counter()
        result = original(payload)
        assert result == payload
        exchanges.append((len(payload), perf_counter()-start))
        return result
    monkeypatch.setattr(mpc, "process_identity", process)
    started = perf_counter()
    assert cli.upload_video(source, CoordinatorClient(os.getenv("COORDINATOR_URL", "http://localhost:8000")),
                            PartyClient(), VideoPreprocessor(), mpc, VideoReconstructor(),
                            NoFaces(), tmp_path / "output") == 0
    result = next((tmp_path / "output").glob("*/reconstructed.mp4"))
    capture = cv2.VideoCapture(str(result))
    assert capture.isOpened()
    assert (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))) == size
    assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(30)
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == frames
    capture.release()
    assert exchanges[0][0] == size[0]*size[1]*frames
    print(f"\nPROFILE {size}, {frames} frames: ExternalIO={exchanges[0][1]:.3f}s total={perf_counter()-started:.3f}s")
