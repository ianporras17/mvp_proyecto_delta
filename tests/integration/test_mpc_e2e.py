"""Smoke test real contra tres runtimes MP-SPDZ levantados por Compose."""

import hashlib
import os
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest

from client import cli
from client.preprocessing import VideoPreprocessor
from client.preprocessing.yunet import DEFAULT_YUNET_MODEL, YuNet
from mpc import MpcExternalClient


pytestmark = pytest.mark.e2e


def _enabled() -> bool:
    return os.getenv("RUN_MPC_E2E", "").lower() in {"1", "true", "yes"}


@pytest.mark.skipif(not _enabled(), reason="requiere Docker Compose con MP-SPDZ")
@pytest.mark.parametrize(
    "source_size,source_fps,source_count,expected_fps,expected_count",
    [
        pytest.param((128, 96), 20.0, 20, 10.0, 10, id="10-fps-sampling"),
        pytest.param((640, 480), 1.0, 1, 1.0, 1, id="640x480-single-frame"),
    ],
)
def test_three_real_parties_reconstruct_tiny_video(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_size: tuple[int, int],
    source_fps: float,
    source_count: int,
    expected_fps: float,
    expected_count: int,
) -> None:
    for port in (8000, 8100, 8101, 8102):
        response = httpx.get(f"http://localhost:{port}/health", timeout=5)
        assert response.status_code == 200

    source = tmp_path / "smoke.mp4"
    writer = cv2.VideoWriter(
        str(source),
        cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps,
        source_size,
    )
    assert writer.isOpened()
    width, height = source_size
    for frame_index in range(source_count):
        writer.write(
            np.full((height, width, 3), frame_index * 11 + 20, dtype=np.uint8)
        )
    writer.release()
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    processed = VideoPreprocessor().process(source, YuNet(DEFAULT_YUNET_MODEL))
    assert processed.metadata.frame_count == expected_count
    assert processed.metadata.fps == expected_fps
    assert (processed.metadata.width, processed.metadata.height) == source_size

    # Observe the real ExternalIO call, not a replacement or a second MPC run.
    original_process = MpcExternalClient.process_identity
    observed_payloads: list[tuple[bytes, bytes]] = []

    def observe_identity(client: MpcExternalClient, payload: bytes) -> bytes:
        result = original_process(client, payload)
        observed_payloads.append((payload, result))
        assert result == payload == processed.payload
        return result

    monkeypatch.setattr(MpcExternalClient, "process_identity", observe_identity)

    output_root = tmp_path / "output"
    monkeypatch.setenv("DELTA_OUTPUT_DIR", str(output_root))
    assert cli.main(["upload", str(source)]) == 0
    assert len(observed_payloads) == 1
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_sha256

    results = list(output_root.glob("*/reconstructed.mp4"))
    assert len(results) == 1
    capture = cv2.VideoCapture(str(results[0]))
    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == expected_count
    assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == width
    assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == height
    assert float(capture.get(cv2.CAP_PROP_FPS)) == pytest.approx(
        expected_fps, abs=0.05
    )
    decoded_frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        decoded_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    capture.release()

    assert len(decoded_frames) == expected_count
    expected_frames = np.frombuffer(processed.payload, dtype=np.uint8).reshape(
        processed.metadata.frame_count,
        processed.metadata.height,
        processed.metadata.width,
    )
    decoded = np.stack(decoded_frames)
    if expected_count > 1:
        # Samples and encoded frames must remain in their original time order.
        assert np.all(np.diff(expected_frames.mean(axis=(1, 2))) > 0)
        assert np.all(np.diff(decoded.mean(axis=(1, 2))) > 0)
    mean_absolute_error = np.abs(
        expected_frames.astype(np.int16) - decoded.astype(np.int16)
    ).mean()
    assert mean_absolute_error < 5

    video_id = results[0].parent.name
    for port in (8100, 8101, 8102):
        assert httpx.get(
            f"http://localhost:{port}/api/v1/jobs/{video_id}", timeout=5
        ).status_code == 404
        health = httpx.get(f"http://localhost:{port}/health", timeout=5)
        assert health.json()["runtime"] == "stopped"

    # El Coordinator conserva metadata terminal, pero el slot único ya quedó libre.
    probe = httpx.post(
        "http://localhost:8000/api/v1/videos",
        json={"filename": "cleanup-probe.mp4", "format": "mp4"},
        timeout=5,
    )
    assert probe.status_code == 201
    probe_id = probe.json()["id"]
    assert httpx.post(
        f"http://localhost:8000/api/v1/videos/{probe_id}/fail",
        json={"detail": "fin de prueba"},
        timeout=5,
    ).status_code == 200
    assert httpx.post(
        f"http://localhost:8000/api/v1/videos/{probe_id}/cleanup", timeout=5
    ).status_code == 200
