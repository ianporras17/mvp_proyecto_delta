"""Smoke test real contra tres runtimes MP-SPDZ levantados por Compose."""

import os
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest

from client import cli
from client.preprocessing import VideoPreprocessor


pytestmark = pytest.mark.e2e


def _enabled() -> bool:
    return os.getenv("RUN_MPC_E2E", "").lower() in {"1", "true", "yes"}


@pytest.mark.skipif(not _enabled(), reason="requiere Docker Compose con MP-SPDZ")
def test_three_real_parties_reconstruct_tiny_video(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for port in (8000, 8100, 8101, 8102):
        response = httpx.get(f"http://localhost:{port}/health", timeout=5)
        assert response.status_code == 200

    source = tmp_path / "smoke.mp4"
    writer = cv2.VideoWriter(
        str(source),
        cv2.VideoWriter_fourcc(*"mp4v"),
        2.0,
        (64, 48),
    )
    assert writer.isOpened()
    for value in (0, 60, 120, 180):
        writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
    writer.release()

    output_root = tmp_path / "output"
    monkeypatch.setenv("DELTA_OUTPUT_DIR", str(output_root))
    assert cli.main(["upload", str(source)]) == 0

    results = list(output_root.glob("*/reconstructed.mp4"))
    assert len(results) == 1
    capture = cv2.VideoCapture(str(results[0]))
    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 4
    assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 64
    assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 48
    assert float(capture.get(cv2.CAP_PROP_FPS)) == pytest.approx(2.0, abs=0.25)
    decoded_frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        decoded_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    capture.release()

    processed = VideoPreprocessor().process(source)
    expected_frames = np.frombuffer(processed.payload, dtype=np.uint8).reshape(
        processed.metadata.frame_count,
        processed.metadata.height,
        processed.metadata.width,
    )
    decoded = np.stack(decoded_frames)
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
