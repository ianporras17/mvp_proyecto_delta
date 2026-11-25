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


class FakeYuNet:
    def setInputSize(self, size: list[int]) -> None:
        pass

    def infer(self, image: np.ndarray) -> np.ndarray:
        return np.empty((0, 15), dtype=np.float32)


def _enabled() -> bool:
    return os.getenv("RUN_MPC_E2E", "").lower() in {"1", "true", "yes"}


@pytest.mark.skipif(not _enabled(), reason="requiere Docker Compose con MP-SPDZ")
def test_three_real_parties_reconstruct_tiny_video(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator_url = os.getenv("COORDINATOR_URL", "http://localhost:8000").rstrip("/")
    configured_parties = os.getenv("PARTY_CONTROL_URLS")
    if configured_parties:
        party_urls = tuple(
            value.strip().rstrip("/")
            for value in configured_parties.split(",") if value.strip()
        )
    else:
        party_urls = tuple(f"http://localhost:{port}" for port in (8100, 8101, 8102))
    assert len(party_urls) == 3, "PARTY_CONTROL_URLS debe contener exactamente tres URLs"
    for service_url in (coordinator_url, *party_urls):
        response = httpx.get(f"{service_url}/health", timeout=5)
        assert response.status_code == 200

    source = tmp_path / "smoke.mp4"
    writer = cv2.VideoWriter(
        str(source),
        cv2.VideoWriter_fourcc(*"mp4v"),
        20.0,
        (64, 48),
    )
    assert writer.isOpened()
    for value in range(0, 200, 10):
        writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
    writer.release()

    output_root = tmp_path / "output"
    monkeypatch.setenv("DELTA_OUTPUT_DIR", str(output_root))
    model_path = (
        Path(__file__).resolve().parents[2]
        / "src/client/preprocessing/face_detection_yunet_2023mar.onnx"
    )
    original_yunet = cli.YuNet

    def local_yunet(**options):
        options["modelPath"] = str(model_path)
        return original_yunet(**options)

    exchanges: list[tuple[bytes, bytes]] = []
    original_identity = cli.MpcExternalClient.process_identity

    def record_identity(client, payload: bytes) -> bytes:
        result = original_identity(client, payload)
        exchanges.append((payload, result))
        return result

    monkeypatch.setattr(cli, "YuNet", local_yunet)
    monkeypatch.setattr(cli.MpcExternalClient, "process_identity", record_identity)
    monkeypatch.chdir(tmp_path)
    assert cli.main(["upload", str(source), "--mpc-fps", "10"]) == 0

    results = list(output_root.glob("*/reconstructed.mp4"))
    assert len(results) == 1
    capture = cv2.VideoCapture(str(results[0]))
    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 10
    assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 64
    assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 48
    output_fps = float(capture.get(cv2.CAP_PROP_FPS))
    assert output_fps == pytest.approx(10.0, abs=0.25)
    assert capture.get(cv2.CAP_PROP_FRAME_COUNT) / output_fps == pytest.approx(1.0, abs=0.05)
    decoded_frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        decoded_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    capture.release()

    processed = VideoPreprocessor(target_fps=10).process(source, FakeYuNet())
    assert processed.metadata.frame_count == 10
    assert processed.metadata.fps == 10
    assert len(processed.payload) == 10 * 64 * 48
    assert exchanges == [(processed.payload, processed.payload)]
    expected_frames = np.frombuffer(processed.payload, dtype=np.uint8).reshape(
        processed.metadata.frame_count,
        processed.metadata.height,
        processed.metadata.width,
    )
    decoded = np.stack(decoded_frames)
    assert decoded.shape == (10, 48, 64)
    mean_absolute_error = np.abs(
        expected_frames.astype(np.int16) - decoded.astype(np.int16)
    ).mean()
    assert mean_absolute_error < 5

    video_id = results[0].parent.name
    for party_url in party_urls:
        assert httpx.get(
            f"{party_url}/api/v1/jobs/{video_id}", timeout=5
        ).status_code == 404
        health = httpx.get(f"{party_url}/health", timeout=5)
        assert health.json()["runtime"] == "stopped"

    # El Coordinator conserva metadata terminal, pero el slot único ya quedó libre.
    probe = httpx.post(
        f"{coordinator_url}/api/v1/videos",
        json={"filename": "cleanup-probe.mp4", "format": "mp4"},
        timeout=5,
    )
    assert probe.status_code == 201
    probe_id = probe.json()["id"]
    assert httpx.post(
        f"{coordinator_url}/api/v1/videos/{probe_id}/fail",
        json={"detail": "fin de prueba"},
        timeout=5,
    ).status_code == 200
    assert httpx.post(
        f"{coordinator_url}/api/v1/videos/{probe_id}/cleanup", timeout=5
    ).status_code == 200
