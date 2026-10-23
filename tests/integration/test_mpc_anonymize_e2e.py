"""T025 real: synthetic T024 regions, official ExternalIO and three MP-SPDZ parties."""

import hashlib
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

import cv2
import httpx
import numpy as np
import pytest

from client import cli
from client.anonymization import prepare_anonymization
from client.networking import CoordinatorClient, PartyClient
from client.preprocessing import VideoPreprocessor, VideoReconstructor
from client.preprocessing.yunet import YuNet
from client.regions import DetectedRegion, FrameDetections, RegionType
from mpc import MpcExternalClient
from mpc.external_io import CHUNK_SIZE


pytestmark = pytest.mark.e2e


def _enabled() -> bool:
    return os.getenv("RUN_MPC_E2E", "").lower() in {"1", "true", "yes"}


def _service_urls() -> tuple[str, tuple[str, ...]]:
    coordinator = os.getenv("COORDINATOR_URL", "http://localhost:8000").rstrip("/")
    parties = tuple(
        item.strip().rstrip("/")
        for item in os.getenv(
            "PARTY_CONTROL_URLS", "http://localhost:8100,http://localhost:8101,http://localhost:8102"
        ).split(",") if item.strip()
    )
    assert len(parties) == 3
    return coordinator, parties


def _write_source(path: Path, size: tuple[int, int]) -> None:
    width, height = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, size)
    assert writer.isOpened()
    y, x = np.indices((height, width))
    try:
        for index in range(4):
            # Distinct, nonzero frames make index shifts and accidental identity visible.
            gray = (30 + index * 35 + (x + y) % 25).astype(np.uint8)
            writer.write(np.stack((gray, gray + 8, gray + 16), axis=-1))
    finally:
        writer.release()


def _detections(width: int, height: int) -> tuple[FrameDetections, ...]:
    return (
        FrameDetections.success(0),
        FrameDetections.success(1, (
            DetectedRegion(RegionType.FACE, 2, 4, width // 3, height // 2),
        )),
        FrameDetections.success(2, (
            DetectedRegion(RegionType.PLATE, width - 16, height - 12, 20, 20),
        )),
        FrameDetections.success(3, (
            DetectedRegion(RegionType.FACE, 8, 8, width // 2, height // 2),
            DetectedRegion(RegionType.PLATE, 16, 16, width // 2, height // 2),
            DetectedRegion(RegionType.PLATE, 0, height - 2, width, 2),
        )),
    )


@pytest.mark.skipif(not _enabled(), reason="requires Docker Compose and real MP-SPDZ")
@pytest.mark.parametrize(
    ("source_size", "target_size"),
    [((64, 48), (64, 48)), ((256, 192), (128, 96))],
    ids=["smoke-64x48-cross-frame-block", "scaled-128x96-mixed-overlapping-regions"],
)
def test_three_real_parties_anonymize_synthetic_t024_regions(
    source_size: tuple[int, int],
    target_size: tuple[int, int],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator_url, party_urls = _service_urls()
    for url in (coordinator_url, *party_urls):
        assert httpx.get(f"{url}/health", timeout=5).status_code == 200

    source = tmp_path / "synthetic.mp4"
    _write_source(source, source_size)
    plan = prepare_anonymization(
        _detections(*source_size), source_size=source_size,
        target_size=target_size, frame_count=4,
    )
    model_path = Path(__file__).resolve().parents[2] / (
        "src/client/preprocessing/face_detection_yunet_2023mar.onnx"
    )
    model = YuNet(str(model_path), confThreshold=0.9)
    inference_sizes = []
    original_infer = model.infer

    def record_inference(frame):
        inference_sizes.append(frame.shape)
        return original_infer(frame)

    monkeypatch.setattr(model, "infer", record_inference)
    monkeypatch.chdir(tmp_path)  # Existing T020 JPGs stay solely on the trusted client.
    preprocessor = VideoPreprocessor(
        max_width=target_size[0], max_height=target_size[1], target_fps=2,
    )
    processed_videos = []
    original_preprocess = preprocessor.process

    def record_preprocess(path, selected_model):
        processed = original_preprocess(path, selected_model)
        processed_videos.append(processed)
        return processed

    monkeypatch.setattr(preprocessor, "process", record_preprocess)
    coordinator = CoordinatorClient(coordinator_url)
    registrations = []
    original_register = coordinator.register_video

    def record_registration(filename, video_format):
        registration = original_register(filename, video_format)
        registrations.append(registration)
        return registration

    monkeypatch.setattr(coordinator, "register_video", record_registration)
    mpc = MpcExternalClient()
    official_module = mpc._load_official_client()
    private_chunks = []
    public_messages = []
    socket_counts = []

    class RecordingOfficialClient(official_module.Client):
        """Only observes calls; cryptography and sockets remain the upstream client."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            socket_counts.append(len(self.sockets))

        def send_private_inputs(self, values):
            private_chunks.append(tuple(values))
            return super().send_private_inputs(values)

        def send_public_inputs(self, values):
            public_messages.append(tuple(values))
            return super().send_public_inputs(values)

    monkeypatch.setattr(
        mpc, "_load_official_client", lambda: SimpleNamespace(Client=RecordingOfficialClient)
    )
    exchanges = []
    original_anonymize = mpc.process_anonymize

    def record_anonymize(payload, selected_plan):
        assert selected_plan is plan
        assert payload == processed_videos[0].payload
        for url in party_urls:
            assert httpx.get(f"{url}/health", timeout=5).json()["runtime"] == "running"
            state = httpx.get(
                f"{url}/api/v1/jobs/{registrations[0].id}", timeout=5,
            ).json()
            assert state["status"] == "PROCESSING"
        started = time.perf_counter()
        result = original_anonymize(payload, selected_plan)
        exchanges.append((payload, result, time.perf_counter() - started))
        return result

    monkeypatch.setattr(mpc, "process_anonymize", record_anonymize)
    started = time.perf_counter()
    assert cli.upload_video(
        video_path=source, coordinator=coordinator, party_client=PartyClient(),
        preprocessor=preprocessor, mpc_client=mpc, reconstructor=VideoReconstructor(),
        model=model, output_root=tmp_path / "output",
        processing_mode="anonymize", anonymization_plan=plan,
    ) == 0
    total_seconds = time.perf_counter() - started

    assert inference_sizes == [(source_size[1], source_size[0], 3)] * 4
    assert len(processed_videos) == len(exchanges) == 1
    processed = processed_videos[0]
    assert (processed.metadata.width, processed.metadata.height) == target_size
    assert processed.metadata.frame_count == 4
    assert processed.metadata.fps == 2
    original_payload, result_payload, mpc_seconds = exchanges[0]
    original = np.frombuffer(original_payload, dtype=np.uint8).reshape(4, target_size[1], target_size[0])
    assert original.min() > 0
    actual = np.frombuffer(result_payload, dtype=np.uint8).reshape(original.shape)
    # Independent NumPy oracle is test-only and NEVER substituted for MPC output.
    mask = np.zeros(original.shape, dtype=np.bool_)
    for region in plan.regions:
        mask[
            region.frame_index,
            region.y:region.y + region.height,
            region.x:region.x + region.width,
        ] = True
    assert not mask[0].any()
    assert all(mask[index].any() for index in (1, 2, 3))
    assert np.all(actual[mask] == 0)
    assert np.array_equal(actual[~mask], original[~mask])
    assert np.array_equal(actual[0], original[0])
    assert result_payload != original_payload

    number_chunks = math.ceil(len(original_payload) / CHUNK_SIZE)
    assert socket_counts == [3]
    assert len(private_chunks) == number_chunks
    private_stream = bytes(value for chunk in private_chunks for value in chunk)
    assert private_stream[:len(original_payload)] == original_payload
    assert private_stream[len(original_payload):] == bytes(len(private_stream) - len(original_payload))
    assert public_messages[0] == (-len(original_payload), number_chunks)
    public_masks = public_messages[1:]
    assert len(public_masks) == number_chunks
    mask_stream = bytes(value for chunk in public_masks for value in chunk)
    assert mask_stream[:len(original_payload)] == mask.astype(np.uint8).tobytes()
    assert mask_stream[len(original_payload):] == bytes(len(mask_stream) - len(original_payload))

    result_path = tmp_path / "output" / registrations[0].id / "reconstructed.mp4"
    capture = cv2.VideoCapture(str(result_path))
    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == target_size[0]
    assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == target_size[1]
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 4
    assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(2.0, abs=0.25)
    decoded = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        decoded.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    capture.release()
    decoded = np.stack(decoded)
    assert decoded.shape == actual.shape
    codec_mae = float(np.abs(decoded.astype(np.int16) - actual.astype(np.int16)).mean())
    assert codec_mae < 5

    terminal = coordinator.get_video(registrations[0].id)
    assert terminal.status == "COMPLETED"
    for url in party_urls:
        assert httpx.get(f"{url}/api/v1/jobs/{registrations[0].id}", timeout=5).status_code == 404
        assert httpx.get(f"{url}/health", timeout=5).json()["runtime"] == "stopped"
    print("T025_METRICS " + json.dumps({
        "source_size": source_size, "target_size": target_size,
        "frame_count": 4, "fps": 2, "mpc_seconds": round(mpc_seconds, 3),
        "total_seconds": round(total_seconds, 3), "chunks": number_chunks,
        "private_pixels": len(original_payload), "masked_pixels": int(mask.sum()),
        "three_official_sockets": True, "original_sent_privately": True,
        "inside_zero": True, "outside_byte_exact": True,
        "output_sha256": hashlib.sha256(result_payload).hexdigest(),
        "codec_mae": round(codec_mae, 3), "result_path": str(result_path),
    }))
