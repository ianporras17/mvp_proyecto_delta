"""T023 real LPD-YuNet -> T024/T025 -> three parties, using video2 clips only."""

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
from client.detection import LPDYuNet
from client.networking import CoordinatorClient, PartyClient
from client.preprocessing import VideoPreprocessor, VideoReconstructor
from client.preprocessing.yunet import YuNet
from client.regions import RegionType
from mpc import MpcExternalClient
import mpc.external_io as external_io
from mpc.external_io import CHUNK_SIZE


pytestmark = pytest.mark.e2e
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _enabled() -> bool:
    return os.getenv("RUN_MPC_E2E", "").lower() in {"1", "true", "yes"}


def _urls() -> tuple[str, tuple[str, ...]]:
    coordinator = os.getenv("COORDINATOR_URL", "http://localhost:8000").rstrip("/")
    parties = tuple(url.strip().rstrip("/") for url in os.getenv(
        "PARTY_CONTROL_URLS", "http://localhost:8100,http://localhost:8101,http://localhost:8102"
    ).split(",") if url.strip())
    assert len(parties) == 3
    return coordinator, parties


def _write_visual_evidence(directory, source_frames, detections, processed, plan):
    metadata = processed.metadata
    originals = cv2.VideoWriter(
        str(directory / "detections_native.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
        metadata.fps, processed.source_size,
    )
    transformed = cv2.VideoWriter(
        str(directory / "regions_mpc.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
        metadata.fps, (metadata.width, metadata.height),
    )
    assert originals.isOpened() and transformed.isOpened()
    gray_frames = np.frombuffer(processed.payload, dtype=np.uint8).reshape(
        metadata.frame_count, metadata.height, metadata.width,
    )
    records = []
    try:
        for index, (frame, plates) in enumerate(zip(source_frames, detections)):
            original = frame.copy()
            for plate in plates:
                box = plate.region
                cv2.rectangle(original, (math.floor(box.x), math.floor(box.y)),
                              (math.ceil(box.x + box.width) - 1, math.ceil(box.y + box.height) - 1),
                              (0, 255, 0), 2)
                cv2.putText(original, f"plate {plate.confidence:.3f}",
                            (max(0, math.floor(box.x)), max(15, math.floor(box.y) - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.putText(original, f"processed frame {index}; plates {len(plates)}", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            originals.write(original)
            small = cv2.cvtColor(gray_frames[index], cv2.COLOR_GRAY2BGR)
            regions = [region for region in plan.regions if region.frame_index == index]
            for region in regions:
                cv2.rectangle(small, (region.x, region.y),
                              (region.x + region.width - 1, region.y + region.height - 1),
                              (0, 255, 0), 1)
            transformed.write(small)
            records.append({
                "frame_index": index, "processed_timestamp_seconds": index / metadata.fps,
                "detector_status": "success", "detection_count": len(plates),
                "original_regions": [{
                    "x": plate.region.x, "y": plate.region.y,
                    "width": plate.region.width, "height": plate.region.height,
                    "confidence": plate.confidence,
                } for plate in plates],
                "mpc_regions": [{
                    "x": region.x, "y": region.y,
                    "width": region.width, "height": region.height,
                } for region in regions],
            })
            if index in (0, 17, 20, 24):
                assert cv2.imwrite(str(directory / f"sample_native_{index:02}.png"), original)
    finally:
        originals.release()
        transformed.release()
    (directory / "detections.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8",
    )


@pytest.mark.skipif(not _enabled(), reason="requires Docker Compose, MP-SPDZ and local video2 clips")
@pytest.mark.parametrize("clip_name", ("video2_0000_0025", "video2_275_300"))
def test_real_lpd_yunet_plates_anonymized_by_three_parties(
    clip_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = PROJECT_ROOT / "results/T023/clips" / f"{clip_name}.mp4"
    if not source.is_file():
        pytest.skip("Prepare ignored native clips of video2.mp4 before this local E2E")
    assert source.name.startswith("video2_")
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    directory = Path(os.getenv("T023_RESULTS_DIR", str(tmp_path))).resolve() / clip_name
    directory.mkdir(parents=True, exist_ok=True)
    coordinator_url, party_urls = _urls()
    for url in (coordinator_url, *party_urls):
        assert httpx.get(f"{url}/health", timeout=5).status_code == 200

    native_frames, detections, inference_seconds = [], [], []
    original_detect = LPDYuNet.detect

    def record_detector(self, frame):
        assert frame.shape == (850, 478, 3) and frame.dtype == np.uint8
        assert self.confidence_threshold == 0.6
        native_frames.append(frame.copy())
        started = time.perf_counter()
        result = original_detect(self, frame)
        inference_seconds.append(time.perf_counter() - started)
        detections.append(result)
        return result

    monkeypatch.setattr(LPDYuNet, "detect", record_detector)
    preprocessor = VideoPreprocessor(max_width=128, max_height=128, target_fps=10)
    processed_videos = []
    original_preprocess = preprocessor.process

    def record_preprocess(*args, **kwargs):
        processed = original_preprocess(*args, **kwargs)
        processed_videos.append(processed)
        return processed

    monkeypatch.setattr(preprocessor, "process", record_preprocess)
    coordinator = CoordinatorClient(coordinator_url)
    registrations = []
    original_register = coordinator.register_video

    def record_registration(*args, **kwargs):
        assert len(detections) == len(native_frames) == 25  # Prepared BEFORE HTTP registration.
        registration = original_register(*args, **kwargs)
        registrations.append(registration)
        return registration

    monkeypatch.setattr(coordinator, "register_video", record_registration)
    mpc = MpcExternalClient()
    official_module = mpc._load_official_client()
    private_chunks, public_messages, socket_counts = [], [], []

    class RecordingOfficialClient(official_module.Client):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            socket_counts.append(len(self.sockets))

        def send_private_inputs(self, values):
            private_chunks.append(tuple(values))
            return super().send_private_inputs(values)

        def send_public_inputs(self, values):
            public_messages.append(tuple(values))
            return super().send_public_inputs(values)

    monkeypatch.setattr(mpc, "_load_official_client", lambda: SimpleNamespace(Client=RecordingOfficialClient))
    # Observe both supported paths without disabling the production optimization.
    original_mask = external_io._VectorizedChannel.send_public_mask
    original_exchange = external_io._VectorizedChannel.exchange

    def record_mask(channel, mask):
        public_messages.append(mask + bytes(CHUNK_SIZE - len(mask)))
        return original_mask(channel, mask)

    def record_private(channel, payload, offset, length):
        private_chunks.append(payload[offset:offset + length] + bytes(CHUNK_SIZE - length))
        return original_exchange(channel, payload, offset, length)

    monkeypatch.setattr(external_io._VectorizedChannel, "send_public_mask", record_mask)
    monkeypatch.setattr(external_io._VectorizedChannel, "exchange", record_private)
    exchanges = []
    original_anonymize = mpc.process_anonymize

    def record_anonymize(payload, plan):
        assert plan.detections_complete and plan.frame_count == 25
        assert payload == processed_videos[0].payload
        for url in party_urls:
            assert httpx.get(f"{url}/health", timeout=5).json()["runtime"] == "running"
            assert httpx.get(f"{url}/api/v1/jobs/{registrations[0].id}", timeout=5).json()["status"] == "PROCESSING"
        started = time.perf_counter()
        result = original_anonymize(payload, plan)
        exchanges.append((payload, result, plan, time.perf_counter() - started))
        return result

    monkeypatch.setattr(mpc, "process_anonymize", record_anonymize)
    monkeypatch.setattr(mpc, "process_identity", lambda _: pytest.fail("T023 may not fall back to identity"))
    model = YuNet(str(PROJECT_ROOT / "src/client/preprocessing/face_detection_yunet_2023mar.onnx"), confThreshold=0.9)
    # T020 annotations are diagnostics on the trusted client only.
    monkeypatch.chdir(directory)
    started = time.perf_counter()
    assert cli.upload_video(
        video_path=source, coordinator=coordinator, party_client=PartyClient(),
        preprocessor=preprocessor, mpc_client=mpc, reconstructor=VideoReconstructor(),
        model=model, output_root=directory / "output", processing_mode="anonymize",
        detect_plates=True, plate_confidence_threshold=0.6,
    ) == 0
    total_seconds = time.perf_counter() - started
    assert len(exchanges) == len(processed_videos) == 1
    processed = processed_videos[0]
    metadata = processed.metadata
    assert processed.source_size == (478, 850)
    assert (metadata.width, metadata.height) == (70, 128)
    assert metadata.frame_count == len(detections) == len(native_frames) == 25
    assert metadata.fps == 10
    payload, result, plan, mpc_seconds = exchanges[0]
    original = np.frombuffer(payload, dtype=np.uint8).reshape(25, 128, 70)
    actual = np.frombuffer(result, dtype=np.uint8).reshape(original.shape)
    # Validate selected native frame -> MPC pixels in the same processed order.
    for index, frame in enumerate(native_frames):
        expected = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (70, 128), interpolation=cv2.INTER_AREA)
        assert np.array_equal(original[index], expected)
    assert any(detections) and plan.regions
    assert all(region.region_type == RegionType.PLATE for region in plan.regions)
    # Independent geometry checks: no extra margins beyond T024 outward rounding.
    expected_regions = []
    for index, plates in enumerate(detections):
        for plate in plates:
            box = plate.region
            x0, y0 = math.floor(box.x * 70 / 478), math.floor(box.y * 128 / 850)
            x1 = min(70, math.ceil((box.x + box.width) * 70 / 478))
            y1 = min(128, math.ceil((box.y + box.height) * 128 / 850))
            expected_regions.append((index, x0, y0, x1 - x0, y1 - y0))
    assert [(r.frame_index, r.x, r.y, r.width, r.height) for r in plan.regions] == expected_regions
    # Test-only oracle never substitutes for cryptographic output or modifies source.
    mask = np.zeros(original.shape, dtype=np.bool_)
    for region in plan.regions:
        mask[region.frame_index, region.y:region.y + region.height, region.x:region.x + region.width] = True
    assert mask.any() and (~mask).any()
    assert np.all(actual[mask] == 0)
    assert np.array_equal(actual[~mask], original[~mask])
    assert (original[mask] > 0).any() and result != payload
    chunks = math.ceil(len(payload) / CHUNK_SIZE)
    assert socket_counts == [3]
    assert len(private_chunks) == len(public_messages) - 1 == chunks
    private_stream = bytes(value for chunk in private_chunks for value in chunk)
    assert private_stream[:len(payload)] == payload
    assert private_stream[len(payload):] == bytes(len(private_stream) - len(payload))
    assert public_messages[0] == (-len(payload), chunks)
    mask_stream = bytes(value for chunk in public_messages[1:] for value in chunk)
    assert mask_stream[:len(payload)] == mask.astype(np.uint8).tobytes()
    assert mask_stream[len(payload):] == bytes(len(mask_stream) - len(payload))

    result_path = directory / "output" / registrations[0].id / "reconstructed.mp4"
    capture = cv2.VideoCapture(str(result_path))
    decoded = []
    try:
        assert capture.isOpened()
        assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(10, abs=0.25)
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    finally:
        capture.release()
    decoded = np.stack(decoded)
    assert decoded.shape == actual.shape
    codec_mae = float(np.abs(decoded.astype(np.int16) - actual.astype(np.int16)).mean())
    assert codec_mae < 5
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash
    assert coordinator.get_video(registrations[0].id).status == "COMPLETED"
    for url in party_urls:
        assert httpx.get(f"{url}/api/v1/jobs/{registrations[0].id}", timeout=5).status_code == 404
    _write_visual_evidence(directory, native_frames, detections, processed, plan)
    summary = {
        "clip": str(source), "clip_sha256_intact": original_hash,
        "source_size": processed.source_size, "mpc_size": [70, 128],
        "frame_count": 25, "fps": 10, "plate_confidence_threshold": 0.6,
        "detection_observations": sum(len(plates) for plates in detections),
        "frames_with_detections": sum(bool(plates) for plates in detections),
        "frames_without_detections": [index for index, plates in enumerate(detections) if not plates],
        "masked_pixels": int(mask.sum()), "private_pixels": len(payload), "chunks": chunks,
        "inference_seconds": round(sum(inference_seconds), 3),
        "mpc_seconds": round(mpc_seconds, 3), "total_seconds": round(total_seconds, 3),
        "three_real_sockets": True, "source_original_bgr": True,
        "source_to_mpc_frame_exact": True, "original_sent_privately": True,
        "public_masks_match_regions": True, "inside_zero_before_codec": True,
        "outside_byte_exact_before_codec": True, "fallback_to_identity": False,
        "codec_mae": round(codec_mae, 3), "result_path": str(result_path),
        "diagnostics": "detections_native.mp4 (sampled original color), regions_mpc.mp4 (public boxes on MPC input), detections.jsonl",
        "quality_caveat": "Detection observations are not verified true plates; successful empty inference may miss a visible plate; no tracking or box expansion",
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("T023_METRICS " + json.dumps(summary))
