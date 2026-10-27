"""T023: real region preparation and CLI control, with network/MPC boundaries stubbed.

The rectangle oracle below belongs only to tests. Production must send the
unchanged private payload to MP-SPDZ and reconstruct only its validated output.
"""

from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest
import cv2

from client import cli
from client.anonymization import prepare_anonymization
from client.detection import PlateDetectionError
from client.networking import PartyEndpoint, VideoRegistration
from client.preprocessing import (
    FaceRegion, PreprocessedVideo, VideoMetadata, VideoPreprocessingError, VideoPreprocessor,
)
from client.regions import (
    DetectedRegion,
    DetectorExecutionError,
    FrameDetections,
    RegionType,
    SensitiveRegion,
)


@dataclass
class Coordinator:
    calls: list[str] = field(default_factory=list)
    video_id: str = field(default_factory=lambda: str(uuid4()))
    on_register: object = None

    def register_video(self, filename, video_format):
        if self.on_register is not None:
            self.on_register()
        self.calls.append("register")
        return VideoRegistration(
            self.video_id,
            "REGISTERED",
            tuple(PartyEndpoint(i, f"http://p{i}") for i in range(3)),
        )

    def prepare(self, video_id, metadata):
        self.calls.append("prepare")

    def wait_for_status(self, video_id, status):
        self.calls.append(f"wait:{status}")

    def start(self, video_id):
        self.calls.append("start")

    def complete(self, video_id):
        self.calls.append("complete")

    def cleanup(self, video_id):
        self.calls.append("cleanup")

    def fail(self, video_id, detail):
        self.calls.append("fail")


@dataclass
class Parties:
    calls: list[str] = field(default_factory=list)

    def prepare_job(self, party_id, url, video_id, metadata):
        self.calls.append(f"prepare:{party_id}")

    def start_job(self, party_id, url, video_id):
        self.calls.append(f"start:{party_id}")

    def complete_job(self, party_id, url, video_id):
        self.calls.append(f"complete:{party_id}")

    def cleanup_job(self, party_id, url, video_id):
        self.calls.append(f"cleanup:{party_id}")

    def fail_job(self, party_id, url, video_id, detail):
        self.calls.append(f"fail:{party_id}")


class SamplePreprocessor:
    def __init__(self, *, indices=(0, 1, 2), source_size=(8, 4)):
        self.indices = indices
        self.calls = []
        self.frames = [
            np.broadcast_to(np.array([11 + i, 71 + i, 181 + i], np.uint8), (4, 8, 3)).copy()
            for i in range(4)
        ]
        self.processed = PreprocessedVideo(
            bytes(range(1, 25)),
            VideoMetadata(frame_count=3, height=2, width=4, channels=1, fps=10, dtype="uint8"),
            source_size=source_size,
        )

    def process(self, path, model, *, on_sample=None):
        self.calls.append(on_sample)
        if on_sample is not None:
            for position, index in enumerate(self.indices):
                on_sample(self.frames[position].copy(), index)
        return self.processed


class BoundaryMpc:
    def __init__(self, *, corrupt=False):
        self.payloads = []
        self.plans = []
        self.identity_calls = []
        self.corrupt = corrupt

    def process_identity(self, payload):
        self.identity_calls.append(payload)
        return payload

    def process_anonymize(self, payload, plan):
        self.payloads.append(payload)
        self.plans.append(plan)
        # Independent test oracle, not client-side production anonymization.
        output = np.frombuffer(payload, np.uint8).reshape(
            plan.frame_count, plan.height, plan.width
        ).copy()
        for region in plan.regions:
            output[
                region.frame_index,
                region.y:region.y + region.height,
                region.x:region.x + region.width,
            ] = 0
        if self.corrupt:
            output[:] = 255
        return output.tobytes()


class Reconstructor:
    def __init__(self):
        self.payloads = []
        self.metadata = []

    def reconstruct(self, payload, metadata, video_id, output_root):
        self.payloads.append(payload)
        self.metadata.append(metadata)
        return Path(output_root) / video_id / "reconstructed.mp4"


@pytest.fixture
def flow(tmp_path, monkeypatch):
    source = tmp_path / "video2.mp4"
    source.write_bytes(b"local video placeholder; capture replaced for CLI boundary tests")
    state = SimpleNamespace(
        source=source,
        coordinator=Coordinator(),
        parties=Parties(),
        preprocessor=SamplePreprocessor(),
        mpc=BoundaryMpc(),
        reconstructor=Reconstructor(),
        model=object(),
        detector_calls=[],
        detector_options=[],
        results=tuple(FrameDetections.success(i) for i in range(3)),
        detector_error=None,
        output_root=tmp_path / "output",
    )

    class Detector:
        def __init__(self, **options):
            state.detector_options.append(options)
            # Keep the real constructor's input validation at the mocked boundary.
            threshold = options["confidence_threshold"]
            if isinstance(threshold, bool) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
                raise ValueError("confidence_threshold debe estar entre 0 y 1")
            if state.detector_error is not None:
                raise state.detector_error

        def detect_frame(self, original_bgr, frame_index):
            state.detector_calls.append((original_bgr.copy(), frame_index))
            if callable(state.results):
                return state.results(original_bgr, frame_index)
            return state.results[frame_index]

    monkeypatch.setattr(cli, "LPDYuNet", Detector)
    return state


def upload(flow, **options):
    return cli.upload_video(
        flow.source,
        flow.coordinator,
        flow.parties,
        flow.preprocessor,
        flow.mpc,
        flow.reconstructor,
        model=flow.model,
        output_root=flow.output_root,
        processing_mode=options.pop("processing_mode", "anonymize"),
        detect_plates=options.pop("detect_plates", True),
        **options,
    )


def assert_no_remote_work(flow):
    assert flow.coordinator.calls == []
    assert flow.parties.calls == []
    assert flow.mpc.payloads == []
    assert flow.mpc.identity_calls == []
    assert flow.reconstructor.payloads == []


@pytest.mark.parametrize("regions", [
    (),
    (DetectedRegion(RegionType.PLATE, 2, 0, 2, 2),),
    (
        DetectedRegion(RegionType.PLATE, 0, 0, 2, 2),
        DetectedRegion(RegionType.PLATE, 4, 2, 4, 2),
    ),
])
def test_detected_single_multiple_and_empty_use_anonymization_without_identity(flow, regions):
    flow.results = (
        FrameDetections.success(0, regions),
        FrameDetections.success(1),
        FrameDetections.success(2, regions),
    )
    assert upload(flow) == 0
    assert flow.mpc.identity_calls == []
    assert flow.mpc.payloads == [flow.preprocessor.processed.payload]
    assert flow.mpc.plans[0].detections_complete
    assert len(flow.mpc.plans[0].regions) == len(regions) * 2
    assert flow.reconstructor.metadata == [flow.preprocessor.processed.metadata]
    flow.mpc.plans[0].validate_result(
        flow.preprocessor.processed.payload, flow.reconstructor.payloads[0]
    )
    assert flow.coordinator.calls[-2:] == ["complete", "cleanup"]
    assert flow.parties.calls[-3:] == ["cleanup:0", "cleanup:1", "cleanup:2"]


def test_detection_uses_original_bgr_in_processed_temporal_order_and_actual_resize(flow):
    flow.results = tuple(
        FrameDetections.success(i, (DetectedRegion(RegionType.PLATE, 2, 0, 2, 2),))
        for i in range(3)
    )
    # No registration is allowed until every selected frame has been detected.
    flow.coordinator.on_register = lambda: _assert_detection_complete(flow)
    assert upload(flow) == 0
    assert [index for _, index in flow.detector_calls] == [0, 1, 2]
    for (image, _), original in zip(flow.detector_calls, flow.preprocessor.frames):
        assert image.shape == (4, 8, 3)
        assert image.dtype == np.uint8
        assert np.array_equal(image, original)
        assert image[0, 0, 0] != image[0, 0, 1]  # Actual color, not repeated gray.
    assert flow.mpc.plans[0].regions == tuple(
        SensitiveRegion(i, RegionType.PLATE, 1, 0, 1, 1) for i in range(3)
    )
    # Original private pixels have not been zeroed before crossing the boundary.
    assert all(flow.mpc.payloads[0])
    result = np.frombuffer(flow.reconstructor.payloads[0], np.uint8).reshape(3, 2, 4)
    source = np.frombuffer(flow.mpc.payloads[0], np.uint8).reshape(3, 2, 4)
    assert (result[:, 0, 1] == 0).all()
    assert np.array_equal(result[:, 1, :], source[:, 1, :])


def _assert_detection_complete(flow):
    assert len(flow.detector_calls) == 3


@pytest.mark.parametrize("threshold", [None, 0.45, 0.8])
def test_plate_threshold_has_independent_point_six_default_and_can_be_changed(flow, threshold):
    options = {} if threshold is None else {"plate_confidence_threshold": threshold}
    assert upload(flow, **options) == 0
    assert flow.detector_options == [{"confidence_threshold": 0.6 if threshold is None else threshold}]


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf"), True])
def test_invalid_plate_threshold_is_rejected_before_any_job(flow, threshold):
    with pytest.raises((ValueError, cli.ClientInputError)):
        upload(flow, plate_confidence_threshold=threshold)
    assert_no_remote_work(flow)


@pytest.mark.parametrize("frame_index", [0, 1, 2])
def test_detector_failure_on_any_frame_cannot_be_empty_success_or_partial_upload(flow, frame_index):
    flow.results = tuple(
        FrameDetections.failure(i, "inference failed") if i == frame_index
        else FrameDetections.success(i)
        for i in range(3)
    )
    with pytest.raises(DetectorExecutionError):
        upload(flow)
    assert_no_remote_work(flow)


def test_model_load_error_stops_before_registration(flow):
    flow.detector_error = PlateDetectionError("ONNX unavailable")
    with pytest.raises(PlateDetectionError):
        upload(flow)
    assert_no_remote_work(flow)


@pytest.mark.parametrize("error", [PlateDetectionError("runtime"), ValueError("malformed box")])
def test_direct_detector_exception_does_not_fallback_to_identity(flow, error):
    def fail(original_bgr, frame_index):
        raise error
    flow.results = fail
    with pytest.raises(type(error)):
        upload(flow)
    assert_no_remote_work(flow)


def test_nonfinite_region_cannot_be_transmitted(flow):
    def nonfinite(original_bgr, frame_index):
        return FrameDetections.success(frame_index, (
            DetectedRegion(RegionType.PLATE, float("nan"), 1, 2, 2),
        ))
    flow.results = nonfinite
    with pytest.raises(ValueError):
        upload(flow)
    assert_no_remote_work(flow)


@pytest.mark.parametrize("indices", [(), (0, 1), (0, 1, 2, 3), (0, 0, 2), (0, 2, 1)])
def test_missing_extra_or_misordered_callbacks_never_start_a_job(flow, indices):
    flow.preprocessor.indices = indices
    # Return whichever index the hook declares, rather than relying on list indexing.
    flow.results = lambda original, index: FrameDetections.success(index)
    with pytest.raises(ValueError):
        upload(flow)
    assert_no_remote_work(flow)


@pytest.mark.parametrize("bad_result", [None, (), FrameDetections.success(7)])
def test_detector_must_return_t024_result_for_the_same_processed_index(flow, bad_result):
    flow.results = lambda original, index: bad_result
    with pytest.raises(ValueError):
        upload(flow)
    assert_no_remote_work(flow)


@pytest.mark.parametrize("source_size", [None, (0, 4), (8, 0)])
def test_missing_or_invalid_original_dimensions_cannot_guess_region_scaling(flow, source_size):
    flow.preprocessor.processed = PreprocessedVideo(
        flow.preprocessor.processed.payload,
        flow.preprocessor.processed.metadata,
        source_size=source_size,
    )
    with pytest.raises(ValueError):
        upload(flow)
    assert_no_remote_work(flow)


def test_payload_plan_size_mismatch_cannot_start_a_job(flow):
    flow.preprocessor.processed = PreprocessedVideo(
        b"too short", flow.preprocessor.processed.metadata, source_size=(8, 4)
    )
    with pytest.raises(ValueError):
        upload(flow)
    assert_no_remote_work(flow)


@pytest.mark.parametrize("options", [
    {"processing_mode": "identity"},
    {"regions_json": Path("manual.json")},
    {"anonymization_plan": "manual plan must not be mixed with detector"},
])
def test_detector_is_explicit_anonymize_mode_and_exclusive_with_manual_regions(flow, options):
    with pytest.raises(cli.ClientInputError):
        upload(flow, **options)
    assert flow.detector_calls == []
    assert_no_remote_work(flow)


def test_corrupt_mpc_output_marks_failure_and_is_not_reconstructed(flow):
    flow.results = tuple(
        FrameDetections.success(i, (DetectedRegion(RegionType.PLATE, 2, 0, 2, 2),))
        for i in range(3)
    )
    flow.mpc.corrupt = True
    with pytest.raises(ValueError):
        upload(flow)
    assert flow.mpc.identity_calls == []
    assert flow.reconstructor.payloads == []
    assert "fail" in flow.coordinator.calls
    assert "cleanup" in flow.coordinator.calls
    assert "complete" not in flow.coordinator.calls


def _patch_main(flow, monkeypatch):
    monkeypatch.setattr(cli, "CoordinatorClient", lambda url: flow.coordinator)
    monkeypatch.setattr(cli, "PartyClient", lambda: flow.parties)
    monkeypatch.setattr(cli, "VideoPreprocessor", lambda **options: flow.preprocessor)
    monkeypatch.setattr(cli, "MpcExternalClient", lambda: flow.mpc)
    monkeypatch.setattr(cli, "VideoReconstructor", lambda: flow.reconstructor)
    monkeypatch.setattr(cli, "YuNet", lambda **options: flow.model)
    monkeypatch.setenv("DELTA_OUTPUT_DIR", str(flow.output_root))


@pytest.mark.parametrize("flags,expected_threshold", [
    ([], 0.6), (["--plate-confidence-threshold", "0.45"], 0.45),
])
def test_main_forwards_explicit_detector_flag_and_threshold(flow, monkeypatch, flags, expected_threshold):
    _patch_main(flow, monkeypatch)
    assert cli.main([
        "upload", str(flow.source), "--processing-mode", "anonymize", "--detect-plates", *flags
    ]) == 0
    assert flow.detector_options == [{"confidence_threshold": expected_threshold}]
    assert len(flow.mpc.plans) == 1
    assert flow.mpc.identity_calls == []


@pytest.mark.parametrize("flags", [
    ["--detect-plates"],
    ["--processing-mode", "anonymize", "--detect-plates", "--regions-json", "manual.json"],
    ["--processing-mode", "anonymize", "--detect-plates", "--plate-confidence-threshold", "nan"],
])
def test_invalid_cli_combinations_return_error_without_remote_calls(flow, monkeypatch, flags):
    _patch_main(flow, monkeypatch)
    assert cli.main(["upload", str(flow.source), *flags]) == 1
    assert_no_remote_work(flow)


def test_main_handles_model_initialization_failure_as_clear_exit_one(flow, monkeypatch, capsys):
    _patch_main(flow, monkeypatch)
    flow.detector_error = PlateDetectionError("model unavailable")
    assert cli.main([
        "upload", str(flow.source), "--processing-mode", "anonymize", "--detect-plates"
    ]) == 1
    assert "model unavailable" in capsys.readouterr().err
    assert_no_remote_work(flow)


def test_identity_remains_unchanged_and_does_not_load_plate_detector(flow, monkeypatch):
    _patch_main(flow, monkeypatch)
    assert cli.main(["upload", str(flow.source)]) == 0
    assert flow.preprocessor.calls == [None]
    assert flow.detector_options == flow.detector_calls == []
    assert flow.mpc.plans == []
    assert flow.mpc.identity_calls == [flow.preprocessor.processed.payload]
    assert flow.reconstructor.payloads == [flow.preprocessor.processed.payload]


def test_detect_faces_prepares_face_mask_and_sends_it_to_mpc(flow, monkeypatch):
    _patch_main(flow, monkeypatch)
    flow.preprocessor.processed = PreprocessedVideo(
        flow.preprocessor.processed.payload,
        flow.preprocessor.processed.metadata,
        source_size=(8, 4),
        regions=(FaceRegion(1, 1, 0, 2, 1, 0.99),),
    )

    assert cli.main([
        "upload", str(flow.source), "--processing-mode", "anonymize", "--detect-faces"
    ]) == 0
    assert len(flow.mpc.plans) == 1
    assert [(r.frame_index, r.region_type.value, r.x, r.y, r.width, r.height)
            for r in flow.mpc.plans[0].regions] == [(1, "face", 1, 0, 2, 1)]
    assert flow.mpc.payloads == [flow.preprocessor.processed.payload]
    expected = bytearray(flow.preprocessor.processed.payload)
    expected[9:11] = b"\x00\x00"
    assert flow.reconstructor.payloads == [bytes(expected)]
    assert flow.mpc.identity_calls == []


@pytest.mark.parametrize("has_faces,has_plates", [
    (False, False), (True, False), (False, True), (True, True),
])
def test_both_detectors_merge_regions_with_empty_frames_and_overlapping_boxes(
    flow, has_faces, has_plates,
):
    faces = (
        FaceRegion(0, 1, 0, 2, 1, 0.99),
        FaceRegion(2, 0, 1, 1, 1, 0.98),
    ) if has_faces else ()
    flow.preprocessor.processed = replace(flow.preprocessor.processed, regions=faces)
    flow.results = tuple(
        FrameDetections.success(index, (
            DetectedRegion(RegionType.PLATE, 2, 0, 2, 2),
        ) if has_plates and index != 1 else ())
        for index in range(3)
    )
    flow.coordinator.on_register = lambda: _assert_detection_complete(flow)

    assert upload(flow, detect_faces=True) == 0

    assert flow.preprocessor.calls and len(flow.preprocessor.calls) == 1
    assert [index for _, index in flow.detector_calls] == [0, 1, 2]
    assert flow.mpc.payloads == [flow.preprocessor.processed.payload]
    assert flow.mpc.identity_calls == []
    plan = flow.mpc.plans[0]
    assert plan.detections_complete
    expected_regions = []
    for index in range(3):
        if has_plates and index != 1:
            expected_regions.append(SensitiveRegion(index, RegionType.PLATE, 1, 0, 1, 1))
        expected_regions.extend(
            SensitiveRegion(index, RegionType.FACE, face.x, face.y, face.width, face.height)
            for face in faces if face.frame_id == index
        )
    assert plan.regions == tuple(expected_regions)
    expected_pixels = np.frombuffer(flow.preprocessor.processed.payload, np.uint8).reshape(3, 2, 4).copy()
    if has_plates:
        expected_pixels[[0, 2], 0, 1] = 0
    if has_faces:
        expected_pixels[0, 0, 1:3] = 0
        expected_pixels[2, 1, 0] = 0
    assert flow.reconstructor.payloads == [expected_pixels.tobytes()]


@pytest.mark.parametrize("failed_index", [0, 1, 2])
def test_combined_plate_failure_never_uploads_partial_face_detections(flow, failed_index):
    flow.preprocessor.processed = replace(
        flow.preprocessor.processed, regions=(FaceRegion(0, 1, 0, 2, 1, 0.99),),
    )
    flow.results = tuple(
        FrameDetections.failure(index, "plate inference failed") if index == failed_index
        else FrameDetections.success(index)
        for index in range(3)
    )
    with pytest.raises(DetectorExecutionError):
        upload(flow, detect_faces=True)
    assert_no_remote_work(flow)


@pytest.mark.parametrize("frame_id", [3, 100])
def test_combined_faces_with_out_of_payload_index_cannot_start_job(flow, frame_id):
    flow.preprocessor.processed = replace(
        flow.preprocessor.processed, regions=(FaceRegion(frame_id, 1, 0, 2, 1, 0.99),),
    )
    with pytest.raises(cli.ClientInputError, match="frame_id"):
        upload(flow, detect_faces=True)
    assert_no_remote_work(flow)


@pytest.mark.parametrize("manual", [
    {"regions_json": Path("manual.json")},
    {"anonymization_plan": "manual plan"},
])
def test_both_detectors_remain_exclusive_with_manual_region_sources(flow, manual):
    with pytest.raises(cli.ClientInputError):
        upload(flow, detect_faces=True, **manual)
    assert flow.detector_calls == []
    assert flow.preprocessor.calls == []
    assert_no_remote_work(flow)


def test_main_accepts_both_detector_flags_and_reports_both(flow, monkeypatch, capsys):
    _patch_main(flow, monkeypatch)
    assert cli.main([
        "upload", str(flow.source), "--processing-mode", "anonymize",
        "--detect-faces", "--detect-plates",
    ]) == 0
    assert "rostros y placas detectados" in capsys.readouterr().out
    assert len(flow.mpc.plans) == 1
    assert flow.mpc.identity_calls == []


@pytest.mark.parametrize("fail_face", [False, True])
def test_both_detectors_share_real_original_samples_before_resize_and_fail_closed(
    flow, tmp_path, fail_face,
):
    source = tmp_path / "native_bgr.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 30, (32, 16))
    assert writer.isOpened()
    for index in range(6):
        frame = np.empty((16, 32, 3), np.uint8)
        frame[:] = (20 + index, 90 + index, 190 + index)
        writer.write(frame)
    writer.release()
    face_inputs = []

    class FaceDetector:
        def setInputSize(self, size):
            assert size == [32, 16]

        def infer(self, original_bgr):
            face_inputs.append(original_bgr.copy())
            if fail_face:
                raise ValueError("face inference failed")
            return np.array([[8, 0, 8, 8, 0.99]])

    flow.source = source
    flow.model = FaceDetector()
    flow.preprocessor = VideoPreprocessor(max_width=16, max_height=8, target_fps=10)
    flow.results = lambda original, index: FrameDetections.success(index, (
        DetectedRegion(RegionType.PLATE, 16, 8, 16, 8),
    ))
    if fail_face:
        with pytest.raises(VideoPreprocessingError):
            upload(flow, detect_faces=True)
        assert len(flow.detector_calls) == len(face_inputs) == 1
        assert_no_remote_work(flow)
        return

    assert upload(flow, detect_faces=True) == 0
    assert len(face_inputs) == len(flow.detector_calls) == 2
    for face_input, (plate_input, _) in zip(face_inputs, flow.detector_calls):
        assert face_input.shape == plate_input.shape == (16, 32, 3)
        assert np.array_equal(face_input, plate_input)
        assert face_input[0, 0, 0] != face_input[0, 0, 1]
    assert [index for _, index in flow.detector_calls] == [0, 1]
    assert face_inputs[0].mean() < face_inputs[1].mean()
    plan = flow.mpc.plans[0]
    assert (plan.width, plan.height, plan.frame_count) == (16, 8, 2)
    assert plan.regions == tuple(
        region for index in range(2) for region in (
            SensitiveRegion(index, RegionType.PLATE, 8, 4, 8, 4),
            SensitiveRegion(index, RegionType.FACE, 4, 0, 4, 4),
        )
    )
    assert all(flow.mpc.payloads[0])
    plan.validate_result(flow.mpc.payloads[0], flow.reconstructor.payloads[0])


def test_manual_json_remains_separate_and_does_not_load_detector(flow, monkeypatch, tmp_path):
    _patch_main(flow, monkeypatch)
    path = tmp_path / "manual.json"
    path.write_text(json.dumps({"width": 4, "height": 2, "frames": [
        {"frame_index": i, "status": "success", "regions": []} for i in range(3)
    ]}), encoding="utf-8")
    assert cli.main([
        "upload", str(flow.source), "--processing-mode", "anonymize", "--regions-json", str(path)
    ]) == 0
    assert flow.preprocessor.calls == [None]
    assert flow.detector_options == flow.detector_calls == []
    assert flow.mpc.identity_calls == []
    assert flow.mpc.plans[0] == prepare_anonymization(
        tuple(FrameDetections.success(i) for i in range(3)),
        source_size=(4, 2), target_size=(4, 2), frame_count=3,
    )


def test_cli_rejects_nonnumeric_plate_threshold():
    with pytest.raises(SystemExit) as error:
        cli.build_parser().parse_args([
            "upload", "video2.mp4", "--plate-confidence-threshold", "high"
        ])
    assert error.value.code == 2
