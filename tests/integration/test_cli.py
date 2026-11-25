"""Flujo de CLI con límites externos sustituidos."""

from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from client import cli
from client.networking import PartyEndpoint, VideoRegistration, VideoState
from client.preprocessing import PreprocessedVideo, VideoMetadata


@dataclass
class StubCoordinator:
    video_id: str = field(default_factory=lambda: str(uuid4()))
    calls: list[str] = field(default_factory=list)
    prepared_metadata: list[dict] = field(default_factory=list)

    def register_video(self, filename: str, video_format: str) -> VideoRegistration:
        self.calls.append(f"register:{filename}:{video_format}")
        return VideoRegistration(
            self.video_id,
            "REGISTERED",
            (
                PartyEndpoint(0, "http://p0"),
                PartyEndpoint(1, "http://p1"),
                PartyEndpoint(2, "http://p2"),
            ),
        )

    def prepare(self, video_id: str, metadata: dict) -> VideoState:
        self.calls.append("prepare")
        self.prepared_metadata.append(metadata.copy())
        return self._state("PREPARING")

    def wait_for_status(self, video_id: str, expected: str) -> VideoState:
        self.calls.append(f"wait:{expected}")
        return self._state(expected)

    def start(self, video_id: str) -> VideoState:
        self.calls.append("start")
        return self._state("PROCESSING")

    def complete(self, video_id: str) -> VideoState:
        self.calls.append("complete")
        return self._state("COMPLETED")

    def fail(self, video_id: str, detail: str) -> VideoState:
        self.calls.append("fail")
        return self._state("FAILED")

    def cleanup(self, video_id: str) -> VideoState:
        self.calls.append("cleanup")
        return self._state("COMPLETED")

    def _state(self, status: str) -> VideoState:
        return VideoState(self.video_id, status, {})


@dataclass
class StubPartyClient:
    calls: list[str] = field(default_factory=list)
    prepared_metadata: list[dict] = field(default_factory=list)

    def prepare_job(self, party_id: int, url: str, video_id: str, metadata: dict) -> None:
        self.calls.append(f"prepare:{party_id}")
        self.prepared_metadata.append(metadata.copy())

    def start_job(self, party_id: int, url: str, video_id: str) -> None:
        self.calls.append(f"start:{party_id}")

    def complete_job(self, party_id: int, url: str, video_id: str) -> None:
        self.calls.append(f"complete:{party_id}")

    def cleanup_job(self, party_id: int, url: str, video_id: str) -> None:
        self.calls.append(f"cleanup:{party_id}")

    def fail_job(self, party_id: int, url: str, video_id: str, detail: str) -> None:
        self.calls.append(f"fail:{party_id}")


class FakeYuNet:
    def setInputSize(self, size: list[int]) -> None:
        pass

    def infer(self, image: np.ndarray) -> np.ndarray:
        return np.empty((0, 15), dtype=np.float32)


class StubPreprocessor:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, FakeYuNet]] = []

    def process(self, path: Path, model: FakeYuNet) -> PreprocessedVideo:
        self.calls.append((path, model))
        return PreprocessedVideo(
            b"\x00\x01\x02\x03",
            VideoMetadata(1, 2, 2, 1, 2.0, "uint8", 2),
        )


class StubMpc:
    def __init__(self, corrupt: bool = False) -> None:
        self.corrupt = corrupt
        self.payloads: list[bytes] = []

    def process_identity(self, payload: bytes) -> bytes:
        self.payloads.append(payload)
        return b"bad!" if self.corrupt else payload


class InterruptingMpc:
    def process_identity(self, payload: bytes) -> bytes:
        raise KeyboardInterrupt


class StubReconstructor:
    def __init__(self) -> None:
        self.payloads: list[bytes] = []

    def reconstruct(
        self, payload: bytes, metadata: VideoMetadata, video_id: str, output_root: Path
    ) -> Path:
        self.payloads.append(payload)
        destination = Path(output_root) / video_id / "reconstructed.mp4"
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b"mp4")
        return destination


def _video(tmp_path: Path) -> Path:
    path = tmp_path / "input.mp4"
    path.write_bytes(b"stub")
    return path


def test_success_runs_mpc_reconstructs_and_cleans(tmp_path: Path) -> None:
    coordinator = StubCoordinator()
    parties = StubPartyClient()
    preprocessor = StubPreprocessor()
    mpc = StubMpc()
    reconstructor = StubReconstructor()
    model = FakeYuNet()
    source = _video(tmp_path)

    result = cli.upload_video(
        source,
        coordinator,
        parties,
        preprocessor,
        mpc,
        reconstructor,
        model=model,
        output_root=tmp_path / "output",
    )

    assert result == 0
    assert (tmp_path / "output" / coordinator.video_id / "reconstructed.mp4").is_file()
    assert coordinator.calls == [
        "register:input.mp4:mp4",
        "prepare",
        "wait:MPC_READY",
        "start",
        "complete",
        "cleanup",
    ]
    assert parties.calls == [
        "prepare:0", "prepare:1", "prepare:2",
        "start:0", "start:1", "start:2",
        "complete:0", "complete:1", "complete:2",
        "cleanup:0", "cleanup:1", "cleanup:2",
    ]
    assert preprocessor.calls == [(source, model)]
    assert mpc.payloads == [b"\x00\x01\x02\x03"]
    assert reconstructor.payloads == mpc.payloads
    public_metadata = VideoMetadata(1, 2, 2, 1, 2.0, "uint8", 2).as_dict()
    assert coordinator.prepared_metadata == [public_metadata]
    assert parties.prepared_metadata == [public_metadata] * 3


def test_mpc_mismatch_marks_failed_and_cleans(tmp_path: Path) -> None:
    coordinator = StubCoordinator()
    parties = StubPartyClient()

    with pytest.raises(cli.MpcResultMismatchError):
        cli.upload_video(
            _video(tmp_path),
            coordinator,
            parties,
            StubPreprocessor(),
            StubMpc(corrupt=True),
            StubReconstructor(),
            model=FakeYuNet(),
            output_root=tmp_path / "output",
        )

    assert "fail" in coordinator.calls
    assert "cleanup" in coordinator.calls
    assert {call for call in parties.calls if call.startswith("fail:")} == {
        "fail:0", "fail:1", "fail:2"
    }


def test_keyboard_interrupt_marks_failed_and_cleans(tmp_path: Path) -> None:
    coordinator = StubCoordinator()
    parties = StubPartyClient()

    with pytest.raises(KeyboardInterrupt):
        cli.upload_video(
            _video(tmp_path),
            coordinator,
            parties,
            StubPreprocessor(),
            InterruptingMpc(),
            StubReconstructor(),
            model=FakeYuNet(),
            output_root=tmp_path / "output",
        )

    assert "fail" in coordinator.calls
    assert "cleanup" in coordinator.calls
    assert "cleanup:0" in parties.calls


def test_cleanup_is_attempted_when_fail_notification_is_rejected() -> None:
    class RejectingCoordinator(StubCoordinator):
        def fail(self, video_id: str, detail: str) -> VideoState:
            self.calls.append("fail")
            raise cli.CoordinatorClientError("already completed")

    class RejectingParties(StubPartyClient):
        def fail_job(
            self,
            party_id: int,
            url: str,
            video_id: str,
            detail: str,
        ) -> None:
            self.calls.append(f"fail:{party_id}")
            raise cli.PartyClientError("already completed")

    coordinator = RejectingCoordinator()
    parties = RejectingParties()
    registration = coordinator.register_video("input.mp4", "mp4")

    cli._best_effort_failure_cleanup(
        registration,
        coordinator,
        parties,
        "response lost",
    )

    assert "cleanup" in coordinator.calls
    assert {call for call in parties.calls if call.startswith("cleanup:")} == {
        "cleanup:0", "cleanup:1", "cleanup:2"
    }


def test_invalid_input_is_rejected_before_network(tmp_path: Path) -> None:
    with pytest.raises(cli.ClientInputError):
        cli.upload_video(
            tmp_path / "missing.mp4",
            StubCoordinator(),
            StubPartyClient(),
            StubPreprocessor(),
            StubMpc(),
            StubReconstructor(),
            model=FakeYuNet(),
        )


@pytest.mark.parametrize(
    ("flags", "environment", "expected_options", "expected_size"),
    [
        ([], {}, {"max_width": None, "max_height": None, "target_fps": 30}, (640, 480)),
        (["--mpc-max-width", "480"], {},
         {"max_width": 480, "max_height": None, "target_fps": 30}, (480, 360)),
        (["--mpc-max-width", "640", "--mpc-max-height", "480"], {},
         {"max_width": 640, "max_height": 480, "target_fps": 30}, (640, 480)),
        ([], {"MPC_MAX_WIDTH": "480"},
         {"max_width": None, "max_height": None, "target_fps": 30}, (480, 360)),
        ([], {"MPC_MAX_WIDTH": "640", "MPC_MAX_HEIGHT": "360"},
         {"max_width": None, "max_height": None, "target_fps": 30}, (480, 360)),
        (["--mpc-fps", "5"], {},
         {"max_width": None, "max_height": None, "target_fps": 5}, (640, 480)),
    ],
)
def test_main_forwards_resolution_and_fps_options_to_preprocessing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    flags: list[str],
    environment: dict[str, str],
    expected_options: dict[str, int | float | None],
    expected_size: tuple[int, int],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MPC_MAX_WIDTH", raising=False)
    monkeypatch.delenv("MPC_MAX_HEIGHT", raising=False)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    coordinator = StubCoordinator()
    parties = StubPartyClient()
    model = FakeYuNet()
    preprocessor = StubPreprocessor()
    actual_options = []
    actual_sizes = []
    original_preprocessor = cli.VideoPreprocessor

    def make_preprocessor(**options) -> StubPreprocessor:
        actual_options.append(options)
        configured = original_preprocessor(**options)
        actual_sizes.append(configured._scaled_size(640, 480))
        return preprocessor

    monkeypatch.setattr(cli, "VideoPreprocessor", make_preprocessor)
    monkeypatch.setattr(cli, "CoordinatorClient", lambda url: coordinator)
    monkeypatch.setattr(cli, "PartyClient", lambda: parties)
    monkeypatch.setattr(cli, "MpcExternalClient", StubMpc)
    monkeypatch.setattr(cli, "VideoReconstructor", StubReconstructor)
    monkeypatch.setattr(cli, "YuNet", lambda **options: model)
    monkeypatch.setenv("DELTA_OUTPUT_DIR", str(tmp_path / "output"))
    source = _video(tmp_path)

    assert cli.main(["upload", str(source), *flags]) == 0
    assert actual_options == [expected_options]
    assert actual_sizes == [expected_size]
    assert preprocessor.calls == [(source, model)]


@pytest.mark.parametrize(
    "flags",
    [["--mpc-max-width", "0"], ["--mpc-max-height", "0"],
     ["--mpc-max-width", "641"], ["--mpc-max-height", "481"],
     ["--mpc-fps", "0"], ["--mpc-fps", "11"],
     ["--mpc-fps", "nan"], ["--mpc-fps", "inf"]],
)
def test_invalid_profile_is_rejected_before_network_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flags: list[str]
) -> None:
    monkeypatch.delenv("MPC_MAX_WIDTH", raising=False)
    monkeypatch.delenv("MPC_MAX_HEIGHT", raising=False)
    coordinator = StubCoordinator()
    parties = StubPartyClient()
    monkeypatch.setattr(cli, "CoordinatorClient", lambda url: coordinator)
    monkeypatch.setattr(cli, "PartyClient", lambda: parties)
    monkeypatch.setattr(cli, "MpcExternalClient", StubMpc)
    monkeypatch.setattr(cli, "YuNet", lambda **options: FakeYuNet())

    assert cli.main(["upload", str(_video(tmp_path)), *flags]) == 1
    assert coordinator.calls == []
    assert parties.calls == []


@pytest.mark.parametrize("option", ["--mpc-max-width", "--mpc-fps"])
def test_profile_flags_reject_non_numeric_input(option: str) -> None:
    with pytest.raises(SystemExit) as error:
        cli.build_parser().parse_args(["upload", "input.mp4", option, "large"])
    assert error.value.code == 2


class StubAnonymizingMpc(StubMpc):
    def __init__(self, result: bytes = b"\x00\x00\x02\x03") -> None:
        super().__init__()
        self.result = result
        self.plans = []

    def process_identity(self, payload: bytes) -> bytes:
        raise AssertionError("Anonimización nunca debe hacer fallback a identidad")

    def process_anonymize(self, payload: bytes, plan) -> bytes:
        self.payloads.append(payload)
        self.plans.append(plan)
        return self.result


def _anonymization_plan():
    from client.anonymization import prepare_anonymization
    from client.regions import DetectedRegion, FrameDetections, RegionType
    return prepare_anonymization(
        (FrameDetections.success(0, (DetectedRegion(RegionType.FACE, 1, 0, 1, 1),)),),
        source_size=(2, 2), target_size=(2, 2), frame_count=1,
    )


def test_anonymize_mode_sends_original_pixels_and_reconstructs_only_validated_result(tmp_path: Path) -> None:
    coordinator, parties = StubCoordinator(), StubPartyClient()
    mpc, reconstructor = StubAnonymizingMpc(), StubReconstructor()
    plan = _anonymization_plan()

    assert cli.upload_video(
        _video(tmp_path), coordinator, parties, StubPreprocessor(), mpc,
        reconstructor, model=FakeYuNet(), output_root=tmp_path / "output",
        processing_mode="anonymize", anonymization_plan=plan,
    ) == 0
    assert mpc.payloads == [b"\x00\x01\x02\x03"]
    assert mpc.plans == [plan]
    assert reconstructor.payloads == [b"\x00\x00\x02\x03"]
    assert coordinator.calls[-2:] == ["complete", "cleanup"]


@pytest.mark.parametrize("invalid_result", [b"\x00\x01\x02\x03", b"\x00\x00\x09\x03", b"short"])
def test_bad_anonymization_result_fails_cleans_and_never_reconstructs(tmp_path: Path, invalid_result: bytes) -> None:
    coordinator, parties = StubCoordinator(), StubPartyClient()
    mpc, reconstructor = StubAnonymizingMpc(invalid_result), StubReconstructor()
    with pytest.raises(ValueError):
        cli.upload_video(
            _video(tmp_path), coordinator, parties, StubPreprocessor(), mpc,
            reconstructor, model=FakeYuNet(), output_root=tmp_path / "output",
            processing_mode="anonymize", anonymization_plan=_anonymization_plan(),
        )
    assert reconstructor.payloads == []
    assert "fail" in coordinator.calls and "cleanup" in coordinator.calls
    assert "complete" not in coordinator.calls


@pytest.mark.parametrize("options", [
    {"processing_mode": "anonymize"},
    {"processing_mode": "unknown"},
    {"processing_mode": "identity", "anonymization_plan": "not allowed"},
    {"processing_mode": "anonymize", "anonymization_plan": "invalid plan"},
    {"processing_mode": "anonymize", "anonymization_plan": "invalid", "regions_json": Path("both.json")},
])
def test_unsafe_mode_or_missing_plan_is_rejected_before_any_job(tmp_path: Path, options: dict) -> None:
    coordinator, parties, mpc = StubCoordinator(), StubPartyClient(), StubAnonymizingMpc()
    with pytest.raises(cli.ClientInputError):
        cli.upload_video(
            _video(tmp_path), coordinator, parties, StubPreprocessor(), mpc,
            StubReconstructor(), model=FakeYuNet(), **options,
        )
    assert coordinator.calls == parties.calls == mpc.payloads == []


def test_plan_must_match_actual_metadata_not_just_total_pixel_count(tmp_path: Path) -> None:
    from client.anonymization import prepare_anonymization
    from client.regions import FrameDetections
    coordinator, parties, mpc = StubCoordinator(), StubPartyClient(), StubAnonymizingMpc()
    with pytest.raises(cli.ClientInputError):
        cli.upload_video(
            _video(tmp_path), coordinator, parties, StubPreprocessor(), mpc,
            StubReconstructor(), model=FakeYuNet(), processing_mode="anonymize",
            anonymization_plan=prepare_anonymization(
                (FrameDetections.success(0),), source_size=(1, 4), target_size=(1, 4), frame_count=1
            ),
        )
    assert coordinator.calls == parties.calls == mpc.payloads == []


def test_json_regions_are_explicit_complete_and_prepared_before_register(tmp_path: Path) -> None:
    import json
    path = tmp_path / "regions.json"
    path.write_text(json.dumps({"width": 2, "height": 2, "frames": [
        {"frame_index": 0, "status": "success", "regions": [
            {"region_type": "face", "x": 1, "y": 0, "width": 1, "height": 1}
        ]}
    ]}), encoding="utf-8")
    coordinator, parties, mpc = StubCoordinator(), StubPartyClient(), StubAnonymizingMpc()
    assert cli.upload_video(
        _video(tmp_path), coordinator, parties, StubPreprocessor(), mpc,
        StubReconstructor(), model=FakeYuNet(), output_root=tmp_path / "output",
        processing_mode="anonymize", regions_json=path,
    ) == 0
    assert mpc.plans == [_anonymization_plan()]
    assert mpc.payloads == [b"\x00\x01\x02\x03"]


@pytest.mark.parametrize("data", [
    {"width": 2, "height": 2, "frames": []},
    {"width": 2, "height": 2, "frames": [{"frame_index": 0, "status": "failed", "error": "detector fallido"}]},
    {"width": 2, "height": 2, "frames": [{"frame_index": 0, "status": "success"}]},
    {"width": 2, "height": 2, "frames": [{"frame_index": 1, "status": "success", "regions": []}]},
    {"width": 4, "height": 1, "frames": [{"frame_index": 0, "status": "success", "regions": []}]},
    {"width": 2, "height": 2, "frames": [{"frame_index": 0, "status": "unknown", "regions": []}]},
])
def test_failed_incomplete_or_ambiguous_json_never_starts_a_job(tmp_path: Path, data: dict) -> None:
    import json
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    coordinator, parties, mpc = StubCoordinator(), StubPartyClient(), StubAnonymizingMpc()
    with pytest.raises((ValueError, cli.DetectorExecutionError)):
        cli.upload_video(
            _video(tmp_path), coordinator, parties, StubPreprocessor(), mpc,
            StubReconstructor(), model=FakeYuNet(), processing_mode="anonymize",
            regions_json=path,
        )
    assert coordinator.calls == parties.calls == mpc.payloads == []


def test_main_forwards_explicit_anonymization_mode_and_diagnostic_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    source = _video(tmp_path)
    path = tmp_path / "regions.json"
    path.write_text(json.dumps({"width": 2, "height": 2, "frames": [
        {"frame_index": 0, "status": "success", "regions": [
            {"region_type": "face", "x": 1, "y": 0, "width": 1, "height": 1}
        ]}
    ]}), encoding="utf-8")
    mpc = StubAnonymizingMpc()
    monkeypatch.setattr(cli, "CoordinatorClient", lambda url: StubCoordinator())
    monkeypatch.setattr(cli, "PartyClient", StubPartyClient)
    monkeypatch.setattr(cli, "VideoPreprocessor", lambda **options: StubPreprocessor())
    monkeypatch.setattr(cli, "MpcExternalClient", lambda: mpc)
    monkeypatch.setattr(cli, "VideoReconstructor", StubReconstructor)
    monkeypatch.setattr(cli, "YuNet", lambda **options: FakeYuNet())
    monkeypatch.setenv("DELTA_OUTPUT_DIR", str(tmp_path / "output"))
    assert cli.main([
        "upload", str(source), "--processing-mode", "anonymize", "--regions-json", str(path)
    ]) == 0
    assert mpc.payloads == [b"\x00\x01\x02\x03"]
    assert mpc.plans == [_anonymization_plan()]


def test_geometry_only_plan_cannot_bypass_complete_detection_results(tmp_path: Path) -> None:
    from client.anonymization import AnonymizationPlan
    coordinator, parties, mpc = StubCoordinator(), StubPartyClient(), StubAnonymizingMpc()
    with pytest.raises(cli.ClientInputError, match="todos los fotogramas"):
        cli.upload_video(
            _video(tmp_path), coordinator, parties, StubPreprocessor(), mpc,
            StubReconstructor(), model=FakeYuNet(), processing_mode="anonymize",
            anonymization_plan=AnonymizationPlan(2, 2, 1, ()),
        )
    assert coordinator.calls == parties.calls == mpc.payloads == []
