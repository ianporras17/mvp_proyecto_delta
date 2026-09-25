"""Flujo de CLI con límites externos sustituidos."""

from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import pytest

from client import cli
from client.networking import PartyEndpoint, VideoRegistration, VideoState
from client.preprocessing import PreprocessedVideo, VideoMetadata


@dataclass
class StubCoordinator:
    video_id: str = field(default_factory=lambda: str(uuid4()))
    calls: list[str] = field(default_factory=list)

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

    def prepare_job(self, party_id: int, url: str, video_id: str, metadata: dict) -> None:
        self.calls.append(f"prepare:{party_id}")

    def start_job(self, party_id: int, url: str, video_id: str) -> None:
        self.calls.append(f"start:{party_id}")

    def complete_job(self, party_id: int, url: str, video_id: str) -> None:
        self.calls.append(f"complete:{party_id}")

    def cleanup_job(self, party_id: int, url: str, video_id: str) -> None:
        self.calls.append(f"cleanup:{party_id}")

    def fail_job(self, party_id: int, url: str, video_id: str, detail: str) -> None:
        self.calls.append(f"fail:{party_id}")


class StubPreprocessor:
    def process(self, path: Path) -> PreprocessedVideo:
        return PreprocessedVideo(
            b"\x00\x01\x02\x03",
            VideoMetadata(1, 2, 2, 1, 2.0, "uint8", 2),
        )


class StubMpc:
    def __init__(self, corrupt: bool = False) -> None:
        self.corrupt = corrupt

    def process_identity(self, payload: bytes) -> bytes:
        return b"bad!" if self.corrupt else payload


class InterruptingMpc:
    def process_identity(self, payload: bytes) -> bytes:
        raise KeyboardInterrupt


class StubReconstructor:
    def reconstruct(
        self, payload: bytes, metadata: VideoMetadata, video_id: str, output_root: Path
    ) -> Path:
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

    result = cli.upload_video(
        _video(tmp_path),
        coordinator,
        parties,
        StubPreprocessor(),
        StubMpc(),
        StubReconstructor(),
        tmp_path / "output",
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
            tmp_path / "output",
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
            tmp_path / "output",
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
        )
