"""Pruebas del ciclo de vida en memoria y exclusión de jobs."""

import pytest

from coordinator.models.video import VideoMetadata, VideoStatus
from coordinator.services.video_service import (
    ActiveJobConflictError,
    InvalidTransitionError,
    VideoService,
)


PARTIES = ("http://p0", "http://p1", "http://p2")
METADATA = VideoMetadata(
    width=64,
    height=48,
    fps=2,
    frame_count=4,
    dtype="uint8",
    channels=1,
    version=2,
)


def test_full_lifecycle_and_cleanup_allows_next_job() -> None:
    service = VideoService(PARTIES)
    registration = service.register_video("video.mp4", "mp4")
    assert registration.status is VideoStatus.REGISTERED

    state = service.prepare(registration.id, METADATA)
    assert state.status is VideoStatus.PREPARING
    for party_id in range(3):
        state = service.mark_party_ready(registration.id, party_id)
    assert state.status is VideoStatus.MPC_READY

    assert service.start(registration.id).status is VideoStatus.PROCESSING
    assert service.complete(registration.id).status is VideoStatus.COMPLETED
    service.cleanup(registration.id)

    second = service.register_video("next.mp4", ".MP4")
    assert second.id != registration.id


def test_rejects_second_job_until_terminal_job_is_cleaned() -> None:
    service = VideoService(PARTIES)
    first = service.register_video("first.mp4", "mp4")

    with pytest.raises(ActiveJobConflictError):
        service.register_video("second.mp4", "mp4")

    service.fail(first.id, "test failure")
    with pytest.raises(ActiveJobConflictError):
        service.register_video("second.mp4", "mp4")
    service.cleanup(first.id)
    assert service.register_video("second.mp4", "mp4")


def test_invalid_transitions_are_rejected() -> None:
    service = VideoService(PARTIES)
    job = service.register_video("video.mp4", "mp4")

    with pytest.raises(InvalidTransitionError):
        service.start(job.id)
    with pytest.raises(InvalidTransitionError):
        service.cleanup(job.id)


def test_party_readiness_is_idempotent() -> None:
    service = VideoService(PARTIES)
    job = service.register_video("video.mp4", "mp4")
    service.prepare(job.id, METADATA)

    first = service.mark_party_ready(job.id, 1)
    second = service.mark_party_ready(job.id, 1)

    assert first == second
    assert second.parties["1"].value == "READY"
