"""Endpoints del plano de control del MVP MPC."""

from uuid import UUID

from fastapi import APIRouter, HTTPException

from coordinator.models.video import (
    FailureRequest,
    PartyReadyRequest,
    VideoPrepareRequest,
    VideoRegistrationRequest,
    VideoRegistrationResponse,
    VideoStateResponse,
)
from coordinator.services.video_service import (
    ActiveJobConflictError,
    InvalidTransitionError,
    UnsupportedVideoFormatError,
    VideoNotFoundError,
    VideoService,
    load_party_urls,
)


router = APIRouter(prefix="/api/v1/videos", tags=["videos"])
video_service = VideoService(load_party_urls())


def _translate_control_error(exc: Exception) -> HTTPException:
    if isinstance(exc, VideoNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, (ActiveJobConflictError, InvalidTransitionError)):
        return HTTPException(status_code=409, detail=str(exc))
    return HTTPException(status_code=422, detail=str(exc))


@router.post("", response_model=VideoRegistrationResponse, status_code=201)
async def register_video(payload: VideoRegistrationRequest) -> VideoRegistrationResponse:
    """Registra solo metadata nominal; jamás acepta bytes del MP4."""

    try:
        return video_service.register_video(payload.filename, payload.format)
    except UnsupportedVideoFormatError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except (ActiveJobConflictError, ValueError) as exc:
        raise _translate_control_error(exc) from exc


@router.get("/{video_id}", response_model=VideoStateResponse)
async def get_video(video_id: UUID) -> VideoStateResponse:
    try:
        return video_service.get_video(video_id)
    except VideoNotFoundError as exc:
        raise _translate_control_error(exc) from exc


@router.post("/{video_id}/prepare", response_model=VideoStateResponse)
async def prepare_video(video_id: UUID, payload: VideoPrepareRequest) -> VideoStateResponse:
    try:
        return video_service.prepare(video_id, payload.metadata)
    except (VideoNotFoundError, InvalidTransitionError, ValueError) as exc:
        raise _translate_control_error(exc) from exc


@router.post("/{video_id}/parties/{party_id}/ready", response_model=VideoStateResponse)
async def party_ready(
    video_id: UUID,
    party_id: int,
    payload: PartyReadyRequest,
) -> VideoStateResponse:
    del payload
    try:
        return video_service.mark_party_ready(video_id, party_id)
    except (VideoNotFoundError, InvalidTransitionError, ValueError) as exc:
        raise _translate_control_error(exc) from exc


@router.post("/{video_id}/start", response_model=VideoStateResponse)
async def start_video(video_id: UUID) -> VideoStateResponse:
    try:
        return video_service.start(video_id)
    except (VideoNotFoundError, InvalidTransitionError) as exc:
        raise _translate_control_error(exc) from exc


@router.post("/{video_id}/complete", response_model=VideoStateResponse)
async def complete_video(video_id: UUID) -> VideoStateResponse:
    try:
        return video_service.complete(video_id)
    except (VideoNotFoundError, InvalidTransitionError) as exc:
        raise _translate_control_error(exc) from exc


@router.post("/{video_id}/fail", response_model=VideoStateResponse)
async def fail_video(video_id: UUID, payload: FailureRequest) -> VideoStateResponse:
    try:
        return video_service.fail(video_id, payload.detail)
    except (VideoNotFoundError, InvalidTransitionError, ValueError) as exc:
        raise _translate_control_error(exc) from exc


@router.post("/{video_id}/cleanup", response_model=VideoStateResponse)
async def cleanup_video(video_id: UUID) -> VideoStateResponse:
    try:
        return video_service.cleanup(video_id)
    except (VideoNotFoundError, InvalidTransitionError) as exc:
        raise _translate_control_error(exc) from exc
