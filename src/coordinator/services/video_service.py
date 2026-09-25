"""Estado en memoria y transiciones del único job MPC activo."""

from dataclasses import dataclass
import os
from threading import RLock
from typing import Dict, Iterable, Tuple
from urllib.parse import urlparse
from uuid import UUID, uuid4

from coordinator.models.video import (
    PartyEndpoint,
    PartyStatus,
    VideoMetadata,
    VideoRegistrationResponse,
    VideoStateResponse,
    VideoStatus,
)


DEFAULT_PARTY_URLS = (
    "http://localhost:8100",
    "http://localhost:8101",
    "http://localhost:8102",
)
SUPPORTED_FORMAT = "mp4"
TERMINAL_STATUSES = {VideoStatus.COMPLETED, VideoStatus.FAILED}


class VideoNotFoundError(LookupError):
    """El UUID no pertenece a un job registrado."""


class UnsupportedVideoFormatError(ValueError):
    """El cliente intentó registrar un formato distinto de MP4."""


class ActiveJobConflictError(RuntimeError):
    """Ya existe un job que todavía no se limpió."""


class InvalidTransitionError(RuntimeError):
    """La transición solicitada no es válida para el estado actual."""


@dataclass
class _VideoJob:
    id: UUID
    filename: str
    format: str
    status: VideoStatus
    parties: Dict[int, PartyStatus]
    metadata: VideoMetadata | None = None
    error: str | None = None


def load_party_urls(raw_value: str | None = None) -> Tuple[str, str, str]:
    """Carga exactamente tres URLs públicas de Node Agents."""

    configured = raw_value if raw_value is not None else os.getenv("PARTY_URLS")
    values = DEFAULT_PARTY_URLS if configured is None else tuple(
        item.strip().rstrip("/") for item in configured.split(",") if item.strip()
    )
    if len(values) != 3:
        raise RuntimeError("PARTY_URLS debe contener exactamente tres URLs")
    for value in values:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise RuntimeError(f"URL pública de party inválida: {value!r}")
    return values[0], values[1], values[2]


class VideoService:
    """Coordina un job a la vez sin recibir el contenido del video."""

    def __init__(self, party_urls: Iterable[str]) -> None:
        urls = tuple(party_urls)
        if len(urls) != 3:
            raise ValueError("El MVP requiere exactamente tres parties")
        self._party_endpoints = tuple(
            PartyEndpoint(id=party_id, url=url)
            for party_id, url in enumerate(urls)
        )
        self._jobs: Dict[UUID, _VideoJob] = {}
        self._active_job_id: UUID | None = None
        self._lock = RLock()

    def register_video(self, filename: str, video_format: str) -> VideoRegistrationResponse:
        normalized_filename = filename.strip()
        normalized_format = video_format.strip().lower().lstrip(".")
        if not normalized_filename:
            raise ValueError("filename no puede estar vacío")
        if normalized_format != SUPPORTED_FORMAT:
            raise UnsupportedVideoFormatError("Solo se admiten videos MP4")

        with self._lock:
            if self._active_job_id is not None:
                raise ActiveJobConflictError(
                    f"El job {self._active_job_id} sigue activo; complételo y límpielo"
                )
            job = _VideoJob(
                id=uuid4(),
                filename=normalized_filename,
                format=normalized_format,
                status=VideoStatus.REGISTERED,
                parties={endpoint.id: PartyStatus.PENDING for endpoint in self._party_endpoints},
            )
            self._jobs[job.id] = job
            self._active_job_id = job.id

        return VideoRegistrationResponse(
            id=job.id,
            status=job.status,
            parties=list(self._party_endpoints),
        )

    def get_video(self, video_id: UUID) -> VideoStateResponse:
        with self._lock:
            return self._state_response(self._find_job(video_id))

    def prepare(self, video_id: UUID, metadata: VideoMetadata) -> VideoStateResponse:
        with self._lock:
            job = self._find_job(video_id)
            if job.status is VideoStatus.PREPARING and job.metadata == metadata:
                return self._state_response(job)
            self._require_status(job, VideoStatus.REGISTERED)
            job.metadata = metadata
            job.status = VideoStatus.PREPARING
            return self._state_response(job)

    def mark_party_ready(self, video_id: UUID, party_id: int) -> VideoStateResponse:
        with self._lock:
            job = self._find_job(video_id)
            if type(party_id) is not int or party_id not in job.parties:
                raise ValueError(f"party_id desconocido: {party_id}")
            if job.status not in {VideoStatus.PREPARING, VideoStatus.MPC_READY}:
                raise InvalidTransitionError(
                    f"No se puede marcar una party lista desde {job.status.value}"
                )
            job.parties[party_id] = PartyStatus.READY
            if all(value is PartyStatus.READY for value in job.parties.values()):
                job.status = VideoStatus.MPC_READY
            return self._state_response(job)

    def start(self, video_id: UUID) -> VideoStateResponse:
        with self._lock:
            job = self._find_job(video_id)
            if job.status is VideoStatus.PROCESSING:
                return self._state_response(job)
            self._require_status(job, VideoStatus.MPC_READY)
            job.status = VideoStatus.PROCESSING
            job.parties = {party_id: PartyStatus.PROCESSING for party_id in job.parties}
            return self._state_response(job)

    def complete(self, video_id: UUID) -> VideoStateResponse:
        with self._lock:
            job = self._find_job(video_id)
            if job.status is VideoStatus.COMPLETED:
                return self._state_response(job)
            self._require_status(job, VideoStatus.PROCESSING)
            job.status = VideoStatus.COMPLETED
            job.parties = {party_id: PartyStatus.COMPLETED for party_id in job.parties}
            return self._state_response(job)

    def fail(self, video_id: UUID, detail: str) -> VideoStateResponse:
        normalized = detail.strip()
        if not normalized:
            raise ValueError("El detalle del fallo no puede estar vacío")
        with self._lock:
            job = self._find_job(video_id)
            if job.status is VideoStatus.COMPLETED:
                raise InvalidTransitionError("Un job completado no puede marcarse fallido")
            job.status = VideoStatus.FAILED
            job.error = normalized[:1000]
            job.parties = {party_id: PartyStatus.FAILED for party_id in job.parties}
            return self._state_response(job)

    def cleanup(self, video_id: UUID) -> VideoStateResponse:
        with self._lock:
            job = self._find_job(video_id)
            if job.status not in TERMINAL_STATUSES:
                raise InvalidTransitionError(
                    "Solo se puede limpiar un job COMPLETED o FAILED"
                )
            if self._active_job_id == video_id:
                self._active_job_id = None
            return self._state_response(job)

    def _find_job(self, video_id: UUID) -> _VideoJob:
        try:
            return self._jobs[video_id]
        except KeyError as exc:
            raise VideoNotFoundError(f"Video {video_id} no encontrado") from exc

    @staticmethod
    def _require_status(job: _VideoJob, expected: VideoStatus) -> None:
        if job.status is not expected:
            raise InvalidTransitionError(
                f"Transición inválida: se esperaba {expected.value} y está {job.status.value}"
            )

    @staticmethod
    def _state_response(job: _VideoJob) -> VideoStateResponse:
        return VideoStateResponse(
            id=job.id,
            status=job.status,
            parties={str(party_id): value for party_id, value in job.parties.items()},
            metadata=job.metadata,
            error=job.error,
        )
