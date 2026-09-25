"""Estado del único job atendido por un Node Agent."""

from dataclasses import dataclass
from threading import RLock
from typing import Any
from uuid import UUID

import httpx

from party_receiver.runtime import MpcRuntimeManager


class PartyJobNotFoundError(LookupError):
    pass


class PartyJobConflictError(RuntimeError):
    pass


class CoordinatorNotificationError(RuntimeError):
    pass


@dataclass
class _PartyJob:
    video_id: UUID
    metadata: dict[str, Any]
    status: str
    error: str | None = None


class PartyJobService:
    """No almacena pixels; solo enlaza job y runtime persistente."""

    def __init__(
        self,
        party_id: int,
        runtime: MpcRuntimeManager,
        coordinator_url: str,
        notification_timeout: float = 10.0,
    ) -> None:
        self.party_id = party_id
        self.runtime = runtime
        self._coordinator_url = coordinator_url.rstrip("/")
        self._notification_timeout = notification_timeout
        self._active: _PartyJob | None = None
        self._lock = RLock()

    async def prepare(self, video_id: UUID, metadata: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._active is not None and self._active.video_id != video_id:
                raise PartyJobConflictError(
                    f"Party {self.party_id} ya atiende {self._active.video_id}"
                )
        # El accept de ExternalIO tiene timeout. El Node Agent vuelve a iniciar
        # el runtime al preparar cada job si este quedo inactivo entre jobs.
        self.runtime.start()
        self.runtime.require_running()
        with self._lock:
            if self._active is not None and self._active.video_id != video_id:
                raise PartyJobConflictError(
                    f"Party {self.party_id} ya atiende {self._active.video_id}"
                )
            if self._active is None:
                self._active = _PartyJob(video_id, dict(metadata), "READY")
            elif self._active.status != "READY":
                raise PartyJobConflictError(
                    f"No se puede preparar desde {self._active.status}"
                )
        await self._notify_ready(video_id)
        return self._response(self._active)

    def start(self, video_id: UUID) -> dict[str, Any]:
        self.runtime.require_running()
        with self._lock:
            job = self._find(video_id)
            if job.status == "PROCESSING":
                return self._response(job)
            if job.status != "READY":
                raise PartyJobConflictError(f"No se puede iniciar desde {job.status}")
            job.status = "PROCESSING"
            return self._response(job)

    def complete(self, video_id: UUID) -> dict[str, Any]:
        with self._lock:
            job = self._find(video_id)
            if job.status == "COMPLETED":
                return self._response(job)
            if job.status != "PROCESSING":
                raise PartyJobConflictError(f"No se puede completar desde {job.status}")
            job.status = "COMPLETED"
            return self._response(job)

    def fail(self, video_id: UUID, detail: str) -> dict[str, Any]:
        with self._lock:
            job = self._find(video_id)
            job.status = "FAILED"
            job.error = detail.strip()[:1000]
            return self._response(job)

    def cleanup(self, video_id: UUID) -> dict[str, Any]:
        with self._lock:
            job = self._find(video_id)
            if job.status not in {"COMPLETED", "FAILED"}:
                raise PartyJobConflictError(
                    f"Solo se puede limpiar COMPLETED/FAILED, no {job.status}"
                )
            response = self._response(job)
            response["status"] = "CLEANED"
            self._active = None
            self.runtime.stop()
            return response

    def get(self, video_id: UUID) -> dict[str, Any]:
        with self._lock:
            return self._response(self._find(video_id))

    def _find(self, video_id: UUID) -> _PartyJob:
        if self._active is None or self._active.video_id != video_id:
            raise PartyJobNotFoundError(f"Job {video_id} no encontrado en esta party")
        return self._active

    async def _notify_ready(self, video_id: UUID) -> None:
        endpoint = (
            f"{self._coordinator_url}/api/v1/videos/{video_id}/parties/"
            f"{self.party_id}/ready"
        )
        try:
            async with httpx.AsyncClient(timeout=self._notification_timeout) as client:
                response = await client.post(endpoint, json={"status": "READY"})
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise CoordinatorNotificationError(
                f"Coordinator rechazó READY de Party {self.party_id} "
                f"(HTTP {exc.response.status_code})"
            ) from exc
        except httpx.RequestError as exc:
            raise CoordinatorNotificationError(
                f"No se pudo notificar READY de Party {self.party_id}"
            ) from exc

    def _response(self, job: _PartyJob) -> dict[str, Any]:
        return {
            "video_id": str(job.video_id),
            "party_id": self.party_id,
            "status": job.status,
            "runtime": self.runtime.status,
            "error": job.error,
        }
