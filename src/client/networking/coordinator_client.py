"""Cliente HTTP del plano de control de Proyecto Delta."""

from dataclasses import dataclass
import time
from typing import Any
from uuid import UUID

import httpx


class CoordinatorClientError(RuntimeError):
    """Error de comunicación o contrato con el Coordinator."""


@dataclass(frozen=True)
class PartyEndpoint:
    id: int
    url: str


@dataclass(frozen=True)
class VideoRegistration:
    id: str
    status: str
    parties: tuple[PartyEndpoint, PartyEndpoint, PartyEndpoint]


@dataclass(frozen=True)
class VideoState:
    id: str
    status: str
    parties: dict[str, str]
    metadata: dict[str, Any] | None = None
    error: str | None = None


class CoordinatorClient:
    """Coordina estados y metadata; nunca transmite pixels."""

    def __init__(self, base_url: str, timeout: float = 15.0) -> None:
        normalized_url = base_url.strip().rstrip("/")
        if not normalized_url:
            raise ValueError("COORDINATOR_URL no puede estar vacía")
        self._base_url = normalized_url
        self._timeout = timeout

    def register_video(self, filename: str, video_format: str) -> VideoRegistration:
        data = self._post_json(
            "/api/v1/videos",
            {"filename": filename, "format": video_format},
            action="registrar el video",
        )
        try:
            video_id = str(UUID(str(data["id"])))
            status = str(data["status"])
            raw_parties = data["parties"]
            if not isinstance(raw_parties, list):
                raise TypeError
            parties = tuple(
                PartyEndpoint(id=int(item["id"]), url=str(item["url"]).rstrip("/"))
                for item in raw_parties
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CoordinatorClientError("Registro inválido del Coordinator") from exc
        if len(parties) != 3 or {party.id for party in parties} != {0, 1, 2}:
            raise CoordinatorClientError("El Coordinator debe devolver parties 0, 1 y 2")
        ordered = tuple(sorted(parties, key=lambda party: party.id))
        return VideoRegistration(
            id=video_id,
            status=status,
            parties=(ordered[0], ordered[1], ordered[2]),
        )

    def get_video(self, video_id: str) -> VideoState:
        return self._decode_state(
            self._get_json(f"/api/v1/videos/{video_id}", "consultar el job")
        )

    def prepare(self, video_id: str, metadata: dict[str, Any]) -> VideoState:
        return self._control(video_id, "prepare", {"metadata": metadata})

    def start(self, video_id: str) -> VideoState:
        return self._control(video_id, "start")

    def complete(self, video_id: str) -> VideoState:
        return self._control(video_id, "complete")

    def fail(self, video_id: str, detail: str) -> VideoState:
        return self._control(video_id, "fail", {"detail": detail[:1000]})

    def cleanup(self, video_id: str) -> VideoState:
        return self._control(video_id, "cleanup")

    def wait_for_status(
        self,
        video_id: str,
        expected: str,
        *,
        timeout: float = 30.0,
        interval: float = 0.25,
    ) -> VideoState:
        deadline = time.monotonic() + timeout
        while True:
            state = self.get_video(video_id)
            if state.status == expected:
                return state
            if state.status == "FAILED":
                raise CoordinatorClientError(state.error or "El job MPC falló")
            if time.monotonic() >= deadline:
                raise CoordinatorClientError(
                    f"Timeout esperando {expected}; estado actual: {state.status}"
                )
            time.sleep(interval)

    def _control(
        self,
        video_id: str,
        action_name: str,
        payload: dict[str, Any] | None = None,
    ) -> VideoState:
        data = self._post_json(
            f"/api/v1/videos/{video_id}/{action_name}",
            payload,
            action=f"ejecutar {action_name}",
        )
        return self._decode_state(data)

    @staticmethod
    def _decode_state(data: dict[str, Any]) -> VideoState:
        try:
            video_id = str(UUID(str(data["id"])))
            status = str(data["status"])
            raw_parties = data.get("parties", {})
            if not isinstance(raw_parties, dict):
                raise TypeError
            parties = {str(key): str(value) for key, value in raw_parties.items()}
            metadata = data.get("metadata")
            if metadata is not None and not isinstance(metadata, dict):
                raise TypeError
            error = data.get("error")
        except (KeyError, TypeError, ValueError) as exc:
            raise CoordinatorClientError("Estado inválido del Coordinator") from exc
        return VideoState(
            id=video_id,
            status=status,
            parties=parties,
            metadata=metadata,
            error=None if error is None else str(error),
        )

    def _post_json(
        self,
        path: str,
        payload: dict[str, Any] | None,
        action: str,
    ) -> dict[str, Any]:
        try:
            kwargs: dict[str, Any] = {"timeout": self._timeout}
            if payload is not None:
                kwargs["json"] = payload
            response = httpx.post(f"{self._base_url}{path}", **kwargs)
        except httpx.RequestError as exc:
            raise CoordinatorClientError(
                f"No se pudo contactar al Coordinator para {action}: {exc}"
            ) from exc
        return self._decode_response(response, action)

    def _get_json(self, path: str, action: str) -> dict[str, Any]:
        try:
            response = httpx.get(f"{self._base_url}{path}", timeout=self._timeout)
        except httpx.RequestError as exc:
            raise CoordinatorClientError(
                f"No se pudo contactar al Coordinator para {action}: {exc}"
            ) from exc
        return self._decode_response(response, action)

    @staticmethod
    def _decode_response(response: httpx.Response, action: str) -> dict[str, Any]:
        if response.is_error:
            detail = response.text
            try:
                body = response.json()
                if isinstance(body, dict) and body.get("detail"):
                    detail = str(body["detail"])
            except ValueError:
                pass
            raise CoordinatorClientError(
                f"Coordinator rechazó {action} (HTTP {response.status_code}): {detail}"
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise CoordinatorClientError("Coordinator devolvió JSON inválido") from exc
        if not isinstance(body, dict):
            raise CoordinatorClientError("Coordinator devolvió un objeto inválido")
        return body
