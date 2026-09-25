"""Control de Node Agents; los pixels no pasan por HTTP."""

import os
from collections.abc import Sequence
from typing import Any
from urllib.parse import urlparse
from uuid import UUID

import httpx


class PartyClientError(RuntimeError):
    """Un Node Agent no pudo preparar o limpiar el job."""


class PartyClient:
    def __init__(
        self,
        timeout: float = 30.0,
        party_urls: Sequence[str] | None = None,
    ) -> None:
        self._timeout = timeout
        configured = party_urls
        if configured is None:
            raw_urls = os.getenv("PARTY_CONTROL_URLS")
            if raw_urls:
                configured = tuple(
                    value.strip() for value in raw_urls.split(",") if value.strip()
                )
        if configured is None:
            self._party_urls: tuple[str, str, str] | None = None
        else:
            values = tuple(value.rstrip("/") for value in configured)
            if len(values) != 3:
                raise ValueError(
                    "PARTY_CONTROL_URLS debe contener exactamente tres URLs"
                )
            for value in values:
                parsed = urlparse(value)
                if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                    raise ValueError(f"URL de control de party inválida: {value!r}")
            self._party_urls = values[0], values[1], values[2]

    def prepare_job(
        self,
        party_id: int,
        party_url: str,
        video_id: str,
        metadata: dict[str, Any],
    ) -> None:
        self._request(
            party_id,
            party_url,
            video_id,
            "prepare",
            {"metadata": metadata},
            expected="READY",
        )

    def start_job(self, party_id: int, party_url: str, video_id: str) -> None:
        self._request(party_id, party_url, video_id, "start", expected="PROCESSING")

    def complete_job(self, party_id: int, party_url: str, video_id: str) -> None:
        self._request(party_id, party_url, video_id, "complete", expected="COMPLETED")

    def fail_job(
        self,
        party_id: int,
        party_url: str,
        video_id: str,
        detail: str,
    ) -> None:
        self._request(
            party_id,
            party_url,
            video_id,
            "fail",
            {"detail": detail[:1000]},
            expected="FAILED",
        )

    def cleanup_job(self, party_id: int, party_url: str, video_id: str) -> None:
        self._request(party_id, party_url, video_id, "cleanup", expected="CLEANED")

    def _request(
        self,
        party_id: int,
        party_url: str,
        video_id: str,
        action: str,
        payload: dict[str, Any] | None = None,
        *,
        expected: str,
    ) -> None:
        if party_id not in (0, 1, 2):
            raise ValueError("party_id debe ser 0, 1 o 2")
        normalized_id = str(UUID(video_id))
        if self._party_urls is not None:
            party_url = self._party_urls[party_id]
        endpoint = f"{party_url.rstrip('/')}/api/v1/jobs/{normalized_id}/{action}"
        try:
            kwargs: dict[str, Any] = {"timeout": self._timeout}
            if payload is not None:
                kwargs["json"] = payload
            response = httpx.post(endpoint, **kwargs)
        except httpx.RequestError as exc:
            raise PartyClientError(f"Party {party_id} no disponible: {exc}") from exc
        if response.is_error:
            raise PartyClientError(
                f"Party {party_id} rechazó {action} (HTTP {response.status_code}): "
                f"{self._error_detail(response)}"
            )
        try:
            body = response.json()
            returned_id = str(UUID(str(body["video_id"])))
            returned_party = int(body["party_id"])
            returned_status = str(body["status"])
        except (ValueError, TypeError, KeyError) as exc:
            raise PartyClientError(f"Party {party_id} devolvió respuesta inválida") from exc
        if (
            returned_id != normalized_id
            or returned_party != party_id
            or returned_status != expected
        ):
            raise PartyClientError(f"Party {party_id} devolvió estado inesperado")

    @staticmethod
    def _error_detail(response: httpx.Response) -> str:
        try:
            body = response.json()
            if isinstance(body, dict) and body.get("detail"):
                return str(body["detail"])
        except ValueError:
            pass
        return response.text
