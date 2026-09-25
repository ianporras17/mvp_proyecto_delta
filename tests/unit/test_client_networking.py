"""Contratos HTTP de los clientes de control."""

from uuid import uuid4

import httpx
import pytest

from client.networking.coordinator_client import CoordinatorClient
from client.networking.party_client import PartyClient, PartyClientError
import client.networking.coordinator_client as coordinator_module
import client.networking.party_client as party_module


def _response(status: int, json: dict, request: httpx.Request) -> httpx.Response:
    return httpx.Response(status, json=json, request=request)


def test_coordinator_registration_and_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    video_id = str(uuid4())
    requests: list[httpx.Request] = []

    def post(url: str, **kwargs) -> httpx.Response:
        request = httpx.Request("POST", url, json=kwargs.get("json"))
        requests.append(request)
        if url.endswith("/api/v1/videos"):
            return _response(
                201,
                {
                    "id": video_id,
                    "status": "REGISTERED",
                    "parties": [
                        {"id": 0, "url": "http://p0"},
                        {"id": 1, "url": "http://p1"},
                        {"id": 2, "url": "http://p2"},
                    ],
                },
                request,
            )
        return _response(
            200,
            {"id": video_id, "status": "PREPARING", "parties": {}},
            request,
        )

    monkeypatch.setattr(coordinator_module.httpx, "post", post)
    client = CoordinatorClient("http://coordinator")
    registration = client.register_video("video.mp4", "mp4")
    state = client.prepare(video_id, {"width": 64})

    assert registration.id == video_id
    assert state.status == "PREPARING"
    assert requests[1].url.path.endswith(f"/{video_id}/prepare")


def test_party_prepare_validates_receipt(monkeypatch: pytest.MonkeyPatch) -> None:
    video_id = str(uuid4())

    def post(url: str, **kwargs) -> httpx.Response:
        request = httpx.Request("POST", url, json=kwargs.get("json"))
        return _response(
            200,
            {"video_id": video_id, "party_id": 2, "status": "READY"},
            request,
        )

    monkeypatch.setattr(party_module.httpx, "post", post)
    PartyClient().prepare_job(2, "http://party-2", video_id, {"width": 64})


def test_party_http_error_is_mapped(monkeypatch: pytest.MonkeyPatch) -> None:
    video_id = str(uuid4())

    def post(url: str, **kwargs) -> httpx.Response:
        request = httpx.Request("POST", url)
        return _response(409, {"detail": "busy"}, request)

    monkeypatch.setattr(party_module.httpx, "post", post)
    with pytest.raises(PartyClientError, match="busy"):
        PartyClient().start_job(0, "http://p0", video_id)


def test_party_control_urls_can_be_overridden_for_docker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_id = str(uuid4())
    requested_urls: list[str] = []

    def post(url: str, **kwargs) -> httpx.Response:
        requested_urls.append(url)
        request = httpx.Request("POST", url, json=kwargs.get("json"))
        return _response(
            200,
            {"video_id": video_id, "party_id": 1, "status": "READY"},
            request,
        )

    monkeypatch.setenv(
        "PARTY_CONTROL_URLS",
        "http://party-0:8000,http://party-1:8000,http://party-2:8000",
    )
    monkeypatch.setattr(party_module.httpx, "post", post)

    PartyClient().prepare_job(1, "http://localhost:8101", video_id, {"width": 64})

    assert requested_urls[0].startswith("http://party-1:8000/")
