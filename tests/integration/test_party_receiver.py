"""Integración liviana del Node Agent sin arrancar MP-SPDZ."""

from collections.abc import Iterator
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from party_receiver.main import _party_id_from_environment, create_app
from party_receiver.runtime import MpcRuntimeManager


@pytest.fixture
def party_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[TestClient, AsyncMock]]:
    monkeypatch.setenv("PARTY_ID", "1")
    monkeypatch.setenv("COORDINATOR_INTERNAL_URL", "http://coordinator-api:8000")
    runtime = MpcRuntimeManager(1, enabled=False)
    app = create_app(runtime)
    notification = AsyncMock()
    monkeypatch.setattr(app.state.job_service, "_notify_ready", notification)
    with TestClient(app) as client:
        yield client, notification


def _metadata() -> dict:
    return {
        "width": 64,
        "height": 48,
        "fps": 2,
        "frame_count": 4,
        "dtype": "uint8",
        "channels": 1,
        "version": 2,
    }


def test_health_identifies_party_and_runtime(party_client: tuple[TestClient, AsyncMock]) -> None:
    client, _ = party_client
    assert client.get("/health").json() == {
        "status": "ok",
        "party_id": 1,
        "runtime": "disabled",
    }


def test_job_lifecycle_notifies_ready_and_cleans_memory(
    party_client: tuple[TestClient, AsyncMock],
) -> None:
    client, notification = party_client
    video_id = uuid4()

    prepared = client.post(
        f"/api/v1/jobs/{video_id}/prepare", json={"metadata": _metadata()}
    )
    assert prepared.status_code == 200
    assert prepared.json()["status"] == "READY"
    notification.assert_awaited_once_with(video_id)
    assert client.post(f"/api/v1/jobs/{video_id}/start").json()["status"] == "PROCESSING"
    assert client.post(f"/api/v1/jobs/{video_id}/complete").json()["status"] == "COMPLETED"
    assert client.post(f"/api/v1/jobs/{video_id}/cleanup").json()["status"] == "CLEANED"
    assert client.get(f"/api/v1/jobs/{video_id}").status_code == 404


def test_one_job_at_a_time(party_client: tuple[TestClient, AsyncMock]) -> None:
    client, _ = party_client
    first = uuid4()
    second = uuid4()
    assert client.post(
        f"/api/v1/jobs/{first}/prepare", json={"metadata": _metadata()}
    ).status_code == 200
    assert client.post(
        f"/api/v1/jobs/{second}/prepare", json={"metadata": _metadata()}
    ).status_code == 409


@pytest.mark.parametrize("party_id", ["-1", "3", "abc", ""])
def test_invalid_party_id_fails_fast(
    monkeypatch: pytest.MonkeyPatch, party_id: str
) -> None:
    monkeypatch.setenv("PARTY_ID", party_id)
    with pytest.raises(RuntimeError, match="PARTY_ID"):
        _party_id_from_environment()
