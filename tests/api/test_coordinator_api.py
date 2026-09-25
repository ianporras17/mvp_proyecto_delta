"""Contrato HTTP del Coordinator MPC."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from coordinator.api import videos
from coordinator.main import app
from coordinator.services.video_service import VideoService


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setattr(
        videos,
        "video_service",
        VideoService(("http://p0", "http://p1", "http://p2")),
    )
    with TestClient(app) as test_client:
        yield test_client


def _register(client: TestClient, name: str = "video.mp4") -> dict:
    response = client.post(
        "/api/v1/videos", json={"filename": name, "format": "mp4"}
    )
    assert response.status_code == 201
    return response.json()


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


def test_full_control_plane_lifecycle(client: TestClient) -> None:
    registration = _register(client)
    video_id = registration["id"]
    assert registration["status"] == "REGISTERED"
    assert {item["id"] for item in registration["parties"]} == {0, 1, 2}

    prepared = client.post(
        f"/api/v1/videos/{video_id}/prepare", json={"metadata": _metadata()}
    )
    assert prepared.status_code == 200
    assert prepared.json()["status"] == "PREPARING"

    for party_id in range(3):
        ready = client.post(
            f"/api/v1/videos/{video_id}/parties/{party_id}/ready",
            json={"status": "READY"},
        )
    assert ready.json()["status"] == "MPC_READY"
    assert client.post(f"/api/v1/videos/{video_id}/start").json()["status"] == "PROCESSING"
    assert client.post(f"/api/v1/videos/{video_id}/complete").json()["status"] == "COMPLETED"
    assert client.post(f"/api/v1/videos/{video_id}/cleanup").status_code == 200

    assert _register(client, "next.mp4")["status"] == "REGISTERED"


def test_second_active_job_returns_conflict(client: TestClient) -> None:
    _register(client)
    response = client.post(
        "/api/v1/videos", json={"filename": "second.mp4", "format": "mp4"}
    )
    assert response.status_code == 409


def test_invalid_metadata_and_transition_are_rejected(client: TestClient) -> None:
    video_id = _register(client)["id"]
    assert client.post(f"/api/v1/videos/{video_id}/start").status_code == 409
    bad = _metadata()
    bad["channels"] = 3
    assert client.post(
        f"/api/v1/videos/{video_id}/prepare", json={"metadata": bad}
    ).status_code == 422


def test_fail_releases_slot_only_after_cleanup(client: TestClient) -> None:
    video_id = _register(client)["id"]
    failed = client.post(
        f"/api/v1/videos/{video_id}/fail", json={"detail": "boom"}
    )
    assert failed.json()["status"] == "FAILED"
    assert failed.json()["error"] == "boom"
    assert client.post(
        "/api/v1/videos", json={"filename": "blocked.mp4", "format": "mp4"}
    ).status_code == 409
    client.post(f"/api/v1/videos/{video_id}/cleanup")
    assert _register(client, "allowed.mp4")


def test_health_and_openapi(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}
    paths = client.get("/openapi.json").json()["paths"]
    assert "/api/v1/videos/{video_id}/prepare" in paths
    assert "/api/v1/videos/{video_id}/start" in paths
    assert "/api/v1/videos/{video_id}/complete" in paths
