"""Contrato HTTP del Coordinator MPC."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from coordinator.api import videos
from coordinator.main import app
from coordinator.models.video import VideoMetadata
from coordinator.services.video_service import VideoService


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("MPC_MAX_WIDTH", "640")
    monkeypatch.setenv("MPC_MAX_HEIGHT", "480")
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


@pytest.mark.parametrize("max_width", [320, 480, 640])
def test_prepare_accepts_configured_resolution(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, max_width: int
) -> None:
    monkeypatch.setenv("MPC_MAX_WIDTH", str(max_width))
    monkeypatch.delenv("MPC_MAX_HEIGHT", raising=False)
    metadata = _metadata()
    metadata.update(width=max_width, height=max_width * 3 // 4, fps=10, frame_count=200)
    video_id = _register(client)["id"]

    response = client.post(
        f"/api/v1/videos/{video_id}/prepare", json={"metadata": metadata}
    )

    assert response.status_code == 200
    assert response.json()["metadata"] == metadata


@pytest.mark.parametrize("dimension", ["width", "height"])
def test_prepare_rejects_resolution_above_configured_limit(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, dimension: str
) -> None:
    monkeypatch.setenv("MPC_MAX_WIDTH", "480")
    monkeypatch.setenv("MPC_MAX_HEIGHT", "240")
    metadata = _metadata()
    metadata.update(width=480, height=240)
    metadata[dimension] += 1
    video_id = _register(client)["id"]

    response = client.post(
        f"/api/v1/videos/{video_id}/prepare", json={"metadata": metadata}
    )

    assert response.status_code == 422


def test_prepare_accepts_explicit_height_limit(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MPC_MAX_HEIGHT", "240")
    metadata = _metadata()
    metadata.update(width=640, height=240)
    video_id = _register(client)["id"]

    response = client.post(
        f"/api/v1/videos/{video_id}/prepare", json={"metadata": metadata}
    )

    assert response.status_code == 200


@pytest.mark.parametrize("fps, frame_count", [(5, 100), (10, 200), (5, 200)])
def test_prepare_accepts_temporal_limits(
    client: TestClient, fps: int, frame_count: int
) -> None:
    metadata = _metadata()
    metadata.update(fps=fps, frame_count=frame_count)
    video_id = _register(client)["id"]

    response = client.post(
        f"/api/v1/videos/{video_id}/prepare", json={"metadata": metadata}
    )

    assert response.status_code == 200
    assert response.json()["metadata"] == metadata


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("width", 0),
        ("height", 0),
        ("fps", 0),
        ("fps", 10.01),
        ("frame_count", 0),
        ("frame_count", 201),
        ("dtype", "uint16"),
        ("channels", 3),
        ("version", 3),
    ],
)
def test_prepare_preserves_metadata_contract(
    client: TestClient, field: str, value: object
) -> None:
    metadata = _metadata()
    metadata[field] = value
    video_id = _register(client)["id"]

    response = client.post(
        f"/api/v1/videos/{video_id}/prepare", json={"metadata": metadata}
    )

    assert response.status_code == 422


@pytest.mark.parametrize("fps", [float("nan"), float("inf")])
def test_metadata_rejects_nonfinite_fps(fps: float) -> None:
    metadata = _metadata()
    metadata["fps"] = fps
    with pytest.raises(ValidationError):
        VideoMetadata.model_validate(metadata)


@pytest.mark.parametrize(
    ("setting", "value"),
    [("MPC_MAX_WIDTH", "641"), ("MPC_MAX_HEIGHT", "481"), ("MPC_MAX_WIDTH", "invalid")],
)
def test_prepare_rejects_incompatible_profile(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, setting: str, value: str
) -> None:
    monkeypatch.setenv(setting, value)
    video_id = _register(client)["id"]
    response = client.post(
        f"/api/v1/videos/{video_id}/prepare", json={"metadata": _metadata()}
    )
    assert response.status_code == 422


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
