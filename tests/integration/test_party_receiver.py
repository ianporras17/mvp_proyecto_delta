"""Integración liviana del Node Agent sin arrancar MP-SPDZ."""

from collections.abc import Iterator
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from party_receiver.api.jobs import JobMetadata
from party_receiver.main import _party_id_from_environment, create_app
from party_receiver.runtime import MpcRuntimeManager


@pytest.fixture
def party_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[TestClient, AsyncMock]]:
    monkeypatch.setenv("MPC_MAX_WIDTH", "640")
    monkeypatch.setenv("MPC_MAX_HEIGHT", "480")
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


@pytest.mark.parametrize("max_width", [320, 480, 640])
def test_prepare_accepts_configured_resolution(
    party_client: tuple[TestClient, AsyncMock],
    monkeypatch: pytest.MonkeyPatch,
    max_width: int,
) -> None:
    client, notification = party_client
    monkeypatch.setenv("MPC_MAX_WIDTH", str(max_width))
    monkeypatch.delenv("MPC_MAX_HEIGHT", raising=False)
    metadata = _metadata()
    metadata.update(width=max_width, height=max_width * 3 // 4, fps=10, frame_count=200)
    video_id = uuid4()

    response = client.post(
        f"/api/v1/jobs/{video_id}/prepare", json={"metadata": metadata}
    )

    assert response.status_code == 200
    notification.assert_awaited_once_with(video_id)


@pytest.mark.parametrize("dimension", ["width", "height"])
def test_prepare_rejects_resolution_above_configured_limit(
    party_client: tuple[TestClient, AsyncMock],
    monkeypatch: pytest.MonkeyPatch,
    dimension: str,
) -> None:
    client, notification = party_client
    monkeypatch.setenv("MPC_MAX_WIDTH", "480")
    monkeypatch.setenv("MPC_MAX_HEIGHT", "240")
    metadata = _metadata()
    metadata.update(width=480, height=240)
    metadata[dimension] += 1

    response = client.post(
        f"/api/v1/jobs/{uuid4()}/prepare", json={"metadata": metadata}
    )

    assert response.status_code == 422
    notification.assert_not_awaited()


def test_prepare_accepts_explicit_height_limit(
    party_client: tuple[TestClient, AsyncMock], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _ = party_client
    monkeypatch.setenv("MPC_MAX_HEIGHT", "240")
    metadata = _metadata()
    metadata.update(width=640, height=240)
    response = client.post(
        f"/api/v1/jobs/{uuid4()}/prepare", json={"metadata": metadata}
    )
    assert response.status_code == 200


@pytest.mark.parametrize("fps, frame_count", [(5, 100), (10, 200), (5, 200)])
def test_prepare_accepts_temporal_limits(
    party_client: tuple[TestClient, AsyncMock], fps: int, frame_count: int
) -> None:
    client, notification = party_client
    metadata = _metadata()
    metadata.update(fps=fps, frame_count=frame_count)
    video_id = uuid4()
    response = client.post(
        f"/api/v1/jobs/{video_id}/prepare", json={"metadata": metadata}
    )
    assert response.status_code == 200
    notification.assert_awaited_once_with(video_id)


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
    party_client: tuple[TestClient, AsyncMock], field: str, value: object
) -> None:
    client, notification = party_client
    metadata = _metadata()
    metadata[field] = value
    response = client.post(
        f"/api/v1/jobs/{uuid4()}/prepare", json={"metadata": metadata}
    )
    assert response.status_code == 422
    notification.assert_not_awaited()


@pytest.mark.parametrize("fps", [float("nan"), float("inf")])
def test_metadata_rejects_nonfinite_fps(fps: float) -> None:
    metadata = _metadata()
    metadata["fps"] = fps
    with pytest.raises(ValidationError):
        JobMetadata.model_validate(metadata)


@pytest.mark.parametrize(
    ("setting", "value"),
    [("MPC_MAX_WIDTH", "641"), ("MPC_MAX_HEIGHT", "481"), ("MPC_MAX_WIDTH", "invalid")],
)
def test_prepare_rejects_incompatible_profile(
    party_client: tuple[TestClient, AsyncMock],
    monkeypatch: pytest.MonkeyPatch,
    setting: str,
    value: str,
) -> None:
    client, notification = party_client
    monkeypatch.setenv(setting, value)
    response = client.post(
        f"/api/v1/jobs/{uuid4()}/prepare", json={"metadata": _metadata()}
    )
    assert response.status_code == 422
    notification.assert_not_awaited()


@pytest.mark.parametrize("party_id", ["-1", "3", "abc", ""])
def test_invalid_party_id_fails_fast(
    monkeypatch: pytest.MonkeyPatch, party_id: str
) -> None:
    monkeypatch.setenv("PARTY_ID", party_id)
    with pytest.raises(RuntimeError, match="PARTY_ID"):
        _party_id_from_environment()
