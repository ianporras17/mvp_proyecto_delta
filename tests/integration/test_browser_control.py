import asyncio
import struct
import time
from unittest.mock import AsyncMock
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from coordinator.api import browser_jobs as browser
from coordinator.services.video_service import VideoService
from coordinator.main import app
from party_receiver.api.browser import read_packet, MAX_MESSAGE, external_io
from fastapi import WebSocketDisconnect
from result_storage.main import create_app


@pytest.fixture
def control(monkeypatch):
    service = VideoService(["http://p0", "http://p1", "http://p2"])
    monkeypatch.setattr(browser, "video_service", service)
    queue = browser.BrowserQueue()
    queue.nodes_call = AsyncMock()
    monkeypatch.setattr(browser, "queue", queue)
    with TestClient(app) as client:
        yield client, queue, service


def enqueue(client, name="video.mp4"):
    response = client.post("/api/v1/browser-jobs", json={"filename": name, "format": "mp4"})
    assert response.status_code == 202
    return response.json()


def test_global_fifo_and_abort(control):
    client, queue, _ = control
    first, second, third = enqueue(client), enqueue(client), enqueue(client)
    assert first["status"] == "READY"
    assert second["position"] == 1 and third["position"] == 2
    assert client.post(f"/api/v1/browser-jobs/{second['id']}/start", json={"metadata": {"width": 2, "height": 2, "fps": 30, "frame_count": 1}}).status_code == 409
    assert client.delete(f"/api/v1/browser-jobs/{first['id']}").status_code == 200
    assert client.get(f"/api/v1/browser-jobs/{second['id']}").json()["status"] == "READY"
    assert queue.nodes_call.await_args_list[0].args[1] == "fail"
    assert queue.nodes_call.await_args_list[1].args[1] == "cleanup"


def test_cleanup_failure_holds_lease_and_retries(control):
    client, queue, _ = control
    first, second = enqueue(client), enqueue(client)
    queue.nodes_call.side_effect = [None, RuntimeError("node unavailable"), None, None]
    with pytest.raises(RuntimeError):
        client.delete(f"/api/v1/browser-jobs/{first['id']}")
    assert queue.active == browser.UUID(first["id"])
    assert client.get(f"/api/v1/browser-jobs/{second['id']}").json()["status"] == "QUEUED"
    assert client.delete(f"/api/v1/browser-jobs/{first['id']}").status_code == 200
    assert client.get(f"/api/v1/browser-jobs/{second['id']}").json()["status"] == "READY"


def test_expired_tab_releases_queue(control):
    client, queue, _ = control
    first, second = enqueue(client), enqueue(client)
    queue.jobs[browser.UUID(first["id"])].touched = time.monotonic() - 100
    asyncio.run(queue.expire())
    assert client.get(f"/api/v1/browser-jobs/{first['id']}").json()["status"] == "FAILED"
    assert client.get(f"/api/v1/browser-jobs/{second['id']}").json()["status"] == "READY"


def test_control_rejects_original_upload(control):
    client, _, _ = control
    assert client.post("/api/v1/browser-jobs", files={"video": ("source.mp4", b"original", "video/mp4")}).status_code == 422
    assert client.post("/api/v1/browser-jobs", json={"filename": "   ", "format": "mp4"}).status_code == 422


def test_storage_atomic_publication_and_restart(tmp_path):
    payload = b"\0\0\0\x18ftypisom" + b"anonymized-test-container"
    client = TestClient(create_app(tmp_path))
    result = client.post("/api/v1/results", files={"video": ("../../source.mp4", payload, "video/mp4")}, data={"counts": '{"people":2}'} )
    assert result.status_code == 201
    record = result.json()
    assert client.get("/api/v1/results").json() == []
    assert client.get(record["download_url"]).status_code == 404
    assert client.post(f"/api/v1/results/{record['id']}/publish").status_code == 200
    restarted = TestClient(create_app(tmp_path))
    assert restarted.get("/api/v1/results").json()[0]["id"] == record["id"]
    assert restarted.get(record["download_url"]).content == payload
    assert len(list(tmp_path.glob("*.mp4"))) == 1
    assert not list(tmp_path.glob("*.partial"))


@pytest.mark.parametrize("payload,content_type", [(b"", "video/mp4"), (b"not an mp4 file", "video/mp4"), (b"image", "image/png")])
def test_storage_rejects_invalid_results(tmp_path, payload, content_type):
    client = TestClient(create_app(tmp_path))
    assert client.post("/api/v1/results", files={"video": ("x.mp4", payload, content_type)}).status_code in (415, 422)
    assert client.get("/api/v1/results").json() == []
    assert not list(tmp_path.glob("*.partial"))


def test_adapter_framing_handles_tcp_fragmentation():
    async def run():
        reader = asyncio.StreamReader()
        for part in [b"\x03", b"\x00\x00", b"\x00a", b"bc"]:
            reader.feed_data(part)
        assert await read_packet(reader) == b"abc"
        reader.feed_data(struct.pack("<I", MAX_MESSAGE + 1))
        with pytest.raises(ValueError):
            await read_packet(reader)
    asyncio.run(run())


def test_adapter_disconnect_during_accept_does_not_keep_party_busy():
    state = SimpleNamespace(party_id=0, job_service=SimpleNamespace(get=lambda _: {"status": "PROCESSING"}))
    websocket = SimpleNamespace(
        headers={"origin": "http://localhost:8080"}, app=SimpleNamespace(state=state),
        accept=AsyncMock(side_effect=WebSocketDisconnect()), close=AsyncMock(side_effect=WebSocketDisconnect()),
    )
    asyncio.run(external_io(websocket, browser.uuid4()))
    assert state.browser_connected is False
