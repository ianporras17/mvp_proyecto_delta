"""One-party WebSocket <-> authenticated ExternalIO TLS transport.

No arithmetic, inference, or aggregation belongs in this adapter. A WebSocket
message is one octetStream body; framing stays on the TCP side.
"""
import asyncio
import os
from pathlib import Path
import ssl
import struct
from uuid import UUID

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from mpc.web_origins import web_origins

router = APIRouter()
MAX_MESSAGE = 262144 * 8


async def read_packet(reader):
    length = struct.unpack("<I", await reader.readexactly(4))[0]
    if not 0 < length <= MAX_MESSAGE:
        raise ValueError("Invalid ExternalIO message length")
    return await reader.readexactly(length)


@router.websocket("/api/v1/jobs/{video_id}/external-io")
async def external_io(websocket: WebSocket, video_id: UUID):
    if websocket.headers.get("origin") not in web_origins():
        await websocket.close(code=1008)
        return
    state = websocket.app.state
    try:
        if state.job_service.get(video_id)["status"] != "PROCESSING":
            raise ValueError("Job not processing")
    except (LookupError, ValueError):
        await websocket.close(code=1008)
        return
    if getattr(state, "browser_connected", False):
        await websocket.close(code=1008)
        return
    state.browser_connected = True
    writer = None
    tasks = []
    try:
        await websocket.accept()
        cert_dir = Path(os.getenv("MPC_CERT_DIR", "/opt/MP-SPDZ/Player-Data"))
        party = state.party_id
        context = ssl.create_default_context(cafile=str(cert_dir / f"P{party}.pem"))
        context.load_cert_chain(cert_dir / "C0.pem", cert_dir / "C0.key")
        # Match MP-SPDZ's certificate CN (P0/P1/P2), including self-signed dev certs.
        reader, writer = await asyncio.wait_for(asyncio.open_connection("127.0.0.1", 14000 + party), 15)
        writer.write(struct.pack("<I", 1) + b"0")
        await writer.drain()
        await writer.start_tls(context, server_hostname=f"P{party}", ssl_handshake_timeout=15)

        async def to_browser():
            while True:
                await websocket.send_bytes(await read_packet(reader))

        async def to_party():
            while True:
                body = await websocket.receive_bytes()
                if not 0 < len(body) <= MAX_MESSAGE:
                    raise ValueError("Invalid browser message length")
                writer.write(struct.pack("<I", len(body)) + body)
                await writer.drain()

        tasks = [asyncio.create_task(to_browser()), asyncio.create_task(to_party())]
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    except Exception:
        # Never log protocol bodies: even individual random shares stay private.
        pass
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if writer:
            writer.close()
        state.browser_connected = False
        try:
            await websocket.close()
        except (RuntimeError, WebSocketDisconnect):
            pass
