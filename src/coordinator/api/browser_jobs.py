"""FIFO browser leases. Only public metadata is kept; originals stay in the tab.

One uvicorn process is required: like the existing control plane, queue state
is volatile. Cleanup must finish on all nodes before the next lease is granted.
"""
import asyncio
from dataclasses import dataclass, field
import os
import time
from uuid import UUID, uuid4

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from coordinator.api.videos import video_service
from coordinator.models.video import VideoMetadata, VideoRegistrationRequest

router = APIRouter(prefix="/api/v1/browser-jobs", tags=["browser jobs"])


@dataclass
class Ticket:
    filename: str
    id: UUID = field(default_factory=uuid4)
    status: str = "QUEUED"
    touched: float = field(default_factory=time.monotonic)
    video_id: UUID | None = None
    error: str | None = None
    finishing: bool | None = None


class Preparation(BaseModel):
    metadata: VideoMetadata


class BrowserQueue:
    def __init__(self):
        self.jobs = {}
        self.active = None
        self.lock = asyncio.Lock()
        self.timeout = float(os.getenv("DELTA_LEASE_SECONDS", "90"))

    @staticmethod
    def nodes():
        return os.getenv("PARTY_CONTROL_URLS", "http://party-0:8000,http://party-1:8000,http://party-2:8000").split(",")

    async def nodes_call(self, job, action, body=None, *, missing_ok=False):
        async with httpx.AsyncClient(timeout=35) as client:
            replies = await asyncio.gather(*(client.post(f"{url.strip()}/api/v1/jobs/{job.video_id}/{action}", json=body) for url in self.nodes()))
            for reply in replies:
                if missing_ok and reply.status_code == 404:
                    continue
                reply.raise_for_status()

    def find(self, job_id):
        if job_id not in self.jobs:
            raise HTTPException(404, "Trabajo no encontrado")
        return self.jobs[job_id]

    def snapshot(self, job):
        waiting = [j.id for j in self.jobs.values() if j.status == "QUEUED"]
        return {"id": str(job.id), "video_id": str(job.video_id) if job.video_id else None,
                "status": job.status, "position": waiting.index(job.id) + 1 if job.id in waiting else 0,
                "error": job.error}

    def promote(self):
        if self.active is None:
            for job in self.jobs.values():
                if job.status == "QUEUED":
                    registered = video_service.register_video(job.filename, "mp4")
                    job.video_id = registered.id
                    job.status = "READY"
                    self.active = job.id
                    break

    async def finish(self, job, success=False):
        if job.status in {"COMPLETED", "FAILED"}:
            return
        if self.active == job.id:
            # On cleanup failure keep the lease occupied: never overlap runtimes.
            if job.finishing is None:
                if success:
                    await self.nodes_call(job, "complete")
                    video_service.complete(job.video_id)
                else:
                    await self.nodes_call(job, "fail", {"detail": "Cliente desconectado o trabajo abortado"}, missing_ok=True)
                    video_service.fail(job.video_id, "Trabajo del navegador abortado")
                job.finishing = success
            await self.nodes_call(job, "cleanup", missing_ok=True)
            video_service.cleanup(job.video_id)
            self.active = None
        job.status = "COMPLETED" if job.finishing is True else "FAILED"
        self.promote()

    async def expire(self):
        async with self.lock:
            for job in list(self.jobs.values()):
                if job.status not in {"COMPLETED", "FAILED"} and time.monotonic() - job.touched > self.timeout:
                    await self.finish(job)


queue = BrowserQueue()


@router.post("", status_code=202)
async def enqueue(payload: VideoRegistrationRequest):
    if payload.format.lower() != "mp4":
        raise HTTPException(415, "Solo MP4")
    if not payload.filename.strip():
        raise HTTPException(422, "El nombre no puede estar vacío")
    async with queue.lock:
        job = Ticket(payload.filename)
        queue.jobs[job.id] = job
        queue.promote()
        return queue.snapshot(job)


@router.get("/{job_id}")
async def heartbeat(job_id: UUID):
    async with queue.lock:
        job = queue.find(job_id)
        job.touched = time.monotonic()
        return queue.snapshot(job)


@router.post("/{job_id}/start")
async def start(job_id: UUID, payload: Preparation):
    async with queue.lock:
        job = queue.find(job_id)
        if job.status != "READY" or queue.active != job.id:
            raise HTTPException(409, "El trabajo aún no tiene turno")
        job.touched = time.monotonic()
        try:
            video_service.prepare(job.video_id, payload.metadata)
            await queue.nodes_call(job, "prepare", payload.model_dump())
            await queue.nodes_call(job, "start")
            video_service.start(job.video_id)
            job.status = "PROCESSING"
        except Exception as exc:
            await queue.finish(job)
            raise HTTPException(502, "No se pudieron preparar las parties MPC") from exc
        return queue.snapshot(job)


@router.post("/{job_id}/storing")
async def storing(job_id: UUID):
    async with queue.lock:
        job = queue.find(job_id)
        if job.status != "PROCESSING":
            raise HTTPException(409, "Estado inválido")
        job.status = "STORING"
        return queue.snapshot(job)


@router.post("/{job_id}/complete")
async def complete(job_id: UUID):
    async with queue.lock:
        job = queue.find(job_id)
        if job.status != "STORING":
            raise HTTPException(409, "El resultado aún no está almacenándose")
        await queue.finish(job, success=True)
        return queue.snapshot(job)


@router.delete("/{job_id}")
async def abort(job_id: UUID):
    async with queue.lock:
        job = queue.find(job_id)
        await queue.finish(job)
        return queue.snapshot(job)
