"""Node Agent FastAPI que supervisa una computing party MP-SPDZ."""

from contextlib import asynccontextmanager
import os

from fastapi import FastAPI

from party_receiver.api.jobs import router as jobs_router
from party_receiver.runtime import MpcRuntimeManager
from party_receiver.services.job_service import PartyJobService


def _party_id_from_environment() -> int:
    raw_party_id = os.getenv("PARTY_ID", "0")
    try:
        party_id = int(raw_party_id)
    except ValueError as exc:
        raise RuntimeError("PARTY_ID debe ser un entero entre 0 y 2") from exc
    if party_id not in (0, 1, 2):
        raise RuntimeError("PARTY_ID debe ser 0, 1 o 2")
    return party_id


def create_app(runtime_manager: MpcRuntimeManager | None = None) -> FastAPI:
    party_id = _party_id_from_environment()
    coordinator_url = os.getenv(
        "COORDINATOR_INTERNAL_URL", "http://coordinator-api:8000"
    ).strip()
    if not coordinator_url:
        raise RuntimeError("COORDINATOR_INTERNAL_URL no puede estar vacío")
    runtime = runtime_manager or MpcRuntimeManager(party_id)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        runtime.start()
        try:
            yield
        finally:
            runtime.stop()

    application = FastAPI(
        title=f"Proyecto Delta - Party {party_id} Node Agent",
        description="Controla un runtime MP-SPDZ; no recibe pixels por HTTP.",
        version="0.2.0",
        lifespan=lifespan,
    )
    application.state.party_id = party_id
    application.state.runtime_manager = runtime
    application.state.job_service = PartyJobService(
        party_id=party_id,
        runtime=runtime,
        coordinator_url=coordinator_url,
    )
    application.include_router(jobs_router)

    @application.get("/health", tags=["health"])
    async def health() -> dict[str, int | str]:
        return {
            "status": "ok",
            "party_id": party_id,
            "runtime": runtime.status,
        }

    return application


app = create_app()
