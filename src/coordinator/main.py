"""FastAPI application for Proyecto Delta's control plane."""

from fastapi import FastAPI

from coordinator.api.videos import router as videos_router


app = FastAPI(
    title="Proyecto Delta Coordinator",
    description="Plano de control del MVP MPC; nunca recibe pixels ni el MP4.",
    version="0.2.0",
)
app.include_router(videos_router)


@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    """Lightweight service health endpoint."""

    return {"status": "ok"}

