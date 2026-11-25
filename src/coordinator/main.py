"""FastAPI application for Proyecto Delta's control plane."""

from fastapi import FastAPI
import asyncio
from contextlib import asynccontextmanager
from coordinator.api.browser_jobs import router as browser_router, queue
from mpc.web_origins import add_web_cors

from coordinator.api.videos import router as videos_router


@asynccontextmanager
async def lifespan(app):
    async def sweep():
        while True:
            await asyncio.sleep(5)
            try:
                await queue.expire()
            except Exception:
                # Failed node cleanup is retried; the active lease stays held.
                pass
    task = asyncio.create_task(sweep())
    try:
        yield
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


app = FastAPI(
    title="Proyecto Delta Coordinator",
    description="Plano de control del MVP MPC; nunca recibe pixels ni el MP4.",
    version="0.2.0",
    lifespan=lifespan,
)
app.include_router(videos_router)
app.include_router(browser_router)
add_web_cors(app)


@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    """Lightweight service health endpoint."""

    return {"status": "ok"}

