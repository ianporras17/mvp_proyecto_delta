"""Explicit development origins shared by the browser-facing services."""
import os
from fastapi.middleware.cors import CORSMiddleware


def web_origins():
    return [x.strip() for x in os.getenv("DELTA_WEB_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8080,http://127.0.0.1:8080,http://localhost:8081,http://127.0.0.1:8081").split(",") if x.strip()]


def add_web_cors(app):
    app.add_middleware(CORSMiddleware, allow_origins=web_origins(), allow_methods=["GET", "POST", "DELETE"], allow_headers=["Content-Type"], expose_headers=["Content-Disposition"])
