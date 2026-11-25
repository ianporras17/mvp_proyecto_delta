"""Stable result contract: POST/GET /api/v1/results, GET /{id}/download.

SQLite holds the catalog; video bytes live in the same persistent volume.
Upload staging is never listed. This demo trusts the browser's MPC validation;
it cannot determine whether arbitrary uploads have been anonymized.
"""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from uuid import UUID, uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from mpc.web_origins import add_web_cors


def create_app(root=None):
    root = Path(root or os.getenv("DELTA_STORAGE_DIR", "data/results"))
    root.mkdir(parents=True, exist_ok=True)

    def connect():
        db = sqlite3.connect(root / "catalog.sqlite3")
        db.row_factory = sqlite3.Row
        return db

    with connect() as db:
        db.execute("CREATE TABLE IF NOT EXISTS results (id TEXT PRIMARY KEY, filename TEXT NOT NULL, size INTEGER NOT NULL, sha256 TEXT NOT NULL, counts TEXT NOT NULL, published INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')))")
        if "published" not in {r[1] for r in db.execute("PRAGMA table_info(results)")}:
            db.execute("ALTER TABLE results ADD COLUMN published INTEGER NOT NULL DEFAULT 1")
    app = FastAPI(title="Delta anonymized results", version="1.0")
    add_web_cors(app)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    def record(row):
        item = dict(row)
        item["counts"] = json.loads(item["counts"])
        item["download_url"] = f"/api/v1/results/{item['id']}/download"
        return item

    @app.post("/api/v1/results", status_code=201)
    async def upload(video: UploadFile = File(...), counts: str = Form("{}")):
        if video.content_type != "video/mp4":
            raise HTTPException(415, "Solo resultados MP4")
        try:
            parsed = json.loads(counts)
            if not isinstance(parsed, dict) or len(counts) > 4096:
                raise ValueError()
        except (ValueError, TypeError) as exc:
            raise HTTPException(422, "Conteos inválidos") from exc
        result_id = str(uuid4())
        staging, target = root / f"{result_id}.partial", root / f"{result_id}.mp4"
        digest, size = hashlib.sha256(), 0
        try:
            with staging.open("xb") as output:
                while chunk := await video.read(1024 * 1024):
                    if size == 0 and (len(chunk) < 12 or chunk[4:8] != b"ftyp"):
                        raise HTTPException(422, "Resultado MP4 inválido")
                    output.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                if size == 0:
                    raise HTTPException(422, "Resultado vacío")
                output.flush()
                os.fsync(output.fileno())
            staging.replace(target)
            with connect() as db:
                db.execute("INSERT INTO results(id,filename,size,sha256,counts) VALUES(?,?,?,?,?)", (result_id, "video-anonimizado.mp4", size, digest.hexdigest(), json.dumps(parsed)))
                row = db.execute("SELECT * FROM results WHERE id=?", (result_id,)).fetchone()
            return record(row)
        except BaseException:
            staging.unlink(missing_ok=True)
            target.unlink(missing_ok=True)
            raise
        finally:
            await video.close()

    @app.post("/api/v1/results/{result_id}/publish")
    def publish(result_id: UUID):
        with connect() as db:
            row = db.execute("SELECT * FROM results WHERE id=?", (str(result_id),)).fetchone()
            if row is None or not (root / f"{result_id}.mp4").is_file():
                raise HTTPException(404, "Resultado no encontrado")
            db.execute("UPDATE results SET published=1 WHERE id=?", (str(result_id),))
            row = db.execute("SELECT * FROM results WHERE id=?", (str(result_id),)).fetchone()
        return record(row)

    @app.get("/api/v1/results")
    def listing():
        with connect() as db:
            return [record(row) for row in db.execute("SELECT * FROM results WHERE published=1 ORDER BY created_at DESC") if (root / f"{row['id']}.mp4").is_file()]

    @app.get("/api/v1/results/{result_id}/download")
    def download(result_id: UUID):
        with connect() as db:
            row = db.execute("SELECT * FROM results WHERE id=?", (str(result_id),)).fetchone()
        path = root / f"{result_id}.mp4"
        if row is None or not row["published"] or not path.is_file():
            raise HTTPException(404, "Resultado no encontrado")
        return FileResponse(path, media_type="video/mp4", filename=row["filename"])

    return app


app = create_app()
