"""SwapMic REST server (FastAPI).

Endpoints:
  POST /swap      multipart: song file + form fields → final WAV
  POST /separate  multipart: song file              → vocals + instrumental
  GET  /health
  GET  /models    list installed RVC models from $SWAPMIC_MODELS_DIR

Run with:
    uvicorn swapmic.server:app --host 127.0.0.1 --port 3699
"""
from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from . import __version__
from .convert import ConversionParams
from .pipeline import SwapConfig, swap
from .remix import RemixParams
from .separate import separate as run_separate


MODELS_DIR = Path(os.environ.get("SWAPMIC_MODELS_DIR", "models")).resolve()
OUTPUT_ROOT = Path(os.environ.get("SWAPMIC_OUTPUT_DIR", "output")).resolve()

app = FastAPI(title="SwapMic", version=__version__)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "version": __version__, "models_dir": str(MODELS_DIR)}


@app.get("/models")
def list_models() -> dict[str, list[dict[str, str]]]:
    if not MODELS_DIR.exists():
        return {"models": []}
    models = []
    for pth in sorted(MODELS_DIR.glob("*.pth")):
        index = pth.with_suffix(".index")
        models.append({
            "name": pth.stem,
            "model": str(pth),
            "index": str(index) if index.exists() else "",
        })
    return {"models": models}


def _save_upload(upload: UploadFile, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(upload.filename or "song.wav").suffix or ".wav"
    dest = dest_dir / f"{uuid.uuid4().hex}{suffix}"
    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return dest


@app.post("/separate")
def api_separate(song: UploadFile = File(...)) -> JSONResponse:
    job_id = uuid.uuid4().hex[:8]
    work = OUTPUT_ROOT / job_id
    song_path = _save_upload(song, work / "input")
    result = run_separate(song_path, work / "stems")
    return JSONResponse({
        "job_id": job_id,
        "vocals": str(result.vocals),
        "instrumental": str(result.instrumental),
        "sample_rate": result.sample_rate,
    })


@app.post("/swap")
def api_swap(
    song: UploadFile = File(...),
    model: str = Form(..., description="Model name (matches /models response)."),
    pitch: int = Form(0),
    index_rate: float = Form(0.75),
    protect: float = Form(0.33),
    vocal_gain_db: float = Form(0.0),
) -> FileResponse:
    model_path = MODELS_DIR / f"{model}.pth"
    index_path = MODELS_DIR / f"{model}.index"
    if not model_path.exists():
        raise HTTPException(404, f"Model not found: {model}")

    job_id = uuid.uuid4().hex[:8]
    work = OUTPUT_ROOT / job_id
    song_path = _save_upload(song, work / "input")

    config = SwapConfig(
        model_path=model_path,
        index_path=index_path if index_path.exists() else None,
        out_dir=work,
        convert_params=ConversionParams(
            pitch_shift=pitch, index_rate=index_rate, protect=protect
        ),
        remix_params=RemixParams(vocal_gain_db=vocal_gain_db),
    )
    result = swap(song_path, config)
    return FileResponse(
        result.final,
        media_type="audio/wav",
        filename=result.final.name,
    )
