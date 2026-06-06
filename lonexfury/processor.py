"""
processor.py — LONEXFURY Ingestion Service
==========================================
FastAPI service that runs on LONEXFURY (Friend 3's laptop).
Accepts files and URLs, runs the full ingestion pipeline,
and sends chunks to VORTEX master node for embedding + storage.

Endpoints:
  POST /process/pdf          → process PDF/DOCX/TXT file
  POST /process/image        → process image / handwritten notes
  POST /process/youtube      → process YouTube / social media URL
  POST /process/batch        → process multiple files at once
  GET  /health               → heartbeat for master node
  GET  /status               → current job queue status

Run with:
  uvicorn processor:app --host 0.0.0.0 --port 8001 --reload

Dependencies:
  pip install fastapi uvicorn python-multipart httpx aiofiles
  pip install pymupdf pdfplumber python-docx pillow pytesseract
  pip install opencv-python-headless numpy langdetect
  pip install yt-dlp youtube-transcript-api faster-whisper
  pip install imageio-ffmpeg deep-translator sentence-transformers
"""

import os
import uuid
import asyncio
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List

import httpx
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

# VORTEX master node — uses mDNS hostname to avoid IP rotation issues
MASTER_URL      = os.getenv("MASTER_URL", "http://vortex.local:8000")
UPLOAD_DIR      = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

MAX_FILE_MB     = 100
SUPPORTED_DOCS  = {".pdf", ".docx", ".doc", ".txt"}
SUPPORTED_IMGS  = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}

# ─────────────────────────────────────────────
# JOB TRACKER (in-memory for hackathon)
# ─────────────────────────────────────────────

jobs: dict = {}      # job_id → {status, progress, error, result}

def new_job(job_type: str, source: str) -> str:
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "id":         job_id,
        "type":       job_type,
        "source":     source,
        "status":     "queued",
        "progress":   0,
        "chunks":     0,
        "error":      None,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "done_at":    None,
    }
    return job_id

def update_job(job_id: str, **kwargs):
    if job_id in jobs:
        jobs[job_id].update(kwargs)

def fail_job(job_id: str, error: str):
    update_job(job_id, status="failed", error=error,
               done_at=datetime.now(timezone.utc).isoformat())

def complete_job(job_id: str, chunks: int):
    update_job(job_id, status="done", progress=100, chunks=chunks,
               done_at=datetime.now(timezone.utc).isoformat())

# ─────────────────────────────────────────────
# OFFLINE QUEUE (persists chunks when master is down)
# ─────────────────────────────────────────────

QUEUE_DIR = Path("./pending_uploads")
QUEUE_DIR.mkdir(exist_ok=True)

def queue_payload_to_disk(payload: dict) -> str:
    """Save payload to disk for retry when master comes back online."""
    queue_file = QUEUE_DIR / f"{payload['job_id']}_{int(time.time())}.json"
    import json
    with open(queue_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return str(queue_file)

async def retry_queued_uploads():
    """Background task: retry any queued uploads if master is back online."""
    import json
    queued = list(QUEUE_DIR.glob("*.json"))
    if not queued:
        return
    print(f"  🔁  Retrying {len(queued)} queued uploads…")
    for qfile in queued:
        try:
            with open(qfile, "r", encoding="utf-8") as f:
                payload = json.load(f)
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(f"{MASTER_URL}/ingest", json=payload)
                resp.raise_for_status()
            qfile.unlink()
            print(f"  ✅  Drained {qfile.name}")
        except Exception as e:
            print(f"  ⏸   Master still unreachable for {qfile.name}: {e}")
            break  # stop trying — master is still down

# ─────────────────────────────────────────────
# SEND TO MASTER (with offline queueing)
# ─────────────────────────────────────────────

async def send_to_master(
    session_id: str,
    metadata:   dict,
    chunks:     list,
    job_id:     str,
):
    """
    Send processed chunks to VORTEX master node for embedding + ChromaDB storage.
    Retries up to 3 times. On total failure, queues to disk for later retry.
    """
    payload = {
        "session_id": session_id,
        "metadata":   metadata,
        "chunks":     chunks,
        "job_id":     job_id,
    }

    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{MASTER_URL}/ingest",
                    json=payload,
                )
                resp.raise_for_status()
                print(f"  ✅  Sent {len(chunks)} chunks to master (job {job_id})")
                return resp.json()
        except Exception as e:
            print(f"  ⚠️  Send attempt {attempt+1} failed: {e}")
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)

    # All retries failed — queue to disk for later retry
    qfile = queue_payload_to_disk(payload)
    print(f"  💾  Queued to disk: {qfile} (will retry when master is back)")
    return {"queued": True, "file": qfile}

# ─────────────────────────────────────────────
# BACKGROUND TASKS
# ─────────────────────────────────────────────

async def run_pdf_job(job_id: str, file_path: str, session_id: str):
    """Background task: extract + chunk + send document."""
    try:
        update_job(job_id, status="processing", progress=10)

        # Import here to avoid slow startup
        from pdf_processor import process_document
        from ocr_processor import process_image_batch

        # Step 1: Extract text + tables
        update_job(job_id, status="extracting", progress=20)
        metadata, chunks = process_document(file_path)

        update_job(job_id, progress=50)

        # Step 2: OCR any embedded images
        image_paths = metadata.get("image_paths", [])
        if image_paths:
            update_job(job_id, status="ocr", progress=55)
            try:
                _, img_chunks = process_image_batch(
                    image_paths,
                    title=metadata.get("title", "")
                )
                # Only add OCR chunks that have meaningful text
                text_img_chunks = [
                    c for c in img_chunks
                    if not c["metadata"].get("is_diagram") == "True"
                    and len(c["english_text"]) > 50
                ]
                chunks.extend(text_img_chunks)
                print(f"  📝  Added {len(text_img_chunks)} OCR text chunks from images")
            except Exception as e:
                print(f"  ⚠️  OCR failed (non-fatal): {e}")

        update_job(job_id, progress=80)

        # Step 3: Send to master
        update_job(job_id, status="sending", progress=85)
        await send_to_master(session_id, metadata, chunks, job_id)

        complete_job(job_id, len(chunks))

    except Exception as e:
        fail_job(job_id, str(e))
        raise
    finally:
        # Cleanup uploaded file
        try:
            Path(file_path).unlink(missing_ok=True)
        except Exception:
            pass
        # Cleanup extracted image folders from CWD (where processor.py runs)
        try:
            cwd = Path(".")
            for folder in list(cwd.glob("__images_*__")) + list(cwd.glob("__rendered_*__")):
                shutil.rmtree(folder, ignore_errors=True)
        except Exception:
            pass


async def run_image_job(job_id: str, file_path: str, session_id: str, title: str):
    """Background task: OCR image + send."""
    try:
        update_job(job_id, status="processing", progress=10)

        from ocr_processor import process_image

        update_job(job_id, status="ocr", progress=20)
        metadata, chunks = process_image(file_path, title=title)

        update_job(job_id, progress=80)

        update_job(job_id, status="sending", progress=85)
        await send_to_master(session_id, metadata, chunks, job_id)

        complete_job(job_id, len(chunks))

    except Exception as e:
        fail_job(job_id, str(e))
        raise
    finally:
        try:
            Path(file_path).unlink(missing_ok=True)
        except Exception:
            pass


async def run_youtube_job(job_id: str, url: str, session_id: str):
    """Background task: scrape YouTube/Instagram/Facebook/Twitter + send."""
    try:
        update_job(job_id, status="fetching_metadata", progress=5)

        from youtube_scraper import scrape_youtube

        update_job(job_id, status="fetching_transcript", progress=10)
        # scrape_youtube handles all stages internally — metadata, transcript,
        # translation, chunking. Progress jumps reflect real pipeline stages.
        # For long videos, transcription can take several minutes on CPU.
        update_job(job_id, status="transcribing", progress=15)
        metadata, chunks = scrape_youtube(url)

        lang = metadata.get("language", "unknown")
        trans = metadata.get("translation_model", "none")
        update_job(job_id,
                   status=f"transcribed ({lang}, translation={trans})",
                   progress=80)

        update_job(job_id, status="sending", progress=85)
        await send_to_master(session_id, metadata, chunks, job_id)

        complete_job(job_id, len(chunks))

    except Exception as e:
        fail_job(job_id, str(e))
        raise


async def run_batch_job(job_id: str, file_paths: List[str], session_id: str):
    """Background task: process multiple files sequentially."""
    total       = len(file_paths)
    all_chunks  = 0

    update_job(job_id, status="processing", progress=5)

    from pdf_processor  import process_document
    from ocr_processor  import process_image

    for i, file_path in enumerate(file_paths):
        ext = Path(file_path).suffix.lower()
        try:
            if ext in SUPPORTED_DOCS:
                metadata, chunks = process_document(file_path)
            elif ext in SUPPORTED_IMGS:
                metadata, chunks = process_image(file_path)
            else:
                print(f"  ⚠️  Skipping unsupported file: {file_path}")
                continue

            await send_to_master(session_id, metadata, chunks, job_id)
            all_chunks += len(chunks)

            progress = int(((i + 1) / total) * 90)
            update_job(job_id, progress=progress,
                       status=f"processed {i+1}/{total}")

        except Exception as e:
            print(f"  ⚠️  Failed to process {file_path}: {e}")
        finally:
            try:
                Path(file_path).unlink(missing_ok=True)
            except Exception:
                pass

    complete_job(job_id, all_chunks)

# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────

app = FastAPI(
    title="StudyMind AI — LONEXFURY Processor",
    description="Ingestion pipeline: PDF, OCR, YouTube → chunks → VORTEX",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    """Heartbeat endpoint — master node polls this every 30s."""
    queued_count = len(list(QUEUE_DIR.glob("*.json")))
    return {
        "status":    "ok",
        "node":      "lonexfury",
        "role":      "processor",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "master_url": MASTER_URL,
        "jobs":      {
            "total":      len(jobs),
            "running":    sum(1 for j in jobs.values() if j["status"] not in {"done", "failed"}),
            "done":       sum(1 for j in jobs.values() if j["status"] == "done"),
            "failed":     sum(1 for j in jobs.values() if j["status"] == "failed"),
            "queued_offline": queued_count,
        }
    }


@app.post("/drain_queue")
async def drain_queue(background_tasks: BackgroundTasks):
    """Manually retry all queued uploads (use after master comes back online)."""
    background_tasks.add_task(retry_queued_uploads)
    queued_count = len(list(QUEUE_DIR.glob("*.json")))
    return {
        "status": "draining",
        "queued_uploads": queued_count,
        "message": "Retrying in background — poll /health to see progress",
    }


@app.get("/status/{job_id}")
async def job_status(job_id: str):
    """Poll job progress."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return jobs[job_id]


@app.get("/jobs")
async def list_jobs():
    """List all jobs (last 50)."""
    recent = sorted(jobs.values(), key=lambda j: j["started_at"], reverse=True)[:50]
    return {"jobs": recent}


@app.post("/process/pdf")
async def process_pdf(
    background_tasks: BackgroundTasks,
    file:       UploadFile = File(...),
    session_id: str        = Form(...),
):
    """
    Upload and process a PDF, DOCX, or TXT file.
    Returns job_id immediately — poll /status/{job_id} for progress.
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_DOCS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type {ext!r}. Supported: {list(SUPPORTED_DOCS)}"
        )

    # Check file size
    contents = await file.read()
    if len(contents) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max {MAX_FILE_MB}MB."
        )

    # Save to disk
    save_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    with open(save_path, "wb") as f:
        f.write(contents)

    job_id = new_job("pdf", file.filename)
    background_tasks.add_task(
        run_pdf_job, job_id, str(save_path), session_id
    )

    return {
        "job_id":     job_id,
        "session_id": session_id,
        "file_name":  file.filename,
        "file_size":  len(contents),
        "message":    "Processing started. Poll /status/{job_id} for progress.",
    }


@app.post("/process/image")
async def process_image_endpoint(
    background_tasks: BackgroundTasks,
    file:       UploadFile = File(...),
    session_id: str        = Form(...),
    title:      str        = Form(default=""),
):
    """
    Upload and OCR an image (handwritten notes, whiteboard photo, scanned page).
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_IMGS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type {ext!r}. Supported: {list(SUPPORTED_IMGS)}"
        )

    contents  = await file.read()
    if len(contents) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large. Max {MAX_FILE_MB}MB."
        )
    save_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    with open(save_path, "wb") as f:
        f.write(contents)

    job_id = new_job("image", file.filename)
    background_tasks.add_task(
        run_image_job, job_id, str(save_path), session_id, title
    )

    return {
        "job_id":     job_id,
        "session_id": session_id,
        "file_name":  file.filename,
        "message":    "OCR started. Poll /status/{job_id} for progress.",
    }


class YouTubeRequest(BaseModel):
    url:        str
    session_id: str


@app.post("/process/youtube")
async def process_youtube(req: YouTubeRequest, background_tasks: BackgroundTasks):
    """
    Process a YouTube, Instagram, Facebook, or Twitter video URL.
    Downloads audio, transcribes, translates if needed, chunks.
    """
    if not req.url.startswith("http"):
        raise HTTPException(status_code=400, detail="Invalid URL")

    job_id = new_job("youtube", req.url)
    background_tasks.add_task(
        run_youtube_job, job_id, req.url, req.session_id
    )

    return {
        "job_id":     job_id,
        "session_id": req.session_id,
        "url":        req.url,
        "message":    "Transcription started. Poll /status/{job_id} for progress.",
    }


@app.post("/process/batch")
async def process_batch(
    background_tasks: BackgroundTasks,
    files:      List[UploadFile] = File(...),
    session_id: str              = Form(...),
):
    """
    Upload multiple files at once (PDFs + images mixed).
    All processed sequentially in background, all sent to master.
    """
    saved_paths = []
    total_size  = 0

    for file in files:
        ext = Path(file.filename).suffix.lower()

        if ext not in SUPPORTED_DOCS | SUPPORTED_IMGS:
            print(f"  ⚠️  Skipping unsupported: {file.filename}")
            continue

        # Stream file to disk in chunks — never loads whole batch into RAM
        save_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
        file_size = 0
        with open(save_path, "wb") as out:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                file_size  += len(chunk)
                total_size += len(chunk)
                if file_size > MAX_FILE_MB * 1024 * 1024:
                    out.close()
                    save_path.unlink(missing_ok=True)
                    print(f"  ⚠️  Skipping {file.filename} — exceeds {MAX_FILE_MB}MB limit")
                    break
                out.write(chunk)
            else:
                saved_paths.append(str(save_path))

    if not saved_paths:
        raise HTTPException(status_code=400, detail="No supported files found in batch")

    job_id = new_job("batch", f"{len(saved_paths)} files")
    background_tasks.add_task(
        run_batch_job, job_id, saved_paths, session_id
    )

    return {
        "job_id":      job_id,
        "session_id":  session_id,
        "file_count":  len(saved_paths),
        "total_size":  total_size,
        "message":    f"Batch processing {len(saved_paths)} files. Poll /status/{job_id}.",
    }


# ─────────────────────────────────────────────
# DEV: run directly
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("=" * 55)
    print("  🔧  StudyMind AI — LONEXFURY Processor")
    print("=" * 55)
    print(f"  Master node : {MASTER_URL}")
    print(f"  Upload dir  : {UPLOAD_DIR.resolve()}")
    print(f"  Docs        : {SUPPORTED_DOCS}")
    print(f"  Images      : {SUPPORTED_IMGS}")
    print()
    uvicorn.run("processor:app", host="0.0.0.0", port=8001, reload=True)