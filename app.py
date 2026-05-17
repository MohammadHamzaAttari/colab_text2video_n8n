"""
Colab Text-to-Video API for n8n - PRODUCTION QUALITY
Supports multiple models optimized for YouTube automation
"""
import os
import re
import time
import uuid
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

# -------- Config --------
# Use ZeroScope for better quality (supports 16:9 aspect ratio)
MODEL_ID = os.environ.get("MODEL_ID", "cerspense/zeroscope_v2_576w")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/content/generated_videos"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Colab Text-to-Video API", version="2.0.0")
app.mount("/files", StaticFiles(directory=str(OUTPUT_DIR)), name="files")

pipe = None
jobs = {}


def slugify(value: str, fallback: str = "video") -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-._")
    return value[:80] or fallback


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=3, description="Text prompt for video generation")
    negative_prompt: str = "low quality, blurry, distorted, watermark, text, logo, bad anatomy, worst quality, low resolution"
    seed: Optional[int] = Field(None, description="Optional deterministic seed")
    num_frames: int = Field(24, ge=8, le=32)
    num_inference_steps: int = Field(40, ge=10, le=100)
    guidance_scale: float = Field(17.5, ge=1.0, le=30.0)
    height: int = Field(320, ge=128, le=576)
    width: int = Field(576, ge=128, le=1024)
    fps: int = Field(8, ge=4, le=30)
    subfolder: str = Field("documentary_series", description="Folder under generated_videos")
    filename_prefix: str = Field("clip", description="Output filename prefix")


class GenerateResponse(BaseModel):
    job_id: str
    status: str
    file_name: Optional[str] = None
    file_path: Optional[str] = None
    video_url: Optional[str] = None
    error: Optional[str] = None
    seed: Optional[int] = None
    generation_time: Optional[float] = None


def get_base_url(request: Request) -> str:
    """Return the public URL seen by n8n through Cloudflare/ngrok/reverse proxy."""
    env_url = os.environ.get("PUBLIC_BASE_URL")
    if env_url:
        return env_url.rstrip("/")
    host = request.headers.get("x-forwarded-host") or request.headers.get("host") or "127.0.0.1:8000"
    proto = request.headers.get("x-forwarded-proto")
    if not proto:
        proto = "https" if "trycloudflare.com" in host or "ngrok" in host else request.url.scheme
    return f"{proto}://{host}".rstrip("/")


def get_pipeline():
    global pipe
    if pipe is not None:
        return pipe

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU found. In Colab select Runtime > Change runtime type > T4 GPU.")

    dtype = torch.float16
    
    # Load model with optimizations
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
    )
    
    # Better scheduler for quality
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    # Memory optimizations
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    
    # Additional quality boost
    try:
        pipe.enable_attention_slicing(1)
    except Exception:
        pass
    
    return pipe


def generate_video(req: GenerateRequest, base_url: str, job_id: Optional[str] = None):
    job_id = job_id or str(uuid.uuid4())
    start_time = time.time()
    jobs[job_id] = {"status": "running", "created_at": start_time, "prompt": req.prompt}
    
    try:
        p = get_pipeline()
        seed = req.seed if req.seed is not None else int(time.time()) % 2_147_483_647
        generator = torch.Generator(device="cuda").manual_seed(seed)

        folder = OUTPUT_DIR / slugify(req.subfolder, "documentary_series")
        folder.mkdir(parents=True, exist_ok=True)
        file_name = f"{slugify(req.filename_prefix, 'clip')}-{job_id[:8]}-seed{seed}.mp4"
        out_path = folder / file_name

        # Generate with quality settings
        with torch.inference_mode():
            result = p(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                num_frames=req.num_frames,
                num_inference_steps=req.num_inference_steps,
                guidance_scale=req.guidance_scale,
                height=req.height,
                width=req.width,
                generator=generator,
            )
        
        frames = result.frames[0]
        
        # Export with higher quality settings
        export_to_video(frames, str(out_path), fps=req.fps)

        generation_time = time.time() - start_time
        rel = out_path.relative_to(OUTPUT_DIR).as_posix()
        video_url = f"{base_url.rstrip('/')}/files/{rel}"
        
        jobs[job_id] = {
            "status": "completed",
            "created_at": jobs[job_id]["created_at"],
            "prompt": req.prompt,
            "seed": seed,
            "file_name": file_name,
            "file_path": str(out_path),
            "video_url": video_url,
            "generation_time": round(generation_time, 2),
        }
    except Exception as e:
        jobs[job_id] = {
            "status": "failed",
            "created_at": jobs.get(job_id, {}).get("created_at", time.time()),
            "prompt": req.prompt,
            "error": repr(e),
        }
    
    return jobs[job_id]


@app.get("/")
def root():
    return {
        "ok": True,
        "model": MODEL_ID,
        "endpoints": {
            "sync_generation": "POST /generate_sync",
            "async_generation": "POST /generate",
            "job_status": "GET /jobs/{job_id}",
            "files": "GET /files/{subfolder}/{filename}.mp4",
        },
    }


@app.get("/health")
def health():
    return {"ok": True, "cuda": torch.cuda.is_available(), "model_loaded": pipe is not None, "model": MODEL_ID}


@app.post("/generate_sync", response_model=GenerateResponse)
def generate_sync(req: GenerateRequest, request: Request):
    """Generate video and wait until finished. Easiest endpoint for n8n."""
    base_url = get_base_url(request)
    job_id = str(uuid.uuid4())
    data = generate_video(req, base_url=base_url, job_id=job_id)
    return GenerateResponse(job_id=job_id, **data)


@app.post("/generate", response_model=GenerateResponse)
def generate_async(req: GenerateRequest, request: Request, background_tasks: BackgroundTasks):
    """Start generation in background. Use GET /jobs/{job_id} to poll."""
    base_url = get_base_url(request)
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "created_at": time.time(), "prompt": req.prompt}
    background_tasks.add_task(generate_video, req, base_url, job_id)
    return GenerateResponse(job_id=job_id, status="queued")


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/download/{subfolder}/{filename}")
def download(subfolder: str, filename: str):
    path = OUTPUT_DIR / slugify(subfolder) / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(path), media_type="video/mp4", filename=filename)