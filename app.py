"""
Colab Text-to-Video API - LTX-Video Edition
Best quality open-source model for Google Colab T4 GPU
Model: Lightricks/LTX-Video (LTXV-13B-0.9.7-distilled)
"""
import gc
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Optional

import torch
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ─────────────────────────── Config ───────────────────────────
# LTX-Video: best quality model that fits on T4
# Distilled variant = fewer steps needed (4-8 steps vs 50)
MODEL_ID = os.environ.get(
    "MODEL_ID",
    "Lightricks/LTX-Video",
)

# Distilled checkpoint - faster + quality boost
CKPT_ID = os.environ.get(
    "CKPT_ID",
    "Lightricks/LTXV-13B-0.9.7-distilled",
)

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/content/generated_videos"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_JOBS_HISTORY = 100  # keep last N jobs in memory

app = FastAPI(title="LTX-Video Documentary API", version="3.0.0")
app.mount("/files", StaticFiles(directory=str(OUTPUT_DIR)), name="files")

_pipe = None
_jobs: dict[str, dict] = {}


# ─────────────────────────── Helpers ──────────────────────────
def slugify(value: str, fallback: str = "video") -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-._")
    return value[:80] or fallback


def free_memory():
    """Aggressively free GPU memory between generations."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def get_vram_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / 1e9
    return 0.0


def get_base_url(request: Request) -> str:
    env_url = os.environ.get("PUBLIC_BASE_URL")
    if env_url:
        return env_url.rstrip("/")
    host = (
        request.headers.get("x-forwarded-host")
        or request.headers.get("host")
        or "127.0.0.1:8000"
    )
    proto = request.headers.get("x-forwarded-proto")
    if not proto:
        proto = (
            "https"
            if "trycloudflare.com" in host or "ngrok" in host
            else request.url.scheme
        )
    return f"{proto}://{host}".rstrip("/")


# ─────────────────────────── Schema ───────────────────────────
class GenerateRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length=3,
        description="Detailed text prompt. More detail = better video.",
    )
    negative_prompt: str = Field(
        default=(
            "worst quality, inconsistent motion, blurry, jittery, distorted, "
            "watermark, text, logo, low resolution, static, overexposed, "
            "underexposed, duplicate frames, flickering"
        ),
    )
    seed: Optional[int] = Field(None, description="Reproducible seed")

    # LTX-Video distilled works great with 4-8 steps
    # Full model needs 40-50 steps
    num_frames: int = Field(
        default=97,
        ge=9,
        le=257,
        description="Must be N*8+1 (e.g. 9,17,25,49,97,121,161,257)",
    )
    num_inference_steps: int = Field(
        default=6,
        ge=4,
        le=50,
        description="4-8 for distilled model. 40-50 for full model.",
    )
    guidance_scale: float = Field(
        default=1.0,
        ge=1.0,
        le=10.0,
        description="1.0 for distilled. 3.5-7.0 for full model.",
    )
    # LTX-Video native resolution - must be divisible by 32
    height: int = Field(
        default=480,
        ge=256,
        le=720,
        description="Must be divisible by 32. Max 720 on T4.",
    )
    width: int = Field(
        default=704,
        ge=256,
        le=1280,
        description="Must be divisible by 32. Max 1280 on T4.",
    )
    fps: int = Field(default=24, ge=8, le=30)

    subfolder: str = Field(
        default="documentary",
        description="Subdirectory under generated_videos/",
    )
    filename_prefix: str = Field(
        default="clip",
        description="Filename prefix for the output MP4.",
    )

    # LTX-Video specific enhancement
    enhance_prompt: bool = Field(
        default=True,
        description="Use built-in prompt enhancer for better results.",
    )


class GenerateResponse(BaseModel):
    job_id: str
    status: str
    file_name: Optional[str] = None
    file_path: Optional[str] = None
    video_url: Optional[str] = None
    error: Optional[str] = None
    seed: Optional[int] = None
    generation_time: Optional[float] = None
    vram_used_gb: Optional[float] = None


# ─────────────────────────── Pipeline ─────────────────────────
def get_pipeline():
    """
    Load LTX-Video pipeline with T4-optimised settings.
    Auto-detects distilled vs full checkpoint.
    """
    global _pipe
    if _pipe is not None:
        return _pipe

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU detected. "
            "In Colab: Runtime > Change runtime type > T4 GPU"
        )

    vram = get_vram_gb()
    log.info("GPU VRAM: %.1f GB", vram)

    from diffusers import LTXPipeline, LTXVideoTransformer3DModel
    from transformers import T5EncoderModel

    log.info("Loading LTX-Video model: %s", MODEL_ID)
    log.info("Checkpoint: %s", CKPT_ID)

    is_distilled = "distilled" in CKPT_ID.lower()

    # Load transformer from distilled checkpoint for quality boost
    if is_distilled:
        log.info("Using distilled transformer checkpoint...")
        transformer = LTXVideoTransformer3DModel.from_pretrained(
            CKPT_ID,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
    else:
        transformer = None

    # Load text encoder in 8-bit to save ~4GB VRAM
    log.info("Loading T5 text encoder (quantised)...")
    try:
        text_encoder = T5EncoderModel.from_pretrained(
            MODEL_ID,
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16,
            load_in_8bit=True,
            device_map="auto",
        )
        log.info("T5 loaded in 8-bit mode - saved ~4GB VRAM")
    except Exception as e:
        log.warning("8-bit load failed (%s), loading in bfloat16...", e)
        text_encoder = T5EncoderModel.from_pretrained(
            MODEL_ID,
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16,
        )

    # Assemble pipeline
    pipe_kwargs = dict(
        torch_dtype=torch.bfloat16,
        text_encoder=text_encoder,
    )
    if transformer is not None:
        pipe_kwargs["transformer"] = transformer

    _pipe = LTXPipeline.from_pretrained(MODEL_ID, **pipe_kwargs)

    # ── Memory optimisations (order matters) ──────────────────
    # 1. CPU offload - moves model parts to CPU when not in use
    _pipe.enable_model_cpu_offload()

    # 2. VAE tiling - process video in tiles to save VRAM
    _pipe.vae.enable_tiling()

    # 3. VAE slicing - process batch slices sequentially
    _pipe.vae.enable_slicing()

    # 4. Attention slicing - chunk attention computation
    try:
        _pipe.enable_attention_slicing(1)
    except Exception:
        pass

    # 5. xFormers memory efficient attention
    try:
        _pipe.enable_xformers_memory_efficient_attention()
        log.info("xFormers enabled")
    except Exception:
        log.info("xFormers not available, using PyTorch SDPA")

    log.info("LTX-Video pipeline ready!")
    log.info("VRAM after load: %.1f GB used", torch.cuda.memory_allocated() / 1e9)
    return _pipe


def _round_to_valid_frames(n: int) -> int:
    """
    LTX-Video requires num_frames = N*8 + 1.
    Round to nearest valid value.
    """
    valid = [9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97,
             105, 113, 121, 129, 137, 145, 153, 161, 169, 177,
             185, 193, 201, 209, 217, 225, 233, 241, 249, 257]
    return min(valid, key=lambda x: abs(x - n))


def _round_to_32(n: int) -> int:
    """LTX-Video requires height/width divisible by 32."""
    return max(32, (n // 32) * 32)


# ─────────────────────────── Generation ───────────────────────
def generate_video(
    req: GenerateRequest,
    base_url: str,
    job_id: Optional[str] = None,
) -> dict:
    job_id = job_id or str(uuid.uuid4())
    start_time = time.time()

    _jobs[job_id] = {
        "status": "running",
        "created_at": start_time,
        "prompt": req.prompt,
    }

    try:
        p = get_pipeline()

        # ── Validate / normalise params ───────────────────────
        seed = (
            req.seed
            if req.seed is not None
            else int(time.time()) % 2_147_483_647
        )
        num_frames = _round_to_valid_frames(req.num_frames)
        height = _round_to_32(req.height)
        width = _round_to_32(req.width)

        if num_frames != req.num_frames:
            log.info("num_frames adjusted %d → %d", req.num_frames, num_frames)
        if height != req.height or width != req.width:
            log.info("Resolution adjusted to %dx%d", width, height)

        # ── Output path ───────────────────────────────────────
        folder = OUTPUT_DIR / slugify(req.subfolder, "documentary")
        folder.mkdir(parents=True, exist_ok=True)
        file_name = (
            f"{slugify(req.filename_prefix, 'clip')}"
            f"-{job_id[:8]}-seed{seed}.mp4"
        )
        out_path = folder / file_name

        # ── Prompt enhancement ────────────────────────────────
        prompt = req.prompt
        if req.enhance_prompt:
            # Append cinematic quality tags for documentary style
            cinematic_tags = (
                ", cinematic, photorealistic, 8K, high detail, "
                "natural lighting, documentary style, realistic motion, "
                "sharp focus, professional cinematography"
            )
            if not any(t in prompt.lower() for t in ["cinematic", "photorealistic", "8k"]):
                prompt = prompt + cinematic_tags
                log.info("Prompt enhanced with cinematic tags")

        log.info(
            "Generating: frames=%d steps=%d guidance=%.1f res=%dx%d seed=%d",
            num_frames,
            req.num_inference_steps,
            req.guidance_scale,
            width,
            height,
            seed,
        )

        # ── Run inference ─────────────────────────────────────
        generator = torch.Generator(device="cuda").manual_seed(seed)
        free_memory()

        vram_before = torch.cuda.memory_allocated() / 1e9

        with torch.inference_mode():
            result = p(
                prompt=prompt,
                negative_prompt=req.negative_prompt,
                num_frames=num_frames,
                num_inference_steps=req.num_inference_steps,
                guidance_scale=req.guidance_scale,
                height=height,
                width=width,
                generator=generator,
            )

        vram_peak = torch.cuda.max_memory_allocated() / 1e9

        # ── Export video ──────────────────────────────────────
        # LTX-Video returns frames as list of PIL images
        frames = result.frames[0]

        # Use imageio for high-quality H.264 export
        _export_video_hq(frames, str(out_path), fps=req.fps)

        free_memory()
        generation_time = time.time() - start_time

        rel = out_path.relative_to(OUTPUT_DIR).as_posix()
        video_url = f"{base_url.rstrip('/')}/files/{rel}"

        log.info(
            "Done in %.1fs | VRAM peak: %.1fGB | %s",
            generation_time,
            vram_peak,
            file_name,
        )

        result_data = {
            "status": "completed",
            "created_at": start_time,
            "prompt": req.prompt,
            "seed": seed,
            "file_name": file_name,
            "file_path": str(out_path),
            "video_url": video_url,
            "generation_time": round(generation_time, 2),
            "vram_used_gb": round(vram_peak, 2),
        }
        _jobs[job_id] = result_data

        # Prune old jobs
        if len(_jobs) > MAX_JOBS_HISTORY:
            oldest = sorted(_jobs, key=lambda k: _jobs[k].get("created_at", 0))
            for old_key in oldest[: len(_jobs) - MAX_JOBS_HISTORY]:
                _jobs.pop(old_key, None)

        return result_data

    except torch.cuda.OutOfMemoryError:
        free_memory()
        msg = (
            "CUDA out of memory. "
            "Reduce width/height/num_frames and retry. "
            f"Try: width=512, height=288, num_frames=49"
        )
        log.error(msg)
        _jobs[job_id] = {
            "status": "failed",
            "created_at": _jobs.get(job_id, {}).get("created_at", time.time()),
            "prompt": req.prompt,
            "error": msg,
        }
        return _jobs[job_id]

    except Exception as exc:
        free_memory()
        log.exception("Generation failed")
        _jobs[job_id] = {
            "status": "failed",
            "created_at": _jobs.get(job_id, {}).get("created_at", time.time()),
            "prompt": req.prompt,
            "error": repr(exc),
        }
        return _jobs[job_id]


def _export_video_hq(frames, path: str, fps: int = 24):
    """
    Export frames to H.264 MP4 with high quality settings.
    Falls back gracefully if ffmpeg options not supported.
    """
    import imageio
    import numpy as np

    # Convert PIL images to numpy arrays
    np_frames = []
    for f in frames:
        if hasattr(f, "numpy"):
            arr = f.numpy()
        else:
            arr = np.array(f)
        # Ensure uint8
        if arr.dtype != np.uint8:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        np_frames.append(arr)

    try:
        # High quality H.264 with good bitrate for documentary
        writer = imageio.get_writer(
            path,
            fps=fps,
            codec="libx264",
            quality=None,
            output_params=[
                "-crf", "18",          # Near-lossless (0=lossless, 51=worst)
                "-preset", "slow",     # Better compression
                "-pix_fmt", "yuv420p", # Wide compatibility
                "-movflags", "+faststart",  # Web-optimized
            ],
        )
        for frame in np_frames:
            writer.append_data(frame)
        writer.close()
        log.info("Exported %d frames to %s (HQ H.264)", len(np_frames), path)

    except Exception as e:
        log.warning("HQ export failed (%s), using standard export...", e)
        # Fallback to standard export
        writer = imageio.get_writer(path, fps=fps, quality=8)
        for frame in np_frames:
            writer.append_data(frame)
        writer.close()


# ─────────────────────────── Routes ───────────────────────────
@app.get("/")
def root():
    vram = get_vram_gb()
    loaded = _pipe is not None
    return {
        "ok": True,
        "model": MODEL_ID,
        "checkpoint": CKPT_ID,
        "gpu_vram_gb": round(vram, 1),
        "model_loaded": loaded,
        "recommended_settings": {
            "distilled_model": {
                "num_frames": 97,
                "num_inference_steps": 6,
                "guidance_scale": 1.0,
                "width": 704,
                "height": 480,
                "fps": 24,
                "note": "Fast generation, excellent quality",
            },
            "memory_safe": {
                "num_frames": 49,
                "num_inference_steps": 6,
                "guidance_scale": 1.0,
                "width": 512,
                "height": 288,
                "fps": 24,
                "note": "Use if OOM errors occur",
            },
        },
        "endpoints": {
            "sync": "POST /generate_sync",
            "async": "POST /generate",
            "status": "GET /jobs/{job_id}",
            "files": "GET /files/{subfolder}/{filename}.mp4",
            "health": "GET /health",
        },
    }


@app.get("/health")
def health():
    vram_total = get_vram_gb()
    vram_used = (
        torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    )
    return {
        "ok": True,
        "cuda": torch.cuda.is_available(),
        "vram_total_gb": round(vram_total, 1),
        "vram_used_gb": round(vram_used, 2),
        "vram_free_gb": round(vram_total - vram_used, 2),
        "model_loaded": _pipe is not None,
        "model": MODEL_ID,
        "checkpoint": CKPT_ID,
        "jobs_tracked": len(_jobs),
    }


@app.post("/generate_sync", response_model=GenerateResponse)
def generate_sync(req: GenerateRequest, request: Request):
    """
    Synchronous generation - blocks until video is ready.
    Best for n8n simple HTTP request nodes.
    Timeout: set n8n HTTP node timeout to 600000ms (10 min).
    """
    base_url = get_base_url(request)
    job_id = str(uuid.uuid4())
    data = generate_video(req, base_url=base_url, job_id=job_id)
    return GenerateResponse(job_id=job_id, **{
        k: v for k, v in data.items()
        if k in GenerateResponse.model_fields
    })


@app.post("/generate", response_model=GenerateResponse)
def generate_async(
    req: GenerateRequest,
    request: Request,
    background_tasks: BackgroundTasks,
):
    """
    Async generation - returns immediately with job_id.
    Poll GET /jobs/{job_id} every 20-30 seconds.
    """
    base_url = get_base_url(request)
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "status": "queued",
        "created_at": time.time(),
        "prompt": req.prompt,
    }
    background_tasks.add_task(generate_video, req, base_url, job_id)
    return GenerateResponse(job_id=job_id, status="queued")


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _jobs[job_id]


@app.get("/jobs")
def list_jobs():
    """List all tracked jobs with their status."""
    return {
        "total": len(_jobs),
        "jobs": {
            jid: {
                "status": j.get("status"),
                "created_at": j.get("created_at"),
                "generation_time": j.get("generation_time"),
            }
            for jid, j in sorted(
                _jobs.items(),
                key=lambda x: x[1].get("created_at", 0),
                reverse=True,
            )[:20]
        },
    }


@app.get("/download/{subfolder}/{filename}")
def download(subfolder: str, filename: str):
    path = OUTPUT_DIR / slugify(subfolder) / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(path), media_type="video/mp4", filename=filename)


@app.post("/warmup")
def warmup(request: Request):
    """
    Pre-load the model into VRAM.
    Call this once after startup to avoid cold-start on first generation.
    """
    try:
        get_pipeline()
        return {
            "ok": True,
            "model_loaded": True,
            "vram_used_gb": round(
                torch.cuda.memory_allocated() / 1e9, 2
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))