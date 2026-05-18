"""
Colab Text-to-Video API - LTX-Video Edition
Model: Lightricks/LTX-Video
Optimised for Google Colab T4 GPU (15 GB VRAM)
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
# FIXED: Only one real HuggingFace repo exists: Lightricks/LTX-Video
# The distilled transformer lives INSIDE this repo as a variant file
# There is NO separate "LTXV-13B-0.9.7-distilled" repo on HuggingFace
MODEL_ID = os.environ.get("MODEL_ID", "Lightricks/LTX-Video")

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/content/generated_videos"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_JOBS_HISTORY = 100

app = FastAPI(title="LTX-Video Documentary API", version="4.0.0")
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
    num_frames: int = Field(
        default=97,
        ge=9,
        le=257,
        description="Must be N*8+1 (e.g. 9,17,25,49,97,121,161,257)",
    )
    num_inference_steps: int = Field(
        default=6,
        ge=1,
        le=50,
        description="4-8 for distilled model. 40-50 for full model.",
    )
    guidance_scale: float = Field(
        default=1.0,
        ge=1.0,
        le=10.0,
        description="1.0 for distilled. 3.5-7.0 for full model.",
    )
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
    enhance_prompt: bool = Field(
        default=True,
        description="Append cinematic quality tags to prompt.",
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
    Load LTX-Video pipeline.

    Strategy (tries in order, falls back gracefully):
      1. Load transformer with variant='distilled'  (best quality)
      2. Load transformer without variant           (standard quality)
      3. Load full pipeline without custom transformer (safe fallback)

    Text encoder is loaded in 8-bit to save ~4 GB VRAM.
    """
    global _pipe
    if _pipe is not None:
        return _pipe

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU detected. "
            "Go to: Runtime > Change runtime type > T4 GPU"
        )

    log.info("GPU VRAM total: %.1f GB", get_vram_gb())
    log.info("Loading pipeline from: %s", MODEL_ID)

    from diffusers import LTXPipeline, LTXVideoTransformer3DModel
    from transformers import T5EncoderModel

    # ── Step 1: Load text encoder (8-bit to save VRAM) ────────
    log.info("Loading T5 text encoder...")
    text_encoder = _load_text_encoder_safe()

    # ── Step 2: Try to load distilled transformer ──────────────
    transformer = _load_transformer_safe()

    # ── Step 3: Assemble pipeline ──────────────────────────────
    pipe_kwargs: dict = {
        "torch_dtype": torch.bfloat16,
        "text_encoder": text_encoder,
    }
    if transformer is not None:
        pipe_kwargs["transformer"] = transformer

    log.info("Assembling LTXPipeline...")
    _pipe = LTXPipeline.from_pretrained(MODEL_ID, **pipe_kwargs)

    # ── Step 4: Memory optimisations ──────────────────────────
    _pipe.enable_model_cpu_offload()
    _pipe.vae.enable_tiling()
    _pipe.vae.enable_slicing()

    try:
        _pipe.enable_attention_slicing(1)
    except Exception:
        pass

    try:
        _pipe.enable_xformers_memory_efficient_attention()
        log.info("xFormers memory efficient attention enabled")
    except Exception:
        log.info("xFormers unavailable — using PyTorch scaled dot-product attention")

    log.info("Pipeline ready! VRAM used: %.2f GB",
             torch.cuda.memory_allocated() / 1e9)
    return _pipe


def _load_text_encoder_safe():
    """Load T5 text encoder, trying 8-bit first then bfloat16."""
    from transformers import T5EncoderModel

    # Try 8-bit quantisation first (saves ~4 GB)
    try:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        encoder = T5EncoderModel.from_pretrained(
            MODEL_ID,
            subfolder="text_encoder",
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        log.info("T5 text encoder loaded in 8-bit (saved ~4 GB VRAM)")
        return encoder
    except Exception as e:
        log.warning("8-bit text encoder failed (%s) — trying bfloat16...", e)

    # Fallback: plain bfloat16
    try:
        encoder = T5EncoderModel.from_pretrained(
            MODEL_ID,
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16,
        )
        log.info("T5 text encoder loaded in bfloat16")
        return encoder
    except Exception as e:
        log.warning("bfloat16 text encoder failed (%s) — pipeline will use default", e)
        return None


def _load_transformer_safe():
    """
    Try to load the distilled transformer variant.
    The distilled weights are stored inside Lightricks/LTX-Video
    as transformer/diffusion_pytorch_model.distilled.safetensors
    Accessed via variant='distilled' in diffusers >= 0.30.
    """
    from diffusers import LTXVideoTransformer3DModel

    # Attempt 1: distilled variant (diffusers >= 0.30)
    try:
        transformer = LTXVideoTransformer3DModel.from_pretrained(
            MODEL_ID,
            subfolder="transformer",
            variant="distilled",
            torch_dtype=torch.bfloat16,
        )
        log.info("Distilled transformer loaded via variant='distilled'")
        return transformer
    except Exception as e:
        log.warning("Distilled variant load failed (%s)", e)

    # Attempt 2: standard transformer (no variant)
    try:
        transformer = LTXVideoTransformer3DModel.from_pretrained(
            MODEL_ID,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
        log.info("Standard transformer loaded (no variant)")
        return transformer
    except Exception as e:
        log.warning(
            "Transformer pre-load failed (%s) — "
            "pipeline will load its own transformer", e
        )
        return None


# ─────────────────────────── Frame / Resolution Utils ─────────
def _round_to_valid_frames(n: int) -> int:
    """LTX-Video requires num_frames = N*8 + 1."""
    valid = [
        9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97,
        105, 113, 121, 129, 137, 145, 153, 161, 169, 177,
        185, 193, 201, 209, 217, 225, 233, 241, 249, 257,
    ]
    return min(valid, key=lambda x: abs(x - n))


def _round_to_32(n: int) -> int:
    """LTX-Video requires height/width divisible by 32."""
    return max(256, (n // 32) * 32)


# ─────────────────────────── Video Export ─────────────────────
def _export_video_hq(frames, path: str, fps: int = 24):
    """Export frames to H.264 MP4. Falls back to standard if HQ fails."""
    import imageio
    import numpy as np

    np_frames = []
    for f in frames:
        arr = f.numpy() if hasattr(f, "numpy") else np.array(f)
        if arr.dtype != np.uint8:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        np_frames.append(arr)

    # High-quality H.264 attempt
    try:
        writer = imageio.get_writer(
            path,
            fps=fps,
            codec="libx264",
            quality=None,
            output_params=[
                "-crf", "18",
                "-preset", "slow",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
            ],
        )
        for frame in np_frames:
            writer.append_data(frame)
        writer.close()
        log.info("Exported %d frames → %s (HQ H.264)", len(np_frames), path)
        return
    except Exception as e:
        log.warning("HQ export failed (%s) — using standard export", e)

    # Standard fallback
    writer = imageio.get_writer(path, fps=fps, quality=8)
    for frame in np_frames:
        writer.append_data(frame)
    writer.close()
    log.info("Exported %d frames → %s (standard)", len(np_frames), path)


# ─────────────────────────── Core Generation ──────────────────
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

        # ── Normalise params ──────────────────────────────────
        seed = (
            req.seed
            if req.seed is not None
            else int(time.time()) % 2_147_483_647
        )
        num_frames = _round_to_valid_frames(req.num_frames)
        height     = _round_to_32(req.height)
        width      = _round_to_32(req.width)

        if num_frames != req.num_frames:
            log.info("num_frames %d → %d (nearest valid)", req.num_frames, num_frames)
        if height != req.height or width != req.width:
            log.info("Resolution adjusted → %dx%d", width, height)

        # ── Output path ───────────────────────────────────────
        folder = OUTPUT_DIR / slugify(req.subfolder, "documentary")
        folder.mkdir(parents=True, exist_ok=True)
        file_name = (
            f"{slugify(req.filename_prefix, 'clip')}"
            f"-{job_id[:8]}-seed{seed}.mp4"
        )
        out_path = folder / file_name

        # ── Optional prompt enhancement ───────────────────────
        prompt = req.prompt
        if req.enhance_prompt:
            cinematic_tags = (
                ", cinematic, photorealistic, 8K, high detail, "
                "natural lighting, documentary style, realistic motion, "
                "sharp focus, professional cinematography"
            )
            if not any(
                t in prompt.lower()
                for t in ["cinematic", "photorealistic", "8k"]
            ):
                prompt = prompt + cinematic_tags
                log.info("Prompt enhanced with cinematic tags")

        log.info(
            "Generating | frames=%d steps=%d guidance=%.1f res=%dx%d seed=%d",
            num_frames, req.num_inference_steps,
            req.guidance_scale, width, height, seed,
        )

        # ── Inference ─────────────────────────────────────────
        generator = torch.Generator(device="cuda").manual_seed(seed)
        free_memory()
        torch.cuda.reset_peak_memory_stats()

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

        # ── Export ────────────────────────────────────────────
        frames = result.frames[0]
        _export_video_hq(frames, str(out_path), fps=req.fps)
        free_memory()

        generation_time = time.time() - start_time
        rel       = out_path.relative_to(OUTPUT_DIR).as_posix()
        video_url = f"{base_url.rstrip('/')}/files/{rel}"

        log.info(
            "✅ Done in %.1fs | VRAM peak: %.2fGB | %s",
            generation_time, vram_peak, file_name,
        )

        result_data = {
            "status":          "completed",
            "created_at":      start_time,
            "prompt":          req.prompt,
            "seed":            seed,
            "file_name":       file_name,
            "file_path":       str(out_path),
            "video_url":       video_url,
            "generation_time": round(generation_time, 2),
            "vram_used_gb":    round(vram_peak, 2),
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
            "Reduce resolution or num_frames. "
            "Safe fallback: width=512, height=288, num_frames=49"
        )
        log.error(msg)
        err_data = {
            "status":     "failed",
            "created_at": _jobs.get(job_id, {}).get("created_at", time.time()),
            "prompt":     req.prompt,
            "error":      msg,
        }
        _jobs[job_id] = err_data
        return err_data

    except Exception as exc:
        free_memory()
        log.exception("Generation failed")
        err_data = {
            "status":     "failed",
            "created_at": _jobs.get(job_id, {}).get("created_at", time.time()),
            "prompt":     req.prompt,
            "error":      repr(exc),
        }
        _jobs[job_id] = err_data
        return err_data


# ─────────────────────────── Routes ───────────────────────────
@app.get("/")
def root():
    return {
        "ok":           True,
        "model":        MODEL_ID,
        "variant":      "distilled (auto-detected at load time)",
        "gpu_vram_gb":  round(get_vram_gb(), 1),
        "model_loaded": _pipe is not None,
        "recommended_settings": {
            "standard": {
                "num_frames": 97,
                "num_inference_steps": 6,
                "guidance_scale": 1.0,
                "width": 704,
                "height": 480,
                "fps": 24,
                "note": "~4s video, distilled model",
            },
            "memory_safe": {
                "num_frames": 49,
                "num_inference_steps": 6,
                "guidance_scale": 1.0,
                "width": 512,
                "height": 288,
                "fps": 24,
                "note": "Use if CUDA OOM errors occur",
            },
        },
        "endpoints": {
            "sync":     "POST /generate_sync",
            "async":    "POST /generate",
            "status":   "GET /jobs/{job_id}",
            "list":     "GET /jobs",
            "files":    "GET /files/{subfolder}/{filename}.mp4",
            "download": "GET /download/{subfolder}/{filename}",
            "health":   "GET /health",
            "warmup":   "POST /warmup",
        },
    }


@app.get("/health")
def health():
    vram_total = get_vram_gb()
    vram_used  = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    return {
        "ok":            True,
        "cuda":          torch.cuda.is_available(),
        "vram_total_gb": round(vram_total, 1),
        "vram_used_gb":  round(vram_used, 2),
        "vram_free_gb":  round(vram_total - vram_used, 2),
        "model_loaded":  _pipe is not None,
        "model":         MODEL_ID,
        "jobs_tracked":  len(_jobs),
    }


@app.post("/generate_sync", response_model=GenerateResponse)
def generate_sync(req: GenerateRequest, request: Request):
    """
    Synchronous generation — blocks until the video is ready.
    Set n8n HTTP node timeout to 600000 ms (10 min).
    """
    base_url = get_base_url(request)
    job_id   = str(uuid.uuid4())
    data     = generate_video(req, base_url=base_url, job_id=job_id)
    return GenerateResponse(
        job_id=job_id,
        **{k: v for k, v in data.items() if k in GenerateResponse.model_fields},
    )


@app.post("/generate", response_model=GenerateResponse)
def generate_async(
    req: GenerateRequest,
    request: Request,
    background_tasks: BackgroundTasks,
):
    """
    Async generation — returns immediately with job_id.
    Poll GET /jobs/{job_id} every 20-30 seconds.
    """
    base_url = get_base_url(request)
    job_id   = str(uuid.uuid4())
    _jobs[job_id] = {
        "status":     "queued",
        "created_at": time.time(),
        "prompt":     req.prompt,
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
    return {
        "total": len(_jobs),
        "jobs": {
            jid: {
                "status":          j.get("status"),
                "created_at":      j.get("created_at"),
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
    Pre-load model into VRAM.
    First call downloads ~12 GB — allow up to 15 minutes.
    """
    try:
        get_pipeline()
        return {
            "ok":           True,
            "model_loaded": True,
            "model":        MODEL_ID,
            "vram_used_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
            "vram_free_gb": round(
                (get_vram_gb() - torch.cuda.memory_allocated() / 1e9), 2
            ),
        }
    except Exception as e:
        log.exception("Warmup failed")
        raise HTTPException(status_code=500, detail=str(e))