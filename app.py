"""
Colab Text-to-Video API - T4 GPU Compatible Edition
Fixed for Google Colab T4 GPU limitations
"""
import gc
import logging
import os
import re
import time
import uuid
import traceback
from pathlib import Path
from typing import Optional

import torch
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

# Suppress some verbose logs
logging.getLogger("diffusers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

# ─────────────────────────── Config ───────────────────────────
MODEL_ID = os.environ.get("MODEL_ID", "Lightricks/LTX-Video")
CKPT_ID = os.environ.get("CKPT_ID", "Lightricks/LTXV-13B-0.9.7-distilled")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/content/generated_videos"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_JOBS_HISTORY = 100
app = FastAPI(title="LTX-Video T4 Compatible API", version="3.1.0")
app.mount("/files", StaticFiles(directory=str(OUTPUT_DIR)), name="files")

_pipe = None
_jobs: dict[str, dict] = {}
_gpu_info = None


# ─────────────────────────── GPU Detection ────────────────────
def detect_gpu_capabilities():
    """Detect GPU capabilities and determine optimal settings."""
    global _gpu_info
    
    if _gpu_info is not None:
        return _gpu_info
    
    log.info("🔍 Detecting GPU capabilities...")
    
    if not torch.cuda.is_available():
        raise RuntimeError("❌ No CUDA GPU detected")
    
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    
    gpu_name = props.name
    total_memory = props.total_memory / 1e9
    compute_capability = f"{props.major}.{props.minor}"
    
    log.info(f"🎯 GPU: {gpu_name}")
    log.info(f"🎯 VRAM: {total_memory:.1f} GB")
    log.info(f"🎯 Compute Capability: {compute_capability}")
    
    # Determine optimal dtype based on GPU capability
    # A100/H100 (8.0+) = bfloat16, T4/V100 (7.5) = float16, older = float32
    if props.major >= 8:
        optimal_dtype = torch.bfloat16
        dtype_name = "bfloat16"
        can_use_xformers = True
        log.info("✅ A100+ GPU detected - using bfloat16")
    elif props.major == 7 and props.minor >= 5:
        optimal_dtype = torch.float16
        dtype_name = "float16"
        can_use_xformers = False  # Often problematic on T4
        log.info("✅ T4/V100 GPU detected - using float16 (T4 compatible)")
    else:
        optimal_dtype = torch.float32
        dtype_name = "float32"
        can_use_xformers = False
        log.info("⚠️ Older GPU detected - using float32")
    
    # Memory optimization settings based on VRAM
    if total_memory >= 40:
        memory_profile = "high"
    elif total_memory >= 15:
        memory_profile = "medium"  # T4 falls here
    else:
        memory_profile = "low"
    
    log.info(f"📊 Memory profile: {memory_profile}")
    log.info(f"🧮 Using dtype: {dtype_name}")
    log.info(f"⚡ xFormers enabled: {can_use_xformers}")
    
    _gpu_info = {
        "name": gpu_name,
        "memory_gb": total_memory,
        "compute_capability": compute_capability,
        "optimal_dtype": optimal_dtype,
        "dtype_name": dtype_name,
        "can_use_xformers": can_use_xformers,
        "memory_profile": memory_profile,
        "is_t4": "T4" in gpu_name,
        "is_a100_plus": props.major >= 8
    }
    
    return _gpu_info


# ─────────────────────────── Helpers ──────────────────────────
def slugify(value: str, fallback: str = "video") -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-._")
    return value[:80] or fallback


def free_memory():
    """Aggressively free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        log.debug("🧹 Memory cleared")


def get_vram_usage():
    """Get current VRAM usage in GB."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return {
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "total_gb": round(total, 2),
            "free_gb": round(total - reserved, 2)
        }
    return {"allocated_gb": 0, "reserved_gb": 0, "total_gb": 0, "free_gb": 0}


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
    prompt: str = Field(..., min_length=3, description="Detailed text prompt")
    negative_prompt: str = Field(
        default=(
            "worst quality, inconsistent motion, blurry, jittery, distorted, "
            "watermark, text, logo, low resolution, static, overexposed, "
            "underexposed, duplicate frames, flickering"
        ),
    )
    seed: Optional[int] = Field(None, description="Reproducible seed")
    num_frames: int = Field(
        default=97, ge=9, le=257,
        description="Must be N*8+1 (e.g. 9,17,25,49,97,121,161,257)"
    )
    num_inference_steps: int = Field(
        default=6, ge=4, le=50,
        description="4-8 for distilled model"
    )
    guidance_scale: float = Field(default=1.0, ge=1.0, le=10.0)
    height: int = Field(default=480, ge=256, le=720, description="Must be divisible by 32")
    width: int = Field(default=704, ge=256, le=1280, description="Must be divisible by 32")
    fps: int = Field(default=24, ge=8, le=30)
    subfolder: str = Field(default="documentary", description="Subdirectory")
    filename_prefix: str = Field(default="clip", description="Filename prefix")
    enhance_prompt: bool = Field(default=True, description="Use prompt enhancer")


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
    """Load LTX-Video pipeline with T4-compatible settings."""
    global _pipe
    
    if _pipe is not None:
        log.info("♻️ Using cached pipeline")
        return _pipe

    log.info("🚀 Loading LTX-Video pipeline...")
    gpu_info = detect_gpu_capabilities()
    
    try:
        from diffusers import LTXPipeline, LTXVideoTransformer3DModel
        from transformers import T5EncoderModel
        
        log.info(f"📦 Loading model: {MODEL_ID}")
        log.info(f"📦 Checkpoint: {CKPT_ID}")
        log.info(f"🧮 Using dtype: {gpu_info['dtype_name']}")
        
        # Load components with GPU-appropriate settings
        dtype = gpu_info["optimal_dtype"]
        
        # 1. Load transformer from distilled checkpoint
        is_distilled = "distilled" in CKPT_ID.lower()
        transformer = None
        
        if is_distilled:
            log.info("📥 Loading distilled transformer...")
            try:
                transformer = LTXVideoTransformer3DModel.from_pretrained(
                    CKPT_ID,
                    subfolder="transformer",
                    torch_dtype=dtype,
                )
                log.info("✅ Distilled transformer loaded")
            except Exception as e:
                log.error(f"❌ Failed to load distilled transformer: {e}")
                log.info("📥 Falling back to base model transformer")
        
        # 2. Load text encoder with memory optimization
        log.info("📥 Loading T5 text encoder...")
        text_encoder = None
        
        # Try 8-bit loading first for memory savings
        if gpu_info["memory_profile"] in ["medium", "low"]:
            try:
                log.info("🔹 Attempting 8-bit quantized T5 loading...")
                text_encoder = T5EncoderModel.from_pretrained(
                    MODEL_ID,
                    subfolder="text_encoder",
                    torch_dtype=dtype,
                    load_in_8bit=True,
                    device_map="auto",
                )
                log.info("✅ T5 loaded in 8-bit (saves ~4GB VRAM)")
            except Exception as e:
                log.warning(f"⚠️ 8-bit loading failed: {e}")
                log.info("📥 Falling back to standard T5 loading...")
        
        # Standard T5 loading fallback
        if text_encoder is None:
            text_encoder = T5EncoderModel.from_pretrained(
                MODEL_ID,
                subfolder="text_encoder",
                torch_dtype=dtype,
            )
            log.info("✅ T5 loaded in standard mode")
        
        # 3. Assemble pipeline
        log.info("🔧 Assembling pipeline...")
        pipe_kwargs = {
            "torch_dtype": dtype,
            "text_encoder": text_encoder,
        }
        
        if transformer is not None:
            pipe_kwargs["transformer"] = transformer
        
        _pipe = LTXPipeline.from_pretrained(MODEL_ID, **pipe_kwargs)
        log.info("✅ Pipeline assembled")
        
        # 4. Apply memory optimizations in correct order
        log.info("⚙️ Applying memory optimizations...")
        
        # CPU offload - essential for T4
        _pipe.enable_model_cpu_offload()
        log.info("✅ CPU offload enabled")
        
        # VAE optimizations
        _pipe.vae.enable_tiling()
        _pipe.vae.enable_slicing()
        log.info("✅ VAE tiling and slicing enabled")
        
        # Attention optimizations
        try:
            _pipe.enable_attention_slicing(1)
            log.info("✅ Attention slicing enabled")
        except Exception as e:
            log.warning(f"⚠️ Attention slicing failed: {e}")
        
        # xFormers - only if supported
        if gpu_info["can_use_xformers"]:
            try:
                _pipe.enable_xformers_memory_efficient_attention()
                log.info("✅ xFormers memory efficient attention enabled")
            except Exception as e:
                log.warning(f"⚠️ xFormers failed: {e}")
                log.info("📝 Using PyTorch SDPA fallback")
        else:
            log.info("📝 Skipping xFormers (not compatible with this GPU)")
        
        # Log final memory usage
        vram = get_vram_usage()
        log.info(f"📊 Pipeline loaded - VRAM: {vram['allocated_gb']}GB allocated")
        log.info("🎉 LTX-Video pipeline ready!")
        
        return _pipe
        
    except Exception as e:
        log.error(f"❌ Pipeline loading failed: {e}")
        log.error(f"📋 Traceback: {traceback.format_exc()}")
        free_memory()
        raise


def _round_to_valid_frames(n: int) -> int:
    """Round to valid LTX-Video frame count (N*8 + 1)."""
    valid = [9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97,
             105, 113, 121, 129, 137, 145, 153, 161, 169, 177,
             185, 193, 201, 209, 217, 225, 233, 241, 249, 257]
    return min(valid, key=lambda x: abs(x - n))


def _round_to_32(n: int) -> int:
    """Round to multiple of 32 for LTX-Video compatibility."""
    return max(32, (n // 32) * 32)


# ─────────────────────────── Generation ───────────────────────
def generate_video(req: GenerateRequest, base_url: str, job_id: Optional[str] = None) -> dict:
    job_id = job_id or str(uuid.uuid4())
    start_time = time.time()
    
    log.info(f"🎬 Starting generation for job {job_id}")
    
    _jobs[job_id] = {
        "status": "running",
        "created_at": start_time,
        "prompt": req.prompt,
    }

    try:
        # Get pipeline and GPU info
        pipeline = get_pipeline()
        gpu_info = detect_gpu_capabilities()
        
        log.info(f"🎯 GPU: {gpu_info['name']} ({gpu_info['dtype_name']})")
        
        # Validate and adjust parameters
        seed = req.seed if req.seed is not None else int(time.time()) % 2_147_483_647
        num_frames = _round_to_valid_frames(req.num_frames)
        height = _round_to_32(req.height)
        width = _round_to_32(req.width)
        
        # Log parameter adjustments
        if num_frames != req.num_frames:
            log.info(f"📐 Frames adjusted: {req.num_frames} → {num_frames}")
        if height != req.height or width != req.width:
            log.info(f"📐 Resolution adjusted: {req.width}x{req.height} → {width}x{height}")
        
        # T4-specific parameter validation
        if gpu_info["is_t4"]:
            max_pixels = 704 * 480  # Safe for T4
            current_pixels = width * height
            if current_pixels > max_pixels:
                log.warning(f"⚠️ Resolution too high for T4: {current_pixels} > {max_pixels}")
                # Reduce to safe T4 resolution
                width = 512
                height = 288
                log.info(f"📐 Reduced to T4-safe resolution: {width}x{height}")
        
        # Setup output path
        folder = OUTPUT_DIR / slugify(req.subfolder, "documentary")
        folder.mkdir(parents=True, exist_ok=True)
        file_name = f"{slugify(req.filename_prefix, 'clip')}-{job_id[:8]}-seed{seed}.mp4"
        out_path = folder / file_name
        
        # Enhance prompt if requested
        prompt = req.prompt
        if req.enhance_prompt:
            cinematic_tags = (
                ", cinematic, photorealistic, 8K, high detail, "
                "natural lighting, documentary style, realistic motion, "
                "sharp focus, professional cinematography"
            )
            if not any(t in prompt.lower() for t in ["cinematic", "photorealistic", "8k"]):
                prompt = prompt + cinematic_tags
                log.info("✨ Prompt enhanced with cinematic tags")
        
        log.info(f"🎬 Generation settings:")
        log.info(f"   📏 Frames: {num_frames}")
        log.info(f"   🔢 Steps: {req.num_inference_steps}")
        log.info(f"   🎯 Guidance: {req.guidance_scale}")
        log.info(f"   📐 Resolution: {width}x{height}")
        log.info(f"   🎲 Seed: {seed}")
        log.info(f"   💾 Output: {out_path}")
        
        # Check memory before generation
        vram_before = get_vram_usage()
        log.info(f"📊 VRAM before generation: {vram_before['allocated_gb']}GB")
        
        # Free memory before generation
        free_memory()
        
        # Setup generator
        generator = torch.Generator(device="cuda").manual_seed(seed)
        
        # Run inference with comprehensive error handling
        log.info("🚀 Starting inference...")
        try:
            with torch.inference_mode():
                result = pipeline(
                    prompt=prompt,
                    negative_prompt=req.negative_prompt,
                    num_frames=num_frames,
                    num_inference_steps=req.num_inference_steps,
                    guidance_scale=req.guidance_scale,
                    height=height,
                    width=width,
                    generator=generator,
                )
            log.info("✅ Inference completed successfully")
            
        except torch.cuda.OutOfMemoryError as oom_err:
            free_memory()
            error_msg = (
                f"CUDA out of memory during generation. "
                f"Try reducing: width={width//2}, height={height//2}, num_frames={num_frames//2}. "
                f"Current settings may be too demanding for {gpu_info['name']}."
            )
            log.error(f"❌ OOM Error: {error_msg}")
            _jobs[job_id].update({"status": "failed", "error": error_msg})
            return _jobs[job_id]
            
        except Exception as inference_err:
            free_memory()
            error_msg = f"Inference failed: {str(inference_err)}"
            log.error(f"❌ Inference Error: {error_msg}")
            log.error(f"📋 Traceback: {traceback.format_exc()}")
            _jobs[job_id].update({"status": "failed", "error": error_msg})
            return _jobs[job_id]
        
        # Get VRAM peak usage
        vram_after = get_vram_usage()
        log.info(f"📊 VRAM after generation: {vram_after['allocated_gb']}GB")
        
        # Export video
        log.info("🎞️ Exporting video...")
        frames = result.frames[0]
        _export_video_hq(frames, str(out_path), fps=req.fps)
        
        # Final cleanup
        free_memory()
        generation_time = time.time() - start_time
        
        # Build response
        rel_path = out_path.relative_to(OUTPUT_DIR).as_posix()
        video_url = f"{base_url.rstrip('/')}/files/{rel_path}"
        
        log.info(f"✅ Generation completed in {generation_time:.1f}s")
        log.info(f"📁 Video saved: {file_name}")
        log.info(f"🔗 Video URL: {video_url}")
        
        result_data = {
            "status": "completed",
            "created_at": start_time,
            "prompt": req.prompt,
            "seed": seed,
            "file_name": file_name,
            "file_path": str(out_path),
            "video_url": video_url,
            "generation_time": round(generation_time, 2),
            "vram_used_gb": round(vram_after["allocated_gb"], 2),
        }
        _jobs[job_id] = result_data
        
        # Cleanup old jobs
        if len(_jobs) > MAX_JOBS_HISTORY:
            oldest = sorted(_jobs, key=lambda k: _jobs[k].get("created_at", 0))
            for old_key in oldest[: len(_jobs) - MAX_JOBS_HISTORY]:
                _jobs.pop(old_key, None)
        
        return result_data
        
    except Exception as e:
        free_memory()
        error_msg = f"Generation failed: {str(e)}"
        log.error(f"❌ {error_msg}")
        log.error(f"📋 Full traceback: {traceback.format_exc()}")
        
        _jobs[job_id] = {
            "status": "failed",
            "created_at": _jobs.get(job_id, {}).get("created_at", time.time()),
            "prompt": req.prompt,
            "error": error_msg,
        }
        return _jobs[job_id]


def _export_video_hq(frames, path: str, fps: int = 24):
    """Export frames to H.264 MP4 with high quality."""
    try:
        import imageio
        import numpy as np
        
        log.info(f"🎞️ Exporting {len(frames)} frames to {path}")
        
        # Convert frames to numpy arrays
        np_frames = []
        for f in frames:
            if hasattr(f, "numpy"):
                arr = f.numpy()
            else:
                arr = np.array(f)
            
            if arr.dtype != np.uint8:
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
            np_frames.append(arr)
        
        # High quality export with fallback
        try:
            writer = imageio.get_writer(
                path, fps=fps, codec="libx264", quality=None,
                output_params=[
                    "-crf", "18", "-preset", "slow", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart"
                ],
            )
            for frame in np_frames:
                writer.append_data(frame)
            writer.close()
            log.info("✅ High quality H.264 export completed")
            
        except Exception as e:
            log.warning(f"⚠️ HQ export failed ({e}), using standard export...")
            writer = imageio.get_writer(path, fps=fps, quality=8)
            for frame in np_frames:
                writer.append_data(frame)
            writer.close()
            log.info("✅ Standard export completed")
            
    except Exception as e:
        log.error(f"❌ Video export failed: {e}")
        raise


# ─────────────────────────── Routes ───────────────────────────
@app.get("/")
def root():
    gpu_info = detect_gpu_capabilities() if torch.cuda.is_available() else None
    vram = get_vram_usage()
    
    return {
        "ok": True,
        "model": MODEL_ID,
        "checkpoint": CKPT_ID,
        "gpu_info": gpu_info,
        "vram_usage": vram,
        "model_loaded": _pipe is not None,
        "recommended_settings": {
            "t4_safe": {
                "num_frames": 49,
                "num_inference_steps": 6,
                "guidance_scale": 1.0,
                "width": 512,
                "height": 288,
                "fps": 24,
                "note": "Safe for T4 GPU"
            },
            "t4_optimal": {
                "num_frames": 97,
                "num_inference_steps": 6,
                "guidance_scale": 1.0,
                "width": 704,
                "height": 480,
                "fps": 24,
                "note": "Optimal for T4 GPU"
            }
        }
    }


@app.get("/health")
def health():
    vram = get_vram_usage()
    gpu_info = detect_gpu_capabilities() if torch.cuda.is_available() else None
    
    return {
        "ok": True,
        "cuda": torch.cuda.is_available(),
        "vram_total_gb": vram["total_gb"],
        "vram_used_gb": vram["allocated_gb"],
        "vram_free_gb": vram["free_gb"],
        "model_loaded": _pipe is not None,
        "model": MODEL_ID,
        "checkpoint": CKPT_ID,
        "jobs_tracked": len(_jobs),
        "gpu_info": gpu_info,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate_async(req: GenerateRequest, request: Request, background_tasks: BackgroundTasks):
    """Async generation - returns immediately with job_id."""
    base_url = get_base_url(request)
    job_id = str(uuid.uuid4())
    
    log.info(f"📨 Async generation request: {job_id}")
    log.info(f"📝 Prompt: {req.prompt[:100]}...")
    
    _jobs[job_id] = {
        "status": "queued",
        "created_at": time.time(),
        "prompt": req.prompt,
    }
    
    background_tasks.add_task(generate_video, req, base_url, job_id)
    return GenerateResponse(job_id=job_id, status="queued")


@app.post("/generate_sync", response_model=GenerateResponse)
def generate_sync(req: GenerateRequest, request: Request):
    """Synchronous generation - blocks until complete."""
    base_url = get_base_url(request)
    job_id = str(uuid.uuid4())
    
    log.info(f"📨 Sync generation request: {job_id}")
    log.info(f"📝 Prompt: {req.prompt[:100]}...")
    
    data = generate_video(req, base_url=base_url, job_id=job_id)
    return GenerateResponse(job_id=job_id, **{
        k: v for k, v in data.items()
        if k in GenerateResponse.model_fields
    })


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = _jobs[job_id].copy()
    
    # Add real-time VRAM info if job is running
    if job_data.get("status") == "running":
        vram = get_vram_usage()
        job_data["current_vram_gb"] = vram["allocated_gb"]
    
    return job_data


@app.get("/jobs")
def list_jobs():
    """List all tracked jobs."""
    vram = get_vram_usage()
    return {
        "total": len(_jobs),
        "vram_usage": vram,
        "jobs": {
            jid: {
                "status": j.get("status"),
                "created_at": j.get("created_at"),
                "generation_time": j.get("generation_time"),
                "error": j.get("error", "")[:100] if j.get("error") else None,
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
def warmup():
    """Pre-load the model."""
    try:
        log.info("🔥 Warmup requested")
        pipeline = get_pipeline()
        vram = get_vram_usage()
        gpu_info = detect_gpu_capabilities()
        
        return {
            "ok": True,
            "model_loaded": True,
            "vram_usage": vram,
            "gpu_info": gpu_info,
        }
    except Exception as e:
        log.error(f"❌ Warmup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug")
def debug_info():
    """Debug endpoint for troubleshooting."""
    gpu_info = detect_gpu_capabilities() if torch.cuda.is_available() else None
    vram = get_vram_usage()
    
    return {
        "gpu_info": gpu_info,
        "vram_usage": vram,
        "model_loaded": _pipe is not None,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "jobs_count": len(_jobs),
        "recent_jobs": {
            jid: j for jid, j in list(_jobs.items())[-5:]
        }
    }