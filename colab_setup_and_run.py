"""
Colab Setup for LTX-Video Documentary API
Model: Lightricks/LTX-Video
Optimised for Google Colab T4 GPU (15 GB VRAM)
"""
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# ─────────────────────────── Environment ──────────────────────
# FIXED: Only set MODEL_ID — no fake CKPT_ID repo
os.environ.setdefault("MODEL_ID",    "Lightricks/LTX-Video")
os.environ.setdefault("OUTPUT_DIR",  "/content/generated_videos")

COLAB_PORT  = 8000
# Warmup timeout: 15 min — first run downloads ~12 GB weights
WARMUP_TIMEOUT_S = 900


def run(cmd: str, check: bool = True, capture: bool = False):
    kwargs = {"shell": True, "check": check}
    if capture:
        kwargs["capture_output"] = True
        kwargs["text"]           = True
    return subprocess.run(cmd, **kwargs)


def section(title: str):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


# ─────────────────────────── GPU Check ────────────────────────
section("1/6  GPU Verification")

gpu_result = run(
    "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
    capture=True, check=False,
)
if gpu_result.returncode != 0 or not gpu_result.stdout.strip():
    print("❌ No GPU detected!")
    print("   Go to: Runtime > Change runtime type > T4 GPU")
    sys.exit(1)

print(f"✅ GPU: {gpu_result.stdout.strip()}")

vram_result = run(
    "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits",
    capture=True, check=False,
)
try:
    vram_mb = int(vram_result.stdout.strip())
    vram_gb = vram_mb / 1024
    print(f"✅ VRAM: {vram_gb:.1f} GB")
    if vram_gb < 12:
        print("⚠️  < 12 GB VRAM — applying memory-safe settings")
        os.environ["MEMORY_SAFE"] = "1"
except Exception:
    vram_gb = 15.0


# ─────────────────────────── Install ──────────────────────────
section("2/6  Installing Dependencies")

run("pip install -q --upgrade pip")

print("📦 Checking PyTorch + CUDA...")
result = run(
    'python -c "import torch; print(torch.__version__, torch.cuda.is_available())"',
    capture=True, check=False,
)
print(f"   Current: {result.stdout.strip()}")

print("📦 Installing requirements...")
run("pip install -q -r requirements_colab.txt")

print("🔍 Verifying key packages...")
verify_imports = [
    "import torch; print(f'torch {torch.__version__}, CUDA={torch.cuda.is_available()}')",
    "import diffusers; print(f'diffusers {diffusers.__version__}')",
    "import transformers; print(f'transformers {transformers.__version__}')",
    "import imageio_ffmpeg; print('imageio-ffmpeg OK')",
    "import bitsandbytes; print(f'bitsandbytes {bitsandbytes.__version__}')",
]
for imp in verify_imports:
    r = run(f'python -c "{imp}"', capture=True, check=False)
    print(f"   {r.stdout.strip() or ('⚠️  ' + r.stderr.strip()[:80])}")

# Verify the actual HuggingFace repo exists and is accessible
print("🔍 Verifying HuggingFace model repo...")
hf_check = run(
    'python -c "'
    'from huggingface_hub import repo_exists; '
    'ok = repo_exists(\"Lightricks/LTX-Video\"); '
    'print(\"✅ Repo accessible\" if ok else \"❌ Repo NOT found\")'
    '"',
    capture=True, check=False,
)
print(f"   {hf_check.stdout.strip()}")


# ─────────────────────────── Output Dir ───────────────────────
section("3/6  Preparing Output Directory")
out_dir = Path(os.environ["OUTPUT_DIR"])
out_dir.mkdir(parents=True, exist_ok=True)
print(f"✅ Output directory: {out_dir}")


# ─────────────────────────── Start API ────────────────────────
section("4/6  Starting FastAPI Server")

run(f"fuser -k {COLAB_PORT}/tcp 2>/dev/null || true", check=False)
time.sleep(2)

server_proc = subprocess.Popen(
    (
        f"uvicorn app:app --host 0.0.0.0 --port {COLAB_PORT} "
        f"--log-level info --timeout-keep-alive 600"
    ),
    shell=True,
    env={**os.environ},
)

print("⏳ Waiting for server to start...")
for i in range(30):
    time.sleep(1)
    check = run(
        f"curl -s http://127.0.0.1:{COLAB_PORT}/health",
        capture=True, check=False,
    )
    if check.returncode == 0 and "ok" in check.stdout:
        print(f"✅ Server ready after {i+1}s")
        break
else:
    print("⚠️  Server still starting, proceeding anyway...")


# ─────────────────────────── Tunnel ───────────────────────────
section("5/6  Creating Public Tunnel")

public_url = None

# ── ngrok ─────────────────────────────────────────────────────
try:
    print("🔄 Trying ngrok tunnel...")
    from pyngrok import ngrok, conf

    ngrok_token = os.environ.get("NGROK_AUTHTOKEN", "")
    if ngrok_token:
        conf.get_default().auth_token = ngrok_token
        print("   Using ngrok auth token")
    else:
        print("   No NGROK_AUTHTOKEN set")

    tunnel     = ngrok.connect(COLAB_PORT, bind_tls=True)
    public_url = str(tunnel.public_url)
    print(f"✅ ngrok tunnel: {public_url}")

except Exception as ngrok_err:
    print(f"⚠️  ngrok failed: {ngrok_err}")

    # ── cloudflared ───────────────────────────────────────────
    print("🔄 Trying Cloudflare tunnel...")

    cf_bin = Path("/usr/local/bin/cloudflared")
    if not cf_bin.exists():
        print("   Downloading cloudflared...")
        run(
            "wget -q https://github.com/cloudflare/cloudflared/releases/"
            "latest/download/cloudflared-linux-amd64 "
            f"-O {cf_bin}"
        )
        run(f"chmod +x {cf_bin}")

    cf_proc = subprocess.Popen(
        f"{cf_bin} tunnel --url http://127.0.0.1:{COLAB_PORT} --no-autoupdate",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    for _ in range(60):
        line = cf_proc.stdout.readline()
        m    = re.search(r"https://[-\w]+\.trycloudflare\.com", line)
        if m:
            public_url = m.group(0)
            print(f"✅ Cloudflare tunnel: {public_url}")
            break
        time.sleep(1)

    if not public_url:
        cf_proc.terminate()
        raise RuntimeError(
            "Both ngrok and Cloudflare tunnels failed.\n"
            "Try: Runtime > Restart runtime, then run again."
        )

os.environ["PUBLIC_BASE_URL"] = public_url


# ─────────────────────────── Warmup ───────────────────────────
section("6/6  Model Warmup — downloading & loading into VRAM")

print("⏳ Downloading & loading LTX-Video weights (~12 GB on first run)...")
print(f"   Timeout: {WARMUP_TIMEOUT_S // 60} minutes")
print("   Subsequent runs use cached weights (~30s)\n")

# FIXED: Use long timeout so the 12 GB download + load completes
warmup_result = run(
    f'curl -s --max-time {WARMUP_TIMEOUT_S} -X POST "{public_url}/warmup"',
    capture=True, check=False,
)

stdout = warmup_result.stdout.strip()
if "model_loaded" in stdout and '"ok":true' in stdout.replace(" ", ""):
    print(f"✅ Model loaded and ready!")
    print(f"   Response: {stdout}")
elif warmup_result.returncode == 28:
    # curl exit code 28 = timeout
    print("⚠️  Warmup timed out (model still loading in background).")
    print("   Wait 2-3 more minutes before sending generation requests.")
    print("   Check: curl -s {public_url}/health | python3 -m json.tool")
else:
    print(f"⚠️  Warmup response: {stdout[:300] or '(empty)'}")
    print("   If model_loaded=false, wait 1-2 minutes and check /health")


# ─────────────────────────── Summary ──────────────────────────
print("\n" + "█" * 65)
print("  ✅  LTX-VIDEO DOCUMENTARY API IS LIVE")
print("█" * 65)
print(f"\n  📡  Base URL      : {public_url}")
print(f"  🎬  Generate sync : {public_url}/generate_sync")
print(f"  🔄  Generate async: {public_url}/generate")
print(f"  💚  Health check  : {public_url}/health")
print(f"  📋  Job list      : {public_url}/jobs")
print(f"\n  🎥  Model  : {os.environ['MODEL_ID']}")
print(f"  🖥️   VRAM   : {vram_gb:.1f} GB")
print("""
  📐  RECOMMENDED SETTINGS FOR n8n:
  ┌──────────────────────────────────────────┐
  │  num_frames         : 97 (4s @ 24fps)   │
  │  num_inference_steps: 6  (distilled)    │
  │  guidance_scale     : 1.0               │
  │  width              : 704               │
  │  height             : 480               │
  │  fps                : 24                │
  │  enhance_prompt     : true              │
  │  n8n HTTP timeout   : 600000ms (10min)  │
  └──────────────────────────────────────────┘

  💾  IF CUDA OUT OF MEMORY — use these:
  ┌──────────────────────────────────────────┐
  │  width              : 512               │
  │  height             : 288               │
  │  num_frames         : 49 (2s @ 24fps)   │
  └──────────────────────────────────────────┘

  ⚠️  Keep this Colab tab ACTIVE during generation
  ⚠️  Free Colab disconnects after ~90 min idle
""")
print("█" * 65 + "\n")


# ─────────────────────────── Keep-Alive ───────────────────────
print("🔄 Server running... (Ctrl+C to stop)\n")
heartbeat = 0
while True:
    time.sleep(30)
    heartbeat += 1
    check = run(
        f"curl -s http://127.0.0.1:{COLAB_PORT}/health",
        capture=True, check=False,
    )
    if heartbeat % 4 == 0:
        vram_check = run(
            "nvidia-smi --query-gpu=memory.used,memory.free "
            "--format=csv,noheader,nounits",
            capture=True, check=False,
        )
        if vram_check.returncode == 0:
            parts = vram_check.stdout.strip().split(",")
            used  = int(parts[0].strip()) // 1024
            free  = int(parts[1].strip()) // 1024
            api_ok = "✅" if "ok" in check.stdout else "❌"
            print(
                f"  💓 Heartbeat #{heartbeat} | "
                f"VRAM: {used}GB used / {free}GB free | "
                f"API: {api_ok}"
            )