"""
Colab Setup for LTX-Video Documentary API
Model: Lightricks/LTX-Video
Optimised for Google Colab T4 GPU (15 GB VRAM)
Uses ngrok for stable tunnel (no Cloudflare)
"""
import os
import subprocess
import sys
import time
from pathlib import Path

# ─────────────────────────── Environment ──────────────────────
os.environ.setdefault("MODEL_ID",   "Lightricks/LTX-Video")
os.environ.setdefault("OUTPUT_DIR", "/content/generated_videos")

# ── NGROK AUTH TOKEN ──────────────────────────────────────────
# Your token — replace if it expires
NGROK_AUTHTOKEN = "2xx0FzRuI3CgwRVHYFDgQEcWqcW_5hw6qC4inSEhVMgdiAwuj"
os.environ["NGROK_AUTHTOKEN"] = NGROK_AUTHTOKEN

COLAB_PORT       = 8000
WARMUP_TIMEOUT_S = 900   # 15 min — first run downloads ~12 GB


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

# Install pyngrok explicitly to make sure it is fresh
print("📦 Installing pyngrok...")
run("pip install -q --upgrade pyngrok")

print("🔍 Verifying key packages...")
verify_imports = [
    "import torch; print(f'torch {torch.__version__}, CUDA={torch.cuda.is_available()}')",
    "import diffusers; print(f'diffusers {diffusers.__version__}')",
    "import transformers; print(f'transformers {transformers.__version__}')",
    "import imageio_ffmpeg; print('imageio-ffmpeg OK')",
    "import bitsandbytes; print(f'bitsandbytes {bitsandbytes.__version__}')",
    "import pyngrok; print(f'pyngrok {pyngrok.__version__}')",
]
for imp in verify_imports:
    r = run(f'python -c "{imp}"', capture=True, check=False)
    print(f"   {r.stdout.strip() or ('⚠️  ' + r.stderr.strip()[:80])}")

# Verify HuggingFace repo
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

# Kill any leftover server
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

print("⏳ Waiting for FastAPI server to start...")
server_ready = False
for i in range(40):
    time.sleep(1)
    check = run(
        f"curl -s --max-time 3 http://127.0.0.1:{COLAB_PORT}/health",
        capture=True, check=False,
    )
    if check.returncode == 0 and "ok" in check.stdout:
        print(f"✅ Server ready after {i+1}s")
        server_ready = True
        break

if not server_ready:
    print("⚠️  Server did not respond in time — proceeding anyway.")
    print("   Check logs above for errors.")


# ─────────────────────────── ngrok Tunnel ─────────────────────
section("5/6  Creating ngrok Tunnel")

public_url = None

def start_ngrok(port: int, token: str) -> str:
    """
    Start ngrok tunnel with auth token.
    Returns public HTTPS URL or raises RuntimeError.
    """
    from pyngrok import ngrok, conf, exception as ngrok_exception

    # Kill any existing ngrok processes cleanly
    try:
        ngrok.kill()
    except Exception:
        pass
    time.sleep(1)

    # Configure auth token
    pyngrok_config = conf.PyngrokConfig(auth_token=token)
    conf.set_default(pyngrok_config)

    print(f"   Auth token: {token[:12]}...{token[-6:]}")
    print(f"   Starting tunnel on port {port}...")

    # Open tunnel
    tunnel     = ngrok.connect(port, "http", bind_tls=True)
    tunnel_url = str(tunnel.public_url)

    # Ensure HTTPS
    if tunnel_url.startswith("http://"):
        tunnel_url = tunnel_url.replace("http://", "https://", 1)

    return tunnel_url


# Attempt ngrok with retries
MAX_RETRIES = 3
for attempt in range(1, MAX_RETRIES + 1):
    try:
        print(f"🔄 ngrok attempt {attempt}/{MAX_RETRIES}...")
        public_url = start_ngrok(COLAB_PORT, NGROK_AUTHTOKEN)
        print(f"✅ ngrok tunnel active: {public_url}")
        break
    except Exception as e:
        print(f"   Attempt {attempt} failed: {e}")
        if attempt < MAX_RETRIES:
            print(f"   Retrying in 5s...")
            time.sleep(5)
        else:
            print("❌ All ngrok attempts failed.")
            print("   Common causes:")
            print("   1. Invalid auth token — check https://dashboard.ngrok.com/get-started/your-authtoken")
            print("   2. Token already in use on another session")
            print("   3. Network issue in Colab")
            print("\n   Falling back to Cloudflare tunnel...")

            # Emergency fallback to cloudflared
            import re as _re
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
            for _ in range(90):
                line = cf_proc.stdout.readline()
                m    = _re.search(r"https://[-\w]+\.trycloudflare\.com", line)
                if m:
                    public_url = m.group(0)
                    print(f"✅ Cloudflare fallback tunnel: {public_url}")
                    break
                time.sleep(1)

            if not public_url:
                cf_proc.terminate()
                raise RuntimeError(
                    "❌ Both ngrok and Cloudflare tunnels failed.\n"
                    "   Try: Runtime > Restart runtime, then run again."
                )

if not public_url:
    raise RuntimeError("No tunnel URL obtained — cannot continue.")

os.environ["PUBLIC_BASE_URL"] = public_url

# Verify tunnel is reachable from outside
print("\n🔍 Verifying tunnel is reachable...")
for attempt in range(5):
    time.sleep(3)
    ping = run(
        f'curl -s --max-time 10 "{public_url}/health"',
        capture=True, check=False,
    )
    if ping.returncode == 0 and "ok" in ping.stdout:
        print(f"✅ Tunnel verified — external access confirmed")
        break
    print(f"   Attempt {attempt+1}/5: waiting for tunnel to stabilise...")
else:
    print("⚠️  Tunnel may need a few more seconds to stabilise.")
    print(f"   Manual check: curl {public_url}/health")


# ─────────────────────────── Warmup ───────────────────────────
section("6/6  Model Warmup — downloading & loading into VRAM")

print(f"⏳ Downloading & loading LTX-Video weights (~12 GB on first run)")
print(f"   Timeout set to {WARMUP_TIMEOUT_S // 60} minutes")
print(f"   Subsequent runs use HuggingFace cache (~30s)\n")

warmup_result = run(
    f'curl -s --max-time {WARMUP_TIMEOUT_S} -X POST "{public_url}/warmup"',
    capture=True, check=False,
)

stdout = warmup_result.stdout.strip()

if '"model_loaded": true' in stdout or '"model_loaded":true' in stdout:
    print(f"✅ Model loaded and ready!")
    # Parse VRAM info if available
    import json as _json
    try:
        info = _json.loads(stdout)
        print(f"   VRAM used : {info.get('vram_used_gb', '?')} GB")
        print(f"   VRAM free : {info.get('vram_free_gb', '?')} GB")
    except Exception:
        print(f"   Response  : {stdout[:200]}")

elif warmup_result.returncode == 28:
    print("⚠️  Warmup curl timed out after 15 min.")
    print("   The model may still be downloading in the background.")
    print(f"   Check status: curl -s {public_url}/health | python3 -m json.tool")

else:
    print(f"⚠️  Unexpected warmup response (exit={warmup_result.returncode}):")
    print(f"   {stdout[:400] or '(empty response)'}")
    print(f"   Check: curl -s {public_url}/health")


# ─────────────────────────── Summary ──────────────────────────
print("\n" + "█" * 65)
print("  ✅  LTX-VIDEO DOCUMENTARY API IS LIVE")
print("█" * 65)
print(f"""
  📡  Base URL       : {public_url}
  🎬  Generate sync  : {public_url}/generate_sync
  🔄  Generate async : {public_url}/generate
  💚  Health check   : {public_url}/health
  📋  Job list       : {public_url}/jobs
  🔥  Warmup         : {public_url}/warmup

  🎥  Model  : {os.environ['MODEL_ID']}
  🖥️   VRAM   : {vram_gb:.1f} GB
  🔗  Tunnel : ngrok (authenticated)

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

  ⚠️  Copy the Base URL above into n8n env var:
      COLAB_BASE_URL = {public_url}

  ⚠️  Keep this Colab tab ACTIVE during generation
  ⚠️  Free Colab disconnects after ~90 min idle
""")
print("█" * 65 + "\n")

# ─────────────────────────── Keep-Alive ───────────────────────
print("🔄 Server running... (Ctrl+C to stop)\n")

heartbeat        = 0
consecutive_fail = 0

while True:
    time.sleep(30)
    heartbeat += 1

    # Ping local server to prevent Colab idle timeout
    local_check = run(
        f"curl -s --max-time 5 http://127.0.0.1:{COLAB_PORT}/health",
        capture=True, check=False,
    )
    local_ok = "ok" in local_check.stdout

    if not local_ok:
        consecutive_fail += 1
        print(f"  ⚠️  Local server not responding (fail #{consecutive_fail})")
        if consecutive_fail >= 4:
            print("  ❌ Server appears to have crashed. Restart the cell.")
    else:
        consecutive_fail = 0

    # Detailed heartbeat every 2 minutes (every 4 x 30s ticks)
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
            api_symbol = "✅" if local_ok else "❌"
            print(
                f"  💓 Heartbeat #{heartbeat} | "
                f"VRAM: {used}GB used / {free}GB free | "
                f"Local API: {api_symbol} | "
                f"Tunnel: {public_url}"
            )