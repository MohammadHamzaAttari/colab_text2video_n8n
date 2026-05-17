# Paste/run this in a Google Colab cell
# PRODUCTION SETUP FOR YOUTUBE QUALITY

import os, re, subprocess, textwrap, time
from pathlib import Path

# IMPORTANT: Use ZeroScope v2 for 16:9 ratio and better quality
os.environ.setdefault("MODEL_ID", "cerspense/zeroscope_v2_576w")
os.environ.setdefault("OUTPUT_DIR", "/content/generated_videos")

print("📦 Installing dependencies...")
subprocess.run("pip -q install -r requirements_colab.txt", shell=True, check=True)

# Install cloudflared
if not Path("/usr/local/bin/cloudflared").exists():
    print("🔧 Installing Cloudflare tunnel...")
    subprocess.run("wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared", shell=True, check=True)
    subprocess.run("chmod +x /usr/local/bin/cloudflared", shell=True, check=True)

# Start FastAPI
print("🚀 Starting FastAPI server...")
api = subprocess.Popen("uvicorn app:app --host 0.0.0.0 --port 8000", shell=True)
time.sleep(8)

# Start Cloudflare tunnel
print("🌐 Creating public tunnel...")
tunnel = subprocess.Popen(
    "cloudflared tunnel --url http://127.0.0.1:8000 --no-autoupdate",
    shell=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)

public_url = None
for _ in range(80):
    line = tunnel.stdout.readline()
    print(line, end="")
    m = re.search(r"https://[-a-zA-Z0-9.]+trycloudflare.com", line)
    if m:
        public_url = m.group(0)
        break
    time.sleep(1)

if not public_url:
    raise RuntimeError("Cloudflare tunnel URL not found. Re-run this cell.")

os.environ["PUBLIC_BASE_URL"] = public_url

print("\n" + "="*70)
print("✅ PRODUCTION TEXT-TO-VIDEO API IS LIVE")
print("="*70)
print(f"📡 Base URL: {public_url}")
print(f"🎬 Generate endpoint: {public_url}/generate")
print(f"💚 Health check: {public_url}/health")
print(f"🎥 Model: {os.environ['MODEL_ID']}")
print(f"📐 Max resolution: 576×320 (16:9 aspect ratio)")
print(f"⚡ Recommended: 24 frames, 40 steps, guidance 17.5")
print("="*70)
print("\n⚠️  Keep this Colab tab open while n8n generates videos\n")

# Keep alive
while True:
    time.sleep(60)