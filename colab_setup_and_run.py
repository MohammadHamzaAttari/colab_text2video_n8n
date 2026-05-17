# Paste/run this in a Google Colab cell after uploading/cloning this repo.
# It installs dependencies, starts the API, and creates a public Cloudflare tunnel URL.

import os, re, subprocess, textwrap, time
from pathlib import Path

# Optional: change the model before starting.
os.environ.setdefault("MODEL_ID", "damo-vilab/text-to-video-ms-1.7b")
os.environ.setdefault("OUTPUT_DIR", "/content/generated_videos")

# Install Python dependencies.
subprocess.run("pip -q install -r requirements_colab.txt", shell=True, check=True)

# Install cloudflared for a public HTTPS URL without an account.
if not Path("/usr/local/bin/cloudflared").exists():
    subprocess.run("wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared", shell=True, check=True)
    subprocess.run("chmod +x /usr/local/bin/cloudflared", shell=True, check=True)

# Start FastAPI locally.
api = subprocess.Popen("uvicorn app:app --host 0.0.0.0 --port 8000", shell=True)
time.sleep(8)

# Start Cloudflare quick tunnel and capture public URL.
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
print("\n✅ Colab Text-to-Video API is live")
print("Base URL:", public_url)
print("Generation endpoint:", public_url + "/generate_sync")
print("Health endpoint:", public_url + "/health")
print("\nKeep this Colab cell running while n8n sends requests.")

# Keep cell alive and stream logs.
while True:
    time.sleep(60)
