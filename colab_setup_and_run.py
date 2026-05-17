# Enhanced setup with multiple tunnel options
import os, re, subprocess, textwrap, time
from pathlib import Path

# IMPORTANT: Use ZeroScope v2 for 16:9 ratio and better quality
os.environ.setdefault("MODEL_ID", "cerspense/zeroscope_v2_576w")
os.environ.setdefault("OUTPUT_DIR", "/content/generated_videos")

print("📦 Installing dependencies...")
subprocess.run("pip -q install -r requirements_colab.txt", shell=True, check=True)

# Install pyngrok for more reliable tunneling
print("🔧 Installing ngrok...")
subprocess.run("pip -q install pyngrok", shell=True, check=True)

# Start FastAPI
print("🚀 Starting FastAPI server...")
api = subprocess.Popen("uvicorn app:app --host 0.0.0.0 --port 8000", shell=True)
time.sleep(8)

# Try ngrok first (more reliable)
try:
    print("🌐 Creating ngrok tunnel...")
    from pyngrok import ngrok
    
    # Create tunnel
    public_url = ngrok.connect(8000, bind_tls=True)
    public_url = str(public_url).replace('NgrokTunnel: "', '').replace('" -> "http://127.0.0.1:8000"', '')
    
    print(f"✅ ngrok tunnel created: {public_url}")
    
except Exception as e:
    print(f"❌ ngrok failed: {e}")
    print("🔄 Falling back to Cloudflare...")
    
    # Fallback to cloudflared
    if not Path("/usr/local/bin/cloudflared").exists():
        print("🔧 Installing Cloudflare tunnel...")
        subprocess.run("wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared", shell=True, check=True)
        subprocess.run("chmod +x /usr/local/bin/cloudflared", shell=True, check=True)

    # Start Cloudflare tunnel with retries
    for attempt in range(3):
        print(f"🌐 Cloudflare tunnel attempt {attempt + 1}/3...")
        tunnel = subprocess.Popen(
            "cloudflared tunnel --url http://127.0.0.1:8000 --no-autoupdate",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        public_url = None
        for _ in range(40):  # Reduced timeout
            line = tunnel.stdout.readline()
            if line:
                print(line, end="")
                m = re.search(r"https://[-a-zA-Z0-9.]+trycloudflare.com", line)
                if m:
                    public_url = m.group(0)
                    break
            time.sleep(1)
        
        if public_url:
            break
        else:
            tunnel.terminate()
            print(f"❌ Attempt {attempt + 1} failed, retrying...")
            time.sleep(5)
    
    if not public_url:
        raise RuntimeError("All tunnel methods failed. Try restarting the Colab runtime and running again.")

os.environ["PUBLIC_BASE_URL"] = public_url

print("\n" + "="*70)
print("✅ PRODUCTION TEXT-TO-VIDEO API IS LIVE")
print("="*70)
print(f"📡 Base URL: {public_url}")
print(f"🎬 Generate endpoint: {public_url}/generate_sync")
print(f"💚 Health check: {public_url}/health")
print(f"🎥 Model: {os.environ['MODEL_ID']}")
print(f"📐 Max resolution: 576×320 (16:9 aspect ratio)")
print(f"⚡ Recommended: 24 frames, 40 steps, guidance 17.5")
print("="*70)
print("\n⚠️  Keep this Colab tab open while n8n generates videos\n")

# Test the API
try:
    import requests
    response = requests.get(f"{public_url}/health", timeout=10)
    if response.status_code == 200:
        print("✅ API health check passed")
    else:
        print(f"⚠️  API health check returned {response.status_code}")
except Exception as e:
    print(f"⚠️  Could not verify API health: {e}")

# Keep alive
while True:
    time.sleep(60)