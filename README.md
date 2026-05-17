# Colab Text-to-Video API + n8n Google Sheets Workflow

This repository gives you a lightweight Google Colab setup that exposes a text-to-video model through an HTTP API, plus an n8n workflow that reads prompts from Google Sheets and saves generated MP4 files on your local disk under `~/video/<subfolder>/` using n8n's SSH node.

## 1) Model used

Default model:

```text
damo-vilab/text-to-video-ms-1.7b
```

It is one of the more practical open text-to-video models for Colab T4 when you keep outputs small, for example `256x256`, `16` frames, and `20-30` steps.

You can change it in Colab:

```python
import os
os.environ["MODEL_ID"] = "damo-vilab/text-to-video-ms-1.7b"
```

## 2) Files

- `app.py` - FastAPI text-to-video server.
- `requirements_colab.txt` - Colab Python dependencies.
- `colab_setup_and_run.py` - code to run inside Google Colab.
- `n8n_workflow.json` - importable n8n workflow template.
- `google_sheet_template.csv` - sample Google Sheet columns.

## 3) Run on Google Colab

Create a new Colab notebook with GPU enabled:

`Runtime > Change runtime type > T4 GPU`

Then run:

```bash
!git clone YOUR_REPOSITORY_URL colab_text2video_n8n
%cd colab_text2video_n8n
```

If you upload these files manually instead of using GitHub, just place them in the Colab working directory and run:

```python
exec(open("colab_setup_and_run.py").read())
```

The cell will print something like:

```text
Base URL: https://example-random.trycloudflare.com
Generation endpoint: https://example-random.trycloudflare.com/generate_sync
Health endpoint: https://example-random.trycloudflare.com/health
```

Keep the Colab cell running while n8n is generating videos.

## 4) API endpoint

### Health

```http
GET https://YOUR-COLAB-TUNNEL.trycloudflare.com/health
```

### Generate video synchronously

```http
POST https://YOUR-COLAB-TUNNEL.trycloudflare.com/generate_sync
Content-Type: application/json
```

Body:

```json
{
  "prompt": "cinematic documentary b-roll of ancient ruins at sunrise, realistic, slow drone movement",
  "negative_prompt": "cartoon, text, watermark, low quality",
  "seed": 12345,
  "num_frames": 16,
  "num_inference_steps": 25,
  "guidance_scale": 9,
  "width": 256,
  "height": 256,
  "fps": 8,
  "subfolder": "episode-01",
  "filename_prefix": "ancient-ruins"
}
```

Response:

```json
{
  "job_id": "...",
  "status": "completed",
  "file_name": "ancient-ruins-abc12345-seed12345.mp4",
  "file_path": "/content/generated_videos/episode-01/ancient-ruins-abc12345-seed12345.mp4",
  "video_url": "https://YOUR-COLAB-TUNNEL.trycloudflare.com/files/episode-01/ancient-ruins-abc12345-seed12345.mp4"
}
```

## 5) Google Sheet setup

Create a Google Sheet with these headers:

```csv
row_id,prompt,negative_prompt,subfolder,filename_prefix,seed,num_frames,steps,guidance_scale,width,height,fps,status,video_url,local_path
```

You can import `google_sheet_template.csv`.

Recommended generation settings for Colab T4:

- `width`: `256`
- `height`: `256`
- `num_frames`: `16`
- `steps`: `20` to `30`
- `fps`: `8`

## 6) n8n setup on local system

Your n8n is running locally at:

```text
http://localhost:5678
```

### Required credentials

1. **Google Sheets OAuth2 credential** in n8n.
2. **SSH credential** pointing to your local machine.

For the SSH node to save files locally, your local machine must accept SSH connections.

Linux example:

```bash
sudo apt update
sudo apt install openssh-server curl -y
sudo systemctl enable --now ssh
ssh localhost
```

macOS:

```text
System Settings > General > Sharing > Remote Login > On
```

Windows:

Install and enable OpenSSH Server, then test:

```powershell
ssh localhost
```

The workflow saves files to:

```text
$HOME/video/<subfolder>/<generated-file>.mp4
```

## 7) Import n8n workflow

1. Open `http://localhost:5678`.
2. Import `n8n_workflow.json`.
3. Replace placeholders:
   - `PUT_GOOGLE_SHEET_ID_HERE`
   - Google Sheets credential ID/name
   - SSH credential ID/name
   - `PASTE_COLAB_PUBLIC_URL_HERE/generate_sync`
4. Or set the n8n environment variable:

```bash
COLAB_T2V_URL=https://YOUR-COLAB-TUNNEL.trycloudflare.com/generate_sync
```

5. Run the workflow manually first.

## 8) What the workflow does

1. Reads rows from Google Sheets.
2. Filters rows where `prompt` exists and `status` is not `done`.
3. Sends each prompt to Colab endpoint `/generate_sync`.
4. Receives `video_url` from Colab.
5. Runs SSH command on your local machine:

```bash
mkdir -p "$HOME/video/<subfolder>" && curl -L --fail --retry 3 "<video_url>" -o "$HOME/video/<subfolder>/<filename>.mp4"
```

6. Updates the Google Sheet status.

## 9) Notes

- Free Colab sessions disconnect; keep the tab active while generating.
- The Cloudflare quick tunnel URL changes every time you restart the Colab cell.
- For longer/higher-resolution documentary clips, generate multiple short b-roll clips and edit/upscale later.
- If generation fails with CUDA out-of-memory, reduce `width`, `height`, `num_frames`, or `steps`.
