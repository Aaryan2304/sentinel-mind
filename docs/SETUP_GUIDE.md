# Phase 1 Setup Guide

Complete environment setup for Sentinel Mind development.
Follow this guide to get from a fresh WSL2 install to a running SimSat simulation.

---

## Prerequisites

- Windows 11 with WSL2 (Ubuntu 24.04)
- Miniconda or Python 3.13+ in WSL2
- Git configured with `main` as default branch

---

## 1. Docker Installation (WSL2)

Docker is required for SimSat. Install Docker Engine directly in WSL2 (no Docker Desktop needed).

### Install

```bash
# Add Docker's official GPG key
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add Docker repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
| sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
```

### Configure Permissions

```bash
# Add your user to docker group (avoids needing sudo for every command)
sudo usermod -aG docker $USER
```

**IMPORTANT:** After running `usermod`, the group change does NOT take effect in the current terminal.
You must either:
- Log out of WSL and log back in (recommended)
- OR run `newgrp docker` in the current terminal (activates for current session only)

### Start Docker Daemon

```bash
# Check if Docker is already running
docker --version
docker compose version

# If "docker: command not found" or permission denied:
sudo dockerd &

# If you see "process with PID X is still running" — Docker is already running.
# That's fine, just proceed.
```

### Verify

```bash
docker --version
# Expected: Docker version 29.x.x, build xxxxx

docker compose version
# Expected: Docker Compose version v5.x.x
```

### Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `permission denied while trying to connect to the docker API` | User not in docker group or group not activated | Run `newgrp docker` or log out/in |
| `process with PID X is still running` | Docker daemon already running | Ignore — Docker is working |
| `unable to connect to docker API` | Docker daemon not started | Run `sudo dockerd &` |

---

## 2. Python Packages

Install all required packages in your existing conda environment.

```bash
# Core ML (likely already installed)
pip install torch torchvision transformers

# Fine-tuning (partner's task, but useful to have)
pip install trl peft accelerate datasets

# Inference
pip install onnxruntime

# Dashboard
pip install streamlit folium streamlit-folium pandas

# Image processing
pip install pillow

# SimSat API client
pip install requests

# Already installed in most ML envs: numpy, matplotlib, scipy, huggingface_hub
```

### Verify

```python
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')

from transformers import __version__ as tv
print(f'Transformers: {tv}')

from trl import __version__ as trl_v
print(f'TRL: {trl_v}')

import streamlit, folium, pandas, onnxruntime
print('Dashboard + inference packages: OK')
"
```

---

## 3. SimSat Setup

SimSat is a Docker-based satellite simulation with:
- **Dashboard** (port 8000): Cesium globe showing satellite orbit
- **Sim API** (port 9005): FastAPI for fetching satellite position and Sentinel-2 images

### Clone and Fix

The SimSat reference is already cloned at `simsat-reference/` in the project root.

**Known Issue:** The sim container crashes without `MAPBOX_ACCESS_TOKEN`.
We fixed this by making the Mapbox provider optional. If you re-clone SimSat,
apply this fix to `simsat-reference/src/sim/api.py`:

```python
# BEFORE (line ~12):
mapbox = MapboxlProvider()

# AFTER:
try:
    mapbox = MapboxlProvider()
except ValueError:
    mapbox = None
    print("WARNING: MAPBOX_ACCESS_TOKEN not set. Mapbox endpoints will be unavailable.")
```

Also add null checks before the two `mapbox.get_target_image()` calls (~line 152 and ~line 257):

```python
if mapbox is None:
    raise HTTPException(status_code=503, detail="Mapbox API unavailable.")
```

### Start SimSat

```bash
cd ~/projects/sentinel-mind/simsat-reference

# Activate docker group if needed
newgrp docker

# Build and start (first run takes ~5 minutes to build images)
docker compose up
```

Expected output:
```
[+] Building ... FINISHED
[+] up 3/3
Attaching to fakesat-dashboard, fakesat-sim
fakesat-dashboard | Watching for file changes with StatReloader
fakesat-sim | WARNING: MAPBOX_ACCESS_TOKEN not set...
fakesat-sim | INFO: Uvicorn running on http://0.0.0.0:8000
```

### Start the Simulation

The simulation does NOT auto-start. You must send a "start" command to the dashboard.

```bash
# Start simulation (10x speed, 20-second time steps)
curl -X POST http://localhost:8000/api/commands/ \
  -H "Content-Type: application/json" \
  -d '{"command": "start", "parameters": {"step_size_seconds": 20, "replay_speed": 10.0}}'

# Expected: {"id": X, "command": "start", "parameters": {}, ...}
```

Other commands:
```bash
# Pause simulation
curl -X POST http://localhost:8000/api/commands/ \
  -H "Content-Type: application/json" \
  -d '{"command": "pause"}'

# Reset simulation
curl -X POST http://localhost:8000/api/commands/ \
  -H "Content-Type: application/json" \
  -d '{"command": "stop"}'
```

---

## 4. Verify SimSat API Endpoints

### 4.1 Satellite Position

```bash
curl -s http://localhost:9005/data/current/position | python3 -m json.tool
```

Expected response:
```json
{
    "lon-lat-alt": [-122.27, 61.09, 798.99],
    "timestamp": "2026-04-19T05:36:11Z"
}
```

- `lon-lat-alt`: [longitude, latitude, altitude_km]
- `timestamp`: UTC time of the satellite position

If position is `[0.0, 0.0, 0.0]` with timestamp `1970-01-01`, the simulation hasn't started.
Send the start command (Section 3).

### 4.2 Sentinel-2 Image

```bash
# Fetch RGB image at current satellite position
curl -s -o /tmp/test.png -w "HTTP %{http_code}, Size: %{size_download} bytes\n" \
  "http://localhost:9005/data/image/sentinel?lon=-122.27&lat=61.09&timestamp=2026-04-19T05:36:11Z&spectral_bands=red&spectral_bands=green&spectral_bands=blue&size_km=5.0&return_type=png"
```

Expected: HTTP 200 with a PNG file (typically 1-50KB depending on scene complexity).

**Check metadata** (included in response header):
```bash
curl -s -D - "http://localhost:9005/data/image/sentinel?lon=-122.27&lat=61.09&timestamp=2026-04-19T05:36:11Z&spectral_bands=red&spectral_bands=green&spectral_bands=blue&size_km=5.0&return_type=png" 2>&1 | grep sentinel_metadata
```

Metadata fields:
- `image_available`: true/false — false if satellite is over ocean or polar regions
- `source`: data source (e.g., "sentinel-2c")
- `spectral_bands`: requested bands
- `cloud_cover`: percentage (0-100)
- `datetime`: image capture time
- `footprint`: [lon_min, lat_min, lon_max, lat_max]

### 4.3 Multi-Spectral Images

Fetch the same location with different band combinations:

```bash
# False Color IR (vegetation analysis)
curl -s -o /tmp/false_color.png \
  "http://localhost:9005/data/image/sentinel?lon=X&lat=Y&timestamp=Z&spectral_bands=nir&spectral_bands=red&spectral_bands=green&size_km=5.0&return_type=png"

# SWIR (burn/clearing detection)
curl -s -o /tmp/swir.png \
  "http://localhost:9005/data/image/sentinel?lon=X&lat=Y&timestamp=Z&spectral_bands=swir22&spectral_bands=nir&spectral_bands=green&size_km=5.0&return_type=png"

# Red-Edge (vegetation stress)
curl -s -o /tmp/rededge.png \
  "http://localhost:9005/data/image/sentinel?lon=X&lat=Y&timestamp=Z&spectral_bands=rededge1&spectral_bands=rededge2&spectral_bands=rededge3&size_km=5.0&return_type=png"
```

Available band names: `red`, `green`, `blue`, `nir`, `swir16`, `swir22`, `rededge1`, `rededge2`, `rededge3`

### 4.4 Dashboard (Web UI)

Open in browser: **http://localhost:8000**

The dashboard shows:
- Cesium 3D globe with satellite orbit path
- Current satellite position
- Satellite footprint on the ground
- Telemetry data

You can also interact with the simulation through the dashboard UI (start/pause/reset buttons).

---

## 5. View Current Satellite State

To see what the satellite is imaging right now:

### Step 1: Get current position
```bash
curl -s http://localhost:9005/data/current/position
```

### Step 2: Fetch image at that position
```bash
# Extract lon, lat, timestamp from step 1, then:
curl -s -o /tmp/current_view.png \
  "http://localhost:9005/data/image/sentinel?lon=LON&lat=LAT&timestamp=TIMESTAMP&spectral_bands=red&spectral_bands=green&spectral_bands=blue&size_km=5.0&return_type=png"

# View it
xdg-open /tmp/current_view.png
```

### Quick Script: View Current Satellite Image

```bash
#!/bin/bash
# save as: view_satellite.sh

# Get position
POS=$(curl -s http://localhost:9005/data/current/position)
LON=$(echo $POS | python3 -c "import sys,json; print(json.load(sys.stdin)['lon-lat-alt'][0])")
LAT=$(echo $POS | python3 -c "import sys,json; print(json.load(sys.stdin)['lon-lat-alt'][1])")
TS=$(echo $POS | python3 -c "import sys,json; print(json.load(sys.stdin)['timestamp'])")

echo "Satellite at: lon=$LON, lat=$LAT, time=$TS"

# Fetch RGB image
curl -s -o /tmp/sat_view.png \
  "http://localhost:9005/data/image/sentinel?lon=$LON&lat=$LAT&timestamp=$TS&spectral_bands=red&spectral_bands=green&spectral_bands=blue&size_km=5.0&return_type=png"

echo "Image saved to /tmp/sat_view.png"
xdg-open /tmp/sat_view.png 2>/dev/null || echo "Open /tmp/sat_view.png manually"
```

---

## 6. Using the Python Collector

Once SimSat is running, you can use our collector module:

```python
from src.pipeline.collector import get_satellite_position, get_sentinel_image

# Get current position
pos = get_satellite_position()
print(f"Satellite at: lon={pos['lon']:.2f}, lat={pos['lat']:.2f}, alt={pos['alt']:.0f}km")

# Get RGB image
image, metadata = get_sentinel_image(
    lon=pos['lon'],
    lat=pos['lat'],
    timestamp=pos['timestamp'],
    bands=["red", "green", "blue"],
    size_km=5.0,
)

if image is not None:
    print(f"Image: {image.size}, cloud cover: {metadata['cloud_cover']:.1f}%")
    image.save("/tmp/satellite_view.png")
else:
    print("No image available (ocean/polar region or cloudy)")
```

---

## 7. Stopping and Cleaning Up

```bash
# Stop SimSat (in the terminal running docker compose up)
Ctrl+C

# Remove containers (keeps built images)
docker compose down

# Remove containers AND built images (full clean)
docker compose down --rmi all -v

# Check running containers
docker ps

# Check Docker disk usage
docker system df
```

---

## Quick Reference

| Service | URL | Purpose |
|---------|-----|---------|
| Dashboard | http://localhost:8000 | Cesium globe, simulation controls |
| Sim API | http://localhost:9005 | Satellite position + Sentinel images |
| API Docs | http://localhost:9005/docs | FastAPI interactive docs (limited) |

| API Endpoint | Method | Description |
|-------------|--------|-------------|
| `/data/current/position` | GET | Current satellite position |
| `/data/current/image/sentinel` | GET | Image at current position |
| `/data/image/sentinel` | GET | Image at specific location/time |
| `/data/image/mapbox` | GET | Mapbox image (requires API key) |
