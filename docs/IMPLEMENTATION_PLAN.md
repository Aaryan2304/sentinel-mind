# Sentinel Mind — Implementation Plan

Autonomous Satellite Intelligence with Investigative VLM Reasoning
Liquid AI x DPhi Space Hackathon | April 13 - May 9, 2026

---

## Assumptions

- Partner's RTX 4060 Ti (16GB) handles fine-tuning
- User's RTX 3050 Ti (4GB) handles inference and dashboard
- Google Colab T4 (16GB) free tier as fallback for fine-tuning
- All services are free: Docker, Python, Streamlit, HuggingFace
- SimSat runs locally via Docker (no cloud costs)
- VRSBench dataset downloaded from HuggingFace (12.5 GB)

## Technical References

- Liquid AI TRL Guide: docs.liquid.ai/customization/finetuning-frameworks/trl
- Liquid AI Satellite Example: docs.liquid.ai/examples/customize-models/satellite-vlm
- Unsloth LFM2.5: unsloth.ai/docs/models/tutorials/lfm2.5
- Unsloth Vision FT: unsloth.ai/docs/basics/vision-fine-tuning
- VRSBench: huggingface.co/datasets/xiang709/VRSBench
- ONNX Model: huggingface.co/LiquidAI/LFM2.5-VL-450M-ONNX
- SimSat: github.com/DPhi-Space/SimSat

## Models

Two variants of LFM2.5-VL-450M are used, each for a specific purpose:

### Fine-tuning: LiquidAI/LFM2.5-VL-450M (HuggingFace Safetensors)
- **Format:** HuggingFace Transformers (safetensors)
- **Size:** ~0.9 GB (FP16)
- **Purpose:** LoRA fine-tuning with TRL SFTTrainer + PEFT
- **Why this one:** TRL and PEFT require the HuggingFace model format. ONNX and GGUF
  variants are inference-only and cannot be trained with LoRA adapters.
- **Used by:** Partner (RTX 4060 Ti 16GB)

### Inference: LiquidAI/LFM2.5-VL-450M-ONNX
- **Format:** ONNX (FP16 encoder + Q4 decoder)
- **Size:** ~770 MB
- **Purpose:** Optimized inference for dashboard and demo
- **Why this one:** ONNX Runtime is faster and more memory-efficient than raw transformers
  inference. Critical for the RTX 3050 Ti (4GB VRAM). Also aligns with the "space compute"
  narrative since ONNX is the deployment format for NVIDIA Orin-class hardware.
- **Used by:** User (RTX 3050 Ti 4GB)
- **Fallback:** If the fine-tuned model needs to be exported to ONNX, use `optimum-cli
  export onnx`. The pre-quantized ONNX from Liquid AI is the fallback if export fails.

**Note:** The GGUF variant (LiquidAI/LFM2.5-VL-450M-GGUF) is for CPU-only inference
via llama.cpp. Not needed for this project since both team members have NVIDIA GPUs.

## Team & Phase Ownership

| Phase | Owner | Rationale |
|-------|-------|-----------|
| Phase 1: Environment Setup | **Common** | Both need Docker/Python env |
| Phase 2: Data Collection | **User** | SimSat API integration, data pipeline |
| Phase 3: Spectral Analysis | **User** | CV/image processing domain expertise |
| Phase 4: VLM Fine-tuning | **Partner** | Training hardware (RTX 4060 Ti), model optimization |
| Phase 5: Inference Pipeline | **User** | Integrates fine-tuned model into dashboard pipeline |
| Phase 6: Dashboard | **User** | Streamlit UI, visualization, demo interface |
| Phase 7: Integration & Testing | **Common** | Both must verify their components work end-to-end |
| Phase 8: Demo & Submission | **Common** | Joint effort for recording and submission |

---

## Phase 1: Environment Setup (Days 1-2) [COMMON]

### 1.1 SimSat Docker Environment

Goal: Get the satellite simulation running locally.

Steps:

1. Verify Docker is installed and running

```bash
docker --version
docker compose version
```

2. Navigate to simsat-reference directory

```bash
cd ~/projects/geospatial-hackathon/simsat-reference
```

3. Start SimSat containers

```bash
docker compose up
```

- Dashboard: http://localhost:8000
- API: http://localhost:9005

4. Verify API endpoints

```
GET /data/current/position          — satellite position
GET /data/current/image/sentinel    — current image (RGB)
GET /data/image/sentinel            — specific location/time
GET /data/image/mapbox              — Mapbox image (needs API key)
```

5. Run existing test script

```bash
python scripts/api_test.py sentinel
python scripts/api_test.py sentinel_current
```

Verification:

- Dashboard loads at localhost:8000 with Cesium globe
- API returns position data with lon/lat/alt/timestamp
- Sentinel image endpoint returns PNG or array data

### 1.2 Python Environment

Goal: Set up development environment.

1. Create virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

2. Install core dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers>=4.55.0 trl>=0.9.0 peft accelerate
pip install datasets huggingface_hub
pip install streamlit folium streamlit-folium
pip install pillow numpy pandas matplotlib
pip install requests pystac-client odc-stac
pip install onnxruntime
```

3. Verify GPU availability

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

4. Download LFM2.5-VL-450M for testing

```python
from transformers import AutoProcessor, AutoModelForImageTextToText

model = AutoModelForImageTextToText.from_pretrained(
    "LiquidAI/LFM2.5-VL-450M",
    dtype="bfloat16"
)
processor = AutoProcessor.from_pretrained("LiquidAI/LFM2.5-VL-450M")
print("Model loaded successfully")
```

Verification:

- All packages install without errors
- GPU detected (if available)
- Model loads and can generate text from an image

### 1.3 Project Structure

Create the following directory layout:

```
src/
  pipeline/
    __init__.py
    collector.py      # SimSat API data collector
    spectral.py       # Spectral index computation
    change.py         # Temporal change detection
  models/
    __init__.py
    finetune.py       # LoRA fine-tuning with TRL+PEFT
    inference.py      # VLM inference pipeline
    prompts.py        # Investigation prompt templates
  dashboard/
    __init__.py
    app.py            # Streamlit application
configs/
  training.yaml       # Training hyperparameters
requirements.txt
```

---

## Phase 2: Data Collection (Days 3-5) [USER]

### 2.1 SimSat Data Collector

Goal: Build a script that collects Sentinel-2 images from SimSat API.

File: `src/pipeline/collector.py`

Functions to implement:

- `get_satellite_position() -> dict`

Calls `GET /data/current/position`. Returns `{lon, lat, alt, timestamp}`.

- `get_sentinel_image(lon, lat, timestamp, bands, size_km) -> tuple[PIL.Image, dict]`

Calls `GET /data/image/sentinel` with `spectral_bands` param. Returns `(PIL Image, metadata dict with cloud_cover, datetime, source)`.

- `get_multispectral_image(lon, lat, timestamp, bands_list) -> dict`

Fetches same location with different band combinations. Returns `{band_combo_name: PIL.Image}`.

- `collect_orbit_sequence(num_positions=20) -> list[dict]`

Starts simulation, collects position + images at intervals. Returns `list of {position, image, metadata, timestamp}`.

- `save_collection(data, output_dir) -> None`

Saves images + metadata to structured directory. Format: `output_dir/{timestamp}/image_{bands}.png + metadata.json`.

Band combinations to collect:

1. **RGB**: `["red", "green", "blue"]` — visualization
2. **False Color IR**: `["nir", "red", "green"]` — vegetation
3. **SWIR**: `["swir22", "nir", "green"]` — burn/clearing detection
4. **Agriculture**: `["swir16", "nir", "red"]` — crop health
5. **Red-Edge**: `["rededge1", "rededge2", "rededge3"]` — vegetation stress

Data collection strategy:

- Run SimSat with fast replay speed (10x)
- Collect images every 60 simulation seconds
- Target: 100-200 images across different regions
- Filter: only collect when `image_available=True`
- Save metadata: cloud_cover, datetime, source, footprint, bands

Estimated time: 2-3 hours (API is slow for Sentinel data).

### 2.2 VRSBench Dataset Download

Goal: Download VRSBench for fine-tuning.

```python
from datasets import load_dataset

dataset = load_dataset("xiang709/VRSBench", split="train")
```

Dataset details:

- Size: ~12.5 GB
- Contains: 29,614 images, 123,221 VQA pairs, 52,472 grounding refs, 29,614 captions
- Save to `data/vrsbench/` directory

Filter for relevant tasks:

- Captioning: 29,614 entries
- VQA: 123,221 entries
- Visual Grounding: 52,472 entries

Note: VRSBench is large. Download on stable connection. Can use Google Colab for download if local bandwidth is limited.

Verification:

- Dataset loads without errors
- Images are accessible
- Text annotations are in expected format

### 2.3 Custom Dataset Creation

Goal: Create spectral analysis Q&A pairs for fine-tuning.

For each collected SimSat image, generate:

1. Spectral index computation (NDVI, NDWI, NBR)
2. Change detection vs baseline (if available)
3. Investigation Q&A pair

Example Q&A pair:

```
Image: False-color IR composite (NIR-Red-Green)

Q: "Analyze this satellite image. What changes are visible compared
    to typical vegetation patterns? Is this natural or human activity?"

A: "The image shows a large clearing in what was previously dense
    forest. NDVI dropped from 0.75 to 0.35 in the cleared area.
    The clearing pattern is rectangular with straight edges, consistent
    with commercial logging rather than natural causes. A road
    construction is visible on the eastern edge. Severity: HIGH.
    Recommend priority downlink for enforcement action."
```

Generate 50-100 such pairs from collected data. Save in JSONL format compatible with TRL SFTTrainer.

---

## Phase 3: Spectral Analysis Engine (Days 5-7) [USER]

### 3.1 Spectral Index Computation

Goal: Compute vegetation/water/burn indices from Sentinel-2 bands.

File: `src/pipeline/spectral.py`

Formulas (Sentinel-2 band names):

- NDVI = `(B8 - B4) / (B8 + B4)` — Vegetation greenness
- NDWI = `(B3 - B8) / (B3 + B8)` — Water/moisture content
- NBR = `(B8 - B12) / (B8 + B12)` — Burn severity
- NDRE = `(B8A - B5) / (B8A + B5)` — Chlorophyll/stress detection
- NDMI = `(B8 - B11) / (B8 + B11)` — Vegetation water content
- EVI2 = `2.5 * (B8 - B4) / (B8 + 2.4*B4 + 1)` — Enhanced vegetation
- OSAVI = `1.16 * (B8 - B4) / (B8 + B4 + 0.16)` — Soil-adjusted

Important: Compute on reflectance (0-1), not raw DN.

- Scale factor: 10000 (Sentinel-2 quantization)
- `reflectance = DN / 10000`

Functions to implement:

- `compute_ndvi(nir, red) -> np.ndarray`
- `compute_ndwi(green, nir) -> np.ndarray`
- `compute_nbr(nir, swir22) -> np.ndarray`
- `compute_all_indices(bands_dict) -> dict[str, np.ndarray]`
- `detect_anomaly(current_indices, baseline_indices) -> dict`

Returns `{index_name: {change_magnitude, direction, severity}}`.

Note: SimSat API returns images as PNG (3-band) or array (multi-band). For multi-spectral analysis, use `return_type="array"` to get raw band data.

### 3.2 Change Detection

Goal: Compare current image with baseline to detect changes.

File: `src/pipeline/change.py`

Functions to implement:

- `load_baseline(region_id) -> dict` — Loads previous image + indices for a region
- `compute_change_map(current_indices, baseline_indices) -> np.ndarray` — Per-pixel change magnitude across all indices
- `detect_change_regions(change_map, threshold) -> list[dict]` — Segments change regions with bounding boxes. Returns `[{bbox, area, mean_change, dominant_index}]`
- `rank_changes(changes) -> list[dict]` — Sorts by severity (high change + large area = high priority)
- `save_baseline(region_id, image, indices, timestamp) -> None` — Saves current data as new baseline for future comparisons

Storage: Simple file-based (JSON + numpy arrays). No database needed for hackathon scope.

---

## Phase 4: VLM Fine-tuning (Days 7-12) [PARTNER]

### 4.1 Dataset Preparation

Goal: Format data for TRL SFTTrainer.

File: `src/models/prepare_dataset.py`

Steps:

1. Load VRSBench dataset
2. Load custom spectral Q&A pairs
3. Convert to TRL conversation format:

```python
{
    "messages": [
        {"role": "system", "content": "You are a satellite imagery analysis AI."},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "Analyze this satellite image..."}
        ]},
        {"role": "assistant", "content": "The image shows..."}
    ],
    "images": [<PIL.Image>]
}
```

4. Split: 90% train, 10% validation
5. Save as HuggingFace dataset

Dataset composition:

- VRSBench captioning: ~26,650 train / ~2,960 val
- VRSBench VQA: ~110,900 train / ~12,320 val
- Custom spectral Q&A: ~80 train / ~20 val
- Total: ~137,630 train / ~15,300 val

### 4.2 LoRA Fine-tuning with TRL

Goal: Fine-tune LFM2.5-VL-450M with LoRA.

File: `src/models/finetune.py`

Based on: docs.liquid.ai/customization/finetuning-frameworks/trl

```python
from transformers import AutoModelForImageTextToText, AutoProcessor
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from datasets import load_dataset

# Load model
model = AutoModelForImageTextToText.from_pretrained(
    "LiquidAI/LFM2.5-VL-450M",
    dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("LiquidAI/LFM2.5-VL-450M")

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    bias="none",
    task_type="CAUSAL_LM"
)

# Training config
training_config = SFTConfig(
    output_dir="./checkpoints/sentinel-mind",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=True,
    dataset_kwargs={"skip_prepare_dataset": False}
)

# Trainer
trainer = SFTTrainer(
    model=model,
    args=training_config,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=lora_config,
    processing_class=processor
)

trainer.train()
trainer.save_model("./checkpoints/sentinel-mind-best")
```

Hardware requirements:

- RTX 4060 Ti (16GB): Should work with batch_size=4, gradient_accumulation=4
- Colab T4 (16GB): Same settings, may need batch_size=2
- Estimated time: 2-4 hours for 3 epochs on ~137K samples

Fallback: If TRL has issues, use Unsloth:

```python
from unsloth import FastVisionModel

model, tokenizer = FastVisionModel.from_pretrained("LiquidAI/LFM2.5-VL-450M")
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    target_modules="all-linear"
)
```

See: unsloth.ai/docs/basics/vision-fine-tuning

### 4.3 Model Evaluation

Goal: Verify fine-tuning improved satellite imagery understanding.

Steps:

1. Run base model on test images — get baseline outputs
2. Run fine-tuned model on same images — get improved outputs
3. Compare: investigation quality, spectral awareness, domain vocabulary
4. Manual evaluation (no automated metric needed for hackathon)

Test prompts:

- "Analyze this satellite image for vegetation health"
- "Is there evidence of deforestation in this region?"
- "What spectral anomalies are visible?"
- "Describe the land use in this image"

Save best checkpoint for inference.

### 4.4 ONNX Export (Optional, Partner Task)

Goal: Export fine-tuned model for optimized inference.

Note: Fine-tuned model export to ONNX is non-trivial. For the hackathon, use the HuggingFace Transformers pipeline for inference. ONNX export is a nice-to-have for the demo but not required.

If pursued:

```bash
pip install optimum[onnxruntime]
optimum-cli export onnx --model ./checkpoints/sentinel-mind-best ./onnx-export/
```

Alternatively, use the pre-quantized ONNX from Liquid AI as fallback:
huggingface.co/LiquidAI/LFM2.5-VL-450M-ONNX (FP16 encoder + Q4 decoder, ~770MB)

---

## Phase 5: Inference Pipeline (Days 12-15) [USER]

### 5.1 VLM Inference Module

Goal: Run the fine-tuned model on satellite imagery.

File: `src/models/inference.py`

Functions to implement:

- `load_model(checkpoint_path) -> (model, processor)` — Loads fine-tuned model from checkpoint
- `investigate_image(image, spectral_data, model, processor) -> str` — Full investigation pipeline:
  1. Prepare multi-spectral composite image
  2. Compute spectral indices
  3. Format investigation prompt with context
  4. Run VLM inference
  5. Parse structured output (severity, confidence, recommendation)
- `parse_investigation(report) -> dict` — Extracts structured fields from VLM output. Returns `{severity, confidence, findings, recommendation}`
- `compress_report(report) -> bytes` — Compresses investigation report for downlink. Returns compressed bytes (target: <50KB)

### 5.2 Investigation Prompts

Goal: Define prompt templates for different use cases.

File: `src/models/prompts.py`

Forest investigation template:

```
You are a satellite-based forest monitoring AI. Analyze this multi-spectral
satellite imagery (NIR-Red-Green composite).

Current image captured: {timestamp}
Region: {region_name}, coordinates [{lat}, {lon}]
Cloud cover: {cloud_cover}%
Image size: {size_km}km x {size_km}km

Spectral Analysis:
- NDVI (vegetation): {ndvi_mean:.2f} (baseline: {ndvi_baseline:.2f})
- NDWI (water): {ndwi_mean:.2f}
- NBR (burn): {nbr_mean:.2f}
- NDVI change from baseline: {ndvi_change:.1f}%

Investigate:
1. What changes are visible in this image?
2. Does this pattern match natural forest dynamics or human activity?
3. What is the severity? (low/medium/high/critical)
4. What action should enforcement or response teams take?
```

Maritime investigation template:

```
You are a maritime surveillance AI. Analyze this satellite imagery
for vessel activity.

Current image captured: {timestamp}
Region: {region_name}, coordinates [{lat}, {lon}]
Expected vessel traffic: {traffic_level}

Investigate:
1. Are any vessels visible in this image?
2. Do you see wake patterns consistent with active fishing or transit?
3. Is there evidence of oil slicks or environmental disturbance?
4. What is the severity? (low/medium/high/critical)
```

Wildfire investigation template:

```
You are a wildfire detection AI. Analyze this satellite imagery
for fire indicators.

Current image captured: {timestamp}
Region: {region_name}, coordinates [{lat}, {lon}]
Recent weather: {weather_conditions}

Spectral Analysis:
- NBR (burn): {nbr_mean:.2f}
- SWIR thermal anomaly: {thermal_status}

Investigate:
1. Are there thermal anomalies or burn scars visible?
2. What is the fire front extent and direction?
3. Are populated areas at risk?
4. What is the severity? (low/medium/high/critical)
```

---

## Phase 6: Streamlit Dashboard (Days 15-19) [USER]

### 6.1 Dashboard Application

Goal: Build the demo interface.

File: `src/dashboard/app.py`

Dependencies: streamlit, folium, streamlit-folium, pillow, matplotlib

Layout:

```
[Header: "Sentinel Mind — Autonomous Satellite Intelligence"]

[Left Sidebar: Simulation Controls]
- Start/Stop/Pause buttons
- Replay speed slider
- Region selector (Amazon, Congo, etc.)

[Main Area - Row 1: Map + Image]
- Left: Folium map showing satellite orbit path
- Right: Current captured image (RGB composite)

[Main Area - Row 2: Spectral Analysis]
- NDVI heatmap overlay
- NDWI heatmap overlay
- Change detection overlay (vs baseline)
- Spectral index statistics panel

[Main Area - Row 3: VLM Investigation]
- Investigation report (formatted text)
- Severity badge (low/medium/high/critical)
- Confidence score
- Recommended action

[Main Area - Row 4: Downlink Queue]
- Priority-sorted table of investigations
- Compression stats (raw size vs compressed)
- Bandwidth utilization bar
```

### 6.2 Dashboard Components

Map component:

- Use streamlit-folium for interactive map
- Show satellite position as moving marker
- Draw orbit path as polyline
- Show captured image footprints as rectangles
- Color-code by severity (green=low, yellow=medium, red=high/critical)

Image component:

- Display current Sentinel-2 image
- Toggle between band combinations (RGB, False Color, SWIR)
- Show cloud cover percentage
- Show capture timestamp

Spectral analysis component:

- Matplotlib heatmaps for NDVI, NDWI, NBR
- Color bars for index ranges
- Statistics panel: mean, min, max, std for each index
- Change indicator: arrow up/down with percentage

VLM component:

- Display investigation report as formatted text
- Color-coded severity badge
- Confidence progress bar
- Expandable details section

Queue component:

- DataFrame table: timestamp, region, severity, confidence, size
- Color rows by severity
- Show total bandwidth used vs available

### 6.3 Dashboard Implementation Notes

Key Streamlit patterns:

- Use `st.session_state` for persistent data across reruns
- Cache model loading with `@st.cache_resource`
- Use `st.spinner` for long operations (API calls, inference)
- Auto-refresh every 10 seconds during simulation

---

## Phase 7: Integration and Testing (Days 19-21) [COMMON]

### 7.1 End-to-End Integration

Goal: Connect all components into working pipeline.

Flow:

1. Dashboard starts SimSat simulation (via API)
2. Collector fetches satellite position + images
3. Spectral engine computes indices
4. Change detector compares with baseline
5. VLM generates investigation report
6. Report scored and added to downlink queue
7. Dashboard updates with all results

Testing checklist:

- [ ] SimSat API responds correctly
- [ ] Images are fetched and displayed
- [ ] Spectral indices compute correctly
- [ ] Change detection produces meaningful results
- [ ] VLM generates coherent investigation reports
- [ ] Dashboard updates in real-time
- [ ] Queue sorts by severity correctly
- [ ] Compression reduces report size

### 7.2 Performance Optimization

Goal: Ensure fast inference for demo.

Target: <5 seconds per image (capture + analysis + VLM inference).

On RTX 3050 Ti with bfloat16:

- Image fetch from API: 2-3 seconds (Sentinel STAC is slow)
- Spectral computation: <0.1 seconds
- VLM inference: 1-2 seconds (450M model is fast)
- Total: ~3-5 seconds per image

Optimizations:

- Pre-load model at dashboard start (`@st.cache_resource`)
- Batch spectral computation (numpy vectorization)
- Use bfloat16 for inference (faster than float32)
- Limit image size to 512x512 (model's native resolution)
- Cache baseline data to avoid recomputation

If too slow on RTX 3050:

- Use ONNX runtime with quantized model (Q4, ~459MB)
- Reduce image resolution to 256x256
- Skip VLM for low-severity detections (threshold)

### 7.3 Error Handling

Goal: App runs without crashing during demo.

Handle these cases:

- SimSat API timeout — increase request timeout to 60s
- No image available (ocean/polar regions) — skip, show message
- Model loading failure — show error, suggest restart
- Out of memory — reduce batch size, use ONNX fallback
- Cloud cover too high (>90%) — skip image, show warning

---

## Phase 8: Demo and Submission (Days 21-22) [COMMON]

### 8.1 Demo Recording

Goal: Record 3-5 minute demo video.

Script:

- **0:00 - 0:30** — Problem statement

"Satellites capture terabytes daily. They can download gigabytes. Current analysis takes days. What if the satellite could think?"

- **0:30 - 1:00** — Solution overview

"Sentinel Mind gives satellites the ability to decide what to image, investigate what they see, and prioritize what to send back."

- **1:00 - 2:30** — Live demo
  - Start SimSat simulation
  - Show satellite orbit on map
  - Satellite captures forest region image
  - Spectral analysis: NDVI drops 35%
  - VLM investigation report appears
  - Downlink queue updates
  - Show compression: 5MB raw to 50KB report

- **2:30 - 3:00** — Use cases

"This works for deforestation, maritime enforcement, wildfire detection, disaster response, and infrastructure monitoring."

- **3:00 - 3:30** — Technical highlights

"Fine-tuned LFM2.5-VL-450M with LoRA. Multi-spectral Sentinel-2 analysis. On-board inference. Bandwidth-aware prioritization."

- **3:30 - 4:00** — Call to action

"From orbit to enforcement in minutes, not days."

### 8.2 README Polish

Ensure README.md is complete and professional.

Include: architecture diagram, setup instructions, use cases, screenshots.

### 8.3 Submission Checklist

- [ ] Code runs without errors from fresh clone
- [ ] README has clear setup instructions
- [ ] Demo video is recorded and linked
- [ ] All dependencies listed in requirements.txt
- [ ] License is Apache 2.0
- [ ] No hardcoded secrets or API keys
- [ ] .gitignore excludes data, models, .venv, .worktrees
- [ ] Pushed to GitHub main branch

---

## Dependencies (requirements.txt)

```
# Core
torch>=2.6.0
torchvision>=0.21.0
transformers>=4.55.0
trl>=0.9.0
peft>=0.15.0
accelerate>=1.0.0
datasets>=3.0.0
huggingface_hub>=0.28.0

# Inference
onnxruntime>=1.20.0
pillow>=11.0.0
numpy>=2.0.0

# SimSat API
requests>=2.32.0
pystac-client>=0.8.0
odc-stac>=0.4.0

# Dashboard
streamlit>=1.42.0
folium>=0.19.0
streamlit-folium>=0.24.0
matplotlib>=3.10.0
pandas>=2.2.0

# Spectral Analysis
scipy>=1.15.0
```
