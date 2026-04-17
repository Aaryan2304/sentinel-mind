# Sentinel Mind

**Autonomous Satellite Intelligence with Investigative VLM Reasoning**

Sentinel Mind gives satellites the ability to decide what to image, investigate what they see, and prioritize what to send back. Built on Liquid AI's LFM2.5-VL-450M vision-language model, fine-tuned on multi-spectral Sentinel-2 imagery for on-board scene understanding.

## The Problem

Satellites capture terabytes of imagery daily but can only download gigabytes through limited downlink bandwidth. Current workflows require downloading raw images to ground stations for analysis — a process that takes hours to days. By the time intelligence reaches decision-makers, the opportunity for action has passed.

## The Solution

An on-board intelligence pipeline that processes satellite imagery in orbit:

1. **Decide** — Autonomously selects imaging targets based on revisit priority, cloud cover, and region importance
2. **Investigate** — Analyzes multi-spectral imagery using spectral indices (NDVI, NBR, NDWI) and a fine-tuned VLM
3. **Prioritize** — Scores findings by severity and compresses investigations for bandwidth-efficient downlink

The result: actionable intelligence delivered in minutes, not days.

## Use Cases

| Domain | Detection | Investigation | Downlink Priority |
|--------|-----------|---------------|-------------------|
| Deforestation | NDVI change | Logging type, severity | Enforcement coordinates |
| Maritime | Vessel wake pattern | Vessel type, intent | Patrol interdiction |
| Wildfire | Thermal anomaly | Fire spread prediction | Evacuation zones |
| Disaster | Structural change | Infrastructure status | Rescue priorities |
| Infrastructure | Subsidence/cracks | Risk assessment | Inspection scheduling |

## Architecture

```
SimSat API → Sentinel-2 Multi-spectral → Spectral Index Computation
→ Change Detection → VLM Investigation → Priority Scoring
→ Downlink Queue → Streamlit Dashboard
```

## Model

Fine-tuned [LFM2.5-VL-450M](https://huggingface.co/LiquidAI/LFM2.5-VL-450M) — a 450M parameter vision-language model from Liquid AI, optimized for edge deployment. Fine-tuned on VRSBench satellite imagery dataset with custom spectral analysis prompts.

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.11+
- GPU recommended (4GB+ VRAM for inference, 16GB+ for fine-tuning)

### Setup

```bash
# Clone the repository
git clone https://github.com/Aaryan2304/sentinel-mind.git
cd sentinel-mind

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Start SimSat simulation
cd simsat-reference
docker compose up
```

### Run Dashboard

```bash
streamlit run src/dashboard/app.py
```

The dashboard will be available at `http://localhost:8501`.
The SimSat API runs at `http://localhost:9005`.
The SimSat dashboard runs at `http://localhost:8000`.

## Project Structure

```
sentinel-mind/
├── src/
│   ├── pipeline/           # Data collection and spectral analysis
│   │   ├── collector.py    # SimSat API data collector
│   │   ├── spectral.py     # Spectral index computation
│   │   └── change.py       # Temporal change detection
│   ├── models/             # VLM fine-tuning and inference
│   │   ├── finetune.py     # LoRA fine-tuning with TRL+PEFT
│   │   ├── inference.py    # VLM inference pipeline
│   │   └── prompts.py      # Investigation prompt templates
│   └── dashboard/          # Streamlit application
│       └── app.py          # Main dashboard
├── configs/                # Training and inference configs
├── docs/                   # Project documentation
├── requirements.txt        # Python dependencies
├── LICENSE                 # Apache 2.0
└── README.md              # This file
```

## Technical Details

### Spectral Indices
- **NDVI** = (B8 - B4) / (B8 + B4) — Vegetation greenness
- **NDWI** = (B3 - B8) / (B3 + B8) — Water/moisture content
- **NBR** = (B8 - B12) / (B8 + B12) — Burn severity
- **NDRE** = (B8A - B5) / (B8A + B5) — Chlorophyll/stress detection
- **NDMI** = (B8 - B11) / (B8 + B11) — Vegetation water content

### Fine-tuning
- Base model: `LiquidAI/LFM2.5-VL-450M`
- Method: LoRA (r=16, alpha=32) via TRL SFTTrainer + PEFT
- Dataset: VRSBench (29,614 images, 123,221 VQA pairs) + custom spectral analysis prompts
- Training: RTX 4060 Ti or Google Colab T4 (free tier)

### Inference
- Model: Fine-tuned LFM2.5-VL-450M
- Framework: HuggingFace Transformers
- Optimization: ONNX export available for deployment

## Industry Context

This project builds on cutting-edge developments in on-board satellite AI:

- **NASA Dynamic Targeting (July 2025)** — Satellite autonomously decides what to photograph in under 90 seconds
- **GalaxEye Drishti (February 2026)** — India's AI-powered satellite with NVIDIA Jetson Orin for on-board processing
- **XPRIZE Wildfire** — Active competition pushing for sub-30-minute satellite-based fire detection
- **Operation Nightwatch (December 2025)** — Marshall Islands compressed dark vessel enforcement from days to 4 hours

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Liquid AI](https://liquid.ai/) for LFM2.5-VL-450M
- [DPhi Space](https://dphi.space/) for SimSat and the hackathon
- [VRSBench](https://huggingface.co/datasets/xiang709/VRSBench) for the satellite imagery dataset
- [Copernicus Sentinel-2](https://sentinel.esa.int/web/sentinel/missions/sentinel-2) for multi-spectral Earth observation data
