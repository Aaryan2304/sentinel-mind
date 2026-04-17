# AI in Space Hackathon - Brainstorming & Project Planning

**Competition:** Liquid AI x DPhi Space - AI in Space Hackathon
**Event:** https://luma.com/n9cw58h0
**Data Source:** SimSat API (DPhi-Space/SimSat)
**Date:** April 2026

---

## 1. Constraint Analysis

### Data Available (from SimSat API)
- **Sentinel-2 multispectral imagery** (13 bands, 10m resolution)
  - Bands: red, green, blue, nir, swir16, swir22, rededge1-3, coastal, aot, scl, wvp
  - Real temporal data from STAC API (Earth Search AWS)
  - Cloud cover metadata, platform info (sentinel-2a/b/c)
  - Size: configurable (default 5km x 5km)
- **Mapbox high-res RGB** (requires API key, static imagery)
- **Satellite position** (lon, lat, alt) with timestamps
- **Simulation control** (step size, replay speed, start time)

### Hardware Constraints (space-based compute)
- NVIDIA Orin 16GB (the satellite's actual compute)
- Limited downlink bandwidth (10MB download credit)
- Continuous data streams (satellite always moving)
- Must process on-board, not on-ground

### Model Options
- **LFM2-VL-450M** / **LFM2.5-VL-450M**: 450M param VLM
  - SigLIP2 vision encoder (86M)
  - Native 512x512 resolution handling
  - Available in GGUF, ONNX, MLX formats
  - Supports bounding box prediction, VQA, captioning
  - Official fine-tuning guide with VRSBench dataset

---

## 2. Track Selection Analysis

### Liquid AI Track
- **Prize:** $5,000 cash + satellite compute credits
- **Weights:** 10% data, 35% innovation, 35% technical, 20% demo
- **Requirement:** Fine-tuning LFM2-VL strongly encouraged
- **Advantage:** Official fine-tuning guide exists, concrete deliverable
- **Risk:** Many teams will follow the same guide

### General AI Track
- **Prize:** Satellite compute credits only
- **Weights:** 20% data, 25% innovation, 35% technical, 20% demo
- **Advantage:** Freedom to innovate architecturally
- **Risk:** Lower prize, less model-specific guidance

**Decision: Go with Liquid AI Track** (higher prize, official support, matches team skills)

---

## 3. Project Ideas (Ranked)

### IDEA 1: On-Board Spectral Anomaly Detector with VLM Summaries [RECOMMENDED]

**Problem:** Satellites generate massive data volumes but downlink is expensive
and limited. Operators need to know what matters NOW, not after downloading
terabytes.

**Solution:** A two-stage on-board pipeline:
1. **Stage 1 - Spectral Triage:** Lightweight CNN computes spectral indices
   (NDVI, NDWI, NDBI) from Sentinel-2 bands, detects anomalies (floods,
   fires, vegetation stress, urban expansion)
2. **Stage 2 - VLM Scene Understanding:** Fine-tuned LFM2.5-VL generates
   compressed natural language summaries of anomalous scenes
3. **Priority Queue:** Scores findings by severity, compresses to fit
   downlink budget

**Why it wins:**
- Uses Sentinel-2 multi-spectral bands (scoring criteria)
- Fine-tunes LFM2-VL on domain-specific satellite data
- Directly addresses space-based compute constraints
- Clear demo: satellite orbit -> image capture -> analysis -> prioritized alerts

**Technical Stack:**
- SimSat API for data acquisition
- Spectral index computation (numpy/opencv)
- LFM2.5-VL-450M fine-tuned on VRSBench + custom spectral anomaly Q&A
- ONNX export for Orin-compatible inference
- Priority queue with bandwidth-aware compression

---

### IDEA 2: Temporal Change Detection with Compressed Reporting

**Problem:** Detecting changes (deforestation, urban growth, disaster damage)
requires comparing images across time, but storing and transmitting full
image histories is impractical on-board.

**Solution:**
1. Maintain a rolling buffer of scene embeddings (not full images)
2. Compare current scene embedding against historical baseline
3. When change exceeds threshold, generate VLM description of what changed
4. Transmit only: change magnitude + text summary + cropped region

**Pros:** Very aligned with "continuous data streams" constraint
**Cons:** More complex, needs careful memory management

---

### IDEA 3: Multi-Spectral Scene Classification with VQA

**Problem:** Different spectral bands reveal different features. Operators
need to query the satellite: "Is there flooding at coordinates X,Y?"

**Solution:**
1. Fine-tune LFM2-VL on multi-band Sentinel-2 composites
2. Create VQA dataset mapping spectral patterns to questions/answers
3. Support queries like: "What is the vegetation health?", "Is there water
   overflow?", "What land use type is this?"

**Pros:** Directly uses VRSBench VQA format
**Cons:** Less compelling on the "space-based compute" narrative

---

### IDEA 4: Intelligent Image Quality Filter

**Problem:** Many satellite images are unusable (clouds, sensor artifacts,
ocean). Transmitting them wastes downlink bandwidth.

**Solution:**
1. Fine-tune LFM2-VL to classify images as: clear/partially-cloudy/
   cloudy/ocean/artifact
2. Assign quality scores
3. Only queue high-quality, land-containing images for downlink

**Pros:** Simple, practical, directly addresses bandwidth constraint
**Cons:** Maybe too simple for 35% innovation score

---

## 4. Recommended Project Plan (Idea 1)

### Phase 1: Data Pipeline (Day 1)
- Set up SimSat Docker environment
- Build data collection script: orbit -> fetch Sentinel-2 images
- Collect images across multiple spectral band combinations
- Generate spectral index maps (NDVI, NDWI, NDBI)
- Create anomaly detection ground truth

### Phase 2: VLM Fine-tuning (Day 1-2)
- Prepare training data in VRSBench-compatible format
- Use Liquid AI's leap-finetune framework (or manual LoRA if Modal unavailable)
- Fine-tune LFM2.5-VL-450M on:
  - Scene captioning for satellite imagery
  - Spectral anomaly Q&A pairs
  - Change detection descriptions
- Export to ONNX for deployment

### Phase 3: Integration & Demo (Day 2-3)
- Build the on-board processing pipeline
- Connect SimSat orbit -> fetch -> analyze -> summarize -> prioritize
- Create demo dashboard showing:
  - Satellite orbit on map
  - Live image feed
  - Spectral analysis overlay
  - VLM-generated summaries
  - Priority-ranked downlink queue
- Write README and record demo video

### Scoring Alignment
- [x] Data (10%): Uses Sentinel-2 multispectral bands from DPhi API
- [x] Innovation (35%): On-board anomaly detection + VLM summaries
- [x] Technical (35%): Fine-tuned VLM, ONNX export, working pipeline
- [x] Demo (20%): End-to-end satellite simulation walkthrough

---

## 5. Technical Decisions

### Fine-tuning Approach
- **Option A:** Liquid AI's leap-finetune + Modal (official guide)
- **Option B:** Manual LoRA with PEFT/transformers (more control)
- **Decision:** Start with Option A, fall back to B if Modal issues

### Inference Optimization
- Partner handles: ONNX export, quantization, TensorRT if needed
- Target: Orin 16GB compatibility
- LFM2.5-VL-450M is already small (450M params) - should run on Orin

### Dataset Creation
- VRSBench has 123K VQA pairs, 52K grounding refs, 29K captions
- Augment with DPhi API spectral anomaly examples
- Generate synthetic Q&A pairs from spectral index computations

---

## 6. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| SimSat API is slow | Cache images, use batch collection |
| Fine-tuning takes too long | Use LoRA (few hours), not full fine-tune |
| Modal free credit insufficient | Fall back to local Colab/WSL training |
| Model doesn't improve over base | Focus on spectral-specific tasks |
| Demo breaks during judging | Pre-record backup, test end-to-end |

---

## 7. Competitive Analysis

**What other teams will likely do:**
- Follow the Liquid AI satellite VLM guide exactly
- Basic VQA on satellite images
- Simple scene captioning

**Our differentiator:**
- Multi-spectral analysis (not just RGB)
- Anomaly detection + prioritized downlink (space compute narrative)
- Clean end-to-end pipeline from orbit to alert
- Both partners' skills utilized (CV + model optimization)
