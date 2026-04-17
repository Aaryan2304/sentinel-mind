# Enhanced Brainstorming - AI in Space Hackathon

## Hardware Analysis

### LFM2.5-VL-450M Memory Requirements
- **Model weights (FP16):** ~0.9 GB
- **Inference (with KV cache):** ~1.5-2.0 GB
- **LoRA fine-tuning:** ~4-6 GB (model + gradients + optimizer)
- **Full fine-tuning:** ~12-16 GB

### Your Hardware Options
| Device | VRAM | Inference | LoRA Fine-tune | Full Fine-tune |
|--------|------|-----------|----------------|----------------|
| RTX 3050 Ti (yours) | 4 GB | YES (tight) | NO | NO |
| RTX 4050 (partner) | 16 GB | YES | YES | YES |
| Colab T4 (free) | 16 GB | YES | YES | NO (full) |

**Conclusion:** Your partner's RTX 4050 or Colab T4 can handle LoRA fine-tuning.
Your RTX 3050 Ti can run inference for testing. The model is designed for
edge deployment, so it's very efficient.

### DPhi Satellite Hardware
- NVIDIA Orin 16GB (the actual satellite compute)
- This is what your solution will eventually run on
- LFM2.5-VL-450M at 450M params fits comfortably on Orin
- ONNX/GGUF quantization can further reduce memory

---

## What Makes Hackathon Winners Stand Out

After analyzing winning patterns:

1. **Emotional resonance** - The problem must MATTER to people
2. **Clear narrative arc** - Problem -> Solution -> Impact in 3 minutes
3. **Live demo that works** - Not slides, actual running code
4. **Technical depth** - Fine-tuning methodology, not just API calls
5. **"Why space?" justification** - Clear reason it must run on-board

---

## Refined Project Ideas (Ranked by Winning Potential)

### IDEA 1: "The On-Board Forest Guardian"
**Illegal Deforestation Detection with Investigative VLM Reasoning**

Why this is powerful:
- **Emotional hook:** Illegal loggers destroy ecosystems, threaten indigenous
  communities, accelerate climate change. Everyone understands this.
- **Technical depth:** Multi-temporal Sentinel-2 analysis + spectral indices
  (NDVI, NBR) + fine-tuned VLM with chain-of-thought reasoning
- **Space compute justification:** By the time images are downloaded and
  analyzed on-ground, loggers have moved. On-board detection enables
  real-time alerts to enforcement agencies.
- **Demo flow:** Satellite orbit -> Forest region capture -> Change detection
  vs baseline -> VLM investigation: "New clearing detected at coordinates X,Y.
  Road construction pattern suggests commercial logging. NDVI dropped 40% in
  2 weeks. Recommend priority downlink to enforcement."
- **Sentinel-2 advantage:** NIR/SWIR bands detect vegetation stress invisible
  in RGB. 5-day revisit catches changes early.

The VLM prompt:
```
You are a satellite-based forest monitoring AI. Analyze this bi-temporal
Sentinel-2 imagery (NIR-Red-Green composite).
Current image: [spectral data]
Baseline (2 weeks ago): [spectral data]
Region: Amazon Basin, coordinates [-3.1, -60.0]

Investigate:
1. What changes are visible?
2. Does this pattern match natural forest dynamics or human activity?
3. What is the severity? (low/medium/high/critical)
4. What action should enforcement take?
```

**Demo story:**
"The satellite passes over the Amazon at 2am local time. On-board AI detects
a new 2-hectare clearing that wasn't there 5 days ago. It analyzes the spectral
signature - the NIR reflectance dropped 40%, consistent with clear-cutting, not
natural causes. It generates an investigation report and flags it CRITICAL for
immediate downlink. By morning, enforcement agents have coordinates and analysis
before the loggers can return."

---

### IDEA 2: "Maritime Shadow Hunter"
**Dark Vessel Detection for Illegal Fishing Enforcement**

Why this is powerful:
- **Global impact:** IUU fishing costs $23B annually, devastates marine
  ecosystems, threatens food security for coastal communities
- **Technical depth:** Multi-temporal analysis + wake pattern detection +
  investigative VLM reasoning + AIS correlation
- **Space justification:** Vessels disable AIS transponders to avoid detection.
  Only satellite surveillance can catch them. On-board analysis means patrol
  vessels get alerts in near-real-time, not days later.
- **Demo flow:** Satellite detects vessel wake pattern -> No AIS signal ->
  VLM analyzes: "Vessel detected at coordinates X,Y moving at estimated 12 knots.
  Wake pattern consistent with trawling vessel. No AIS transmission. Located
  3km inside Marine Protected Area. HIGH probability of IUU fishing."

---

### IDEA 3: "Disaster Response Autopilot"
**Autonomous Disaster Assessment with Prioritized Downlink**

Why this is powerful:
- **Emotional hook:** After earthquakes, floods, wildfires - every hour matters
- **Space justification:** Ground communication infrastructure is often destroyed.
  Satellite must be the first intelligence source.
- **Demo flow:** Earthquake detected -> Satellite autonomously repositions ->
  Captures affected area -> VLM generates damage assessment ->
  Prioritizes: "Bridge collapsed at X,Y - critical for rescue access.
  Hospital at X,Y appears intact. Residential area X,Y - 60% structures damaged.
  Recommend priority 1 downlink for bridge coordinates."
- **Technical depth:** Change detection + structural damage classification +
  VLM-generated action priorities

---

### IDEA 4: "Crop Guardian" (your friend's idea, enhanced)
**Conversational Crop Insurance Verification**

Why this works:
- **Market:** $3B+ agricultural insurance market in emerging economies
- **Technical:** VLM mediates between satellite data and farmer claims
- **Demo:** "Farmer claims drought damage. Satellite shows NDWI indicates
  adequate soil moisture. But farmer's photo shows wilting. Reconciling:
  Likely cause is pest infestation, not drought. SWIR bands show unusual
  thermal signature consistent with locust activity."

---

### IDEA 5: "Energy Transition Scout"
**Renewable Energy Site Assessment Agent**

Why this works:
- **Timely:** Global push for solar/wind, governments need site assessments
- **Technical:** VLM + tool use (cross-reference with regulatory data)
- **Demo:** "50MW solar farm request. Analyzing 500km² via satellite.
  Parcel A: low ecological value, flat terrain, near transmission line - SUITABLE.
  Parcel B: seasonal wetland detected via NDWI - AVOID.
  Parcel C: existing agriculture, high conflict potential - NOT RECOMMENDED."

---

## Recommendation: IDEA 1 - Forest Guardian

### Why This Wins

1. **Innovation (35%):** Investigative VLM reasoning on multi-spectral data
   is novel. Not just detection, but autonomous investigation with
   chain-of-thought.

2. **Technical (35%):**
   - Fine-tuned LFM2.5-VL-450M on forest imagery + investigation prompts
   - Multi-spectral analysis (NIR, SWIR, Red-Edge)
   - Temporal change detection
   - ONNX export for Orin deployment
   - Priority queue with bandwidth-aware compression

3. **Data (10%):** Uses Sentinel-2 multi-spectral bands from DPhi API
   (not just RGB). Uses temporal data. Uses cloud cover metadata.

4. **Demo (20%):**
   - Satellite orbit visualization
   - Live image capture from SimSat
   - Spectral analysis overlay
   - VLM investigation report
   - Priority downlink queue
   - Clear narrative: "From orbit to enforcement in minutes, not days"

### Why NOT the Others

- **Maritime (Idea 2):** Harder to demo - ocean imagery is less visually
  compelling, vessel detection at 10m resolution is challenging
- **Disaster (Idea 3):** Good but more common - many teams will do
  disaster response
- **Crop (Idea 4):** Good but less emotionally compelling
- **Energy (Idea 5):** Good but less "space-native"

### What Makes Forest Guardian "Mind-Blowing"

The narrative arc:
1. "Illegal loggers work at night. They move fast. But they can't hide
   from satellites."
2. "The satellite doesn't just take pictures. It INVESTIGATES."
3. "On-board AI detects change, analyzes spectral signatures, reasons
   about cause, and generates an enforcement report."
4. "By the time the image reaches the ground, the analysis is already done."
5. "From orbit to enforcement in minutes, not days."

This is a story that resonates with judges, media, and policymakers.

---

## Technical Implementation Plan

### Data Collection (from SimSat API)
1. Define forest regions (Amazon, Congo, Southeast Asia)
2. Collect Sentinel-2 imagery in multiple band combinations:
   - RGB (for visualization)
   - NIR-Red-Green (for vegetation analysis)
   - SWIR-NIR-Green (for burn/clearing detection)
   - Red-Edge bands (for vegetation health)
3. Compute spectral indices: NDVI, NBR, NDWI
4. Generate synthetic change scenarios for training

### Fine-tuning
1. Base: LFM2.5-VL-450M
2. Method: LoRA (r=16, alpha=32) on vision encoder + language head
3. Dataset: Custom investigation prompts + VRSBench forest imagery
4. Training: Colab T4 or partner's RTX 4050
5. Export: ONNX for Orin compatibility

### Pipeline
```
SimSat API -> Sentinel-2 Multi-spectral -> Spectral Index Computation
-> Change Detection -> VLM Investigation -> Priority Scoring
-> Downlink Queue -> Dashboard
```

### Demo Script
1. Start SimSat simulation
2. Show satellite orbit on Cesium globe
3. Satellite passes over forest region
4. Capture Sentinel-2 imagery (multiple bands)
5. Compute NDVI/NBR, compare with baseline
6. Show change detection overlay
7. VLM generates investigation report
8. Show priority queue with compression stats
9. "This analysis happened ON the satellite, before any data
   was sent to ground."
