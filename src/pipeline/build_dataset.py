"""Build custom spectral analysis dataset for VLM fine-tuning.

Takes collected SimSat images, computes spectral indices, and generates
investigation Q&A pairs in TRL SFTTrainer conversation format.

Output format (JSONL, one sample per line):
{
    "messages": [
        {"role": "system", "content": "You are a satellite imagery analysis AI..."},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "Analyze this satellite image..."}
        ]},
        {"role": "assistant", "content": "The image shows..."}
    ],
    "images": ["path/to/image.png"]
}
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.pipeline.spectral import (
    compute_all_indices,
    index_statistics,
    dn_to_reflectance,
)


SYSTEM_PROMPT = (
    "You are a satellite-based Earth observation AI specialized in multi-spectral "
    "image analysis. You analyze Sentinel-2 imagery using spectral indices (NDVI, "
    "NDWI, NBR) to detect environmental changes, anomalies, and land use patterns. "
    "You provide structured investigation reports with severity assessments and "
    "recommended actions."
)

INVESTIGATION_PROMPTS = [
    "Analyze this satellite image for vegetation health and land cover changes. "
    "What spectral anomalies are visible? Is this natural or human activity?",

    "Examine this multi-spectral satellite image. Detect any signs of deforestation, "
    "urban expansion, or environmental disturbance. What is the severity?",

    "This satellite image was captured over a region of interest. Analyze the "
    "spectral data and describe what changes or anomalies you observe. "
    "What action should be taken?",

    "Investigate this satellite imagery for signs of environmental change. "
    "Compare the observed patterns with typical vegetation and land use signatures. "
    "Classify the severity (low/medium/high/critical).",

    "You are analyzing satellite imagery for an on-board intelligence system. "
    "Examine the spectral signatures and determine if there are any anomalies "
    "that require priority downlink for ground investigation.",
]


def generate_investigation_response(
    indices: dict[str, np.ndarray],
    metadata: dict[str, Any],
    band_combo: str,
) -> str:
    """Generate a synthetic investigation response based on spectral data.

    Args:
        indices: Dict of spectral index maps.
        metadata: Image metadata (cloud_cover, source, position, etc.).
        band_combo: Band combination name (e.g., "nir_red_green").

    Returns:
        Investigation report string.
    """
    stats = {}
    for name, arr in indices.items():
        stats[name] = index_statistics(arr)

    ndvi_mean = stats.get("ndvi", {}).get("mean", 0)
    ndwi_mean = stats.get("ndwi", {}).get("mean", 0)
    nbr_mean = stats.get("nbr", {}).get("mean", 0)
    ndmi_mean = stats.get("ndmi", {}).get("mean", 0)

    cloud_cover = metadata.get("cloud_cover", 0)
    position = metadata.get("position", {})
    lon = position.get("lon", 0)
    lat = position.get("lat", 0)

    # Determine scene type based on indices
    if ndvi_mean > 0.5:
        scene = "dense vegetation"
        health = "healthy"
    elif ndvi_mean > 0.3:
        scene = "moderate vegetation"
        health = "moderately healthy"
    elif ndvi_mean > 0.1:
        scene = "sparse vegetation"
        health = "stressed"
    elif ndwi_mean > 0.2:
        scene = "water body or wetland"
        health = "n/a"
    else:
        scene = "bare soil or urban area"
        health = "n/a"

    # Determine severity
    if ndvi_mean < 0.1 and nbr_mean < 0.1:
        severity = "HIGH"
        action = "Recommend priority downlink for ground investigation. Potential clear-cutting or fire damage detected."
    elif ndvi_mean < 0.2:
        severity = "MEDIUM"
        action = "Schedule follow-up observation in 5 days. Monitor for further degradation."
    elif cloud_cover > 50:
        severity = "LOW"
        action = "High cloud cover limits analysis. Re-image when clear."
    else:
        severity = "LOW"
        action = "No significant anomalies detected. Continue routine monitoring."

    # Build response
    response_parts = [
        f"Scene Analysis: The image shows {scene} at coordinates ({lon:.2f}, {lat:.2f}). "
        f"Cloud cover: {cloud_cover:.0f}%.",
        "",
        f"Spectral Indices:",
        f"- NDVI (vegetation): {ndvi_mean:.2f} — {health}",
        f"- NDWI (water/moisture): {ndwi_mean:.2f}",
        f"- NBR (burn): {nbr_mean:.2f}",
        f"- NDMI (vegetation moisture): {ndmi_mean:.2f}",
        "",
        f"Assessment: {'Vegetation appears ' + health + '.' if health != 'n/a' else 'Non-vegetated area.'} "
        f"{'No significant anomalies detected.' if severity == 'LOW' else 'Anomalous spectral signatures detected.'}",
        "",
        f"Severity: {severity}.",
        f"Action: {action}",
    ]

    return "\n".join(response_parts)


def build_sample(
    image_path: Path,
    band_combo: str,
    indices: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build a single training sample in TRL conversation format.

    Args:
        image_path: Path to the satellite image.
        band_combo: Band combination name.
        indices: Computed spectral indices.
        metadata: Image metadata.

    Returns:
        Dict in TRL SFTTrainer format.
    """
    prompt = random.choice(INVESTIGATION_PROMPTS)
    response = generate_investigation_response(indices, metadata, band_combo)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]},
            {"role": "assistant", "content": response},
        ],
        "images": [str(image_path)],
    }


def build_dataset_from_collection(
    raw_dir: str | Path = "data/raw",
    output_path: str | Path = "data/custom/spectral_qa.jsonl",
    band_combo: str = "nir_red_green",
) -> Path:
    """Build training dataset from collected SimSat images.

    Args:
        raw_dir: Directory containing collected images (from save_collection).
        output_path: Output JSONL file path.
        band_combo: Which band combination to use for training images.
                   Default: "nir_red_green" (false color IR — best for vegetation analysis).

    Returns:
        Path to the output JSONL file.
    """
    raw_path = Path(raw_dir)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        raise FileNotFoundError(f"Collection directory not found: {raw_path}")

    entries = sorted(raw_path.iterdir())
    entries = [d for d in entries if d.is_dir()]

    print(f"Building dataset from {len(entries)} collected positions...")
    print(f"Band combo: {band_combo}")
    print(f"Output: {output}")

    samples_written = 0
    skipped = 0

    with open(output, "w") as f:
        for entry_dir in entries:
            meta_path = entry_dir / "metadata.json"
            img_filename = f"image_{band_combo}.png"
            img_path = entry_dir / img_filename

            if not meta_path.exists() or not img_path.exists():
                skipped += 1
                continue

            with open(meta_path) as mf:
                full_meta = json.load(mf)

            # Load image and compute spectral indices
            image = Image.open(img_path).convert("RGB")
            img_array = np.array(image)

            # For RGB images, we can only compute NDVI-like indices
            # if we have NIR data. For the nir_red_green combo:
            # R channel = NIR, G channel = Red, B channel = Green
            if band_combo == "nir_red_green":
                bands = {
                    "nir": img_array[:, :, 0].astype(np.float32) / 255.0,
                    "red": img_array[:, :, 1].astype(np.float32) / 255.0,
                    "green": img_array[:, :, 2].astype(np.float32) / 255.0,
                }
            elif band_combo == "red_green_blue":
                bands = {
                    "red": img_array[:, :, 0].astype(np.float32) / 255.0,
                    "green": img_array[:, :, 1].astype(np.float32) / 255.0,
                    "blue": img_array[:, :, 2].astype(np.float32) / 255.0,
                }
            else:
                # Generic: R=nir, G=red, G=green (best guess)
                bands = {
                    "nir": img_array[:, :, 0].astype(np.float32) / 255.0,
                    "red": img_array[:, :, 1].astype(np.float32) / 255.0,
                    "green": img_array[:, :, 2].astype(np.float32) / 255.0,
                }

            indices = compute_all_indices(bands)

            # Build metadata for sample generation
            band_meta = full_meta.get("metadata", {}).get(band_combo, {})
            sample_meta = {
                "cloud_cover": band_meta.get("cloud_cover", 0),
                "position": full_meta.get("position", {}),
                "source": band_meta.get("source", "unknown"),
            }

            sample = build_sample(img_path, band_combo, indices, sample_meta)
            f.write(json.dumps(sample) + "\n")
            samples_written += 1

    print(f"Dataset built: {samples_written} samples, {skipped} skipped")
    print(f"Saved to: {output}")
    return output


if __name__ == "__main__":
    import sys
    raw_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    build_dataset_from_collection(raw_dir)
