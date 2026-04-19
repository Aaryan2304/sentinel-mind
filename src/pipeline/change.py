"""Temporal change detection for satellite imagery.

Compares current spectral indices against a stored baseline to detect
significant changes (deforestation, burns, flooding, etc.).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_baseline(region_id: str, baseline_dir: str | Path = "data/baselines") -> dict[str, Any]:
    """Load previous image and indices for a region.

    Args:
        region_id: Unique region identifier.
        baseline_dir: Directory containing baseline data.

    Returns:
        Dict with keys: indices (dict of np.ndarray), timestamp, image_path.
        Returns empty dict if no baseline exists.
    """
    baseline_path = Path(baseline_dir) / region_id
    meta_path = baseline_path / "metadata.json"
    if not meta_path.exists():
        return {}
    with open(meta_path) as f:
        metadata = json.load(f)

    indices = {}
    for name in metadata.get("index_names", []):
        arr_path = baseline_path / f"{name}.npy"
        if arr_path.exists():
            indices[name] = np.load(arr_path)

    metadata["indices"] = indices
    return metadata


def compute_change_map(
    current_indices: dict[str, np.ndarray],
    baseline_indices: dict[str, np.ndarray],
) -> np.ndarray:
    """Compute per-pixel change magnitude across all indices.

    Args:
        current_indices: Dict of current spectral index maps.
        baseline_indices: Dict of baseline spectral index maps.

    Returns:
        2D array of change magnitudes (mean absolute difference across indices).
    """
    changes = []
    for name in current_indices:
        if name in baseline_indices:
            diff = np.abs(current_indices[name] - baseline_indices[name])
            changes.append(diff)

    if not changes:
        return np.zeros_like(next(iter(current_indices.values())))

    return np.mean(changes, axis=0)


def detect_change_regions(
    change_map: np.ndarray,
    threshold: float = 0.15,
    min_area_pixels: int = 100,
) -> list[dict[str, Any]]:
    """Segment change regions from the change map.

    Args:
        change_map: 2D change magnitude array.
        threshold: Minimum change magnitude to consider significant.
        min_area_pixels: Minimum region area in pixels.

    Returns:
        List of dicts: {bbox: (y1,x1,y2,x2), area, mean_change, dominant_index}.
    """
    raise NotImplementedError("Phase 3 implementation - requires scipy.ndimage")


def rank_changes(changes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort change regions by severity.

    Ranking: high change magnitude + large area = high priority.

    Args:
        changes: List of change region dicts from detect_change_regions.

    Returns:
        Sorted list (highest severity first), each dict gains 'severity' key.
    """
    for c in changes:
        c["severity_score"] = c["mean_change"] * c["area"]
    return sorted(changes, key=lambda c: c["severity_score"], reverse=True)


def save_baseline(
    region_id: str,
    indices: dict[str, np.ndarray],
    timestamp: str,
    baseline_dir: str | Path = "data/baselines",
) -> None:
    """Save current spectral indices as new baseline.

    Args:
        region_id: Unique region identifier.
        indices: Dict of spectral index maps.
        timestamp: ISO 8601 timestamp of capture.
        baseline_dir: Directory to store baselines.
    """
    baseline_path = Path(baseline_dir) / region_id
    baseline_path.mkdir(parents=True, exist_ok=True)

    index_names = []
    for name, arr in indices.items():
        np.save(baseline_path / f"{name}.npy", arr)
        index_names.append(name)

    metadata = {
        "region_id": region_id,
        "timestamp": timestamp,
        "index_names": index_names,
    }
    with open(baseline_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
