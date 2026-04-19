"""SimSat API data collector.

Fetches satellite position and Sentinel-2 imagery from the SimSat simulation API.
API base URL: http://localhost:9005
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Optional

import requests
from PIL import Image

BASE_URL = "http://localhost:9005"
REQUEST_TIMEOUT_SECONDS = 60


def _api_get(endpoint: str, params: Optional[dict[str, Any]] = None) -> requests.Response:
    """GET request to SimSat API. Raises on failure."""
    url = f"{BASE_URL}{endpoint}"
    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response


def get_satellite_position() -> dict[str, Any]:
    """Fetch current satellite position.

    Returns:
        dict with keys: lon, lat, alt, timestamp
    """
    response = _api_get("/data/current/position")
    data = response.json()
    lon_lat_alt = data["lon-lat-alt"]
    return {
        "lon": lon_lat_alt[0],
        "lat": lon_lat_alt[1],
        "alt": lon_lat_alt[2],
        "timestamp": data["timestamp"],
    }


def get_sentinel_image(
    lon: float,
    lat: float,
    timestamp: str,
    bands: list[str],
    size_km: float = 5.0,
) -> tuple[Optional[Image.Image], dict[str, Any]]:
    """Fetch Sentinel-2 image for a specific location and time.

    Args:
        lon: Longitude.
        lat: Latitude.
        timestamp: ISO 8601 timestamp.
        bands: Spectral bands to request, e.g. ["red", "green", "blue"].
        size_km: Image footprint size in km.

    Returns:
        Tuple of (PIL Image or None, metadata dict).
        metadata includes: image_available, cloud_cover, datetime, source.
    """
    params = {
        "lon": lon,
        "lat": lat,
        "timestamp": timestamp,
        "spectral_bands": bands,
        "size_km": size_km,
        "return_type": "png",
    }
    response = _api_get("/data/image/sentinel", params=params)

    metadata = json.loads(response.headers.get("sentinel_metadata", "{}"))
    if not metadata.get("image_available", False):
        return None, metadata

    image = Image.open(io.BytesIO(response.content)).convert("RGB")
    return image, metadata


def get_multispectral_image(
    lon: float,
    lat: float,
    timestamp: str,
    band_sets: list[list[str]],
    size_km: float = 5.0,
) -> dict[str, tuple[Optional[Image.Image], dict[str, Any]]]:
    """Fetch multiple band combinations for the same location.

    Args:
        lon: Longitude.
        lat: Latitude.
        timestamp: ISO 8601 timestamp.
        band_sets: List of band combinations, e.g. [["red","green","blue"], ["nir","red","green"]].
        size_km: Image footprint size in km.

    Returns:
        Dict mapping band combo name ("red_green_blue") to (Image, metadata).
    """
    results: dict[str, tuple[Optional[Image.Image], dict[str, Any]]] = {}
    for bands in band_sets:
        image, metadata = get_sentinel_image(lon, lat, timestamp, bands, size_km)
        key = "_".join(bands)
        results[key] = (image, metadata)
    return results


def collect_orbit_sequence(
    num_positions: int = 20,
    interval_seconds: float = 60.0,
    bands: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Collect satellite position + images along an orbit sequence.

    Args:
        num_positions: Number of positions to collect.
        interval_seconds: Time between collections.
        bands: Spectral bands. Defaults to ["red", "green", "blue"].

    Returns:
        List of dicts with keys: position, image, metadata, timestamp.
    """
    raise NotImplementedError("Requires SimSat simulation running (Phase 2)")


def save_collection(
    data: list[dict[str, Any]],
    output_dir: str | Path,
) -> None:
    """Save collected images and metadata to disk.

    Structure: output_dir/{timestamp}/image_{bands}.png + metadata.json
    """
    raise NotImplementedError("Phase 2 implementation")
