"""SimSat API data collector.

Fetches satellite position and Sentinel-2 imagery from the SimSat simulation API.
API base URL: http://localhost:9005
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Any, Optional

import requests
from PIL import Image

BASE_URL = "http://localhost:9005"
REQUEST_TIMEOUT_SECONDS = 120
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5


def _api_get(
    endpoint: str,
    params: Optional[dict[str, Any]] = None,
    retries: int = MAX_RETRIES,
) -> requests.Response:
    """GET request to SimSat API with retry logic for transient failures."""
    url = f"{BASE_URL}{endpoint}"
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response
        except (requests.ReadTimeout, requests.ConnectionError) as e:
            last_error = e
            if attempt < retries:
                print(f"  [retry {attempt}/{retries}] {type(e).__name__}: {e}")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                raise
        except requests.HTTPError:
            raise

    raise last_error  # type: ignore[misc]


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


# Band combinations for multi-spectral collection
DEFAULT_BAND_SETS: list[list[str]] = [
    ["red", "green", "blue"],           # RGB — visualization
    ["nir", "red", "green"],            # False Color IR — vegetation
    ["swir22", "nir", "green"],         # SWIR — burn/clearing detection
    ["swir16", "nir", "red"],           # Agriculture — crop health
    ["rededge1", "rededge2", "rededge3"],  # Red-Edge — vegetation stress
]


def collect_orbit_sequence(
    num_positions: int = 20,
    poll_interval_seconds: float = 5.0,
    band_sets: list[list[str]] | None = None,
    size_km: float = 5.0,
    max_cloud_cover: float | None = None,
) -> list[dict[str, Any]]:
    """Collect satellite position + multi-spectral images along an orbit sequence.

    Polls the SimSat API at regular intervals. For each position, fetches images
    in multiple band combinations. Skips positions where image_available=False.

    Args:
        num_positions: Number of valid image positions to collect.
        poll_interval_seconds: Seconds between position polls (real time).
        band_sets: Band combinations to fetch. Defaults to DEFAULT_BAND_SETS.
        size_km: Image footprint size in km.
        max_cloud_cover: Skip images with cloud cover above this (0-100). None = no filter.

    Returns:
        List of dicts, each with keys:
            position: {lon, lat, alt, timestamp}
            images: {band_combo_name: PIL.Image}
            metadata: {band_combo_name: metadata_dict}
    """
    if band_sets is None:
        band_sets = DEFAULT_BAND_SETS

    import time

    collected: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = num_positions * 5  # Safety limit to avoid infinite loops

    print(f"Collecting {num_positions} positions with {len(band_sets)} band sets each...")
    print(f"Poll interval: {poll_interval_seconds}s, size: {size_km}km")

    while len(collected) < num_positions and attempts < max_attempts:
        attempts += 1

        try:
            position = get_satellite_position()
        except (requests.RequestException, KeyError) as e:
            print(f"[{attempts}] Failed to get position: {e}")
            time.sleep(poll_interval_seconds)
            continue

        lon, lat, timestamp = position["lon"], position["lat"], position["timestamp"]

        # Fetch first band set to check availability and cloud cover
        first_bands = band_sets[0]
        first_image, first_metadata = get_sentinel_image(
            lon, lat, timestamp, first_bands, size_km
        )

        if first_image is None:
            print(f"[{attempts}] No image at ({lon:.2f}, {lat:.2f}) — skipping")
            time.sleep(poll_interval_seconds)
            continue

        cloud_cover = first_metadata.get("cloud_cover", 0)
        if max_cloud_cover is not None and cloud_cover > max_cloud_cover:
            print(f"[{len(collected)+1}/{num_positions}] Cloud cover {cloud_cover:.0f}% > {max_cloud_cover}% — skipping")
            time.sleep(poll_interval_seconds)
            continue

        # First image is valid, fetch remaining band sets
        images: dict[str, Any] = {"_".join(first_bands): first_image}
        metadata: dict[str, Any] = {"_".join(first_bands): first_metadata}

        for bands in band_sets[1:]:
            key = "_".join(bands)
            try:
                img, meta = get_sentinel_image(lon, lat, timestamp, bands, size_km)
                images[key] = img
                metadata[key] = meta
            except (requests.RequestException, Exception) as e:
                print(f"  Warning: failed to fetch {key}: {e}")
                images[key] = None
                metadata[key] = {"error": str(e)}

        collected.append({
            "position": position,
            "images": images,
            "metadata": metadata,
        })

        print(
            f"[{len(collected)}/{num_positions}] "
            f"({lon:.2f}, {lat:.2f}) cloud={cloud_cover:.0f}% "
            f"source={first_metadata.get('source', '?')}"
        )

        time.sleep(poll_interval_seconds)

    print(f"Collection complete: {len(collected)} positions collected in {attempts} attempts")
    return collected


def save_collection(
    data: list[dict[str, Any]],
    output_dir: str | Path = "data/raw",
) -> Path:
    """Save collected images and metadata to disk.

    Creates directory structure:
        output_dir/
            2026-04-19T05-36-11Z_lon-122.27_lat61.09/
                image_red_green_blue.png
                image_nir_red_green.png
                ...
                metadata.json

    Args:
        data: List of collected position+image dicts from collect_orbit_sequence().
        output_dir: Root output directory.

    Returns:
        Path to the output directory.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    for entry in data:
        position = entry["position"]
        images = entry["images"]
        metadata = entry["metadata"]

        # Sanitize timestamp for directory name (replace : with -)
        ts_safe = position["timestamp"].replace(":", "-")
        lon_str = f"lon{position['lon']:.2f}"
        lat_str = f"lat{position['lat']:.2f}"
        dir_name = f"{ts_safe}_{lon_str}_{lat_str}"

        entry_dir = output_path / dir_name
        entry_dir.mkdir(exist_ok=True)

        # Save each band combination image
        saved_images: dict[str, str] = {}
        for band_key, image in images.items():
            if image is not None:
                img_filename = f"image_{band_key}.png"
                image.save(entry_dir / img_filename)
                saved_images[band_key] = img_filename

        # Save metadata
        meta_out = {
            "position": position,
            "images_saved": saved_images,
            "metadata": {
                band_key: {
                    k: v for k, v in meta.items()
                    if k != "image"  # Exclude raw image data if present
                }
                for band_key, meta in metadata.items()
            },
        }
        with open(entry_dir / "metadata.json", "w") as f:
            json.dump(meta_out, f, indent=2, default=str)

        saved_count += 1

    print(f"Saved {saved_count} entries to {output_path}")
    return output_path
