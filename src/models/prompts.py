"""Investigation prompt templates for VLM inference.

Each template is designed for a specific use case. The prompts include
spectral analysis context so the VLM can reason about multi-spectral data.
"""

from __future__ import annotations

from typing import Any


FOREST_INVESTIGATION_TEMPLATE = """You are a satellite-based forest monitoring AI. Analyze this multi-spectral \
satellite imagery (NIR-Red-Green composite).

Current image captured: {timestamp}
Region: {region_name}, coordinates [{lat}, {lon}]
Cloud cover: {cloud_cover}%
Image size: {size_km}km x {size_km}km

Spectral Analysis:
- NDVI (vegetation): {ndvi_mean:.2f} (baseline: {ndvi_baseline:.2f})
- NDWI (water): {ndwi_mean:.2f}
- NBR (burn): {nbr_mean:.2f}
- NDVI change from baseline: {ndvi_change_pct:.1f}%

Investigate:
1. What changes are visible in this image?
2. Does this pattern match natural forest dynamics or human activity?
3. What is the severity? (low/medium/high/critical)
4. What action should enforcement or response teams take?"""

MARITIME_INVESTIGATION_TEMPLATE = """You are a maritime surveillance AI. Analyze this satellite imagery \
for vessel activity.

Current image captured: {timestamp}
Region: {region_name}, coordinates [{lat}, {lon}]
Expected vessel traffic: {traffic_level}

Investigate:
1. Are any vessels visible in this image?
2. Do you see wake patterns consistent with active fishing or transit?
3. Is there evidence of oil slicks or environmental disturbance?
4. What is the severity? (low/medium/high/critical)"""

WILDFIRE_INVESTIGATION_TEMPLATE = """You are a wildfire detection AI. Analyze this satellite imagery \
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
4. What is the severity? (low/medium/high/critical)"""

TEMPLATES = {
    "forest": FOREST_INVESTIGATION_TEMPLATE,
    "maritime": MARITIME_INVESTIGATION_TEMPLATE,
    "wildfire": WILDFIRE_INVESTIGATION_TEMPLATE,
}


def build_investigation_prompt(
    spectral_data: dict[str, Any],
    template_name: str = "forest",
) -> str:
    """Build investigation prompt from spectral data.

    Args:
        spectral_data: Dict with keys matching template placeholders.
            Required: timestamp, region_name, lat, lon, cloud_cover, size_km,
            ndvi_mean, ndvi_baseline, ndwi_mean, nbr_mean, ndvi_change_pct.
        template_name: Which template to use ("forest", "maritime", "wildfire").

    Returns:
        Formatted prompt string.
    """
    template = TEMPLATES.get(template_name, FOREST_INVESTIGATION_TEMPLATE)

    defaults = {
        "timestamp": "unknown",
        "region_name": "unknown",
        "lat": 0.0,
        "lon": 0.0,
        "cloud_cover": 0.0,
        "size_km": 5.0,
        "ndvi_mean": 0.0,
        "ndvi_baseline": 0.0,
        "ndwi_mean": 0.0,
        "nbr_mean": 0.0,
        "ndvi_change_pct": 0.0,
        "traffic_level": "unknown",
        "weather_conditions": "unknown",
        "thermal_status": "none detected",
    }

    filled = {**defaults, **spectral_data}
    return template.format(**filled)
