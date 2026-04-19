"""Spectral index computation for Sentinel-2 imagery.

All indices operate on reflectance values (0-1), not raw DN.
Sentinel-2 scale factor: 10000 (reflectance = DN / 10000).

Band mapping (Sentinel-2):
    B2=Blue(490nm), B3=Green(560nm), B4=Red(665nm),
    B5=RedEdge1(705nm), B6=RedEdge2(740nm), B7=RedEdge3(783nm),
    B8=NIR(842nm), B8A=RedEdge4(865nm),
    B11=SWIR1(1610nm), B12=SWIR2(2190nm)
"""

from __future__ import annotations

import numpy as np

SENTINEL2_SCALE_FACTOR = 10000.0


def dn_to_reflectance(dn: np.ndarray) -> np.ndarray:
    """Convert Sentinel-2 DN to reflectance.

    Args:
        dn: Raw digital number array.

    Returns:
        Reflectance array in range [0, 1].
    """
    return dn.astype(np.float32) / SENTINEL2_SCALE_FACTOR


def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Normalized Difference Vegetation Index.

    NDVI = (B8 - B4) / (B8 + B4)
    Range: -1 to 1. >0.6 dense vegetation, <0.2 bare/water.
    """
    nir = nir.astype(np.float32)
    red = red.astype(np.float32)
    denominator = nir + red
    return np.where(denominator > 0, (nir - red) / denominator, 0.0)


def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Normalized Difference Water Index (McFeeters).

    NDWI = (B3 - B8) / (B3 + B8)
    Range: -1 to 1. >0.3 water bodies.
    """
    green = green.astype(np.float32)
    nir = nir.astype(np.float32)
    denominator = green + nir
    return np.where(denominator > 0, (green - nir) / denominator, 0.0)


def compute_nbr(nir: np.ndarray, swir22: np.ndarray) -> np.ndarray:
    """Normalized Burn Ratio.

    NBR = (B8 - B12) / (B8 + B12)
    Range: -1 to 1. High=healthy vegetation, Low=burned.
    """
    nir = nir.astype(np.float32)
    swir22 = swir22.astype(np.float32)
    denominator = nir + swir22
    return np.where(denominator > 0, (nir - swir22) / denominator, 0.0)


def compute_ndre(nir: np.ndarray, red_edge: np.ndarray) -> np.ndarray:
    """Normalized Difference Red Edge Index.

    NDRE = (B8A - B5) / (B8A + B5)
    Range: -1 to 1. Chlorophyll content, crop stress detection.
    """
    nir = nir.astype(np.float32)
    red_edge = red_edge.astype(np.float32)
    denominator = nir + red_edge
    return np.where(denominator > 0, (nir - red_edge) / denominator, 0.0)


def compute_ndmi(nir: np.ndarray, swir16: np.ndarray) -> np.ndarray:
    """Normalized Difference Moisture Index.

    NDMI = (B8 - B11) / (B8 + B11)
    Range: -1 to 1. Vegetation water content.
    """
    nir = nir.astype(np.float32)
    swir16 = swir16.astype(np.float32)
    denominator = nir + swir16
    return np.where(denominator > 0, (nir - swir16) / denominator, 0.0)


def compute_evi2(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Enhanced Vegetation Index 2.

    EVI2 = 2.5 * (B8 - B4) / (B8 + 2.4*B4 + 1)
    Minimizes atmospheric effects.
    """
    nir = nir.astype(np.float32)
    red = red.astype(np.float32)
    return 2.5 * (nir - red) / (nir + 2.4 * red + 1.0)


def compute_osavi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Optimized Soil-Adjusted Vegetation Index.

    OSAVI = 1.16 * (B8 - B4) / (B8 + B4 + 0.16)
    Vegetation in areas with soil background influence.
    """
    nir = nir.astype(np.float32)
    red = red.astype(np.float32)
    return 1.16 * (nir - red) / (nir + red + 0.16)


def compute_all_indices(bands: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Compute all spectral indices from a band dictionary.

    Args:
        bands: Dict mapping band names to reflectance arrays.
               Expected keys: "nir", "red", "green", "swir22", "swir16", "rededge1".

    Returns:
        Dict mapping index name to index array.
        Keys: ndvi, ndwi, nbr, ndmi, evi2, osavi.
        ndre included only if "rededge1" is available.
    """
    indices: dict[str, np.ndarray] = {}

    if "nir" in bands and "red" in bands:
        indices["ndvi"] = compute_ndvi(bands["nir"], bands["red"])
        indices["evi2"] = compute_evi2(bands["nir"], bands["red"])
        indices["osavi"] = compute_osavi(bands["nir"], bands["red"])

    if "green" in bands and "nir" in bands:
        indices["ndwi"] = compute_ndwi(bands["green"], bands["nir"])

    if "nir" in bands and "swir22" in bands:
        indices["nbr"] = compute_nbr(bands["nir"], bands["swir22"])

    if "nir" in bands and "swir16" in bands:
        indices["ndmi"] = compute_ndmi(bands["nir"], bands["swir16"])

    if "nir" in bands and "rededge1" in bands:
        indices["ndre"] = compute_ndre(bands["nir"], bands["rededge1"])

    return indices


def index_statistics(index_map: np.ndarray) -> dict[str, float]:
    """Compute summary statistics for a spectral index map.

    Args:
        index_map: 2D spectral index array.

    Returns:
        Dict with mean, min, max, std values.
    """
    valid = index_map[np.isfinite(index_map)]
    if valid.size == 0:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    return {
        "mean": float(np.mean(valid)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "std": float(np.std(valid)),
    }
