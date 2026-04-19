"""VLM inference pipeline.

Runs fine-tuned LFM2.5-VL-450M on satellite imagery to generate
investigation reports. Designed for RTX 3050 Ti (4GB VRAM).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


def load_model(
    checkpoint_path: str | Path,
    base_model: str = "LiquidAI/LFM2.5-VL-450M",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[Any, Any]:
    """Load fine-tuned model from checkpoint.

    Args:
        checkpoint_path: Path to LoRA checkpoint directory.
        base_model: Base model name (used if checkpoint is LoRA adapter).
        dtype: Model dtype.

    Returns:
        Tuple of (model, processor).
    """
    processor = AutoProcessor.from_pretrained(str(checkpoint_path))
    model = AutoModelForImageTextToText.from_pretrained(
        str(checkpoint_path),
        dtype=dtype,
        device_map="auto",
    )
    model.eval()
    return model, processor


def investigate_image(
    image: Image.Image,
    spectral_data: dict[str, Any],
    model: Any,
    processor: Any,
    prompt_template: Optional[str] = None,
) -> str:
    """Run full investigation pipeline on a satellite image.

    Args:
        image: PIL Image (RGB or false-color composite).
        spectral_data: Dict with spectral index stats and change info.
        model: Loaded VLM model.
        processor: Model processor.
        prompt_template: Optional custom prompt template.

    Returns:
        Investigation report string.
    """
    from .prompts import build_investigation_prompt

    if prompt_template is None:
        prompt_template = build_investigation_prompt(spectral_data)

    messages = [
        {
            "role": "system",
            "content": "You are a satellite imagery analysis AI specialized in Earth observation.",
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_template},
            ],
        },
    ]

    inputs = processor(
        text=processor.apply_chat_template(messages, tokenize=False),
        images=[image],
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )

    output = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    return output


def parse_investigation(report: str) -> dict[str, Any]:
    """Extract structured fields from VLM investigation output.

    Args:
        report: Raw VLM output string.

    Returns:
        Dict with keys: severity, confidence, findings, recommendation.
    """
    result: dict[str, Any] = {
        "severity": "unknown",
        "confidence": 0.0,
        "findings": "",
        "recommendation": "",
    }

    severity_match = re.search(
        r"severity[:\s]*(low|medium|high|critical)",
        report,
        re.IGNORECASE,
    )
    if severity_match:
        result["severity"] = severity_match.group(1).lower()

    confidence_match = re.search(r"(\d{1,3})\s*%", report)
    if confidence_match:
        result["confidence"] = min(float(confidence_match.group(1)) / 100.0, 1.0)

    result["findings"] = report
    return result


def compress_report(report: dict[str, Any]) -> bytes:
    """Compress investigation report for downlink.

    Target: <50KB for a full investigation report.

    Args:
        report: Parsed investigation dict.

    Returns:
        Compressed bytes.
    """
    import zlib

    json_bytes = json.dumps(report, separators=(",", ":")).encode("utf-8")
    return zlib.compress(json_bytes, level=9)
