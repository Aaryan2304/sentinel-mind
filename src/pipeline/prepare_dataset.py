"""Load and prepare VRSBench + custom spectral dataset for TRL fine-tuning.

VRSBench is stored as raw JSON + ZIP files in the HuggingFace cache.
The dataset has a known type mismatch issue preventing standard load_dataset() usage.
This module loads the raw JSON directly.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

VRSBENCH_CACHE = Path.home() / ".cache/huggingface/hub/datasets--xiang709--VRSBench/snapshots/6c39918caaf19fc876d8afb2c4658c999b670a59"


def load_vrsbench_annotations(
    split: str = "train",
    cache_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Load VRSBench annotations from cached JSON.

    Args:
        split: "train" or "val".
        cache_dir: Path to the VRSBench snapshot directory.

    Returns:
        List of annotation dicts with keys: id, image, conversations.
    """
    if cache_dir is None:
        cache_dir = VRSBENCH_CACHE

    json_path = cache_dir / f"VRSBench_{split}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"VRSBench JSON not found: {json_path}")

    with open(json_path) as f:
        data = json.load(f)

    print(f"Loaded VRSBench {split}: {len(data)} samples")
    return data


def convert_to_trl_format(
    entry: dict[str, Any],
    image_dir: Path | None = None,
) -> dict[str, Any]:
    """Convert a VRSBench entry to TRL SFTTrainer messages format.

    VRSBench format:
        {"from": "human", "value": "<image>\n[caption] ..."}
        {"from": "gpt", "value": "..."}

    TRL format:
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "..."}]}
        {"role": "assistant", "content": "..."}
    """
    messages = []
    images = []

    for conv in entry.get("conversations", []):
        role = "user" if conv["from"] == "human" else "assistant"
        value = conv["value"]

        if role == "user" and "<image>" in value:
            # Extract text after <image> token
            text = value.replace("<image>\n", "").replace("<image>", "").strip()
            content = [
                {"type": "image"},
                {"type": "text", "text": text},
            ]
            messages.append({"role": "user", "content": content})

            # Image path
            img_name = entry.get("image", "")
            if image_dir and img_name:
                images.append(str(image_dir / img_name))
            elif img_name:
                images.append(img_name)
        else:
            messages.append({"role": role, "content": value})

    return {"messages": messages, "images": images}


def build_combined_dataset(
    vrsbench_limit: int = 5000,
    custom_qa_path: str | Path = "data/custom/spectral_qa.jsonl",
    output_path: str | Path = "data/custom/combined_train.jsonl",
    val_split: float = 0.1,
) -> tuple[Path, Path]:
    """Build combined dataset: VRSBench + custom spectral Q&A.

    Args:
        vrsbench_limit: Max VRSBench samples to include.
        custom_qa_path: Path to custom spectral Q&A JSONL.
        output_path: Output path for combined training data.
        val_split: Fraction for validation.

    Returns:
        Tuple of (train_path, val_path).
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Load VRSBench
    print("Loading VRSBench annotations...")
    vrsbench = load_vrsbench_annotations("train")

    # Convert to TRL format
    print(f"Converting {min(vrsbench_limit, len(vrsbench))} VRSBench samples...")
    vrsbench_samples = []
    for entry in vrsbench[:vrsbench_limit]:
        sample = convert_to_trl_format(entry)
        if sample["messages"]:
            vrsbench_samples.append(sample)

    # Load custom spectral Q&A
    custom_samples = []
    custom_path = Path(custom_qa_path)
    if custom_path.exists():
        print(f"Loading custom spectral Q&A from {custom_path}...")
        with open(custom_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    custom_samples.append(json.loads(line))
        print(f"Loaded {len(custom_samples)} custom samples")
    else:
        print(f"Warning: Custom Q&A file not found: {custom_path}")

    # Combine
    all_samples = vrsbench_samples + custom_samples
    random.shuffle(all_samples)

    # Split
    val_size = int(len(all_samples) * val_split)
    val_samples = all_samples[:val_size]
    train_samples = all_samples[val_size:]

    # Write
    train_path = output.parent / "combined_train.jsonl"
    val_path = output.parent / "combined_val.jsonl"

    with open(train_path, "w") as f:
        for sample in train_samples:
            f.write(json.dumps(sample) + "\n")

    with open(val_path, "w") as f:
        for sample in val_samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Combined dataset:")
    print(f"  Train: {len(train_samples)} samples -> {train_path}")
    print(f"  Val: {len(val_samples)} samples -> {val_path}")
    print(f"  VRSBench: {len(vrsbench_samples)}, Custom: {len(custom_samples)}")

    return train_path, val_path


if __name__ == "__main__":
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    build_combined_dataset(vrsbench_limit=limit)
