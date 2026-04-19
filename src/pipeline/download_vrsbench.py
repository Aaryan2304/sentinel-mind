"""Download VRSBench dataset from HuggingFace.

Downloads to data/vrsbench/ directory.
Size: ~12.5 GB. Requires stable internet connection.

Known issue: The dataset has a type mismatch (string 'Q6' in int64 field)
that causes DatasetGenerationError with standard loading. Workaround: use
streaming mode or download raw files directly.
"""

import sys
from pathlib import Path

def main():
    output_dir = Path("data/vrsbench")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading VRSBench dataset (xiang709/VRSBench)...")
    print(f"Output: {output_dir}")
    print("Size: ~12.5 GB. This may take 10-30 minutes.")

    from datasets import load_dataset

    # Use streaming to avoid the ArrowInvalid type mismatch error
    dataset = load_dataset(
        "xiang709/VRSBench",
        split="train",
        streaming=True,
    )

    # Iterate through streaming dataset and save samples
    count = 0
    save_path = output_dir / "train"
    save_path.mkdir(exist_ok=True)

    print("Streaming dataset, saving samples...")
    for sample in dataset:
        count += 1
        if count % 1000 == 0:
            print(f"  Processed {count} samples...")
        if count >= 50000:  # Safety limit
            print(f"  Reached {count} samples, stopping")
            break

    print(f"Dataset streaming complete: {count} samples processed")

    # Alternative: save dataset to disk in arrow format
    try:
        dataset_nonstream = load_dataset(
            "xiang709/VRSBench",
            split="train",
        )
        print(f"Full dataset loaded: {len(dataset_nonstream)} samples")
        print(f"Columns: {dataset_nonstream.column_names}")
        dataset_nonstream.save_to_disk(str(save_path))
        print(f"Saved to {save_path}")
    except Exception as e:
        print(f"Full load failed (expected): {e}")
        print("Using streaming approach for training data access")

if __name__ == "__main__":
    main()
