"""Run full SimSat data collection.

Collects 100 positions with all 5 band combinations.
Estimated time: 25-40 minutes.
"""

import sys
sys.path.insert(0, ".")

from src.pipeline.collector import collect_orbit_sequence, save_collection

def main():
    print("=" * 60)
    print("Sentinel Mind — Full Data Collection")
    print("=" * 60)

    data = collect_orbit_sequence(
        num_positions=100,
        poll_interval_seconds=5.0,
        max_cloud_cover=80.0,
        size_km=5.0,
    )

    if data:
        output_dir = save_collection(data, output_dir="data/raw")
        print(f"\nCollection complete!")
        print(f"Positions: {len(data)}")
        print(f"Output: {output_dir}")
    else:
        print("No data collected")

if __name__ == "__main__":
    main()
