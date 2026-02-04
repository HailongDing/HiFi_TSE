#!/usr/bin/env python3
"""Pack only noise into HDF5 (filtered AudioSet). Stops before RIR.

Run with: conda run -n tse_dataset python data/pack_noise_only.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from data.prepare_manifest import (
    pack_noise_hdf5, scan_noise_dir, scan_audioset_filtered,
)

with open("configs/hifi_tse.yaml") as f:
    cfg = yaml.safe_load(f)

hdf5_dir = cfg["data"]["hdf5_dir"]
manifest_dir = os.path.join(hdf5_dir, "manifests")
noise_paths = cfg["data"]["noise"]

# Rebuild noise manifest with AudioSet filtering
print("Scanning noise sources (with AudioSet filtering)...")
noise_sources = {}
noise_sources["demand"] = scan_noise_dir(noise_paths[0], "DEMAND")
noise_sources["wham"] = scan_noise_dir(noise_paths[1], "WHAM")
noise_sources["audioset"] = scan_audioset_filtered(noise_paths[2])

total_noise = sum(len(v) for v in noise_sources.values())
print(f"Total noise files (filtered): {total_noise}")

noise_manifest_path = os.path.join(manifest_dir, "noise_manifest.json")
with open(noise_manifest_path, "w") as f:
    json.dump(noise_sources, f, indent=0)
print(f"Saved: {noise_manifest_path}")

# Remove incomplete file if exists
noise_h5 = os.path.join(hdf5_dir, "noise", "noise.h5")
if os.path.exists(noise_h5):
    os.remove(noise_h5)
    print(f"Removed incomplete: {noise_h5}")

print("\n--- Packing Noise HDF5 ---")
pack_noise_hdf5(noise_sources, noise_h5)

print("\nNoise packing complete! Review RIR list before proceeding.")
