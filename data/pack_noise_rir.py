#!/usr/bin/env python3
"""Pack only noise and RIR into HDF5 (clean speech already done).

Regenerates noise manifest with AudioSet filtering, then packs.

Run with: cd /home/hailong/code/from_draft/v1 && conda run -n tse_dataset python data/pack_noise_rir.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from data.prepare_manifest import (
    pack_noise_hdf5, pack_rir_hdf5,
    scan_noise_dir, scan_audioset_filtered,
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

# Load RIR manifest (unchanged)
with open(os.path.join(manifest_dir, "rir_manifest.json")) as f:
    rir_sources = json.load(f)

print("\n--- Packing Noise HDF5 ---")
pack_noise_hdf5(noise_sources, os.path.join(hdf5_dir, "noise", "noise.h5"))

print("\n--- Packing RIR HDF5 ---")
pack_rir_hdf5(rir_sources, os.path.join(hdf5_dir, "rir", "rir.h5"))

print("\nAll done!")
