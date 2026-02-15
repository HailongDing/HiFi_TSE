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
    pack_noise_hdf5_single, pack_rir_hdf5,
    scan_noise_dir, scan_audioset_filtered, scan_rir_dir,
)

with open("configs/hifi_tse.yaml") as f:
    cfg = yaml.safe_load(f)

hdf5_dir = cfg["data"]["hdf5_dir"]
manifest_dir = os.path.join(hdf5_dir, "manifests")
noise_cfg = cfg["data"]["noise"]

# Rebuild noise manifest with AudioSet filtering
print("Scanning noise sources (with AudioSet filtering)...")
noise_sources = {}
noise_sources["demand"] = scan_noise_dir(noise_cfg["demand"], "DEMAND")
noise_sources["wham"] = scan_noise_dir(noise_cfg["wham"], "WHAM")
noise_sources["audioset"] = scan_audioset_filtered(noise_cfg["audioset"])

total_noise = sum(len(v) for v in noise_sources.values())
print(f"Total noise files (filtered): {total_noise}")

noise_manifest_path = os.path.join(manifest_dir, "noise_manifest.json")
with open(noise_manifest_path, "w") as f:
    json.dump(noise_sources, f, indent=0)
print(f"Saved: {noise_manifest_path}")

# Re-scan RIR directories (manifest only stores counts, not file lists)
rir_cfg = cfg["data"]["rir"]
rir_sources = {}
rir_sources["slr26"] = scan_rir_dir(rir_cfg["slr26"], "SLR26")
rir_sources["slr28"] = scan_rir_dir(rir_cfg["slr28"], "SLR28")

print("\n--- Packing Noise HDF5 ---")
noise_out = os.path.join(hdf5_dir, "train", "noise")
for source_name, file_list in noise_sources.items():
    out_path = os.path.join(noise_out, f"{source_name}.h5")
    pack_noise_hdf5_single(file_list, out_path, source_name)

print("\n--- Packing RIR HDF5 ---")
pack_rir_hdf5(rir_sources, os.path.join(hdf5_dir, "train", "rir", "rir.h5"))

print("\nAll done!")
