#!/usr/bin/env python3
"""Scan raw audio data, build JSON manifests, and pack into HDF5 files.

Run with the tse_dataset conda environment:
    conda run -n tse_dataset python data/prepare_manifest.py --config configs/hifi_tse.yaml

Produces:
    /data/tse_hdf5/train/clean_speech/{dns4,ears,vctk}.h5
    /data/tse_hdf5/train/noise/{demand,wham,audioset,dns5_noise_0..3}.h5
    /data/tse_hdf5/train/rir/rir.h5
    /data/tse_hdf5/val/val_mixtures.h5
    /data/tse_hdf5/manifests/{speaker,noise,rir}_manifest.json
"""

import argparse
import json
import os
import re
from pathlib import Path

import h5py
import numpy as np
import soundfile as sf
import yaml
from tqdm import tqdm


SAMPLE_RATE = 48000


# ---------------------------------------------------------------------------
# Scanning functions
# ---------------------------------------------------------------------------

def scan_dns4(root):
    """Scan DNS4 read_speech directory, group by reader ID."""
    speaker_files = {}
    wav_dir = Path(root)
    files = sorted(wav_dir.glob("*.wav"))
    print(f"[DNS4] Found {len(files)} wav files in {root}")

    reader_re = re.compile(r"reader_(\d+)")
    for f in files:
        match = reader_re.search(f.name)
        if match:
            spk_id = "dns4_reader_" + match.group(1)
            speaker_files.setdefault(spk_id, []).append(str(f))

    return speaker_files


def scan_ears(root):
    """Scan EARS dataset: each pXXX folder is a speaker."""
    speaker_files = {}
    root_path = Path(root)
    for spk_dir in sorted(root_path.iterdir()):
        if spk_dir.is_dir() and spk_dir.name.startswith("p"):
            spk_id = "ears_" + spk_dir.name
            wavs = sorted(str(f) for f in spk_dir.glob("*.wav"))
            if wavs:
                speaker_files[spk_id] = wavs
    return speaker_files


def scan_vctk(root):
    """Scan VCTK dataset: pXXX folders nested deep in the extracted structure."""
    speaker_files = {}
    root_path = Path(root)
    for spk_dir in sorted(root_path.rglob("p[0-9]*")):
        if spk_dir.is_dir():
            wavs = sorted(str(f) for f in spk_dir.glob("*.wav"))
            if wavs:
                spk_id = "vctk_" + spk_dir.name
                speaker_files[spk_id] = wavs
    return speaker_files


def scan_noise_dir(root, source_name):
    """Scan a noise directory recursively for all wav files."""
    files = sorted(str(f) for f in Path(root).rglob("*.wav"))
    print(f"[{source_name}] Found {len(files)} wav files in {root}")
    return files


# AudioSet categories suitable for TSE noise (all except Music)
AUDIOSET_ALLOWED_CATEGORIES = {"Natural_sounds", "Vehicle", "Domestic_sounds", "Animal"}


def scan_audioset_filtered(root):
    """Scan AudioSet directory, keeping all except Music."""
    all_files = sorted(Path(root).glob("*.wav"))
    filtered = []
    category_counts = {}
    for f in all_files:
        name = f.stem
        # Exclude Music
        if "_Music" in name:
            category_counts["Music_excluded"] = category_counts.get("Music_excluded", 0) + 1
            continue
        # Identify category for reporting
        matched_cat = "other"
        for cat in AUDIOSET_ALLOWED_CATEGORIES:
            if cat in name:
                matched_cat = cat
                break
        category_counts[matched_cat] = category_counts.get(matched_cat, 0) + 1
        filtered.append(str(f))

    total = len(all_files)
    kept = len(filtered)
    print(f"[AudioSet] {total} total -> {kept} kept ({total - kept} dropped: Music)")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count}")
    return filtered


def scan_rir_dir(root, source_name):
    """Scan an RIR directory recursively for all wav files."""
    files = sorted(str(f) for f in Path(root).rglob("*.wav"))
    print(f"[{source_name}] Found {len(files)} wav files in {root}")
    return files


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_wav_float32(path):
    """Load wav file as float32 numpy array, force mono, verify sample rate."""
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        return None, sr
    return data, sr


def split_speakers_train_val(speaker_ids, val_ratio=0.05):
    """Deterministic train/val split by sorted speaker ID.

    Last val_ratio fraction of sorted IDs go to validation.
    Returns (train_ids, val_ids).
    """
    sorted_ids = sorted(speaker_ids)
    n_val = max(1, int(len(sorted_ids) * val_ratio))
    train_ids = sorted_ids[:-n_val]
    val_ids = sorted_ids[-n_val:]
    return train_ids, val_ids


# ---------------------------------------------------------------------------
# HDF5 packing: Clean Speech
# ---------------------------------------------------------------------------

def pack_clean_speech_hdf5(speaker_files, output_path, source_name,
                           min_utt_length, speaker_subset=None):
    """Pack speaker-grouped clean speech into HDF5 (per-utterance storage).

    Layout:
        /speaker_ids: (num_speakers,) string
        /<speaker_id>/count: attr int
        /<speaker_id>/<idx>: (L,) float32 per utterance
        /<speaker_id>/durations: (N,) int64
        /<speaker_id>/filenames: (N,) string
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    min_samples = int(min_utt_length * SAMPLE_RATE)

    # Use subset if provided (for train/val split)
    if speaker_subset is not None:
        speaker_files = {k: v for k, v in speaker_files.items()
                         if k in speaker_subset}

    speaker_ids = sorted(speaker_files.keys())
    print(f"  Packing {len(speaker_ids)} speakers into {output_path}")

    dt = h5py.string_dtype()
    valid_speaker_ids = []

    with h5py.File(output_path, "w") as hf:
        hf.attrs["sample_rate"] = SAMPLE_RATE
        hf.attrs["source"] = source_name

        for spk_id in tqdm(speaker_ids, desc=f"  {source_name} speakers"):
            paths = speaker_files[spk_id]
            waveforms = []
            durations = []
            filenames = []

            for p in paths:
                wav, sr = load_wav_float32(p)
                if wav is None:
                    continue
                if len(wav) < min_samples:
                    continue
                waveforms.append(wav)
                durations.append(len(wav))
                filenames.append(os.path.basename(p))

            if len(waveforms) < 2:
                continue

            valid_speaker_ids.append(spk_id)
            grp = hf.create_group(spk_id)
            grp.attrs["count"] = len(waveforms)

            # Per-utterance datasets (no padding, no compression for fast I/O)
            for i, wav in enumerate(waveforms):
                grp.create_dataset(str(i), data=wav)

            grp.create_dataset("durations",
                               data=np.array(durations, dtype=np.int64))
            grp.create_dataset("filenames", data=filenames, dtype=dt)

        # Store final speaker IDs (only those with >= 2 valid utterances)
        hf.create_dataset("speaker_ids", data=valid_speaker_ids, dtype=dt)
        hf.attrs["num_speakers"] = len(valid_speaker_ids)

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"  Done: {output_path} ({size_mb:.0f} MB, "
          f"{len(valid_speaker_ids)} speakers)")


# ---------------------------------------------------------------------------
# HDF5 packing: Noise
# ---------------------------------------------------------------------------

def pack_noise_hdf5_single(file_list, output_path, source_name,
                           chunk_seconds=0, mono_downmix=False):
    """Pack noise files into a single HDF5 file.

    Layout:
        /count: attr int
        /0, 1, 2, ...: (L,) float32 per clip
        /filenames: (N,) string

    Args:
        chunk_seconds: if > 0, split long files into chunks of this duration.
        mono_downmix: if True, average stereo channels to mono.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    chunk_samples = int(chunk_seconds * SAMPLE_RATE) if chunk_seconds > 0 else 0

    dt = h5py.string_dtype()
    valid_count = 0
    filenames = []

    with h5py.File(output_path, "w") as hf:
        hf.attrs["sample_rate"] = SAMPLE_RATE
        hf.attrs["source"] = source_name

        for p in tqdm(file_list, desc=f"  {source_name}"):
            try:
                data, sr = sf.read(p, dtype="float32")
            except Exception as e:
                print(f"  WARNING: failed to read {p}: {e}")
                continue

            if data.ndim > 1:
                if mono_downmix:
                    data = data.mean(axis=1)
                else:
                    data = data[:, 0]  # take first channel

            if sr != SAMPLE_RATE:
                continue

            basename = os.path.basename(p)

            if chunk_samples > 0 and len(data) > chunk_samples:
                # Split into chunks
                n_chunks = (len(data) + chunk_samples - 1) // chunk_samples
                for c in range(n_chunks):
                    start = c * chunk_samples
                    end = min(start + chunk_samples, len(data))
                    chunk = data[start:end]
                    if len(chunk) < SAMPLE_RATE:  # skip < 1s tail
                        continue
                    hf.create_dataset(str(valid_count), data=chunk)
                    filenames.append(f"{basename}_chunk{c}")
                    valid_count += 1
            else:
                hf.create_dataset(str(valid_count), data=data)
                filenames.append(basename)
                valid_count += 1

        hf.attrs["count"] = valid_count
        hf.create_dataset("filenames", data=filenames, dtype=dt)

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"  Done: {output_path} ({size_mb:.0f} MB, {valid_count} clips)")


def pack_noise_hdf5_sharded(file_list, output_dir, source_name, num_shards):
    """Pack noise files into multiple sharded HDF5 files.

    Splits file_list evenly across num_shards.
    Output: {source_name}_0.h5, {source_name}_1.h5, ...
    """
    sorted_files = sorted(file_list)
    shard_size = (len(sorted_files) + num_shards - 1) // num_shards

    for shard_idx in range(num_shards):
        start = shard_idx * shard_size
        end = min(start + shard_size, len(sorted_files))
        shard_files = sorted_files[start:end]
        if not shard_files:
            continue

        output_path = os.path.join(output_dir, f"{source_name}_{shard_idx}.h5")
        print(f"\n  Shard {shard_idx}: {len(shard_files)} files")
        pack_noise_hdf5_single(
            shard_files, output_path,
            f"{source_name}_shard{shard_idx}",
        )


# ---------------------------------------------------------------------------
# HDF5 packing: RIR
# ---------------------------------------------------------------------------

def pack_rir_hdf5(rir_sources, output_path):
    """Pack RIR files into HDF5, grouped by source.

    Layout:
        /<source_name>/count: attr int
        /<source_name>/0, 1, ...: (L,) float32 per RIR
        /<source_name>/filenames: (N,) string
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with h5py.File(output_path, "w") as hf:
        hf.attrs["sample_rate"] = SAMPLE_RATE

        for source_name, file_list in rir_sources.items():
            print(f"  Packing {source_name}: {len(file_list)} files")
            grp = hf.create_group(source_name)
            filenames = []
            valid_count = 0

            for p in tqdm(file_list, desc=f"  {source_name}"):
                wav, sr = load_wav_float32(p)
                if wav is None:
                    continue
                grp.create_dataset(str(valid_count), data=wav)
                filenames.append(os.path.basename(p))
                valid_count += 1

            if valid_count == 0:
                print(f"  WARNING: No valid files for {source_name}, skipping")
                continue

            grp.attrs["count"] = valid_count
            dt = h5py.string_dtype()
            grp.create_dataset("filenames", data=filenames, dtype=dt)

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"  Done: {output_path} ({size_mb:.0f} MB)")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_scan(all_speakers, noise_sources, rir_sources, min_utt_length):
    """Print validation summary of scanned data."""
    min_samples = int(min_utt_length * SAMPLE_RATE)

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    # Speaker / utterance stats
    print("\n--- Clean Speech ---")
    total_spk = 0
    total_utt = 0
    for spk_id, paths in sorted(all_speakers.items()):
        total_spk += 1
        total_utt += len(paths)
    print(f"  Total speakers: {total_spk}")
    print(f"  Total utterances: {total_utt}")
    print(f"  Min utterance length filter: {min_utt_length}s ({min_samples} samples)")

    # Sample a few files to check durations
    print("\n  Duration spot-check (20 random files):")
    import random
    random.seed(42)
    all_paths = []
    for paths in all_speakers.values():
        all_paths.extend(paths)
    sample_paths = random.sample(all_paths, min(20, len(all_paths)))
    short_count = 0
    for p in sample_paths:
        try:
            info = sf.info(p)
            dur = info.duration
            if dur < min_utt_length:
                short_count += 1
                print(f"    SHORT: {os.path.basename(p)} = {dur:.2f}s")
        except Exception:
            pass
    if short_count == 0:
        print(f"    All {len(sample_paths)} sampled files >= {min_utt_length}s")

    # Noise stats
    print("\n--- Noise ---")
    for name, files in noise_sources.items():
        print(f"  {name}: {len(files)} files")

    # RIR stats
    print("\n--- RIR ---")
    for name, files in rir_sources.items():
        print(f"  {name}: {len(files)} files")

    # RIR sample rate spot-check
    print("\n  RIR sample rate spot-check:")
    for name, files in rir_sources.items():
        sample = files[:10]
        bad = 0
        for p in sample:
            try:
                info = sf.info(p)
                if info.samplerate != SAMPLE_RATE:
                    bad += 1
                    print(f"    {name}: {os.path.basename(p)} sr={info.samplerate}")
            except Exception:
                bad += 1
        if bad == 0:
            print(f"    {name}: all {len(sample)} sampled files are {SAMPLE_RATE}Hz")

    print("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare HDF5 datasets for HiFi-TSE")
    parser.add_argument("--config", type=str, default="configs/hifi_tse.yaml")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only scan and validate, no HDF5 packing")
    parser.add_argument("--source", type=str, default=None,
                        help="Pack a single source: dns4/ears/vctk/demand/"
                             "wham/audioset/dns5_noise/rir")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    hdf5_dir = cfg["data"]["hdf5_dir"]
    train_dir = os.path.join(hdf5_dir, "train")
    manifest_dir = os.path.join(hdf5_dir, "manifests")
    os.makedirs(manifest_dir, exist_ok=True)

    min_utt_length = cfg["audio"].get("min_utterance_length", 3.0)
    val_ratio = cfg["data"].get("val_split_ratio", 0.05)
    dns5_shards = cfg["data"].get("dns5_noise_shards", 4)

    # ---- Scan Clean Speech ----
    print("=" * 60)
    print("Scanning clean speech sources...")
    clean_paths = cfg["data"]["clean_speech"]

    dns4_speakers = scan_dns4(clean_paths[0])
    print(f"  DNS4: {len(dns4_speakers)} speakers, "
          f"{sum(len(v) for v in dns4_speakers.values())} utterances")

    ears_speakers = scan_ears(clean_paths[1])
    print(f"  EARS: {len(ears_speakers)} speakers, "
          f"{sum(len(v) for v in ears_speakers.values())} utterances")

    vctk_speakers = scan_vctk(clean_paths[2])
    print(f"  VCTK: {len(vctk_speakers)} speakers, "
          f"{sum(len(v) for v in vctk_speakers.values())} utterances")

    # Train/val split per source
    all_speakers = {}
    all_speakers.update(dns4_speakers)
    all_speakers.update(ears_speakers)
    all_speakers.update(vctk_speakers)

    train_spk_dns4, val_spk_dns4 = split_speakers_train_val(
        dns4_speakers.keys(), val_ratio)
    train_spk_ears, val_spk_ears = split_speakers_train_val(
        ears_speakers.keys(), val_ratio)
    train_spk_vctk, val_spk_vctk = split_speakers_train_val(
        vctk_speakers.keys(), val_ratio)

    train_speakers = set(train_spk_dns4 + train_spk_ears + train_spk_vctk)
    val_speakers = set(val_spk_dns4 + val_spk_ears + val_spk_vctk)

    print(f"\n  Train/Val split ({val_ratio:.0%} val):")
    print(f"    DNS4:  {len(train_spk_dns4)} train, {len(val_spk_dns4)} val")
    print(f"    EARS:  {len(train_spk_ears)} train, {len(val_spk_ears)} val")
    print(f"    VCTK:  {len(train_spk_vctk)} train, {len(val_spk_vctk)} val")
    print(f"    Total: {len(train_speakers)} train, {len(val_speakers)} val")

    # Save speaker manifest
    speaker_manifest = {
        "train": sorted(train_speakers),
        "val": sorted(val_speakers),
        "all_speakers": {k: v for k, v in sorted(all_speakers.items())},
    }
    manifest_path = os.path.join(manifest_dir, "speaker_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(speaker_manifest, f, indent=0)
    print(f"Saved: {manifest_path}")

    # ---- Scan Noise ----
    print("\n" + "=" * 60)
    print("Scanning noise sources...")
    noise_cfg = cfg["data"]["noise"]
    noise_sources = {}
    noise_sources["demand"] = scan_noise_dir(noise_cfg["demand"], "DEMAND")
    noise_sources["wham"] = scan_noise_dir(noise_cfg["wham"], "WHAM")
    noise_sources["audioset"] = scan_audioset_filtered(noise_cfg["audioset"])
    noise_sources["dns5_noise"] = scan_noise_dir(
        noise_cfg["dns5_noise"], "DNS5_noise")

    total_noise = sum(len(v) for v in noise_sources.values())
    print(f"Total noise files: {total_noise}")

    noise_manifest_path = os.path.join(manifest_dir, "noise_manifest.json")
    with open(noise_manifest_path, "w") as f:
        json.dump({k: len(v) for k, v in noise_sources.items()}, f, indent=2)
    print(f"Saved: {noise_manifest_path}")

    # ---- Scan RIR ----
    print("\n" + "=" * 60)
    print("Scanning RIR sources...")
    rir_cfg = cfg["data"]["rir"]
    rir_sources = {}
    rir_sources["slr26"] = scan_rir_dir(rir_cfg["slr26"], "SLR26")
    rir_sources["slr28"] = scan_rir_dir(rir_cfg["slr28"], "SLR28")

    total_rir = sum(len(v) for v in rir_sources.values())
    print(f"Total RIR files: {total_rir}")

    rir_manifest_path = os.path.join(manifest_dir, "rir_manifest.json")
    with open(rir_manifest_path, "w") as f:
        json.dump({k: len(v) for k, v in rir_sources.items()}, f, indent=2)
    print(f"Saved: {rir_manifest_path}")

    # ---- Validate ----
    validate_scan(all_speakers, noise_sources, rir_sources, min_utt_length)

    if args.validate_only:
        print("\n--validate-only specified, skipping HDF5 packing.")
        return

    # ---- Pack HDF5 ----
    print("\n" + "=" * 60)
    print("Packing HDF5 files...")

    sources_to_pack = args.source.split(",") if args.source else [
        "dns4", "ears", "vctk",
        "demand", "wham", "audioset", "dns5_noise",
        "rir",
    ]

    # Clean speech
    clean_out = os.path.join(train_dir, "clean_speech")

    if "dns4" in sources_to_pack:
        print("\n--- DNS4 (train) ---")
        pack_clean_speech_hdf5(
            dns4_speakers,
            os.path.join(clean_out, "dns4.h5"),
            "dns4", min_utt_length,
            speaker_subset=train_speakers,
        )

    if "ears" in sources_to_pack:
        print("\n--- EARS (train) ---")
        pack_clean_speech_hdf5(
            ears_speakers,
            os.path.join(clean_out, "ears.h5"),
            "ears", min_utt_length,
            speaker_subset=train_speakers,
        )

    if "vctk" in sources_to_pack:
        print("\n--- VCTK (train) ---")
        pack_clean_speech_hdf5(
            vctk_speakers,
            os.path.join(clean_out, "vctk.h5"),
            "vctk", min_utt_length,
            speaker_subset=train_speakers,
        )

    # Noise
    noise_out = os.path.join(train_dir, "noise")

    if "demand" in sources_to_pack:
        print("\n--- DEMAND ---")
        pack_noise_hdf5_single(
            noise_sources["demand"],
            os.path.join(noise_out, "demand.h5"),
            "demand",
        )

    if "wham" in sources_to_pack:
        print("\n--- WHAM (30s chunks, mono downmix) ---")
        pack_noise_hdf5_single(
            noise_sources["wham"],
            os.path.join(noise_out, "wham.h5"),
            "wham",
            chunk_seconds=30.0,
            mono_downmix=True,
        )

    if "audioset" in sources_to_pack:
        print("\n--- AudioSet (filtered) ---")
        pack_noise_hdf5_single(
            noise_sources["audioset"],
            os.path.join(noise_out, "audioset.h5"),
            "audioset",
        )

    if "dns5_noise" in sources_to_pack:
        print("\n--- DNS5 Noise (sharded) ---")
        pack_noise_hdf5_sharded(
            noise_sources["dns5_noise"],
            noise_out,
            "dns5_noise",
            num_shards=dns5_shards,
        )

    # RIR
    if "rir" in sources_to_pack:
        print("\n--- RIR ---")
        pack_rir_hdf5(
            rir_sources,
            os.path.join(train_dir, "rir", "rir.h5"),
        )

    print("\n" + "=" * 60)
    print("All done!")
    print(f"HDF5 files written to: {train_dir}")
    print(f"Manifests written to: {manifest_dir}")


if __name__ == "__main__":
    main()
