#!/usr/bin/env python3
"""Evaluate a trained HiFi-TSE model on held-out validation speakers.

Generates test mixtures from val speakers (raw wav files, not in training
HDF5), runs inference, and computes objective metrics.

Run with:
    conda run -n USEF-TFGridNet python evaluate.py \
        --config configs/hifi_tse.yaml \
        --checkpoint checkpoints/checkpoint_final.pt \
        --num-samples 200
"""

import argparse
import json
import os
import random

import numpy as np
import soundfile as sf
import torch
import yaml
from scipy.signal import fftconvolve
from pathlib import Path

from models.generator import Generator


SAMPLE_RATE = 48000


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def compute_si_sdr(estimate, target):
    """SI-SDR in dB (numpy, single pair)."""
    estimate = estimate - np.mean(estimate)
    target = target - np.mean(target)
    dot = np.sum(target * estimate)
    s_target_energy = np.sum(target ** 2) + 1e-8
    proj = dot * target / s_target_energy
    noise = estimate - proj
    si_sdr_val = 10.0 * np.log10(
        np.sum(proj ** 2) + 1e-8) - 10.0 * np.log10(
        np.sum(noise ** 2) + 1e-8)
    return float(si_sdr_val)


def compute_pesq(estimate, target, sr=48000):
    """PESQ score. Returns None if pesq is unavailable or fails."""
    try:
        from pesq import pesq
        # pesq library supports 8000 and 16000 Hz only; resample
        import torchaudio
        est_t = torch.from_numpy(estimate).float().unsqueeze(0)
        tgt_t = torch.from_numpy(target).float().unsqueeze(0)
        est_16k = torchaudio.functional.resample(est_t, sr, 16000).squeeze().numpy()
        tgt_16k = torchaudio.functional.resample(tgt_t, sr, 16000).squeeze().numpy()
        return float(pesq(16000, tgt_16k, est_16k, "wb"))
    except Exception:
        return None


def compute_stoi(estimate, target, sr=48000):
    """STOI score. Returns None if pystoi is unavailable or fails."""
    try:
        from pystoi import stoi
        # pystoi works with any sample rate
        return float(stoi(target, estimate, sr, extended=False))
    except Exception:
        return None


def compute_energy_db(signal):
    """RMS energy in dB."""
    energy = np.mean(signal ** 2)
    return float(10.0 * np.log10(energy + 1e-8))


# ---------------------------------------------------------------------------
# Audio utilities (mirrors data/dataset.py for test mixture creation)
# ---------------------------------------------------------------------------

def load_wav(path):
    """Load wav as float32 mono, skip if not 48kHz."""
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        return None
    return data


def random_crop(wav, length):
    """Random crop to length. Assumes len(wav) >= length."""
    if len(wav) >= length:
        offset = random.randint(0, len(wav) - length)
        return wav[offset:offset + length]
    # Should not happen with min_utterance_length >= segment length
    out = np.zeros(length, dtype=wav.dtype)
    out[:len(wav)] = wav
    return out


def loop_to_length(wav, length):
    """Loop/crop noise to fill target length."""
    if len(wav) >= length:
        offset = random.randint(0, len(wav) - length)
        return wav[offset:offset + length]
    reps = (length // len(wav)) + 1
    looped = np.tile(wav, reps)
    offset = random.randint(0, len(looped) - length)
    return looped[offset:offset + length]


def apply_rir(wav, rir, crop_length=None):
    """Convolve with RIR. If crop_length is given, crop result to that length."""
    rir = rir / (np.linalg.norm(rir) + 1e-8)
    out = fftconvolve(wav, rir, mode="full")
    if crop_length is not None:
        out = out[:crop_length]
    return out


def mix_at_snr(signal, noise, snr_db):
    """Mix signal and noise at target SNR."""
    sig_power = np.mean(signal ** 2) + 1e-8
    noise_power = np.mean(noise ** 2) + 1e-8
    snr_linear = 10.0 ** (snr_db / 10.0)
    scale = np.sqrt(sig_power / (noise_power * snr_linear + 1e-8))
    return signal + scale * noise


# ---------------------------------------------------------------------------
# Test data generation
# ---------------------------------------------------------------------------

def load_speaker_files(speaker_files, min_length):
    """Load and filter wav files for a speaker."""
    min_samples = int(min_length * SAMPLE_RATE)
    wavs = []
    for path in speaker_files:
        wav = load_wav(path)
        if wav is not None and len(wav) >= min_samples:
            wavs.append(wav)
    return wavs


def scan_noise_files(noise_cfg):
    """Collect noise file paths from config."""
    files = []
    for source_name, path in noise_cfg.items():
        for f in Path(path).rglob("*.wav"):
            files.append(str(f))
    return files


def scan_rir_files(rir_cfg):
    """Collect RIR file paths from config."""
    files = []
    for source_name, path in rir_cfg.items():
        for f in Path(path).rglob("*.wav"):
            files.append(str(f))
    return files


def generate_test_sample(target_wavs, interferer_pool, noise_files,
                         rir_files, seg_samples, target_present, snr_db,
                         num_interferers):
    """Generate a single test mixture.

    Returns: (mix, ref, target_clean, metadata_dict) or None on failure.
    """
    if len(target_wavs) < 2:
        return None

    # Pick target and reference (different utterances)
    indices = random.sample(range(len(target_wavs)), 2)
    ref_wav = random_crop(target_wavs[indices[1]], seg_samples)

    # Load a random RIR
    def get_rir():
        for _ in range(10):
            rir = load_wav(random.choice(rir_files))
            if rir is not None:
                return rir
        return None

    # Convolve-then-crop: apply RIR to full utterance, then crop to segment
    rir = get_rir()
    if rir is None:
        return None
    clean_target = random_crop(target_wavs[indices[0]], seg_samples)
    target_wav = random_crop(apply_rir(target_wavs[indices[0]], rir), seg_samples)

    # Interferers (convolve-then-crop each)
    interferers_reverbed = []
    for _ in range(num_interferers):
        int_spk_wavs = random.choice(interferer_pool)
        if not int_spk_wavs:
            continue
        r = get_rir()
        if r is not None:
            int_wav = random_crop(apply_rir(random.choice(int_spk_wavs), r), seg_samples)
            interferers_reverbed.append(int_wav)

    # Load noise
    noise_wav = None
    for _ in range(10):
        noise_wav = load_wav(random.choice(noise_files))
        if noise_wav is not None:
            break
    if noise_wav is None:
        return None
    noise_wav = loop_to_length(noise_wav, seg_samples)

    # Build mixture (target_wav is already reverbed from convolve-then-crop)
    if target_present:
        speech_sum = target_wav + sum(interferers_reverbed)
        mix_wav = mix_at_snr(speech_sum, noise_wav, snr_db)
        reverbed_target = target_wav
    else:
        if len(interferers_reverbed) == 0:
            speech_sum = np.zeros(seg_samples, dtype=np.float32)
        else:
            speech_sum = sum(interferers_reverbed)
        mix_wav = mix_at_snr(speech_sum, noise_wav, snr_db)
        clean_target = np.zeros(seg_samples, dtype=np.float32)
        reverbed_target = np.zeros(seg_samples, dtype=np.float32)

    # Normalize
    mix_max = np.abs(mix_wav).max()
    if mix_max > 0.95:
        scale = 0.9 / mix_max
        mix_wav = mix_wav * scale
        clean_target = clean_target * scale
        reverbed_target = reverbed_target * scale

    return mix_wav, ref_wav, clean_target, reverbed_target, {
        "target_present": target_present,
        "snr_db": snr_db,
        "num_interferers": num_interferers,
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate HiFi-TSE")
    parser.add_argument("--config", type=str, default="configs/hifi_tse.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=200,
                        help="Number of test mixtures to generate")
    parser.add_argument("--ta-ratio", type=float, default=0.2,
                        help="Fraction of target-absent test samples")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible evaluation")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Save audio samples to this directory")
    parser.add_argument("--save-audio", type=int, default=10,
                        help="Number of audio examples to save (0=none)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seg_seconds = cfg["audio"]["mix_segment_seconds"]
    seg_samples = int(seg_seconds * SAMPLE_RATE)
    min_utt = cfg["audio"]["min_utterance_length"]

    # ---- Load model ----
    print("Loading model...")
    generator = Generator(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    generator.load_state_dict(ckpt["generator"])
    generator.eval()
    step = ckpt.get("step", "unknown")
    print("  Checkpoint: {} (step {})".format(args.checkpoint, step))

    # ---- Load val speakers ----
    print("Loading validation speakers...")
    manifest_path = os.path.join(cfg["data"]["hdf5_dir"],
                                 "manifests", "speaker_manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    val_speaker_ids = manifest["val"]
    all_speaker_files = manifest["all_speakers"]

    # Load utterances for each val speaker
    val_speakers = {}
    for spk_id in val_speaker_ids:
        files = all_speaker_files.get(spk_id, [])
        wavs = load_speaker_files(files, min_utt)
        if len(wavs) >= 2:
            val_speakers[spk_id] = wavs

    print("  {} val speakers with >= 2 utterances".format(len(val_speakers)))
    if len(val_speakers) == 0:
        print("ERROR: No valid val speakers found.")
        return

    # Interferer pool: all val speakers (list of wav lists)
    interferer_pool = list(val_speakers.values())

    # ---- Load noise and RIR file lists ----
    print("Scanning noise and RIR files...")
    noise_files = scan_noise_files(cfg["data"]["noise"])
    rir_files = scan_rir_files(cfg["data"]["rir"])
    print("  {} noise files, {} RIR files".format(len(noise_files), len(rir_files)))

    # ---- Output directory ----
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # ---- Generate and evaluate ----
    print("\nGenerating {} test samples (seed={}, ta_ratio={})...".format(
        args.num_samples, args.seed, args.ta_ratio))

    snr_range = cfg["data"]["snr_range"]
    interferer_range = cfg["data"]["num_interferers_range"]
    val_spk_list = list(val_speakers.keys())

    # Metric accumulators (primary: against reverbed target; secondary: against clean for dereverb tracking)
    tp_metrics = {"si_sdr": [], "si_sdri": [], "pesq": [], "stoi": [],
                  "si_sdr_clean": [], "pesq_clean": [], "stoi_clean": []}
    ta_metrics = {"energy_db": [], "suppression_db": []}
    saved_count = 0

    for i in range(args.num_samples):
        # Pick a random val speaker
        spk_id = random.choice(val_spk_list)
        target_wavs = val_speakers[spk_id]

        # TP or TA
        target_present = random.random() >= args.ta_ratio
        snr_db = random.uniform(*snr_range)
        num_int = random.randint(*interferer_range)

        result = generate_test_sample(
            target_wavs, interferer_pool, noise_files, rir_files,
            seg_samples, target_present, snr_db, num_int)

        if result is None:
            continue

        mix_wav, ref_wav, clean_target, reverbed_target, meta = result

        # Run inference
        mix_t = torch.from_numpy(mix_wav).float().unsqueeze(0).to(device)
        ref_t = torch.from_numpy(ref_wav).float().unsqueeze(0).to(device)
        with torch.no_grad():
            est_t = generator(mix_t, ref_t)
        est_wav = est_t.squeeze(0).cpu().numpy()

        # Compute metrics (primary: against reverbed target)
        if target_present:
            si_sdr_val = compute_si_sdr(est_wav, reverbed_target)
            si_sdr_input = compute_si_sdr(mix_wav, reverbed_target)
            si_sdri = si_sdr_val - si_sdr_input
            tp_metrics["si_sdr"].append(si_sdr_val)
            tp_metrics["si_sdri"].append(si_sdri)

            pesq_val = compute_pesq(est_wav, reverbed_target)
            if pesq_val is not None:
                tp_metrics["pesq"].append(pesq_val)

            stoi_val = compute_stoi(est_wav, reverbed_target)
            if stoi_val is not None:
                tp_metrics["stoi"].append(stoi_val)

            # Secondary metrics: against clean target (dereverberation tracking)
            si_sdr_clean = compute_si_sdr(est_wav, clean_target)
            tp_metrics["si_sdr_clean"].append(si_sdr_clean)
            pesq_clean = compute_pesq(est_wav, clean_target)
            if pesq_clean is not None:
                tp_metrics["pesq_clean"].append(pesq_clean)
            stoi_clean = compute_stoi(est_wav, clean_target)
            if stoi_clean is not None:
                tp_metrics["stoi_clean"].append(stoi_clean)
        else:
            est_energy = compute_energy_db(est_wav)
            mix_energy = compute_energy_db(mix_wav)
            ta_metrics["energy_db"].append(est_energy)
            ta_metrics["suppression_db"].append(mix_energy - est_energy)

        # Save audio examples
        if args.output_dir and saved_count < args.save_audio:
            tag = "tp" if target_present else "ta"
            prefix = os.path.join(args.output_dir,
                                  "{:03d}_{}".format(saved_count, tag))
            sf.write(prefix + "_mix.wav", mix_wav, SAMPLE_RATE)
            sf.write(prefix + "_ref.wav", ref_wav, SAMPLE_RATE)
            sf.write(prefix + "_est.wav", est_wav, SAMPLE_RATE)
            sf.write(prefix + "_target.wav", clean_target, SAMPLE_RATE)
            saved_count += 1

        # Progress
        if (i + 1) % 50 == 0 or i == 0:
            print("  [{}/{}] ...".format(i + 1, args.num_samples))

    # ---- Report ----
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (step {})".format(step))
    print("=" * 60)

    n_tp = len(tp_metrics["si_sdr"])
    n_ta = len(ta_metrics["energy_db"])
    print("\nSamples: {} TP, {} TA".format(n_tp, n_ta))

    if n_tp > 0:
        print("\n--- Target Present (TP) — Primary (vs reverbed target) ---")
        print("  SI-SDR:  {:.2f} dB  (std {:.2f})".format(
            np.mean(tp_metrics["si_sdr"]), np.std(tp_metrics["si_sdr"])))
        print("  SI-SDRi: {:.2f} dB  (std {:.2f})".format(
            np.mean(tp_metrics["si_sdri"]), np.std(tp_metrics["si_sdri"])))
        if tp_metrics["pesq"]:
            print("  PESQ:    {:.2f}      (std {:.2f})".format(
                np.mean(tp_metrics["pesq"]), np.std(tp_metrics["pesq"])))
        if tp_metrics["stoi"]:
            print("  STOI:    {:.3f}     (std {:.3f})".format(
                np.mean(tp_metrics["stoi"]), np.std(tp_metrics["stoi"])))
        # Secondary metrics (vs clean target — dereverberation tracking)
        if tp_metrics["si_sdr_clean"]:
            print("\n  --- Secondary (vs clean target, dereverb tracking) ---")
            print("  SI-SDR (clean):  {:.2f} dB  (std {:.2f})".format(
                np.mean(tp_metrics["si_sdr_clean"]), np.std(tp_metrics["si_sdr_clean"])))
            if tp_metrics["pesq_clean"]:
                print("  PESQ (clean):    {:.2f}      (std {:.2f})".format(
                    np.mean(tp_metrics["pesq_clean"]), np.std(tp_metrics["pesq_clean"])))
            if tp_metrics["stoi_clean"]:
                print("  STOI (clean):    {:.3f}     (std {:.3f})".format(
                    np.mean(tp_metrics["stoi_clean"]), np.std(tp_metrics["stoi_clean"])))

    if n_ta > 0:
        print("\n--- Target Absent (TA) ---")
        print("  Output energy:  {:.1f} dB  (lower = better suppression)".format(
            np.mean(ta_metrics["energy_db"])))
        print("  Suppression:    {:.1f} dB  (mix energy - output energy)".format(
            np.mean(ta_metrics["suppression_db"])))

    # Save results to JSON
    if args.output_dir:
        results = {
            "checkpoint": args.checkpoint,
            "step": step,
            "seed": args.seed,
            "num_samples": args.num_samples,
            "n_tp": n_tp,
            "n_ta": n_ta,
        }
        if n_tp > 0:
            results["tp"] = {
                "si_sdr_mean": float(np.mean(tp_metrics["si_sdr"])),
                "si_sdr_std": float(np.std(tp_metrics["si_sdr"])),
                "si_sdri_mean": float(np.mean(tp_metrics["si_sdri"])),
                "si_sdri_std": float(np.std(tp_metrics["si_sdri"])),
            }
            if tp_metrics["pesq"]:
                results["tp"]["pesq_mean"] = float(np.mean(tp_metrics["pesq"]))
                results["tp"]["pesq_std"] = float(np.std(tp_metrics["pesq"]))
            if tp_metrics["stoi"]:
                results["tp"]["stoi_mean"] = float(np.mean(tp_metrics["stoi"]))
                results["tp"]["stoi_std"] = float(np.std(tp_metrics["stoi"]))
            # Secondary metrics (vs clean target)
            if tp_metrics["si_sdr_clean"]:
                results["tp"]["si_sdr_clean_mean"] = float(np.mean(tp_metrics["si_sdr_clean"]))
                results["tp"]["si_sdr_clean_std"] = float(np.std(tp_metrics["si_sdr_clean"]))
            if tp_metrics["pesq_clean"]:
                results["tp"]["pesq_clean_mean"] = float(np.mean(tp_metrics["pesq_clean"]))
                results["tp"]["pesq_clean_std"] = float(np.std(tp_metrics["pesq_clean"]))
            if tp_metrics["stoi_clean"]:
                results["tp"]["stoi_clean_mean"] = float(np.mean(tp_metrics["stoi_clean"]))
                results["tp"]["stoi_clean_std"] = float(np.std(tp_metrics["stoi_clean"]))
        if n_ta > 0:
            results["ta"] = {
                "energy_db_mean": float(np.mean(ta_metrics["energy_db"])),
                "suppression_db_mean": float(np.mean(ta_metrics["suppression_db"])),
            }
        results_path = os.path.join(args.output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print("\nResults saved to:", results_path)

    if args.output_dir and saved_count > 0:
        print("Audio samples saved to:", args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
