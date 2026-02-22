#!/usr/bin/env python3
"""Generate 10 groups of evaluation audio files with variable lengths.

Each group contains:
  - mix.wav:    mixture (5-8 seconds)
  - ref.wav:    reference enrollment (4-6 seconds)
  - target.wav: ground truth reverbed target
  - est.wav:    model's estimated output

Run with:
    conda run -n USEF-TFGridNet python generate_eval_audios.py \
        --config configs/hifi_tse.yaml \
        --checkpoint checkpoints/checkpoint_best.pt
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


def load_wav(path):
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        return None
    return data


def random_crop(wav, length):
    if len(wav) >= length:
        offset = random.randint(0, len(wav) - length)
        return wav[offset:offset + length]
    out = np.zeros(length, dtype=wav.dtype)
    out[:len(wav)] = wav
    return out


def loop_to_length(wav, length):
    if len(wav) >= length:
        offset = random.randint(0, len(wav) - length)
        return wav[offset:offset + length]
    reps = (length // len(wav)) + 1
    looped = np.tile(wav, reps)
    offset = random.randint(0, len(looped) - length)
    return looped[offset:offset + length]


def apply_rir(wav, rir, crop_length=None):
    rir = rir / (np.linalg.norm(rir) + 1e-8)
    out = fftconvolve(wav, rir, mode="full")
    if crop_length is not None:
        out = out[:crop_length]
    return out


def mix_at_snr(signal, noise, snr_db):
    sig_power = np.mean(signal ** 2) + 1e-8
    noise_power = np.mean(noise ** 2) + 1e-8
    snr_linear = 10.0 ** (snr_db / 10.0)
    scale = np.sqrt(sig_power / (noise_power * snr_linear + 1e-8))
    return signal + scale * noise


def load_speaker_files(speaker_files, min_length):
    min_samples = int(min_length * SAMPLE_RATE)
    wavs = []
    for path in speaker_files:
        wav = load_wav(path)
        if wav is not None and len(wav) >= min_samples:
            wavs.append(wav)
    return wavs


def scan_noise_files(noise_cfg):
    files = []
    for source_name, path in noise_cfg.items():
        for f in Path(path).rglob("*.wav"):
            files.append(str(f))
    return files


def scan_rir_files(rir_cfg):
    files = []
    for source_name, path in rir_cfg.items():
        for f in Path(path).rglob("*.wav"):
            files.append(str(f))
    return files


def generate_sample(target_wavs, interferer_pool, noise_files, rir_files,
                    mix_samples, ref_samples, snr_db, num_interferers):
    """Generate one TP sample with specified mix and ref lengths."""
    if len(target_wavs) < 2:
        return None

    indices = random.sample(range(len(target_wavs)), 2)
    ref_wav = random_crop(target_wavs[indices[1]], ref_samples)

    def get_rir():
        for _ in range(10):
            rir = load_wav(random.choice(rir_files))
            if rir is not None:
                return rir
        return None

    rir = get_rir()
    if rir is None:
        return None

    clean_target = random_crop(target_wavs[indices[0]], mix_samples)
    target_reverbed = random_crop(apply_rir(target_wavs[indices[0]], rir), mix_samples)

    # Interferers
    interferers_reverbed = []
    for _ in range(num_interferers):
        int_spk_wavs = random.choice(interferer_pool)
        if not int_spk_wavs:
            continue
        r = get_rir()
        if r is not None:
            int_wav = random_crop(apply_rir(random.choice(int_spk_wavs), r), mix_samples)
            interferers_reverbed.append(int_wav)

    # Noise
    noise_wav = None
    for _ in range(10):
        noise_wav = load_wav(random.choice(noise_files))
        if noise_wav is not None:
            break
    if noise_wav is None:
        return None
    noise_wav = loop_to_length(noise_wav, mix_samples)

    # Build mixture
    speech_sum = target_reverbed + sum(interferers_reverbed)
    mix_wav = mix_at_snr(speech_sum, noise_wav, snr_db)

    # Normalize
    mix_max = np.abs(mix_wav).max()
    if mix_max > 0.95:
        scale = 0.9 / mix_max
        mix_wav = mix_wav * scale
        target_reverbed = target_reverbed * scale

    return mix_wav.astype(np.float32), ref_wav.astype(np.float32), target_reverbed.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/hifi_tse.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_best.pt")
    parser.add_argument("--output-dir", type=str, default="audios")
    parser.add_argument("--num-groups", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print("Loading model...")
    generator = Generator(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    generator.load_state_dict(ckpt["generator"])
    generator.eval()
    step = ckpt.get("step", "unknown")
    print("  Checkpoint: {} (step {})".format(args.checkpoint, step))

    # Load val speakers
    print("Loading validation speakers...")
    manifest_path = os.path.join(cfg["data"]["hdf5_dir"], "manifests", "speaker_manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    val_speaker_ids = manifest["val"]
    all_speaker_files = manifest["all_speakers"]

    # Need longer utterances since mix can be up to 8s
    min_utt_length = 8.0
    val_speakers = {}
    for spk_id in val_speaker_ids:
        files = all_speaker_files.get(spk_id, [])
        wavs = load_speaker_files(files, min_utt_length)
        if len(wavs) >= 2:
            val_speakers[spk_id] = wavs

    print("  {} val speakers with >= 2 utterances (>= {:.0f}s)".format(
        len(val_speakers), min_utt_length))
    if len(val_speakers) == 0:
        print("ERROR: No valid val speakers found. Trying with shorter min length...")
        min_utt_length = 5.0
        for spk_id in val_speaker_ids:
            files = all_speaker_files.get(spk_id, [])
            wavs = load_speaker_files(files, min_utt_length)
            if len(wavs) >= 2:
                val_speakers[spk_id] = wavs
        print("  {} val speakers with >= 2 utterances (>= {:.0f}s)".format(
            len(val_speakers), min_utt_length))

    interferer_pool = list(val_speakers.values())
    val_spk_list = list(val_speakers.keys())

    # Noise and RIR
    print("Scanning noise and RIR files...")
    noise_files = scan_noise_files(cfg["data"]["noise"])
    rir_files = scan_rir_files(cfg["data"]["rir"])
    print("  {} noise files, {} RIR files".format(len(noise_files), len(rir_files)))

    snr_range = cfg["data"]["snr_range"]
    interferer_range = cfg["data"]["num_interferers_range"]

    os.makedirs(args.output_dir, exist_ok=True)

    print("\nGenerating {} groups...".format(args.num_groups))
    generated = 0
    attempts = 0

    while generated < args.num_groups and attempts < args.num_groups * 20:
        attempts += 1

        # Random lengths for this group
        mix_seconds = random.uniform(5.0, 8.0)
        ref_seconds = random.uniform(4.0, 6.0)
        mix_samples = int(mix_seconds * SAMPLE_RATE)
        ref_samples = int(ref_seconds * SAMPLE_RATE)

        spk_id = random.choice(val_spk_list)
        target_wavs = val_speakers[spk_id]
        snr_db = random.uniform(*snr_range)
        num_int = random.randint(*interferer_range)

        result = generate_sample(
            target_wavs, interferer_pool, noise_files, rir_files,
            mix_samples, ref_samples, snr_db, num_int)

        if result is None:
            continue

        mix_wav, ref_wav, target_wav = result

        # Run inference
        mix_t = torch.from_numpy(mix_wav).float().unsqueeze(0).to(device)
        ref_t = torch.from_numpy(ref_wav).float().unsqueeze(0).to(device)
        with torch.no_grad():
            est_t = generator(mix_t, ref_t)
        est_wav = est_t.squeeze(0).cpu().numpy()

        # Save
        group_dir = os.path.join(args.output_dir, "group_{:02d}".format(generated + 1))
        os.makedirs(group_dir, exist_ok=True)

        sf.write(os.path.join(group_dir, "mix.wav"), mix_wav, SAMPLE_RATE)
        sf.write(os.path.join(group_dir, "ref.wav"), ref_wav, SAMPLE_RATE)
        sf.write(os.path.join(group_dir, "target.wav"), target_wav, SAMPLE_RATE)
        sf.write(os.path.join(group_dir, "est.wav"), est_wav, SAMPLE_RATE)

        # Compute SI-SDRi for this sample
        def si_sdr(est, ref):
            est = est - np.mean(est)
            ref = ref - np.mean(ref)
            dot = np.sum(ref * est)
            s_energy = np.sum(ref ** 2) + 1e-8
            proj = dot * ref / s_energy
            noise = est - proj
            return 10.0 * np.log10(np.sum(proj ** 2) + 1e-8) - 10.0 * np.log10(np.sum(noise ** 2) + 1e-8)

        si_sdr_out = si_sdr(est_wav, target_wav)
        si_sdr_in = si_sdr(mix_wav, target_wav)
        si_sdri = si_sdr_out - si_sdr_in

        generated += 1
        print("  Group {:02d}: mix {:.1f}s, ref {:.1f}s, SNR {:.0f}dB, {} interferers, SI-SDRi {:.2f} dB".format(
            generated, mix_seconds, ref_seconds, snr_db, num_int, si_sdri))

    print("\nDone. {} groups saved to: {}".format(generated, args.output_dir))


if __name__ == "__main__":
    main()
