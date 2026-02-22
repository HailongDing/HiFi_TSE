#!/usr/bin/env python3
"""Test case: target speaker present in only part of the audio.

Creates an 8-second mixture where:
  - Seconds 0-2:   target ABSENT (interferer + noise only)
  - Seconds 2-5.5: target PRESENT (target + interferer + noise)
  - Seconds 5.5-8: target ABSENT (interferer + noise only)

Saves audio files and plots energy over time to visualize the model's behavior.
"""

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


def energy_over_time(wav, frame_ms=50):
    """Compute RMS energy in dB over time with given frame size."""
    frame_samples = int(SAMPLE_RATE * frame_ms / 1000)
    n_frames = len(wav) // frame_samples
    times = []
    energies = []
    for i in range(n_frames):
        chunk = wav[i * frame_samples:(i + 1) * frame_samples]
        rms = np.sqrt(np.mean(chunk ** 2) + 1e-10)
        times.append((i + 0.5) * frame_ms / 1000)
        energies.append(20 * np.log10(rms + 1e-10))
    return np.array(times), np.array(energies)


def main():
    random.seed(77)
    np.random.seed(77)
    torch.manual_seed(77)

    config_path = "configs/hifi_tse.yaml"
    checkpoint_path = "checkpoints/checkpoint_best.pt"
    output_dir = "audios/partial_presence_test"

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print("Loading model...")
    generator = Generator(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(ckpt["generator"])
    generator.eval()
    print("  Loaded step {}".format(ckpt.get("step", "?")))

    # Load val speakers
    manifest_path = os.path.join(cfg["data"]["hdf5_dir"], "manifests", "speaker_manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    val_speaker_ids = manifest["val"]
    all_speaker_files = manifest["all_speakers"]

    min_utt = 8.0
    val_speakers = {}
    for spk_id in val_speaker_ids:
        files = all_speaker_files.get(spk_id, [])
        wavs = load_speaker_files(files, min_utt)
        if len(wavs) >= 2:
            val_speakers[spk_id] = wavs

    print("  {} val speakers available".format(len(val_speakers)))
    val_spk_list = list(val_speakers.keys())

    # Scan noise and RIR
    noise_files = []
    for _, path in cfg["data"]["noise"].items():
        for f in Path(path).rglob("*.wav"):
            noise_files.append(str(f))
    rir_files = []
    for _, path in cfg["data"]["rir"].items():
        for f in Path(path).rglob("*.wav"):
            rir_files.append(str(f))

    def get_rir():
        for _ in range(10):
            r = load_wav(random.choice(rir_files))
            if r is not None:
                return r
        return None

    # ---- Build the test mixture ----
    total_seconds = 8.0
    total_samples = int(total_seconds * SAMPLE_RATE)

    # Target present region: 2.0s to 5.5s
    tp_start = int(2.0 * SAMPLE_RATE)
    tp_end = int(5.5 * SAMPLE_RATE)

    # Pick target speaker and a different interferer speaker
    target_spk = random.choice(val_spk_list)
    interferer_spk = random.choice([s for s in val_spk_list if s != target_spk])

    target_wavs = val_speakers[target_spk]
    interferer_wavs = val_speakers[interferer_spk]

    # Reference: a separate utterance from the target speaker
    ref_wav = random_crop(target_wavs[1], int(5.0 * SAMPLE_RATE))

    # Target: reverbed, only in the middle region
    rir_target = get_rir()
    target_full_utt = target_wavs[0]
    target_reverbed_full = apply_rir(target_full_utt, rir_target, len(target_full_utt))
    target_segment = random_crop(target_reverbed_full, tp_end - tp_start)

    # Create the target track (silence + speech + silence)
    target_track = np.zeros(total_samples, dtype=np.float32)
    target_track[tp_start:tp_end] = target_segment

    # Apply fade in/out (50ms) at boundaries to avoid clicks
    fade_samples = int(0.05 * SAMPLE_RATE)
    fade_in = np.linspace(0, 1, fade_samples, dtype=np.float32)
    fade_out = np.linspace(1, 0, fade_samples, dtype=np.float32)
    target_track[tp_start:tp_start + fade_samples] *= fade_in
    target_track[tp_end - fade_samples:tp_end] *= fade_out

    # Interferer: present throughout the full 8 seconds
    rir_interferer = get_rir()
    int_utt = random_crop(random.choice(interferer_wavs), total_samples)
    interferer_track = random_crop(apply_rir(int_utt, rir_interferer, total_samples + len(rir_interferer)), total_samples)

    # Noise: present throughout
    noise_wav = None
    for _ in range(10):
        noise_wav = load_wav(random.choice(noise_files))
        if noise_wav is not None:
            break
    noise_track = loop_to_length(noise_wav, total_samples)

    # Mix at SNR 5 dB (speech vs noise)
    speech_sum = target_track + interferer_track
    snr_db = 5.0
    mix_wav = mix_at_snr(speech_sum, noise_track, snr_db)

    # Normalize
    mx = np.abs(mix_wav).max()
    if mx > 0.95:
        scale = 0.9 / mx
        mix_wav = (mix_wav * scale).astype(np.float32)
        target_track = (target_track * scale).astype(np.float32)

    # ---- Run inference ----
    print("\nRunning inference on 8s mixture...")
    mix_t = torch.from_numpy(mix_wav).float().unsqueeze(0).to(device)
    ref_t = torch.from_numpy(ref_wav).float().unsqueeze(0).to(device)
    with torch.no_grad():
        est_t = generator(mix_t, ref_t)
    est_wav = est_t.squeeze(0).cpu().numpy()

    # ---- Save audio files ----
    os.makedirs(output_dir, exist_ok=True)
    sf.write(os.path.join(output_dir, "mix.wav"), mix_wav, SAMPLE_RATE)
    sf.write(os.path.join(output_dir, "ref.wav"), ref_wav, SAMPLE_RATE)
    sf.write(os.path.join(output_dir, "target.wav"), target_track, SAMPLE_RATE)
    sf.write(os.path.join(output_dir, "est.wav"), est_wav, SAMPLE_RATE)

    # ---- Analyze energy over time ----
    print("\n--- Energy Analysis (50ms frames) ---")
    print("Target present: {:.1f}s - {:.1f}s".format(tp_start / SAMPLE_RATE, tp_end / SAMPLE_RATE))

    times, est_energy = energy_over_time(est_wav)
    _, target_energy = energy_over_time(target_track)
    _, mix_energy = energy_over_time(mix_wav)

    # Compute average energy in TA and TP regions
    ta_mask = (times < 2.0) | (times > 5.5)
    tp_mask = (times >= 2.0) & (times <= 5.5)

    est_tp_energy = np.mean(est_energy[tp_mask])
    est_ta_energy = np.mean(est_energy[ta_mask])
    mix_tp_energy = np.mean(mix_energy[tp_mask])
    mix_ta_energy = np.mean(mix_energy[ta_mask])
    target_tp_energy = np.mean(target_energy[tp_mask])

    print("\n  Region         | Mix (dB)  | Est (dB)  | Target (dB)")
    print("  ---------------+-----------+-----------+------------")
    print("  TA (0-2s)      | {:>7.1f}   | {:>7.1f}   |   (silence)".format(
        np.mean(mix_energy[times < 2.0]), np.mean(est_energy[times < 2.0])))
    print("  TP (2-5.5s)    | {:>7.1f}   | {:>7.1f}   | {:>7.1f}".format(
        mix_tp_energy, est_tp_energy, target_tp_energy))
    print("  TA (5.5-8s)    | {:>7.1f}   | {:>7.1f}   |   (silence)".format(
        np.mean(mix_energy[times > 5.5]), np.mean(est_energy[times > 5.5])))

    print("\n  TP vs TA contrast in output: {:.1f} dB".format(est_tp_energy - est_ta_energy))
    print("  TA suppression (mix - est):  {:.1f} dB (0-2s), {:.1f} dB (5.5-8s)".format(
        np.mean(mix_energy[times < 2.0]) - np.mean(est_energy[times < 2.0]),
        np.mean(mix_energy[times > 5.5]) - np.mean(est_energy[times > 5.5])))

    # ---- Plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

        # Waveforms
        t_wav = np.arange(total_samples) / SAMPLE_RATE

        axes[0].plot(t_wav, mix_wav, linewidth=0.3, color="gray")
        axes[0].set_ylabel("Mix")
        axes[0].set_title("Partial Presence Test: Target present 2.0s - 5.5s")
        axes[0].axvspan(2.0, 5.5, alpha=0.15, color="green", label="Target present")
        axes[0].legend(loc="upper right", fontsize=8)

        axes[1].plot(t_wav, target_track, linewidth=0.3, color="blue")
        axes[1].set_ylabel("Target (GT)")
        axes[1].axvspan(2.0, 5.5, alpha=0.15, color="green")

        axes[2].plot(t_wav, est_wav, linewidth=0.3, color="red")
        axes[2].set_ylabel("Estimated")
        axes[2].axvspan(2.0, 5.5, alpha=0.15, color="green")

        # Energy over time
        axes[3].plot(times, mix_energy, label="Mix", color="gray", alpha=0.6)
        axes[3].plot(times, target_energy, label="Target", color="blue", alpha=0.8)
        axes[3].plot(times, est_energy, label="Estimated", color="red", linewidth=1.5)
        axes[3].axvspan(2.0, 5.5, alpha=0.15, color="green")
        axes[3].set_ylabel("Energy (dB)")
        axes[3].set_xlabel("Time (s)")
        axes[3].legend(loc="upper right", fontsize=8)
        axes[3].set_ylim([-80, 0])

        plt.tight_layout()
        plot_path = os.path.join(output_dir, "analysis.png")
        plt.savefig(plot_path, dpi=150)
        print("\n  Plot saved to:", plot_path)
    except ImportError:
        print("\n  (matplotlib not available, skipping plot)")

    print("\nAudio files saved to:", output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
