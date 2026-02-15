#!/usr/bin/env python3
"""HiFi-TSE single-file inference.

Run with the USEF-TFGridNet conda environment:
    conda run -n USEF-TFGridNet python inference.py \
        --config configs/hifi_tse.yaml \
        --checkpoint checkpoints/checkpoint_final.pt \
        --mix /path/to/mixture.wav \
        --ref /path/to/reference.wav \
        -o /path/to/output.wav

For long files, use --chunk-seconds to process in overlapping chunks.
"""

import argparse

import torch
import torchaudio
import yaml

from models.generator import Generator


def load_audio(path, sr=48000):
    """Load audio file, force mono, resample if needed."""
    waveform, orig_sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if orig_sr != sr:
        waveform = torchaudio.functional.resample(waveform, orig_sr, sr)
    return waveform  # (1, L)


def inference_full(generator, mix_wav, ref_wav, device):
    """Run full-length inference (no chunking)."""
    mix_wav = mix_wav.to(device)
    ref_wav = ref_wav.to(device)
    with torch.no_grad():
        est_wav = generator(mix_wav, ref_wav)
    return est_wav.cpu()


def inference_chunked(generator, mix_wav, ref_wav, device, chunk_samples, hop_samples):
    """Run inference with overlapping chunks for long files.

    Uses overlap-add with a triangular (Bartlett) window for smooth transitions.

    Args:
        generator: Generator model
        mix_wav: (1, L) mixture waveform
        ref_wav: (1, L_ref) reference waveform
        device: torch device
        chunk_samples: chunk size in samples
        hop_samples: hop between chunks in samples
    """
    L = mix_wav.shape[1]
    ref_wav = ref_wav.to(device)

    output = torch.zeros(1, L)
    weight = torch.zeros(1, L)

    # Triangular window for overlap-add
    win = torch.bartlett_window(chunk_samples, dtype=torch.float32)

    start = 0
    while start < L:
        end = min(start + chunk_samples, L)
        chunk = mix_wav[:, start:end]

        # Pad last chunk if shorter
        if chunk.shape[1] < chunk_samples:
            pad_len = chunk_samples - chunk.shape[1]
            chunk = torch.nn.functional.pad(chunk, (0, pad_len))

        chunk = chunk.to(device)
        with torch.no_grad():
            est_chunk = generator(chunk, ref_wav).cpu()

        actual_len = min(chunk_samples, L - start)
        w = win[:actual_len]
        output[:, start:start + actual_len] += est_chunk[:, :actual_len] * w
        weight[:, start:start + actual_len] += w

        start += hop_samples

    # Normalize by accumulated weight
    output = output / weight.clamp(min=1e-8)
    return output


def main():
    parser = argparse.ArgumentParser(description="HiFi-TSE Inference")
    parser.add_argument("--config", type=str, default="configs/hifi_tse.yaml")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to generator checkpoint (.pt)")
    parser.add_argument("--mix", type=str, required=True,
                        help="Path to mixture audio file")
    parser.add_argument("--ref", type=str, required=True,
                        help="Path to reference speaker audio file")
    parser.add_argument("-o", "--output", type=str, default="output.wav",
                        help="Output file path")
    parser.add_argument("--chunk-seconds", type=float, default=0,
                        help="Process in chunks of this many seconds (0=no chunking)")
    parser.add_argument("--overlap", type=float, default=0.5,
                        help="Overlap ratio between chunks (default 0.5)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    sr = cfg["audio"]["sample_rate"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load model
    generator = Generator(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    generator.load_state_dict(ckpt["generator"])
    generator.eval()
    print("Loaded checkpoint:", args.checkpoint)

    # Load audio
    mix_wav = load_audio(args.mix, sr)  # (1, L)
    ref_wav = load_audio(args.ref, sr)  # (1, L_ref)
    print("Mix: {} ({:.2f}s)".format(args.mix, mix_wav.shape[1] / sr))
    print("Ref: {} ({:.2f}s)".format(args.ref, ref_wav.shape[1] / sr))

    # Inference
    if args.chunk_seconds > 0:
        chunk_samples = int(args.chunk_seconds * sr)
        overlap = max(0.0, min(args.overlap, 0.99))
        hop_samples = max(1, int(chunk_samples * (1.0 - overlap)))
        print("Chunked inference: {:.1f}s chunks, {:.1f}s hop".format(
            chunk_samples / sr, hop_samples / sr))
        est_wav = inference_chunked(generator, mix_wav, ref_wav, device,
                                    chunk_samples, hop_samples)
    else:
        print("Full-length inference")
        est_wav = inference_full(generator, mix_wav, ref_wav, device)

    # Save output
    torchaudio.save(args.output, est_wav, sr)
    print("Saved: {} ({:.2f}s)".format(args.output, est_wav.shape[1] / sr))


if __name__ == "__main__":
    main()
