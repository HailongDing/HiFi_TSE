#!/usr/bin/env python3
"""Pre-training sanity check: verify reverb mismatch is fixed.

Loads N samples from the training dataset and computes SI-SDR(mix, target).
- With the fix: target is reverbed, SI-SDR should be ~0 to +5 dB
- Without fix: target is clean, SI-SDR would be ~-25 to -27 dB
"""

import argparse
import sys

import numpy as np
import torch
import yaml

from data.dataset import HiFiTSEDataset
from losses.separation import si_sdr


def main():
    parser = argparse.ArgumentParser(description="Reverb mismatch sanity check")
    parser.add_argument("--config", default="configs/hifi_tse.yaml")
    parser.add_argument("--num-samples", type=int, default=100)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print("Loading dataset...")
    dataset = HiFiTSEDataset(cfg)
    print("  {} speakers, {} samples requested".format(
        len(dataset.clean_index.all_speakers), args.num_samples))

    si_sdr_vals = []
    tp_count = 0
    ta_count = 0

    for i in range(args.num_samples):
        mix_wav, ref_wav, target_wav, tp_flag = dataset[i % len(dataset)]

        if tp_flag.item() < 0.5:
            ta_count += 1
            continue

        tp_count += 1
        # Compute SI-SDR between mixture and target
        mix = mix_wav.unsqueeze(0)
        tgt = target_wav.unsqueeze(0)
        val = si_sdr(mix, tgt).item()
        si_sdr_vals.append(val)

    si_sdr_arr = np.array(si_sdr_vals)
    print("\n" + "=" * 50)
    print("Reverb Mismatch Sanity Check")
    print("=" * 50)
    print("  Samples: {} TP, {} TA".format(tp_count, ta_count))
    print("  SI-SDR(mix, target): {:.2f} dB  (std {:.2f})".format(
        si_sdr_arr.mean(), si_sdr_arr.std()))
    print("  Min: {:.2f} dB, Max: {:.2f} dB".format(
        si_sdr_arr.min(), si_sdr_arr.max()))
    print()

    if si_sdr_arr.mean() > -10.0:
        print("  PASS: Target is aligned with mixture (reverb fix working)")
        print("  Expected range: ~0 to +5 dB")
    else:
        print("  FAIL: Target is misaligned with mixture (reverb mismatch still present)")
        print("  Expected ~0 to +5 dB, got {:.2f} dB".format(si_sdr_arr.mean()))
        sys.exit(1)


if __name__ == "__main__":
    main()
