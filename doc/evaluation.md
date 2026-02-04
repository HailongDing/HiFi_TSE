# Evaluation Strategy

## 1. Training Curve Monitoring

Tensorboard logs are written to `checkpoints/logs/` during training. Key
scalars to monitor:

| Scalar | What to look for |
|--------|-----------------|
| `train/loss_sep` | Steady decrease in phase 1; stable through phase 2 TA introduction |
| `train/loss_stft` | Should decrease alongside separation loss |
| `train/loss_D` | Should stabilize (not collapse to 0 or diverge) in phase 3 |
| `train/loss_adv` | Generator adversarial loss; should decrease as G improves |
| `train/loss_fm` | Feature matching loss; should decrease |
| `val/scene_aware_loss` | Validation loss; should track training loss without large gap |
| `train/lr` | Cosine decay from 2e-4 to 0 over 500k steps |

Launch tensorboard:

```bash
conda run -n USEF-TFGridNet tensorboard --logdir=checkpoints/logs --port=6006
```

Warning signs:
- `loss_D` collapsing to 0 → discriminator is too strong, generator can't learn
- `loss_D` diverging → discriminator gradient contamination (fixed in code review)
- `loss_sep` spiking at phase 2 transition → normal, should recover within 1-2k steps
- Large train/val gap → overfitting (unlikely with 215k utterances and dynamic mixing)

## 2. Objective Metrics

The primary evaluation uses `evaluate.py`, which generates test mixtures from
the 107 held-out validation speakers and computes standard metrics.

### Metrics

**Target-present (TP) samples:**

| Metric | Range | Description |
|--------|-------|-------------|
| SI-SDR | dB (higher=better) | Scale-invariant signal-to-distortion ratio. Primary metric for extraction quality. |
| SI-SDRi | dB (higher=better) | SI-SDR improvement over input mixture. Measures how much the model helps vs raw input. |
| PESQ | 1.0-4.5 (higher=better) | ITU P.862 perceptual speech quality. Computed at 16kHz (resampled). Wideband mode. |
| STOI | 0.0-1.0 (higher=better) | Short-time objective intelligibility. Measures whether speech remains understandable. |

**Target-absent (TA) samples:**

| Metric | Range | Description |
|--------|-------|-------------|
| Output energy | dB (lower=better) | RMS energy of the model output. Should be very low (silence). |
| Suppression | dB (higher=better) | Difference between mixture energy and output energy. Measures how well the model suppresses non-target audio. |

### Running evaluation

After training (or at any checkpoint):

```bash
conda run -n USEF-TFGridNet python evaluate.py \
    --config configs/hifi_tse.yaml \
    --checkpoint checkpoints/checkpoint_final.pt \
    --num-samples 200 \
    --output-dir eval_results/
```

Options:
- `--num-samples 200`: Number of test mixtures (default 200)
- `--ta-ratio 0.2`: Fraction of target-absent samples (default 0.2)
- `--seed 42`: Random seed for reproducible results
- `--output-dir eval_results/`: Save metrics JSON + audio examples
- `--save-audio 10`: Number of audio examples to save for listening

The script saves:
- `results.json`: All metric values
- `NNN_tp_mix.wav`, `NNN_tp_est.wav`, `NNN_tp_target.wav`, `NNN_tp_ref.wav`:
  Audio examples for manual listening

### Test data

Evaluation uses 107 held-out speakers (97 DNS4, 5 EARS, 5 VCTK) that were
excluded from training HDF5 files. Test mixtures are generated from raw wav
files using the same mixing pipeline as training (RIR convolution, noise
mixing at -5 to 15 dB SNR, 1-3 interferers). A fixed random seed ensures
reproducibility.

### Baseline comparison

To establish whether the model is actually learning, compute SI-SDR of the
raw mixture vs clean target (the "input SI-SDR"). SI-SDRi = output SI-SDR
minus input SI-SDR. A positive SI-SDRi means the model improves over doing
nothing.

### Checkpoint comparison

Run evaluation at multiple checkpoints to track improvement:

```bash
for ckpt in checkpoints/checkpoint_*.pt; do
    echo "=== $ckpt ==="
    conda run -n USEF-TFGridNet python evaluate.py \
        --config configs/hifi_tse.yaml \
        --checkpoint "$ckpt" \
        --num-samples 100
done
```

## 3. Subjective Listening Tests

Objective metrics don't capture everything — especially high-frequency
naturalness and artifacts that matter at 48kHz. After objective evaluation:

1. **Spot-check audio examples** from `eval_results/` directory. Listen for:
   - Target voice clarity and naturalness
   - Residual interferer or noise leakage
   - High-frequency artifacts (8-24 kHz)
   - Silence quality in TA samples

2. **Test on real-world audio** using `inference.py`:
   ```bash
   conda run -n USEF-TFGridNet python inference.py \
       --config configs/hifi_tse.yaml \
       --checkpoint checkpoints/checkpoint_final.pt \
       --mix real_mixture.wav \
       --ref speaker_reference.wav \
       -o extracted.wav
   ```

3. **A/B preference test** (if resources allow): Present listeners with
   pairs of (model output, clean target) and ask which sounds more natural.
   This is the gold standard for perceptual quality.

## 4. What Good Results Look Like

Rough benchmarks for a well-trained 48kHz TSE system:

| Metric | Reasonable | Good |
|--------|-----------|------|
| SI-SDRi | > 8 dB | > 12 dB |
| PESQ | > 2.5 | > 3.0 |
| STOI | > 0.85 | > 0.92 |
| TA suppression | > 15 dB | > 25 dB |

These are approximate targets based on published results for similar
(mostly 16kHz) systems. 48kHz evaluation may show slightly lower PESQ since
the metric is computed at 16kHz after resampling and doesn't capture
high-frequency improvements.
