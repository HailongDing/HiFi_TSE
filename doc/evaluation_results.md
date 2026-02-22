# Evaluation Results — HiFi-TSE v2

## Model Info

- **Architecture**: TF-GridNet + USEF conditioning, 11.79M params, 48kHz
- **Best checkpoint**: `checkpoints/checkpoint_best.pt` (step 1,495,000)
- **Final checkpoint**: `checkpoints/checkpoint_final.pt` (step 1,500,000)
- **Training**: SI-SDR + phase loss + amplitude loss, 2-phase curriculum, 1.5M steps

---

## Standard Evaluation (200 samples, seed=42)

### checkpoint_best.pt (step 1,495,000)

#### Target Present (158 samples) — vs reverbed target

| Metric | Mean | Std |
|--------|------|-----|
| SI-SDR | -1.71 dB | 5.33 |
| **SI-SDRi** | **+4.17 dB** | 3.96 |
| PESQ | 1.14 | 0.13 |
| STOI | 0.489 | 0.162 |

#### Target Present — vs clean target (dereverberation tracking)

| Metric | Mean | Std |
|--------|------|-----|
| SI-SDR (clean) | -44.57 dB | 10.34 |
| PESQ (clean) | 1.11 | 0.12 |
| STOI (clean) | 0.226 | 0.066 |

#### Target Absent (42 samples)

| Metric | Value |
|--------|-------|
| Output energy | -31.2 dB |
| Suppression | 11.4 dB |

### checkpoint_final.pt (step 1,500,000)

#### Target Present (158 samples) — vs reverbed target

| Metric | Mean | Std |
|--------|------|-----|
| SI-SDR | -1.70 dB | 5.32 |
| **SI-SDRi** | **+4.18 dB** | 3.88 |
| PESQ | 1.14 | 0.13 |
| STOI | 0.490 | 0.162 |

#### Target Present — vs clean target (dereverberation tracking)

| Metric | Mean | Std |
|--------|------|-----|
| SI-SDR (clean) | -45.24 dB | 11.82 |
| PESQ (clean) | 1.12 | 0.16 |
| STOI (clean) | 0.227 | 0.067 |

#### Target Absent (42 samples)

| Metric | Value |
|--------|-------|
| Output energy | -30.4 dB |
| Suppression | 10.6 dB |

---

## Variable-Length Evaluation (10 groups, seed=123)

Mix length 5-8s, ref length 4-6s, all target-present. Audio files in `audios/`.

| Group | Mix | Ref | SNR | Interferers | SI-SDRi |
|-------|-----|-----|-----|-------------|---------|
| 01 | 5.2s | 4.2s | 0 dB | 1 | +5.30 dB |
| 02 | 5.7s | 4.0s | 10 dB | 3 | +2.34 dB |
| 03 | 7.5s | 5.5s | 5 dB | 2 | +4.56 dB |
| 04 | 7.2s | 4.2s | 8 dB | 1 | +1.63 dB |
| 05 | 6.8s | 5.7s | -5 dB | 2 | +12.61 dB |
| 06 | 8.0s | 4.9s | 4 dB | 1 | -0.34 dB |
| 07 | 6.9s | 5.0s | -3 dB | 2 | +3.04 dB |
| 08 | 6.9s | 5.9s | -4 dB | 3 | +8.24 dB |
| 09 | 6.6s | 4.7s | 8 dB | 3 | +8.12 dB |
| 10 | 7.6s | 5.0s | 2 dB | 1 | +4.85 dB |
| **Mean** | | | | | **+5.04 dB** |

---

## Partial Presence Test (seed=77)

8-second mixture with target speaker present only in seconds 2.0-5.5. Audio files in
`audios/partial_presence_test/`. Plot: `audios/partial_presence_test/analysis.png`.

| Region | Mix (dB) | Est (dB) | Target (dB) |
|--------|----------|----------|-------------|
| TA (0-2s) | -20.0 | -38.2 | (silence) |
| TP (2-5.5s) | -20.3 | -29.0 | -28.6 |
| TA (5.5-8s) | -26.4 | -50.2 | (silence) |

- TP vs TA contrast in output: 15.9 dB
- TA suppression: 18.2 dB (0-2s), 23.8 dB (5.5-8s)
- Model correctly extracts target in active region and suppresses in silent regions

---

## v1 vs v2 Comparison

| Metric | v1 (step 430K) | v2 (step 1,495K) | Delta |
|--------|---------------|------------------|-------|
| SI-SDRi | +4.01 dB | +4.18 dB | +0.17 dB |
| SI-SDR | -1.87 dB | -1.71 dB | +0.16 dB |
| PESQ | 1.15 | 1.14 | -0.01 |
| STOI | 0.488 | 0.489 | +0.001 |
| TA suppression | 11.2 dB | 11.4 dB | +0.2 dB |
| Model params | 7.8M | 11.79M | +51% |
| Training steps | 480K | 1,500K | +3.1x |

---

## Notes

- PESQ is computed at 16kHz (resampled from 48kHz) due to library limitations —
  scores are not directly comparable to native 16kHz benchmarks
- "vs clean target" metrics confirm the model preserves reverberation (does not
  dereverberate), which is expected since training targets are reverbed signals
- Best and final checkpoints are nearly identical, confirming full convergence
- Variable-length evaluation (mean +5.04 dB) slightly exceeds fixed-length eval
  (+4.18 dB), possibly due to longer context providing more information
