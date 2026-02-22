# Performance Improvement Plan — HiFi-TSE v2

## Current Baseline (Step 1,495K / 1.5M — Training Complete)

- **Model**: 11.79M params, TF-GridNet + USEF conditioning, 48kHz
- **SI-SDRi**: +4.18 dB (200-sample eval, seed=42)
- **SI-SDR**: -1.71 dB (std 5.33)
- **PESQ**: 1.14 (std 0.13)
- **STOI**: 0.489 (std 0.162)
- **TA suppression**: -31.2 dB output energy (11.4 dB suppression)

---

## 1. GAN Fine-Tuning (Post-Training)

**What**: Add a HiFi-GAN discriminator (multi-period + multi-scale) and fine-tune the
converged generator for 100-200K additional steps with adversarial + feature matching
losses at low LR (1e-5).

**Why it helps**: SI-SDR optimizes waveform-level accuracy but doesn't capture perceptual
quality. GAN training pushes the output toward natural-sounding speech distributions —
sharper harmonics, less "mushy" artifacts. v1 showed GAN Phase 3 added +0.5 dB SI-SDR
on top of perceptual improvement.

**Risk**: Medium. GAN training can be unstable. Use low LR, freeze batch norm, and keep
the pre-GAN checkpoint as a safe fallback.

**Expected gain**: +0.3 to +0.8 dB SI-SDRi, significant PESQ/STOI improvement.

---

## 2. Increased Model Capacity

**What**: Scale the model up:
- Option A: `lstm_hidden` 256 -> 384 (~18M params)
- Option B: `num_gridnet_blocks` 6 -> 8 (~16M params)
- Option C: Both (~24M params, would need gradient checkpointing tuning)

**Why it helps**: The current model is relatively small. X-TF-GridNet achieves 19.7 dB
SI-SDRi on WSJ0-2mix with a larger model. More capacity = better modeling of complex
multi-speaker + noise + reverb interactions. The 48kHz task has 1025 frequency bins —
far more spectral detail to model than 8/16kHz benchmarks.

**Risk**: Requires retraining from scratch (1.5M steps, ~4 days). GPU memory: current
usage is 6.5/24.5 GB, so there's headroom for ~2x model size.

**Expected gain**: +0.5 to +1.5 dB SI-SDRi per scaling step.

---

## 3. Training on Longer Segments

**What**: Increase `mix_segment_seconds` from 4.0 to 6.0 or 8.0 seconds.

**Why it helps**: The partial presence test showed the model generalizes to 8s inputs,
but it was only trained on 4s segments. Longer training segments expose the model to
more temporal context — longer silences, speaker turns, natural speech rhythm. This
directly improves the model's ability to handle real-world scenarios (partial speaker
presence, transitions).

**Risk**: Higher memory usage per sample. At 8s the batch may need to drop to 1 with
more grad accumulation, or need gradient checkpointing tuning. Also requires retraining.

**Expected gain**: +0.3 to +0.8 dB SI-SDRi, especially on longer test utterances.

---

## 4. Data Augmentation Enhancements

**What**: Three additions:
- **SpecAugment on input mixture**: randomly mask frequency bands and time steps,
  forcing the model to be robust to missing information
- **Random EQ on reference audio**: apply random equalization to enrollment clips,
  simulating varying recording conditions
- **Multi-enrollment training**: provide 2-3 reference utterances (average embeddings)
  — at inference, use multiple enrollments for better speaker representation

**Why it helps**: The model currently sees clean reference audio most of the time
(noisy_ref_prob=0.5, and only light corruption). Real-world enrollment audio may be
from different devices, rooms, and quality levels. More aggressive augmentation improves
generalization.

**Risk**: Low. These are additive changes to the data pipeline.

**Expected gain**: +0.2 to +0.5 dB SI-SDRi; multi-enrollment can add +0.3 to +0.5 dB
at inference.

---

## 5. Curriculum Phase 3 — Hard Conditions

**What**: Add a third curriculum phase after 1M steps with harder conditions:
- Lower SNR range: [-10, 10] dB (from [-5, 15])
- More interferers: [1, 4] or [2, 5] (from [1, 3])
- Higher TA ratio: 0.3 (from 0.2)

**Why it helps**: The model has already learned the basics by 1M steps. Exposing it to
harder conditions in the final phase pushes performance on difficult cases — the long
tail of the SI-SDRi distribution (the std of 3.96 dB suggests some samples are much
harder than others).

**Risk**: Low. Easy to implement. Can hurt average metrics if pushed too hard, but
careful tuning prevents this.

**Expected gain**: +0.2 to +0.5 dB SI-SDRi, with larger gains on hard test cases.

---

## 6. Loss Function Improvements

**What**:
- **Multi-resolution SI-SDR**: Compute SI-SDR at multiple STFT window sizes (e.g.,
  512, 1024, 2048) to capture both fine temporal detail and coarse spectral structure
- **Learnable loss weighting**: Use uncertainty-based weighting (Kendall et al.) to
  automatically balance `lambda_sep`, `lambda_phase`, `lambda_amp` instead of fixed
  weights
- **Contrastive speaker loss**: Auxiliary loss on speaker embeddings to improve target
  vs interferer discrimination

**Why it helps**: The current loss is a hand-tuned weighted sum. Multi-resolution SI-SDR
captures structure that single-resolution misses. Auto-weighting removes the guesswork
of lambda tuning.

**Risk**: Medium. Multi-resolution SI-SDR is well-established. Contrastive loss adds
complexity.

**Expected gain**: +0.3 to +0.8 dB SI-SDRi.

---

## 7. Stronger Speaker Conditioning

**What**:
- **Speaker-conditioned LayerNorm (AdaLN)**: modulate normalization parameters with
  speaker embedding at each layer, providing finer-grained conditioning
- **Multi-scale reference encoding**: extract features from enrollment at multiple time
  scales, not just a single cross-attention pass

**Why it helps**: Better speaker conditioning = better discrimination between target and
interferers with similar voices. Current USEF cross-attention + FiLM is effective but a
single mechanism. AdaLN adds per-channel modulation at every normalization point.

**Risk**: Medium. Requires architecture changes and retraining.

**Expected gain**: +0.3 to +0.7 dB SI-SDRi.

---

## 8. Data Loading Stage 2 (Throughput)

**What**: The deferred infrastructure optimizations:
- Replace `scipy.signal.resample` with `torchaudio.functional.resample` (faster speed
  perturbation)
- Replace `scipy.signal.fftconvolve` with `scipy.signal.oaconvolve` (faster RIR
  convolution)
- Increase `batch_size` 2->4, reduce `grad_accum` 32->16 (same effective batch, fewer
  DataLoader round-trips)

**Why it helps**: Faster training throughput means more experiments per day. Doesn't
directly improve model quality, but enables faster iteration on all the above methods.

**Risk**: Low for oaconvolve (mathematically identical). Low-medium for torchaudio
resample (different filter). Medium for batch size change (different micro-batch
dynamics).

**Expected gain**: 20-40% faster training throughput.

---

## Summary — Ranked by Impact vs Effort

| # | Method                  | Expected SI-SDRi Gain | Effort | Requires Retrain |
|---|-------------------------|----------------------|--------|-----------------|
| 1 | GAN fine-tuning         | +0.3 to +0.8 dB     | Medium | No (fine-tune)  |
| 2 | Larger model            | +0.5 to +1.5 dB     | High   | Yes             |
| 3 | Longer segments         | +0.3 to +0.8 dB     | Medium | Yes             |
| 4 | Data augmentation       | +0.2 to +0.5 dB     | Low    | Yes             |
| 5 | Curriculum Phase 3      | +0.2 to +0.5 dB     | Low    | Yes             |
| 6 | Loss improvements       | +0.3 to +0.8 dB     | Medium | Yes             |
| 7 | Stronger conditioning   | +0.3 to +0.7 dB     | High   | Yes             |
| 8 | Data loading Stage 2    | Throughput only      | Low    | No              |

The quickest win is **#1 GAN fine-tuning** — it builds directly on the current checkpoint
with no retraining. For the next full training run, combining **#2 + #3 + #4 + #5**
would give the largest compounded gain.
