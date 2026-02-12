# HiFi_TSE — Deep Code Analysis & Potential Issues

## Project Overview

HiFi_TSE is a **Target Speaker Extraction** system operating at **48kHz** that combines a Band-Split Encoder, USEF (Universal Speaker Embedding Free) cross-attention conditioning, a TF-GridNet backbone, and HiFi-GAN-style adversarial training with a 3-phase curriculum. The pipeline is:

```
mix_wav → STFT → BandSplitEncoder → Z_mix ─┐
                                             ├→ USEF → TFGridNet → BandMergeDecoder → est_wav
ref_wav → STFT → BandSplitEncoder → Z_ref ─┘
                    (shared weights)
```

---

## 🔴 Critical Issues

### 1. Incorrect Masking in BandMergeDecoder (Semantic Bug)

**File:** `models/band_split.py`, `BandMergeDecoder.forward()`

The decoder applies the mask as `mask * mix_spec` where both tensors are `(B, 2, T, F)` with dim=1 representing `[real, imag]`. This performs **element-wise multiplication of the stacked real/imaginary channels independently**:

```
output_real = mask_real × mix_real
output_imag = mask_imag × mix_imag
```

This is **not** a complex ratio mask (cRM). A proper complex mask would be:
```
output_real = mask_real × mix_real − mask_imag × mix_imag
output_imag = mask_real × mix_imag + mask_imag × mix_real
```

The current approach cannot model any phase rotation — it can only scale the real and imaginary parts independently. This severely limits the model's ability to reconstruct phase-accurate speech, which is particularly harmful at 48kHz where high-frequency harmonics are phase-sensitive. This likely caps the achievable PESQ and SI-SDR significantly.

**Fix:** Replace `masked_spec = mask * mix_spec` with proper complex multiplication:
```python
mask_r, mask_i = mask[:, 0], mask[:, 1]
mix_r, mix_i = mix_spec[:, 0], mix_spec[:, 1]
out_r = mask_r * mix_r - mask_i * mix_i
out_i = mask_r * mix_i + mask_i * mix_r
masked_spec = torch.stack([out_r, out_i], dim=1)
```

### 2. Silent Mixture for Target-Absent Samples Without Interferers

**File:** `data/dataset.py`, `__getitem__()`, TA branch

When `target_present=False` and `interferers_reverbed` is empty (edge case where `num_interferers=0` is impossible given `num_interferers_range: [1, 3]`, but if interferer lookup fails or the range is changed):

```python
speech_sum = np.zeros(mix_seg, dtype=np.float32)
mix_wav = _mix_at_snr_np(speech_sum, noise_wav, snr_db)
```

Inside `_mix_at_snr_np`, `sig_power ≈ 1e-8` (all zeros), so `scale ≈ sqrt(1e-8 / (noise_power × snr_linear))` which is extremely small. The resulting mixture is near-silent noise. The model would learn to output silence for near-silent input, which isn't useful.

More broadly, the same issue occurs when `speech_sum` has very low energy (e.g., interferers are short and mostly padded with zeros), producing abnormally quiet mixtures.

### 3. Scene-Aware Loss Weighting Imbalance

**File:** `losses/separation.py`, `scene_aware_loss()`

The loss divides by `count` (the number of active loss branches, 1 or 2), not by the number of samples in each branch. With `ta_ratio=0.2`, a typical batch of 2 might have 1 or 2 TP and 0 or 1 TA samples. When both are present, the loss is:

```
loss = (TP_loss + TA_loss) / 2
```

This gives the TA energy loss **equal weight** to the TP SI-SDR loss, regardless of the actual TP:TA ratio in the batch. A single TA sample in a batch of 2 gets 50% of the loss weight, despite representing 20% of the data distribution. This systematically over-weights TA suppression, potentially degrading TP extraction quality.

**Fix:** Weight by the number of samples in each branch, or use a fixed TP/TA loss ratio.

---

## 🟠 Significant Issues

### 4. STFT Loss: Window Recreated Every Forward Pass

**File:** `losses/stft_loss.py`, `SingleResolutionSTFTLoss.forward()`

```python
window = torch.hann_window(self.win_length, device=estimate.device)
```

A new Hann window tensor is created on **every forward call** (3 resolutions × every training step). This is wasteful and, more importantly, can cause issues with `torch.compile()` or mixed-precision training if the window ends up on the wrong device or dtype.

**Fix:** Register the window as a buffer in `__init__`:
```python
self.register_buffer('window', torch.hann_window(win_length))
```

### 5. Validation Uses Training Data (Data Leakage)

**File:** `train.py`, `main()`

```python
val_dataset = HiFiTSEDataset(cfg, phase=2)
```

The validation dataset is constructed from the **same HDF5 files** as training. Although `HiFiTSEDataset` performs dynamic mixing, it draws from the same pool of speakers/utterances. This means validation loss tracks training dynamics, not generalization. The code comment acknowledges this: *"Full validation would use a separate held-out set"* — but this makes the validation metrics unreliable.

**Evaluate.py** does use separate val speakers (via `speaker_manifest.json`), but the training loop's online validation doesn't.

### 6. Target Waveform Not Scaled Consistently With Mixture

**File:** `data/dataset.py`, `__getitem__()`

In the TP branch:
1. `target_wav` is peak-normalized clean speech
2. `target_reverbed = _apply_rir_np(target_wav, rir)` changes the amplitude
3. The mix is built from `target_reverbed + interferers + noise`
4. Final normalization scales both `mix_wav` and `target_wav` by the same factor

But the model is trained to predict `target_wav` (clean, dry) from the mixture (which contains `target_reverbed`). The normalization at step 4 applies a scale factor derived from the **mixture's peak**, which includes RIR-altered speech. Since the RIR changes the target's amplitude, the clean `target_wav` and its contribution to the mixture are at different amplitudes. While SI-SDR is scale-invariant, the **multi-resolution STFT loss is not**, so this amplitude mismatch between target and its contribution in the mix introduces noise into the spectral loss gradients.

### 7. Missing Dependencies in requirements.txt

**File:** `requirements.txt`

The file lists only 8 packages but the project needs:
- `pesq` (used in `evaluate.py`)
- `pystoi` (used in `evaluate.py`)
- `torchaudio` (used throughout but not listed)
- `scipy` (used for `fftconvolve` in dataset)

The evaluation script silently degrades (returns `None`) if `pesq`/`pystoi` aren't installed, which could lead to confusing results with missing PESQ/STOI metrics.

### 8. Hardcoded Sample Rate in Multiple Files

**Files:** `data/dataset.py`, `evaluate.py`

Both files have `SAMPLE_RATE = 48000` as a module-level constant, independent of the config's `audio.sample_rate`. If someone changes the config to 16kHz or 24kHz for experiments, the dataset and evaluator would still use 48kHz, causing silent data corruption.

---

## 🟡 Moderate Issues

### 9. Gradient Accumulation: D Loss Backward Timing

**File:** `train.py`, training loop, phase 3

```python
(loss_D / grad_accum).backward()  # Backward D first
# ... then ...
(loss_G / grad_accum).backward()  # Backward G second
```

The discriminator loss is backwarded **before** the generator loss. While functionally correct (D uses detached estimates), this means two backward passes coexist in the same step, which nearly doubles peak memory usage. A more memory-efficient approach would backward D, step D, then forward G's adversarial pass, backward G, step G.

### 10. Discriminator Not Frozen During D Loss Backward

When computing discriminator loss, `discriminator.requires_grad_(True)` is the default state. The D forward passes are through `est_wav_detach` and `target_wav`, so D gradients will only flow through D parameters (correct). However, `loss_D.backward()` computes gradients for D parameters while the generator's computational graph (from `loss_sep` and `loss_stft`) is still alive in memory. This doesn't cause incorrect gradients, but it keeps the entire generator graph in memory until `loss_G.backward()` frees it.

### 11. Overlap-Add Band Averaging May Cause Spectral Discontinuities

**File:** `models/band_split.py`, `BandMergeDecoder`

Overlapping bands are merged by averaging:
```python
mask[:, :, start:end + 1, :] += band_out
weight[:, :, start:end + 1, :] += 1.0
mask = mask / weight.clamp(min=1.0)
```

The number of overlapping bands changes at region boundaries (e.g., between fine/medium/coarse regions at ~1kHz and ~8kHz). This creates step changes in the effective weighting at these transitions, potentially introducing artifacts at perceptually important frequency boundaries.

### 12. No Gradient Clipping for Discriminator in Initial Accumulation Steps

**File:** `train.py`, training loop

Gradient clipping and `opt_D.step()` only happen when `(step + 1) % grad_accum == 0`. But `loss_D.backward()` happens every step. Gradients accumulate over 32 micro-steps before being clipped and applied. If early micro-steps produce large D gradients, they accumulate without bounds until the clipping step. With `grad_accum=32`, this could cause gradient explosion in the discriminator.

### 13. BiLSTM Hidden States Not Reset Between Batches

**File:** `models/tf_gridnet.py`, `GridNetBlock`

The BiLSTMs in `GridNetBlock` don't explicitly initialize hidden states (`h_0`, `c_0` default to zeros in PyTorch). This is correct for training, but during inference with chunked processing (`inference.py`), each chunk processes through the LSTM independently with zero initial hidden states. There's no mechanism to carry hidden states between chunks, which may cause discontinuities at chunk boundaries despite the overlap-add windowing.

### 14. Evaluation: SI-SDR Input Computed on Reverbed Target vs. Clean Mix

**File:** `evaluate.py`, main loop

```python
si_sdr_input = compute_si_sdr(mix_wav, clean_target)
```

SI-SDRi is computed as `SI-SDR(estimated, clean) - SI-SDR(mixture, clean)`. But the mixture contains the reverbed target, while `clean_target` is dry. The baseline SI-SDR(`mix`, `clean_target`) will be lower than expected because of the RIR mismatch, artificially inflating SI-SDRi. Standard TSE benchmarks (Libri2Mix etc.) don't apply RIR to the target, making these metrics non-comparable to published results.

---

## 🔵 Minor Issues / Code Quality

### 15. Lazy scipy Import Inside Hot Path
**File:** `data/dataset.py`, `_apply_rir_np()`
```python
from scipy.signal import fftconvolve
```
This import runs on every RIR application (every `__getitem__` call). Move to module level.

### 16. Google Verification File in Repository
`google8e1ca91efe5f5a03.html` is a Google Search Console verification file. This exposes website ownership and has no place in a research code repository.

### 17. No Random Seed Control in Dataset Workers
`data/dataset.py` uses `random.randint()` and `random.random()` without per-worker seed management. With `num_workers=4`, all workers share the same Python `random` state (forked), leading to correlated random augmentations in the first epoch. PyTorch's `worker_init_fn` should be used.

### 18. Missing `__init__.py` Verification
The generator imports `from models.band_split import ...`, `from models.usef import ...`, `from models.tf_gridnet import ...`. These require proper `__init__.py` files in `models/`, `losses/`, and `data/` packages. If these are empty or misconfigured, the project won't run.

### 19. Validation Cap Too Low
Validation is capped at 50 batches (100 samples with batch_size=2). For a 48kHz system with complex acoustic scenarios, this produces noisy validation metrics and may miss systematic failure modes.

### 20. No Checkpoint Rotation/Cleanup
Checkpoints are saved every 5000 steps over 500k total steps, producing ~100 checkpoint files. At potentially several GB each (due to discriminator + optimizers), this could consume hundreds of GB without cleanup.

### 21. `tse_collate_fn` Doesn't Handle Variable-Length Edge Cases
The collate function assumes all samples have identical shapes (fixed segments). While the dataset guarantees this, there's no error checking. A malformed HDF5 entry producing a wrong-length array would silently corrupt the batch.

---

## 📊 Architecture / Design Concerns

### 22. Single-Layer USEF Cross-Attention May Be Insufficient
The USEF module uses a **single** `nn.MultiheadAttention` layer. State-of-the-art TSE models (USEF-TSE, SMMA-Net) use multi-layer cross-attention with more sophisticated fusion mechanisms. A single layer may not learn robust speaker conditioning, especially with noisy references (50% noise ref probability).

### 23. No Data Augmentation Beyond RIR/Noise
Common speech augmentation techniques (speed perturbation, pitch shifting, SpecAugment-style masking, codec simulation) are absent. For 48kHz full-band audio, codec artifacts and bandwidth limitations are important real-world conditions to handle.

### 24. Batch Size 2 With 32-Step Accumulation
The effective batch size is 64, but each micro-step processes only 2 samples. This means batch normalization (if added) would be unreliable, and the gradient estimate per micro-step is very noisy. The current architecture uses LayerNorm, which is batch-size agnostic, so this is partially mitigated.

### 25. 500k Steps at Batch Size 2 With 4-Second Segments
At 48kHz × 4s = 192k samples/segment × 2 segments/batch, each step processes ~768k audio samples. Over 500k steps, the model sees ~384 billion samples total, which seems very high. But with grad_accum=32, there are only ~15.6k optimizer updates. This is actually relatively few optimizer updates for a complex model, suggesting training may be slow to converge.

---

## Summary Priority Table

| Priority | Issue | Impact |
|----------|-------|--------|
| 🔴 Critical | #1 — Incorrect complex masking | Caps output quality |
| 🔴 Critical | #2 — Near-silent TA mixtures | Corrupts TA training |
| 🔴 Critical | #3 — Loss weighting imbalance | Degrades TP performance |
| 🟠 Significant | #5 — Validation data leakage | Unreliable checkpointing |
| 🟠 Significant | #6 — Target amplitude mismatch | Noisy STFT loss |
| 🟠 Significant | #7 — Missing requirements | Broken eval metrics |
| 🟠 Significant | #8 — Hardcoded sample rate | Config flexibility broken |
| 🟡 Moderate | #4 — Window re-creation | Performance drag |
| 🟡 Moderate | #9-14 | Various training/eval issues |
| 🔵 Minor | #15-25 | Code quality, design improvements |
