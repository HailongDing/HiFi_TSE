# Training Convergence Investigation Report

**Date:** 2026-02-05
**Training run:** Steps 0–34,100 (~15 hours), Phase 1 only
**Checkpoint:** `checkpoint_0030000.pt` (last saved, ~4,100 steps lost due to system reboot)

---

## Executive Summary

After 34,100 steps, the training is **not converging effectively**. Statistical analysis reveals the primary separation loss (`loss_sep`) has **no statistically significant downward trend** (p=0.46). Multiple critical bugs and architectural issues were identified through deep investigation and Codex (o3) code review.

---

## 1. Statistical Analysis of Training Logs

### 1.1 Descriptive Statistics

| Scalar | Count | Mean | Std | Min (step) | Max (step) | CV |
|---|---|---|---|---|---|---|
| `train/loss_G` | 342 | 30.636 | 8.339 | 13.120 (16200) | 71.068 (23100) | 27.22% |
| `train/loss_sep` | 342 | 29.800 | 8.285 | 12.357 (16200) | 69.037 (23100) | 27.80% |
| `train/loss_stft` | 342 | 1.673 | 0.427 | 1.115 (26800) | 4.062 (23100) | 25.51% |
| `val/scene_aware_loss` | 6 | 18.918 | 1.712 | 16.289 (15000) | 20.586 (10000) | 9.05% |

- `loss_sep` dominates 94.7% of `loss_G` (mean 29.8 vs 1.67 for STFT)
- `loss_sep` mean of 29.8 = SI-SDR of **-29.8 dB** — model output is essentially uncorrelated with target
- Worst step: 23100 (loss_G=71.07, loss_sep=69.04, loss_stft=4.06)

### 1.2 Validation Loss (All 6 Data Points)

| Step | Value |
|---|---|
| 5000 | 19.838 |
| 10000 | 20.586 |
| 15000 | **16.289** (best) |
| 20000 | 19.981 |
| 25000 | 17.296 |
| 30000 | 19.520 |

Oscillatory with no clear improvement trend.

### 1.3 Rolling Average Trends (window=20)

| Scalar | Start | End | Trend |
|---|---|---|---|
| `train/loss_G` | 34.089 | 30.060 | -11.82% |
| `train/loss_sep` | 33.094 | 29.301 | -11.46% |
| `train/loss_stft` | 1.991 | 1.519 | -23.71% |
| `val/scene_aware_loss` | 18.918 | 18.918 | +0.00% |

### 1.4 Linear Regression Trend Analysis

| Scalar | Slope | R² | p-value | Significant? |
|---|---|---|---|---|
| `train/loss_G` | -4.064e-05 | 0.002 | 0.374 | **NO** |
| `train/loss_sep` | -3.358e-05 | 0.002 | 0.460 | **NO** |
| `train/loss_stft` | -1.413e-05 | 0.107 | **5.67e-10** | **YES** (downward) |
| `val/scene_aware_loss` | -4.438e-05 | 0.059 | 0.643 | **NO** |

**Key finding:** STFT loss improves (model learns spectral shape) but SI-SDR does not (model fails to learn speaker-specific extraction). This is consistent with a model outputting mixture passthrough rather than extracting the target.

### 1.5 Component Breakdown

- `loss_sep` CV: 27.80%, `loss_stft` CV: 25.51%
- Pearson correlation (loss_sep vs loss_stft): r=0.239, p=8.19e-06 (weak positive)
- `loss_sep` is 17.8x larger than `loss_stft`
- For good separation, `loss_sep` should become **negative** (SI-SDR > 0 dB)

---

## 2. Issues Found — Detailed Explanations

### 2.1 CRITICAL — LR Scheduler Is Non-Functional

**File:** `train.py:121,267`

**What's wrong:** The training loop iterates 500K micro-steps (`for step in range(0, 500000)`). The scheduler `CosineAnnealingLR(T_max=500000)` expects 500K calls to `.step()`. But `.step()` is only called every 32 iterations (inside the grad accumulation block), giving ~15,625 actual calls. The cosine curve barely starts — LR stays at 0.0002 for the entire run.

**Observed:** LR went from 1.99999995e-04 to 1.99997754e-04 over 34K steps — essentially zero decay.

**Why it matters:** Cosine annealing is designed to gradually reduce LR so the model makes large updates early (exploration) and small updates late (fine-tuning). Without decay, the model never enters a fine-tuning regime. In Phase 3 especially, constant high LR + adversarial losses = training instability.

**Suggestion:** Change `T_max = total_steps // grad_accum` so the scheduler completes one full cosine cycle over the 15,625 actual optimizer steps.

**Alternative considered:** Moving `sched_G.step()` outside the accumulation block (call every micro-step). Rejected because stepping the scheduler 32x per optimizer step would cause LR to decay far too fast.

---

### 2.2 CRITICAL — Phase Transitions Invisible to DataLoader Workers

**File:** `train.py:131,194` | `data/dataset.py:288-290`

**What's wrong:** The DataLoader uses `num_workers=4`. Workers are forked processes with their own copy of the dataset. When the main process calls `train_dataset.set_phase(2)` at step 100K, it only modifies its own copy. Workers keep generating phase-1 data (TP-only) until they're destroyed and recreated at the next epoch boundary.

**Why it matters:** The entire curriculum design depends on phase transitions:
- Phase 2 introduces TA samples (20%) and energy loss — crucial for learning to suppress absent speakers
- Phase 3 enables GAN losses — crucial for waveform quality

If workers don't see phase changes promptly, the model trains on wrong data for up to one full epoch after each transition. With a large dataset, one epoch could be tens of thousands of steps.

**Suggestion:** Recreate the DataLoader at each phase transition:
```python
if step == phase1_steps:
    train_dataset.set_phase(2)
    train_loader = DataLoader(train_dataset, ...)
    train_iter = infinite_loader(train_loader)
```

**Alternative considered:** Using `num_workers=0` (single-process loading). Rejected because it would severely slow training — HDF5 I/O and dynamic mixing are CPU-heavy and benefit from parallelism.

**Alternative considered:** Using `multiprocessing.Value` shared memory for the phase flag. More complex to implement and workers still have stale prefetch buffers.

---

### 2.3 HIGH — No Learning Rate Warmup

**File:** `train.py:121`

**What's wrong:** The model starts training at full LR=0.0002 from step 0. The USEF cross-attention and TFGridNet self-attention layers have randomly initialized weights. Large gradients + full LR on random attention weights can destabilize the attention softmax early on, causing the model to learn degenerate attention patterns that are hard to recover from.

**Why it matters:** Attention layers compute `softmax(QK^T / sqrt(d))`. With random Q and K, the dot products can be large and concentrated, causing the softmax to saturate. Full-strength gradient updates on saturated softmax produce near-zero gradients for most attention positions — the "dead attention" problem. A warmup phase lets the weights settle into a reasonable range before applying full LR.

**Suggestion:** Add a linear warmup over 2,000 optimizer steps (~64K micro-steps). Use a `LambdaLR` wrapper:
```python
warmup_steps = 2000
def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (total_opt_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))
```

**Why 2,000 steps:** At effective batch=64, 2,000 optimizer steps = 128K samples seen. This is enough for attention weights to stabilize without wasting too much of the training budget (2000/15625 = 12.8% of total optimizer steps).

---

### 2.4 HIGH — Single-Layer USEF Conditioning Bottleneck

**File:** `models/usef.py`, `models/generator.py`

**What's wrong:** Speaker identity is extracted from the reference via a single cross-attention layer (128-dim, 4 heads). The conditioned representation then passes through 6 TFGridNet blocks — none of which can re-access the reference. The speaker signal must survive through 18 sequential transformations (6 blocks x 3 stages each) via residual connections alone.

**Why it matters:** This explains the core convergence symptom: STFT loss improves (the model learns to produce speech-like spectrograms) but SI-SDR doesn't improve (the model doesn't learn to pick the right speaker). The model effectively learns a speaker-agnostic denoising solution because the speaker conditioning dilutes through the deep network.

For comparison, SpEx+ re-injects speaker embeddings at multiple network stages. SpeakerBeam uses FiLM conditioning at every layer. A single injection point is insufficient for deep networks.

**Suggestion:** Add cross-attention conditioning at every other TFGridNet block (blocks 0, 2, 4). Each conditioned block would have an additional USEF-style cross-attention that re-queries the reference features:
```python
class GridNetBlock(nn.Module):
    def __init__(self, feature_dim, ..., use_speaker_cond=False):
        ...
        if use_speaker_cond:
            self.speaker_attn = nn.MultiheadAttention(feature_dim, num_heads)
            self.speaker_norm = nn.LayerNorm(feature_dim)
```

**Why not every block:** Adding cross-attention to all 6 blocks increases parameters by ~400K and computation. Every-other-block is a good balance — the model re-consults the reference 3 times total (initial USEF + blocks 0, 2, 4), keeping the speaker signal fresh through the entire network.

**Alternative considered:** FiLM conditioning (scale and shift features using a speaker embedding). Simpler and cheaper, but less expressive than cross-attention for capturing time-varying speaker characteristics.

---

### 2.5 HIGH — `load_checkpoint` Device Mismatch on Resume

**File:** `train.py:58-67`

**What's wrong:**
```python
ckpt = torch.load(path, map_location="cpu")
opt_G.load_state_dict(ckpt["opt_G"])  # momentum buffers now on CPU
```
The optimizer's internal state (Adam momentum `exp_avg` and `exp_avg_sq`) gets loaded to CPU, but model parameters are on GPU. The next `opt_G.step()` tries to compute `param.data - lr * exp_avg` across devices → RuntimeError.

**Why it matters:** This hasn't crashed yet because we haven't resumed from a checkpoint. But we have 6 checkpoints saved (5K–30K). When we restart training from `checkpoint_0030000.pt`, it will crash immediately.

**Suggestion:** Change to `map_location=device`:
```python
ckpt = torch.load(path, map_location=device)
```

---

### 2.6 MODERATE — GAN Feature Matching Shape Mismatch

**File:** `train.py:232,243`

**What's wrong:**
```python
d_real_feat = discriminator(target_wav[tp_mask])   # batch = N_tp (e.g. 1)
d_fake_feat_g = discriminator(est_wav)              # batch = 2
loss_fm = feature_matching_loss(d_real_feat, d_fake_feat_g)  # broadcasting!
```
When a batch has 1 TP + 1 TA sample (32% of Phase 3 batches), the real features have batch=1 but fake features have batch=2. PyTorch broadcasts the single real feature across both fake features, producing a mathematically incorrect loss.

**Why it matters:** The feature matching loss is supposed to compare "real sample i" with "fake sample i". Broadcasting means both fake samples are compared against the same real sample, and the TA sample gets adversarial gradients it shouldn't receive. This corrupts the G gradient signal in ~32% of Phase 3 batches.

**Suggestion:** Mask the generator's adversarial forward too:
```python
d_fake_out_g, d_fake_feat_g = discriminator(est_wav[tp_mask])
```

---

### 2.7 MODERATE — STFT Loss Explodes for TA Samples

**File:** `train.py:217`, `losses/stft_loss.py:40`

**What's wrong:** The STFT loss is computed on all samples unconditionally. For TA samples, `target_wav` is all zeros. The spectral convergence term computes:
```
sc = ||est_mag - 0||_F / (||0||_F + 1e-8)
```
The denominator is 1e-8, so even small estimated magnitudes produce enormous loss values (potentially millions).

**Why it matters:** In Phase 1, this doesn't matter (no TA samples). But in Phase 2+, 20% of samples are TA. When a TA sample hits this code path, it produces a massive STFT loss that dominates the gradient, creating a spike that overwhelms the separation learning signal.

**Suggestion:** Only compute STFT loss on TP samples:
```python
if current_phase >= 2:
    tp_mask = tp_flag.bool()
    if tp_mask.any():
        loss_stft = stft_loss_fn(est_wav[tp_mask], target_wav[tp_mask])
    else:
        loss_stft = torch.tensor(0.0, device=device)
else:
    loss_stft = stft_loss_fn(est_wav, target_wav)
```

---

### 2.8 MODERATE — No SIR Control in Data Pipeline

**File:** `data/dataset.py`

**What's wrong:** The mixture pipeline controls SNR (noise level relative to speech sum) but not SIR (target level relative to interferers). The relative amplitude between target and interferers is determined entirely by whatever levels the raw utterances have in the HDF5 files. Different datasets (DNS4, EARS, VCTK) have different recording levels, and there's no per-utterance normalization.

**Why it matters:** A quiet DNS4 target mixed with a loud EARS interferer creates an implicitly very low SIR (maybe -10 dB), making separation nearly impossible. A loud EARS target with quiet DNS4 interferers creates a high SIR (+15 dB), making it trivial. This uncontrolled variance adds ~10-20 dB of difficulty spread on top of the already-wide SNR range, making the gradient signal noisier.

Compound variance sources:

| Source | Impact |
|---|---|
| SNR range: -5 to +15 dB (20 dB span, 100x power) | Very High |
| No SIR control (target/interferer levels uncontrolled) | High |
| 1-3 interferers (uniform) | High |
| RIR diversity (anechoic to highly reverberant) | Moderate-High |
| Noisy reference (50% probability, 5-20 dB) | Moderate |

**Suggestion:** Add per-utterance peak normalization before mixing:
```python
def _normalize_utterance(wav):
    peak = np.abs(wav).max()
    if peak > 0:
        wav = wav / peak * 0.9
    return wav
```
Apply to target, each interferer, and reference before mixing.

**Alternative considered:** Explicit SIR sampling (e.g., uniform [-5, 10] dB). More complex but gives finer control. Can be added later if normalization alone isn't sufficient.

---

### 2.9 LOW — Unconstrained Mask in BandMergeDecoder (No Fix Needed)

**File:** `models/band_split.py`

The decoder applies `mask * mix_spec` where `mask` has no activation function bounding its values. With random initialization, mask values can be large positive or negative numbers, producing extreme spectral estimates.

**Assessment:** No change needed if we add LR warmup (Issue 2.3). Warmup effectively handles the early instability period. An unconstrained mask is actually beneficial once training stabilizes — it allows spectral enhancement (mask > 1.0) that bounded masks can't achieve.

---

### 2.10 LOW — No Global Skip Connection Around TFGridNet (No Fix Needed)

**File:** `models/tf_gridnet.py`

TFGridNet is `for block in self.blocks: x = block(x)` with no global residual from input to output. Each block has internal residuals, but there's no shortcut for the network to default to "pass through + refine."

**Assessment:** No change recommended. The decoder's multiplicative masking of `mix_spec` already provides the right inductive bias — the model just needs to learn a mask close to 1.0 for "pass through" and close to 0.0 for "suppress." Adding a global skip inside TFGridNet would be redundant.

---

## 3. Loss Function Reference

### SI-SDR Loss (`loss_sep`)
- Formula: `-10 * log10(proj² / noise²)` where proj is the projection of estimate onto target
- Scale-invariant (zero-mean normalization first)
- Loss > 0 means model is worse than mixture; loss < 0 means model separates successfully
- Good target: loss_sep around -10 to -15 (SI-SDR of 10–15 dB)

### Multi-Resolution STFT Loss (`loss_stft`)
- 3 scales: (512,240,50), (1024,600,120), (2048,1200,240)
- Each: spectral convergence (Frobenius norm ratio) + log-magnitude L1
- Averaged over 6 terms; weighted by `lambda_stft=0.5` in loss_G
- NOT scale-invariant (log-magnitude depends on absolute levels)

### Combined: `loss_G = 1.0 * loss_sep + 0.5 * loss_stft`

---

## 4. Architecture Summary

| Module | Params | Notes |
|---|---|---|
| BandSplitEncoder | ~200K | 53 bands, shared for mix & ref |
| USEFModule | ~66K | Single cross-attention layer (bottleneck) |
| TFGridNet (6 blocks) | ~6.9M | BiLSTM + self-attention per block, residual connections |
| BandMergeDecoder | ~340K | Unconstrained multiplicative mask |
| **Generator Total** | **7.61M** | Borderline sufficient for 48kHz TSE |

- LayerNorm throughout (appropriate for batch_size=2)
- Post-LN style (acceptable at 6 blocks depth)
- No BatchNorm (correct choice given small batch size)

---

## 5. Training Configuration Reference

```yaml
batch_size: 2
grad_accum_steps: 32       # effective batch = 64
lr: 0.0002
betas: [0.8, 0.99]
grad_clip: 5.0
total_steps: 500000
optimizer: AdamW
scheduler: CosineAnnealingLR (T_max=500000, but only 15625 actual calls — BUG)
warmup: none

curriculum:
  phase1_steps: 100000    # TP only, SI-SDR + STFT
  phase2_steps: 300000    # add TA + energy loss; GAN starts here
```

---

## 6. Fix Plan

### Fixes to implement (8 items):

| # | Issue | Severity | Action |
|---|---|---|---|
| 1 | LR scheduler T_max | CRITICAL | Change to `total_steps // grad_accum` |
| 2 | Phase transition workers | CRITICAL | Recreate DataLoader at transitions |
| 3 | No LR warmup | HIGH | Add linear warmup with cosine decay |
| 4 | Single-layer USEF | HIGH | Re-inject speaker info at TFGridNet blocks 0, 2, 4 |
| 5 | Checkpoint device mismatch | HIGH | Use `map_location=device` |
| 6 | GAN feature matching mismatch | MODERATE | Mask `est_wav` to TP for G adversarial |
| 7 | STFT loss on TA samples | MODERATE | Only compute on TP in Phase 2+ |
| 8 | No SIR control | MODERATE | Add per-utterance peak normalization |

### No fix needed (2 items):

| # | Issue | Severity | Reason |
|---|---|---|---|
| 9 | Unconstrained mask | LOW | LR warmup handles early instability; unbounded mask enables spectral enhancement |
| 10 | No global skip in TFGridNet | LOW | Decoder masking provides equivalent inductive bias |
