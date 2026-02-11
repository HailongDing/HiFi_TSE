# Over-Suppression Analysis: Phase 2 Amplitude Decay

**Date**: 2026-02-10 (updated 2026-02-11 after deep review)
**Observed at**: Step 257K (phase 2, resumed from 100K)
**Symptom**: Validation rms_ratio steadily declining from 1.1 (100K) to 0.59 (255K)

---

## Observed Data

### Validation rms_ratio Trend (Phase 2)

| Step | Val Loss | SI-SDR (dB) | rms_ratio | ta_energy (dB) |
|------|----------|-------------|-----------|----------------|
| 100K | 17.21 | -40.87 | 1.105 | -24.8 |
| 105K | 18.37 | -40.96 | 1.077 | -25.0 |
| 110K | 17.75 | -42.83 | 0.930 | -25.5 |
| 120K | 17.59 | -41.12 | 0.900 | -27.1 |
| 140K | 16.01 | -42.47 | 0.945 | -26.9 |
| 155K | 16.97 | -39.15 | 0.872 | -27.1 |
| 170K | 17.11 | -39.56 | 0.724 | -28.4 |
| 180K | 16.11 | -39.12 | 0.646 | -30.2 |
| 190K | 17.96 | -43.03 | 0.761 | -29.8 |
| 255K | 14.72 | -38.00 | 0.593 | -30.5 |

### Milestone Real Audio Check

| Milestone | SI-SDR (dB) | rms_ratio | Real audio ratio |
|-----------|-------------|-----------|-----------------|
| 100K | -28.97 | 0.915 | 0.497 |
| 200K | -24.32 | 0.583 | 0.279 |

### Training Log rms Values (Step ~258K, TP batches only)

Training rms on TP batches: 0.49-0.57 range (was 0.8-1.1 at 100K).

---

## Root Cause Analysis

### Current Loss Weights (from `configs/hifi_tse.yaml`)

```yaml
loss_weights:
  lambda_sep: 1.0    # SI-SDR (scene-aware in phase 2+)
  lambda_stft: 1.5   # Multi-resolution STFT
  lambda_l1: 0.5     # L1 waveform
  ta_weight: 0.1     # TA energy loss weight inside scene_aware_loss
```

### Effective Gradient Contribution

| Loss | Weight (λ) | Typical Value | **Weighted Contribution** | Scale-Sensitive? |
|------|-----------|---------------|--------------------------|-----------------|
| `sep` (SI-SDR) | 1.0 | ~20 | **~20** | NO (scale-invariant) |
| `stft` | 1.5 | ~1.3 | **~2.0** | Partially (log compresses) |
| `l1` | 0.5 | ~0.05 | **~0.025** | YES (but negligible weight) |

**SI-SDR dominates the total loss by 10× over scale-sensitive losses.**

### Primary Cause: Three Compounding Forces

The over-suppression is driven by three forces acting together with no counterbalance:

#### Force 1: SI-SDR Scale-Invariance

SI-SDR zero-means both estimate and target, then computes a projection (`losses/separation.py` lines 17-30). The loss sees **no difference** between 100% and 10% amplitude output. It only optimizes waveform *shape*, not *scale*.

#### Force 2: AdamW Weight Decay (Previously Overlooked)

**This was missing from the original analysis.** The optimizer is `AdamW(generator.parameters(), lr=..., betas=...)` with no explicit `weight_decay` parameter, so PyTorch's default of **0.01** applies. AdamW's update rule includes `w = w - lr * weight_decay * w`, which continuously shrinks all parameters toward zero — including the BandMergeDecoder's mask projection weights. Over 100K+ steps, this directly reduces mask magnitudes and output amplitude.

With SI-SDR providing no gradient to resist this shrinkage, and STFT/L1 too weak to counterbalance, weight decay acts as a persistent inward force on the mask. This explains why suppression occurs even in Phase 1 (rms_ratio already 0.915 at 100K, before TA loss is introduced).

#### Force 3: TA Energy Loss — Unbounded Suppression Gradient

The TA energy loss `10 * log10(mean(est²) + eps)` has gradient:

```
d/d(est_i) = 20 * est_i / (N * ln(10) * (mean(est²) + eps))
```

This gradient is **O(1/energy)** — it pushes harder as the output gets quieter, and **never stops**. There is no floor. Once the output reaches -30 dB (where validation shows it), the gradient is still pushing toward -40, -50, -60 dB. This persistent suppression gradient flows through all shared model layers (encoder, USEF, TFGridNet), creating a general bias toward reduced output energy that bleeds into TP predictions.

**The original analysis underestimated this factor.** The plan's statement "No change needed to TA energy loss" was incorrect.

### Why Phase 2 Accelerates Suppression

In Phase 1 (TP-only):
- STFT and L1 operate on all samples, providing some amplitude anchor
- No TA energy loss pulling toward suppression
- But weight decay + scale-invariant SI-SDR already cause slow drift (rms 1.0 → 0.915)

In Phase 2:
- `scene_aware_loss` wraps SI-SDR with `(1.0 - ta_weight) * tp_loss = 0.9 × SI-SDR` — still dominant
- TA energy loss adds persistent suppression gradient on shared weights — **no floor**
- STFT/L1 only on TP samples (correct), but too weak to counterbalance
- **Optimizer moment carryover**: Adam's variance estimate `v` is low for the new TA gradient directions, causing disproportionately large effective step sizes for the suppression signal initially. This explains the rapid rms_ratio drop in the first 20K steps of Phase 2 (1.105 → 0.93).

### Quantitative STFT Scale Sensitivity

For uniform scaling by factor α=0.5:
- Spectral convergence: `|α - 1| = 0.5`
- Log-magnitude L1: `|log(α)| = 0.693` per TF bin
- Combined per resolution: 1.193. Averaged over 3 resolutions × 2 terms ÷ 6: **0.597**
- Weighted at λ=1.5: **~0.9**

At 50% suppression, STFT contributes only **4.5% of the total loss** vs SI-SDR. This confirms the STFT loss is too weakly weighted to anchor amplitude.

### Note on SI-SDR Values

The SI-SDR values of -38 to -46 dB in validation are alarmingly poor. A trained separation model at 250K steps should achieve +10 to +20 dB SI-SDR. Values of -40 dB indicate the estimate is essentially uncorrelated with the target. This may indicate a problem beyond amplitude scaling (e.g., the model is not extracting the correct speaker, or there's a data pipeline issue). **This warrants separate investigation.** However, the amplitude decay is independently observable from the rms_ratio trend and real_audio_check results, so the over-suppression fix should proceed regardless.

---

## Revised Fix Plan

### Change 1: New `amplitude_loss` in `losses/separation.py`

```python
def amplitude_loss(estimate, target, eps=1e-8):
    """Penalize RMS ratio deviation from 1.0.

    Returns: scalar loss = mean((rms_est / rms_tgt - 1)^2)
    """
    rms_est = estimate.pow(2).mean(dim=-1).sqrt()
    rms_tgt = target.pow(2).mean(dim=-1).sqrt().clamp(min=eps)
    ratio = rms_est / rms_tgt
    return (ratio - 1.0).pow(2).mean()
```

**Gradient analysis**: `dL/d(est_i) = 2 * (r-1) * est_i / (N * rms_est * rms_tgt)`. This is proportional to `est_i` (multiplicative correction) and has a `1/rms_est` term that **increases corrective force as suppression worsens** — a desirable self-correcting property.

### Change 2: Config — Revised Weights

```yaml
loss_weights:
  lambda_sep: 1.0
  lambda_stft: 5.0     # increased from 1.5 (simple, effective scale anchor)
  lambda_l1: 0.5       # keep as-is (increase is cosmetic at any reasonable λ)
  lambda_amp: 15.0      # NEW: strong amplitude anchor
  ta_weight: 0.1
```

**Weight justification for lambda_amp=15.0** (revised up from original 5.0):

| rms_ratio | Loss (λ=15) | % of SI-SDR (~20) | Gradient strength |
|-----------|------------|-------------------|-------------------|
| 0.5 | 3.75 | 18.8% | Strong correction |
| 0.7 | 1.35 | 6.8% | Moderate correction |
| 0.9 | 0.15 | 0.75% | Gentle nudge |
| 1.0 | 0.0 | 0% | No interference |
| 1.5 | 3.75 | 18.8% | Strong correction |

At λ=5 (original plan), the loss at rms_ratio=0.59 would be only 4.2% of SI-SDR — insufficient. At λ=15, it's ~12.6%, providing meaningful corrective force.

**Weight justification for lambda_stft=5.0** (revised up from 1.5):

At 50% suppression with λ=5.0: weighted contribution = **3.0** (15% of SI-SDR). This is a zero-code-change fix that significantly strengthens the existing scale anchor. The STFT log-magnitude component already penalizes uniform scaling by `|log(scale)|` per TF bin.

**Lambda_l1 kept at 0.5**: The deep review confirmed that increasing L1 to 2.0 is essentially cosmetic. At λ=2.0 the contribution is 0.1 vs SI-SDR at 20 (a 200:1 ratio). For L1 to matter you'd need λ=100+, which would fight SI-SDR on waveform shape. Not worth it.

### Change 3: Clamp TA Energy Loss at -40 dB Floor

In `losses/separation.py`, modify `energy_loss`:

```python
def energy_loss(estimate, floor_db=-40.0, eps=1e-8):
    """Energy loss for target-absent samples.

    Forces output energy toward zero, with a floor to prevent unbounded
    suppression gradients on shared model layers.
    """
    energy = estimate.pow(2).mean(dim=-1)
    energy_db = 10.0 * torch.log10(energy + eps)
    return energy_db.clamp(min=floor_db)
```

**Why**: Without a floor, the gradient `O(1/energy)` grows unboundedly as output approaches zero. Once the output is at -40 dB (essentially inaudible), there is no perceptual benefit to pushing further, but the growing gradient still bleeds through shared layers and suppresses TP output. The clamp stops gradient flow once the TA output is quiet enough.

### Change 4: Add Amplitude Loss to Training Loop

In `train.py`:
- Import `amplitude_loss` from `losses.separation`
- Compute `loss_amp = amplitude_loss(est_wav[tp_mask], target_wav[tp_mask])` alongside STFT/L1 for TP samples
- Add `lambda_amp * loss_amp` to `loss_G`
- Add `amp` column to console log and `train/loss_amp` to TensorBoard
- Align rms_ratio monitoring to use per-sample computation (matching the loss)

### Change 5: Warm Up lambda_amp Over 5K Micro-Steps

When resuming from the 100K checkpoint, the AdamW optimizer moments have no history for amplitude_loss gradients. Sudden introduction at full weight can cause instability because Adam's variance estimate `v` is low for these gradient directions, causing disproportionately large effective steps.

```python
# In training loop, after computing loss_amp:
amp_warmup_steps = 5000
if step < phase1_steps + amp_warmup_steps:
    amp_scale = (step - phase1_steps) / amp_warmup_steps
else:
    amp_scale = 1.0
loss_G = loss_G + loss_w["lambda_amp"] * amp_scale * loss_amp
```

This linearly ramps lambda_amp from 0 to full weight over the first 5K steps of Phase 2.

---

## Effective Loss Balance After Fix

| Loss | Weight | At rms=0.5 | At rms=1.0 | Purpose |
|------|--------|-----------|-----------|---------|
| sep (SI-SDR) | 1.0 | ~20 | ~20 | Waveform shape (scale-invariant) |
| stft | **5.0** | **~3.0** | ~6.5 | Spectral fidelity + scale anchor |
| l1 | 0.5 | ~0.025 | ~0.025 | Waveform amplitude (minor) |
| **amp (new)** | **15.0** | **~3.75** | **0** | **Direct RMS anchor to 1.0** |

Total scale-sensitive contribution at 50% suppression: **~6.75** (34% of SI-SDR). Previously: **~2.0** (10% of SI-SDR). This is a 3.4× improvement in scale-preserving gradient strength.

---

## Implementation Plan

1. Add `amplitude_loss()` to `losses/separation.py`
2. Modify `energy_loss()` to add `-40 dB` floor clamp
3. Update `configs/hifi_tse.yaml`: `lambda_stft: 5.0`, add `lambda_amp: 15.0`
4. In `train.py`: import amplitude_loss, add to TP loss branch with warmup, add logging
5. Resume training from step 100K checkpoint
   - Current 260K weights have suppressed amplitude baked in, not recoverable
   - Phase 2 restarts at 100K, now with amplitude anchor + TA floor preventing decay
   - Lambda_amp warms up over first 5K steps to avoid optimizer shock
6. Monitor rms_ratio should stabilize at 0.8-1.2 through phases 2 and 3

## What This Does NOT Change

- SI-SDR loss function (standard metric, correct for separation quality)
- Model architecture (no structural changes)
- Phase transitions or curriculum schedule
- GAN losses (phase 3) — amplitude loss provides a stable base for GAN training
- Lambda_l1 (increase is cosmetic; not worth changing)

## Risk Assessment

- **Low-moderate risk**: amplitude_loss gradient increases with suppression (self-correcting) but also increases with low rms_tgt (edge case for very quiet targets). The `eps=1e-8` clamp on rms_tgt prevents division by zero but very quiet TP targets could produce large gradients.
- **Mitigated by warmup**: lambda_amp ramps from 0 to full weight over 5K steps, allowing optimizer moments to adapt gradually.
- **Possible concern**: if lambda_amp=15 is too high, the model may over-correct and oscillate around rms_ratio=1.0. Monitor `train/loss_amp` — if it oscillates rather than decaying, reduce lambda_amp.
- **AdamW weight decay**: still active at default 0.01. The amplitude loss should be strong enough to counteract this, but if suppression recurs, consider setting `weight_decay=0` for the BandMergeDecoder projection layers.

## Deferred Investigation

- **SI-SDR at -40 dB**: Alarmingly poor for 250K steps. May indicate deeper issues beyond amplitude (speaker confusion, data pipeline, reverbed target alignment). Should investigate separately after the amplitude fix stabilizes rms_ratio.
