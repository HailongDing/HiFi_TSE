# Fix Plan: Over-Suppression Root Cause & Remediation

**Date:** 2026-02-09 (updated after Codex MCP review)
**Issue:** Model produces output at ~8% of expected energy, destroying TP extraction quality.
**Evaluation:** SI-SDRi -19.64 dB, PESQ 1.09, STOI 0.473 (checkpoint_best.pt, step 365K)

---

## Root Cause Analysis

The over-suppression results from four interacting factors:

### 1. SI-SDR is completely scale-invariant (primary cause)

SI-SDR computes the projection of the estimate onto the target direction, then measures the ratio of projected signal to residual noise. If `estimate = alpha * target` for any alpha > 0, the projection perfectly recovers the estimate, noise = 0, and SI-SDR = +infinity.

This means SI-SDR provides **zero gradient with respect to output amplitude**. The model can produce arbitrarily quiet output and still achieve excellent SI-SDR, as long as the waveform shape is correct.

### 2. Complex mask initializes at (0, 0) — fully suppressed

The BandMergeDecoder projection layers (`nn.Linear`) use PyTorch default initialization (Kaiming uniform, centered at zero). The complex mask starts at approximately (0+0i), meaning the model begins from a fully-suppressed state.

To pass the input signal through unchanged, the mask needs to be (1+0i) — the complex identity. But since SI-SDR doesn't reward amplitude, there is insufficient gradient to push the mask from (0,0) toward (1,0). The model stays in the suppressed regime.

### 3. TA energy loss pushes shared weights toward suppression

In Phase 2+ (step 100K+), the energy loss for target-absent samples minimizes output energy via the shared backbone (TFGridNet, speaker encoder, cross-attention). Since SI-SDR provides no counter-gradient to maintain TP amplitude, the shared weights drift toward global suppression.

With `ta_weight=0.2` and batch_size=2:
- 64% of batches are all-TP (no TA gradient)
- 32% have 1 TP + 1 TA (TA energy loss actively pushes toward suppression)
- 4% are all-TA (full suppression gradient)

### 4. STFT loss is too weak to compensate

STFT loss (`lambda_stft=0.5`) is internally averaged over 6 terms (3 resolutions x 2 loss types), giving ~0.083 effective weight per term vs 1.0 for SI-SDR. The log-magnitude L1 component IS scale-sensitive and does fight suppression, but at this weight it cannot overcome the combined scale-invariance of SI-SDR + TA energy gradient + initialization bias.

### Degenerate minimum

The optimizer converges to: mask values are small but correctly-shaped (good SI-SDR), STFT loss is elevated but not dominant, TA energy loss is well minimized. The model produces correctly-shaped but heavily attenuated output (~8% amplitude).

---

## Proposed Fixes (updated with Codex MCP review)

### Fix 1: Add time-domain L1 loss for TP samples [CRITICAL]

**Files:** `losses/separation.py`, `train.py`, `configs/hifi_tse.yaml`

Add an L1 waveform loss that directly penalizes amplitude mismatch:

```python
def l1_waveform_loss(estimate, target):
    """Scale-sensitive L1 loss on time-domain waveform."""
    return torch.nn.functional.l1_loss(estimate, target)
```

Apply only to TP samples in the training loop:

```python
if tp_mask.any():
    loss_l1 = l1_waveform_loss(est_wav[tp_mask], target_wav[tp_mask])
    loss_G += lambda_l1 * loss_l1
```

Config: `lambda_l1: 0.5` (Codex recommendation: 0.5-1.0, with warmup ramp)

**Rationale:** L1 loss IS scale-sensitive. If estimate is 8% of target, L1 = 0.92 * mean(|target|), providing strong gradient to increase amplitude. This directly breaks the scale-invariance trap.

### Fix 2: Initialize complex mask near identity (1+0i) [HIGH]

**File:** `models/band_split.py`

Add custom initialization in BandMergeDecoder.__init__. **Important (Codex correction):** the bias layout is `(band_width * 2,)` with interleaved real/imag pairs (from reshape to `(B, T, band_width, 2)`), so use stride-2 indexing:

```python
for proj in self.projections:
    nn.init.normal_(proj.weight, std=1e-4)  # tiny perturbation
    with torch.no_grad():
        proj.bias.zero_()
        proj.bias[0::2] = 1.0  # real part = 1 (pass-through)
        # imag part (odd indices) stays 0
```

**Rationale:** The model starts from "pass through the input" instead of "suppress everything." Combined with the L1 loss, the optimizer has both a good starting point and a gradient signal to maintain amplitude.

### Fix 3: Increase STFT loss weight [MODERATE]

**File:** `configs/hifi_tse.yaml`

Change `lambda_stft` from `0.5` to `1.5` (Codex recommendation: 1.0-1.5 when combined with L1):

```yaml
loss_weights:
  lambda_stft: 1.5  # was 0.5
```

**Rationale:** With /6 internal averaging, effective weight per STFT term goes from 0.083 to 0.25. The log-magnitude L1 component provides scale-aware gradients that help anchor output amplitude. This reinforces Fix 1.

### Fix 4: Reduce and configure TA weight [MODERATE]

**File:** `losses/separation.py`, `configs/hifi_tse.yaml`

Instead of full stop-gradient (which may weaken TA suppression too much), reduce `ta_weight` and make it configurable:

```yaml
loss_weights:
  ta_weight: 0.1  # was 0.2 (Codex recommendation: 0.05-0.1)
```

Pass `ta_weight` from config to `scene_aware_loss()` instead of hardcoding.

**Rationale:** Reduces the suppression pressure from TA energy loss on shared backbone while still maintaining some TA suppression learning. More conservative than full detach, easier to tune.

---

## Implementation Order

1. **Fix 1** (L1 loss) — directly addresses root cause
2. **Fix 2** (mask init) — prevents cold-start suppression
3. **Fix 3** (STFT weight) — strengthens existing amplitude signal
4. **Fix 4** (TA weight) — reduces suppression pressure

All fixes bundled into one commit. Requires training from scratch.

---

## Expected Outcome

| Metric | Current | Target (after fixes) |
|--------|---------|---------------------|
| SI-SDRi | -19.64 dB | > +5 dB |
| PESQ | 1.09 | > 2.0 |
| STOI | 0.473 | > 0.75 |
| TA suppression | 19.4 dB | > 15 dB (may decrease slightly with amplitude fix) |
| Output RMS ratio | 8% | > 80% |

---

## Risk Assessment

- Fix 1 (L1 loss) may reduce SI-SDR metric since the optimizer now balances two objectives. But real quality (PESQ, STOI, perceptual) should improve dramatically.
- Fix 2 (mask init) changes early training dynamics. The model may overfit to pass-through early on, but the separation losses will push it away from identity.
- Fix 3 (STFT weight) may slow convergence of SI-SDR since STFT gradients compete more, but overall quality should improve.
- Fix 4 (TA weight reduction) may slightly weaken TA suppression learning. Monitor TA metrics during training.
- TA suppression may slightly decrease as the model shifts from "suppress everything" to "selectively extract." This is the correct trade-off.

---

## Evaluation Plan

Evaluate at checkpoints:
- Step 100K (end of Phase 1) — verify amplitude is maintained
- Step 200K (mid Phase 2) — verify TA doesn't cause suppression regression
- Step 300K (Phase 3 start) — pre-GAN baseline **[PRESERVE THIS CHECKPOINT]**
- Step 350K, 400K, 450K, 500K — track GAN effect

**Critical:** Configure checkpoint rotation to always preserve phase-boundary checkpoints (100K, 300K) so we can isolate phase effects.
