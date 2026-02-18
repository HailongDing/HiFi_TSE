# Over-Suppression Fix — Phase 2 Training (Step 200K-230K)

## Problem

Training entered Phase 2 (TP+TA) at step 200K. Since then, rms_ratio has crashed:

| Step | Val rms_ratio | Val ta_energy | Val SI-SDR |
|------|--------------|---------------|------------|
| 200K | 1.327 | -23.8 dB | -4.32 dB |
| 210K | 1.038 | -25.9 dB | -3.72 dB |
| 220K (best) | 0.746 | -30.6 dB | -2.83 dB |
| 230K | 0.633 | -30.6 dB | -3.74 dB |

The model is learning to suppress all output (TP and TA) to minimize TA energy loss.

## Root Cause

**The `energy_loss` function returns raw negative dB values** (`losses/separation.py:59-61`). When added to total loss via `ta_weight * mean(energy_db)`, more suppression produces more negative values, directly reducing total loss. The optimizer is **rewarded** for over-suppression.

Example at step 220K:
- TP component: `0.9 * (+4 to +12)` = +3.6 to +10.8
- TA component: `0.1 * (-30.6)` = **-3.06** (reduces total loss!)

The floor clamp at -40 dB means the optimizer can extract up to `-4.0` of free loss reduction by suppressing TA to silence. Since model weights are shared, this suppression bias bleeds into TP outputs.

**Secondary**: `checkpoint_best.pt` (step 220K, val_loss=0.775) is actually the most-suppressed model — best model selection is rewarding suppression.

## Codex o3 Review

- Root cause confirmed correct
- `relu(energy_db - target_db)` fix is sound; `softplus` is a smoother alternative but `relu` is fine
- `target_db=-30` is heuristic — may need tuning
- TP-only SI-SDR for best model is OK
- Resume from 200K is safest (pre-dates bad gradients)
- Bug found: `validate()` calls `scene_aware_loss` without `ta_weight`, using function default 0.2 instead of config's 0.1 — should fix for consistency

## Fix: 4 Changes

### Change 1: Fix `energy_loss` to be non-negative (`losses/separation.py:46-61`)

Replace the raw dB return with a thresholded positive loss:

```python
def energy_loss(estimate, target_db=-30.0, eps=1e-8):
    energy = estimate.pow(2).mean(dim=-1)
    energy_db = 10.0 * torch.log10(energy + eps)
    # Only penalize energy ABOVE target — gradient stops once suppressed enough
    return torch.relu(energy_db - target_db)
```

Behavior:
- Output at -20 dB: loss = relu(-20 - (-30)) = 10 (penalized)
- Output at -30 dB: loss = relu(0) = 0 (target reached, no gradient)
- Output at -40 dB: loss = relu(-10) = 0 (no over-suppression incentive)

This eliminates the negative loss contribution. TA loss is always >= 0.

### Change 2: Use TP-only SI-SDR for best model selection (`train.py:525-548`)

Currently `val_loss` (scene_aware_loss including TA term) is used for best checkpoint. Replace with TP-only validation SI-SDR as the selection metric (higher is better).

### Change 3: Fix validation ta_weight consistency (`train.py:154`)

`validate()` calls `scene_aware_loss(est_wav, target_wav, tp_flag)` without passing `ta_weight`, so it uses function default 0.2 instead of config's 0.1. Pass `ta_weight` explicitly.

### Change 4: Resume from step 200K checkpoint (`configs/hifi_tse.yaml`)

The Phase 1 checkpoint (step 200K) was saved before over-suppression began. Resume from it with the fixed loss, using `--reset-optimizer` to clear contaminated Adam momentum.

## Verification

1. Smoke test: Run 100 steps from step 200K, verify TA loss is non-negative
2. Early check at step 205K: rms_ratio should stay > 0.8 (vs 1.265 in broken run)
3. Monitor: TA energy should stabilize around -30 dB and stop dropping
4. Watch: SI-SDR should improve without rms_ratio collapse

---

## Continued Over-Suppression (Step 200K–245K) — Amplitude Loss Fix

### Problem

After Changes 1–4 were applied (commit `6766f72`) and training resumed from step 200K, rms_ratio declined again:

| Val step | SI-SDR | rms_ratio | ta_energy |
|----------|--------|-----------|-----------|
| 200K (resume) | -3.87 dB | 1.180 | -23.8 dB |
| 210K | -3.67 dB | 1.038 | -25.5 dB |
| 225K (best) | -2.88 dB | 0.906 | -29.3 dB |
| 235K | -2.98 dB | 0.738 | -28.6 dB |
| 240K | -4.12 dB | 0.752 | -30.7 dB |

The relu fix eliminated the negative-loss exploit, but rms_ratio still drifts from 1.18 → 0.738. The over-suppression pattern recurs, just more slowly.

### Root Cause

Three compounding factors with **insufficient countermeasure** (PhaseSensitiveLoss is scale-sensitive but too weak to anchor amplitude alone):

1. **energy_loss saturates to zero** once TA output < -30 dB (relu clamps to 0). After saturation, TA samples contribute **zero gradient** — no resistance to further suppression.

2. **SI-SDR is scale-invariant** — it zero-means both estimate and target, then computes a ratio. A signal scaled to 0.5x has the exact same SI-SDR. There is **no amplitude anchor** for TP samples anywhere in the current loss.

3. **Shared model weights** learn a suppressive prior from TA training that bleeds into TP outputs, systematically reducing output amplitude.

### Fix: Change 5 — Add `amplitude_loss` for TP samples

An `amplitude_loss` function **already exists** in `losses/separation.py:67-85` but was never used:

```python
def amplitude_loss(estimate, target, eps=1e-8):
    rms_est = estimate.pow(2).mean(dim=-1).sqrt()
    rms_tgt = target.pow(2).mean(dim=-1).sqrt().clamp(min=eps)
    ratio = (rms_est / rms_tgt).clamp(min=0.05)
    return torch.log(ratio).pow(2).mean()
```

Properties:
- Zero gradient at ratio=1.0 (no interference when amplitude is correct)
- Symmetric in log-space (0.5x and 2.0x penalized equally)
- Self-correcting: gradient diminishes as ratio approaches 1.0
- At current rms_ratio=0.738: loss = log(0.738)^2 = 0.092

Applied to **TP samples only** (all samples in Phase 1, tp_flag-masked in Phase 2).

Config: `lambda_amp: 1.0` — conservative. At ratio=0.738 this adds ~0.092 to loss_G, about 2-3% of typical loss_sep magnitude. Large enough for clear gradient signal, small enough not to disrupt SI-SDR optimization.

### Files Modified

| File | Change |
|------|--------|
| `train.py` | Import `amplitude_loss`, compute for TP samples, add to `loss_G`, add `amp_l` logging + TensorBoard |
| `configs/hifi_tse.yaml` | Add `lambda_amp: 1.0` to loss_weights, resume from `checkpoint_0245000.pt` |

### Resume Strategy

Resume from step 245K (not rollback to 200K):
- SI-SDR improved during 200K–245K despite amplitude drift
- amplitude_loss will directly correct the amplitude without undoing separation quality
- Use `--reset-optimizer` to clear Adam momentum shaped by the suppressed-amplitude regime

### Codex Review

- Root cause confirmed. Notes PhaseSensitiveLoss already provides some scale-sensitive signal for TP, but too weak to prevent drift — amplitude_loss is a more targeted anchor.
- lambda_amp=1.0 is conservative; may need tuning upward if recovery is too slow.
- Risks: ratio clamp at 0.05 creates zero-gradient zone below 5% (not an issue at current 0.738). Low-energy TP targets could spike log-ratio — mitigated by existing `rms_tgt.clamp(min=eps)`.
- Must guard with `tp_flag.any()` to avoid empty-tensor issues (already planned).

### Verification

1. After 500 steps: rms_ratio should start increasing from ~0.68
2. After 5K steps: loss_amp should decrease, rms_ratio trending toward 0.9+
3. After 20K steps: rms_ratio should stabilize between 0.9–1.1
4. SI-SDR should not degrade (amplitude_loss has zero gradient at ratio=1.0)

---

## Validation Stabilization Fix (Step ~495K)

### Problem

Validation SI-SDR has extremely high variance: swings from -1.97 to -4.96 dB between consecutive evaluations (5K steps apart). This causes:
- Unreliable best model selection (best checkpoint may be a lucky roll, not genuinely best)
- Early stopping patience counter oscillates instead of giving a clear signal
- Difficult to assess whether training is actually improving

### Root Cause

Two compounding factors in `validate()` (`train.py:138-202`):

1. **Only 50 batches** (line 184: `if count >= 50: break`). With batch_size=2, that's ~100 samples. ~80 TP samples for SI-SDR, ~20 TA samples. Far too few — standard practice is 500-2000+ samples.

2. **Non-deterministic dynamic mixing**. The val dataset is `HiFiTSEDataset(cfg, phase=2)` — same class as training, generating random mixtures on the fly each time. Every validation sees completely different speakers, noise, SNR, and TA/TP splits. This adds random noise on top of the already small sample size.

### Fix: Change 6

Two changes to `train.py`:

1. **Increase validation batches from 50 to 200** — 400 samples total (~320 TP, ~80 TA). 4x more samples reduces standard error by ~2x. Adds ~30-45s per validation (every 5K steps = negligible overhead).

2. **Fix random seed for validation** — seed Python `random`, `numpy`, and `torch` at the start of each `validate()` call with a fixed seed. This ensures the same mixtures are generated every validation, making metrics directly comparable across evaluations. Reset RNG state after validation to avoid affecting training randomness.

### Files Modified

| File | Change |
|------|--------|
| `train.py` | In `validate()`: save RNG states, set fixed seed, increase batch cap to 200, restore RNG states after |

### Expected Impact

- SI-SDR variance between consecutive validations should drop from ~3 dB to ~1 dB or less
- Best model selection becomes meaningful — genuine improvements, not noise
- Early stopping patience counter gives a reliable signal
- No impact on training (RNG states restored after validation)
