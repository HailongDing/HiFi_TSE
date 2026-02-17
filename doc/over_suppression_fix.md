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
