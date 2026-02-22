# Implementation Plan: HiFi-TSE v2 (11 Changes) [ALL COMPLETED]

## Context

HiFi-TSE v1 achieved SI-SDRi +4.01 dB at 48kHz with 7.8M params. The v2 codebase is a copy of v1 source code with all 11 changes implemented. Training completed at step 1,500,000 with **SI-SDRi +4.18 dB**.

### Codex o3 Review Summary

**Strengths of original plan:**
- Clear baseline and explicit SI-SDRi objective
- Loss simplification aligns training with target metric
- Capacity and memory strategy is cohesive (scale model, remove GAN, add checkpointing)
- Phased training with verification milestones

**Weaknesses identified:**
- Impact estimates appear additive and optimistic
- Removing all auxiliary losses may reduce stability
- FiLM + all-block reinjection could over-condition (model ignores mixture)
- 1.5M-step run lacks EMA, mixed precision, and LR floor

**Recommended additions:** EMA, mixed precision (bfloat16), min_lr floor, conditioning dropout

**Realistic SI-SDRi estimate:** +4.5 to +5.5 dB (conservative); original +8-10 dB target retained as stretch goal

---

## All 11 Changes

### Original 7 Changes (from improvement_plan.md) — ALL COMPLETED

| # | Change | Files | Status |
|---|--------|-------|--------|
| 1 | Scale TFGridNet: lstm_hidden 192→256 (~7.8M→~11.5M params) | `configs/hifi_tse.yaml` | DONE |
| 2 | Simplify loss: keep SI-SDR + phase only, remove STFT/L1/amp/GAN losses | `configs/hifi_tse.yaml`, `train.py` | DONE |
| 3 | Strengthen speaker conditioning: FiLM + reinject at all 6 blocks | `models/usef.py`, `models/generator.py`, `configs/hifi_tse.yaml` | DONE |
| 4 | Extend training: 500K→1.5M micro-steps, 2 phases (no GAN phase) | `configs/hifi_tse.yaml`, `train.py` | DONE |
| 5 | Remove GAN from training loop (defer to fine-tuning) | `configs/hifi_tse.yaml`, `train.py` | DONE |
| 6 | Speed perturbation augmentation (0.9x-1.1x) | `data/dataset.py` | DONE |
| 7 | Gradient checkpointing on TFGridNet blocks | `models/tf_gridnet.py`, `configs/hifi_tse.yaml` | DONE |

### 4 Additional Changes (from Codex review) — ALL COMPLETED

| # | Change | Files | Status |
|---|--------|-------|--------|
| 8 | EMA of generator weights (decay=0.999) | `train.py` | DONE |
| 9 | Mixed precision (bfloat16, SI-SDR in FP32) | `train.py` | DONE |
| 10 | min_lr floor (1e-5) in cosine schedule | `train.py`, `configs/hifi_tse.yaml` | DONE |
| 11 | Conditioning dropout (ref_dropout_prob=0.1) | `models/generator.py`, `configs/hifi_tse.yaml` | DONE |

---

## Current Code State (v1 baseline)

Verified by code inspection:
- **Gradient clipping**: Yes, grad_clip=5.0 (train.py)
- **Mixed precision**: No
- **EMA**: No
- **LR schedule**: Linear warmup 2000 steps + cosine decay to 0
- **Loss terms**: 6 (SI-SDR, multi-res STFT, phase, L1, amplitude, + GAN in Phase 3)
- **GAN**: Full implementation with MPD+MSD discriminator (70.7M params)
- **Speed perturbation**: No
- **Gradient checkpointing**: No
- **FiLM conditioning**: No
- **Reinject at**: [0, 2, 4]

---

## Implementation Order

### Step 1: `configs/hifi_tse.yaml`
All config changes in one pass:
- `lstm_hidden`: 192 → 256
- `reinject_at`: [0, 2, 4] → [0, 1, 2, 3, 4, 5]
- `use_checkpoint`: true (new)
- `ref_dropout_prob`: 0.1 (new)
- `loss_weights`: keep only `lambda_sep: 1.0`, `lambda_phase: 0.5`, `ta_weight: 0.1`
- `total_steps`: 500000 → 1500000
- `warmup_steps`: 2000 → 5000
- `phase1_steps`: 100000 → 200000
- `phase2_steps`: 1500000
- Remove GAN config sections
- `speed_perturb`: true (new)
- `min_lr`: 1e-5 (new)
- `ema_decay`: 0.999 (new)

### Step 2: `models/usef.py` — Add FiLMLayer class
- gamma/beta projection from time-pooled speaker embedding (~65K params)
- **Correct init** (per review): `zeros_(gamma_proj.weight)` + `ones_(gamma_proj.bias)` so `gamma_proj(any_input) = 1`; `zeros_(beta_proj.weight)` + `zeros_(beta_proj.bias)` so `beta_proj(any_input) = 0`. This produces identity transform `1 * z_cond + 0 = z_cond` at init.

### Step 3: `models/generator.py` — FiLM + conditioning dropout
- Import and instantiate FiLMLayer, call after USEF
- **Conditioning dropout scope** (per review): Zero out z_ref before USEF, which disables ALL conditioning paths (USEF cross-attention + reinject layers + FiLM) simultaneously. This is intentionally aggressive — forces the model to learn some separation even without speaker reference, preventing over-reliance on conditioning. Prob=0.1 means 90% of samples still get full conditioning.

### Step 4: `models/tf_gridnet.py` — Gradient checkpointing
- `torch.utils.checkpoint.checkpoint` on GridNet blocks when training

### Step 5: `data/dataset.py` — Speed perturbation
- **Use scipy.signal.resample** (per review): consistent with existing numpy pipeline, avoids numpy→tensor→sox→numpy overhead. Resample to achieve speed factor, then resample back to original rate.

### Step 6: `train.py` — Major rewrite
1. Remove GAN (discriminator, opt_D, sched_D, GAN losses, GAN logging)
2. Simplify loss to SI-SDR + phase only
3. **Add EMA** (per review): Use separate `AveragedModel` instance for validation — no weight swapping needed. Validate with EMA model, save EMA state in checkpoints.
4. **Add AMP** (per review): bfloat16 autocast for forward pass. SI-SDR computed in FP32 via `autocast(enabled=False)`. PhaseSensitiveLoss uses `register_buffer` for windows (already FP32), and `torch.stft` will use input dtype — ensure estimate/target are cast to float32 before phase loss.
5. Add min_lr floor in cosine schedule
6. Update checkpoint save/load (add EMA, remove discriminator)
7. Simplify to 2 phases (Phase 1: TP-only 0-200K, Phase 2: TP+TA 200K-1.5M)
8. Update TensorBoard logging

### Step 7: `monitor_training.sh` — Remove GAN patterns

### Step 8: Smoke test (100 steps)

### Step 9: Start training in tmux

---

## Files Modified

| File | Scope |
|------|-------|
| `configs/hifi_tse.yaml` | Extensive config updates |
| `train.py` | Major: remove GAN, simplify loss, add EMA/AMP/min_lr |
| `models/tf_gridnet.py` | Small: gradient checkpointing |
| `models/usef.py` | Small: add FiLMLayer class |
| `models/generator.py` | Small: FiLM integration + ref dropout |
| `data/dataset.py` | Small: speed perturbation |
| `monitor_training.sh` | Minor: remove GAN patterns |

---

## Verification — ALL PASSED

1. **Smoke test (100 steps)**: PASSED — Model instantiates, loss decreasing, no OOM, checkpoint round-trips
2. **Param count**: PASSED — 11.79M generator params
3. **Memory check**: PASSED — 6.5 / 24.5 GB GPU memory
4. **Early milestone (50K steps)**: PASSED — SI-SDR improving
5. **Final evaluation**: PASSED — SI-SDRi **+4.18 dB** (exceeds v1 baseline +4.01 dB)

## Final Results (Step 1,500,000)

| Metric | v1 | v2 |
|--------|----|----|
| SI-SDRi | +4.01 dB | **+4.18 dB** |
| SI-SDR | -1.87 dB | **-1.71 dB** |
| PESQ | 1.15 | 1.14 |
| STOI | 0.488 | 0.489 |
| TA suppression | 11.2 dB | **11.4 dB** |
| Model params | 7.8M | 11.79M |
| Training steps | 480K | 1,500K |
