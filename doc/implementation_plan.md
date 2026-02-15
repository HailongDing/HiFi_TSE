# Implementation Plan: HiFi-TSE v2 (11 Changes)

## Context

HiFi-TSE v1 achieved SI-SDRi +4.01 dB at 48kHz with 7.8M params. The v2 codebase is a copy of v1 source code with no changes applied yet. The goal is to implement 7 planned improvements (from `improvement_plan.md`) plus 4 additional improvements identified during Codex o3 review, then train from scratch.

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

### Original 7 Changes (from improvement_plan.md)

| # | Change | Files |
|---|--------|-------|
| 1 | Scale TFGridNet: lstm_hidden 192→256 (~7.8M→~11.5M params) | `configs/hifi_tse.yaml` |
| 2 | Simplify loss: keep SI-SDR + phase only, remove STFT/L1/amp/GAN losses | `configs/hifi_tse.yaml`, `train.py` |
| 3 | Strengthen speaker conditioning: FiLM + reinject at all 6 blocks | `models/usef.py`, `models/generator.py`, `configs/hifi_tse.yaml` |
| 4 | Extend training: 500K→1.5M micro-steps, 2 phases (no GAN phase) | `configs/hifi_tse.yaml`, `train.py` |
| 5 | Remove GAN from training loop (defer to fine-tuning) | `configs/hifi_tse.yaml`, `train.py` |
| 6 | Speed perturbation augmentation (0.9x-1.1x) | `data/dataset.py` |
| 7 | Gradient checkpointing on TFGridNet blocks | `models/tf_gridnet.py`, `configs/hifi_tse.yaml` |

### 4 Additional Changes (from Codex review)

| # | Change | Files | Rationale |
|---|--------|-------|-----------|
| 8 | EMA of generator weights (decay=0.999) | `train.py` | Smoother val/checkpoint weights for 1.5M-step run |
| 9 | Mixed precision (bfloat16, SI-SDR in FP32) | `train.py` | ~30% faster training; SI-SDR log10/ratio ops need FP32 |
| 10 | min_lr floor (1e-5) in cosine schedule | `train.py`, `configs/hifi_tse.yaml` | Prevents LR hitting zero too early in 3x longer run |
| 11 | Conditioning dropout (ref_dropout_prob=0.1) | `models/generator.py`, `configs/hifi_tse.yaml` | Prevents over-conditioning from FiLM + all-block reinject |

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
- Identity init (gamma→1, beta→0) for stable training start

### Step 3: `models/generator.py` — FiLM + conditioning dropout
- Import and instantiate FiLMLayer, call after USEF
- Add ref dropout: zero out z_ref per-sample with `ref_dropout_prob` during training

### Step 4: `models/tf_gridnet.py` — Gradient checkpointing
- `torch.utils.checkpoint.checkpoint` on GridNet blocks when training

### Step 5: `data/dataset.py` — Speed perturbation
- torchaudio sox_effects, 0.9x-1.1x on clean speech before mixing

### Step 6: `train.py` — Major rewrite
1. Remove GAN (discriminator, opt_D, sched_D, GAN losses, GAN logging)
2. Simplify loss to SI-SDR + phase only
3. Add EMA (AveragedModel, decay=0.999)
4. Add AMP (bfloat16 autocast, SI-SDR in FP32)
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

## Verification

1. **Smoke test (100 steps)**: Model instantiates, loss decreasing, no OOM, checkpoint round-trips
2. **Param count**: Verify ~11.5M generator params (no discriminator)
3. **Memory check**: batch_size=2 + lstm_hidden=256 + grad_checkpoint fits in 24GB RTX 4090
4. **Early milestone (50K steps)**: SI-SDR improving, EMA val loss tracking
5. **Final evaluation**: evaluate.py on best EMA checkpoint, target SI-SDRi > +4.01 dB
