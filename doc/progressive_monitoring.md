# Progressive Training Monitoring

**Commit**: `6d464ef` (2026-02-09)
**Files modified**: `train.py`, `monitor_training.sh`

## Problem

Model trained 415K steps before evaluation revealed severe over-suppression (output at 8% amplitude, SI-SDRi -19.6 dB). The issue was invisible because training logs only showed losses — SI-SDR is scale-invariant and looks fine even with 8% amplitude output. No amplitude-aware metric was logged during training.

## Changes

### 1. rms_ratio in Training Log [CRITICAL]

Added `rms_ratio = est_rms / target_rms` computed every step, logged every `log_interval` (100 steps).

- Healthy model: rms_ratio 0.8-1.2
- Over-suppressed model: rms_ratio ~0.08
- Appears as `rms X.XX` in console log line
- Logged to TensorBoard as `train/rms_ratio`
- Cost: ~0.01ms/step, negligible

### 2. Enhanced validate() with Decomposed Metrics

Replaced single `scene_aware_loss` scalar with:
- **SI-SDR (dB)**: per-sample quality metric for TP samples
- **rms_ratio**: amplitude preservation check
- **ta_energy (dB)**: target-absent suppression level

Print format: `Validation loss: X.XX | si_sdr: X.XX dB | rms_ratio: X.XXX | ta_energy: X.X dB`

All metrics logged to TensorBoard under `val/`.

### 3. Protected Phase Boundary Checkpoints

Phase boundary checkpoints (step 100K, 300K) are never deleted by rotation. Added `protected_steps` parameter to `save_checkpoint()` and `_step_from_filename()` helper.

This ensures we always have a checkpoint from right before each phase transition for rollback.

### 4. Mini-Evaluation at Milestones

New `mini_evaluate()` function runs at steps `{5000, 100K, 200K, 300K, 400K, 500K}`:
- 10 val batches (~20 samples)
- Reports SI-SDR + rms_ratio
- Prints `MILESTONE_EVAL step N | si_sdr: X.XX dB | rms_ratio: X.XXX`
- At step 5000: fires `EARLY_CHECK WARNING` if rms_ratio < 0.3
- Cost: ~5 seconds per milestone

### 5. Real-Audio Inference Check at Milestones

New `real_audio_check()` function at each milestone:
- Loads `real_audio/mix_16k.wav` + `ref_16k.wav`
- Resamples to 48kHz, runs model inference
- Logs `est_rms / mix_rms` ratio
- Saves output wav to `checkpoints/real_audio_step_XXXXXXX.wav`
- Prints `REAL_AUDIO_CHECK step N | est_rms: X.XX | mix_rms: X.XX | ratio: X.XXX`
- Cost: ~2 seconds per milestone, ~25MB disk total

### 6. Monitor Script Integration

New patterns in `monitor_training.sh`:
- `MILESTONE_EVAL` → WeCom notification
- `REAL_AUDIO_CHECK` → WeCom notification
- `EARLY_CHECK WARNING` → urgent WeCom warning with action guidance
- `rms < 0.3` in training log → over-suppression alert

## Detection Timeline

| Step | What Fires | Catches |
|------|-----------|---------|
| 0-100 | rms_ratio in training log | Immediate amplitude check (~1.0 with identity init) |
| 5,000 | MILESTONE_EVAL + early warning | Over-suppression (rms < 0.3), SI-SDR baseline |
| 5K-100K | rms_ratio every 100 steps, validation every 5K | Amplitude drift, loss plateau |
| 100,000 | Protected checkpoint + milestone eval + real audio | Phase 2 transition health |
| 300,000 | Protected checkpoint + milestone eval + real audio | Pre-GAN baseline |
| 300K-500K | All above + best-model tracking + early stopping | GAN phase regression |

## What This Does NOT Include

- **No PESQ/STOI during training**: Too slow, SI-SDR + rms_ratio sufficient for early detection
- **No auto-stopping on anomaly**: Transient dips during phase transitions could cause false positives; alerts only, human decides
- **No subprocess evaluate.py**: Would OOM (two model instances on 24GB GPU)
