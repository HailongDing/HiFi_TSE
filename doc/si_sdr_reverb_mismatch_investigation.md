# SI-SDR Stagnation Investigation: Reverb Mismatch Root Cause

**Date:** 2026-02-11
**Training range:** Steps 100K–138K (Phase 2)
**Symptom:** SI-SDR stuck at -25 to -27 dB despite stable rms_ratio (~0.88–0.97)

---

## Investigation Summary

Three parallel investigations were conducted:
1. Reverb mismatch in training target vs mixture
2. Speaker embedding pathway adequacy
3. Complex mask and model output analysis

## Primary Root Cause: Reverb Mismatch

The dataset has a fundamental contradiction between the mixture and the training target:

```
Mixture    = target_REVERBED + interferers_reverbed + noise
Target     = target_CLEAN (anechoic)
SI-SDR     = si_sdr(estimate, target_CLEAN)
```

### Code Evidence

In `data/dataset.py`:
- **Line 322:** `target_wav = self.clean_index.get_utterance(...)` — reads clean speech from HDF5
- **Line 323-324:** Peak normalize and random crop — still clean
- **Line 346:** `target_reverbed = _apply_rir_np(target_wav, ...)` — reverbed version created
- **Line 359:** `speech_sum = target_reverbed + sum(interferers_reverbed)` — reverbed goes into mixture
- **Line 394:** Returns `target_wav` (the CLEAN version) — not `target_reverbed`

### Why SI-SDR Is Stuck at -27 dB

Even a **perfect model** that perfectly isolates the target from the mixture would output `target_reverbed` (because that's what's in the mix). SI-SDR against `target_clean` would still be ~-27 dB because:
- RIR convolution changes amplitude AND phase of the target
- Reverb tails (0.5–2s decay) create permanent mismatch
- SI-SDR is scale-invariant but NOT reverb-invariant

**-27 dB is not a model failure — it is the theoretical ceiling for this data pipeline.**

*Note:* The -27 dB figure was observed from training metric stagnation (steps 100K–138K), not derived analytically. The actual ceiling depends on RIR characteristics (RT60, direct-to-reverberant ratio) and varies per sample. The pre-training validation step below will empirically confirm this.

### Supporting Evidence

- `evaluate.py` (lines 368-370) explicitly acknowledges this issue with the comment: "Use reverbed target for baseline so SI-SDRi isn't inflated"
- STFT loss IS decreasing (model learning spectral patterns), but SI-SDR is flat (reverb mismatch is the floor)
- Amplitude fixes stabilized rms_ratio at 0.88–0.97 but had zero effect on SI-SDR

## Secondary Finding: Weak Speaker Conditioning

The speaker embedding pathway is undersized relative to the backbone:

| Component | Parameters | Share |
|-----------|-----------|-------|
| USEF cross-attention | 66K | 1.0% |
| Reinject layers (3x) | 199K | 3.0% |
| **Total speaker pathway** | **265K** | **4.0%** |
| TFGridNet backbone | 5.7M | 87% |

- Only 3 re-injection points (blocks 0, 2, 4) out of 6 blocks
- Speaker signal dilutes through residual connections at blocks 1, 3, 5
- Literature (SpEx+, SpeakerBeam) typically re-injects at every block

**This is a real concern but secondary** — we cannot evaluate speaker extraction quality until the reverb mismatch is fixed.

## Fix Options

### Option A: Train against reverbed target (Recommended)

| Mixture | Training Target | Task |
|---------|----------------|------|
| target_reverbed + interf_reverbed + noise | **target_reverbed** | Extract only |

- Standard approach in TSE literature (SpeakerBeam, SpEx+)
- Realistic mixture, model only needs to learn extraction
- Dereverberation can be added as a separate downstream module later
- Expected SI-SDR ceiling: +10 to +20 dB (assuming good extraction)

**Implementation:** Change `data/dataset.py` to return `target_reverbed` instead of `target_clean` for TP samples.

### Option B: Keep current setup (joint extract + dereverb)

| Mixture | Training Target | Task |
|---------|----------------|------|
| target_reverbed + interf_reverbed + noise | target_clean | Extract + dereverberate |

- Current setup — asks model to do two tasks simultaneously
- Very hard for a 7.8M param model
- SI-SDR ceiling limited by reverb mismatch (~-27 dB in current RIR conditions)
- Would need much larger model and longer training

### Option C: Use clean target in mixture

| Mixture | Training Target | Task |
|---------|----------------|------|
| **target_clean** + interf_reverbed + noise | target_clean | Extract only (unrealistic mix) |

- Fixes the mismatch but creates physically unrealistic mixtures
- In real rooms, ALL speakers are reverbed
- Model trained this way would struggle on real reverbed audio at inference

## Decision

**Option A selected.** Return `target_reverbed` as training target. This is the most principled fix:
- Maintains acoustic realism in training data
- Focuses model capacity on speaker extraction (the primary task)
- Common approach in TSE literature (SpeakerBeam, SpEx+), though some larger models (e.g., BSRNN-based) do train with joint extraction+dereverberation
- Requires retraining from scratch

---

## External Review (Opus Agent)

An independent review confirmed the diagnosis and implementation plan, with the following critical additions:

### CRITICAL: Clipping Normalization Must Scale `target_reverbed`

In `data/dataset.py` lines 387–391, the clipping normalization scales `mix_wav` and `target_wav` by the same factor, but `target_reverbed` is **not scaled**:

```python
# Current code (lines 387-391):
peak = np.abs(mix_wav).max()
if peak > 0.95:
    scale = 0.95 / peak
    mix_wav *= scale
    target_wav *= scale    # ← clean target scaled
    # target_reverbed NOT scaled — becomes misaligned with mix_wav
```

If we switch to returning `target_reverbed` as the training target, this scaling **must** also apply to `target_reverbed`. Otherwise the target will have a different amplitude than what's in the mixture, creating a systematic amplitude bias.

*Note:* With multiple reverbed speakers + noise at SNRs as low as -5 dB, clipping (`mix_max > 0.95`) likely triggers on a significant fraction of samples — this is not a rare edge case. The reassignment strategy below handles this automatically.

### Cleanest Implementation Strategy

Instead of adding `target_reverbed` to the clipping normalization block, the cleanest approach is to **replace `target_wav` with `target_reverbed` immediately after line 346**:

```python
target_reverbed = _apply_rir_np(target_wav, rir)
target_wav = target_reverbed  # ← reassign here
```

This way all downstream operations (interferer mixing, clipping normalization, SNR scaling) automatically apply to the reverbed target. No other code changes needed in the dataset pipeline.

### evaluate.py: All Metrics Are Broken (Not Just SI-SDR)

**SI-SDR (line 367):** `compute_si_sdr(est_wav, clean_target)` — measures against clean, artificially depressed.

**SI-SDRi (lines 367–370) has a mixed-reference bug:**
```python
si_sdr_val   = compute_si_sdr(est_wav, clean_target)      # ← clean ref
si_sdr_input = compute_si_sdr(mix_wav, reverbed_target)    # ← reverbed ref
si_sdri      = si_sdr_val - si_sdr_input                   # ← INVALID: different references
```
SI-SDRi requires both terms against the **same** reference to be meaningful. The earlier review statement "this is consistent" was **incorrect**.

**PESQ (line 374) and STOI (line 378)** also measure against `clean_target`. PESQ especially penalizes reverb-vs-clean comparisons heavily. All eval numbers are currently meaningless.

**Fix:** Change all primary metrics to use `reverbed_target`. Optionally keep `clean_target` metrics as secondary for dereverberation tracking.

### TA Branch Interaction

For TA (target-absent) samples, `target_wav` is zeroed at line 370 (`target_wav = np.zeros(...)`). This happens **after** the proposed reassignment at line 347, so TA behavior is preserved — the reassignment is overwritten to zeros as expected. Note: if code is reordered in the future, ensure the TA zeroing still comes after the reverbed reassignment.

### Loss Component Compatibility

All existing loss components work correctly with reverbed targets:
- **SI-SDR**: Will measure extraction quality (ceiling moves from -27 dB to +10–20 dB)
- **STFT loss**: Spectral convergence + log-magnitude — works with any target
- **L1 waveform**: Also had a floor due to reverb mismatch (not just SI-SDR) — fixed by this change
- **Amplitude loss**: `log(rms_est/rms_tgt)^2` — works correctly since both are reverbed
- **GAN losses**: Discriminator learns from reverbed targets — fine for phase 3

### Pre-Training Validation (Recommended)

Before starting full retraining, run a quick sanity check:
- Load 100 samples from the fixed dataset
- Compute `si_sdr(mix, target_reverbed)` — should be ~0 to +5 dB (confirming ceiling is removed)
- Compare with current `si_sdr(mix, target_clean)` — should be ~-27 dB (confirming the mismatch)

This takes <1 minute and confirms the fix is working before committing to a full training run.

### Future Lambda Tuning Notes

- `lambda_sep=0.5`: May need **increasing** back toward 1.0 as SI-SDR improves and raw loss values decrease (currently inflated at -27 dB)
- `lambda_stft=5.0`: May need **reducing** to 2.0–3.0 if STFT loss dominates in late training once SI-SDR is healthy

---

## Impact on Existing Fixes

The over-suppression fixes (amplitude loss, lambda rebalancing) remain valid and should be kept:
- rms_ratio stability at 0.88–0.97 is working correctly
- Log-space amplitude loss with lambda_amp=20.0 is well-tuned
- lambda_sep=0.5 rebalancing is sound

These fixes address amplitude dynamics, which is orthogonal to the reverb mismatch issue. Both fixes are needed for healthy training.

## Second Review Corrections

A second review (`doc/comments_on_si_sdr_reverb_mismatch_investigation.md`) identified several issues in the original document, now corrected:

1. **SI-SDRi mixed-reference bug**: The earlier claim that SI-SDRi was "consistent" was wrong — `si_sdr_val` uses `clean_target` while `si_sdr_input` uses `reverbed_target`, making SI-SDRi mathematically invalid. Fixed in the evaluate.py section above.
2. **PESQ/STOI also broken**: Both compute against `clean_target` — PESQ especially penalizes reverb-vs-clean. Now documented and fixed.
3. **TA branch interaction**: Explicitly verified that the reassignment approach is safe for TA samples. Added note above.
4. **Clipping normalization frequency**: Not a rare edge case — added note about expected trigger rate.
5. **-27 dB derivation**: Clarified this was observed from training stagnation, not analytically derived.
6. **"Standard in TSE literature"**: Qualified — common but not universal; larger models do joint extraction+dereverberation.
7. **L1 waveform loss also affected**: Added to loss compatibility section — not just SI-SDR had a floor.

## Deferred Items

After fixing the reverb mismatch and retraining:
1. Evaluate whether speaker conditioning pathway needs strengthening
2. Consider adding re-injection at all 6 blocks (not just 0, 2, 4)
3. Consider adding a separate dereverberation module after extraction
4. Tune lambda_sep and lambda_stft as SI-SDR improves (see Future Lambda Tuning Notes)
5. Consider joint extraction+dereverberation if model capacity increases
