# SI-SDR Stagnation Investigation (Phase 2)

**Date:** 2026-02-12
**Training range:** Steps 0–141K (Phase 1: 0–100K, Phase 2: 100K–141K)
**Symptom:** SI-SDR (sep) stagnated at -3 to -5 dB throughout Phase 2, no improvement over Phase 1

---

## Background

### Project

HiFi-TSE is a 48kHz Target Speaker Extraction system built on TFGridNet (7.8M params generator) with GAN-based enhancement (70.7M params discriminator). Given a mixture of reverbed speakers + noise and a reference utterance from the target speaker, the model extracts the target speaker's reverbed speech.

### Training Configuration

- **3-phase curriculum**: Phase 1 (0–100K): TP-only SI-SDR loss; Phase 2 (100K–300K): +TA/energy/amplitude losses; Phase 3 (300K+): +GAN
- **Optimizer**: AdamW (lr=2e-4, betas=[0.8, 0.99]), grad_accum=32, cosine decay with 2K-step warmup
- **Total**: 500K micro-steps = 15,625 optimizer steps
- **Config**: `configs/hifi_tse.yaml`

### Prior Fixes Applied Before This Training Run

1. **Analysis-1 fixes** (commit `18193d5`): 8 architectural fixes including speaker re-injection at blocks [0,2,4], LR warmup, DataLoader recreation at phase transitions.

2. **Analysis-2 fixes** (7 code review fixes): Complex ratio mask proper multiplication, loss weighting (`ta_weight=0.2`), STFT window register_buffer, worker seeds, checkpoint rotation, eval SI-SDRi baseline.

3. **Over-suppression fix v1** (commit `69c51f7`): Added quadratic amplitude loss, TA energy floor at -40 dB, boosted `lambda_stft` to 5.0.

4. **Over-suppression fix v2** (commit `b758e15`): Changed amplitude loss to log-space L2 (`log(ratio)^2`), reduced `lambda_sep` from 1.0 to 0.5, increased `lambda_amp` to 20.0. This fixed rms_ratio (0.85–0.97) but introduced the loss imbalance that contributes to the current stagnation.

5. **Reverb mismatch fix** (commit `21e94b2`): Changed dataset to return `target_reverbed` instead of `target_clean`. Fixed evaluate.py to measure all metrics against reverbed target. Sanity check confirmed ceiling moved from -27 dB to -7.5 dB. **This is the current training run** — started from scratch after this fix.

### Current Loss Weights

```yaml
lambda_sep: 0.5    # reduced from 1.0 in over-suppression fix v2
lambda_stft: 5.0   # boosted in over-suppression fix v1
lambda_l1: 0.5
lambda_amp: 20.0   # log-space L2 amplitude loss
lambda_adv: 0.1    # Phase 3 only
lambda_fm: 2.0     # Phase 3 only
ta_weight: 0.1
```

### Training Progress Leading to This Investigation

- **Steps 0–87K (Phase 1)**: Sep slowly improved from ~6.3 to ~3.7 (slope -0.22/10K steps). RMS stuck at 0.58 (no amplitude loss in Phase 1). STFT improved from 1.83 to 1.10.
- **Steps 100K–105K (Phase 2 start)**: Amplitude loss kicked in, rms recovered from 0.65 to 0.89. Sep briefly improved.
- **Steps 105K–141K (Phase 2)**: Sep stagnated at 2.2–3.6, oscillating with no trend (slope ≈ 0). STFT flat at ~1.08. RMS healthy at 0.86–1.0. Validation SI-SDR confirmed stagnation at -3 to -6 dB.

The reverb mismatch fix was validated (old ceiling was -27 dB, now removed), but the model's actual separation ability plateaued at -3 to -5 dB SI-SDR, prompting this investigation.

---

## Investigation Summary

Four parallel investigations conducted:
1. Loss component balance analysis
2. Learning rate schedule and optimizer state
3. Speaker conditioning pathway architecture
4. Validation metrics and checkpoint analysis

---

## Finding 1: SI-SDR is Gradient-Starved

Weighted contribution to `loss_G` in Phase 2:

| Component | Lambda | Raw Value (mean) | Weighted | % of loss_G |
|-----------|--------|-------------------|----------|-------------|
| **sep** | 0.5 | ~3.0 | ~1.5 | **13–15%** |
| **stft** | 5.0 | ~1.1 | ~5.5 | **50–61%** |
| amp | 20.0 | ~0.15 | ~3.0 | 25–35% |
| l1 | 0.5 | ~0.04 | ~0.02 | 0.2% |

SI-SDR provides only ~14% of the gradient signal. The optimizer primarily optimizes STFT (spectral shape) and amplitude matching. The `lambda_sep=0.5` reduction (from the over-suppression fix v2) combined with `lambda_stft=5.0` created this severe imbalance.

### Phase 2 Windowed Breakdown

| Window | sep % | stft % | amp % | mean loss_G |
|--------|-------|--------|-------|-------------|
| 100K–110K | 13.8% | 56.6% | 29.3% | 8.72 |
| 110K–120K | 14.3% | 54.4% | 31.0% | 9.95 |
| 120K–130K | 14.7% | 50.6% | 34.5% | 10.72 |
| 130K–142K | 13.4% | 60.7% | 25.6% | 9.28 |

STFT dominates at 50–61%. Sep has been squeezed to ~14% — effectively gradient-starved.

---

## Finding 2: Speaker Conditioning is Critically Weak (3.41% of params)

### Parameter Breakdown

| Component | Parameters | Share |
|-----------|-----------|-------|
| TFGridNet backbone | 6.91M | 88.7% |
| BandSplitEncoder | 311K | 4.0% |
| BandMergeDecoder | 302K | 3.9% |
| **USEF cross-attention** | **66K** | **0.9%** |
| **SpeakerReinjectLayers (3x)** | **199K** | **2.6%** |
| **Total speaker pathway** | **265K** | **3.41%** |

Typical TSE systems allocate 10–15% to speaker pathway. HiFi-TSE is **3–4x under-parameterized**.

### Eight Critical Weaknesses

1. **Severe under-parameterization (3.41% vs 10–15%)**: Only 265K of 7.8M parameters dedicated to speaker pathway. Typical systems: 400K–600K in speaker encoder alone.

2. **Sparse speaker injection (50% of blocks)**: Only blocks [0, 2, 4] get speaker reinject. Blocks [1, 3, 5] have none.
   - Block 0: ✓ reinject
   - Block 1: ✗
   - Block 2: ✓ reinject
   - Block 3: ✗
   - Block 4: ✓ reinject
   - Block 5: ✗

3. **Fixed reference embeddings**: `z_ref` is created once after BandSplitEncoder and reused at all injection points without adaptation. Never updates to match evolving mixture features.

4. **Early fusion → dilution through 18 LSTM layers**: Speaker info injected at USEF then processed through 18 BiLSTM layers with forget gates. By Block 5, ~88% of speaker signal is diluted.

5. **Per-band processing loses cross-band patterns**: Both USEF and SpeakerReinjectLayers reshape `(B, T, 53, 128)` to `(B*53, T, 128)`. Cross-band speaker characteristics (pitch formants, harmonics) cannot be captured.

6. **No dedicated speaker encoder**: Uses raw reference spectrogram without learned disentanglement. Typical TSE systems have a dedicated Conv1D speaker encoder.

7. **Residual connection bottleneck**: `x_out = LayerNorm(x + attn_out)` — if attention learns near-zero output (early training), speaker info is completely bypassed.

8. **Temporal misalignment**: `z_ref` has fixed T_ref = 401 frames. When reference is shorter, model must extrapolate.

### Direct Connection to SI-SDR Stagnation

This explains the observed behavior:
- **STFT improves ✓**: Model learns spectral shape from mixture — doesn't need speaker identity for this
- **SI-SDR stagnates ✗**: Model can't use speaker embedding to identify *which* speaker to extract
- The model converges to a "mixture spectral shape matcher" rather than a "target speaker extractor"

---

## Finding 3: Validation Confirms Stagnation

28 validation runs across training (every 5K optimizer steps):

| Phase | Val SI-SDR Range | Val SI-SDR Mean | Val RMS Ratio |
|-------|-----------------|-----------------|---------------|
| Phase 1 early (5K–50K) | -3.05 to -5.94 | -4.60 | 0.54–0.95 |
| Phase 1 late (55K–100K) | -3.05 to -5.37 | -4.13 | 0.59–0.70 |
| Phase 2 (105K–140K) | -3.01 to -5.64 | -4.24 | 0.77–1.02 |

Key observations:
- Val SI-SDR has been **bouncing between -3 and -6 dB for the entire training run**
- No improvement from Phase 1 to Phase 2 — rules out logging artifacts
- Best val SI-SDR: -3.01 dB (step 105K), worst: -5.94 dB (step 10K)
- RMS ratio improved in Phase 2 (amplitude loss working) but didn't help SI-SDR
- Phase 2 val SI-SDR is actually slightly worse than late Phase 1 (-4.24 vs -4.13)

### Selected Validation Results

| Step | Val Loss | SI-SDR (dB) | RMS Ratio | TA Energy (dB) |
|------|----------|-------------|-----------|----------------|
| 5000 | 3.290 | -5.58 | 1.762 | -18.8 |
| 30000 | 0.928 | -3.73 | 0.544 | -28.8 |
| 60000 | -0.452 | -4.12 | 0.548 | -33.0 |
| 80000 | 0.659 | -3.05 | 0.641 | -30.9 |
| 100000 | 0.745 | -3.38 | 0.594 | -29.1 |
| 105000 | 0.575 | -3.01 | 0.937 | -25.3 |
| 125000 | 1.433 | -3.29 | 0.972 | -27.2 |
| 140000 | 2.200 | -5.48 | 1.024 | -24.6 |

---

## Finding 4: LR Schedule is Not the Issue

| Parameter | Value |
|-----------|-------|
| Base LR | 0.0002 |
| Optimizer | AdamW (betas=[0.8, 0.99]) |
| Grad accumulation | 32 |
| Total optimizer steps | 15,625 |
| Current optimizer steps | ~4,400 |
| Current LR | ~0.000198 (99.2% of base) |
| Warmup | 2,000 optimizer steps (linear) |
| Decay | Cosine, currently at 17.6% progress |

LR is still near peak. No optimizer reset at phase transitions. Momentum from Phase 1 carries into Phase 2. LR schedule is not limiting.

---

## Finding 5: Mixed-Batch Noise in Training Logs

With batch_size=2 and ta_ratio=0.2 in Phase 2:
- P(both TP) = 64% → sep = 0.9 × (−SI-SDR)
- P(1 TP + 1 TA) = 32% → sep contaminated by TA energy term (~-2.5)
- P(both TA) = 4% → sep = TA energy only

This makes the training log's `sep` column noisy (~32% of values contaminated), but **does not affect actual training** — validation SI-SDR (TP-only, per-sample) independently confirms the stagnation.

---

## Training Sep Trend (10K windows, Phase 2)

| Window | sep mean | sep p50 | stft | rms mean | SI-SDR > 0 dB |
|--------|----------|---------|------|----------|---------------|
| 100K–110K | 2.61 | 2.92 | 1.069 | 0.915 | 26.0% |
| 110K–120K | 2.85 | 3.07 | 1.083 | 0.959 | 23.0% |
| 120K–130K | 3.15 | 3.50 | 1.086 | 0.958 | 25.0% |
| 130K–140K | 2.24 | 2.28 | 1.139 | 0.990 | 31.0% |

Phase 2 sep linear trend slope: **+0.006 per 10K steps** — essentially flat.

For comparison, Phase 1 reference (90–100K): sep=3.62, rms=0.647, SI-SDR > 0: 15%.

---

## Correlation Analysis (Phase 2)

| Pair | Correlation | Interpretation |
|------|------------|----------------|
| sep vs rms | +0.616 | Higher rms → worse SI-SDR (model passing interference) |
| rms vs amp | +0.732 | Expected — amp directly measures rms deviation |
| sep vs stft | +0.590 | Hard samples are hard for both metrics |

Samples with rms < 0.8: sep mean = 0.80, 44% have SI-SDR > 0 dB
Samples with rms > 1.0: sep mean = 5.78, 3% have SI-SDR > 0 dB

Low-RMS samples paradoxically have better SI-SDR — likely because the model is suppressing everything (including interference) rather than selectively extracting. This is consistent with weak speaker conditioning.

---

## Root Cause Summary

Two interacting problems:

### Primary: Weak Speaker Conditioning (Architectural)

The model has only 265K parameters (3.41%) dedicated to speaker identification and conditioning. It cannot effectively use the reference audio to guide target speaker extraction. The model converges to a "mixture spectral shape matcher" — it learns to produce reasonable spectral envelopes (STFT improves) but cannot distinguish which speaker to extract (SI-SDR stagnates). This is likely an architectural limitation, though this needs confirmation via the diagnostic experiment below.

*Note:* The specific claim that "88% of speaker signal is diluted by Block 5" is a qualitative estimate, not a measured value. LSTM forget gate behavior is learned and varies. The directional concern (signal dilution through deep layers) is valid, but the specific percentage is uncertain. Similarly, the "10–15% typical allocation" for speaker pathway in TSE literature varies widely across systems.

### Secondary: Sep Gradient Starvation (Loss Configuration)

At 13–15% of loss_G, the SI-SDR gradient is drowned out by STFT (50–61%) and amplitude (25–35%). Even if the speaker pathway were stronger, the optimizer would prioritize spectral and amplitude matching over extraction quality.

*Note:* In Phase 2, the effective SI-SDR weight is further reduced by `scene_aware_loss` which scales the TP SI-SDR by `(1 - ta_weight) = 0.9`. So the effective weight is `0.5 × 0.9 = 0.45`, making the starvation slightly worse than the raw lambda suggests.

---

## External Review (Opus Agent)

An independent review confirmed both root causes and identified several additional issues:

### CRITICAL: SI-SDR Ceiling Clarification Needed

The sanity check showed SI-SDR(mix, target_reverbed) = **-7.5 dB**. This is the *input* SI-SDR — how well the mixture already matches the target before any processing. The model at -3 to -5 dB val SI-SDR is actually **worse than the unprocessed mixture**. This means the model is actively degrading the target signal rather than extracting it. This is consistent with the model being a "spectral shape matcher" that distorts the signal while trying to match spectral patterns.

### CRITICAL: Run Speaker Reference Swap Test First

Before committing to any fix, run a diagnostic experiment:
1. Take the best checkpoint (step 105K, val SI-SDR = -3.01 dB)
2. Evaluate on the validation set with **correct** speaker references
3. Evaluate again with **swapped** references (wrong speaker)
4. Compare SI-SDR between the two:
   - **If similar**: Model is ignoring the speaker embedding entirely → architecture fix needed
   - **If worse with wrong reference**: Model IS using the embedding → gradient starvation is the bottleneck, loss rebalancing may suffice

This takes ~30 minutes and determines the entire fix strategy.

### More Moderate Loss Rebalancing Suggested

The review recommends more conservative values than originally proposed:

```yaml
lambda_sep: 1.5    # up from 0.5 (3x, not 4-6x)
lambda_stft: 2.5   # down from 5.0 (2x, not 2.5-5x)
lambda_amp: 10.0   # down from 20.0 (amp is healthy, reduce dominance)
lambda_l1: 0.5     # unchanged
```

Estimated balance: sep ~35%, stft ~22%, amp ~12%, l1 ~0.2%.

**Important**: Add a gradual weight transition over 10K steps to avoid optimizer shock. Adam's momentum terms are adapted to the current loss landscape; sudden weight changes create gradient distribution shift.

### FiLM Conditioning Preferred Over More Cross-Attention

For speaker pathway strengthening, Feature-wise Linear Modulation (FiLM) is recommended over adding more per-band cross-attention layers:

```python
# FiLM: x = gamma * x + beta
# gamma, beta derived from global speaker embedding
speaker_emb = speaker_encoder(z_ref)  # (B, 256) global embedding
gamma = linear_gamma(speaker_emb)     # (B, D) per-feature scale
beta = linear_beta(speaker_emb)       # (B, D) per-feature shift
x = gamma.unsqueeze(1).unsqueeze(2) * x + beta.unsqueeze(1).unsqueeze(2)
```

Advantages over cross-attention:
- Captures **cross-band** speaker patterns (global embedding averages across bands)
- More parameter-efficient (no Q/K/V projections per band)
- Stronger conditioning signal (multiplicative, not additive)
- Applied at every block trivially

### Additional Issues Identified

1. **No dedicated speaker encoder**: The reference audio goes through the same BandSplitEncoder as the mixture (shared weights). This encoder is designed for separation features, not speaker identification. A lightweight dedicated encoder (Linear layers + temporal average pooling → 256-dim global embedding) would provide cleaner speaker representations.

2. **Reference audio has different RIR than target** (`dataset.py` line 347 vs 381): When `noisy_ref_prob=0.5`, half the references have reverb from a *different* room than the target. Matching speakers across different room acoustics requires learning room-invariant features — very hard without a dedicated speaker encoder.

3. **Gradient clipping interaction**: With `grad_clip=5.0`, if STFT and amplitude losses produce large gradients that trigger clipping, the SI-SDR gradient (already a minority) gets further suppressed proportionally. Recommend logging gradient norms before/after clipping to verify.

4. **Validation set may not be properly held out**: `val_split_ratio: 0.05` in config may not be implemented in the dataset code — needs verification. If validation overlaps with training, val metrics may be unreliable.

5. **Local minimum risk**: The identity mask initialization (BandMergeDecoder) starts the model as passthrough. If suppression is rewarded more than selective extraction early in training (because STFT+amp dominate), the model may be stuck in a suppression-oriented basin. This is a strong argument for having strong SI-SDR signal from the very beginning of training, not just after rebalancing.

---

## Second External Review (Codex MCP)

A second independent review via Codex MCP confirmed all findings and identified additional root causes and refinements.

### Additional Root Causes Identified

6. **Magnitude-only STFT loss + identity mask init = phase basin trap**: The STFT loss (`losses/stft_loss.py:37`) penalizes magnitude errors only, not phase. Combined with identity-initialized masks (`models/band_split.py:93`) and small `lambda_sep`, the model can converge to a "mixture-phase basin" — correct magnitude but wrong phase. SI-SDR is the only loss sensitive to phase errors, but at 14% of loss_G it lacks the gradient strength to escape this basin.

7. **RIR truncation weakens conditioning cues**: `_apply_rir_np` (`data/dataset.py:67`) uses `fftconvolve(mode="full")[:len(wav)]`, truncating reverb tails. Each source and the reference get independent random RIRs, so there are no consistent room cues that the model could exploit for conditioning. This makes speaker matching across different acoustic conditions harder.

8. **Unconstrained complex mask**: The complex mask in `band_split.py:88` uses linear projections with no magnitude or phase constraints, allowing arbitrarily large phase rotations. SI-SDR must correct what STFT/amp losses ignore, but it's too weak to do so.

9. **Phase transition momentum carry-over**: Adam optimizer state carries across the Phase 1→2 boundary (`train.py:376, 407`). The momentum terms accumulated under the Phase 1 loss landscape (SI-SDR only) become misaligned when Phase 2 adds amplitude/TA losses, slowing adaptation to the new loss distribution.

10. **Speaker conditioning modules lack FFN**: USEF and SpeakerReinjectLayer are MHA+LayerNorm only (no feedforward network) (`models/usef.py:25`, `models/tf_gridnet.py:85`). This limits the non-linear transformation capacity of the speaker pathway — the attention output has no further processing before being added to the mixture features.

### Refined Fix Suggestions

**Step 0 refinement — Add zero-ref control:**
- In addition to the wrong-speaker swap test, also test with **zeroed reference** (feed zeros as ref_wav). This gives three conditions:
  - Correct reference → baseline SI-SDR
  - Wrong speaker reference → measures speaker discrimination
  - Zero reference → measures how much the model relies on any reference at all
- If zero-ref ≈ correct-ref, the model completely ignores the conditioning pathway.

**Step 1 refinement — Phase-aware loss and optimizer reset:**
- Add a **phase-sensitive loss** (complex STFT or phase-sensitive L1) to escape the magnitude-only basin. This directly addresses root cause #6.
- Consider **GradNorm or Dynamic Weight Averaging (DWA)** instead of manual lambda tuning — automatically balances gradient magnitudes across losses.
- **Reset Adam state** (or warm-restart) at phase boundaries to eliminate momentum carry-over (root cause #9).

**Step 2 refinement — Pretrained speaker encoder:**
- Use a **pretrained speaker encoder** (ECAPA-TDNN or d-vector) instead of training one from scratch. This provides immediately useful speaker representations.
- Add a **speaker verification auxiliary loss** on the embedding to ensure it captures speaker identity.
- **Reduce `noisy_ref_prob`** or **match RIR between target and reference** early in training to strengthen conditioning before making it harder.

**Pre-Step 2 data fixes (before architecture changes):**
- **Shared RIR per scene**: Use the same RIR for target + interferers within a mixture (different speakers still have independent RIRs in real rooms, but this simplifies the initial learning problem).
- **Convolve-then-crop** instead of crop-then-convolve: Apply RIR to the full utterance, then crop to segment length, to preserve reverb tails within the segment.
- **Explicit TP/TA batch composition**: Guarantee each micro-batch has 1 TP + 1 TA (or all TP) to reduce gradient variance from random composition.
- **Monitor gradient norms** before/after `clip_grad_norm_` to quantify how much clipping suppresses SI-SDR gradients.

---

## Recommended Fixes (Revised After Both Reviews)

### Step 0 — Speaker Conditioning Diagnostic (~30 min, no code changes)

Run three evaluation conditions on the best checkpoint (step 105K):
1. **Correct reference** — baseline SI-SDR
2. **Wrong speaker reference** — measures speaker discrimination ability
3. **Zero reference** (feed zeros as ref_wav) — measures reliance on any reference

Interpretation:
- If correct ≈ wrong ≈ zero: Model completely ignores conditioning → architecture fix mandatory
- If correct > wrong > zero: Model uses conditioning but weakly → loss rebalancing may help
- If correct >> wrong: Model has strong discrimination → stagnation is from gradient starvation, not architecture

### Step 1 — Loss Rebalancing + Phase-Aware Loss (resume from checkpoint)

```yaml
lambda_sep: 1.5      # up from 0.5 (target ~35% of loss_G)
lambda_stft: 2.5     # down from 5.0
lambda_amp: 10.0     # down from 20.0 (rms is healthy, reduce dominance)
```

Additional changes:
- **Gradual weight transition** over 10K steps to avoid optimizer shock
- **Add phase-sensitive loss** (complex STFT or phase-sensitive L1) to escape magnitude-only basin
- Consider **resetting Adam state** at the rebalancing point to eliminate stale momentum
- **Log gradient norms** before/after clipping to verify starvation hypothesis
- Expected sep contribution: ~35% (up from 14%)

### Step 1.5 — Data Pipeline Fixes (can combine with Step 1 or 2)

Before architecture changes, try these data-level improvements:
- **Convolve-then-crop**: Apply RIR to full utterance, then crop to segment length — preserves reverb tails
- **Reduce `noisy_ref_prob`** early in training (e.g., 0.0 for first 100K, then ramp to 0.5) — makes conditioning easier initially
- **Explicit TP/TA batch composition**: Guarantee each micro-batch has at least 1 TP sample to reduce gradient variance

### Step 2 — Speaker Encoder + FiLM (retrain from scratch)

- **Pretrained speaker encoder** (ECAPA-TDNN or d-vector) preferred over training from scratch — provides immediately useful speaker representations
- FiLM conditioning (`x = gamma * x + beta`) at all 6 blocks
- **Speaker verification auxiliary loss** on the embedding
- Keep loss weights from Step 1

### Step 3 — If Still Insufficient: Full Architecture Revision

- Cross-band speaker attention (operate on full `(B, T, 53*D)` instead of per-band)
- Add FFN after speaker attention layers (currently MHA+LayerNorm only, no non-linear transform)
- Larger speaker encoder (Conv1D stack)
- Consider GradNorm/DWA for automatic loss balancing

---

## Deferred Considerations

- `lambda_sep` may need dynamic adjustment as SI-SDR improves
- Cross-band attention in speaker pathway could further improve performance
- Speaker auxiliary loss (e.g., speaker verification loss) could provide additional gradient signal
- Gradient norm monitoring to verify starvation hypothesis and clipping effects
- Verify validation set is properly held out from training data
- Shared RIR per scene (same room for all speakers in a mixture) could simplify the learning problem
- Complex mask constraints (bounded magnitude, limited phase rotation) could stabilize training
