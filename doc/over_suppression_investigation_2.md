# Over-Suppression Investigation #2: Amplitude Loss Insufficient

**Date:** 2026-02-11
**Training range:** Steps 100K–106.8K (Phase 2, resumed from 100K checkpoint)
**Fixes applied:** amplitude_loss (lambda_amp=15.0 with 5K warmup), energy_loss -40dB floor, lambda_stft=5.0

---

## Observation

Despite the amplitude loss fix, rms_ratio dropped from 1.06 to 0.61 in just 6K steps, then stabilized around 0.61. The previous run (without amplitude loss) dropped to 0.39 by step 274K, so the fix slowed the decline but did not prevent it.

## Training Data Analysis

### Windowed Averages (TP batches only, 67 of 69 total)

| Window | sep | stft | amp | eff_sep | eff_stft | eff_amp | rms_μ | rms_σ |
|--------|-----|------|-----|---------|----------|---------|-------|-------|
| 100K | 20.9 | 1.413 | 0.17 | 20.9 (75%) | 7.1 (25%) | 0.5 (2%) | **1.06** | 0.30 |
| 102K | 21.6 | 1.251 | 0.17 | 21.6 (73%) | 6.3 (21%) | 1.6 (5%) | **0.65** | 0.13 |
| 104K | 23.8 | 1.336 | 0.23 | 23.8 (70%) | 6.7 (20%) | 3.4 (10%) | **0.61** | 0.20 |
| 106K | 27.4 | 1.349 | 0.14 | 27.4 (76%) | 6.7 (19%) | 2.1 (6%) | **0.61** | 0.14 |

*Loss weights: lambda_sep=1.0, lambda_stft=5.0, lambda_amp=15.0*
*eff_X = lambda_X * raw_X (x warmup scale for amp)*

### RMS Ratio Distribution

| Range | Count | Percentage |
|-------|-------|------------|
| < 0.5 (suppressed) | 10/67 | 15% |
| 0.5-0.8 (moderate) | 34/67 | 51% |
| 0.8-1.2 (healthy) | 16/67 | 24% |
| > 1.2 (amplified) | 7/67 | 10% |

### Loss Contribution at Full Warmup (105K+)

| Component | Effective | Share |
|-----------|-----------|-------|
| eff_sep | 25.6 | **73%** |
| eff_stft | 6.7 | 19% |
| eff_amp | 2.7 | 8% |

**sep dominates amp by 10x.**

## Root Cause Analysis

### 1. SI-SDR loss values are disproportionately large

SI-SDR loss values are in the 15-35 range, while STFT is ~1.3 and amplitude is ~0.15. Even with lambda_amp=15.0, the effective amplitude contribution (2.1-3.4) is dwarfed by the separation loss (20-28). The amplitude gradient simply cannot compete.

### 2. Quadratic amplitude penalty is too gentle

The current `amplitude_loss = mean((rms_est/rms_tgt - 1)^2)` is quadratic. At rms_ratio=0.6:
- Loss value: (0.6 - 1.0)^2 = 0.16
- Gradient magnitude: 2 * (0.6 - 1.0) * 15.0 = -12.0

Compare to sep loss gradient which is ~25 in the opposite direction.

### 3. sep loss is increasing, not decreasing

The separation loss went from 20.9 (100K) to 27.4 (106K) -- the model is getting *worse* at SI-SDR despite training. This suggests the introduction of TA data and the energy loss are confusing the model's optimization landscape, pulling it in conflicting directions.

### 4. Amplitude loss stabilized but didn't recover

The good news: rms_ratio stabilized at ~0.61 instead of continuing to drop (the previous run hit 0.39). The amplitude loss is working as a brake, but not as a correction force. The equilibrium point of 0.61 reflects where the sep gradient pushing down balances the amp gradient pushing up.

### 5. TP/TA conflict is the actual suppression mechanism (from review)

Phase 2 introduces a fundamental gradient conflict on shared backbone layers: the model must simultaneously extract the target speaker (for TP) and suppress everything (for TA). The optimizer resolves this conflict via uniform amplitude reduction -- a compromise that reduces TA energy loss while costing nothing in SI-SDR (scale-invariant). This is the path of least resistance.

### 6. Phase 1 optimizer momentum carry-over (from review)

At step 100K, Adam's first-moment and second-moment estimates are shaped entirely by SI-SDR-only gradients from Phase 1. When Phase 2 introduces new loss terms, the existing momentum carries directional bias that takes time to overcome. The 5K warmup helps but only affects the amplitude_loss contribution.

### 7. STFT loss budget consumed by shape errors (from review)

With SI-SDR at -28 dB, spectral shape errors dominate the STFT loss, leaving little gradient budget for scale correction. The STFT loss is doing double duty (shape + scale) but shape errors consume most of its capacity.

### 8. SI-SDR at -28 dB is alarmingly poor (from review)

A working TFGridNet should reach positive SI-SDR within 20-50K steps. At -28 dB after 100K steps, the estimate is essentially uncorrelated with the target. This may indicate deeper issues beyond amplitude: reverb mismatch in SI-SDR target, weak speaker embeddings, or mask still near identity. **This needs separate investigation.**

## Candidate Solutions

### Option 1: Increase lambda_amp dramatically (e.g., 50-100)
- **Pros:** Simple, direct
- **Cons:** Adam normalizes gradients by variance -- large lambda inflates `v` estimate, partially undoing the effect. Creates oscillatory dynamics near ratio=1.0. Does not address fundamental sep dominance.
- **Expected impact:** Shift equilibrium to ~0.75-0.85 with oscillation. Not recommended standalone.

### Option 2: Switch to L1 amplitude loss |ratio - 1|
- **Pros:** Constant gradient magnitude regardless of deviation
- **Cons:** Only 25% improvement over L2 (15.0 vs 12.0). Constant gradient at ratio=1.0 means perpetual micro-adjustments wasting gradient budget.
- **Expected impact:** Marginal. Shift equilibrium from 0.61 to ~0.65.

### Option 3: Log-space amplitude loss |log(ratio)|
- **Pros:** At rms_ratio=0.6, loss = 0.51 (3.2x larger than L2's 0.16). Gradient 1/r provides stronger correction when more suppressed.
- **Cons:** Non-smooth at r=1.0. Gradient never vanishes (1/r = 1.0 at r=1.0), interferes with optimization even when amplitude is correct.
- **Expected impact:** Good. Would likely shift equilibrium to ~0.8-0.9.

### Option 4: Reduce lambda_sep (e.g., 0.3-0.5)
- **Pros:** Most architecturally correct -- sep at 20-30 is the outlier. Rebalances entire loss landscape.
- **Cons:** Changes primary training objective. Risk of slowing separation learning (though sep is already failing at -28 dB).
- **Expected impact:** High for amplitude. Risk for separation if bottleneck is gradient magnitude (but likely isn't given 100K steps at full lambda produced only -28 dB).

### Option 5: Combine lambda_sep reduction + log-space amp loss
- **Pros:** Addresses both dominance and penalty shape. Most robust.
- **Cons:** Two simultaneous changes make attribution harder.

## Implemented Fix (from review recommendation)

**Modified Option 5 with log-space L2 (not L1):**

### Changes Applied

1. **`lambda_sep`: 1.0 -> 0.5** -- reduce sep dominance from 73% to ~52%
2. **`lambda_amp`: 15.0 -> 20.0** -- slightly stronger amplitude signal
3. **`amplitude_loss`: `(ratio-1)^2` -> `log(ratio)^2`** -- log-space L2

Why `log(r)^2` over `|log(r)|`:
- **Gradient is exactly zero at r=1.0** -- when amplitude is correct, loss contributes no gradient, letting sep/stft optimize freely
- Smooth everywhere (no non-differentiable point)
- At r=0.6: gradient = 2*log(0.6)/0.6 = -1.70, weighted: 20*1.70 = 34. Competitive with eff_sep at 12.5
- `clamp(min=0.05)` on ratio prevents log explosion for near-zero values

### Expected Loss Balance (at rms_ratio=0.6)

| Component | Weight | Raw Value | Effective | Share |
|-----------|--------|-----------|-----------|-------|
| sep | 0.5 | ~25 | 12.5 | 52% |
| stft | 5.0 | ~1.3 | 6.5 | 27% |
| amp (log-L2) | 20.0 | ~0.26 | 5.2 | **22%** |
| l1 | 0.5 | ~0.05 | 0.025 | ~0% |

### At rms_ratio=1.0 (target achieved)

| Component | Weight | Raw Value | Effective | Share |
|-----------|--------|-----------|-----------|-------|
| sep | 0.5 | ~20 | 10.0 | 61% |
| stft | 5.0 | ~1.3 | 6.5 | 39% |
| amp | 20.0 | 0.0 | 0.0 | 0% |

Ideal behavior: when amplitude is correct, sep and stft have full control.

## Monitoring Checklist

Track every 1K steps after resuming:
1. **rms_ratio** (training TP): should stabilize at 0.8-1.2 within 5-10K steps
2. **loss_amp** (raw): should decrease from ~0.26 toward ~0.05
3. **loss_sep** (raw SI-SDR): should decrease. If rises above 30, model is destabilizing
4. **Validation SI-SDR**: should trend toward positive values. **If still below -20 dB at 120K, there is a deeper problem**
5. **ta_energy**: should reach -35 to -40 dB and stabilize (held by floor)
6. **Gradient norm**: watch for spikes in first 1K steps

## Safety Fallback

If SI-SDR has not improved by step 110K (10K into Phase 2), revert lambda_sep to 1.0 and rely solely on log-space amplitude loss with lambda_amp=20.

## Deferred Investigation

SI-SDR at -28 dB after 100K steps is alarmingly poor and needs separate investigation:
- Reverb mismatch between training target and SI-SDR reference
- Speaker embedding quality from USEF cross-attention
- Complex mask learning dynamics (still near identity?)

## Validation Metrics at Key Points

| Step | si_sdr | rms_ratio | ta_energy |
|------|--------|-----------|-----------|
| 100K | -27.07 dB | 1.131 | -25.1 dB |
| 105K | -28.00 dB | 0.653 | -29.8 dB |
