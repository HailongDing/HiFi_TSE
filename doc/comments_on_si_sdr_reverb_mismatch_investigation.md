Now I have a thorough understanding of both the codebase and the document. Here's my review:

---

## Review: `doc/si_sdr_reverb_mismatch_investigation.md`

### Diagnosis: Correct and well-supported

The document correctly identifies the fundamental problem: the training target is **clean** (`target_wav`) while the mixture contains **reverbed** audio (`target_reverbed`). I verified this against the source code:

- `dataset.py:322-324` — target is loaded clean and peak-normalized
- `dataset.py:346-347` — `target_reverbed` is created via RIR convolution
- `dataset.py:358-359` — `target_reverbed` goes into the mixture
- `dataset.py:394-397` — but `target_wav` (clean) is returned as the label

This means SI-SDR is measured against an impossible target — a perfect model would output the reverbed version (which is what's in the mix), so the -27 dB ceiling is real. The analysis is sound.

### Potential Issues Found

**1. evaluate.py has an inconsistency that the doc partially acknowledges but underplays (line 367)**

`evaluate.py:367` computes the primary SI-SDR metric against `clean_target`, not `reverbed_target`. This means **evaluation is also broken** — the reported SI-SDR numbers during evaluation are also artificially depressed by the same reverb mismatch. The doc mentions this but frames it only as a "future update" rather than flagging it as an active measurement bug that makes all current eval numbers meaningless.

**2. The `evaluate.py` SI-SDRi calculation is internally contradictory (lines 367-370)**

```
si_sdr_val    = compute_si_sdr(est_wav, clean_target)      # line 367
si_sdr_input  = compute_si_sdr(mix_wav, reverbed_target)    # line 369
si_sdri       = si_sdr_val - si_sdr_input                   # line 370
```

SI-SDRi subtracts two SI-SDRs computed against **different references** (clean vs reverbed). This is mathematically invalid — SI-SDR improvement requires both measurements against the same reference to be meaningful. The doc notes "the existing SI-SDRi baseline already uses reverbed target — this is consistent" (line 149) which is **wrong** — it's actually inconsistent because numerator and denominator use different targets.

**3. The proposed fix's interaction with the TA branch is unmentioned**

In `dataset.py:370`, for TA samples, `target_wav` is set to zeros. The proposed "reassign immediately after line 346" approach (`target_wav = target_reverbed`) would be overwritten to zeros anyway for TA, so this is fine — but the document doesn't explicitly verify this code path. If someone later reorders the code, the reassignment could be placed after the TA zeroing, silently breaking TA behavior.

**4. Clipping normalization bug scope is understated**

The doc correctly identifies the clipping normalization issue (lines 387-391) where `target_reverbed` isn't scaled with the mixture. But it doesn't quantify how often `mix_max > 0.95` triggers. With multiple reverbed speakers + noise at SNRs as low as -5 dB, clipping could happen on a significant fraction of samples — this isn't a rare edge case, it could be affecting a large portion of training data.

**5. The -27 dB "ceiling" number is asserted without derivation**

The document states the ceiling is ~-27 dB but doesn't show how this was measured. Is it an average over the training set? A single sample? The reverb mismatch magnitude depends heavily on the RIR characteristics (RT60, direct-to-reverberant ratio). Different RIR distributions would give different ceilings. A proper sanity check (as suggested in the "Pre-Training Validation" section) should have been done **before** writing the report, not deferred.

**6. PESQ and STOI in evaluate.py also measure against clean target**

`evaluate.py:374,378` compute PESQ and STOI against `clean_target`. The doc only discusses SI-SDR but these perceptual metrics are also affected by the mismatch. PESQ especially penalizes reverb vs clean comparisons heavily.

**7. The doc claims Option A is "standard in TSE literature" — this is mostly but not entirely accurate**

Training against reverbed target is common, but several modern systems (e.g., BSRNN-based approaches) do train with joint extraction+dereverberation when the model is large enough. The blanket claim that Option A is the standard could discourage future exploration of joint training once the model scales up.

### Minor Issues

- **Line numbers are off by a few**: The doc references "Line 322" for `get_utterance` — in the actual code it's at line 322, so these check out. However, line 394 "Returns target_wav" is described as a return statement but it's actually the tensor conversion; the actual return is at line 399. This is a minor description inaccuracy.

- **The doc doesn't mention that the L1 waveform loss (`losses/separation.py:85-95`) is also affected**: L1 loss between reverbed estimate and clean target would also have a floor. All waveform-domain losses (SI-SDR, L1, amplitude) share this ceiling issue, not just SI-SDR.

### Summary

The core diagnosis is **correct and well-reasoned**. The recommended fix (Option A) is appropriate. The main concerns are:
1. The `evaluate.py` SI-SDRi calculation is mixing references — this is a real bug the doc mischaracterizes as "consistent"
2. PESQ/STOI evaluation has the same mismatch issue but is unmentioned
3. The -27 dB ceiling should have been empirically validated before the report
4. The clipping normalization frequency/impact is not quantified