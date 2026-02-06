# Future Actions & Deferred Issues

Deferred items from the code review in `doc/HiFi_TSE_Analysis_2.md`.
These are not bugs but improvements to consider after the current training run stabilizes.

---

## Training & Loss

### Separate D-step / G-step for memory efficiency (Analysis #9)
Currently D backward and G backward coexist in the same micro-step, nearly
doubling peak GPU memory.  Refactoring the training loop to: backward D → step D
→ forward G adversarial → backward G → step G would reduce peak VRAM.
**When:** If GPU OOM becomes a problem or we want to increase batch size.

### Per-micro-step gradient clipping for discriminator (Analysis #12)
D gradients accumulate over 32 micro-steps before clipping.  Large early
gradients accumulate without bound until the clip step.  Adding per-step clipping
(or gradient scaling) would make D training more stable.
**When:** If D loss diverges or training becomes unstable in phase 3.

---

## Data Pipeline

### Validation on held-out data during training (Analysis #5)
In-training validation draws from the same HDF5 as training.  `evaluate.py`
already uses separate val speakers, so this only affects mid-training
monitoring.  A proper held-out val split in the training loop would make
checkpoint selection more reliable.
**When:** When implementing early stopping or automated checkpoint selection.

### Data augmentation beyond RIR/noise (Analysis #23)
Missing: speed perturbation, pitch shift, SpecAugment-style masking, codec
simulation.  These are important for robustness to real-world 48kHz audio.
**When:** After baseline training converges and we want to improve generalization.

### Read sample_rate from config instead of hardcoding (Analysis #8)
`data/dataset.py` and `evaluate.py` hardcode `SAMPLE_RATE = 48000`.  Low risk
since everything is 48kHz-specific, but would improve config flexibility.
**When:** If we ever experiment with 16kHz or 24kHz variants.

---

## Inference

### Carry BiLSTM hidden states across inference chunks (Analysis #13)
Chunked inference in `inference.py` resets LSTM hidden states per chunk.
Overlap-add windowing partially mitigates edge artifacts, but carrying hidden
states would improve temporal continuity.
**When:** When optimizing inference quality for production deployment.

---

## Architecture

### Multi-layer USEF cross-attention (Analysis #22)
Currently single-layer `nn.MultiheadAttention`.  State-of-the-art TSE models
use multi-layer cross-attention with more sophisticated fusion.  We partially
addressed this with `SpeakerReinjectLayer` at blocks [0,2,4], but a deeper
USEF module could improve speaker conditioning robustness.
**When:** Next architecture iteration.

### Band overlap weighting at region boundaries (Analysis #11)
Overlapping bands are merged by simple averaging.  The number of overlaps
changes at region boundaries (~1kHz, ~8kHz), creating step changes in
effective weighting.  A learned or smooth weighting scheme could help.
**When:** If spectral artifacts are audible at region boundaries.

---

## Evaluation

### Add pesq/pystoi to requirements.txt as optional deps (Analysis #7)
`evaluate.py` gracefully handles missing deps, but explicit listing helps
reproducibility.  Consider `requirements-eval.txt` or extras in setup.py.
**When:** When packaging for distribution.

### Increase validation batch cap (Analysis #19)
Currently 50 batches (100 samples).  Fine for training monitoring but may miss
systematic failure modes.  Increase when compute budget allows.

---

## Infrastructure

### Worker seed management improvements (Analysis #17)
Basic `worker_init_fn` added.  Could also seed `torch` random state per worker
and use `generator` argument in DataLoader for full reproducibility.
**When:** If reproducibility becomes important for ablation studies.

### Effective batch size considerations (Analysis #24-25)
Batch size 2 with grad_accum 32 = effective 64.  Only ~15.6K optimizer updates
over 500K micro-steps.  If convergence is slow, consider increasing batch size
(requires more VRAM) or reducing grad_accum.
**When:** After evaluating convergence speed of current training run.
