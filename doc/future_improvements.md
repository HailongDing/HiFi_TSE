# Future Improvements — HiFi-TSE v2

## Current Status — TRAINING COMPLETE (Step 1,500,000)

- **SI-SDRi**: +4.18 dB (200-sample eval, seed=42)
- **SI-SDR**: -1.71 dB (std 5.33)
- **PESQ**: 1.14 (std 0.13)
- **STOI**: 0.489 (std 0.162)
- **TA suppression**: -31.2 dB output energy (11.4 dB suppression)
- **Model**: 11.79M params, 48kHz, TF-GridNet + USEF conditioning
- **Training**: SI-SDR + phase loss + amplitude loss, 2-phase curriculum
- **v1 baseline**: SI-SDRi +4.01 dB — v2 improves by +0.17 dB

See `doc/performance_improvement_plan.md` for detailed next-step improvement methods.

---

## Data Loading Optimization

### Stage 1 — Infrastructure [COMPLETED]

- num_workers 4 -> 8, prefetch_factor 2 -> 4, OMP_NUM_THREADS=1
- persistent_workers=True initially, reverted to False due to h5py crash
- GPU utilization improved from periodic 0% drops to sustained 83-99%

### Stage 2 — Algorithm-level (Deferred, NOT YET DONE)

Infrastructure-only changes (Stage 1) eliminated GPU stalls. These algorithm-level
changes were deferred for safety but can provide additional throughput:

### Replace scipy.signal.resample with torchaudio.functional.resample
- `data/dataset.py:_speed_perturb_np` — speed perturbation (0.9-1.1x)
- scipy uses FFT-based O(N log N) on entire 288K-sample arrays
- torchaudio uses polyphase sinc filtering — much faster for near-unity ratios
- Risk: slightly different filter characteristics (acceptable for augmentation)

### Replace scipy.signal.fftconvolve with scipy.signal.oaconvolve
- `data/dataset.py:_apply_rir_np` — RIR convolution
- oaconvolve (overlap-add) faster when kernel (RIR ≤96K) < signal (~288K)
- Drop-in replacement, mathematically identical (different floating-point rounding)

### Increase batch_size 2→4, reduce grad_accum 32→16
- Same effective batch (64), half the DataLoader round-trips
- GPU memory: 6.5 / 24.5 GB — ample headroom
- Risk: changes micro-batch dynamics (TA/TP split, amplitude_loss averaging)

---

## Model Architecture Improvements (NOT YET DONE)

### Larger model capacity
- Current: 11.79M params (lstm_hidden=256, 6 blocks)
- Option A: lstm_hidden 256→384 (~18M params, may need larger GPU)
- Option B: num_gridnet_blocks 6→8 (~16M params)
- Expected: +0.5 to +1.5 dB SI-SDRi per scaling step
- Reference: X-TF-GridNet achieves 20.7 dB SI-SDRi with larger model on WSJ0-2mix

### Stronger speaker conditioning
- Current: USEF cross-attention + FiLM, reinject at all 6 blocks
- Option: Add speaker-conditioned LayerNorm (AdaLN) — modulate norm parameters
  with speaker embedding, providing finer-grained conditioning per layer
- Option: Multi-scale reference encoding — extract features at multiple time scales
  from enrollment, not just a single cross-attention pass
- Reference: USEF-TFGridNet shows 5-6% relative SI-SDRi improvement from
  better enrollment utilization

### Dual-path attention improvements
- Replace BiLSTM with Mamba/S4 for sub-band modeling — O(N) vs O(N²) attention
- Could enable longer segments (4s → 8s) without memory explosion
- Reference: SpeakerBeam-SS achieves 78% RTF reduction with state space models

---

## Training Strategy Improvements (NOT YET DONE)

### Curriculum refinement
- Current: Phase 1 (TP-only, 0-200K) → Phase 2 (TP+TA, 200K-1.5M)
- Option: Phase 3 with harder conditions (lower SNR, more interferers) after 1M steps
- Option: Progressive SNR hardening — start with easy (5-15 dB) then ramp to (-5, 15 dB)
- Option: Increase num_interferers_range from [1,3] to [1,4] in final phase

### Loss function refinements
- Multi-resolution SI-SDR: compute SI-SDR at multiple window sizes to capture
  both fine and coarse temporal structure
- Learnable loss weighting: use uncertainty-based weighting (Kendall et al.) to
  automatically balance lambda_sep, lambda_phase, lambda_amp
- Contrastive speaker loss: add auxiliary contrastive loss on speaker embeddings
  to improve target/interferer discrimination

### Data augmentation
- SpecAugment on input mixture — mask frequency bands and time steps
- Random EQ on reference audio — improve robustness to enrollment quality
- Multi-enrollment: provide 2-3 reference utterances during training
  (average embeddings) — improves robustness at inference

---

## Post-Training / Fine-Tuning (NOT YET DONE)

### GAN fine-tuning for perceptual quality
- v1 showed GAN Phase 3 added +0.5 dB SI-SDR + perceptual improvement
- After convergence at 1.5M, add discriminator for 100-200K steps
- Use HiFi-GAN discriminator (multi-period + multi-scale)
- Lower LR (1e-5) to avoid destabilizing learned separation

### Knowledge distillation
- Train a smaller student model (4-6M params) from the v2 teacher
- Target: real-time inference on edge devices
- Use soft SI-SDR targets + intermediate feature matching

### Test-time augmentation
- Multi-enrollment averaging: use 3-5 reference utterances at inference
- Expected: +0.3 to +0.5 dB SI-SDRi from better speaker representation

---

## Benchmark Evaluation (NOT YET DONE)

### Standard benchmarks to evaluate on
- **WSJ0-2mix**: 8kHz, 2-speaker clean separation (standard, easy)
- **WHAM!**: 8kHz, 2-speaker with noise
- **WHAMR!**: 8kHz, 2-speaker with noise + reverb
- **LibriMix**: 16kHz, 2-3 speakers with noise
- Need to implement evaluation pipeline with proper input SI-SDR measurement
  for SI-SDRi computation

### Competitive targets (similar model size, ~10M params)
| Model | Dataset | SI-SDRi |
|-------|---------|---------|
| SpEx+ | WSJ0-2mix | ~17 dB |
| X-TF-GridNet | WSJ0-2mix | 19.7 dB |
| USEF-TFGridNet | WHAMR! | SOTA |

Note: These are 8/16kHz benchmarks. 48kHz performance cannot be directly compared.

---

## Infrastructure Improvements

### NVIDIA driver stability [PARTIALLY COMPLETED]
- Applied `uvm_disable_hmm=1` workaround — no reboots since
- Still on 580.126.09; monitor for stable driver with proper UVM fix
- Consider open-source NVIDIA kernel modules for better stability

### Auto-restart on crash (NOT YET DONE)
- Add systemd service or cron watchdog to restart training after GPU driver crashes
- Monitor PID of training process, restart from latest checkpoint if missing

### Distributed training (NOT YET DONE)
- Current: single RTX 4090 D (24.5 GB)
- If available, multi-GPU with DDP would enable larger batch size and faster iteration
- Gradient accumulation could be reduced/eliminated with 2-4 GPUs
