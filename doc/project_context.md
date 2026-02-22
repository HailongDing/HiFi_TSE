# HiFi-TSE v2 — Project Context

This document provides all necessary background for resuming work on the HiFi-TSE v2 project in a new Claude Code session.

## CURRENT STATUS: v2 Training COMPLETE (Feb 22, 2026)

All 11 implementation changes done. Training finished at step 1,500,000.

| Metric | v1 | v2 |
|--------|----|----|
| SI-SDRi | +4.01 dB | **+4.18 dB** |
| SI-SDR | -1.87 dB | **-1.71 dB** |
| PESQ | 1.15 | 1.14 |
| STOI | 0.488 | 0.489 |
| TA suppression | 11.2 dB | 11.4 dB |
| Model params | 7.8M | 11.79M |

See `doc/performance_improvement_plan.md` for next-step improvements.

---

## 1. Project Background

### Goal
Build an industrial-grade **Target Speaker Extraction (TSE)** system operating at **48kHz full-band** audio. Given a mixture of multiple speakers + noise + reverberation, and a reference utterance of the target speaker, the system extracts only the target speaker's voice at 48kHz.

### Architecture: HiFi-USEF-Grid-GAN
The system fuses four core techniques:

| Component | Reference | Role |
|-----------|-----------|------|
| **BSRNN** (Band-Split RNN) | [chenxinliang/BSRNN](https://github.com/chenxinliang/BSRNN) | Band-Split Encoder/Decoder — splits 1025 freq bins into 53 overlapping sub-bands |
| **USEF** | [X-LANCE/USEF-Speech-Extraction](https://github.com/X-LANCE/USEF-Speech-Extraction) | Cross-attention conditioning (Q=mix, K/V=ref) for noise-robust speaker identification |
| **TF-GridNet** | [espnet/espnet](https://github.com/espnet/espnet) | Backbone: 6 GridNet blocks with intra-frame BiLSTM (freq), sub-band BiLSTM (time), self-attention |
| **HiFi-GAN** | [jik876/hifi-gan](https://github.com/jik876/hifi-gan) | Discriminator: MPD (periods 2,3,5,7,11) + MSD (3 scales) for perceptual quality |

### Signal Flow
```
Mix WAV -> STFT -> BandSplitEncoder -> Z_mix --+
                                                +-> USEF CrossAttn -> [FiLM] -> TFGridNet x6 -> BandMergeDecoder -> iSTFT -> Est WAV
Ref WAV -> STFT -> BandSplitEncoder -> Z_ref --+
                   (shared weights)
```
- STFT: n_fft=2048, win_length=960 (20ms), hop_length=480 (10ms) at 48kHz
- 53 overlapping sub-bands: fine (0-1kHz), medium (1-8kHz), coarse (8-24kHz)
- BandMergeDecoder applies complex ratio mask with proper complex multiplication
- SpeakerReinjectLayers (cross-attention) at configurable GridNet block indices

---

## 2. v1 Results and Problems

### v1 Final Results (best checkpoint at step 430K)
| Metric | Value |
|--------|-------|
| SI-SDR (mean) | -1.87 dB |
| **SI-SDRi (mean)** | **+4.01 dB** |
| PESQ (mean) | 1.15 |
| STOI (mean) | 0.488 |
| TA suppression | 11.2 dB |
| Training steps | 480K (early-stopped from 500K target) |
| Evaluation samples | 200 (158 TP, 42 TA) |

### Key Problems Identified in v1

1. **Small model capacity**: Generator only 7.8M params (lstm_hidden=192). Undersized for 48kHz (1025 freq bins). SOTA 16kHz models use 20-60M params.

2. **Loss competition**: 6 competing loss terms (SI-SDR, multi-res STFT, phase, L1, amplitude, TA energy) diluted SI-SDR gradient to ~48% of total. Multi-res STFT and amplitude losses optimize spectral fidelity, which can conflict with SI-SDR.

3. **Insufficient training**: 500K micro-steps with grad_accum=32 = only ~15,625 optimizer steps = ~10 epochs over 2,057 speakers. Model was still improving when early stopping triggered.

4. **Weak speaker conditioning**: USEF is a single cross-attention layer (66K params, 0.9% of generator). Speaker identity fades through 6 GridNet blocks. Re-injection only at blocks [0, 2, 4].

5. **GAN overhead**: 70.7M discriminator params consumed GPU memory and added training complexity (instability risk, D-only pre-training, warmup ramp) for only marginal SI-SDR benefit (+0.5 dB).

### v1 Issues Resolved During Training
- **Reverb mismatch**: Fixed train-against-reverbed-target (convolve-then-crop)
- **SI-SDR stagnation**: Rebalanced loss weights (lambda_sep 0.5->1.5, lambda_amp 20->10)
- **Over-suppression**: Added phase-sensitive loss, rms_ratio monitoring
- **Complex ratio mask**: Fixed proper complex multiplication in band_split.py
- **GAN warm-up**: Implemented 3-sub-phase strategy (D-only -> ramp -> full)
- **Data pipeline**: Worker seeds, scipy module-level import, checkpoint rotation

### v1 Source Code
The v1 project is at `/home/hailong/code/from_draft/v1/`. v1 checkpoints and training logs are there; they are NOT copied to v2.

---

## 3. v2 Improvement Plan Summary

See `doc/improvement_plan.md` for the full plan. Key changes:

1. **Scale TFGridNet**: lstm_hidden 192->256 (7.8M -> ~11.5M params)
2. **Simplify loss**: SI-SDR + phase only (remove STFT/L1/amp/GAN)
3. **Strengthen speaker conditioning**: FiLM layer + reinject at ALL 6 blocks
4. **Extend training**: 500K -> 1.5M micro-steps
5. **Remove GAN**: Defer to optional fine-tuning phase
6. **Speed perturbation**: 0.9x-1.1x augmentation on clean speech
7. **Gradient checkpointing**: Memory savings for larger model

Training from scratch required (architecture changes).

---

## 4. Environment Setup

### Hardware
| Component | Spec |
|-----------|------|
| GPU | NVIDIA GeForce RTX 4090 D (24GB VRAM) |
| OS | Ubuntu 22.04, Linux 5.15.0-164-generic |
| Machine | Single GPU workstation |

### Conda Environments

#### `USEF-TFGridNet` — Training & Inference (PRIMARY)
| Property | Value |
|----------|-------|
| Python | 3.8.20 |
| PyTorch | 2.4.1+cu118 |
| torchaudio | 2.4.1+cu118 |
| CUDA | 11.8 |
| Location | `/home/hailong/miniforge3/envs/USEF-TFGridNet` |

Key packages: scipy 1.10.1, soundfile 0.13.1, h5py 3.11.0, numpy 1.24.4, tensorboard 2.14.0, pesq 0.0.4, pystoi 0.4.1, mir_eval 0.8.2

```bash
conda activate USEF-TFGridNet
# or
conda run -n USEF-TFGridNet --no-capture-output python train.py --config configs/hifi_tse.yaml
```

#### `tse_dataset` — Dataset Preparation (one-time)
| Property | Value |
|----------|-------|
| Python | 3.10.19 |
| PyTorch | 2.9.1+cpu |
| Location | `/home/hailong/miniforge3/envs/tse_dataset` |

Key packages: scipy 1.15.3, soundfile 0.13.1, h5py 3.15.1, numpy 2.2.6

Used for: `data/prepare_manifest.py`, HDF5 packing scripts. NOT used for training.

### tmux Sessions Convention
| Session | Purpose |
|---------|---------|
| `train` | Long-running training process |
| `monitor` | WeCom webhook notification script |

### Key Commands
```bash
# Training (in tmux 'train' session)
PYTHONUNBUFFERED=1 conda run -n USEF-TFGridNet --no-capture-output \
    python train.py --config configs/hifi_tse.yaml 2>&1 | tee checkpoints/train.log

# Monitor (in tmux 'monitor' session, start BEFORE training)
bash monitor_training.sh checkpoints/train.log

# Evaluation
conda run -n USEF-TFGridNet python evaluate.py \
    --config configs/hifi_tse.yaml \
    --checkpoint checkpoints/checkpoint_best.pt \
    --num-samples 200 --output-dir checkpoints/eval_best

# Inference
conda run -n USEF-TFGridNet python inference.py \
    --config configs/hifi_tse.yaml \
    --mix mix.wav --ref ref.wav -o output.wav
```

### Important Notes
- `conda run` buffers stdout — always use `PYTHONUNBUFFERED=1` and `--no-capture-output` when piping to tee/log
- Always launch long training in **tmux**, not background shell tasks
- Start monitor script **before** or simultaneously with training (it uses `tail -n 0 -F`, new lines only)
- WeCom webhook URL: `https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=08365047-6447-4f2c-81d1-6975fd0e1fa0`

---

## 5. Dataset Information

All audio is 48kHz, mono, float32. Raw WAV files are pre-packed into HDF5 for fast I/O. Total dataset: ~769 GB at `/data/tse_hdf5/`.

### Clean Speech (2,057 train + 107 val speakers)

| Source | Path | Train Spkrs | Val Spkrs | Utterances | Hours | HDF5 |
|--------|------|-------------|-----------|------------|-------|------|
| DNS4 | `/data/48khz_dataset/DNS4_raw/.../read_speech` | 1,759 | 92 | 185,826 | 533 h | `dns4.h5` (344 GB) |
| EARS | `/data/48khz_dataset/ears_dataset` | 97 | 5 | 16,406 | 95 h | `ears.h5` (62 GB) |
| VCTK | `/data/48khz_dataset/dns_clean_dataset/vctk_wav48_silence_trimmed` | 99 | 5 | 12,812 | 14 h | `vctk.h5` (9.3 GB) |

Speaker ID format: `dns4_reader_XXXXX`, `ears_pXXX`, `vctk_pXXX`
Val split: last 5% of sorted speaker IDs per source (deterministic).

### Noise (107,940 clips total)

| Source | Path | Clips | Disk | Characteristic |
|--------|------|-------|------|----------------|
| DEMAND | `/data/48khz_dataset/dns_noise_dataset/demand` | 272 | 15.7 GB | Stationary environmental |
| WHAM | `/data/48khz_dataset/wham_dataset/high_res_wham/audio` | 10,083 | 54.3 GB | Urban ambient |
| AudioSet | `/data/48khz_dataset/audioset_dataset/audio` | 33,775 | 120.4 GB | Transient events (music excluded) |
| DNS5 | `/data/48khz_dataset/dns_noise_dataset/dns5_noise` | 63,810 | 122.4 GB | General (4 shards) |

### RIR (60,248 total)

| Source | Path | Count | Type |
|--------|------|-------|------|
| SLR26 | `.../SLR26/simulated_rirs_48k` | 60,000 | Simulated (small/medium/large rooms) |
| SLR28 | `.../SLR28/RIRS_NOISES/real_rirs_isotropic_noises` | 248 | Real measured |

### HDF5 Layout
```
/data/tse_hdf5/
├── train/
│   ├── clean_speech/
│   │   ├── dns4.h5             344 GB
│   │   ├── ears.h5              62 GB
│   │   └── vctk.h5             9.3 GB
│   ├── noise/
│   │   ├── demand.h5            15 GB
│   │   ├── wham.h5              51 GB
│   │   ├── audioset.h5         113 GB
│   │   ├── dns5_noise_0.h5      29 GB
│   │   ├── dns5_noise_1.h5      29 GB
│   │   ├── dns5_noise_2.h5      29 GB
│   │   └── dns5_noise_3.h5      29 GB
│   └── rir/
│       └── rir.h5               11 GB
└── manifests/
    ├── speaker_manifest.json
    ├── noise_manifest.json
    └── rir_manifest.json
```

### Dynamic Mixing Pipeline (data/dataset.py)
Training samples are constructed on-the-fly, NOT pre-mixed:
1. Sample target speaker + utterance, random-crop to 4.0s (192,000 samples)
2. TP (80%) or TA (20%) decision (Phase 2+)
3. Sample 1-3 interferer speakers (different from target)
4. Convolve target + interferers with random RIRs (convolve-then-crop for proper reverb tails)
5. Mix with random noise at SNR [-5, 15] dB
6. Reference: different utterance from same speaker, 4.0s. Noisy ref with probability ramped 0->0.5 over steps 100K-150K
7. Peak normalization if mixture exceeds 0.95

---

## 6. Source Code Structure

```
v2/
├── configs/
│   └── hifi_tse.yaml          # All hyperparameters, dataset paths, training config
├── data/
│   ├── audio_utils.py          # STFT/iSTFT helpers
│   ├── dataset.py              # HiFiTSEDataset, dynamic mixing, collate_fn
│   ├── prepare_manifest.py     # Scan -> manifest -> HDF5 packing
│   ├── pack_noise_only.py      # Repack noise
│   ├── pack_noise_rir.py       # Repack noise + RIR
│   ├── download_dns5_noise.sh  # DNS5 noise download
│   └── pack_all_hdf5.sh        # Pack all sources
├── models/
│   ├── band_split.py           # BandSplitEncoder + BandMergeDecoder (53 bands, complex ratio mask)
│   ├── generator.py            # Full generator: STFT -> Encoder -> USEF -> TFGridNet -> Decoder -> iSTFT
│   ├── tf_gridnet.py           # TFGridNet backbone (6 blocks) + SpeakerReinjectLayer
│   ├── usef.py                 # USEF cross-attention module
│   └── discriminator.py        # HiFi-GAN MPD+MSD (kept for future GAN fine-tuning)
├── losses/
│   ├── separation.py           # SI-SDR, scene_aware_loss, energy_loss, amplitude_loss, l1_waveform_loss
│   ├── stft_loss.py            # MultiResolutionSTFTLoss + PhaseSensitiveLoss
│   └── adversarial.py          # generator_adv_loss, discriminator_loss, feature_matching_loss
├── train.py                    # 3-phase curriculum training loop (to be simplified to 2-phase)
├── evaluate.py                 # SI-SDR/SI-SDRi/PESQ/STOI evaluation
├── inference.py                # Single-file inference
├── monitor_training.sh         # WeCom webhook notifications
├── requirements.txt            # Python dependencies
├── .gitignore
└── doc/
    ├── improvement_plan.md          # Original v2 improvement plan (7 changes) [COMPLETED]
    ├── implementation_plan.md       # Full implementation plan (11 changes) [COMPLETED]
    ├── another_review.md            # Codex review of implementation plan [ADDRESSED]
    ├── over_suppression_fix.md      # Training fixes (changes 1-8) [COMPLETED]
    ├── future_improvements.md       # Early improvement ideas (partially superseded)
    ├── performance_improvement_plan.md  # Next-step improvements for v3/fine-tuning
    └── project_context.md           # This file
```

### Key Model Parameters (v1 -> v2)
| Parameter | v1 | v2 (IMPLEMENTED) |
|-----------|-----|------|
| feature_dim | 128 | 128 |
| lstm_hidden | 192 | **256** |
| num_gridnet_blocks | 6 | 6 |
| num_heads | 4 | 4 |
| reinject_at | [0, 2, 4] | **[0, 1, 2, 3, 4, 5]** |
| FiLM conditioning | N/A | **Added** |
| Gradient checkpointing | No | **Yes** |
| Generator params | 7.8M | **11.79M** |
| Discriminator | 70.7M (active) | **Removed from training** |

---

## 7. Implementation Checklist for v2 — ALL COMPLETED

- [x] `configs/hifi_tse.yaml` — Updated all config values (lstm_hidden, loss weights, phases, etc.)
- [x] `models/usef.py` — Added FiLMLayer class
- [x] `models/generator.py` — Instantiated and called FiLMLayer after USEF
- [x] `models/tf_gridnet.py` — Added gradient checkpointing support
- [x] `train.py` — Simplified loss (remove STFT/L1/amp), removed GAN, updated logging
- [x] `data/dataset.py` — Added speed perturbation augmentation
- [x] `monitor_training.sh` — Simplified (no GAN warnings needed)
- [x] Smoke test (100 steps) — PASSED
- [x] Memory check (batch_size=2 + lstm_hidden=256 + grad_checkpoint) — PASSED (6.5/24.5 GB)
- [x] Full training completed (1,500,000 steps)
- [x] Evaluation completed (SI-SDRi +4.18 dB)

### Additional fixes applied during training:
- [x] energy_loss relu fix (over-suppression)
- [x] amplitude_loss for TP samples (over-suppression)
- [x] Validation stabilization (200 batches, fixed seed)
- [x] GPU utilization fix (8 workers, OMP_NUM_THREADS=1)
- [x] NVIDIA driver fix (uvm_disable_hmm=1)
- [x] h5py crash mitigation (persistent_workers=False)

---

## 8. Resuming in a New Claude Code Session

When starting a new Claude Code session in `/home/hailong/code/from_draft/v2/`:

1. Read `doc/project_context.md` (this file) for full background
2. v2 training is COMPLETE — all 11 changes implemented, 1.5M steps trained
3. Final result: SI-SDRi +4.18 dB (vs v1 baseline +4.01 dB)
4. Best checkpoint: `checkpoints/checkpoint_best.pt` (step 1,495,000)
5. Final checkpoint: `checkpoints/checkpoint_final.pt` (step 1,500,000)
6. See `doc/performance_improvement_plan.md` for next improvement methods
7. Use conda env `USEF-TFGridNet` for all training/inference
8. Use tmux sessions `train` and `monitor` for long-running processes
