# Conda Environment Configuration

## 1. `tse_dataset` — Dataset Preparation

**Purpose:** One-time data scanning, manifest generation, and dataset preparation scripts.
- `data/prepare_manifest.py`

| Property | Value |
|----------|-------|
| Python | 3.10.19 |
| PyTorch | 2.9.1+cpu |
| CUDA | N/A (CPU-only) |
| Location | `/home/hailong/miniforge3/envs/tse_dataset` |

**Key packages:** scipy 1.15.3, soundfile 0.13.1, librosa 0.11.0, h5py 3.15.1, PyYAML 6.0.3, numpy 2.2.6, tqdm 4.67.1, pandas 2.3.3

**Activation:**
```bash
conda activate tse_dataset
```

---

## 2. `USEF-TFGridNet` — Training & Inference

**Purpose:** Model training, validation, and inference. All GPU-dependent work.
- `train.py`
- `inference.py`

| Property | Value |
|----------|-------|
| Python | 3.8.20 |
| PyTorch | 2.4.1+cu118 |
| torchaudio | 2.4.1+cu118 |
| CUDA | 11.8 |
| GPU | NVIDIA GeForce RTX 4090 D |
| Location | `/home/hailong/miniforge3/envs/USEF-TFGridNet` |

**Key packages:** scipy 1.10.1, soundfile 0.13.1, librosa 0.11.0, h5py 3.11.0, PyYAML 6.0.3, numpy 1.24.4, tensorboard 2.14.0, tqdm 4.67.1, pesq 0.0.4, pystoi 0.4.1, mir_eval 0.8.2

**Activation:**
```bash
conda activate USEF-TFGridNet
```

---

## Usage Summary

| Task | Conda Environment | Command |
|------|-------------------|---------|
| Build speaker/noise/RIR manifests | `tse_dataset` | `conda run -n tse_dataset python data/prepare_manifest.py` |
| Train model | `USEF-TFGridNet` | `conda run -n USEF-TFGridNet python train.py --config configs/hifi_tse.yaml` |
| Run inference | `USEF-TFGridNet` | `conda run -n USEF-TFGridNet python inference.py --config configs/hifi_tse.yaml --mix mix.wav --ref ref.wav -o output.wav` |

---

## Reference Codebases

The HiFi-TSE implementation draws from four core systems:

| Component | Reference | Role in HiFi-TSE |
|-----------|-----------|-------------------|
| **BSRNN** (Band-Split RNN) | [chenxinliang/BSRNN](https://github.com/chenxinliang/BSRNN) | Band-Split Encoder / Band-Merge Decoder — splits 1025 frequency bins into 53 psychoacoustically-motivated sub-bands for computational efficiency at 48 kHz |
| **USEF** | [X-LANCE/USEF-Speech-Extraction](https://github.com/X-LANCE/USEF-Speech-Extraction) | Cross-Multi-Head-Attention conditioning module — shared encoder for mix/ref, robust to noisy reference audio (Q=mix, K/V=ref) |
| **TF-GridNet** | [espnet/espnet](https://github.com/espnet/espnet) | Backbone separation network — 6 GridNet blocks with intra-frame BiLSTM (frequency), sub-band BiLSTM (time), and self-attention |
| **HiFi-GAN** | [jik876/hifi-gan](https://github.com/jik876/hifi-gan) | Discriminator design — Multi-Period Discriminator (periods 2,3,5,7,11) + Multi-Scale Discriminator (3 scales), adversarial + feature matching losses |

### Architecture Fusion

```
Mix WAV ──→ STFT ──→ [Band-Split Encoder] ──→ Z_mix ──┐
                          (BSRNN)                       ├──→ [USEF Cross-Attn] ──→ [TF-GridNet ×6] ──→ [Band-Merge Decoder] ──→ iSTFT ──→ Est WAV
Ref WAV ──→ STFT ──→ [Band-Split Encoder] ──→ Z_ref ──┘        (USEF)              (TF-GridNet)           (BSRNN)
                      (shared weights)

Est WAV ──→ [MPD + MSD Discriminator] ──→ Adversarial + Feature Matching Loss
                  (HiFi-GAN)
```

---

## HDF5 Dataset Format

All raw WAV files are pre-packed into HDF5 files during preparation to eliminate per-file I/O overhead at training time.

### File Layout

```
/data/tse_hdf5/
├── clean_speech/
│   ├── dns4.h5          # ~322k utterances, grouped by speaker
│   ├── ears.h5          # ~17k utterances, grouped by speaker
│   └── vctk.h5          # ~80k utterances, grouped by speaker
├── noise/
│   └── noise.h5         # DEMAND + AudioSet + WHAM merged
└── rir/
    └── rir.h5           # SLR26 + SLR28 merged
```

### HDF5 Internal Structure

**Clean speech files** (`dns4.h5`, `ears.h5`, `vctk.h5`):
```
/
├── speaker_ids          # dataset: list of speaker ID strings
├── <speaker_id>/        # one group per speaker
│   ├── utterances       # dataset: (N, max_samples) float32, zero-padded
│   ├── lengths          # dataset: (N,) int64, actual sample count per utterance
│   └── filenames        # dataset: (N,) strings, original filenames
└── attrs:
    ├── sample_rate: 48000
    ├── source: "dns4" | "ears" | "vctk"
    └── num_speakers: int
```

**Noise file** (`noise.h5`):
```
/
├── demand/
│   ├── waveforms        # dataset: (274, max_samples) float32
│   ├── lengths          # dataset: (274,) int64
│   └── filenames        # dataset: (274,) strings
├── audioset/
│   ├── waveforms        # dataset: (80189, max_samples) float32
│   ├── lengths          # dataset: (80189,) int64
│   └── filenames        # dataset: (80189,) strings
├── wham/
│   ├── waveforms        # dataset: (N, max_samples) float32, downmixed to mono
│   ├── lengths          # dataset: (N,) int64
│   └── filenames        # dataset: (N,) strings
└── attrs:
    └── sample_rate: 48000
```

**RIR file** (`rir.h5`):
```
/
├── slr26/
│   ├── waveforms        # dataset: (60006, max_samples) float32
│   ├── lengths          # dataset: (60006,) int64
│   └── filenames        # dataset: (60006,) strings
├── slr28/
│   ├── waveforms        # dataset: (248, max_samples) float32
│   ├── lengths          # dataset: (248,) int64
│   └── filenames        # dataset: (248,) strings
└── attrs:
    └── sample_rate: 48000
```

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| Separate HDF5 per source (dns4/ears/vctk) | Avoids single huge file; allows incremental rebuilds |
| Group by speaker for clean speech | Enables fast same-speaker reference lookup without index |
| Zero-padded + lengths array | HDF5 requires rectangular datasets; `lengths` restores true length at read time |
| `float32` storage | Avoids int16↔float conversion at training time |
| Noise/RIR grouped by source | Preserves provenance; flat index built at dataset init |
| Store at `/data/tse_hdf5/` | Co-located with raw data on same filesystem |

### Performance Impact

| Operation | WAV files | HDF5 |
|-----------|-----------|------|
| Open file | 1 syscall per utterance | 1 file handle, kept open |
| Random access | seek + read per file | single dataset slice `[i, :length]` |
| Metadata lookup | parse filename / scan dir | in-memory index from `speaker_ids` + `lengths` |
| DataLoader workers | contend on filesystem metadata | contend only on HDF5 read lock (thread-safe) |

### Pipeline Change Summary

| Step | Original Plan | With HDF5 |
|------|---------------|-----------|
| `data/prepare_manifest.py` | Scan → JSON manifests | Scan → JSON manifests **+ pack into HDF5 files** |
| `data/dataset.py` | `torchaudio.load(wav_path)` per sample | `h5py` slice from pre-opened file handles |
| Config `hifi_tse.yaml` | `manifest_path` pointing to JSON | `hdf5_dir` pointing to `/data/tse_hdf5/` + JSON index |
