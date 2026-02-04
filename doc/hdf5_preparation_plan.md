# HDF5 Data Preparation Plan (Revised)

## Overview

Pack raw 48kHz WAV files into HDF5 for fast training I/O. All data consumed by
`dataset.py` via single-item random access — design optimized for that pattern.

---

## Variable-Length Training Segments (4–6 seconds)

Training uses variable-length mixture and reference segments to improve
robustness. This affects both HDF5 packing and the training pipeline.

### Config

```yaml
audio:
  sample_rate: 48000
  mix_segment_range: [4.0, 6.0]   # seconds, sampled per batch
  ref_segment_range: [4.0, 6.0]   # seconds, sampled per batch (independent)
  min_utterance_length: 3.0       # skip utterances shorter than this during packing
```

### HDF5 Packing Impact

- Utterances are stored at their **full original length** — no cropping during
  packing. Cropping to a random segment happens at training time.
- **Filter**: skip utterances shorter than `min_utterance_length` (3.0s =
  144,000 samples). This avoids excessive zero-padding when a 6s segment is
  requested from a very short utterance.
- Store each utterance's duration (samples) as metadata for fast filtering at
  training time without reading the waveform.

### Dataset Batching Strategy

- **Per-item random segment length**: each `__getitem__` independently
  samples `mix_len` from `mix_segment_range` and `ref_len` from
  `ref_segment_range`. Mix and ref lengths are independent.
- **Collate crops to batch minimum**: the custom `collate_fn` crops all
  items to the shortest `mix_len` and `ref_len` in the batch. No padding.
  Max waste is ~2s per item (range is 4–6s). Works with `num_workers > 0`.
- Utterances shorter than the sampled segment length are rejected and
  re-drawn (pick a different utterance instead of zero-padding the target).

### Code Changes Required

**`configs/hifi_tse.yaml`**:
- Replace `segment_length: 4.0` with `mix_segment_range` and
  `ref_segment_range`.
- Add `min_utterance_length: 3.0`.

**`data/prepare_manifest.py`**:
- Store utterance duration (in samples) alongside each waveform in HDF5
  as a dataset attribute (`duration_samples`).
- Filter utterances shorter than `min_utterance_length`.

**`data/dataset.py`**:
- `HiFiTSEDataset.__init__()`: read segment ranges from config.
- `HiFiTSEDataset.__getitem__()`: sample `mix_len` and `ref_len` per call.
  Reject and re-draw utterances shorter than `mix_len` for target/interferer,
  shorter than `ref_len` for reference.
- Add custom `collate_fn`: crop all items in the batch to the batch-minimum
  `mix_len` and `ref_len`. Since each item independently samples from
  [4.0, 6.0]s, the batch min is close to most items (max waste ~2s).
  This avoids padding entirely and works correctly with `num_workers > 0`.

**`train.py`**:
- Pass custom `collate_fn` to DataLoader.
- No loss masking needed since all items in a batch share the same lengths.

---

## Storage Strategy: Per-Utterance Datasets

Every audio file is stored as its own HDF5 dataset (variable-length 1D float32
array). No rectangular padding. This matches the training access pattern where
each `__getitem__` reads individual utterances by index.

---

## Clean Speech

### Sources

| Source | Path | Speakers | Files | Raw size |
|--------|------|----------|-------|----------|
| DNS4 | `/data/48khz_dataset/DNS4_raw/.../read_speech/` | ~1,948 | 196,044 | ~37 GB |
| EARS | `/data/48khz_dataset/ears_dataset/` | 107 | 17,227 | ~12 GB |
| VCTK | `/data/48khz_dataset/dns_clean_dataset/vctk_wav48_silence_trimmed/` | 109 | 79,417 | ~8 GB |

### HDF5 Layout (per source file: `dns4.h5`, `ears.h5`, `vctk.h5`)

```
/
├── speaker_ids                    # dataset: (num_speakers,) string
├── <speaker_id>/                  # one group per speaker
│   ├── count                      # attr: int, number of utterances
│   ├── <idx>                      # dataset: (L,) float32, one per utterance
│   ├── durations                  # dataset: (N,) int64, length in samples per utt
│   └── filenames                  # dataset: (N,) string
└── attrs:
    ├── sample_rate: 48000
    └── source: "dns4" | "ears" | "vctk"
```

- Speaker grouping preserved (needed for same-speaker reference lookup)
- Each utterance stored individually — no zero-padding
- `durations` dataset enables fast length-based filtering at training time
  without reading waveform data (needed for variable-length segment sampling)
- Filter: skip speakers with < 2 valid utterances (after length filter)
- Filter: skip files with sample rate ≠ 48000
- Filter: skip utterances shorter than `min_utterance_length` (3.0s = 144,000 samples)

### Estimated HDF5 Sizes

Stored as uncompressed float32 (4 bytes/sample) for fast training I/O.
Roughly 2x the raw WAV size (WAV uses 16-bit PCM = 2 bytes/sample).

| File | Estimated |
|------|-----------|
| `dns4.h5` | ~70 GB |
| `ears.h5` | ~24 GB |
| `vctk.h5` | ~16 GB |

---

## Noise

### Design Goal: Indoor-Focused TSE

This model targets indoor environments. Noise sources should emphasize:
- HVAC / air conditioning, appliance hum, kitchen sounds
- Office equipment, typing, fans
- Café / restaurant ambient
- TV / radio background bleed
- General room tone, footsteps, door sounds

Outdoor noise is still useful (attenuated through windows, robustness) but
indoor coverage is the priority.

### Sources

| Source | Path | Files | Raw size | Notes |
|--------|------|-------|----------|-------|
| DEMAND | `.../dns_noise_dataset/demand/` | 272 | ~19 GB | 17 environment categories (8 indoor) |
| WHAM | `.../wham_dataset/high_res_wham/audio/` | 1,578 | ~76 GB | Stereo 24-bit, ~150s each, urban/café ambient |
| AudioSet (filtered) | `.../audioset_dataset/audio/` | ~33,775 | ~8 GB | All except Music (see below) |
| DNS5 noise (new) | `.../dns_noise_dataset/dns5_noise/` | ~62,000 | ~58 GB | 150 fine-grained classes, native 48kHz |

**AudioSet filtering:**
- Included: Natural_sounds (685), Vehicle (400), Domestic_sounds (38), Animal (32,652)
- Excluded: Music (46,414) — pitched/harmonic content interferes with speech extraction
- Animal sounds add diversity and robustness; some indoor-relevant (pets)

**DNS5 noise_fullband:**
- Downloaded from DNS Challenge 5 (ICASSP 2023)
- 63,811 files (~10s each) from AudioSet + Freesound, all at 48kHz
- Curated by Microsoft for speech enhancement — unsuitable noise types
  already filtered out. No per-file category labels available.
- Flat directory of YouTube-ID-named WAVs (no subfolder structure)
- Note: sourced from AudioSet/Freesound, so partial content overlap with
  the existing AudioSet dataset above. YouTube IDs do not overlap (verified),
  but similar sound categories may be represented in both. Keeping both
  maximizes clip-level diversity.

### WHAM Pre-processing

WHAM files are stereo 24-bit, ~150s each. At training time only a 4–6s random
crop is used. Storing full 150s wastes I/O bandwidth — each read pulls ~28 MB
to use 384–576 KB.

**Plan:** Pre-segment WHAM files into 30s chunks during packing.
- 1,578 files × ~5 chunks each = ~7,890 chunks
- Each chunk: 30s × 48000 × 4 bytes = ~5.5 MB (vs ~28 MB for full file)
- Reduces per-read I/O by ~5x with no loss of diversity

### HDF5 Layout (per source file: `demand.h5`, `wham.h5`, `audioset.h5`, `dns5_noise.h5`)

Each noise source gets its own HDF5 file for resumability and manageability.
All share the same internal layout:

```
/
├── count                          # attr: int, total number of clips
├── 0, 1, 2, ...                  # datasets: (L,) float32, one per clip
├── filenames                      # dataset: (N,) string
└── attrs:
    └── sample_rate: 48000
```

### Estimated Sizes

Stored as uncompressed float32 for fast training I/O (no decompression
overhead). Target max file size: **~40 GB** (suitable for 64 GB RAM system).

| File | Files | Estimated |
|------|-------|-----------|
| `demand.h5` | 272 | ~38 GB |
| `wham.h5` | ~7,890 chunks | ~24 GB (after mono downmix + 30s chunking) |
| `audioset.h5` | ~33,775 | ~8 GB |
| `dns5_noise_0.h5` | ~15,953 | ~30 GB |
| `dns5_noise_1.h5` | ~15,953 | ~30 GB |
| `dns5_noise_2.h5` | ~15,953 | ~30 GB |
| `dns5_noise_3.h5` | ~15,952 | ~30 GB |
| **Total noise** | | **~190 GB** |

DNS5 noise is sharded into 4 files (~16k files each, ~30 GB per shard).
Files are assigned to shards by sorted filename order for deterministic splits.

---

## RIR (Room Impulse Response)

### Sources

| Source | Path | Files | Notes |
|--------|------|-------|-------|
| SLR26 simulated | `.../SLR26/simulated_rirs_48k/{small,medium,large}room/` | 60,000 | Simulated, 48kHz |
| SLR28 real | `.../SLR28/.../real_rirs_isotropic_noises/` | 248 | Real recordings |

### Pre-packing Validation Required

Before packing, verify:
1. **Sample rate**: confirm all files are 48kHz (SLR28 real RIRs may vary)
2. **File format**: confirm all are readable mono WAV
3. **Duration**: RIRs should be short (< 10s typically). Flag any outliers.

### HDF5 Layout (`rir.h5`)

```
/
├── slr26/
│   ├── count                      # attr: int
│   ├── 0, 1, 2, ...              # datasets: (L,) float32 each
│   └── filenames                  # dataset: (N,) string
├── slr28/
│   ├── count                      # attr: int
│   ├── 0, 1, 2, ...              # datasets: (L,) float32 each
│   └── filenames                  # dataset: (N,) string
└── attrs:
    └── sample_rate: 48000
```

### Estimated Size

RIRs are short (~0.1-10s). 60,248 files at average ~1s:
- 60,248 × 48,000 × 4 bytes ≈ ~11 GB uncompressed
- With gzip: **~2-3 GB**

---

## Output Structure

```
/data/tse_hdf5/
├── train/
│   ├── clean_speech/
│   │   ├── dns4.h5          # ~70 GB
│   │   ├── ears.h5          # ~24 GB
│   │   └── vctk.h5          # ~16 GB
│   ├── noise/
│   │   ├── demand.h5        # ~38 GB
│   │   ├── wham.h5          # ~24 GB
│   │   ├── audioset.h5      # ~8 GB
│   │   ├── dns5_noise_0.h5  # ~30 GB
│   │   ├── dns5_noise_1.h5  # ~30 GB
│   │   ├── dns5_noise_2.h5  # ~30 GB
│   │   └── dns5_noise_3.h5  # ~30 GB
│   └── rir/
│       └── rir.h5           # ~2-3 GB
├── val/
│   └── val_mixtures.h5      # pre-generated fixed validation mixtures
└── manifests/
    ├── speaker_manifest.json       # all speakers with train/val split labels
    ├── noise_manifest.json
    └── rir_manifest.json
```

**Total estimated: ~310-330 GB**

---

## Execution Plan

### Step 0: Download DNS5 Noise Fullband

Download the DNS5 noise_fullband dataset (48kHz, ~62,000 files, 150 classes).

**Source:** Azure blob `https://dnschallengepublic.blob.core.windows.net/dns5archive/V5_training_dataset/`

**Archives (9 files, ~39 GB compressed → ~58 GB extracted):**

| Archive | Source |
|---------|--------|
| `datasets_fullband.noise_fullband.audioset_000.tar.bz2` | AudioSet noise |
| `datasets_fullband.noise_fullband.audioset_001.tar.bz2` | AudioSet noise |
| `datasets_fullband.noise_fullband.audioset_002.tar.bz2` | AudioSet noise |
| `datasets_fullband.noise_fullband.audioset_003.tar.bz2` | AudioSet noise |
| `datasets_fullband.noise_fullband.audioset_004.tar.bz2` | AudioSet noise |
| `datasets_fullband.noise_fullband.audioset_005.tar.bz2` | AudioSet noise |
| `datasets_fullband.noise_fullband.audioset_006.tar.bz2` | AudioSet noise |
| `datasets_fullband.noise_fullband.freesound_000.tar.bz2` | Freesound noise |
| `datasets_fullband.noise_fullband.freesound_001.tar.bz2` | Freesound noise |

**Destination:** `/data/48khz_dataset/dns_noise_dataset/dns5_noise/`
- All files extracted into a single flat directory (YouTube-ID-named WAVs)
- Download in tmux to survive connection drops
- Archives stored in `dns5_noise/_archives/` — delete after verification

**Post-download:** ✅ DONE (63,811 WAV files, 95 GB incl. archives)
- Verified: all files 48kHz, ~10s each, mono
- No category labels available — use all files as-is

### Step 1: Scan & Validate

Run with `--skip-hdf5` to generate manifests and print validation summary:
- Speaker counts and utterance counts per source
- Noise file counts per category (with AudioSet filtering and DNS5 categories)
- RIR sample rate validation (flag non-48kHz files)
- WHAM file duration distribution

### Step 2: Pack Clean Speech

Pack DNS4, EARS, VCTK sequentially. Per-utterance storage, no padding.
One source at a time so partial completion is resumable.

### Step 3: Pack Noise

Pack each source into its own HDF5 file, sequentially:
1. `demand.h5` — 272 files, straightforward
2. `wham.h5` — 1,578 files → ~7,890 chunks (30s chunking + mono downmix)
3. `audioset.h5` — ~33,775 files (filtered: all except Music)
4. `dns5_noise_{0..3}.h5` — 63,811 files across 4 shards (~30 GB each)

Per-source files allow resuming if one source fails without repacking others.

### Step 4: Pack RIR

Pack SLR26 + SLR28 (only 48kHz files, skip others).

### Step 5: Verify

- Open each HDF5, check structure matches expected layout
- Read a random sample from each, verify shape and value range
- Print total file sizes

### Step 6: Generate Validation Mixtures

- Load held-out speakers from training HDF5 + noise/RIR data
- Generate 1,000 fixed mixtures (800 TP, 200 TA) at 6.0s segment length
- Save to `val/val_mixtures.h5`

---

## Changes to Code

### `configs/hifi_tse.yaml`

- Replace `audio.segment_length: 4.0` with:
  - `audio.mix_segment_range: [4.0, 6.0]`
  - `audio.ref_segment_range: [4.0, 6.0]`
  - `audio.min_utterance_length: 3.0`
- Add DNS5 noise path to `data.noise` list

### `data/prepare_manifest.py`

- `pack_clean_speech_hdf5()`: change from rectangular `(N, max_len)` to
  per-utterance datasets `<speaker_id>/<idx>` with `count` attr.
  Add `durations` dataset (int64 array of sample counts per utterance).
  Filter utterances shorter than `min_utterance_length`.
  Implement train/val speaker split (last 5% by sorted ID).
- `pack_noise_hdf5()`: produce per-source HDF5 files (`demand.h5`,
  `wham.h5`, `audioset.h5`, `dns5_noise.h5`). Add WHAM 30s chunking.
- `generate_val_mixtures()`: new function for Step 6.
- Add `--validate-only` flag for Step 1 pre-checks.
- Add `--source` flag to pack a single source (for resumability).
- Output paths now under `train/` subdirectory.

### `data/dataset.py`

- `CleanSpeechIndex`: read per-utterance datasets via
  `hf[speaker_id][str(utt_idx)][:]`. Load `durations` array per speaker
  for fast length-based filtering during segment sampling.
- `HiFiTSEDataset.__init__()`: read `mix_segment_range` and
  `ref_segment_range` from config.
- `HiFiTSEDataset.__getitem__()`: sample `mix_len` and `ref_len` per call.
  Reject and re-draw utterances shorter than the requested segment length
  (for target, interferer, and reference). No audio resampling needed —
  all sources are native 48kHz.
- Add custom `collate_fn` for DataLoader batching.
- `NoiseIndex`: open multiple per-source HDF5 files (`demand.h5`,
  `wham.h5`, `audioset.h5`, `dns5_noise_{0..3}.h5`). Build unified flat
  index across all files. Paths now under `train/noise/`.
- `RIRIndex`: path now under `train/rir/`. No layout change.
- h5py multi-worker safety: open file handles lazily per worker
  (defer `h5py.File()` open to first access in each DataLoader worker).

### `train.py`

- Pass custom `collate_fn` to DataLoader.
- No loss masking needed (uniform lengths within each batch).

---

## Validation Split

**Strategy: Held-out speakers + pre-generated fixed validation mixtures.**

- Reserve ~5% of speakers from each source for validation:
  - DNS4: ~97 speakers (out of ~1,948)
  - EARS: ~5 speakers (out of ~107)
  - VCTK: ~5 speakers (out of ~109)
- Speaker assignment is deterministic (sorted by ID, last N% reserved)
- Held-out speakers are **excluded** from training HDF5 packing
- Pre-generate ~1,000 fixed validation mixtures from held-out speakers
- Ensures reproducible evaluation across training epochs/runs

### Validation HDF5 Layout (`val/val_mixtures.h5`)

```
/
├── count                          # attr: int, number of mixtures
├── <idx>/                         # one group per mixture
│   ├── mix                        # dataset: (L,) float32, the mixture waveform
│   ├── ref                        # dataset: (L',) float32, the reference waveform
│   ├── target                     # dataset: (L,) float32, clean target (zeros if TA)
│   └── target_present             # attr: bool
└── attrs:
    ├── sample_rate: 48000
    ├── num_mixtures: 1000
    └── mix_segment_seconds: 6.0   # fixed at max length for consistent eval
```

- Fixed segment length (6.0s) for consistent metrics across checkpoints
- ~80% TP, ~20% TA to match training TA ratio
- Generated after training HDF5 is complete (uses same noise + RIR data)

---

## Conda Environments

| Step | Environment | Command |
|------|-------------|---------|
| Scan & Validate | `tse_dataset` | `conda run -n tse_dataset python data/prepare_manifest.py --config configs/hifi_tse.yaml --skip-hdf5` |
| Pack HDF5 | `tse_dataset` | `tmux` + `conda run -n tse_dataset python data/prepare_manifest.py --config configs/hifi_tse.yaml` |
| Train | `USEF-TFGridNet` | `conda run -n USEF-TFGridNet python train.py --config configs/hifi_tse.yaml` |
