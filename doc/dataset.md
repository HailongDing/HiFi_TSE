# HiFi-TSE Dataset

All audio is 48 kHz, mono, float32. Raw WAV files are pre-packed into HDF5 for
fast random-access I/O at training time. The packed dataset lives at
`/data/tse_hdf5/` (~769 GB total).

---

## 1. Audio Source Selection

The dataset is composed of three categories: clean speech (for target and
interferer signals), noise, and room impulse responses (RIR). Sources were
chosen to maximize diversity in speakers, acoustic environments, and noise
types while staying at a native 48 kHz sample rate to avoid resampling
artifacts.

### 1.1 Clean Speech

Three multi-speaker corpora provide broad coverage of speaker identities,
recording conditions, and speaking styles.

#### DNS4 Read Speech

- **Path**: `/data/48khz_dataset/DNS4_raw/datasets_fullband/clean_fullband/read_speech`
- **Why**: Largest available 48 kHz read-speech corpus. High speaker count
  ensures the model sees many distinct voices.
- **Raw files**: 321,996 WAV files in a flat directory.
- **Naming**: `book_XXXXX_chp_XXXX_reader_XXXXX_N_seg_N.wav`. Speaker ID is
  extracted from the `reader_XXXXX` field.
- **After packing**: 1,851 speakers, 185,826 utterances, 533.4 hours
  (368.8 GB). Average utterance ~10 s.

#### EARS

- **Path**: `/data/48khz_dataset/ears_dataset`
- **Why**: Studio-quality emotional / expressive speech. Covers adoration,
  anger, amusement, singing, whisper, and other styles that read-speech
  corpora lack.
- **Raw files**: 17,227 WAV files across 107 speaker directories (`p001`
  through `p107`). Per-speaker files include styles like
  `emo_adoration_freeform.wav`, `singing_verse.wav`.
- **After packing**: 102 speakers, 16,406 utterances, 95.1 hours (65.7 GB).
  Utterance durations range from 3.0 s to 234.8 s.

#### VCTK

- **Path**: `/data/48khz_dataset/dns_clean_dataset/vctk_wav48_silence_trimmed`
- **Why**: Well-known multi-accent English corpus (109 speakers). Adds accent
  diversity (Scottish, Indian, American, etc.) with silence-trimmed 48 kHz
  recordings.
- **Raw files**: 80,207 WAV files across 109 speaker directories.
- **After packing**: 104 speakers, 12,812 utterances, 14.4 hours (10.0 GB).
  Utterance durations range from 3.0 s to 13.7 s.

#### Clean speech filtering rules

| Rule | Detail |
|------|--------|
| Sample rate | Must be 48 kHz; non-48 kHz files are skipped |
| Minimum duration | Utterances shorter than 3.0 s are excluded (`audio.min_utterance_length`) |
| Minimum utterances per speaker | Speakers with fewer than 2 valid utterances are excluded (need at least 1 target + 1 reference) |

#### Train / val split

5% of speakers are held out for validation (`data.val_split_ratio: 0.05`),
selected deterministically by taking the last 5% of sorted speaker IDs per
source.

| Source | Train speakers | Val speakers |
|--------|---------------|-------------|
| DNS4 | 1,759 | 92 |
| EARS | 97 | 5 |
| VCTK | 99 | 5 |
| **Total** | **2,057** | **107** |

Speaker ID convention: `dns4_reader_XXXXX`, `ears_pXXX`, `vctk_pXXX`.

### 1.2 Noise

Four noise sources cover domestic, environmental, urban, and general
non-speech audio. All sources are 48 kHz.

#### DEMAND

- **Path**: `/data/48khz_dataset/dns_noise_dataset/demand`
- **Why**: High-quality multi-channel environmental recordings in 16+
  real-world locations (kitchen, park, bus, cafeteria, etc.). Provides
  long-duration stationary and quasi-stationary noise.
- **Raw files**: 272 WAV files, all exactly 300.0 s (5 minutes) each.
  Multi-channel files are downmixed to first channel at pack time.
- **After packing**: 272 clips, 15.7 GB.

#### WHAM

- **Path**: `/data/48khz_dataset/wham_dataset/high_res_wham/audio`
- **Why**: Real ambient noise from urban environments (streets, restaurants,
  bars). Recorded at 48 kHz to complement the clean speech. Long recordings
  provide varied temporal noise profiles.
- **Raw files**: 1,578 WAV files. Durations vary widely: 0.4 s to 1,560 s
  (26 minutes), mean 171.2 s. Filenames encode location and day.
- **Processing**: Stereo files are downmixed to mono. Files longer than 30 s
  are split into 30 s chunks (tail chunks shorter than 1 s are discarded).
- **After packing**: 10,083 clips (after chunking), 54.3 GB.

#### AudioSet

- **Path**: `/data/48khz_dataset/audioset_dataset/audio`
- **Why**: Broad coverage of real-world sound events (animal sounds, vehicles,
  domestic sounds, natural sounds). Adds non-stationary transient noise that
  DEMAND and WHAM lack.
- **Raw files**: 80,189 WAV files total. Durations 0.7 s to 20.0 s, mean
  18.5 s. Filenames contain category tags.
- **Filtering**: Files with `_Music` in the filename are excluded (46,414
  files, 57.9%) because music overlaps with speech harmonics and confuses
  the extractor. Kept categories include:
  - `Natural_sounds`
  - `Vehicle`
  - `Domestic_sounds`
  - `Animal`
  - Other uncategorized non-music files
- **After packing**: 33,775 clips, 120.4 GB.

#### DNS5 Noise

- **Path**: `/data/48khz_dataset/dns_noise_dataset/dns5_noise`
- **Why**: Large-scale noise collection from the DNS5 challenge. Adds volume
  and variety beyond the other three sources.
- **Raw files**: 63,810 WAV files. Durations 7.0 s to 10.0 s (mean 10.0 s).
  YouTube-ID-style filenames.
- **Sharding**: Split across 4 HDF5 shards (~15,953 clips each) to avoid a
  single oversized file.
- **After packing**: 63,810 clips across `dns5_noise_0.h5` through
  `dns5_noise_3.h5`, 122.4 GB total.

#### Noise summary

| Source | Raw files | After packing | Avg duration | Disk | Key characteristic |
|--------|-----------|---------------|-------------|------|--------------------|
| DEMAND | 272 | 272 clips | 300.0 s | 15.7 GB | Stationary environmental |
| WHAM | 1,578 | 10,083 clips | ~28 s | 54.3 GB | Urban ambient |
| AudioSet | 80,189 | 33,775 clips | ~17 s | 120.4 GB | Transient sound events |
| DNS5 | 63,810 | 63,810 clips | ~10 s | 122.4 GB | General-purpose |
| **Total** | **145,849** | **107,940 clips** | — | **312.8 GB** | — |

### 1.3 Room Impulse Responses (RIR)

Two RIR collections provide both large-scale simulated room diversity and
real-world acoustic measurements.

#### SLR26 — Simulated RIRs

- **Path**: `/data/48khz_dataset/DNS4_RIRs/datasets_fullband/impulse_responses/SLR26/simulated_rirs_48k`
- **Why**: 60,000 simulated RIRs covering small, medium, and large rooms.
  Provides massive room diversity for training.
- **Raw files**: 60,000 WAV files across 3 subdirectories: `smallroom`,
  `mediumroom`, `largeroom`. All exactly 1.0 s.
- **After packing**: 60,000 RIRs.

#### SLR28 — Real RIRs

- **Path**: `/data/48khz_dataset/DNS4_RIRs/datasets_fullband/impulse_responses/SLR28/RIRS_NOISES/real_rirs_isotropic_noises`
- **Why**: Real measured RIRs from actual rooms (lecture halls, offices,
  stairways, booths). Captures acoustic properties that simulation cannot
  fully replicate.
- **Raw files**: 248 WAV files. Durations 0.1 s to 10.0 s (mean 1.8 s).
  Includes AIR binaural (aula carolina, booth, lecture, office, stairway),
  AIR phone, and RWCP recordings.
- **After packing**: 248 RIRs.

#### RIR summary

| Source | Count | Duration | Type |
|--------|-------|----------|------|
| SLR26 | 60,000 | 1.0 s each | Simulated (small/medium/large rooms) |
| SLR28 | 248 | 0.1 - 10.0 s | Real measured |
| **Total** | **60,248** | — | `/data/tse_hdf5/train/rir/rir.h5` (11.6 GB) |

---

## 2. HDF5 File Layout

All packed data is stored under `/data/tse_hdf5/train/`.

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

### Clean speech HDF5 structure

```
/<file>.h5
  attrs:
    sample_rate: 48000
    source: "dns4" | "ears" | "vctk"
    num_speakers: int
  /speaker_ids              dataset (N,) string
  /<speaker_id>/            group
    attrs:
      count: int
    /0                      dataset (L0,) float32
    /1                      dataset (L1,) float32
    ...
    /durations              dataset (count,) int64
    /filenames              dataset (count,) string
```

Each utterance is stored as an individual variable-length dataset (no
zero-padding).

### Noise HDF5 structure

```
/<file>.h5
  attrs:
    sample_rate: 48000
    source: str
    count: int
  /0                        dataset (L0,) float32
  /1                        dataset (L1,) float32
  ...
  /filenames                dataset (count,) string
```

### RIR HDF5 structure

```
/rir.h5
  attrs:
    sample_rate: 48000
  /<source>/                group ("slr26" or "slr28")
    attrs:
      count: int
    /0                      dataset (L,) float32
    ...
    /filenames              dataset (count,) string
```

### I/O design decisions

| Decision | Rationale |
|----------|-----------|
| Per-utterance datasets (no padding) | Avoids wasting disk on zeros; each waveform is read at true length |
| Separate HDF5 per source | Avoids single huge file; allows incremental rebuilds |
| DNS5 noise sharded into 4 files | Keeps individual files under ~30 GB for filesystem friendliness |
| No compression | Maximizes read speed; GPU compute is the bottleneck, not disk I/O |
| Lazy file handles | Opened per DataLoader worker on first access, avoiding forked-fd issues |
| Metadata loaded at init | Speaker lists, counts, flat index read once via short-lived opens |

---

## 3. Dynamic Mixing Pipeline

Training samples are constructed on-the-fly in `data/dataset.py`
(`HiFiTSEDataset.__getitem__`). No pre-mixed audio is stored on disk.

### Per-sample construction

1. **Target selection** — Pick a speaker and utterance from the flat index.
   Random-crop to a segment length sampled uniformly from
   `mix_segment_range` (4.0 - 5.0 s at 48 kHz = 192,000 - 240,000 samples).

2. **Target-present (TP) vs target-absent (TA)** — Phase 1: always TP.
   Phases 2-3: TA with probability `ta_ratio` (0.2).

3. **Interferers** — Sample 1 to 3 interferer speakers
   (`num_interferers_range`), each excluded from the target speaker.
   Each interferer gets a random utterance, cropped to the same segment
   length.

4. **RIR convolution** — Target and each interferer are each convolved
   with an independently sampled random RIR (FFT convolution, L2-normalized
   RIR).

5. **Noise mixing** — A random noise clip is looped/cropped to fill the
   segment length, then mixed at a random SNR from `snr_range` (-5 to 15 dB).

6. **Reference enrollment** — A different utterance from the same target
   speaker, cropped to a length from `ref_segment_range` (4.0 - 5.0 s).
   With probability `noisy_ref_prob` (0.5), the reference is also convolved
   with a random RIR and mixed with noise at 5 - 20 dB SNR.

7. **Normalization** — If the mixture peak exceeds 0.95, both mixture and
   target are scaled to 0.9 peak.

For TA samples, the target waveform is replaced with zeros.

### Collation

`tse_collate_fn` crops all items in a batch to the shortest mixture length
and shortest reference length in that batch. No zero-padding is used.

### Output per sample

| Tensor | Shape | Description |
|--------|-------|-------------|
| `mix_wav` | `(T_mix,)` float32 | Mixture waveform |
| `ref_wav` | `(T_ref,)` float32 | Reference enrollment from same speaker |
| `target_wav` | `(T_mix,)` float32 | Clean target (zeros if TA) |
| `tp_flag` | scalar float | 1.0 (TP) or 0.0 (TA) |

---

## 4. Preparation Scripts

| Script | Environment | Purpose |
|--------|-------------|---------|
| `data/prepare_manifest.py` | `tse_dataset` | Full pipeline: scan, manifest, pack HDF5 |
| `data/pack_all_hdf5.sh` | `tse_dataset` | Shell wrapper: pack all sources sequentially |
| `data/pack_noise_only.py` | `tse_dataset` | Repack noise only (with AudioSet filtering) |
| `data/pack_noise_rir.py` | `tse_dataset` | Repack noise + RIR only |
| `data/download_dns5_noise.sh` | — | Download DNS5 noise dataset |

Rebuild everything:

```bash
tmux new-session -s pack 'bash data/pack_all_hdf5.sh'
```

Rebuild a single source:

```bash
conda run -n tse_dataset python data/prepare_manifest.py \
    --config configs/hifi_tse.yaml --source dns4
```

Validate scans without packing:

```bash
conda run -n tse_dataset python data/prepare_manifest.py \
    --config configs/hifi_tse.yaml --validate-only
```
