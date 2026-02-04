# Code Review Issues

Analysis performed by Codex (o3) and verified against source code.
Date: 2026-02-04

---

## CRITICAL — Will cause incorrect training

### 1. GAN generator loss contaminates discriminator gradients

- **File**: `train.py:236,244`
- **Description**: In phase 3, `discriminator(est_wav)` at line 236 builds a
  computation graph through the discriminator. When `loss_G.backward()` runs
  at line 244, gradients from the generator's adversarial loss accumulate onto
  discriminator parameters on top of the D loss gradients from line 229. When
  `opt_D.step()` fires at line 254, the D update is contaminated — it gets
  pushed in the direction the generator wants, undermining adversarial training.
- **Fix**: Zero D gradients after D backward/step and before the G forward
  through the discriminator, or freeze D parameters during the G adversarial
  forward pass.

### 2. Discriminator LR scheduler decays during phases 1-2 when D is unused

- **File**: `train.py:258`
- **Description**: `sched_D.step()` is called every accumulation step starting
  from step 0. By the time phase 3 starts at step 300k, cosine annealing has
  decayed the D learning rate to near zero (300k/500k = 60% through the
  schedule). The discriminator effectively begins training with a crippled
  learning rate.
- **Fix**: Only step `sched_D` when `current_phase >= 3`, or create `sched_D`
  with `T_max` based on the number of GAN-phase steps only (500k - 300k =
  200k).

### 3. Discriminator trained on zero-valued targets for TA samples

- **File**: `train.py:226`, `data/dataset.py:365`
- **Description**: In phases 2-3, 20% of samples are target-absent with
  `target_wav = zeros`. The discriminator receives these zeros as "real"
  targets at line 226, biasing it to classify silence as real speech. This
  degrades the discriminator's ability to judge speech quality.
- **Fix**: Mask TA samples out of the discriminator loss. Only pass TP samples
  to the discriminator:
  ```python
  tp_mask = tp_flag.bool()
  if tp_mask.any():
      d_real_out, d_real_feat = discriminator(target_wav[tp_mask])
      d_fake_out, d_fake_feat = discriminator(est_wav_detach[tp_mask])
  ```

---

## MAJOR — Broken helper scripts (do not affect training)

### 4. pack_noise_rir.py and pack_noise_only.py import nonexistent function

- **File**: `data/pack_noise_rir.py:17`, `data/pack_noise_only.py:15`
- **Description**: Both scripts import `pack_noise_hdf5` from
  `data.prepare_manifest`, but that function does not exist. The actual
  functions are `pack_noise_hdf5_single` and `pack_noise_hdf5_sharded`.
  Running either script will crash with `ImportError`.
- **Impact**: Standalone helper scripts only. The main pipeline
  (`prepare_manifest.py` and `pack_all_hdf5.sh`) is unaffected. The HDF5
  files were built correctly.
- **Fix**: Update the imports to reference `pack_noise_hdf5_single`.

### 5. pack_noise_rir.py indexes noise config as a list, but config is a dict

- **File**: `data/pack_noise_rir.py:31-33`
- **Description**: `noise_paths[0]`, `noise_paths[1]`, `noise_paths[2]` treat
  `cfg["data"]["noise"]` as a list, but in the current config it is a dict
  with keys `demand`, `wham`, `audioset`, `dns5_noise`. This raises
  `TypeError`.
- **Impact**: Standalone helper script only.
- **Fix**: Index by key: `noise_paths["demand"]`, `noise_paths["wham"]`,
  `noise_paths["audioset"]`.

### 6. RIR manifest stores counts instead of file lists

- **File**: `data/prepare_manifest.py:543`, `data/pack_noise_rir.py:44-51`
- **Description**: `rir_manifest.json` is written as
  `{"slr26": 60000, "slr28": 248}` (counts only). But `pack_noise_rir.py`
  loads this and passes it to `pack_rir_hdf5()`, which expects dicts of file
  path lists. Iterating over integers instead of paths will fail.
- **Impact**: Standalone helper script only. The main pipeline passes in-memory
  file lists directly and never reads the RIR manifest back for packing.
- **Fix**: Either store full file lists in `rir_manifest.json`, or have
  `pack_noise_rir.py` re-scan the RIR directories instead of loading the
  manifest.

---

## MINOR

### 7. TA fallback crashes if num_interferers_range allows 0

- **File**: `data/dataset.py:361-363`
- **Description**: If `num_interferers_range` is configured as `[0, 3]`, a TA
  sample with 0 interferers produces an empty `interferers_reverbed` list.
  `sum([])` returns integer 0, and `interferers_reverbed[0]` raises
  `IndexError`.
- **Current config**: `[1, 3]` — minimum 1 interferer, so this does not
  trigger today.
- **Fix**: Add a guard:
  ```python
  if len(interferers_reverbed) == 0:
      speech_sum = np.zeros(mix_seg, dtype=np.float32)
  else:
      speech_sum = sum(interferers_reverbed)
  ```

### 8. Chunked inference infinite loop if overlap >= 1.0

- **File**: `inference.py:131`
- **Description**: `hop_samples = int(chunk_samples * (1.0 - overlap))`. If
  the user passes `--overlap 1.0` (or greater), `hop_samples` becomes 0 (or
  negative), and the while loop at line 66 never advances.
- **Fix**: Clamp overlap and validate hop:
  ```python
  args.overlap = min(args.overlap, 0.99)
  hop_samples = max(1, int(chunk_samples * (1.0 - args.overlap)))
  ```
