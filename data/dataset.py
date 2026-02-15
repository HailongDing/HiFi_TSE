"""HDF5-backed dynamic mixing dataset for HiFi-TSE training.

Reads per-utterance HDF5 files and performs on-the-fly mixing:
  - Target speaker selection + reference from same speaker
  - 1-3 interferer speakers
  - RIR convolution for reverb simulation
  - Noise mixing at random SNR
  - Target-absent (TA) samples at configurable ratio
  - Fixed-length segments (default 4s), all samples same size per batch

HDF5 layout (produced by data/prepare_manifest.py):
  clean_speech/{dns4,ears,vctk}.h5:
      /speaker_ids             (N,) string
      /<speaker_id>/count      attr int
      /<speaker_id>/<idx>      (L,) float32 per utterance
      /<speaker_id>/durations  (N,) int64

  noise/{demand,wham,audioset,dns5_noise_0..3}.h5:
      /count                   attr int
      /0, 1, ...               (L,) float32 per clip

  rir/rir.h5:
      /<source>/count          attr int
      /<source>/0, 1, ...      (L,) float32 per RIR
"""

import multiprocessing as mp
import os
import random

import h5py
import numpy as np
import torch
from scipy.signal import fftconvolve
from torch.utils.data import Dataset


SAMPLE_RATE = 48000


# ---------------------------------------------------------------------------
# Audio utilities (numpy, used in __getitem__)
# ---------------------------------------------------------------------------

def _random_crop_np(wav, target_length):
    """Random crop or zero-pad a numpy array to target_length."""
    length = len(wav)
    if length >= target_length:
        offset = random.randint(0, length - target_length)
        return wav[offset:offset + target_length]
    else:
        out = np.zeros(target_length, dtype=wav.dtype)
        out[:length] = wav
        return out


def _loop_to_length_np(wav, target_length):
    """Loop a numpy array to fill target_length."""
    if len(wav) >= target_length:
        offset = random.randint(0, len(wav) - target_length)
        return wav[offset:offset + target_length]
    reps = (target_length // len(wav)) + 1
    looped = np.tile(wav, reps)
    offset = random.randint(0, len(looped) - target_length)
    return looped[offset:offset + target_length]


def _apply_rir_np(wav, rir):
    """Convolve waveform with RIR using numpy FFT convolution.

    Returns the full convolution result (length = len(wav) + len(rir) - 1).
    Caller is responsible for cropping to desired length afterwards.
    """
    rir = rir / (np.linalg.norm(rir) + 1e-8)
    return fftconvolve(wav, rir, mode="full")


def _precrop_for_rir_np(wav, target_length, max_rir_length=96000):
    """Pre-crop a long waveform before RIR convolution for efficiency.

    Crops to (target_length + max_rir_length) so that after full convolution
    and final crop, reverb tails from preceding audio are preserved.
    Returns the pre-cropped waveform (or original if already short enough).
    """
    safe_length = target_length + max_rir_length
    if len(wav) <= safe_length:
        return wav
    offset = random.randint(0, len(wav) - safe_length)
    return wav[offset:offset + safe_length]


def _mix_at_snr_np(signal, noise, snr_db):
    """Mix signal and noise at target SNR."""
    sig_power = np.mean(signal ** 2) + 1e-8
    noise_power = np.mean(noise ** 2) + 1e-8
    snr_linear = 10.0 ** (snr_db / 10.0)
    scale = np.sqrt(sig_power / (noise_power * snr_linear + 1e-8))
    return signal + scale * noise


def _peak_normalize_np(wav, target_peak=0.9):
    """Peak-normalize to target amplitude."""
    peak = np.abs(wav).max()
    if peak > 1e-8:
        wav = wav * (target_peak / peak)
    return wav


# ---------------------------------------------------------------------------
# HDF5 Index classes (lazy-open for multi-worker DataLoader safety)
#
# Metadata (speaker lists, counts, flat index) is read once in __init__
# via context-managed opens.  Persistent file handles are opened lazily
# on first access inside a worker process (_get_handle), so forked/spawned
# workers each get their own independent h5py handles.
# ---------------------------------------------------------------------------

class CleanSpeechIndex:
    """Index for per-utterance clean speech HDF5 files."""

    def __init__(self, hdf5_dir):
        clean_dir = os.path.join(hdf5_dir, "train", "clean_speech")
        self.h5_paths = sorted(
            os.path.join(clean_dir, f)
            for f in os.listdir(clean_dir) if f.endswith(".h5")
        )

        self.speaker_to_file = {}   # speaker_id -> h5 path
        self.speaker_counts = {}    # speaker_id -> utterance count
        self.all_speakers = []
        self.flat_index = []        # [(speaker_id, utt_idx), ...]

        for path in self.h5_paths:
            with h5py.File(path, "r") as hf:
                speaker_ids = [
                    s.decode() if isinstance(s, bytes) else s
                    for s in hf["speaker_ids"][:]
                ]
                for spk_id in speaker_ids:
                    if spk_id not in hf:
                        continue
                    count = hf[spk_id].attrs.get("count", 0)
                    if count < 2:
                        continue
                    self.speaker_to_file[spk_id] = path
                    self.speaker_counts[spk_id] = count
                    self.all_speakers.append(spk_id)
                    for i in range(count):
                        self.flat_index.append((spk_id, i))

        self._handles = {}  # lazily opened per worker

    def _get_handle(self, path):
        if path not in self._handles:
            self._handles[path] = h5py.File(path, "r", rdcc_nbytes=512*1024)
        return self._handles[path]

    def get_utterance(self, speaker_id, utt_idx):
        """Read a single utterance as numpy float32 array."""
        path = self.speaker_to_file[speaker_id]
        hf = self._get_handle(path)
        return hf[speaker_id][str(utt_idx)][:]

    def get_random_utterance(self, speaker_id, exclude_idx=None):
        """Get a random utterance from a speaker, optionally excluding one."""
        count = self.speaker_counts[speaker_id]
        candidates = list(range(count))
        if exclude_idx is not None and len(candidates) > 1:
            candidates.remove(exclude_idx)
        idx = random.choice(candidates)
        return self.get_utterance(speaker_id, idx)

    def get_random_speaker(self, exclude=None):
        """Get a random speaker ID, optionally excluding some."""
        if exclude:
            candidates = [s for s in self.all_speakers if s not in exclude]
        else:
            candidates = self.all_speakers
        return random.choice(candidates)

    def num_utterances(self):
        return len(self.flat_index)

    def close(self):
        for hf in self._handles.values():
            hf.close()
        self._handles.clear()


class NoiseIndex:
    """Index for per-clip noise HDF5 files (multiple files)."""

    def __init__(self, hdf5_dir):
        noise_dir = os.path.join(hdf5_dir, "train", "noise")
        self.h5_paths = sorted(
            os.path.join(noise_dir, f)
            for f in os.listdir(noise_dir) if f.endswith(".h5")
        )

        # (h5_path, clip_index_str) for every clip across all files
        self.flat_index = []
        for path in self.h5_paths:
            with h5py.File(path, "r") as hf:
                count = hf.attrs.get("count", 0)
                for i in range(count):
                    self.flat_index.append((path, str(i)))

        self._handles = {}

    def _get_handle(self, path):
        if path not in self._handles:
            self._handles[path] = h5py.File(path, "r", rdcc_nbytes=512*1024)
        return self._handles[path]

    def get_random(self):
        """Get a random noise waveform."""
        path, idx_str = random.choice(self.flat_index)
        hf = self._get_handle(path)
        return hf[idx_str][:]

    def close(self):
        for hf in self._handles.values():
            hf.close()
        self._handles.clear()


class RIRIndex:
    """Index for RIR HDF5 file (grouped by source)."""

    def __init__(self, hdf5_dir):
        self.h5_path = os.path.join(hdf5_dir, "train", "rir", "rir.h5")

        # (source_name, clip_index_str) for every RIR
        self.flat_index = []
        with h5py.File(self.h5_path, "r") as hf:
            for src in hf.keys():
                if not isinstance(hf[src], h5py.Group):
                    continue
                count = hf[src].attrs.get("count", 0)
                for i in range(count):
                    self.flat_index.append((src, str(i)))

        self._handle = None

    def _get_handle(self):
        if self._handle is None:
            self._handle = h5py.File(self.h5_path, "r", rdcc_nbytes=512*1024)
        return self._handle

    def get_random(self):
        """Get a random RIR waveform."""
        src, idx_str = random.choice(self.flat_index)
        hf = self._get_handle()
        return hf[src][idx_str][:]

    def close(self):
        if self._handle is not None:
            self._handle.close()
            self._handle = None


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def tse_collate_fn(batch):
    """Stack fixed-length samples into a batch.

    All samples have identical mix and ref lengths (fixed segment config),
    so this is a straightforward stack.
    """
    mix_wavs, ref_wavs, target_wavs, tp_flags = zip(*batch)

    mix_batch = torch.stack(mix_wavs)
    ref_batch = torch.stack(ref_wavs)
    target_batch = torch.stack(target_wavs)
    tp_batch = torch.stack(list(tp_flags))

    return mix_batch, ref_batch, target_batch, tp_batch


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HiFiTSEDataset(Dataset):
    """Dynamic mixing dataset for HiFi-TSE training.

    Each __getitem__ constructs a fresh mixture on the fly using fixed
    segment lengths.  All samples in a batch have identical tensor shapes,
    so tse_collate_fn is a simple torch.stack.
    """

    def __init__(self, cfg, phase=1):
        data_cfg = cfg["data"]
        hdf5_dir = data_cfg["hdf5_dir"]
        audio_cfg = cfg["audio"]

        # Fixed segment lengths (in samples)
        self.mix_seg = int(audio_cfg["mix_segment_seconds"] * SAMPLE_RATE)
        self.ref_seg = int(audio_cfg["ref_segment_seconds"] * SAMPLE_RATE)

        self.ta_ratio = data_cfg["ta_ratio"]
        self.noisy_ref_prob = data_cfg["noisy_ref_prob"]
        # Shared state for dynamic noisy_ref_prob (readable by DataLoader workers)
        self._shared_noisy_ref_prob = mp.Value('d', 0.0)
        self.snr_range = data_cfg["snr_range"]
        self.num_interferers_range = data_cfg["num_interferers_range"]
        self.phase = phase

        # Build indices (metadata only; HDF5 handles opened lazily)
        self.clean_index = CleanSpeechIndex(hdf5_dir)
        self.noise_index = NoiseIndex(hdf5_dir)
        self.rir_index = RIRIndex(hdf5_dir)

    def set_phase(self, phase):
        """Update training phase (controls TA ratio)."""
        self.phase = phase

    def set_noisy_ref_prob(self, prob):
        """Update noisy_ref_prob (propagates to DataLoader workers via shared memory)."""
        self._shared_noisy_ref_prob.value = prob

    def close_handles(self):
        """Close all HDF5 file handles to free memory."""
        self.clean_index.close()
        self.noise_index.close()
        self.rir_index.close()

    def __len__(self):
        return self.clean_index.num_utterances()

    def __getitem__(self, idx):
        """
        Returns:
            mix_wav:  (mix_seg,) float32 tensor
            ref_wav:  (ref_seg,) float32 tensor
            target_wav: (mix_seg,) float32 tensor (zeros if TA)
            target_present: float scalar, 1.0 for TP, 0.0 for TA
        """
        mix_seg = self.mix_seg
        ref_seg = self.ref_seg

        # 1. Pick target speaker and utterance
        #    Convolve-then-crop: apply RIR to full utterance, then crop to segment
        spk_id, utt_idx = self.clean_index.flat_index[idx]
        target_wav = self.clean_index.get_utterance(spk_id, utt_idx)
        target_wav = _peak_normalize_np(target_wav)
        target_wav = _precrop_for_rir_np(target_wav, mix_seg)
        target_wav = _apply_rir_np(target_wav, self.rir_index.get_random())
        target_wav = _random_crop_np(target_wav, mix_seg)

        # 2. Decide TP or TA
        if self.phase == 1:
            target_present = True
        else:
            target_present = random.random() >= self.ta_ratio

        # 3. Pick interferers (convolve-then-crop each)
        num_interferers = random.randint(*self.num_interferers_range)
        interferer_wavs = []
        exclude_speakers = {spk_id}
        for _ in range(num_interferers):
            int_spk = self.clean_index.get_random_speaker(
                exclude=exclude_speakers)
            exclude_speakers.add(int_spk)
            int_wav = self.clean_index.get_random_utterance(int_spk)
            int_wav = _peak_normalize_np(int_wav)
            int_wav = _precrop_for_rir_np(int_wav, mix_seg)
            int_wav = _apply_rir_np(int_wav, self.rir_index.get_random())
            int_wav = _random_crop_np(int_wav, mix_seg)
            interferer_wavs.append(int_wav)

        # 4. Build mixture
        noise_wav = self.noise_index.get_random()
        noise_wav = _loop_to_length_np(noise_wav, mix_seg)
        snr_db = random.uniform(*self.snr_range)

        if target_present:
            speech_sum = target_wav + sum(interferer_wavs)
            mix_wav = _mix_at_snr_np(speech_sum, noise_wav, snr_db)
        else:
            # Target absent: no target in mixture
            if len(interferer_wavs) == 0:
                speech_sum = np.zeros(mix_seg, dtype=np.float32)
            else:
                speech_sum = sum(interferer_wavs)
                if np.abs(speech_sum).max() < 1e-8:
                    speech_sum = interferer_wavs[0]  # fallback
            mix_wav = _mix_at_snr_np(speech_sum, noise_wav, snr_db)
            target_wav = np.zeros(mix_seg, dtype=np.float32)

        # 5. Reference: different utterance from same speaker
        ref_wav = self.clean_index.get_random_utterance(
            spk_id, exclude_idx=utt_idx)
        ref_wav = _peak_normalize_np(ref_wav)

        # Optionally add noise/reverb to reference (convolve-then-crop)
        noisy_ref_prob = self._shared_noisy_ref_prob.value
        if random.random() < noisy_ref_prob:
            ref_wav = _precrop_for_rir_np(ref_wav, ref_seg)
            ref_wav = _apply_rir_np(ref_wav, self.rir_index.get_random())
            ref_wav = _random_crop_np(ref_wav, ref_seg)
            ref_noise = self.noise_index.get_random()
            ref_noise = _loop_to_length_np(ref_noise, ref_seg)
            ref_snr = random.uniform(5.0, 20.0)
            ref_wav = _mix_at_snr_np(ref_wav, ref_noise, ref_snr)
        else:
            ref_wav = _random_crop_np(ref_wav, ref_seg)

        # Normalize mixture to prevent clipping
        mix_max = np.abs(mix_wav).max()
        if mix_max > 0.95:
            scale = 0.9 / mix_max
            mix_wav = mix_wav * scale
            target_wav = target_wav * scale

        # Convert to tensors
        mix_wav = torch.from_numpy(mix_wav.astype(np.float32))
        ref_wav = torch.from_numpy(ref_wav.astype(np.float32))
        target_wav = torch.from_numpy(target_wav.astype(np.float32))
        tp_flag = torch.tensor(1.0 if target_present else 0.0)

        return mix_wav, ref_wav, target_wav, tp_flag
