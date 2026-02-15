"""Audio utility functions for HiFi-TSE.

All functions operate on torch tensors unless noted otherwise.
Designed for 48kHz mono audio.
"""

import torch
import torch.nn.functional as F
import torchaudio


def stft(waveform, n_fft, win_length, hop_length):
    """Compute STFT, returning real and imaginary parts stacked on dim=1.

    Args:
        waveform: (B, L) float tensor
        n_fft: FFT size
        win_length: window length in samples
        hop_length: hop length in samples

    Returns:
        (B, 2, T, F) float tensor where dim=1 is [real, imag], F = n_fft//2 + 1
    """
    window = torch.hann_window(win_length, device=waveform.device)
    # torch.stft returns (B, F, T, 2) with return_complex=False
    spec = torch.stft(
        waveform, n_fft, hop_length=hop_length, win_length=win_length,
        window=window, return_complex=True
    )
    # spec: (B, F, T) complex
    real = spec.real  # (B, F, T)
    imag = spec.imag  # (B, F, T)
    # Stack to (B, 2, T, F)
    out = torch.stack([real, imag], dim=1)  # (B, 2, F, T)
    out = out.permute(0, 1, 3, 2)  # (B, 2, T, F)
    return out


def istft(complex_spec, n_fft, win_length, hop_length, length=None):
    """Compute inverse STFT from stacked real/imag representation.

    Args:
        complex_spec: (B, 2, T, F) float tensor where dim=1 is [real, imag]
        n_fft: FFT size
        win_length: window length in samples
        hop_length: hop length in samples
        length: desired output length (samples), or None

    Returns:
        (B, L) float tensor
    """
    # (B, 2, T, F) -> (B, F, T) complex
    spec = complex_spec.permute(0, 1, 3, 2)  # (B, 2, F, T)
    spec_complex = torch.complex(spec[:, 0], spec[:, 1])  # (B, F, T)
    window = torch.hann_window(win_length, device=complex_spec.device)
    waveform = torch.istft(
        spec_complex, n_fft, hop_length=hop_length, win_length=win_length,
        window=window, length=length
    )
    return waveform


def apply_rir(waveform, rir):
    """Convolve waveform with room impulse response.

    Args:
        waveform: (L,) or (B, L) float tensor
        rir: (R,) float tensor (single RIR)

    Returns:
        Convolved waveform, same shape as input, trimmed to original length.
    """
    single = waveform.dim() == 1
    if single:
        waveform = waveform.unsqueeze(0)

    # Normalize RIR to unit energy
    rir = rir / (rir.norm() + 1e-8)

    # (B, L) -> (B, 1, L) for conv1d
    x = waveform.unsqueeze(1)
    kernel = rir.flip(0).unsqueeze(0).unsqueeze(0).to(x.device)  # (1, 1, R)
    pad_len = rir.shape[0] - 1
    x_padded = F.pad(x, (pad_len, 0))
    out = F.conv1d(x_padded, kernel)  # (B, 1, L)
    out = out.squeeze(1)  # (B, L)

    # Trim to original length
    out = out[:, :waveform.shape[1]]

    if single:
        out = out.squeeze(0)
    return out


def mix_at_snr(signal, noise, snr_db):
    """Mix signal and noise at a target SNR.

    Args:
        signal: (L,) or (B, L) float tensor (clean signal)
        noise: same shape as signal
        snr_db: float, target SNR in dB

    Returns:
        mixture: signal + scaled noise, same shape
    """
    sig_power = signal.pow(2).mean(dim=-1, keepdim=True).clamp(min=1e-8)
    noise_power = noise.pow(2).mean(dim=-1, keepdim=True).clamp(min=1e-8)
    snr_linear = 10.0 ** (snr_db / 10.0)
    scale = (sig_power / (noise_power * snr_linear + 1e-8)).sqrt()
    return signal + scale * noise


def random_crop(waveform, target_length):
    """Randomly crop or zero-pad waveform to target_length.

    Args:
        waveform: (L,) float tensor
        target_length: int

    Returns:
        (target_length,) float tensor
    """
    length = waveform.shape[0]
    if length >= target_length:
        offset = torch.randint(0, length - target_length + 1, (1,)).item()
        return waveform[offset:offset + target_length]
    else:
        # Zero-pad
        pad_amount = target_length - length
        return F.pad(waveform, (0, pad_amount))


def load_audio(path, sr=48000):
    """Load audio file, force mono, resample if needed.

    Args:
        path: str, path to audio file
        sr: int, target sample rate

    Returns:
        (L,) float tensor
    """
    waveform, orig_sr = torchaudio.load(path)
    # Force mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)  # (L,)
    # Resample if needed
    if orig_sr != sr:
        waveform = torchaudio.functional.resample(waveform, orig_sr, sr)
    return waveform
