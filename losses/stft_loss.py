"""Multi-resolution STFT loss for spectral fidelity at 48kHz."""

import torch
import torch.nn as nn


class SingleResolutionSTFTLoss(nn.Module):
    """STFT loss at a single resolution: spectral convergence + log magnitude L1."""

    def __init__(self, n_fft, win_length, hop_length):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.register_buffer('window', torch.hann_window(win_length))

    def forward(self, estimate, target):
        """
        Args:
            estimate: (B, L) waveform
            target: (B, L) waveform

        Returns:
            (spectral_convergence, log_magnitude_l1) tuple of scalars
        """
        window = self.window

        est_spec = torch.stft(
            estimate, self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=window, return_complex=True
        )
        tgt_spec = torch.stft(
            target, self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=window, return_complex=True
        )

        est_mag = est_spec.abs()
        tgt_mag = tgt_spec.abs()

        # Spectral convergence: ||mag_est - mag_tgt||_F / ||mag_tgt||_F
        sc = torch.norm(tgt_mag - est_mag, p="fro") / (torch.norm(tgt_mag, p="fro") + 1e-8)

        # Log magnitude L1
        eps = 1e-8
        log_mag_l1 = nn.functional.l1_loss(
            torch.log(est_mag + eps), torch.log(tgt_mag + eps)
        )

        return sc, log_mag_l1


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT loss across 3 scales.

    Scales tuned for 48kHz:
      - (512, 240, 50): transients, plosives
      - (1024, 600, 120): mid-frequency voice
      - (2048, 1200, 240): high-frequency harmonics
    """

    def __init__(self):
        super().__init__()
        self.losses = nn.ModuleList([
            SingleResolutionSTFTLoss(512, 240, 50),
            SingleResolutionSTFTLoss(1024, 600, 120),
            SingleResolutionSTFTLoss(2048, 1200, 240),
        ])

    def forward(self, estimate, target):
        """
        Args:
            estimate: (B, L) waveform
            target: (B, L) waveform

        Returns:
            scalar loss (mean of all 6 terms: 3 scales x 2 loss types)
        """
        total = torch.tensor(0.0, device=estimate.device)
        for loss_fn in self.losses:
            sc, log_l1 = loss_fn(estimate, target)
            total = total + sc + log_l1
        return total / (len(self.losses) * 2)


class PhaseSensitiveLoss(nn.Module):
    """Complex-domain STFT loss that penalizes both magnitude and phase errors.

    Computes L1 loss on real and imaginary parts of the STFT, which implicitly
    penalizes phase errors (unlike magnitude-only STFT loss). Uses the same
    3-scale configuration as MultiResolutionSTFTLoss.
    """

    def __init__(self):
        super().__init__()
        self.configs = [
            (512, 240, 50),
            (1024, 600, 120),
            (2048, 1200, 240),
        ]
        for n_fft, win_length, hop_length in self.configs:
            self.register_buffer(
                "window_{}".format(n_fft),
                torch.hann_window(win_length),
            )

    def forward(self, estimate, target):
        """
        Args:
            estimate: (B, L) waveform
            target: (B, L) waveform

        Returns:
            scalar loss (mean complex L1 across 3 scales)
        """
        total = torch.tensor(0.0, device=estimate.device)
        for n_fft, win_length, hop_length in self.configs:
            window = getattr(self, "window_{}".format(n_fft))
            est_spec = torch.stft(
                estimate, n_fft, hop_length=hop_length,
                win_length=win_length, window=window, return_complex=True,
            )
            tgt_spec = torch.stft(
                target, n_fft, hop_length=hop_length,
                win_length=win_length, window=window, return_complex=True,
            )
            # L1 on real and imaginary parts
            total = total + nn.functional.l1_loss(est_spec.real, tgt_spec.real)
            total = total + nn.functional.l1_loss(est_spec.imag, tgt_spec.imag)
        return total / (len(self.configs) * 2)
