"""Band-Split Encoder and Band-Merge Decoder for 48kHz processing.

Splits 1025 frequency bins into 53 overlapping psychoacoustic sub-bands,
projects each to a shared feature dimension, then merges back.
"""

import torch
import torch.nn as nn

from data.audio_utils import istft


class BandSplitEncoder(nn.Module):
    """Split complex spectrogram into sub-bands and project to feature space.

    Input: (B, 2, T, F) where F=1025, dim=1 is [real, imag]
    Output: (B, T, num_bands, feature_dim)
    """

    def __init__(self, bands, feature_dim):
        """
        Args:
            bands: list of [start_bin, end_bin] pairs (inclusive)
            feature_dim: output feature dimension per band
        """
        super().__init__()
        self.bands = bands
        self.num_bands = len(bands)
        self.feature_dim = feature_dim

        self.norms = nn.ModuleList()
        self.projections = nn.ModuleList()
        for start, end in bands:
            band_width = end - start + 1
            input_dim = band_width * 2  # real + imag concatenated
            self.norms.append(nn.LayerNorm(input_dim))
            self.projections.append(nn.Linear(input_dim, feature_dim))

    def forward(self, x):
        """
        Args:
            x: (B, 2, T, F) complex spectrogram

        Returns:
            (B, T, num_bands, feature_dim)
        """
        B, _, T, F = x.shape
        band_features = []
        for i, (start, end) in enumerate(self.bands):
            # Extract sub-band: (B, 2, T, band_width)
            band = x[:, :, :, start:end + 1]
            # Reshape to (B, T, band_width * 2) by concatenating real and imag
            band = band.permute(0, 2, 3, 1).contiguous()  # (B, T, band_width, 2)
            band = band.reshape(B, T, -1)  # (B, T, band_width * 2)
            # Normalize and project
            band = self.norms[i](band)
            band = self.projections[i](band)  # (B, T, feature_dim)
            band_features.append(band)

        # Stack bands: (B, T, num_bands, feature_dim)
        return torch.stack(band_features, dim=2)


class BandMergeDecoder(nn.Module):
    """Project sub-band features back to frequency bins and reconstruct waveform.

    Produces a complex mask that is applied to the mixture spectrogram,
    then converts to waveform via iSTFT.
    """

    def __init__(self, bands, feature_dim, n_fft, win_length, hop_length, freq_bins=1025):
        """
        Args:
            bands: list of [start_bin, end_bin] pairs (inclusive), same as encoder
            feature_dim: input feature dimension per band
            n_fft: FFT size for iSTFT
            win_length: window length for iSTFT
            hop_length: hop length for iSTFT
            freq_bins: total number of frequency bins
        """
        super().__init__()
        self.bands = bands
        self.freq_bins = freq_bins
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        self.projections = nn.ModuleList()
        for start, end in bands:
            band_width = end - start + 1
            output_dim = band_width * 2  # real + imag
            proj = nn.Linear(feature_dim, output_dim)
            # Initialize mask near identity (1+0i) so model starts from
            # pass-through instead of full suppression.  The output is
            # reshaped to (B, T, band_width, 2) with interleaved [real, imag],
            # so even indices are real and odd indices are imaginary.
            nn.init.normal_(proj.weight, std=1e-4)
            with torch.no_grad():
                proj.bias.zero_()
                proj.bias[0::2] = 1.0   # real part = 1 (pass-through)
                # imag part (odd indices) stays 0
            self.projections.append(proj)

    def forward(self, z, mix_spec, length=None):
        """
        Args:
            z: (B, T, num_bands, feature_dim) processed features
            mix_spec: (B, 2, T, F) original mixture spectrogram
            length: desired output waveform length, or None

        Returns:
            (B, L) estimated waveform
        """
        B, T, num_bands, D = z.shape
        F = self.freq_bins

        # Accumulate mask with overlap-add for overlapping bands
        mask = torch.zeros(B, T, F, 2, device=z.device)
        weight = torch.zeros(1, 1, F, 1, device=z.device)

        for i, (start, end) in enumerate(self.bands):
            band_width = end - start + 1
            band_feat = z[:, :, i, :]  # (B, T, D)
            band_out = self.projections[i](band_feat)  # (B, T, band_width * 2)
            band_out = band_out.reshape(B, T, band_width, 2)
            mask[:, :, start:end + 1, :] += band_out
            weight[:, :, start:end + 1, :] += 1.0

        # Average overlapping regions
        mask = mask / weight.clamp(min=1.0)

        # mask: (B, T, F, 2) -> (B, 2, T, F)
        mask = mask.permute(0, 3, 1, 2)

        # Apply complex ratio mask (proper complex multiplication)
        mask_r, mask_i = mask[:, 0], mask[:, 1]
        mix_r, mix_i = mix_spec[:, 0], mix_spec[:, 1]
        out_r = mask_r * mix_r - mask_i * mix_i
        out_i = mask_r * mix_i + mask_i * mix_r
        masked_spec = torch.stack([out_r, out_i], dim=1)

        # iSTFT to waveform
        return istft(masked_spec, self.n_fft, self.win_length, self.hop_length, length=length)
