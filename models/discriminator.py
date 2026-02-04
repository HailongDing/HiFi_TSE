"""HiFi-GAN style discriminators: Multi-Period Discriminator + Multi-Scale Discriminator."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class PeriodDiscriminator(nn.Module):
    """Single period sub-discriminator.

    Reshapes 1D waveform into 2D by folding at a given period,
    then applies 2D convolutions to detect periodic artifacts.
    """

    def __init__(self, period):
        super().__init__()
        self.period = period

        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 32, (5, 1), (3, 1), (2, 0))),
            weight_norm(nn.Conv2d(32, 128, (5, 1), (3, 1), (2, 0))),
            weight_norm(nn.Conv2d(128, 512, (5, 1), (3, 1), (2, 0))),
            weight_norm(nn.Conv2d(512, 1024, (5, 1), (3, 1), (2, 0))),
            weight_norm(nn.Conv2d(1024, 1024, (5, 1), (1, 1), (2, 0))),
        ])
        self.output_conv = weight_norm(nn.Conv2d(1024, 1, (3, 1), (1, 1), (1, 0)))

    def forward(self, x):
        """
        Args:
            x: (B, 1, L) waveform

        Returns:
            (output, feature_maps): output is final prediction, feature_maps is list of intermediates
        """
        features = []
        B, C, L = x.shape

        # Pad to multiple of period
        if L % self.period != 0:
            pad_len = self.period - (L % self.period)
            x = F.pad(x, (0, pad_len), mode='reflect')
            L = x.shape[2]

        # Reshape to 2D: (B, 1, L//p, p)
        x = x.view(B, C, L // self.period, self.period)

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            features.append(x)

        x = self.output_conv(x)
        features.append(x)
        x = x.flatten(1, -1)

        return x, features


class MultiPeriodDiscriminator(nn.Module):
    """Multi-Period Discriminator with periods [2, 3, 5, 7, 11]."""

    def __init__(self, periods=None):
        super().__init__()
        if periods is None:
            periods = [2, 3, 5, 7, 11]
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(p) for p in periods
        ])

    def forward(self, x):
        """
        Args:
            x: (B, 1, L) waveform

        Returns:
            (outputs_list, features_list)
        """
        outputs = []
        features = []
        for disc in self.discriminators:
            out, feat = disc(x)
            outputs.append(out)
            features.append(feat)
        return outputs, features


class ScaleDiscriminator(nn.Module):
    """Single scale sub-discriminator using 1D convolutions."""

    def __init__(self, use_spectral_norm=False):
        super().__init__()
        norm_fn = nn.utils.spectral_norm if use_spectral_norm else weight_norm

        self.convs = nn.ModuleList([
            norm_fn(nn.Conv1d(1, 128, 15, 1, 7)),
            norm_fn(nn.Conv1d(128, 128, 41, 2, 20, groups=4)),
            norm_fn(nn.Conv1d(128, 256, 41, 2, 20, groups=16)),
            norm_fn(nn.Conv1d(256, 512, 41, 4, 20, groups=16)),
            norm_fn(nn.Conv1d(512, 1024, 41, 4, 20, groups=16)),
            norm_fn(nn.Conv1d(1024, 1024, 41, 1, 20, groups=16)),
            norm_fn(nn.Conv1d(1024, 1024, 5, 1, 2)),
        ])
        self.output_conv = norm_fn(nn.Conv1d(1024, 1, 3, 1, 1))

    def forward(self, x):
        """
        Args:
            x: (B, 1, L) waveform

        Returns:
            (output, feature_maps)
        """
        features = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            features.append(x)

        x = self.output_conv(x)
        features.append(x)
        x = x.flatten(1, -1)

        return x, features


class MultiScaleDiscriminator(nn.Module):
    """Multi-Scale Discriminator operating at 3 scales: raw, x2 downsample, x4 downsample."""

    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(use_spectral_norm=True),
            ScaleDiscriminator(),
            ScaleDiscriminator(),
        ])
        self.downsample = nn.AvgPool1d(kernel_size=4, stride=2, padding=2)

    def forward(self, x):
        """
        Args:
            x: (B, 1, L) waveform

        Returns:
            (outputs_list, features_list)
        """
        outputs = []
        features = []
        for i, disc in enumerate(self.discriminators):
            if i > 0:
                x = self.downsample(x)
            out, feat = disc(x)
            outputs.append(out)
            features.append(feat)
        return outputs, features


class Discriminator(nn.Module):
    """Combined MPD + MSD discriminator."""

    def __init__(self, cfg):
        """
        Args:
            cfg: dict with key 'discriminator' containing 'mpd_periods'
        """
        super().__init__()
        periods = cfg['discriminator']['mpd_periods']
        self.mpd = MultiPeriodDiscriminator(periods)
        self.msd = MultiScaleDiscriminator()

    def forward(self, x):
        """
        Args:
            x: (B, L) or (B, 1, L) waveform

        Returns:
            (all_outputs, all_features): concatenated from MPD and MSD
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, L)

        mpd_out, mpd_feat = self.mpd(x)
        msd_out, msd_feat = self.msd(x)

        return mpd_out + msd_out, mpd_feat + msd_feat
