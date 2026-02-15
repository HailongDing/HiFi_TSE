"""Full generator assembly: Band-Split Encoder → USEF → TF-GridNet → Band-Merge Decoder."""

import torch
import torch.nn as nn

from data.audio_utils import stft
from models.band_split import BandSplitEncoder, BandMergeDecoder
from models.usef import FiLMLayer, USEFModule
from models.tf_gridnet import TFGridNet


class Generator(nn.Module):
    """HiFi-TSE Generator.

    Pipeline:
        mix_wav → STFT → BandSplitEncoder → Z_mix ─┐
                                                     ├→ USEF → TFGridNet → BandMergeDecoder → est_wav
        ref_wav → STFT → BandSplitEncoder → Z_ref ─┘
                         (shared weights)
    """

    def __init__(self, cfg):
        """
        Args:
            cfg: dict with keys 'stft', 'band_split', 'model'
        """
        super().__init__()
        stft_cfg = cfg['stft']
        self.n_fft = stft_cfg['n_fft']
        self.win_length = stft_cfg['win_length']
        self.hop_length = stft_cfg['hop_length']

        bands = [tuple(b) for b in cfg['band_split']['bands']]
        feature_dim = cfg['model']['feature_dim']
        num_heads = cfg['model']['num_heads']
        lstm_hidden = cfg['model']['lstm_hidden']
        num_blocks = cfg['model']['num_gridnet_blocks']

        # Shared encoder for mix and ref
        self.encoder = BandSplitEncoder(bands, feature_dim)

        # USEF cross-attention conditioning
        self.usef = USEFModule(feature_dim, num_heads)

        # FiLM conditioning (v2)
        self.film = FiLMLayer(feature_dim)

        # Conditioning dropout probability (v2)
        self.ref_dropout_prob = cfg['model'].get('ref_dropout_prob', 0.0)

        # TF-GridNet backbone
        reinject_at = cfg['model'].get('reinject_at', [])
        use_checkpoint = cfg['model'].get('use_checkpoint', False)
        self.gridnet = TFGridNet(feature_dim, lstm_hidden, num_heads, num_blocks,
                                  reinject_at=reinject_at, use_checkpoint=use_checkpoint)

        # Band-merge decoder
        self.decoder = BandMergeDecoder(
            bands, feature_dim,
            self.n_fft, self.win_length, self.hop_length,
        )

    def forward(self, mix_wav, ref_wav):
        """
        Args:
            mix_wav: (B, L) mixture waveform at 48kHz
            ref_wav: (B, L') reference waveform at 48kHz (may differ in length)

        Returns:
            (B, L) estimated target waveform
        """
        length = mix_wav.shape[1]

        # STFT
        mix_spec = stft(mix_wav, self.n_fft, self.win_length, self.hop_length)
        ref_spec = stft(ref_wav, self.n_fft, self.win_length, self.hop_length)

        # Band-split encode (shared weights)
        z_mix = self.encoder(mix_spec)  # (B, T_mix, 53, D)
        z_ref = self.encoder(ref_spec)  # (B, T_ref, 53, D)

        # Conditioning dropout (v2): zero out z_ref for random samples during training.
        # This disables ALL conditioning paths (USEF + reinject + FiLM) simultaneously,
        # forcing the model to learn some separation without speaker reference.
        if self.training and self.ref_dropout_prob > 0.0:
            drop_mask = (torch.rand(z_ref.size(0), 1, 1, 1, device=z_ref.device)
                         < self.ref_dropout_prob)
            z_ref = z_ref.masked_fill(drop_mask, 0.0)

        # USEF cross-attention conditioning
        z_cond = self.usef(z_mix, z_ref)  # (B, T_mix, 53, D)

        # FiLM modulation (v2)
        z_cond = self.film(z_cond, z_ref)

        # TF-GridNet backbone
        z_out = self.gridnet(z_cond, z_ref=z_ref)  # (B, T_mix, 53, D)

        # Band-merge decode to waveform
        est_wav = self.decoder(z_out, mix_spec, length=length)

        return est_wav
