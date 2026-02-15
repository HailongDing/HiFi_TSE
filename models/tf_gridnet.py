"""TF-GridNet backbone: 6 GridNet blocks with intra-frame BiLSTM,
sub-band BiLSTM, and self-attention on the time-frequency grid.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint


class GridNetBlock(nn.Module):
    """Single GridNet block with three processing stages:

    1. Intra-frame BiLSTM (frequency axis across sub-bands)
    2. Sub-band BiLSTM (time axis within each sub-band)
    3. Self-attention (temporal, per sub-band)
    """

    def __init__(self, feature_dim, lstm_hidden, num_heads):
        """
        Args:
            feature_dim: feature dimension (D=128)
            lstm_hidden: LSTM hidden size per direction (192)
            num_heads: number of self-attention heads (4)
        """
        super().__init__()

        # 1. Intra-frame BiLSTM (frequency axis)
        self.freq_lstm = nn.LSTM(
            feature_dim, lstm_hidden, bidirectional=True, batch_first=True,
        )
        self.freq_proj = nn.Linear(lstm_hidden * 2, feature_dim)
        self.freq_norm = nn.LayerNorm(feature_dim)

        # 2. Sub-band BiLSTM (time axis)
        self.time_lstm = nn.LSTM(
            feature_dim, lstm_hidden, bidirectional=True, batch_first=True,
        )
        self.time_proj = nn.Linear(lstm_hidden * 2, feature_dim)
        self.time_norm = nn.LayerNorm(feature_dim)

        # 3. Self-attention (time axis)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads, batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(feature_dim)

    def forward(self, x):
        """
        Args:
            x: (B, T, N, D) where N=num_bands, D=feature_dim

        Returns:
            (B, T, N, D)
        """
        B, T, N, D = x.shape

        # --- 1. Intra-frame BiLSTM (frequency axis) ---
        # Reshape: (B*T, N, D)
        h = x.reshape(B * T, N, D)
        h, _ = self.freq_lstm(h)           # (B*T, N, 2*lstm_hidden)
        h = self.freq_proj(h)              # (B*T, N, D)
        h = h.reshape(B, T, N, D)
        x = self.freq_norm(x + h)          # residual + norm

        # --- 2. Sub-band BiLSTM (time axis) ---
        # Reshape: (B*N, T, D)
        h = x.permute(0, 2, 1, 3).reshape(B * N, T, D)
        h, _ = self.time_lstm(h)           # (B*N, T, 2*lstm_hidden)
        h = self.time_proj(h)              # (B*N, T, D)
        h = h.reshape(B, N, T, D).permute(0, 2, 1, 3)  # (B, T, N, D)
        x = self.time_norm(x + h)          # residual + norm

        # --- 3. Self-attention (time axis, per sub-band) ---
        # Reshape: (B*N, T, D)
        h = x.permute(0, 2, 1, 3).reshape(B * N, T, D)
        attn_out, _ = self.self_attn(h, h, h)  # (B*N, T, D)
        attn_out = attn_out.reshape(B, N, T, D).permute(0, 2, 1, 3)
        x = self.attn_norm(x + attn_out)   # residual + norm

        return x


class SpeakerReinjectLayer(nn.Module):
    """Lightweight cross-attention for re-injecting speaker info."""

    def __init__(self, feature_dim, num_heads):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads, batch_first=True,
        )
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x, z_ref):
        """
        Args:
            x: (B, T, N, D)
            z_ref: (B, T_ref, N, D)
        Returns:
            (B, T, N, D)
        """
        B, T, N, D = x.shape
        T_ref = z_ref.shape[1]
        q = x.permute(0, 2, 1, 3).reshape(B * N, T, D)
        kv = z_ref.permute(0, 2, 1, 3).reshape(B * N, T_ref, D)
        attn_out, _ = self.cross_attn(q, kv, kv)
        out = self.norm(q + attn_out)
        return out.reshape(B, N, T, D).permute(0, 2, 1, 3)


class TFGridNet(nn.Module):
    """Stack of GridNet blocks forming the TF-GridNet backbone."""

    def __init__(self, feature_dim, lstm_hidden, num_heads, num_blocks,
                 reinject_at=None, use_checkpoint=False):
        """
        Args:
            feature_dim: feature dimension (128)
            lstm_hidden: LSTM hidden size per direction (256)
            num_heads: attention heads (4)
            num_blocks: number of GridNet blocks (6)
            reinject_at: list of block indices for speaker re-injection
            use_checkpoint: enable gradient checkpointing to save memory
        """
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            GridNetBlock(feature_dim, lstm_hidden, num_heads)
            for _ in range(num_blocks)
        ])
        self.reinject_at = reinject_at or []
        self.reinject_layers = nn.ModuleDict({
            str(i): SpeakerReinjectLayer(feature_dim, num_heads)
            for i in self.reinject_at
        })

    def forward(self, x, z_ref=None):
        """
        Args:
            x: (B, T, N, D)
            z_ref: (B, T_ref, N, D) optional speaker reference embedding

        Returns:
            (B, T, N, D)
        """
        for i, block in enumerate(self.blocks):
            if z_ref is not None and i in self.reinject_at:
                x = self.reinject_layers[str(i)](x, z_ref)
            if self.use_checkpoint and self.training:
                x = grad_checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x
