import torch
import torch.nn as nn
import math


class PatchEmbed(nn.Module):
    """
    Splits a 2D feature map (e.g., [B, C, H, W]) into patches
    and projects each patch into a vector of dimension embed_dim.
    """

    def __init__(self, in_chans=1, embed_dim=64, patch_size=8):
        super(PatchEmbed, self).__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x shape: [B, in_chans, H, W]
        # After conv: [B, embed_dim, H/patch_size, W/patch_size]
        x = self.proj(x)
        # Flatten spatial dims, then transpose so shape = [B, num_patches, embed_dim]
        x = x.flatten(2).transpose(1, 2)
        return x


class TransformerEncoderBlock(nn.Module):
    """
    A single Transformer block: Multi-Head Self-Attention + Feed-Forward.
    """

    def __init__(self, embed_dim=64, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x shape: [B, N, embed_dim]

        # 1) Multi-Head Self-Attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)  # (B, N, embed_dim)
        x = x + attn_out

        # 2) Feed-forward
        x_norm = self.norm2(x)
        feedforward_out = self.mlp(x_norm)
        x = x + feedforward_out

        return x


class SimpleVisionTransformer(nn.Module):
    """
    A minimal Vision Transformer with:
      - Patch Embedding
      - A sequence of Transformer Encoder blocks
    """

    def __init__(self, in_chans=1, embed_dim=64, patch_size=8, depth=2, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super(SimpleVisionTransformer, self).__init__()

        self.patch_embed = PatchEmbed(in_chans, embed_dim, patch_size)

        # Create multiple Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # Optional: a final LayerNorm
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        x shape: [B, in_chans, H, W]
        Returns a [B, embed_dim] after global pooling the patch embeddings.
        """
        # 1) Patch embedding
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]

        # 2) Forward through each Transformer Encoder block
        for block in self.blocks:
            x = block(x)

        # 3) Global average pooling over patch dimension
        x = self.norm(x)  # [B, num_patches, embed_dim]
        x = x.mean(dim=1)  # [B, embed_dim]

        return x
