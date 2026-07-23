from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class VARCConfig:
    num_colors: int = 12
    canvas_size: int = 64
    patch_size: int = 2
    pixel_embed_dim: int = 128
    d_model: int = 512
    n_layers: int = 10
    n_heads: int = 8
    dim_feedforward: int = 512
    dropout: float = 0.1
    num_tasks: int = 400
    max_train_pairs: int = 4


class PixelEmbed(nn.Module):
    def __init__(self, config: VARCConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.pixel_embed = nn.Embedding(
            config.num_colors, config.pixel_embed_dim
        )
        patch_dim = config.patch_size * config.patch_size * config.pixel_embed_dim
        self.proj = nn.Linear(patch_dim, config.d_model)
        num_patches = config.canvas_size // config.patch_size
        self.num_patches = num_patches
        pos_h = nn.Parameter(torch.randn(1, num_patches, config.d_model // 2) * 0.02)
        pos_w = nn.Parameter(torch.randn(1, num_patches, config.d_model // 2) * 0.02)
        self.pos_embed_h = pos_h
        self.pos_embed_w = pos_w
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W = x.shape
        ps = self.patch_size
        x = self.pixel_embed(x)
        x = x.reshape(B, H // ps, ps, W // ps, ps, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(B, H // ps, W // ps, -1)
        x = self.proj(x)
        np = self.num_patches
        pos = torch.cat([
            self.pos_embed_h.unsqueeze(2).expand(-1, -1, np, -1),
            self.pos_embed_w.unsqueeze(1).expand(-1, np, -1, -1),
        ], dim=-1)
        x = x + pos
        x = x.flatten(1, 2)
        return self.drop(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: VARCConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attn = nn.MultiheadAttention(
            config.d_model, config.n_heads, dropout=config.dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.dim_feedforward),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x


class VARCModel(nn.Module):
    def __init__(self, config: VARCConfig):
        super().__init__()
        self.config = config
        self.pixel_embed = PixelEmbed(config)
        self.task_embed = nn.Embedding(config.num_tasks + 1, config.d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(
            config.d_model,
            config.patch_size * config.patch_size * config.num_colors,
        )
        self.patch_size = config.patch_size
        self.canvas_size = config.canvas_size
        self.pad_token_id = 10

    def forward(
        self,
        x: torch.Tensor,
        task_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = x.shape[0]
        x = self.pixel_embed(x)
        if task_ids is not None:
            task_tok = self.task_embed(task_ids).unsqueeze(1)
            x = x + task_tok
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.head(x)
        ps = self.patch_size
        cs = self.canvas_size
        nc = self.config.num_colors
        h = w = cs // ps
        x = x.view(B, h, w, ps, ps, nc)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, nc, cs, cs)
        return x
