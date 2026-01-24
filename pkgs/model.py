import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm, remove_spectral_norm


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (non-learnable)."""

    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)

        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


def dihedral_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    MSE over sin/cos features.
    x_hat, x: (B, N, 4)
    mask: (N,) bool (True=valid)
    """
    if mask is not None:
        mask = mask.to(x.device)
        m = mask.unsqueeze(0).unsqueeze(-1).float()  # (1, N, 1)
        diff2 = ((x_hat - x) ** 2) * m
        n_valid = mask.sum().clamp(min=1).float()
        denom = x.shape[0] * n_valid * x.shape[2]  # B * N_valid * 4
        return diff2.sum() / denom
    return ((x_hat - x) ** 2).mean()


class WarmupCosineScheduler:
    """Per-step LR scheduler with linear warmup + cosine decay."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.01):
        self.optimizer = optimizer
        self.warmup_steps = int(max(0, warmup_steps))
        self.total_steps = int(max(1, total_steps))
        self.min_lr_ratio = float(min_lr_ratio)

        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.step_num = 0

    def step(self) -> None:
        self.step_num += 1
        denom = max(1, self.total_steps - self.warmup_steps)

        for i, pg in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[i]

            if self.warmup_steps > 0 and self.step_num <= self.warmup_steps:
                lr = base_lr * (self.step_num / self.warmup_steps)
            else:
                progress = (self.step_num - self.warmup_steps) / denom
                progress = float(min(max(progress, 0.0), 1.0))
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

                lr_min = base_lr * self.min_lr_ratio
                lr = lr_min + (base_lr - lr_min) * cosine

            pg["lr"] = lr

    def get_last_lr(self):
        return [self.optimizer.param_groups[0]["lr"]]


class DihedralTransformerAE(nn.Module):
    """
    Transformer Autoencoder for dihedral features:
      x[..., 0:2] = (sin(phi), cos(phi))
      x[..., 2:4] = (sin(psi), cos(psi))

    TorchScript-safe:
      - encode() ALWAYS returns Tensor (no Union)
      - remove_spectral_norm_() removes SN hooks before scripting
    """

    def __init__(
        self,
        n_tokens: int,
        d_model: int = 64,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        latent_dim: int = 2,
        use_spectral_norm: bool = True,
        pool_temp: float = 2.0,
    ):
        super().__init__()
        self.n_tokens = int(n_tokens)
        self.d_model = int(d_model)
        self.latent_dim = int(latent_dim)
        self.pool_temp = float(pool_temp)
        self.use_spectral_norm = bool(use_spectral_norm)

        sn = spectral_norm if self.use_spectral_norm else (lambda m: m)

        self.input_proj = sn(nn.Linear(4, d_model))
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=n_tokens)
        self.input_norm = nn.LayerNorm(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        self.attn_pool = nn.Sequential(
            sn(nn.Linear(d_model, d_model)),
            nn.Tanh(),
            sn(nn.Linear(d_model, 1)),
        )

        self.to_latent = nn.Sequential(
            sn(nn.Linear(d_model, dim_feedforward)),
            nn.LayerNorm(dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            sn(nn.Linear(dim_feedforward, latent_dim)),
        )

        self.residue_queries = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)

        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, dim_feedforward),
            nn.LayerNorm(dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        dec_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=num_decoder_layers)

        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 4),
        )

        self._init_parameters()

    def _init_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def _normalize_sincos_pairs(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        phi = x[..., 0:2]
        psi = x[..., 2:4]
        phi = phi / (phi.norm(p=2, dim=-1, keepdim=True) + eps)
        psi = psi / (psi.norm(p=2, dim=-1, keepdim=True) + eps)
        return torch.cat([phi, psi], dim=-1)

    def remove_spectral_norm_(self) -> None:
        """Remove spectral_norm hooks to make TorchScript export safe."""
        for m in self.modules():
            try:
                remove_spectral_norm(m)
            except Exception:
                pass

    def encode(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Returns ONLY z: (B, latent_dim). TorchScript-safe.
        """
        B, N, _ = x.shape

        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.input_norm(h)

        if mask is not None:
            src_key_padding_mask = (~mask).expand(B, -1)  # True=ignore
            h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)
        else:
            h = self.encoder(h)

        scores = self.attn_pool(h).squeeze(-1)  # (B, N)
        if mask is not None:
            scores = scores.masked_fill((~mask).unsqueeze(0), -1e9)

        w = torch.softmax(scores / self.pool_temp, dim=1)  # (B, N)
        pooled = (h * w.unsqueeze(-1)).sum(dim=1)          # (B, d_model)

        z = self.to_latent(pooled)                         # (B, latent_dim)
        return z

    def encode_with_pool(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (z, w, pooled). Not needed for PLUMED export.
        """
        B, N, _ = x.shape

        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.input_norm(h)

        if mask is not None:
            src_key_padding_mask = (~mask).expand(B, -1)
            h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)
        else:
            h = self.encoder(h)

        scores = self.attn_pool(h).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill((~mask).unsqueeze(0), -1e9)

        w = torch.softmax(scores / self.pool_temp, dim=1)
        pooled = (h * w.unsqueeze(-1)).sum(dim=1)
        z = self.to_latent(pooled)
        return z, w, pooled

    def decode(self, z: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = z.size(0)
        context = self.from_latent(z).unsqueeze(1)  # (B, 1, d_model)

        queries = self.residue_queries.expand(B, -1, -1)  # (B, N, d_model)
        h = queries + context
        h = self.pos_enc(h)

        if mask is not None:
            src_key_padding_mask = (~mask).unsqueeze(0).expand(B, -1)
            h = self.decoder(h, src_key_padding_mask=src_key_padding_mask)
        else:
            h = self.decoder(h)

        x_hat = self.output_proj(h)
        x_hat = self._normalize_sincos_pairs(x_hat)
        return x_hat

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        z = self.encode(x, mask)       # z is ALWAYS Tensor now
        x_hat = self.decode(z, mask)
        return x_hat, z