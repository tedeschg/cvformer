import math
from typing import Optional

import torch
import torch.nn as nn


class DihedralTransformerAE(nn.Module):
    """
    Dihedral Transformer Autoencoder with:
      - optional attention pooling temperature
      - optional mean pooling
      - optional bounded latent (tanh/softsign) to limit CV excursions / gradients
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
        pool_type: str = "attn",          # "attn" or "mean"
        attn_temperature: float = 1.0,    # >1.0 => softer softmax
        latent_bound: float = 0.0,        # 0.0 disables; >0 enables tanh bounding
    ):
        super().__init__()

        if pool_type not in ("attn", "mean"):
            raise ValueError("pool_type must be 'attn' or 'mean'")
        if attn_temperature <= 0.0:
            raise ValueError("attn_temperature must be > 0")
        if latent_bound < 0.0:
            raise ValueError("latent_bound must be >= 0")

        self.n_tokens = n_tokens
        self.d_model = d_model
        self.latent_dim = latent_dim

        self.pool_type = pool_type
        self.attn_temperature = float(attn_temperature)
        self.latent_bound = float(latent_bound)

        self.input_proj = nn.Linear(4, d_model)
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

        # Attention pooling
        self.attn_pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
        )

        self.to_latent = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.LayerNorm(dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, latent_dim),
        )

        # Decoder init from latent + learned residue queries
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

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def _normalize_sincos_pairs(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Project (sin,cos) onto unit circle for phi and psi separately.
        x: (B, N, 4)
        """
        phi = x[..., 0:2]
        psi = x[..., 2:4]
        phi = phi / (phi.norm(p=2, dim=-1, keepdim=True) + eps)
        psi = psi / (psi.norm(p=2, dim=-1, keepdim=True) + eps)
        return torch.cat([phi, psi], dim=-1)

    def _apply_latent_bounding(self, z: torch.Tensor) -> torch.Tensor:
        """
        If latent_bound > 0: z <- latent_bound * tanh(z / latent_bound).
        This keeps latent in [-latent_bound, +latent_bound] and tends to soften
        gradients in extreme regions.
        """
        if self.latent_bound <= 0.0:
            return z
        lb = self.latent_bound
        return lb * torch.tanh(z / lb)

    def encode(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, N, 4)
        mask: (N,) bool True=valid, False=ignore
        """
        B, N, _ = x.shape

        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.input_norm(h)

        if mask is not None:
            # src_key_padding_mask: (B, N) True=ignore
            src_key_padding_mask = (~mask).expand(B, -1)
            h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)
        else:
            h = self.encoder(h)

        if self.pool_type == "mean":
            if mask is None:
                pooled = h.mean(dim=1)
            else:
                m = mask.to(h.device).float().unsqueeze(0).unsqueeze(-1)  # (1,N,1)
                denom = m.sum(dim=1).clamp(min=1.0)                        # (1,1)
                pooled = (h * m).sum(dim=1) / denom
        else:
            # attention pooling with temperature
            scores = self.attn_pool(h).squeeze(-1)  # (B, N)
            if mask is not None:
                scores = scores.masked_fill((~mask).unsqueeze(0), -1e9)

            # temperature >1 => softer weights, less "sharp" changes
            tau = self.attn_temperature
            w = torch.softmax(scores / tau, dim=1)  # (B, N)
            pooled = (h * w.unsqueeze(-1)).sum(dim=1)  # (B, d_model)

        z = self.to_latent(pooled)
        z = self._apply_latent_bounding(z)
        return z

    def decode(self, z: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        z: (B, latent_dim)
        mask: (N,) bool (used to ignore invalid tokens inside decoder if desired)
        """
        B = z.size(0)

        # Ensure decode sees the same bounded z (symmetry helps)
        z = self._apply_latent_bounding(z)

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
        z = self.encode(x, mask)
        x_hat = self.decode(z, mask)
        return x_hat, z


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding (buffer, no learned params).
    """

    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


def dihedral_loss(x_hat: torch.Tensor, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    MSE on sin/cos, averaged over B*N_valid*4.
    """
    if mask is not None:
        mask = mask.to(x.device)
        m = mask.unsqueeze(0).unsqueeze(-1).float()  # (1, N, 1)
        diff2 = ((x_hat - x) ** 2) * m
        n_valid = mask.sum().clamp(min=1).float()
        denom = x.shape[0] * n_valid * x.shape[2]  # B*N_valid*4
        return diff2.sum() / denom

    return ((x_hat - x) ** 2).mean()


class WarmupCosineScheduler:
    """
    Per-step scheduler.
    Warmup linearly to base_lr, then cosine decay to base_lr*min_lr_ratio.
    """

    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.01):
        self.optimizer = optimizer
        self.warmup_steps = int(max(0, warmup_steps))
        self.total_steps = int(max(1, total_steps))
        self.min_lr_ratio = float(min_lr_ratio)

        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.step_num = 0

    def step(self):
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
