import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


# ==========================================================
# Dihedral Transformer Autoencoder (TorchScript + PLUMED safe)
# - Proper TransformerDecoder (cross-attn to latent-derived memory)
# - Mask handling fixed + TorchScript-safe
# - No torch.finfo / no torch.device args in scripted paths
# - Includes WarmupCosineScheduler (for your training script)
# ==========================================================


# -----------------------------
# Positional Encoding
# -----------------------------
class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding (buffer, no learned parameters).
    """
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


# -----------------------------
# Model
# -----------------------------
class DihedralTransformerAE(nn.Module):
    """
    Autoencoder for dihedral angles represented as:
      (sin_phi, cos_phi, sin_psi, cos_psi)

    Encoder: TransformerEncoder + attention pooling -> latent z
    Decoder: TransformerDecoder cross-attending to latent-derived memory

    TorchScript/PLUMED constraints:
      - avoid torch.finfo
      - avoid passing torch.device in scripted code paths
      - mask handling must be deterministic and typed
    """

    def __init__(
        self,
        n_tokens: int,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        latent_dim: int = 2,
        memory_tokens: int = 4,
    ):
        super().__init__()
        self.n_tokens = int(n_tokens)
        self.d_model = int(d_model)
        self.latent_dim = int(latent_dim)
        self.memory_tokens = int(memory_tokens)

        # ---- Encoder ----
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

        # ---- Decoder (proper TransformerDecoder) ----
        self.residue_queries = nn.Parameter(torch.empty(1, n_tokens, d_model))

        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, dim_feedforward),
            nn.LayerNorm(dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, memory_tokens * d_model),
        )

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 4),
        )

        self._init_parameters()

    def _init_parameters(self):
        # Xavier for most weights; keep residue_queries small
        for name, p in self.named_parameters():
            if p.dim() > 1 and name != "residue_queries":
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.residue_queries, mean=0.0, std=0.02)

    # -----------------------------
    # TorchScript-safe mask handling
    # -----------------------------
    @staticmethod
    def _canonicalize_mask(
        mask: Optional[torch.Tensor],
        B: int,
        N: int,
        ref: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Returns mask (B,N) bool where True=valid and False=padding.
        Accepts None, (N,), or (B,N).
        TorchScript-safe: uses ref tensor to move device.
        """
        if mask is None:
            return None

        mask = mask.to(ref).to(dtype=torch.bool)

        if mask.dim() == 1:
            if mask.numel() != N:
                raise RuntimeError("mask (N,) wrong length")
            return mask.unsqueeze(0).expand(B, -1)

        if mask.dim() == 2:
            if mask.size(0) != B or mask.size(1) != N:
                raise RuntimeError("mask (B,N) wrong shape")
            return mask

        raise RuntimeError("mask must be None, (N,), or (B,N)")

    @staticmethod
    def _normalize_sincos_pairs(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        phi = x[..., 0:2]
        psi = x[..., 2:4]
        phi = phi / (phi.norm(p=2, dim=-1, keepdim=True) + eps)
        psi = psi / (psi.norm(p=2, dim=-1, keepdim=True) + eps)
        return torch.cat([phi, psi], dim=-1)

    # -----------------------------
    # Encode
    # -----------------------------
    def _encode_core(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = x.shape
        mask_bn = self._canonicalize_mask(mask, B, N, x)

        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.input_norm(h)

        if mask_bn is not None:
            h = self.encoder(h, src_key_padding_mask=~mask_bn)  # True=ignore
        else:
            h = self.encoder(h)

        scores = self.attn_pool(h).squeeze(-1)  # (B, N)

        if mask_bn is not None:
            # TorchScript-safe constant (works in fp16/bf16/fp32)
            scores = scores.masked_fill(~mask_bn, -1.0e4)

        w = torch.softmax(scores, dim=1)         # (B, N)
        pooled = (h * w.unsqueeze(-1)).sum(1)    # (B, d_model)
        z = self.to_latent(pooled)               # (B, latent_dim)
        return z, w

    def encode(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        z, _ = self._encode_core(x, mask)
        return z

    @torch.jit.ignore
    def encode_with_attention(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        return self._encode_core(x, mask)

    # -----------------------------
    # Decode
    # -----------------------------
    def decode(self, z: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = z.size(0)
        N = self.n_tokens
        mask_bn = self._canonicalize_mask(mask, B, N, z)

        mem = self.from_latent(z).view(B, self.memory_tokens, self.d_model)

        tgt = self.residue_queries.expand(B, -1, -1)
        tgt = self.pos_enc(tgt)

        tgt_key_padding_mask = (~mask_bn) if mask_bn is not None else None

        h = self.decoder(
            tgt=tgt,
            memory=mem,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=None,
        )

        x_hat = self.output_proj(h)
        x_hat = self._normalize_sincos_pairs(x_hat)
        return x_hat

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        z = self.encode(x, mask)
        x_hat = self.decode(z, mask)
        return x_hat, z


# -----------------------------
# Loss
# -----------------------------
def dihedral_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    MSE on sin/cos, averaged over valid tokens and channels.
    mask: None, (N,), or (B,N) with True=valid.
    """
    if mask is None:
        return ((x_hat - x) ** 2).mean()

    mask = mask.to(x).to(dtype=torch.bool)
    B, N, C = x.shape

    if mask.dim() == 1:
        if mask.numel() != N:
            raise RuntimeError("mask (N,) wrong length")
        m = mask.unsqueeze(0).unsqueeze(-1).float()  # (1, N, 1)
        diff2 = ((x_hat - x) ** 2) * m
        n_valid = mask.sum().clamp(min=1).float()
        denom = float(B) * n_valid * float(C)
        return diff2.sum() / denom

    if mask.dim() == 2:
        if mask.size(0) != B or mask.size(1) != N:
            raise RuntimeError("mask (B,N) wrong shape")
        m = mask.unsqueeze(-1).float()  # (B, N, 1)
        diff2 = ((x_hat - x) ** 2) * m
        n_valid_total = mask.sum().clamp(min=1).float()
        denom = n_valid_total * float(C)
        return diff2.sum() / denom

    raise RuntimeError("mask must be None, (N,), or (B,N)")


# -----------------------------
# Scheduler: warmup + cosine
# -----------------------------
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
        return [pg["lr"] for pg in self.optimizer.param_groups]