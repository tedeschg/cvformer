import torch
import torch.nn as nn
import math

# -----------------------------
# Latent prior sampling
# -----------------------------
def sample_prior(
    batch_size: int,
    latent_dim: int,
    kind: str = "gaussian",
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    kind:
      - "gaussian": N(0,1)
      - "uniform":  U(-1,1)
    """
    if kind == "gaussian":
        return torch.randn(batch_size, latent_dim, device=device)
    elif kind == "uniform":
        return 2.0 * torch.rand(batch_size, latent_dim, device=device) - 1.0
    raise ValueError(f"Unknown prior kind: {kind}")


# -----------------------------
# Critic on latent space (WGAN-GP)
# -----------------------------
class LatentCritic(nn.Module):
    """
    Simple MLP critic over z.
    Outputs scores (no sigmoid).
    """
    def __init__(self, latent_dim: int, hidden: int = 128, depth: int = 3, dropout: float = 0.1, use_layer_norm: bool = True):
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be >= 2")

        layers: list[nn.Module] = []
        d_in = latent_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(d_in, hidden)]
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden))
            layers += [
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            d_in = hidden

        layers += [nn.Linear(d_in, 1)]
        self.net = nn.Sequential(*layers)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)  # (B,)


# -----------------------------
# Model
# -----------------------------
class DihedralTransformerAE(nn.Module):
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
        latent_activation: str = "linear",  # "linear" | "tanh"
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_model = d_model
        self.latent_dim = latent_dim

        if latent_activation not in ("linear", "tanh"):
            raise ValueError("latent_activation must be 'linear' or 'tanh'")
        self.latent_activation = latent_activation

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

    def _encode_core(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        B, N, _ = x.shape

        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.input_norm(h)

        if mask is not None:
            src_key_padding_mask = (~mask).expand(B, -1)  # (B, N) True=ignore
            h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)
        else:
            h = self.encoder(h)

        scores = self.attn_pool(h).squeeze(-1)  # (B, N)
        if mask is not None:
            scores = scores.masked_fill((~mask).unsqueeze(0), -1e9)

        w = torch.softmax(scores, dim=1)  # (B, N)
        pooled = (h * w.unsqueeze(-1)).sum(dim=1)  # (B, d_model)
        z = self.to_latent(pooled)  # (B, latent_dim)

        if self.latent_activation == "tanh":
            z = torch.tanh(z)

        return z, w

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        z, _ = self._encode_core(x, mask)
        return z

    @torch.jit.ignore
    def encode_with_attention(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        return self._encode_core(x, mask)

    def decode(self, z: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B = z.size(0)
        context = self.from_latent(z).unsqueeze(1)  # (B, 1, d_model)

        queries = self.residue_queries.expand(B, -1, -1)  # (B, N, d_model)
        h = queries + context
        h = self.pos_enc(h)

        if mask is not None:
            src_key_padding_mask = (~mask).unsqueeze(0).expand(B, -1)  # (B, N)
            h = self.decoder(h, src_key_padding_mask=src_key_padding_mask)
        else:
            h = self.decoder(h)

        x_hat = self.output_proj(h)
        x_hat = self._normalize_sincos_pairs(x_hat)
        return x_hat

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        z = self.encode(x, mask)
        x_hat = self.decode(z, mask)
        return x_hat, z


# -----------------------------
# Positional Encoding
# -----------------------------
class SinusoidalPositionalEncoding(nn.Module):
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


# -----------------------------
# Loss
# -----------------------------
def dihedral_loss(x_hat: torch.Tensor, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
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


# -----------------------------
# Scheduler: warmup + cosine
# -----------------------------
class WarmupCosineScheduler:
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
                t = min(1.0, (self.step_num - self.warmup_steps) / denom)
                cosine = 0.5 * (1.0 + math.cos(math.pi * t))
                lr = base_lr * (self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine)

            pg["lr"] = lr

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]
