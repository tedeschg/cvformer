# ==========================================================
# Dihedral Transformer Autoencoder - VERSION 2.2 (REWRITE)
# ==========================================================
# Fixes & upgrades applied:
# - Robust phi/psi alignment by residue using mdtraj phi_idx/psi_idx
# - Mask is constant (NOT batched by DataLoader)
# - Proper unit-circle projection for sin/cos pairs (phi and psi)
# - Stable loss scaling (MSE over B*N_valid*4)
# - Warmup + cosine scheduler with safety guards and clamped progress
# - Validation adds circular angular MAE for phi/psi
# - Saves latents as .npy and .txt (+ residue id mapping)
# ==========================================================

import math
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import mdtraj as md


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# -----------------------------
# Utilities: sin/cos
# -----------------------------
def angles_to_sincos(phi: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """
    Args:
        phi, psi: (n_frames, n_tokens) radians
    Returns:
        X: (n_frames, n_tokens, 4) = [sin_phi, cos_phi, sin_psi, cos_psi]
    """
    return np.stack([np.sin(phi), np.cos(phi), np.sin(psi), np.cos(psi)], axis=-1)


def sincos_to_angle_torch(sin_t: torch.Tensor, cos_t: torch.Tensor) -> torch.Tensor:
    """atan2(sin, cos)"""
    return torch.atan2(sin_t, cos_t)


def circular_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Circular difference (wrap to [-pi, pi]) using atan2(sinΔ, cosΔ).
    Args:
        a, b: angles in radians
    Returns:
        delta in [-pi, pi]
    """
    d = a - b
    return torch.atan2(torch.sin(d), torch.cos(d))


# -----------------------------
# Robust dihedral alignment
# -----------------------------
def compute_aligned_phi_psi(traj: md.Trajectory):
    """
    Align phi/psi by residue id so token i corresponds to the SAME residue for both angles.

    mdtraj dihedral conventions:
      phi atoms: [C(i-1), N(i), CA(i), C(i)] -> central residue is atom 1 (N(i))
      psi atoms: [N(i), CA(i), C(i), N(i+1)] -> central residue is atom 0 (N(i))

    Returns:
      phi_aligned, psi_aligned: (n_frames, n_tokens)
      residue_ids: (n_tokens,) residue indices in topology
      mask: (n_tokens,) bool (all True, kept for API consistency)
    """
    phi_idx, phi = md.compute_phi(traj)  # phi: (n_frames, n_phi)
    psi_idx, psi = md.compute_psi(traj)  # psi: (n_frames, n_psi)

    phi_res = np.array([traj.topology.atom(int(a[1])).residue.index for a in phi_idx], dtype=int)
    psi_res = np.array([traj.topology.atom(int(a[0])).residue.index for a in psi_idx], dtype=int)

    common = np.intersect1d(phi_res, psi_res)
    common.sort()
    if common.size == 0:
        raise RuntimeError(
            "No residues found that have BOTH phi and psi. "
            "Check topology/chains or whether this is a protein-like system."
        )

    phi_pos = {r: i for i, r in enumerate(phi_res)}
    psi_pos = {r: i for i, r in enumerate(psi_res)}

    phi_cols = np.array([phi_pos[r] for r in common], dtype=int)
    psi_cols = np.array([psi_pos[r] for r in common], dtype=int)

    phi_aligned = phi[:, phi_cols]
    psi_aligned = psi[:, psi_cols]

    mask = np.ones(common.size, dtype=bool)
    return phi_aligned, psi_aligned, common, mask


# -----------------------------
# Dataset (mask NOT returned)
# -----------------------------
class DihedralDataset(Dataset):
    def __init__(self, X: np.ndarray, mask: np.ndarray):
        """
        Args:
            X: (n_frames, n_tokens, 4) float32
            mask: (n_tokens,) bool
        """
        self.X = torch.from_numpy(X).float()
        self.mask = torch.from_numpy(mask).bool()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]


# -----------------------------
# Positional Encoding
# -----------------------------
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
        # x: (B, L, d_model)
        return x + self.pe[:, : x.size(1), :]


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
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_model = d_model
        self.latent_dim = latent_dim

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
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        # Attention pooling (more expressive than mean pooling)
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
        phi = phi / (phi.norm(dim=-1, keepdim=True) + eps)
        psi = psi / (psi.norm(dim=-1, keepdim=True) + eps)
        return torch.cat([phi, psi], dim=-1)

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (B, N, 4)
        mask: (N,) bool True=valid, False=ignore
        """
        B, N, _ = x.shape
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.input_norm(h)

        src_key_padding_mask = None
        if mask is not None:
            src_key_padding_mask = (~mask).unsqueeze(0).expand(B, -1)  # (B, N) True=ignore

        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)  # (B, N, d_model)

        # Attention pooling
        scores = self.attn_pool(h).squeeze(-1)  # (B, N)
        if mask is not None:
            scores = scores.masked_fill((~mask).unsqueeze(0), -1e9)
        w = torch.softmax(scores, dim=1)  # (B, N)
        pooled = (h * w.unsqueeze(-1)).sum(dim=1)  # (B, d_model)

        z = self.to_latent(pooled)  # (B, latent_dim)
        return z

    def decode(self, z: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        z: (B, latent_dim)
        mask: (N,) bool (used to ignore invalid tokens inside decoder if desired)
        """
        B = z.size(0)
        context = self.from_latent(z).unsqueeze(1)  # (B, 1, d_model)

        queries = self.residue_queries.expand(B, -1, -1)  # (B, N, d_model)
        h = queries + context  # broadcast add
        h = self.pos_enc(h)

        src_key_padding_mask = None
        if mask is not None:
            src_key_padding_mask = (~mask).unsqueeze(0).expand(B, -1)

        h = self.decoder(h, src_key_padding_mask=src_key_padding_mask)

        x_hat = self.output_proj(h)
        x_hat = self._normalize_sincos_pairs(x_hat)
        return x_hat

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        z = self.encode(x, mask)
        x_hat = self.decode(z, mask)
        return x_hat, z


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


# -----------------------------
# Metrics (angular MAE)
# -----------------------------
@torch.no_grad()
def angular_mae(x_hat: torch.Tensor, x: torch.Tensor, mask: torch.Tensor | None = None):
    """
    Returns:
      mae_phi, mae_psi in radians
    """
    phi_pred = sincos_to_angle_torch(x_hat[..., 0], x_hat[..., 1])
    psi_pred = sincos_to_angle_torch(x_hat[..., 2], x_hat[..., 3])

    phi_true = sincos_to_angle_torch(x[..., 0], x[..., 1])
    psi_true = sincos_to_angle_torch(x[..., 2], x[..., 3])

    dphi = circular_diff(phi_pred, phi_true).abs()
    dpsi = circular_diff(psi_pred, psi_true).abs()

    if mask is not None:
        mask = mask.to(x.device)
        m = mask.unsqueeze(0).float()  # (1, N)
        denom = m.sum().clamp(min=1.0) * x.shape[0]
        mae_phi = (dphi * m).sum() / denom
        mae_psi = (dpsi * m).sum() / denom
    else:
        mae_phi = dphi.mean()
        mae_psi = dpsi.mean()

    return mae_phi.item(), mae_psi.item()


# -----------------------------
# Train / Validate
# -----------------------------
def train_epoch(model, loader, optimizer, scheduler, device, mask):
    model.train()
    total = 0.0

    for batch in loader:
        batch = batch.to(device)

        x_hat, _ = model(batch, mask)
        loss = dihedral_loss(x_hat, batch, mask)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total += loss.item()

    return total / max(1, len(loader))


@torch.no_grad()
def validate(model, loader, device, mask):
    model.eval()
    total = 0.0
    for batch in loader:
        batch = batch.to(device)
        x_hat, _ = model(batch, mask)
        total += dihedral_loss(x_hat, batch, mask).item()
    return total / max(1, len(loader))


@torch.no_grad()
def validate_with_metrics(model, loader, device, mask):
    model.eval()
    total_loss = 0.0
    total_mae_phi = 0.0
    total_mae_psi = 0.0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        x_hat, _ = model(batch, mask)

        total_loss += dihedral_loss(x_hat, batch, mask).item()
        mae_phi, mae_psi = angular_mae(x_hat, batch, mask)
        total_mae_phi += mae_phi
        total_mae_psi += mae_psi

        n_batches += 1

    n_batches = max(1, n_batches)
    return (
        total_loss / n_batches,
        total_mae_phi / n_batches,
        total_mae_psi / n_batches,
    )


@torch.no_grad()
def extract_latents(model, loader, device, mask):
    model.eval()
    latents = []
    for batch in loader:
        batch = batch.to(device)
        z = model.encode(batch, mask)
        latents.append(z.cpu())
    return torch.cat(latents, dim=0).numpy()


# -----------------------------
# Main
# -----------------------------
def main(args):
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.txt", "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"\nLoading trajectory: {args.trajectory}")
    traj = md.load(args.trajectory, top=args.topology)
    print(f"Loaded frames={traj.n_frames} residues={traj.n_residues}")

    # Compute and align phi/psi robustly
    phi, psi, residue_ids, mask_np = compute_aligned_phi_psi(traj)
    X = angles_to_sincos(phi, psi).astype(np.float32)

    n_frames, n_tokens, _ = X.shape
    print(f"Aligned tokens (residues with BOTH φ and ψ): {n_tokens} / topology residues={traj.n_residues}")
    print(f"Residue id range (topology indices): {residue_ids.min()}..{residue_ids.max()}")

    # Dataset
    dataset = DihedralDataset(X, mask_np)

    # Constant mask on device (NOT batched)
    mask_t = dataset.mask.to(device)

    # Temporal split
    train_size = int(args.train_split * len(dataset))
    train_size = max(1, min(train_size, len(dataset) - 1))
    train_idx = list(range(train_size))
    val_idx = list(range(train_size, len(dataset)))

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    print(f"\nData split (temporal):")
    print(f"  Train: {len(train_dataset)} frames (0..{train_size-1})")
    print(f"  Val:   {len(val_dataset)} frames ({train_size}..{len(dataset)-1})")

    pin_memory = (device.type == "cuda")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    # Model
    model = DihedralTransformerAE(
        n_tokens=n_tokens,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        latent_dim=args.latent_dim,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    total_steps = args.epochs * max(1, len(train_loader))
    warmup_steps = args.warmup_epochs * max(1, len(train_loader))

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=args.min_lr_ratio,
    )

    print(f"\nTraining config:")
    print(f"  epochs={args.epochs}  batch_size={args.batch_size}")
    print(f"  lr={args.lr}  weight_decay={args.weight_decay}")
    print(f"  warmup_epochs={args.warmup_epochs}  warmup_steps={warmup_steps}  total_steps={total_steps}")
    print(f"  min_lr_ratio={args.min_lr_ratio}")
    print(f"  early_stop_patience={args.patience}")

    best_val = float("inf")
    patience_ctr = 0
    train_hist = []
    val_hist = []
    mae_phi_hist = []
    mae_psi_hist = []

    print("\nStarting training...")
    for epoch in range(args.epochs):
        tr = train_epoch(model, train_loader, optimizer, scheduler, device, mask_t)
        val, mae_phi, mae_psi = validate_with_metrics(model, val_loader, device, mask_t)

        train_hist.append(tr)
        val_hist.append(val)
        mae_phi_hist.append(mae_phi)
        mae_psi_hist.append(mae_psi)

        if epoch % args.log_interval == 0 or epoch < 10:
            lr_now = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch:04d} | "
                f"Train {tr:.6f} | Val {val:.6f} | "
                f"MAEφ {mae_phi:.4f} rad | MAEψ {mae_psi:.4f} rad | "
                f"LR {lr_now:.2e}"
            )

        if val < best_val:
            best_val = val
            patience_ctr = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val,
                    "config": vars(args),
                    "token_residue_ids": residue_ids,  # mapping token -> topology residue id
                    "mask": mask_np,
                },
                output_dir / "best_model.pt",
            )
        else:
            patience_ctr += 1

        if patience_ctr >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (best val={best_val:.6f})")
            break

    # Save history
    np.savetxt(output_dir / "train_losses.txt", np.array(train_hist))
    np.savetxt(output_dir / "val_losses.txt", np.array(val_hist))
    np.savetxt(output_dir / "mae_phi.txt", np.array(mae_phi_hist))
    np.savetxt(output_dir / "mae_psi.txt", np.array(mae_psi_hist))

    # Load best model
    ckpt = torch.load(output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"\nLoaded best model: val_loss={ckpt['val_loss']:.6f} at epoch={ckpt['epoch']}")

    # Extract latents for full dataset
    full_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory)
    latents = extract_latents(model, full_loader, device, mask_t)

    np.save(output_dir / "latents.npy", latents)
    np.savetxt(output_dir / "latents.txt", latents)
    np.savetxt(output_dir / "token_residue_ids.txt", residue_ids, fmt="%d")

    print(f"\nSaved:")
    print(f"  latents.npy / latents.txt  (shape={latents.shape})")
    print(f"  token_residue_ids.txt      (len={len(residue_ids)})")
    print(f"  training histories         (train/val/mae_phi/mae_psi)")

    # =============================
    # PLUMED EXPORT
    # =============================
    print(f"\n{'=' * 60}")
    print("Exporting model for PLUMED integration...")
    print(f"{'=' * 60}")

    # FlattenEncoder wrapper for PLUMED (expects flat input)
    class FlattenEncoder(nn.Module):
        """Wrapper that takes flat (4*n_tokens,) input for PLUMED."""
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
            self.n_tokens = encoder.n_tokens

        def forward(self, x_flat):
            # x_flat: (batch, 4*n_tokens)
            batch_size = x_flat.shape[0]
            x = x_flat.view(batch_size, self.n_tokens, 4)
            z = self.encoder.encode(x, mask=None)  # PLUMED will handle masking externally if needed
            return z

    flat_encoder = FlattenEncoder(model)
    scripted_encoder = torch.jit.trace(flat_encoder, torch.zeros(1, 4 * n_tokens, device=device))
    scripted_encoder.save(output_dir / "dihedral_encoder_plumed.pt")

    print(f"✓ Saved PLUMED-compatible encoder: {output_dir / 'dihedral_encoder_plumed.pt'}")
    print(f"  Input shape: (batch, {4 * n_tokens}) = flat sin/cos for {n_tokens} residues")
    print(f"  Output shape: (batch, {args.latent_dim})")

    # Save metadata for PLUMED integration
    import json
    plumed_info = {
        "n_tokens": int(n_tokens),
        "latent_dim": int(args.latent_dim),
        "residue_ids": residue_ids.tolist(),
        "input_shape": [4 * n_tokens],
        "output_shape": [args.latent_dim],
        "notes": "Input: flat array [sin(phi_0), cos(phi_0), sin(psi_0), cos(psi_0), sin(phi_1), ...] for residues in residue_ids"
    }
    with open(output_dir / "plumed_info.json", "w") as f:
        json.dump(plumed_info, f, indent=2)

    print(f"✓ Saved PLUMED metadata: {output_dir / 'plumed_info.json'}")

    print(f"\n{'=' * 60}")
    print(f"Done. Output dir: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Dihedral Transformer Autoencoder (aligned phi/psi, fixed masking, angular metrics)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument("--trajectory", type=str, required=True, help="Trajectory file (.xtc, .dcd, ...)")
    parser.add_argument("--topology", type=str, required=True, help="Topology file (.pdb, .gro, ...)")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")

    # Model
    parser.add_argument("--d_model", type=int, default=64, help="Model dimension")
    parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_encoder_layers", type=int, default=3, help="Encoder layers")
    parser.add_argument("--num_decoder_layers", type=int, default=3, help="Decoder layers")
    parser.add_argument("--dim_feedforward", type=int, default=256, help="FFN dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--latent_dim", type=int, default=2, help="Latent dimension")

    # Training
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=1000, help="Max epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--train_split", type=float, default=0.9, help="Temporal split fraction for train")
    parser.add_argument("--warmup_epochs", type=int, default=10, help="Warmup epochs")
    parser.add_argument("--min_lr_ratio", type=float, default=0.01, help="Min LR ratio for cosine decay")

    # Early stopping / misc
    parser.add_argument("--patience", type=int, default=100, help="Early stopping patience (epochs)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--log_interval", type=int, default=10, help="Log every N epochs")

    args = parser.parse_args()
    main(args)
