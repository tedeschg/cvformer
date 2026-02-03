# ==========================================================
# Dihedral Transformer Autoencoder - VERSION 0.1
# ==========================================================
# Fixes & upgrades applied:
# - Robust phi/psi alignment by residue using mdtraj phi_idx/psi_idx
# - Mask is constant (NOT batched by DataLoader)
# - Proper unit-circle projection for sin/cos pairs (phi and psi)
# - Stable loss scaling (MSE over B*N_valid*4)
# - Warmup + cosine scheduler with safety guards and clamped progress
# - Validation adds circular angular MAE for phi/psi
# - Saves latents as .npy and .txt (+ residue id mapping)
# - NEW: Extract attention pooling weights w per frame and compute per-residue statistics
# ==========================================================

import math
import argparse
import numpy as np
import torch
import mdtraj as md
import torch.nn as nn

from pathlib import Path
from torch.utils.data import DataLoader, Subset

from pkgs.utils import angles_to_sincos, compute_aligned_phi_psi, DihedralDataset
from pkgs.model import DihedralTransformerAE, WarmupCosineScheduler
from pkgs.train import validate_with_metrics, train_epoch, extract_latents, extract_attention_weights
from pkgs.plumed_export import export_plumed_encoder

# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

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
    ckpt = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=False)
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
    # NEW: ATTENTION WEIGHTS EXPORT
    # =============================
    print(f"\n{'=' * 60}")
    print("Extracting attention pooling weights w per frame...")
    print(f"{'=' * 60}")

    W = extract_attention_weights(model, full_loader, device, mask_t)  # (n_frames, n_tokens)

    w_mean = W.mean(axis=0)              # (n_tokens,)
    w_median = np.median(W, axis=0)      # (n_tokens,)

    np.save(output_dir / "attn_weights.npy", W)
    np.savetxt(output_dir / "attn_weights_mean.txt", w_mean)
    np.savetxt(output_dir / "attn_weights_median.txt", w_median)

    attn_table = np.column_stack([residue_ids.astype(int), w_mean, w_median])
    np.savetxt(
        output_dir / "attn_importance_by_residue.txt",
        attn_table,
        header="residue_id w_mean w_median",
        fmt=["%d", "%.8e", "%.8e"],
    )

    print(f"Saved attention outputs:")
    print(f"  attn_weights.npy                 (shape={W.shape})")
    print(f"  attn_weights_mean.txt            (shape={w_mean.shape})")
    print(f"  attn_weights_median.txt          (shape={w_median.shape})")
    print(f"  attn_importance_by_residue.txt   (residue_id, mean, median)")

    topk = min(20, len(residue_ids))
    idx = np.argsort(-w_mean)[:topk]
    print("\nTop residues by MEAN attention weight:")
    for r, m, med in zip(residue_ids[idx], w_mean[idx], w_median[idx]):
        print(f"  residue {int(r):4d} | mean={m:.6e} | median={med:.6e}")

    # =============================
    # PLUMED EXPORT
    # =============================
    print(f"\n{'=' * 60}")
    print("Exporting model for PLUMED integration...")
    print(f"{'=' * 60}")

    pt_path, info_path = export_plumed_encoder(
        model=model,
        output_dir=output_dir,
        n_tokens=n_tokens,
        latent_dim=args.latent_dim,
        residue_ids=residue_ids,
        mask_np=np.asarray(mask_np, dtype=bool),
        pt_name="dihedral_encoder_plumed.pt",
        info_name="plumed_info.json",
    )

    print(f"✓ Saved PLUMED-compatible encoder: {pt_path}")
    print(f"  Input shape: (batch, {4 * n_tokens}) = flat sin/cos for {n_tokens} residues")
    print(f"  Output shape: (batch, {args.latent_dim})")
    print(f"✓ Saved PLUMED metadata: {info_path}")

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