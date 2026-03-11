# ==========================================================
# Dihedral Transformer Autoencoder - VERSION 0.3 (no PLUMED)
# ==========================================================
#
# - Training + latents export + attention weights export
# - NO PLUMED / TorchScript export (removed)
#
# ==========================================================

import argparse
from pathlib import Path

import mdtraj as md
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from pkgs.model import DihedralTransformerAE, WarmupCosineScheduler
from pkgs.train import (
    extract_attention_weights,
    extract_latents,
    train_epoch,
    validate_with_metrics,
)
from pkgs.utils import DihedralDataset, angles_to_sincos, compute_aligned_phi_psi


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

    print(f"\nLoading training trajectory: {args.trajectory}")
    traj = md.load(args.trajectory, top=args.topology)
    print(f"Loaded frames={traj.n_frames} residues={traj.n_residues}")

    # Compute aligned phi/psi
    phi, psi, residue_ids, mask_np = compute_aligned_phi_psi(traj)
    X = angles_to_sincos(phi, psi).astype(np.float32)

    n_frames, n_tokens, _ = X.shape
    print(f"Aligned tokens: {n_tokens}")
    print(f"Residue id range: {residue_ids.min()}..{residue_ids.max()}")

    dataset = DihedralDataset(X, mask_np)

    # Constant mask tensor (global over tokens)
    mask_t = dataset.mask.to(device)

    # Temporal split
    train_size = int(args.train_split * len(dataset))
    train_size = max(1, min(train_size, len(dataset) - 1))
    train_dataset = Subset(dataset, list(range(train_size)))
    val_dataset = Subset(dataset, list(range(train_size, len(dataset))))

    print(f"\nSplit:")
    print(f"  Train frames: {len(train_dataset)}")
    print(f"  Val frames:   {len(val_dataset)}")

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
        memory_tokens=args.memory_tokens,
    ).to(device)

    print(f"\nModel params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = args.epochs * max(1, len(train_loader))
    warmup_steps = args.warmup_epochs * max(1, len(train_loader))

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=args.min_lr_ratio,
    )

    # Training loop
    best_val = float("inf")
    patience_ctr = 0

    print("\nStarting training...\n")

    for epoch in range(args.epochs):
        tr_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            mask_t,
            contrastive_weight=args.contrastive_weight,
            contrastive_temp=args.contrastive_temp,
            contrastive_noise=args.contrastive_noise,
        )
        val_loss, mae_phi, mae_psi = validate_with_metrics(model, val_loader, device, mask_t)

        if epoch % args.log_interval == 0:
            print(
                f"Epoch {epoch:04d} | "
                f"Train {tr_loss:.6f} | Val {val_loss:.6f} | "
                f"MAEφ {mae_phi:.4f} | MAEψ {mae_psi:.4f}"
            )

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            patience_ctr = 0
            torch.save(model.state_dict(), output_dir / "best_model_weights.pt")
        else:
            patience_ctr += 1

        if patience_ctr >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Load best weights
    model.load_state_dict(torch.load(output_dir / "best_model_weights.pt", map_location=device))
    model.eval()

    print("\nLoaded best model weights.")

    # Extract latents
    full_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    latents = extract_latents(model, full_loader, device, mask_t)
    np.save(output_dir / "latents.npy", latents)
    print(f"Saved latents.npy shape={latents.shape}")

    # =============================
    # ATTENTION WEIGHTS EXPORT
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
    np.savetxt(output_dir / "token_residue_ids.txt", residue_ids, fmt="%d")

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
    print(f"  token_residue_ids.txt            (shape={residue_ids.shape})")
    print(f"  attn_importance_by_residue.txt   (residue_id, mean, median)")

    topk = min(20, len(residue_ids))
    idx = np.argsort(-w_mean)[:topk]
    print("\nTop residues by MEAN attention weight:")
    for r, m, med in zip(residue_ids[idx], w_mean[idx], w_median[idx]):
        print(f"  residue {int(r):4d} | mean={m:.6e} | median={med:.6e}")

    print(f"\nAll outputs saved in: {output_dir}")
    print(f"{'=' * 60}")


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train DihedralTransformerAE (no PLUMED export) + save latents/attention",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Training data
    parser.add_argument("--trajectory", type=str, required=True)
    parser.add_argument("--topology", type=str, required=True)

    # Output
    parser.add_argument("--output_dir", type=str, default="output")

    # Model hyperparams (recommended defaults for your case: ~10k frames, ~20 residues, latent 2D)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_encoder_layers", type=int, default=3)
    parser.add_argument("--num_decoder_layers", type=int, default=3)
    parser.add_argument("--dim_feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--latent_dim", type=int, default=2)
    parser.add_argument("--memory_tokens", type=int, default=4)

    # Training params
    parser.add_argument("--batch_size", type=int, default=128)  # set 64 if GPU memory issues
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--train_split", type=float, default=0.9)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--min_lr_ratio", type=float, default=0.05)

    # Contrastive loss (optional)
    parser.add_argument("--contrastive_weight", type=float, default=0.0)
    parser.add_argument("--contrastive_temp", type=float, default=0.1)
    parser.add_argument("--contrastive_noise", type=float, default=0.05)

    # Misc
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--log_interval", type=int, default=10)

    args = parser.parse_args()
    main(args)