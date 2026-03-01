# ==========================================================
# Dihedral Transformer Autoencoder - VERSION 0.2 (patched)
# ==========================================================
#
# Training is unchanged.
# At the end you can export for PLUMED:
#
#   --plumed_export legacy   -> dihedral_encoder_plumed.pt
#   --plumed_export coords   -> dihedral_encoder_fromcoords_plumed.pt
#   --plumed_export both     -> both exports
#
# ==========================================================

import argparse
import numpy as np
import torch
import mdtraj as md

from pathlib import Path
from torch.utils.data import DataLoader, Subset

from pkgs.utils import angles_to_sincos, compute_aligned_phi_psi, DihedralDataset
from pkgs.model import DihedralTransformerAE, WarmupCosineScheduler
from pkgs.train import (
    validate_with_metrics,
    train_epoch,
    extract_latents,
    extract_attention_weights,
)

from pkgs.plumed_export import (
    export_plumed_encoder,                 # legacy sin/cos export
    export_plumed_encoder_from_coords,     # coords-only export
)


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

    # Constant mask tensor
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
        tr_loss = train_epoch(model, train_loader, optimizer, scheduler, device, mask_t)
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
    full_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
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
    print(f"  attn_importance_by_residue.txt   (residue_id, mean, median)")

    topk = min(20, len(residue_ids))
    idx = np.argsort(-w_mean)[:topk]
    print("\nTop residues by MEAN attention weight:")
    for r, m, med in zip(residue_ids[idx], w_mean[idx], w_median[idx]):
        print(f"  residue {int(r):4d} | mean={m:.6e} | median={med:.6e}")

    # =============================
    # PLUMED EXPORT OPTIONS
    # =============================
    print(f"\n{'=' * 60}")
    print("PLUMED EXPORT")
    print(f"Mode selected: {args.plumed_export}")
    print(f"{'=' * 60}")

    # --- LEGACY EXPORT ---
    if args.plumed_export in ("legacy", "both"):
        print("\n[1] Exporting LEGACY encoder (sin/cos input via ARG=...)")

        export_plumed_encoder(
            model=model,
            output_dir=output_dir,
            n_tokens=n_tokens,
            latent_dim=args.latent_dim,
            residue_ids=residue_ids,
            mask_np=np.asarray(mask_np, dtype=bool),
            pt_name="dihedral_encoder_plumed.pt",
            info_name="plumed_info.json",
        )

        print("✓ Saved dihedral_encoder_plumed.pt")
        print("✓ Saved plumed_info.json")

    # --- COORDS EXPORT ---
    if args.plumed_export in ("coords", "both"):
        if args.plumed_top is None:
            raise RuntimeError(
                "coords export requires --plumed_top (e.g. npt.gro)"
            )

        print("\n[2] Exporting COORDS-only encoder (ATOMS=... input)")

        export_plumed_encoder_from_coords(
            model=model,
            output_dir=output_dir,
            n_tokens=n_tokens,
            latent_dim=args.latent_dim,
            residue_ids=residue_ids,
            mask_np=np.asarray(mask_np, dtype=bool),
            training_top=args.topology,
            training_traj=args.trajectory,
            plumed_top=args.plumed_top,
            plumed_traj=None,
            pt_name="dihedral_encoder_fromcoords_plumed.pt",
            info_name="plumed_info_fromcoords.json",
        )

        print("✓ Saved dihedral_encoder_fromcoords_plumed.pt")
        print("✓ Saved plumed_info_fromcoords.json")

    print(f"\nAll exports saved in: {output_dir}")
    print(f"{'=' * 60}")


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train DihedralTransformerAE + export PLUMED TorchScript encoders",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Training data
    parser.add_argument("--trajectory", type=str, required=True)
    parser.add_argument("--topology", type=str, required=True)

    # Output
    parser.add_argument("--output_dir", type=str, default="output")

    # PLUMED export mode
    parser.add_argument(
        "--plumed_export",
        choices=["legacy", "coords", "both"],
        default="both",
        help="Which PLUMED encoder export to generate",
    )

    parser.add_argument(
        "--plumed_top",
        type=str,
        default=None,
        help="Topology used in production MD/PLUMED (e.g. npt.gro). Required for coords export.",
    )

    # Model hyperparams
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_encoder_layers", type=int, default=3)
    parser.add_argument("--num_decoder_layers", type=int, default=3)
    parser.add_argument("--dim_feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--latent_dim", type=int, default=2)

    # Training params
    parser.add_argument("--batch_size", type=int, default=128) #64 if gpu problems
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--train_split", type=float, default=0.9)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--min_lr_ratio", type=float, default=0.05)

    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--log_interval", type=int, default=10)

    args = parser.parse_args()
    main(args)
