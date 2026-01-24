import argparse
import copy
import json
import numpy as np
import torch
import mdtraj as md
import torch.nn as nn

from pathlib import Path
from torch.utils.data import DataLoader, Subset

from pkgs.utils import angles_to_sincos, compute_aligned_phi_psi, DihedralDataset
from pkgs.model import DihedralTransformerAE, WarmupCosineScheduler
from pkgs.train import validate_with_metrics, train_epoch, extract_latents


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main(args) -> None:
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    traj = md.load(args.trajectory, top=args.topology)

    phi, psi, residue_ids, mask_np = compute_aligned_phi_psi(traj)
    X = angles_to_sincos(phi, psi).astype(np.float32)

    n_frames, n_tokens, _ = X.shape
    dataset = DihedralDataset(X, mask_np)
    mask_t = dataset.mask.to(device)

    train_size = int(args.train_split * len(dataset))
    train_size = max(1, min(train_size, len(dataset) - 1))

    train_dataset = Subset(dataset, list(range(train_size)))
    val_dataset = Subset(dataset, list(range(train_size, len(dataset))))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = DihedralTransformerAE(
        n_tokens=n_tokens,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        latent_dim=args.latent_dim,
        use_spectral_norm=True,  # training is unchanged
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = args.epochs * max(1, len(train_loader))
    warmup_steps = args.warmup_epochs * max(1, len(train_loader))

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=args.min_lr_ratio,
    )

    best_val = float("inf")
    patience_ctr = 0

    print("Training...")
    for epoch in range(args.epochs):
        tr = train_epoch(model, train_loader, optimizer, scheduler, device, mask_t)
        val, mae_phi, mae_psi = validate_with_metrics(model, val_loader, device, mask_t)

        if epoch % args.log_interval == 0:
            print(f"Epoch {epoch:04d} | Train {tr:.6f} | Val {val:.6f} | MAEφ {mae_phi:.4f} | MAEψ {mae_psi:.4f}")

        if val < best_val:
            best_val = val
            patience_ctr = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val,
                    "config": vars(args),
                    "token_residue_ids": residue_ids,
                    "mask": mask_np,
                },
                output_dir / "best_model.pt",
            )
        else:
            patience_ctr += 1

        if patience_ctr >= args.patience:
            print("Early stopping")
            break

    ckpt = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    full_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    latents = extract_latents(model, full_loader, device, mask_t)
    np.save(output_dir / "latents.npy", latents)

    # =============================
    # PLUMED EXPORT (TorchScript)
    # =============================
    print("=" * 60)
    print("Exporting model for PLUMED integration...")
    print("=" * 60)

    class FlattenEncoder(nn.Module):
        """
        Input:  (B, 4*N)
        Output: (B, latent_dim)
        """

        def __init__(self, encoder: DihedralTransformerAE):
            super().__init__()
            self.encoder = encoder
            self.n_tokens = encoder.n_tokens

        def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
            B = x_flat.shape[0]
            x = x_flat.view(B, self.n_tokens, 4)
            return self.encoder.encode(x, mask=None)

    # Export safe: CPU copy + remove spectral_norm hooks + script
    model_export = copy.deepcopy(model).cpu().eval()
    model_export.remove_spectral_norm_()

    flat_encoder = FlattenEncoder(model_export).eval()
    scripted_encoder = torch.jit.script(flat_encoder)
    scripted_encoder.save(output_dir / "dihedral_encoder_plumed.pt")

    plumed_info = {
        "n_tokens": int(n_tokens),
        "latent_dim": int(args.latent_dim),
        "residue_ids": residue_ids.tolist(),
        "input_shape": [4 * n_tokens],
        "output_shape": [args.latent_dim],
    }
    with open(output_dir / "plumed_info.json", "w") as f:
        json.dump(plumed_info, f, indent=2)

    print("Saved TorchScript:", output_dir / "dihedral_encoder_plumed.pt")
    print("Saved metadata:  ", output_dir / "plumed_info.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--trajectory", type=str, required=True)
    parser.add_argument("--topology", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output")

    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_encoder_layers", type=int, default=3)
    parser.add_argument("--num_decoder_layers", type=int, default=3)
    parser.add_argument("--dim_feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--latent_dim", type=int, default=2)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--train_split", type=float, default=0.9)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--min_lr_ratio", type=float, default=0.01)

    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=10)

    args = parser.parse_args()
    main(args)
