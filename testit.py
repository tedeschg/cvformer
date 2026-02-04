# ==========================================================
# Dihedral Transformer Autoencoder / AAE (WGAN-GP) - VERSION 0.3
# ==========================================================

import argparse
import numpy as np
import torch
import mdtraj as md

from pathlib import Path
from torch.utils.data import DataLoader, Subset

from pkgs.utils import angles_to_sincos, compute_aligned_phi_psi, DihedralDataset
from pkgs.model import DihedralTransformerAE, WarmupCosineScheduler, LatentCritic
from pkgs.train import (
    validate_with_metrics,
    train_epoch,
    train_epoch_aae_wgangp,
    extract_latents,
    extract_attention_weights,
)
from pkgs.plumed_export import export_plumed_encoder


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main(args):
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.txt", "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"\nLoading trajectory: {args.trajectory}")
    traj = md.load(args.trajectory, top=args.topology)
    print(f"Loaded frames={traj.n_frames} residues={traj.n_residues}")

    phi, psi, residue_ids, mask_np = compute_aligned_phi_psi(traj)
    X = angles_to_sincos(phi, psi).astype(np.float32)

    n_frames, n_tokens, _ = X.shape
    print(f"Aligned tokens: {n_tokens} | frames={n_frames}")

    dataset = DihedralDataset(X, mask_np)
    mask_t = dataset.mask.to(device)  # (N,) bool

    train_size = int(args.train_split * len(dataset))
    train_size = max(1, min(train_size, len(dataset) - 1))
    train_idx = list(range(train_size))
    val_idx = list(range(train_size, len(dataset)))

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    print(f"\nData split (temporal): Train={len(train_dataset)} | Val={len(val_dataset)}")

    pin_memory = (device.type == "cuda")
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory
    )

    # Uniform prior: force z in [-1,1] with tanh (stability)
    latent_activation = "tanh" if (args.use_aae and args.prior_kind == "uniform") else "linear"

    model = DihedralTransformerAE(
        n_tokens=n_tokens,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        latent_dim=args.latent_dim,
        latent_activation=latent_activation,
    ).to(device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    steps_per_epoch = max(1, len(train_loader))
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    if not args.use_aae:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = WarmupCosineScheduler(
            optimizer, warmup_steps=warmup_steps, total_steps=total_steps, min_lr_ratio=args.min_lr_ratio
        )
        critic = None
        opt_critic = None
        sch_critic = None
    else:
        # AE optimizer (2 updates per batch: recon + adv) => scheduler steps doubled
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = WarmupCosineScheduler(
            optimizer, warmup_steps=warmup_steps, total_steps=2 * total_steps, min_lr_ratio=args.min_lr_ratio
        )

        critic = LatentCritic(
            latent_dim=args.latent_dim,
            hidden=args.critic_hidden,
            depth=args.critic_depth,
            dropout=args.critic_dropout,
        ).to(device)

        opt_critic = torch.optim.AdamW(
            critic.parameters(), lr=args.critic_lr, weight_decay=args.critic_weight_decay
        )
        sch_critic = WarmupCosineScheduler(
            opt_critic,
            warmup_steps=warmup_steps,
            total_steps=args.epochs * steps_per_epoch * max(1, int(args.n_critic)),
            min_lr_ratio=args.min_lr_ratio,
        )

    print("\nTraining config:")
    print(f"  epochs={args.epochs} batch_size={args.batch_size} lr={args.lr}")
    print(f"  warmup_epochs={args.warmup_epochs} min_lr_ratio={args.min_lr_ratio} patience={args.patience}")
    if args.use_aae:
        print("\nAAE (WGAN-GP) config:")
        print(f"  prior_kind={args.prior_kind} latent_activation={latent_activation}")
        print(f"  lambda_adv={args.lambda_adv} n_critic={args.n_critic} lambda_gp={args.lambda_gp}")
        print(f"  critic_lr={args.critic_lr} critic_wd={args.critic_weight_decay}")

    best_val = float("inf")
    patience_ctr = 0

    train_hist, val_hist = [], []
    mae_phi_hist, mae_psi_hist = [], []

    crit_hist, gen_hist = [], []

    print("\nStarting training...")
    for epoch in range(args.epochs):
        if not args.use_aae:
            tr = train_epoch(model, train_loader, optimizer, scheduler, device, mask_t)
        else:
            tr_recon, tr_crit, tr_gen = train_epoch_aae_wgangp(
                ae=model,
                critic=critic,
                loader=train_loader,
                opt_ae=optimizer,
                sch_ae=scheduler,
                opt_critic=opt_critic,
                sch_critic=sch_critic,
                device=device,
                mask=mask_t,
                prior_kind=args.prior_kind,
                lambda_adv=args.lambda_adv,
                n_critic=args.n_critic,
                lambda_gp=args.lambda_gp,
                freeze_decoder_on_adv=not args.adv_updates_decoder,
            )
            tr = tr_recon
            crit_hist.append(tr_crit)
            gen_hist.append(tr_gen)

        val, mae_phi, mae_psi = validate_with_metrics(model, val_loader, device, mask_t)

        train_hist.append(tr)
        val_hist.append(val)
        mae_phi_hist.append(mae_phi)
        mae_psi_hist.append(mae_psi)

        lr_now = scheduler.get_last_lr()[0]
        if not args.use_aae:
            print(
                f"Epoch {epoch:04d} | Train {tr:.6f} | Val {val:.6f} | "
                f"MAEφ {mae_phi:.4f} | MAEψ {mae_psi:.4f} | LR {lr_now:.2e}"
            )
        else:
            print(
                f"Epoch {epoch:04d} | Recon {tr_recon:.6f} | Crit {tr_crit:.4f} | Gen {tr_gen:.4f} | "
                f"Val {val:.6f} | MAEφ {mae_phi:.4f} | MAEψ {mae_psi:.4f} | LR {lr_now:.2e}"
            )

        # Early stopping monitors ONLY validation reconstruction loss
        if val < best_val:
            best_val = val
            patience_ctr = 0

            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val,
                "config": vars(args),
                "token_residue_ids": residue_ids,
                "mask": mask_np,
            }
            if args.use_aae:
                ckpt["critic_state_dict"] = critic.state_dict()
                ckpt["opt_critic_state_dict"] = opt_critic.state_dict()

            torch.save(ckpt, output_dir / "best_model.pt")
        else:
            patience_ctr += 1

        if patience_ctr >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (best val={best_val:.6f})")
            break

    np.savetxt(output_dir / "train_losses.txt", np.array(train_hist))
    np.savetxt(output_dir / "val_losses.txt", np.array(val_hist))
    np.savetxt(output_dir / "mae_phi.txt", np.array(mae_phi_hist))
    np.savetxt(output_dir / "mae_psi.txt", np.array(mae_psi_hist))

    if args.use_aae:
        np.savetxt(output_dir / "critic_losses.txt", np.array(crit_hist))
        np.savetxt(output_dir / "gen_losses.txt", np.array(gen_hist))

    ckpt = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"\nLoaded best model: val_loss={ckpt['val_loss']:.6f} at epoch={ckpt['epoch']}")

    full_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory)
    latents = extract_latents(model, full_loader, device, mask_t)

    np.save(output_dir / "latents.npy", latents)
    np.savetxt(output_dir / "latents.txt", latents)
    np.savetxt(output_dir / "token_residue_ids.txt", residue_ids, fmt="%d")

    print(f"\nSaved latents: shape={latents.shape}")

    W = extract_attention_weights(model, full_loader, device, mask_t)
    w_mean = W.mean(axis=0)
    np.save(output_dir / "attn_w.npy", W)
    np.savetxt(output_dir / "attn_w_mean.txt", w_mean)
    print(f"Saved attention weights: attn_w.npy shape={W.shape}")

    if args.export_plumed:
        pt_path, info_path = export_plumed_encoder(
            model=model,
            output_dir=output_dir,
            n_tokens=n_tokens,
            latent_dim=args.latent_dim,
            residue_ids=residue_ids,
            mask_np=mask_np,
            pt_name="dihedral_encoder_plumed.pt",
            info_name="plumed_info.json",
        )
        print(f"\nExported TorchScript encoder for PLUMED:")
        print(f"  {pt_path}")
        print(f"  {info_path}")


def build_parser():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--trajectory", type=str, required=True)
    p.add_argument("--topology", type=str, required=True)

    # output
    p.add_argument("--output_dir", type=str, default="out")
    p.add_argument("--export_plumed", action="store_true")

    # training
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--min_lr_ratio", type=float, default=0.01)
    p.add_argument("--train_split", type=float, default=0.8)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)

    # model
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num_encoder_layers", type=int, default=3)
    p.add_argument("--num_decoder_layers", type=int, default=3)
    p.add_argument("--dim_feedforward", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--latent_dim", type=int, default=2)

    # AAE (WGAN-GP)
    p.add_argument("--use_aae", action="store_true")
    p.add_argument("--prior_kind", type=str, default="gaussian", choices=["gaussian", "uniform"])
    p.add_argument("--lambda_adv", type=float, default=0.2)
    p.add_argument("--n_critic", type=int, default=5)
    p.add_argument("--lambda_gp", type=float, default=10.0)
    p.add_argument("--adv_updates_decoder", action="store_true")

    # critic
    p.add_argument("--critic_lr", type=float, default=5e-4)
    p.add_argument("--critic_weight_decay", type=float, default=1e-4)
    p.add_argument("--critic_hidden", type=int, default=128)
    p.add_argument("--critic_depth", type=int, default=3)
    p.add_argument("--critic_dropout", type=float, default=0.1)

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(args)
