import argparse
import numpy as np
import torch
import mdtraj as md
from pathlib import Path
from torch.utils.data import DataLoader

from pkgs.utils import angles_to_sincos, compute_aligned_phi_psi, DihedralDataset
from pkgs.model import DihedralTransformerAE
from pkgs.train import extract_latents, extract_attention_weights
from pkgs.plumed_export import export_plumed_encoder, export_plumed_encoder_from_coords

def main():
    parser = argparse.ArgumentParser(description="Extract attention weights and latents from a pre-trained model.")
    parser.add_argument("--trajectory", type=str, required=True)
    parser.add_argument("--topology", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True, help="Path to best_model_weights.pt")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--plumed_top", type=str, default=None)
    
    # Model architecture (must match training)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_encoder_layers", type=int, default=3)
    parser.add_argument("--num_decoder_layers", type=int, default=3)
    parser.add_argument("--dim_feedforward", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=2)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading trajectory: {args.trajectory}")
    traj = md.load(args.trajectory, top=args.topology)
    phi, psi, residue_ids, mask_np = compute_aligned_phi_psi(traj)
    X = angles_to_sincos(phi, psi).astype(np.float32)
    n_frames, n_tokens, _ = X.shape
    
    dataset = DihedralDataset(X, mask_np)
    mask_t = dataset.mask.to(device)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    print(f"Loading model from {args.weights}")
    model = DihedralTransformerAE(
        n_tokens=n_tokens,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        latent_dim=args.latent_dim,
    ).to(device)
    
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    print("Extracting latents...")
    latents = extract_latents(model, loader, device, mask_t)
    np.save(output_dir / "latents.npy", latents)

    print("Extracting attention weights...")
    W = extract_attention_weights(model, loader, device, mask_t)
    w_mean = W.mean(axis=0)
    w_median = np.median(W, axis=0)

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

    print(f"\nTop 20 residues by MEAN attention weight:")
    idx = np.argsort(-w_mean)[:20]
    for r, m, med in zip(residue_ids[idx], w_mean[idx], w_median[idx]):
        print(f"  residue {int(r):4d} | mean={m:.6e} | median={med:.6e}")

    # Export PLUMED
    print("\nRe-exporting PLUMED encoders...")
    export_plumed_encoder(
        model=model,
        output_dir=output_dir,
        n_tokens=n_tokens,
        latent_dim=args.latent_dim,
        residue_ids=residue_ids,
        mask_np=mask_np,
        pt_name="dihedral_encoder_plumed.pt",
        info_name="plumed_info.json",
    )
    
    if args.plumed_top:
        export_plumed_encoder_from_coords(
            model=model,
            output_dir=output_dir,
            n_tokens=n_tokens,
            latent_dim=args.latent_dim,
            residue_ids=residue_ids,
            mask_np=mask_np,
            training_top=args.topology,
            training_traj=args.trajectory,
            plumed_top=args.plumed_top,
            plumed_traj=None,
            pt_name="dihedral_encoder_fromcoords_plumed.pt",
            info_name="plumed_info_fromcoords.json",
        )

    print(f"\nDone. Outputs in {output_dir}")

if __name__ == "__main__":
    main()
