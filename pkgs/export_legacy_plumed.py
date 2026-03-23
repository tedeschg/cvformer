#!/usr/bin/env python3
"""Export trained model to PLUMED-compatible TorchScript format (LEGACY MODE)."""

import sys
import argparse
from pathlib import Path
import numpy as np
import torch

# Add parent directory to path to import pkgs
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pkgs.model import DihedralTransformerAE
from pkgs.plumed_export import export_plumed_encoder

def main():
    parser = argparse.ArgumentParser(description="Export trained model to PLUMED (LEGACY MODE)")
    parser.add_argument("--output_dir", "-o", required=True, help="Path to output directory containing model and config")
    args = parser.parse_args()

    # Paths
    output_dir = Path(args.output_dir).resolve()

    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    # Model checkpoint
    checkpoint_path = output_dir / "best_model_weights.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    # Config file
    config_path = output_dir / "config.txt"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load config
    config = {}
    with open(config_path) as f:
        for line in f:
            line = line.strip()
            if line and ':' in line:
                key, value = line.split(':', 1)
                config[key.strip()] = value.strip()

    # Load residue IDs
    residue_ids = np.loadtxt(output_dir / "token_residue_ids.txt", dtype=int)
    n_tokens = len(residue_ids)

    # Model hyperparameters (from config.txt)
    d_model = int(config.get('d_model', 128))
    nhead = int(config.get('nhead', 4))
    num_encoder_layers = int(config.get('num_encoder_layers', 3))
    num_decoder_layers = int(config.get('num_decoder_layers', 3))
    dim_feedforward = int(config.get('dim_feedforward', 256))
    dropout = float(config.get('dropout', 0.2))
    latent_dim = int(config.get('latent_dim', 2))
    memory_tokens = int(config.get('memory_tokens', 4))

    print(f"Loading model from {checkpoint_path}")
    print(f"n_tokens: {n_tokens}")
    print(f"residue_ids: {residue_ids}")

    # Create model
    model = DihedralTransformerAE(
        n_tokens=n_tokens,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        latent_dim=latent_dim,
        memory_tokens=memory_tokens,
    )

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint)
    model.eval()

    # Create mask (all tokens are valid)
    mask = np.ones(n_tokens, dtype=bool)

    print(f"\nExporting to PLUMED format (LEGACY MODE)...")

    # Export in legacy mode
    pt_path, info_path = export_plumed_encoder(
        model=model,
        output_dir=output_dir,
        n_tokens=n_tokens,
        latent_dim=latent_dim,
        residue_ids=residue_ids,
        mask_np=mask,
        pt_name="dihedral_encoder_plumed.pt",
        info_name="plumed_info.json",
    )

    print(f"\n✓ Export successful!")
    print(f"  TorchScript model: {pt_path}")
    print(f"  Info JSON: {info_path}")
    print(f"\nNext step: generate plumed.dat with make_plumed_file.py in flat_sincos mode")

if __name__ == "__main__":
    main()
