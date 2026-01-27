# pkgs/plumed_export.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn


class _FlattenEncoderPlumed(nn.Module):
    """
    TorchScript wrapper for PLUMED.

    Input:  (B, 4*N)  with token layout [sin(phi), cos(phi), sin(psi), cos(psi)] repeated N times
    Output: (B, latent_dim)

    Uses the same token mask as training (stored as a buffer).
    """

    def __init__(self, encoder_model: nn.Module, n_tokens: int, mask_np: np.ndarray):
        super().__init__()
        self.encoder = encoder_model
        self.n_tokens = int(n_tokens)

        mask = torch.from_numpy(mask_np.astype(np.bool_))
        if mask.ndim != 1 or mask.shape[0] != self.n_tokens:
            raise ValueError(f"mask shape mismatch: got {mask.shape}, expected ({self.n_tokens},)")
        self.register_buffer("mask", mask)

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        B = x_flat.shape[0]
        x = x_flat.view(B, self.n_tokens, 4)
        # IMPORTANT: use training mask
        return self.encoder.encode(x, mask=self.mask)


def export_plumed_encoder(
    *,
    model: nn.Module,
    output_dir: str | Path,
    n_tokens: int,
    latent_dim: int,
    residue_ids: np.ndarray | list,
    mask_np: np.ndarray,
    pt_name: str = "dihedral_encoder_plumed.pt",
    info_name: str = "plumed_info.json",
) -> tuple[Path, Path]:
    """
    Exports a TorchScript encoder for PLUMED (mask-aware) and writes metadata JSON.

    Returns:
      (pt_path, info_path)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # CPU + eval is safest for TorchScript in PLUMED usage
    model_export = model.cpu().eval()

    wrapper = _FlattenEncoderPlumed(model_export, n_tokens=n_tokens, mask_np=np.asarray(mask_np)).eval()
    scripted = torch.jit.script(wrapper)
    pt_path = out / pt_name
    scripted.save(str(pt_path))

    info: Dict[str, Any] = {
        "n_tokens": int(n_tokens),
        "latent_dim": int(latent_dim),
        "residue_ids": residue_ids.tolist() if hasattr(residue_ids, "tolist") else list(residue_ids),
        "mask": np.asarray(mask_np, dtype=bool).tolist(),
        "input_shape": [4 * int(n_tokens)],
        "output_shape": [int(latent_dim)],
    }
    info_path = out / info_name
    info_path.write_text(json.dumps(info, indent=2))

    return pt_path, info_path
