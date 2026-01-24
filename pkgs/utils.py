from typing import Tuple

import numpy as np
import torch
import mdtraj as md
from torch.utils.data import Dataset


def angles_to_sincos(phi: np.ndarray, psi: np.ndarray) -> np.ndarray:
    return np.stack([np.sin(phi), np.cos(phi), np.sin(psi), np.cos(psi)], axis=-1)


def sincos_to_angle_torch(sin_t: torch.Tensor, cos_t: torch.Tensor) -> torch.Tensor:
    return torch.atan2(sin_t, cos_t)


def circular_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    d = a - b
    return torch.atan2(torch.sin(d), torch.cos(d))


def compute_aligned_phi_psi(traj: md.Trajectory) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    phi_idx, phi = md.compute_phi(traj)
    psi_idx, psi = md.compute_psi(traj)

    phi_res = np.array([traj.topology.atom(int(a[1])).residue.index for a in phi_idx], dtype=int)
    psi_res = np.array([traj.topology.atom(int(a[0])).residue.index for a in psi_idx], dtype=int)

    common = np.intersect1d(phi_res, psi_res)
    common.sort()
    if common.size == 0:
        raise RuntimeError("No residues found that have BOTH phi and psi.")

    phi_pos = {r: i for i, r in enumerate(phi_res)}
    psi_pos = {r: i for i, r in enumerate(psi_res)}

    phi_cols = np.array([phi_pos[r] for r in common], dtype=int)
    psi_cols = np.array([psi_pos[r] for r in common], dtype=int)

    phi_aligned = phi[:, phi_cols]
    psi_aligned = psi[:, psi_cols]

    mask = np.ones(common.size, dtype=bool)
    return phi_aligned, psi_aligned, common, mask


class DihedralDataset(Dataset):
    def __init__(self, X: np.ndarray, mask: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.mask = torch.from_numpy(mask).bool()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.X[idx]
