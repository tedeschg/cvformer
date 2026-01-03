import numpy as np
import torch
import mdtraj as md
from torch.utils.data import Dataset

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
