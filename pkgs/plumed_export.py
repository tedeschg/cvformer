# pkgs/plumed_export.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import mdtraj as md
except Exception as e:  # pragma: no cover
    md = None


# =============================================================================
# Helpers shared with make_plumed_file.py
# =============================================================================
def residue_key(res) -> tuple:
    """Stable residue identifier across different topologies."""
    resseq = getattr(res, "resSeq", None)
    if resseq is None:
        resseq = res.index

    ins = getattr(res, "insertion_code", None)
    if ins is None:
        ins = getattr(res, "insertionCode", "")
    if ins is None:
        ins = ""

    return (int(res.chain.index), int(resseq), str(ins), str(res.name))


def _build_phi_psi_maps(traj_or_top) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Return dicts: {residue_index_in_that_topology: quadruple(0-based atom idx)} for phi and psi.
    Needs a Trajectory (1 frame is fine). If passed a Topology, we create dummy xyz.
    """
    if md is None:
        raise RuntimeError("mdtraj is required for export_plumed_encoder_from_coords")

    if isinstance(traj_or_top, md.Topology):
        top = traj_or_top
        xyz = np.zeros((1, top.n_atoms, 3), dtype=np.float32)
        traj = md.Trajectory(xyz, top)
    else:
        traj = traj_or_top
        top = traj.topology

    phi_quads, _ = md.compute_phi(traj)
    psi_quads, _ = md.compute_psi(traj)

    def resid_of_quad(q):
        # second atom is "central" residue for mdtraj phi/psi definition
        return top.atom(int(q[1])).residue.index

    phi_map = {int(resid_of_quad(q)): np.asarray(q, dtype=np.int64) for q in phi_quads}
    psi_map = {int(resid_of_quad(q)): np.asarray(q, dtype=np.int64) for q in psi_quads}
    return phi_map, psi_map


def _load_top_or_traj(top_path: str | Path, traj_path: str | None = None):
    if md is None:
        raise RuntimeError("mdtraj is required for export_plumed_encoder_from_coords")
    if traj_path:
        traj = md.load(str(traj_path), top=str(top_path))
        return traj
    top = md.load_topology(str(top_path))
    xyz = np.zeros((1, top.n_atoms, 3), dtype=np.float32)
    return md.Trajectory(xyz, top)


def _unique_sorted(it: Iterable[int]) -> list[int]:
    return sorted({int(x) for x in it})


# =============================================================================
# TorchScript wrappers
# =============================================================================
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
        return self.encoder.encode(x, mask=self.mask)


@torch.jit.script
def _dihedral_sincos(p0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, eps: float = 1e-9):
    """
    Compute sin/cos of dihedral angle using a stable formulation.

    Inputs p*: (..., 3)
    Returns: (sin, cos) with shape (...,)
    """
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2

    b1n = b1 / (torch.linalg.norm(b1, dim=-1, keepdim=True) + eps)

    v = b0 - (b0 * b1n).sum(-1, keepdim=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdim=True) * b1n

    v_hat = v / (torch.linalg.norm(v, dim=-1, keepdim=True) + eps)
    w_hat = w / (torch.linalg.norm(w, dim=-1, keepdim=True) + eps)

    cos = (v_hat * w_hat).sum(-1)
    sin = (torch.cross(b1n, v_hat, dim=-1) * w_hat).sum(-1)
    return sin, cos


class _DihedralEncoderFromCoordsPlumed(nn.Module):
    """
    TorchScript wrapper for PLUMED when you want *only* WHOLEMOLECULES + ATOMS.

    Input shapes supported (depending on PLUMED build):
      - (3*N,)
      - (B, 3*N)
      - (N, 3)
      - (B, N, 3)
    """

    def __init__(
        self,
        encoder_model: nn.Module,
        n_tokens: int,
        mask_np: np.ndarray,
        phi_quads_local: np.ndarray,
        psi_quads_local: np.ndarray,
        n_atoms: int,
    ):
        super().__init__()
        self.encoder = encoder_model
        self.n_tokens = int(n_tokens)
        self.n_atoms = int(n_atoms)

        mask = torch.from_numpy(np.asarray(mask_np, dtype=np.bool_))
        if mask.ndim != 1 or mask.shape[0] != self.n_tokens:
            raise ValueError("mask shape mismatch")
        self.register_buffer("mask", mask)

        phi = torch.as_tensor(np.asarray(phi_quads_local, dtype=np.int64))
        psi = torch.as_tensor(np.asarray(psi_quads_local, dtype=np.int64))
        if phi.shape != (self.n_tokens, 4):
            raise ValueError("phi_quads_local shape mismatch")
        if psi.shape != (self.n_tokens, 4):
            raise ValueError("psi_quads_local shape mismatch")

        # bounds check in eager mode (OK)
        if int(phi.max()) >= self.n_atoms or int(psi.max()) >= self.n_atoms:
            raise ValueError("phi/psi quad indices exceed n_atoms")

        self.register_buffer("phi_quads", phi)
        self.register_buffer("psi_quads", psi)

    def _reshape_coords(self, x: torch.Tensor) -> torch.Tensor:
        # Output: (B, N, 3)

        # Case: (3N,)
        if x.dim() == 1:
            x = x.view(1, self.n_atoms, 3)

        # Case: (B, 3N)  OR (N,3)
        elif x.dim() == 2:
            if x.size(1) == 3 and x.size(0) == self.n_atoms:
                # (N,3) -> (1,N,3)
                x = x.unsqueeze(0)
            else:
                # assume (B,3N)
                x = x.view(x.size(0), self.n_atoms, 3)

        # Case: already (B,N,3)
        elif x.dim() == 3:
            # do nothing
            pass

        else:
            raise RuntimeError("Unexpected input ndim for PLUMED coords")

        # TorchScript-safe checks (no tuple(x.shape))
        if x.size(1) != self.n_atoms:
            raise RuntimeError("Bad coord shape: wrong N")
        if x.size(2) != 3:
            raise RuntimeError("Bad coord shape: last dim != 3")

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._reshape_coords(x)

        B = x.size(0)
        T = self.n_tokens

        tok = torch.empty((B, T, 4), dtype=x.dtype, device=x.device)

        # PHI
        i0 = self.phi_quads[:, 0]
        i1 = self.phi_quads[:, 1]
        i2 = self.phi_quads[:, 2]
        i3 = self.phi_quads[:, 3]
        p0 = x.index_select(1, i0)
        p1 = x.index_select(1, i1)
        p2 = x.index_select(1, i2)
        p3 = x.index_select(1, i3)
        sphi, cphi = _dihedral_sincos(p0, p1, p2, p3)

        # PSI
        j0 = self.psi_quads[:, 0]
        j1 = self.psi_quads[:, 1]
        j2 = self.psi_quads[:, 2]
        j3 = self.psi_quads[:, 3]
        q0 = x.index_select(1, j0)
        q1 = x.index_select(1, j1)
        q2 = x.index_select(1, j2)
        q3 = x.index_select(1, j3)
        spsi, cpsi = _dihedral_sincos(q0, q1, q2, q3)

        tok[:, :, 0] = sphi
        tok[:, :, 1] = cphi
        tok[:, :, 2] = spsi
        tok[:, :, 3] = cpsi

        return self.encoder.encode(tok, mask=self.mask)



# =============================================================================
# Exports
# =============================================================================
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
    """Legacy export: PLUMED computes torsions; model input is flat sin/cos."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_export = model.cpu().eval()
    wrapper = _FlattenEncoderPlumed(model_export, n_tokens=int(n_tokens), mask_np=np.asarray(mask_np)).eval()
    scripted = torch.jit.script(wrapper)
    pt_path = out / pt_name
    scripted.save(str(pt_path))

    info: Dict[str, Any] = {
        "mode": "flat_sincos",
        "n_tokens": int(n_tokens),
        "latent_dim": int(latent_dim),
        "residue_ids": residue_ids.tolist() if hasattr(residue_ids, "tolist") else list(residue_ids),
        "mask": np.asarray(mask_np, dtype=bool).tolist(),
        "input_shape": [4 * int(n_tokens)],
        "output_shape": [int(latent_dim)],
        "notes": "Input: flat array [sin(phi_0), cos(phi_0), sin(psi_0), cos(psi_0), sin(phi_1), ...] for residues in residue_ids",
    }
    info_path = out / info_name
    info_path.write_text(json.dumps(info, indent=2))
    return pt_path, info_path


def export_plumed_encoder_from_coords(
    *,
    model: nn.Module,
    output_dir: str | Path,
    n_tokens: int,
    latent_dim: int,
    residue_ids: np.ndarray | list,
    mask_np: np.ndarray,
    training_top: str | Path,
    plumed_top: str | Path,
    training_traj: str | Path | None = None,
    plumed_traj: str | Path | None = None,
    pt_name: str = "dihedral_encoder_fromcoords_plumed.pt",
    info_name: str = "plumed_info_fromcoords.json",
) -> tuple[Path, Path]:
    """
    Export a TorchScript model that takes ONLY atom coordinates from PLUMED (ATOMS=...).

    It embeds:
      - ATOMS list (from plumed_top indices)
      - local phi/psi quadruples indices into that ATOMS list
      - training mask

    So plumed.dat can be reduced to WHOLEMOLECULES + PYTORCH_MODEL_CV ATOMS=...
    """
    if md is None:
        raise RuntimeError("mdtraj is required for export_plumed_encoder_from_coords")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    residue_ids_arr = np.asarray(residue_ids, dtype=int)
    mask_arr = np.asarray(mask_np, dtype=bool)
    if residue_ids_arr.shape[0] != int(n_tokens):
        raise ValueError("residue_ids length must match n_tokens")
    if mask_arr.shape[0] != int(n_tokens):
        raise ValueError("mask length must match n_tokens")

    # Load topologies/trajectories (1 frame is enough)
    tr_traj = _load_top_or_traj(training_top, training_traj)
    pl_traj = _load_top_or_traj(plumed_top, plumed_traj)
    tr_top = tr_traj.topology
    pl_top = pl_traj.topology

    # Map training residue.index -> plumed residue.index via stable key
    train_res_by_key = {residue_key(r): int(r.index) for r in tr_top.residues}
    plumed_res_by_key = {residue_key(r): int(r.index) for r in pl_top.residues}
    train_to_plumed: Dict[int, int] = {}
    for key, tr_residx in train_res_by_key.items():
        if key in plumed_res_by_key:
            train_to_plumed[int(tr_residx)] = int(plumed_res_by_key[key])

    # Compute phi/psi on PLUMED topology (gives correct atom indices for npt.gro)
    pl_phi_map, pl_psi_map = _build_phi_psi_maps(pl_traj)

    # Build per-token global (plumed_top) quads
    phi_quads_global = np.zeros((int(n_tokens), 4), dtype=np.int64)
    psi_quads_global = np.zeros((int(n_tokens), 4), dtype=np.int64)
    used_atoms: list[int] = []

    for t, tr_resid in enumerate(residue_ids_arr.tolist()):
        tr_resid = int(tr_resid)
        if tr_resid not in train_to_plumed:
            tr_res = tr_top.residue(tr_resid)
            raise RuntimeError(
                "Residue from training residue_ids not found in plumed topology.\n"
                f"  token={t}\n"
                f"  training residue.index={tr_resid}\n"
                f"  training key={residue_key(tr_res)}\n"
                "This indicates a mismatch in chain/resSeq/insertion/resname between training and plumed structures."
            )
        pl_resid = train_to_plumed[tr_resid]

        if pl_resid not in pl_phi_map or pl_resid not in pl_psi_map:
            pl_res = pl_top.residue(pl_resid)
            raise RuntimeError(
                "Mapped residue exists in plumed topology but missing phi/psi quadruple there.\n"
                f"  token={t}\n"
                f"  plumed residue.index={pl_resid}\n"
                f"  plumed key={residue_key(pl_res)}\n"
                "Often happens at termini or if backbone atoms are missing."
            )

        qphi = np.asarray(pl_phi_map[pl_resid], dtype=np.int64)
        qpsi = np.asarray(pl_psi_map[pl_resid], dtype=np.int64)
        phi_quads_global[t] = qphi
        psi_quads_global[t] = qpsi
        used_atoms.extend(qphi.tolist())
        used_atoms.extend(qpsi.tolist())

    # ATOMS list = unique set of all atoms that appear in any phi/psi quadruple
    atoms_global_0based = _unique_sorted(used_atoms)
    n_atoms = len(atoms_global_0based)
    if n_atoms == 0:
        raise RuntimeError("No atoms collected for phi/psi. Unexpected.")

    # Global->local index mapping
    g2l = {int(g): i for i, g in enumerate(atoms_global_0based)}
    phi_local = np.vectorize(lambda a: g2l[int(a)])(phi_quads_global).astype(np.int64)
    psi_local = np.vectorize(lambda a: g2l[int(a)])(psi_quads_global).astype(np.int64)

    # Export TorchScript wrapper
    model_export = model.cpu().eval()
    wrapper = _DihedralEncoderFromCoordsPlumed(
        model_export,
        n_tokens=int(n_tokens),
        mask_np=mask_arr,
        phi_quads_local=phi_local,
        psi_quads_local=psi_local,
        n_atoms=int(n_atoms),
    ).eval()
    scripted = torch.jit.script(wrapper)
    pt_path = out / pt_name
    scripted.save(str(pt_path))

    atoms_plumed_1based = [int(a) + 1 for a in atoms_global_0based]

    info: Dict[str, Any] = {
        "mode": "coords_to_sincos",
        "n_tokens": int(n_tokens),
        "latent_dim": int(latent_dim),
        "residue_ids": residue_ids_arr.tolist(),
        "mask": mask_arr.tolist(),
        "n_atoms": int(n_atoms),
        "atoms_plumed_1based": atoms_plumed_1based,
        "phi_quads_local": phi_local.tolist(),
        "psi_quads_local": psi_local.tolist(),
        "input_shape": [3 * int(n_atoms)],
        "output_shape": [int(latent_dim)],
        "notes": "Input: atom coordinates for ATOMS list (plumed_top indices). Model computes sin/cos(phi,psi) internally and returns latent z.",
        "training_top": str(training_top),
        "plumed_top": str(plumed_top),
    }
    info_path = out / info_name
    info_path.write_text(json.dumps(info, indent=2))

    return pt_path, info_path
