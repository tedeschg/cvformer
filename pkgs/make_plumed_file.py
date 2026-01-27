#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

# -----------------------------------------------------------------------------
# Make script runnable both ways:
#   1) python /abs/path/to/pkgs/make_plumed_file.py ...
#   2) python -m pkgs.make_plumed_file ...
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import mdtraj as md

from pkgs.utils import compute_aligned_phi_psi


def plumed_atoms(quad_0based) -> str:
    """MDTraj: 0-based -> PLUMED: 1-based"""
    return ",".join(str(int(a) + 1) for a in quad_0based)


def _infer_and_validate_interface(info: dict, n_tokens: int) -> str:
    """
    Backward-compatible interface check.
    - New format: info["interface"] must be "sincos_only"
    - Old format: infer from input_shape if present; otherwise assume sincos_only (warn)
    """
    interface = info.get("interface", None)
    input_shape = info.get("input_shape", None)

    # If interface explicitly set, enforce it.
    if interface is not None:
        if interface != "sincos_only":
            raise RuntimeError(
                f"plumed_info.json interface mismatch: expected 'sincos_only', got {interface!r}. "
                "Refuse to generate potentially incompatible plumed.dat."
            )
        # Optional structural check if input_shape exists
        if isinstance(input_shape, list) and len(input_shape) == 1:
            if int(input_shape[0]) != 4 * n_tokens:
                raise RuntimeError(
                    "plumed_info.json input_shape mismatch for sincos_only.\n"
                    f"  expected: [4*n_tokens] = [{4*n_tokens}]\n"
                    f"  got:      {input_shape}\n"
                )
        return interface

    # If interface missing, try to infer from input_shape
    if isinstance(input_shape, list) and len(input_shape) == 1:
        if int(input_shape[0]) == 4 * n_tokens:
            # inferred OK
            return "sincos_only"
        raise RuntimeError(
            "plumed_info.json missing 'interface' and input_shape is not consistent with sincos_only.\n"
            f"  expected: [4*n_tokens] = [{4*n_tokens}]\n"
            f"  got:      {input_shape}\n"
        )

    # If both missing, assume sincos_only but warn (keeps old JSON working)
    print(
        "[WARN] plumed_info.json has no 'interface' and no 'input_shape'. "
        "Assuming interface='sincos_only' (backward-compat)."
    )
    return "sincos_only"


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Generate plumed.dat using the same dihedral selection/alignment as training.\n"
            "SINCOS-ONLY: always expands phi/psi into sin/cos and passes 4*N ARGs to PYTORCH_MODEL.\n"
            "Optionally, pass --plumed_info to enforce strict consistency with training metadata."
        )
    )
    ap.add_argument("--top", required=True, help="Topology file used in training/MD (.pdb/.gro)")
    ap.add_argument("--traj", default=None, help="Trajectory file (.xtc/.dcd). Optional; if omitted loads topology only.")
    ap.add_argument("--pt", default="dihedral_encoder_plumed.pt", help="TorchScript encoder filename (as referenced in plumed.dat)")
    ap.add_argument("--out", default="plumed.dat", help="Output plumed.dat path")

    ap.add_argument("--whole_entity0", default=None, help="WHOLEMOLECULES ENTITY0 range e.g. '1-272'")

    ap.add_argument(
        "--plumed_info",
        default=None,
        help="Path to plumed_info.json saved by training (checks interface and n_tokens match).",
    )

    # Optional METAD block from precomputed latents
    ap.add_argument("--latents", default=None, help="latents.npy to set SIGMA and GRID_{MIN,MAX}")
    ap.add_argument("--pace", type=int, default=1000)
    ap.add_argument("--height", type=float, default=1.0)
    ap.add_argument("--biasfactor", type=float, default=15.0)
    ap.add_argument("--sigma_div", type=float, default=20.0, help="SIGMA = range/sigma_div")
    ap.add_argument("--grid_margin", type=float, default=3.0, help="GRID margin multiplier (range*grid_margin on both sides)")
    ap.add_argument("--hills", default="HILLS")

    ap.add_argument("--stride", type=int, default=100)
    args = ap.parse_args()

    # -------------------------
    # Load traj (1 frame enough)
    # -------------------------
    if args.traj:
        traj = md.load(args.traj, top=args.top)
    else:
        top = md.load_topology(args.top)
        xyz = np.zeros((1, top.n_atoms, 3), dtype=np.float32)
        traj = md.Trajectory(xyz, top)

    # -------------------------
    # Token order (current alignment)
    # -------------------------
    _, _, residue_ids, _ = compute_aligned_phi_psi(traj)
    n_tokens = int(len(residue_ids))

    # -------------------------
    # Optional: validate against plumed_info.json
    # -------------------------
    if args.plumed_info is not None:
        info_path = Path(args.plumed_info)
        if not info_path.exists():
            raise FileNotFoundError(f"--plumed_info not found: {info_path}")

        info = json.loads(info_path.read_text())

        # backward-compatible interface inference + validation
        interface = _infer_and_validate_interface(info, n_tokens)

        info_n_tokens = int(info.get("n_tokens", -1))
        if info_n_tokens != n_tokens:
            raise RuntimeError(
                "n_tokens mismatch between topology/alignment and training metadata.\n"
                f"  computed n_tokens = {n_tokens}\n"
                f"  plumed_info n_tokens = {info_n_tokens}\n"
                "This usually means you're using a different topology, chain selection, or different phi/psi alignment.\n"
                "Fix by using the exact same topology used during training and regenerate."
            )

        latent_dim = info.get("latent_dim", None)
        if latent_dim is not None:
            print(f"[OK] plumed_info: interface={interface}, n_tokens={info_n_tokens}, latent_dim={latent_dim}")
        else:
            print(f"[OK] plumed_info: interface={interface}, n_tokens={info_n_tokens}")

    # -------------------------
    # Backbone dihedral quads
    # -------------------------
    phi_idx, _ = md.compute_phi(traj)
    psi_idx, _ = md.compute_psi(traj)
    top = traj.topology

    # Map residue_index -> quadruple (0-based)
    # phi central residue is atom 1 (N(i)); psi central residue is atom 0 (N(i))
    def resid_of_phi_quad(q):
        return top.atom(int(q[1])).residue.index

    def resid_of_psi_quad(q):
        return top.atom(int(q[0])).residue.index

    phi_map = {int(resid_of_phi_quad(q)): q for q in phi_idx}
    psi_map = {int(resid_of_psi_quad(q)): q for q in psi_idx}

    # -------------------------
    # Write plumed.dat (SINCOS-ONLY)
    # -------------------------
    lines = []
    lines.append("RESTART")
    if args.whole_entity0:
        lines.append(f"WHOLEMOLECULES ENTITY0={args.whole_entity0}")

    arg_labels = []

    for t, r in enumerate(residue_ids):
        r = int(r)
        if r not in phi_map or r not in psi_map:
            raise RuntimeError(
                f"Residue {r} is in residue_ids but missing phi/psi quadruple. "
                "Mismatch between topology and compute_phi/psi indexing."
            )

        qphi = phi_map[r]
        qpsi = psi_map[r]

        lab_phi = f"phi{t}"
        lab_psi = f"psi{t}"

        lines.append(f"{lab_phi}: TORSION ATOMS={plumed_atoms(qphi)}")
        lines.append(f"{lab_psi}: TORSION ATOMS={plumed_atoms(qpsi)}")

        # Always expand to sin/cos to match training/model interface
        lab_sphi = f"sphi{t}"
        lab_cphi = f"cphi{t}"
        lab_spsi = f"spsi{t}"
        lab_cpsi = f"cpsi{t}"

        lines.append(f"{lab_sphi}: MATHEVAL ARG={lab_phi} FUNC=sin(x) PERIODIC=NO")
        lines.append(f"{lab_cphi}: MATHEVAL ARG={lab_phi} FUNC=cos(x) PERIODIC=NO")
        lines.append(f"{lab_spsi}: MATHEVAL ARG={lab_psi} FUNC=sin(x) PERIODIC=NO")
        lines.append(f"{lab_cpsi}: MATHEVAL ARG={lab_psi} FUNC=cos(x) PERIODIC=NO")

        # Model inputs: [sin(phi), cos(phi), sin(psi), cos(psi)] per token
        arg_labels.extend([lab_sphi, lab_cphi, lab_spsi, lab_cpsi])

    # Sanity: 4*N ARGs
    if len(arg_labels) != 4 * n_tokens:
        raise RuntimeError(
            f"Internal error: built {len(arg_labels)} ARG labels, expected {4*n_tokens} (4*n_tokens)."
        )

    # PYTORCH_MODEL uses ARG (not ATOMS)
    lines.append(f"model: PYTORCH_MODEL FILE={args.pt} ARG={','.join(arg_labels)}")

    # Optional METAD computed from training latents range
    if args.latents:
        lat = np.load(args.latents)
        if lat.ndim != 2 or lat.shape[1] < 2:
            raise RuntimeError(f"latents.npy must be (n_frames, >=2). Got {lat.shape}")

        lat2 = lat[:, :2]
        lmin = np.min(lat2, axis=0)
        lmax = np.max(lat2, axis=0)
        llen = lmax - lmin

        # guard against degenerate range
        llen = np.maximum(llen, 1e-6)

        sigma = llen / float(args.sigma_div)
        grid_min = lmin - llen * float(args.grid_margin)
        grid_max = lmax + llen * float(args.grid_margin)

        lines.append(
            "metad: METAD "
            "ARG=model.node-0,model.node-1 "
            f"PACE={args.pace} HEIGHT={args.height} BIASFACTOR={args.biasfactor} "
            f"SIGMA={sigma[0]},{sigma[1]} "
            f"GRID_MIN={grid_min[0]},{grid_min[1]} "
            f"GRID_MAX={grid_max[0]},{grid_max[1]} "
            f"FILE={args.hills}"
        )
        lines.append(f"PRINT FILE=COLVAR ARG=model.node-0,model.node-1,metad.bias STRIDE={args.stride}")
    else:
        lines.append(f"PRINT FILE=COLVAR ARG=model.node-0,model.node-1 STRIDE={args.stride}")

    Path(args.out).write_text("\n".join(lines) + "\n")

    print(f"Wrote {args.out}")
    print(f"  tokens            : {n_tokens}")
    print(f"  model inputs (ARG): {4 * n_tokens}  (sin/cos only)")
    print(f"  encoder file      : {args.pt}")
    print(f"First 5 residue_ids : {residue_ids[:5].tolist()}")
    print(f"Last  5 residue_ids : {residue_ids[-5:].tolist()}")


if __name__ == "__main__":
    main()

