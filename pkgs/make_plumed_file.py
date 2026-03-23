#!/usr/bin/env python3
"""pkgs/make_plumed_file.py

Two supported modes:

1) mode=coords (RECOMMENDED)
   - You export a TorchScript model with pkgs.plumed_export.export_plumed_encoder_from_coords
   - That .pt takes ONLY coordinates for ATOMS=... and computes phi/psi internally
   - plumed.dat contains only WHOLEMOLECULES + PYTORCH_MODEL_CV ATOMS=...

2) mode=flat_sincos (LEGACY)
   - PLUMED computes TORSION + sin/cos and passes a flat ARG vector to PYTORCH_MODEL
   - This keeps backward compatibility with earlier outputs
"""

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import mdtraj as md
except Exception:
    md = None


# =============================================================================
# Helpers (used only in legacy mode)
# =============================================================================
def _plumed_atoms(quad_0based) -> str:
    """MDTraj 0-based atom indices -> PLUMED 1-based."""
    return ",".join(str(int(a) + 1) for a in quad_0based)


def _residue_key(res) -> tuple:
    resseq = getattr(res, "resSeq", None)
    if resseq is None:
        resseq = res.index
    ins = getattr(res, "insertion_code", None)
    if ins is None:
        ins = getattr(res, "insertionCode", "")
    if ins is None:
        ins = ""
    return (int(res.chain.index), int(resseq), str(ins), str(res.name))


def _build_phi_psi_maps(traj_or_top):
    if md is None:
        raise RuntimeError("mdtraj is required for legacy mode")

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
        return top.atom(int(q[1])).residue.index

    phi_map = {int(resid_of_quad(q)): q for q in phi_quads}
    psi_map = {int(resid_of_quad(q)): q for q in psi_quads}
    return phi_map, psi_map


def _load_top_or_traj(top_path: str, traj_path: str | None = None):
    if md is None:
        raise RuntimeError("mdtraj is required for legacy mode")
    if traj_path:
        return md.load(traj_path, top=top_path)
    top = md.load_topology(top_path)
    xyz = np.zeros((1, top.n_atoms, 3), dtype=np.float32)
    return md.Trajectory(xyz, top)


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Generate plumed.dat for either (coords->model) or (legacy sincos->model) workflows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    ap.add_argument("--mode", choices=["coords", "flat_sincos"], default="coords")

    # Common
    ap.add_argument("--pt", required=True, help="TorchScript model (.pt) for PLUMED")
    ap.add_argument("--info", required=True, help="plumed_info*.json exported alongside the .pt")
    ap.add_argument("--out", default="plumed.dat", help="Output plumed.dat path")
    ap.add_argument("--whole_entity0", default=None, help="WHOLEMOLECULES ENTITY0 range, e.g. '1-272'")

    # METAD block from latents.npy
    ap.add_argument("--latents", default=None, help="latents.npy to set SIGMA and GRID_{MIN,MAX}")
    ap.add_argument("--pace", type=int, default=1000)
    ap.add_argument("--height", type=float, default=1.0)
    ap.add_argument("--biasfactor", type=float, default=15.0)
    ap.add_argument("--sigma_div", type=float, default=20.0, help="SIGMA = range/sigma_div")
    ap.add_argument("--grid_margin", type=float, default=3.0, help="GRID margin multiplier")
    ap.add_argument("--hills", default="HILLS")
    ap.add_argument("--stride", type=int, default=100)

    # Optional debug
    ap.add_argument("--print_inputs", action="store_true", help="(legacy) PRINT first few sin/cos inputs")

    # Legacy-only inputs
    ap.add_argument("--top", default=None, help="(legacy) Training topology used to export plumed_info.json")
    ap.add_argument("--traj", default=None, help="(legacy) Optional training trajectory")
    ap.add_argument("--plumed_top", default=None, help="(legacy) Topology used by PLUMED/MD (e.g., npt.gro)")
    ap.add_argument("--plumed_traj", default=None, help="(legacy) Optional trajectory for plumed_top")

    args = ap.parse_args()

    info = json.loads(Path(args.info).read_text())
    n_tokens = int(info["n_tokens"])
    latent_dim = int(info["latent_dim"])

    lines: list[str] = []
    lines.append("RESTART")
    if args.whole_entity0:
        lines.append(f"WHOLEMOLECULES ENTITY0={args.whole_entity0}")

    # ---------------------------------------------------------------------
    # MODE 1: coords -> model (recommended)
    # ---------------------------------------------------------------------
    if args.mode == "coords":
        if info.get("mode") not in ("coords_to_sincos", "coords"):
            raise RuntimeError(
                "You selected --mode coords, but the provided info JSON does not look like a coords-export.\n"
                "Expected info['mode'] == 'coords_to_sincos' and presence of 'atoms_plumed_1based'."
            )

        atoms_1based = info.get("atoms_plumed_1based", None)
        if not atoms_1based:
            raise RuntimeError("coords mode requires atoms_plumed_1based in info JSON")

        atoms_str = ",".join(str(int(a)) for a in atoms_1based)
        lines.append(f"model: PYTORCH_MODEL_CV FILE={args.pt} ATOMS={atoms_str}")

        cv_arg0, cv_arg1 = "model.node-0", "model.node-1"

        if args.latents:
            lat = np.load(args.latents)
            if lat.ndim != 2 or lat.shape[1] < 2:
                raise RuntimeError(f"latents.npy must be (n_frames, >=2). Got {lat.shape}")

            lat2 = lat[:, :2]
            lmin = np.min(lat2, axis=0)
            lmax = np.max(lat2, axis=0)
            llen = np.maximum(lmax - lmin, 1e-6)

            sigma = llen / float(args.sigma_div)
            grid_min = lmin - llen * float(args.grid_margin)
            grid_max = lmax + llen * float(args.grid_margin)

            lines.append(
                "metad: METAD "
                f"ARG={cv_arg0},{cv_arg1} "
                f"PACE={args.pace} HEIGHT={args.height} BIASFACTOR={args.biasfactor} "
                f"SIGMA={sigma[0]},{sigma[1]} "
                f"GRID_MIN={grid_min[0]},{grid_min[1]} "
                f"GRID_MAX={grid_max[0]},{grid_max[1]} "
                f"FILE={args.hills}"
            )
            lines.append(f"PRINT FILE=COLVAR ARG={cv_arg0},{cv_arg1},metad.bias STRIDE={args.stride}")
        else:
            lines.append(f"PRINT FILE=COLVAR ARG={cv_arg0},{cv_arg1} STRIDE={args.stride}")

        Path(args.out).write_text("\n".join(lines) + "\n")
        print(f"Wrote {args.out}")
        print(f"  Mode: coords  | Tokens: {n_tokens} | Latent dim: {latent_dim}")
        print(f"  TorchScript: {args.pt}")
        print(f"  Info: {args.info}")
        print(f"  ATOMS count: {len(info['atoms_plumed_1based'])}")
        return

    # ---------------------------------------------------------------------
    # MODE 2: legacy flat sin/cos -> model (kept for compatibility)
    # ---------------------------------------------------------------------
    if args.top is None or args.plumed_top is None:
        raise RuntimeError("legacy mode requires --top and --plumed_top")

    if md is None:
        raise RuntimeError("mdtraj is required for legacy mode")

    residue_ids = np.asarray(info["residue_ids"], dtype=int)
    mask = np.asarray(info.get("mask", [True] * n_tokens), dtype=bool)
    if residue_ids.shape[0] != n_tokens:
        raise RuntimeError(f"plumed_info.json mismatch: residue_ids len {len(residue_ids)} != n_tokens {n_tokens}")
    if mask.shape[0] != n_tokens:
        raise RuntimeError(f"plumed_info.json mismatch: mask len {len(mask)} != n_tokens {n_tokens}")

    tr_traj = _load_top_or_traj(args.top, args.traj)
    pl_traj = _load_top_or_traj(args.plumed_top, args.plumed_traj)
    tr_top = tr_traj.topology
    pl_top = pl_traj.topology

    train_res_by_key = {_residue_key(r): r.index for r in tr_top.residues}
    plumed_res_by_key = {_residue_key(r): r.index for r in pl_top.residues}
    train_to_plumed = {}
    for key, tr_residx in train_res_by_key.items():
        if key in plumed_res_by_key:
            train_to_plumed[int(tr_residx)] = int(plumed_res_by_key[key])

    pl_phi_map, pl_psi_map = _build_phi_psi_maps(pl_traj)

    arg_labels = []
    input_labels_for_print = []

    for t, tr_resid in enumerate(residue_ids):
        tr_resid = int(tr_resid)
        if tr_resid not in train_to_plumed:
            tr_res = tr_top.residue(tr_resid)
            raise RuntimeError(
                "Residue from training residue_ids not found in plumed topology.\n"
                f"  token={t}\n"
                f"  training residue.index={tr_resid}\n"
                f"  training key={_residue_key(tr_res)}\n"
            )
        pl_resid = train_to_plumed[tr_resid]
        if pl_resid not in pl_phi_map or pl_resid not in pl_psi_map:
            pl_res = pl_top.residue(pl_resid)
            raise RuntimeError(
                "Mapped residue exists in plumed topology but missing phi/psi quadruple there.\n"
                f"  token={t}\n"
                f"  plumed residue.index={pl_resid}\n"
                f"  plumed key={_residue_key(pl_res)}\n"
            )

        qphi = pl_phi_map[pl_resid]
        qpsi = pl_psi_map[pl_resid]

        lab_phi = f"phi{t}"
        lab_psi = f"psi{t}"
        lines.append(f"{lab_phi}: TORSION ATOMS={_plumed_atoms(qphi)}")
        lines.append(f"{lab_psi}: TORSION ATOMS={_plumed_atoms(qpsi)}")

        lab_sphi = f"sphi{t}"
        lab_cphi = f"cphi{t}"
        lab_spsi = f"spsi{t}"
        lab_cpsi = f"cpsi{t}"

        lines.append(f"{lab_sphi}: MATHEVAL ARG={lab_phi} FUNC=sin(x) PERIODIC=NO")
        lines.append(f"{lab_cphi}: MATHEVAL ARG={lab_phi} FUNC=cos(x) PERIODIC=NO")
        lines.append(f"{lab_spsi}: MATHEVAL ARG={lab_psi} FUNC=sin(x) PERIODIC=NO")
        lines.append(f"{lab_cpsi}: MATHEVAL ARG={lab_psi} FUNC=cos(x) PERIODIC=NO")

        arg_labels.extend([lab_sphi, lab_cphi, lab_spsi, lab_cpsi])
        if args.print_inputs and t < 5:
            input_labels_for_print.extend([lab_sphi, lab_cphi, lab_spsi, lab_cpsi])

    expected = 4 * n_tokens
    if len(arg_labels) != expected:
        raise RuntimeError(f"Built {len(arg_labels)} ARG labels, expected {expected} (4*n_tokens).")

    lines.append(f"model: PYTORCH_MODEL FILE={args.pt} ARG={','.join(arg_labels)}")
    cv_arg0, cv_arg1 = "model.node-0", "model.node-1"

    if args.latents:
        lat = np.load(args.latents)
        if lat.ndim != 2 or lat.shape[1] < 2:
            raise RuntimeError(f"latents.npy must be (n_frames, >=2). Got {lat.shape}")

        lat2 = lat[:, :2]
        lmin = np.min(lat2, axis=0)
        lmax = np.max(lat2, axis=0)
        llen = np.maximum(lmax - lmin, 1e-6)

        sigma = llen / float(args.sigma_div)
        grid_min = lmin - llen * float(args.grid_margin)
        grid_max = lmax + llen * float(args.grid_margin)

        lines.append(
            "metad: METAD "
            f"ARG={cv_arg0},{cv_arg1} "
            f"PACE={args.pace} HEIGHT={args.height} BIASFACTOR={args.biasfactor} "
            f"SIGMA={sigma[0]},{sigma[1]} "
            f"GRID_MIN={grid_min[0]},{grid_min[1]} "
            f"GRID_MAX={grid_max[0]},{grid_max[1]} "
            f"FILE={args.hills}"
        )

        if args.print_inputs and input_labels_for_print:
            lines.append(
                f"PRINT FILE=COLVAR ARG={cv_arg0},{cv_arg1},metad.bias,{','.join(input_labels_for_print)} STRIDE={args.stride}"
            )
        else:
            lines.append(f"PRINT FILE=COLVAR ARG={cv_arg0},{cv_arg1},metad.bias STRIDE={args.stride}")
    else:
        if args.print_inputs and input_labels_for_print:
            lines.append(f"PRINT FILE=COLVAR ARG={cv_arg0},{cv_arg1},{','.join(input_labels_for_print)} STRIDE={args.stride}")
        else:
            lines.append(f"PRINT FILE=COLVAR ARG={cv_arg0},{cv_arg1} STRIDE={args.stride}")

    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.out}")
    print(f"  Mode: flat_sincos  | Tokens: {n_tokens} | Latent dim: {latent_dim}")
    print(f"  TorchScript: {args.pt}")
    print(f"  Info: {args.info}")
    print(f"  Training top: {args.top}")
    print(f"  Plumed top:   {args.plumed_top}")


if __name__ == "__main__":
    main()
