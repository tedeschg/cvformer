#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import mdtraj as md


def plumed_atoms(quad_0based) -> str:
    """MDTraj 0-based atom indices -> PLUMED 1-based."""
    return ",".join(str(int(a) + 1) for a in quad_0based)


def main():
    ap = argparse.ArgumentParser(
        description="Generate plumed.dat using training-aligned residue_ids from plumed_info.json "
                    "and sin/cos(phi,psi) inputs for a TorchScript encoder."
    )

    # Required
    ap.add_argument("--top", required=True, help="Topology file used in MD (.pdb/.gro)")
    ap.add_argument("--pt", required=True, help="TorchScript encoder (.pt) exported for PLUMED")
    ap.add_argument("--info", required=True, help="plumed_info.json exported alongside the .pt")
    ap.add_argument("--out", default="plumed.dat", help="Output plumed.dat path")

    # Optional: trajectory only used for topology sanity (one frame enough)
    ap.add_argument("--traj", default=None, help="Optional trajectory file (.xtc/.dcd). If omitted loads topology only.")

    # Optional: WHOLEMOLECULES
    ap.add_argument("--whole_entity0", default=None, help="WHOLEMOLECULES ENTITY0 range, e.g. '1-272'")

    # Optional: add extra CV bounding (recommended to reduce LINCS risk)
    ap.add_argument("--cv_tanh", action="store_true", help="Apply tanh bounding to model outputs before METAD.")
    ap.add_argument("--cv_tanh_scale", type=float, default=2.0, help="cv = tanh(x/scale). Used if --cv_tanh.")

    # Optional: METAD block from latents.npy
    ap.add_argument("--latents", default=None, help="latents.npy to set SIGMA and GRID_{MIN,MAX}")
    ap.add_argument("--pace", type=int, default=1000)
    ap.add_argument("--height", type=float, default=1.0)
    ap.add_argument("--biasfactor", type=float, default=15.0)
    ap.add_argument("--sigma_div", type=float, default=20.0, help="SIGMA = range/sigma_div")
    ap.add_argument("--grid_margin", type=float, default=3.0, help="GRID margin multiplier")
    ap.add_argument("--hills", default="HILLS")
    ap.add_argument("--stride", type=int, default=100)

    # Optional: small safety knobs
    ap.add_argument("--print_inputs", action="store_true", help="Also PRINT the first few sin/cos inputs (debug).")
    ap.add_argument("--debug_tokens", type=int, default=0,
                    help="If >0, limit to first N tokens (debug only). Must match exported model if used in MD (normally keep 0).")

    args = ap.parse_args()

    # --------------------------
    # Load metadata from training
    # --------------------------
    info = json.loads(Path(args.info).read_text())
    n_tokens = int(info["n_tokens"])
    latent_dim = int(info["latent_dim"])
    residue_ids = np.asarray(info["residue_ids"], dtype=int)
    mask = np.asarray(info.get("mask", [True] * n_tokens), dtype=bool)

    if residue_ids.shape[0] != n_tokens:
        raise RuntimeError(f"plumed_info.json mismatch: residue_ids len {len(residue_ids)} != n_tokens {n_tokens}")
    if mask.shape[0] != n_tokens:
        raise RuntimeError(f"plumed_info.json mismatch: mask len {len(mask)} != n_tokens {n_tokens}")

    if args.debug_tokens and args.debug_tokens > 0:
        # WARNING: only for debugging; will NOT match exported model unless you exported with same n_tokens.
        residue_ids = residue_ids[: args.debug_tokens]
        mask = mask[: args.debug_tokens]
        n_tokens = len(residue_ids)

    # --------------------------
    # Load a trajectory/topology (one frame enough)
    # --------------------------
    if args.traj:
        traj = md.load(args.traj, top=args.top)
    else:
        top = md.load_topology(args.top)
        xyz = np.zeros((1, top.n_atoms, 3), dtype=np.float32)
        traj = md.Trajectory(xyz, top)

    top = traj.topology

    # --------------------------
    # Build residue_index -> dihedral quadruple maps using MDTraj conventions
    # --------------------------
    phi_quads, _ = md.compute_phi(traj)  # list of quadruples (0-based atom idx)
    psi_quads, _ = md.compute_psi(traj)

    def resid_of_quad(q):
        # Use second atom as "central" residue (standard for phi/psi definitions)
        return top.atom(int(q[1])).residue.index

    phi_map = {int(resid_of_quad(q)): q for q in phi_quads}
    psi_map = {int(resid_of_quad(q)): q for q in psi_quads}

    # --------------------------
    # Write plumed.dat
    # --------------------------
    lines = []
    lines.append("RESTART")
    if args.whole_entity0:
        lines.append(f"WHOLEMOLECULES ENTITY0={args.whole_entity0}")

    arg_labels = []
    input_labels_for_print = []  # optional debug

    # IMPORTANT: keep token order exactly as training (residue_ids order)
    for t, resid in enumerate(residue_ids):
        resid = int(resid)
        if resid not in phi_map or resid not in psi_map:
            raise RuntimeError(
                f"Residue {resid} (token {t}) missing phi/psi quadruple in this topology. "
                "This indicates a mismatch between training topology and MD topology."
            )

        qphi = phi_map[resid]
        qpsi = psi_map[resid]

        # Define torsions
        lab_phi = f"phi{t}"
        lab_psi = f"psi{t}"
        lines.append(f"{lab_phi}: TORSION ATOMS={plumed_atoms(qphi)}")
        lines.append(f"{lab_psi}: TORSION ATOMS={plumed_atoms(qpsi)}")

        # sin/cos feature expansion
        lab_sphi = f"sphi{t}"
        lab_cphi = f"cphi{t}"
        lab_spsi = f"spsi{t}"
        lab_cpsi = f"cpsi{t}"

        # sin/cos are non-periodic variables in [-1,1]
        lines.append(f"{lab_sphi}: MATHEVAL ARG={lab_phi} FUNC=sin(x) PERIODIC=NO")
        lines.append(f"{lab_cphi}: MATHEVAL ARG={lab_phi} FUNC=cos(x) PERIODIC=NO")
        lines.append(f"{lab_spsi}: MATHEVAL ARG={lab_psi} FUNC=sin(x) PERIODIC=NO")
        lines.append(f"{lab_cpsi}: MATHEVAL ARG={lab_psi} FUNC=cos(x) PERIODIC=NO")

        arg_labels.extend([lab_sphi, lab_cphi, lab_spsi, lab_cpsi])

        if args.print_inputs and t < 5:
            input_labels_for_print.extend([lab_sphi, lab_cphi, lab_spsi, lab_cpsi])

    # Validate 4N args
    expected = 4 * n_tokens
    if len(arg_labels) != expected:
        raise RuntimeError(f"Internal error: built {len(arg_labels)} ARG labels, expected {expected} (4*n_tokens).")

    # Torch model expects ARG=... (not ATOMS=...)
    lines.append(f"model: PYTORCH_MODEL FILE={args.pt} ARG={','.join(arg_labels)}")

    # Optional: apply tanh bounding to CVs before metadynamics
    if args.cv_tanh:
        sc = float(args.cv_tanh_scale)
        # Output node names: model.node-0, model.node-1, ...
        # For latent_dim>2, user can extend later. Here we do first two as usual for METAD.
        if latent_dim < 2:
            raise RuntimeError(f"latent_dim={latent_dim}, cannot define cv0/cv1.")
        lines.append(f"cv0: MATHEVAL ARG=model.node-0 FUNC=tanh(x/{sc}) PERIODIC=NO")
        lines.append(f"cv1: MATHEVAL ARG=model.node-1 FUNC=tanh(x/{sc}) PERIODIC=NO")
        cv_arg0, cv_arg1 = "cv0", "cv1"
    else:
        cv_arg0, cv_arg1 = "model.node-0", "model.node-1"

    # METAD from latents range
    if args.latents:
        lat = np.load(args.latents)
        if lat.ndim != 2 or lat.shape[1] < 2:
            raise RuntimeError(f"latents.npy must be (n_frames, >=2). Got {lat.shape}")

        lat2 = lat[:, :2]
        lmin = np.min(lat2, axis=0)
        lmax = np.max(lat2, axis=0)
        llen = lmax - lmin

        # Guard against degenerate ranges
        llen = np.maximum(llen, 1e-6)

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
        # Print
        if args.print_inputs and input_labels_for_print:
            lines.append(
                f"PRINT FILE=COLVAR ARG={cv_arg0},{cv_arg1},metad.bias,"
                f"{','.join(input_labels_for_print)} STRIDE={args.stride}"
            )
        else:
            lines.append(f"PRINT FILE=COLVAR ARG={cv_arg0},{cv_arg1},metad.bias STRIDE={args.stride}")
    else:
        if args.print_inputs and input_labels_for_print:
            lines.append(
                f"PRINT FILE=COLVAR ARG={cv_arg0},{cv_arg1},{','.join(input_labels_for_print)} STRIDE={args.stride}"
            )
        else:
            lines.append(f"PRINT FILE=COLVAR ARG={cv_arg0},{cv_arg1} STRIDE={args.stride}")

    Path(args.out).write_text("\n".join(lines) + "\n")

    # Summary
    print(f"Wrote {args.out}")
    print(f"  Tokens: {n_tokens}  (expect model input ARGs={4*n_tokens})")
    print(f"  Latent dim: {latent_dim}  (using first two CVs)")
    print(f"  TorchScript: {args.pt}")
    print(f"  Info: {args.info}")
    if args.cv_tanh:
        print(f"  CV bounding: tanh(x/{args.cv_tanh_scale}) enabled")
    if args.latents:
        print(f"  METAD: enabled (ranges from {args.latents})")
    else:
        print("  METAD: not written (no --latents)")

    # Mask sanity
    n_valid = int(mask.sum())
    if n_valid != n_tokens:
        print(f"  Note: mask has {n_valid}/{n_tokens} valid tokens (model export uses this mask).")


if __name__ == "__main__":
    main()
