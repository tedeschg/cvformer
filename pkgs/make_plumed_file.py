#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import mdtraj as md


def plumed_atoms(quad_0based) -> str:
    """MDTraj 0-based atom indices -> PLUMED 1-based."""
    return ",".join(str(int(a) + 1) for a in quad_0based)


def residue_key(res) -> tuple:
    """
    Build a stable residue identifier across different topologies.
    Uses: chain index + PDB resSeq + insertion code + residue name.
    """
    # MDTraj exposes res.resSeq for PDB-style numbering when available; fallback to res.index.
    resseq = getattr(res, "resSeq", None)
    if resseq is None:
        resseq = res.index

    ins = getattr(res, "insertion_code", None)
    if ins is None:
        # some versions use res.insertionCode; others have nothing
        ins = getattr(res, "insertionCode", "")
    if ins is None:
        ins = ""
    ins = str(ins)

    return (int(res.chain.index), int(resseq), ins, str(res.name))


def build_phi_psi_maps(traj_or_top):
    """
    Return dicts: {residue_index_in_that_topology: quadruple} for phi and psi.
    Needs a Trajectory (1 frame is fine). If passed a Topology, we create dummy xyz.
    """
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

    phi_map = {int(resid_of_quad(q)): q for q in phi_quads}
    psi_map = {int(resid_of_quad(q)): q for q in psi_quads}
    return phi_map, psi_map


def main():
    ap = argparse.ArgumentParser(
        description="Generate plumed.dat using training-aligned residue_ids from plumed_info.json "
                    "and sin/cos(phi,psi) inputs for a TorchScript encoder, but emitting ATOMS indices "
                    "for the actual PLUMED/MD topology."
    )

    # Required training artifacts
    ap.add_argument("--top", required=True, help="Training topology used to export plumed_info.json (.pdb/.gro)")
    ap.add_argument("--pt", required=True, help="TorchScript encoder (.pt) exported for PLUMED")
    ap.add_argument("--info", required=True, help="plumed_info.json exported alongside the .pt")
    ap.add_argument("--out", default="plumed.dat", help="Output plumed.dat path")

    # Optional: training trajectory only used for topology sanity (one frame enough)
    ap.add_argument("--traj", default=None, help="Optional training trajectory (.xtc/.dcd). If omitted loads topology only.")

    # NEW: topology actually used in MD/PLUMED (indices must match THIS)
    ap.add_argument("--plumed_top", required=True, help="Topology actually used for PLUMED/MD (.pdb/.gro)")
    ap.add_argument("--plumed_traj", default=None, help="Optional trajectory for plumed_top (.xtc/.dcd). One frame enough.")

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
        residue_ids = residue_ids[: args.debug_tokens]
        mask = mask[: args.debug_tokens]
        n_tokens = len(residue_ids)

    # --------------------------
    # Load training topology (for residue_ids reference)
    # --------------------------
    if args.traj:
        tr_traj = md.load(args.traj, top=args.top)
    else:
        tr_top = md.load_topology(args.top)
        xyz = np.zeros((1, tr_top.n_atoms, 3), dtype=np.float32)
        tr_traj = md.Trajectory(xyz, tr_top)
    tr_top = tr_traj.topology

    # --------------------------
    # Load plumed topology (THIS defines correct ATOMS indices)
    # --------------------------
    if args.plumed_traj:
        pl_traj = md.load(args.plumed_traj, top=args.plumed_top)
    else:
        pl_top = md.load_topology(args.plumed_top)
        xyz = np.zeros((1, pl_top.n_atoms, 3), dtype=np.float32)
        pl_traj = md.Trajectory(xyz, pl_top)
    pl_top = pl_traj.topology

    # --------------------------
    # Build mapping: training residue.index -> plumed residue.index
    # via stable residue_key (chain,resSeq,ins,resname)
    # --------------------------
    train_res_by_key = {residue_key(r): r.index for r in tr_top.residues}
    plumed_res_by_key = {residue_key(r): r.index for r in pl_top.residues}

    train_to_plumed = {}
    missing = []
    for key, tr_residx in train_res_by_key.items():
        if key in plumed_res_by_key:
            train_to_plumed[int(tr_residx)] = int(plumed_res_by_key[key])
        else:
            missing.append(key)

    # --------------------------
    # Build phi/psi maps on PLUMED topology (correct atom indices)
    # --------------------------
    pl_phi_map, pl_psi_map = build_phi_psi_maps(pl_traj)

    # --------------------------
    # Write plumed.dat
    # --------------------------
    lines = []
    lines.append("RESTART")
    if args.whole_entity0:
        lines.append(f"WHOLEMOLECULES ENTITY0={args.whole_entity0}")

    arg_labels = []
    input_labels_for_print = []

    for t, tr_resid in enumerate(residue_ids):
        tr_resid = int(tr_resid)

        if tr_resid not in train_to_plumed:
            # provide a very explicit error
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

        qphi = pl_phi_map[pl_resid]
        qpsi = pl_psi_map[pl_resid]

        # Define torsions (using PLUMED topology atom indices!)
        lab_phi = f"phi{t}"
        lab_psi = f"psi{t}"
        lines.append(f"{lab_phi}: TORSION ATOMS={plumed_atoms(qphi)}")
        lines.append(f"{lab_psi}: TORSION ATOMS={plumed_atoms(qpsi)}")

        # sin/cos feature expansion
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
        raise RuntimeError(f"Internal error: built {len(arg_labels)} ARG labels, expected {expected} (4*n_tokens).")

    lines.append(f"model: PYTORCH_MODEL FILE={args.pt} ARG={','.join(arg_labels)}")

    if args.cv_tanh:
        sc = float(args.cv_tanh_scale)
        if latent_dim < 2:
            raise RuntimeError(f"latent_dim={latent_dim}, cannot define cv0/cv1.")
        lines.append(f"cv0: MATHEVAL ARG=model.node-0 FUNC=tanh(x/{sc}) PERIODIC=NO")
        lines.append(f"cv1: MATHEVAL ARG=model.node-1 FUNC=tanh(x/{sc}) PERIODIC=NO")
        cv_arg0, cv_arg1 = "cv0", "cv1"
    else:
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

    print(f"Wrote {args.out}")
    print(f"  Tokens: {n_tokens}  (expect model input ARGs={4*n_tokens})")
    print(f"  Latent dim: {latent_dim}  (using first two CVs)")
    print(f"  TorchScript: {args.pt}")
    print(f"  Info: {args.info}")
    print(f"  Training top: {args.top}")
    print(f"  Plumed top:   {args.plumed_top}")
    if args.cv_tanh:
        print(f"  CV bounding: tanh(x/{args.cv_tanh_scale}) enabled")
    if args.latents:
        print(f"  METAD: enabled (ranges from {args.latents})")
    else:
        print("  METAD: not written (no --latents)")

    n_valid = int(mask.sum())
    if n_valid != n_tokens:
        print(f"  Note: mask has {n_valid}/{n_tokens} valid tokens (model export uses this mask).")


if __name__ == "__main__":
    main()
