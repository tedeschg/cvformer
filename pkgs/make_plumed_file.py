import argparse
from pathlib import Path

import numpy as np
import mdtraj as md

# Importa la tua funzione (se vuoi continuare a usarla)
from utils import compute_aligned_phi_psi


def plumed_atoms(quad_0based) -> str:
    """MDTraj: 0-based -> PLUMED: 1-based"""
    return ",".join(str(int(a) + 1) for a in quad_0based)


def main():
    ap = argparse.ArgumentParser(
        description="Generate plumed.dat using the same dihedral selection/alignment as training, "
                    "and optionally expand phi/psi to sin/cos via MATHEVAL."
    )
    ap.add_argument("--top", required=True, help="Topology file used in training/MD (.pdb/.gro)")
    ap.add_argument("--traj", default=None, help="Trajectory file (.xtc/.dcd). Optional; if omitted loads topology only.")
    ap.add_argument("--pt", default="dihedral_encoder_plumed.pt", help="TorchScript encoder filename (as referenced in plumed.dat)")
    ap.add_argument("--out", default="plumed.dat", help="Output plumed.dat path")

    ap.add_argument("--whole_entity0", default=None, help="WHOLEMOLECULES ENTITY0 range e.g. '1-272'")

    # NEW: control whether to output sin/cos MATHEVAL and feed 4N args to the model
    ap.add_argument(
        "--use_sincos",
        action="store_true",
        help="If set, generate MATHEVAL sin/cos variables and pass 4N ARGs to PYTORCH_MODEL "
             "(sphi,cphi,spsi,cpsi per token). If not set, pass 2N ARGs (phi,psi per token).",
    )

    # Optional METAD block from precomputed latents (same logic you described)
    ap.add_argument("--latents", default=None, help="latents.npy to set SIGMA and GRID_{MIN,MAX}")
    ap.add_argument("--pace", type=int, default=1000)
    ap.add_argument("--height", type=float, default=1.0)
    ap.add_argument("--biasfactor", type=float, default=15.0)
    ap.add_argument("--sigma_div", type=float, default=20.0, help="SIGMA = range/sigma_div")
    ap.add_argument("--grid_margin", type=float, default=3.0, help="GRID margin multiplier (range*grid_margin on both sides)")
    ap.add_argument("--hills", default="HILLS")

    ap.add_argument("--stride", type=int, default=100)
    args = ap.parse_args()

    # Load a trajectory (1 frame is enough)
    if args.traj:
        traj = md.load(args.traj, top=args.top)
    else:
        top = md.load_topology(args.top)
        xyz = np.zeros((1, top.n_atoms, 3), dtype=np.float32)
        traj = md.Trajectory(xyz, top)

    # 1) Get residue_ids (token order) consistent with training
    phi, psi, residue_ids, mask_np = compute_aligned_phi_psi(traj)

    # 2) Compute MDTraj backbone dihedral quadruples
    phi_idx, _ = md.compute_phi(traj)
    psi_idx, _ = md.compute_psi(traj)

    top = traj.topology

    # Map residue_index -> quadruple (0-based)
    def resid_of_quad(q):
        return top.atom(int(q[1])).residue.index

    phi_map = {int(resid_of_quad(q)): q for q in phi_idx}
    psi_map = {int(resid_of_quad(q)): q for q in psi_idx}

    # 3) Write plumed.dat
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

        if args.use_sincos:
            # Create sin/cos via MATHEVAL
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
        else:
            # Model inputs: [phi, psi] per token
            arg_labels.extend([lab_phi, lab_psi])

    # PYTORCH_MODEL MUST use ARG, not ATOMS
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

    n_tokens = len(residue_ids)
    if args.use_sincos:
        print(f"Wrote {args.out} with sin/cos MATHEVAL: tokens={n_tokens}, model ARGs={4*n_tokens}")
    else:
        print(f"Wrote {args.out} with raw phi/psi: tokens={n_tokens}, model ARGs={2*n_tokens}")

    print(f"First 5 residue_ids: {residue_ids[:5].tolist()}")
    print(f"Last  5 residue_ids: {residue_ids[-5:].tolist()}")


if __name__ == "__main__":
    main()
