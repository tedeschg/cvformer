#!/usr/bin/env python3
import argparse
import math
from dataclasses import dataclass
from typing import Tuple, Optional

import torch


# ----------------------------
# Utilities: sampling + manifold projection
# ----------------------------
def sample_unit_sincos(batch: int, n_tokens: int, device, dtype) -> torch.Tensor:
    """Sample random valid sin/cos pairs. Returns (B, 4*N)"""
    angles = (torch.rand(batch, n_tokens, 2, device=device, dtype=dtype) * 2 * math.pi) - math.pi
    phi = angles[..., 0]
    psi = angles[..., 1]
    x = torch.stack([torch.sin(phi), torch.cos(phi), torch.sin(psi), torch.cos(psi)], dim=-1)
    return x.reshape(batch, n_tokens * 4)


def sample_critical_angles(batch: int, n_tokens: int, device, dtype, noise_scale: float = 0.05) -> Tuple[
    torch.Tensor, torch.Tensor]:
    """Sample near critical angles (0, ±π/2, ±π) with small noise. Returns angles (B,N,2) and sin/cos (B,4*N)"""
    critical = torch.tensor([0, math.pi / 2, math.pi, -math.pi / 2, -math.pi], device=device, dtype=dtype)

    angles = torch.zeros(batch, n_tokens, 2, device=device, dtype=dtype)
    for i in range(n_tokens):
        idx_phi = torch.randint(0, len(critical), (batch,), device=device)
        idx_psi = torch.randint(0, len(critical), (batch,), device=device)
        angles[:, i, 0] = critical[idx_phi] + noise_scale * torch.randn(batch, device=device, dtype=dtype)
        angles[:, i, 1] = critical[idx_psi] + noise_scale * torch.randn(batch, device=device, dtype=dtype)

    phi = angles[..., 0]
    psi = angles[..., 1]
    x = torch.stack([torch.sin(phi), torch.cos(phi), torch.sin(psi), torch.cos(psi)], dim=-1)
    return angles, x.reshape(batch, n_tokens * 4)


def project_unit_pairs_(x_flat: torch.Tensor, eps: float = 1e-8) -> None:
    """In-place projection to unit circles"""
    B, D = x_flat.shape
    assert D % 4 == 0
    N = D // 4
    x = x_flat.view(B, N, 4)

    phi = x[..., 0:2] / (x[..., 0:2].norm(dim=-1, keepdim=True) + eps)
    psi = x[..., 2:4] / (x[..., 2:4].norm(dim=-1, keepdim=True) + eps)

    x[..., 0:2] = phi
    x[..., 2:4] = psi
    x_flat.copy_(x.view(B, D))


# ----------------------------
# Jacobian computation
# ----------------------------
def jacobian_per_sample(model, x_flat: torch.Tensor) -> torch.Tensor:
    """
    Compute Jacobian dz/dx for a batch.
    z: (B, L), x: (B, D)
    returns J: (B, L, D)

    Uses torch.autograd.grad for efficiency.
    """
    B, D = x_flat.shape
    x = x_flat.detach().clone().requires_grad_(True)
    z = model(x)  # (B, L)
    L = z.shape[1]

    J = torch.zeros((B, L, D), device=x.device, dtype=x.dtype)
    
    for j in range(L):
        # Gradient of the j-th output component for all samples in batch
        grad_outputs = torch.zeros_like(z)
        grad_outputs[:, j] = 1.0
        
        grads = torch.autograd.grad(
            outputs=z,
            inputs=x,
            grad_outputs=grad_outputs,
            retain_graph=True,
            create_graph=False,
            allow_unused=True
        )[0]
        
        if grads is not None:
            J[:, j, :] = grads

    return J.detach()


def jacobian_wrt_angles(model, angles: torch.Tensor) -> torch.Tensor:
    """
    Compute dz/dφ and dz/dψ (full chain rule through sin/cos).
    angles: (B, N, 2) where [...,0]=phi, [...,1]=psi
    Returns: (B, L, 2*N) where L is latent dimension.
    """
    B, N, _ = angles.shape
    angles = angles.detach().clone().requires_grad_(True)

    phi = angles[..., 0]
    psi = angles[..., 1]
    x = torch.stack([torch.sin(phi), torch.cos(phi), torch.sin(psi), torch.cos(psi)], dim=-1)
    x_flat = x.reshape(B, N * 4)

    z = model(x_flat)  # (B, L)
    L = z.shape[1]

    J = torch.zeros((B, L, 2 * N), device=angles.device, dtype=angles.dtype)
    for j in range(L):
        grad_outputs = torch.zeros_like(z)
        grad_outputs[:, j] = 1.0
        
        grads = torch.autograd.grad(
            outputs=z,
            inputs=angles,
            grad_outputs=grad_outputs,
            retain_graph=True,
            create_graph=False,
            allow_unused=True
        )[0]  # (B, N, 2)
        
        if grads is not None:
            J[:, j, :N] = grads[:, :, 0]  # dz_j/dφ for all tokens
            J[:, j, N:] = grads[:, :, 1]  # dz_j/dψ for all tokens

    return J.detach()


def grad_stats_from_J(J: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Frobenius norm and max abs per sample"""
    frob = torch.sqrt((J * J).sum(dim=(1, 2)))
    maxabs = J.abs().amax(dim=(1, 2))
    return frob, maxabs


def token_contrib_sincos(J: torch.Tensor) -> torch.Tensor:
    """Contribution per token from sin/cos Jacobian (B,2,4*N) -> (B,N)"""
    B, L, D = J.shape
    assert D % 4 == 0
    N = D // 4
    Jt = J.view(B, L, N, 4)
    mag = torch.sqrt((Jt * Jt).sum(dim=(1, 3)))
    return mag


def token_contrib_angles(J: torch.Tensor) -> torch.Tensor:
    """Contribution per token from angle Jacobian (B,2,2*N) -> (B,N)"""
    B, L, D = J.shape
    assert D % 2 == 0
    N = D // 2
    J_phi = J[..., :N]  # (B, 2, N)
    J_psi = J[..., N:]  # (B, 2, N)
    mag = torch.sqrt((J_phi * J_phi).sum(dim=1) + (J_psi * J_psi).sum(dim=1))  # (B, N)
    return mag


def percentile(x: torch.Tensor, q: float) -> float:
    x = x.flatten()
    if x.numel() == 0:
        return float("nan")
    xs = x.sort().values
    idx = int(round((q / 100.0) * (xs.numel() - 1)))
    return float(xs[idx])


# ----------------------------
# Worst-case search
# ----------------------------
def worstcase_search_hillclimb(
        model,
        n_tokens: int,
        device,
        dtype,
        n_seeds: int,
        iters: int,
        step_scale: float,
        mode: str = "sincos",  # "sincos" or "angles"
        patience: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    """Random + hill-climb search for worst-case gradients"""
    best_x = None
    best_frob = -1.0
    best_maxabs = -1.0
    best_J = None
    best_z = None

    # Initial seeds
    for _ in range(n_seeds):
        x = sample_unit_sincos(1, n_tokens, device=device, dtype=dtype)

        if mode == "sincos":
            J = jacobian_per_sample(model, x)
        else:  # angles
            # Convert x back to angles for gradient computation
            x_view = x.view(1, n_tokens, 4)
            phi = torch.atan2(x_view[..., 0], x_view[..., 1]).unsqueeze(-1)
            psi = torch.atan2(x_view[..., 2], x_view[..., 3]).unsqueeze(-1)
            angles = torch.cat([phi, psi], dim=-1)  # (1, N, 2)
            J = jacobian_wrt_angles(model, angles)

        frob, maxabs = grad_stats_from_J(J)
        f = float(frob.item())
        if f > best_frob:
            best_frob = f
            best_maxabs = float(maxabs.item())
            best_x = x.detach()
            best_J = J.detach()

    # Hill-climb
    x = best_x.clone()
    no_improve = 0
    for i in range(iters):
        # Adaptive step: reduce step if no improvement
        current_step = step_scale * (0.5 ** (no_improve // (patience // 2)))
        cand = x + current_step * torch.randn_like(x)
        project_unit_pairs_(cand)

        if mode == "sincos":
            J = jacobian_per_sample(model, cand)
        else:
            cand_view = cand.view(1, n_tokens, 4)
            phi = torch.atan2(cand_view[..., 0], cand_view[..., 1]).unsqueeze(-1)
            psi = torch.atan2(cand_view[..., 2], cand_view[..., 3]).unsqueeze(-1)
            angles = torch.cat([phi, psi], dim=-1)
            J = jacobian_wrt_angles(model, angles)

        frob, maxabs = grad_stats_from_J(J)
        f = float(frob.item())
        if f > best_frob:
            best_frob = f
            best_maxabs = float(maxabs.item())
            x = cand.detach()
            best_x = x.detach()
            best_J = J.detach()
            no_improve = 0
        else:
            no_improve += 1
            
        if no_improve >= patience:
            break

    with torch.no_grad():
        best_z = model(best_x).detach()

    return best_x, best_z, best_J, best_frob, best_maxabs


# ----------------------------
# Main
# ----------------------------
@dataclass
class Report:
    name: str
    frob_mean: float
    frob_p50: float
    frob_p95: float
    frob_max: float
    maxabs_mean: float
    maxabs_p50: float
    maxabs_p95: float
    maxabs_max: float
    mask_leak: float = 0.0


def run_diagnostic(
        model,
        n_tokens: int,
        device,
        dtype,
        batch: int,
        n_batches: int,
        mode: str,  # "sincos" or "angles" or "critical"
        mask: Optional[torch.Tensor] = None,
        topk: int = 8,
) -> Tuple[Report, Optional[torch.Tensor]]:
    """Run gradient diagnostic"""

    frob_all = []
    maxabs_all = []
    mask_leaks = []
    worst_f = -1.0
    worst_tokmag = None

    for bi in range(n_batches):
        if mode == "critical":
            angles, x = sample_critical_angles(batch, n_tokens, device, dtype)
            J = jacobian_wrt_angles(model, angles)
            tokmag = token_contrib_angles(J)
        elif mode == "angles":
            angles = (torch.rand(batch, n_tokens, 2, device=device, dtype=dtype) * 2 * math.pi) - math.pi
            J = jacobian_wrt_angles(model, angles)
            tokmag = token_contrib_angles(J)
        else:  # sincos
            x = sample_unit_sincos(batch, n_tokens, device, dtype)
            J = jacobian_per_sample(model, x)
            tokmag = token_contrib_sincos(J)

        frob, maxabs = grad_stats_from_J(J)
        frob_all.append(frob.detach().cpu())
        maxabs_all.append(maxabs.detach().cpu())

        # Check mask leakage if applicable
        if mask is not None:
            # tokmag is (B, N), mask is (N,)
            inactive_mag = tokmag[:, ~mask]
            if inactive_mag.numel() > 0:
                mask_leaks.append(inactive_mag.max().item())

        loc = int(torch.argmax(frob).item())
        f = float(frob[loc].item())
        if f > worst_f:
            worst_f = f
            worst_tokmag = tokmag[loc].detach().cpu()

        if (bi + 1) % 10 == 0:
            print(f"  Batch {bi + 1}/{n_batches} done...", end="\r")

    print()
    frob_all = torch.cat(frob_all)
    maxabs_all = torch.cat(maxabs_all)
    avg_mask_leak = sum(mask_leaks) / len(mask_leaks) if mask_leaks else 0.0

    rep = Report(
        name=mode,
        frob_mean=float(frob_all.mean()),
        frob_p50=percentile(frob_all, 50.0),
        frob_p95=percentile(frob_all, 95.0),
        frob_max=float(frob_all.max()),
        maxabs_mean=float(maxabs_all.mean()),
        maxabs_p50=percentile(maxabs_all, 50.0),
        maxabs_p95=percentile(maxabs_all, 95.0),
        maxabs_max=float(maxabs_all.max()),
        mask_leak=avg_mask_leak,
    )

    return rep, worst_tokmag


def print_report(rep: Report, worst_tokmag: Optional[torch.Tensor], topk: int, has_mask: bool = False):
    """Print diagnostic report"""
    print(f"\n=== {rep.name.upper()} ===")
    print(
        f"||J||_F:   mean={rep.frob_mean:.3e}  p50={rep.frob_p50:.3e}  p95={rep.frob_p95:.3e}  max={rep.frob_max:.3e}")
    print(
        f"max|J_ij|: mean={rep.maxabs_mean:.3e}  p50={rep.maxabs_p50:.3e}  p95={rep.maxabs_p95:.3e}  max={rep.maxabs_max:.3e}")
    if has_mask:
        print(f"mask leak: avg_max={rep.mask_leak:.3e} (should be ~0)")

    if worst_tokmag is not None:
        topk = min(topk, worst_tokmag.numel())
        vals, idxs = torch.topk(worst_tokmag, k=topk)
        print(f"Top {topk} token contributions (worst sample):")
        for i in range(topk):
            print(f"  token {int(idxs[i].item()):2d}: {float(vals[i].item()):.3e}")


def main():
    ap = argparse.ArgumentParser(description="Diagnostic tool for CVFormer stability")
    ap.add_argument("--pt", required=True, help="TorchScript .pt model")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--batch", type=int, default=16, help="Batch size for sampling")
    ap.add_argument("--n-batches", type=int, default=50, help="Number of batches to run")
    ap.add_argument("--topk", type=int, default=8, help="Show top-K token contributions")

    # Worst-case search
    ap.add_argument("--do-worstcase", action="store_true", help="Perform hill-climbing search for worst-case")
    ap.add_argument("--wc-seeds", type=int, default=512, help="Random seeds for worst-case search")
    ap.add_argument("--wc-iters", type=int, default=800, help="Hill-climbing iterations")
    ap.add_argument("--wc-step", type=float, default=0.05, help="Search step size")

    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    model = torch.jit.load(args.pt, map_location=device)
    model.eval()

    try:
        n_tokens = int(model.n_tokens)
    except Exception:
        raise RuntimeError("Cannot read model.n_tokens from TorchScript wrapper.")

    # Check for token mask
    mask = None
    try:
        mask = model.mask.detach().to(device)
        print(f"[info] Mask found: {int(mask.sum().item())}/{n_tokens} active tokens.")
    except Exception:
        print("[info] No token mask found.")

    print(f"Model: {args.pt}")
    print(f"Device: {device}, dtype: {dtype}, n_tokens: {n_tokens}")
    print(f"Total samples: {args.batch * args.n_batches}")

    # Test 1: dz/d(sin,cos) - NN architecture only
    rep1, worst1 = run_diagnostic(model, n_tokens, device, dtype, args.batch, args.n_batches, "sincos", mask, args.topk)
    print_report(rep1, worst1, args.topk, has_mask=(mask is not None))

    # Test 2: dz/dφ, dz/dψ - full chain rule
    rep2, worst2 = run_diagnostic(model, n_tokens, device, dtype, args.batch, args.n_batches, "angles", mask, args.topk)
    print_report(rep2, worst2, args.topk, has_mask=(mask is not None))

    # Test 3: Critical angles (near discontinuities)
    rep3, worst3 = run_diagnostic(model, n_tokens, device, dtype, args.batch, args.n_batches, "critical", mask, args.topk)
    print_report(rep3, worst3, args.topk, has_mask=(mask is not None))

    # Worst-case search
    if args.do_worstcase:
        print(f"\n=== WORST-CASE SEARCH (seeds={args.wc_seeds}, iters={args.wc_iters}) ===")

        # Worst for sin/cos
        _, z_wc1, J_wc1, frob_wc1, maxabs_wc1 = worstcase_search_hillclimb(
            model, n_tokens, device, dtype, args.wc_seeds, args.wc_iters, args.wc_step, mode="sincos"
        )
        print(f"\nWorst sin/cos input: ||J||_F={frob_wc1:.3e}  max|J|={maxabs_wc1:.3e}")
        print(f"  z={z_wc1.flatten().tolist()}")
        tokmag1 = token_contrib_sincos(J_wc1)[0].cpu()
        vals, idxs = torch.topk(tokmag1, k=min(args.topk, tokmag1.numel()))
        for i in range(len(vals)):
            print(f"  token {int(idxs[i].item()):2d}: {float(vals[i].item()):.3e}")

        # Worst for angles
        _, z_wc2, J_wc2, frob_wc2, maxabs_wc2 = worstcase_search_hillclimb(
            model, n_tokens, device, dtype, args.wc_seeds, args.wc_iters, args.wc_step, mode="angles"
        )
        print(f"\nWorst angle input: ||J||_F={frob_wc2:.3e}  max|J|={maxabs_wc2:.3e}")
        print(f"  z={z_wc2.flatten().tolist()}")
        tokmag2 = token_contrib_angles(J_wc2)[0].cpu()
        vals, idxs = torch.topk(tokmag2, k=min(args.topk, tokmag2.numel()))
        for i in range(len(vals)):
            print(f"  token {int(idxs[i].item()):2d}: {float(vals[i].item()):.3e}")

    # Interpretation
    print("\n" + "=" * 60)
    print("DIAGNOSTIC INTERPRETATION:")
    print("=" * 60)

    ratio = rep2.frob_max / rep1.frob_max
    print(f"\n1. NN ARCHITECTURE (sin/cos → z):")
    print(f"   Max ||J||_F = {rep1.frob_max:.3e}")
    if rep1.frob_max > 1e4:
        print("   ⚠️  ISSUE: Huge gradients in the NN itself.")
    else:
        print("   ✓  OK: NN does not amplify excessively.")

    print(f"\n2. CHAIN RULE (φ,ψ → sin/cos → z):")
    print(f"   Max ||J||_F = {rep2.frob_max:.3e}")
    print(f"   Ratio angles/sincos = {ratio:.2f}x")
    if ratio > 5:
        print("   ⚠️  ISSUE: d(sin,cos)/dφ amplifies significantly (near discontinuity).")
    else:
        print("   ✓  OK: Chain rule amplification is acceptable.")

    print(f"\n3. CRITICAL ANGLES (0, ±π/2, ±π):")
    print(f"   Max ||J||_F = {rep3.frob_max:.3e}")
    if rep3.frob_max > 2 * rep2.frob_max:
        print("   ⚠️  ISSUE: Gradients explode near critical angles.")
    else:
        print("   ✓  OK: No pathology near discontinuities.")

    print("\nNEXT STEPS:")
    if rep1.frob_max > 1e4:
        print("→ NN Architecture: Add LayerNorm, reduce learning rate, or use gradient clipping during training.")
    if ratio > 5 or rep3.frob_max > 2 * rep2.frob_max:
        print("→ PLUMED: Verify dihedral calculation; consider using MATHEVAL for smoothing.")
    print("→ Bias scale: Test with a smaller bias (e.g., 1/10 of current value).")
    print("→ MD: Enable gradient clipping in PLUMED using MAX_GRADIENT.")


if __name__ == "__main__":
    main()