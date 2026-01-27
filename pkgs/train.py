from typing import Optional

import torch

from pkgs.utils import sincos_to_angle_torch, circular_diff
from pkgs.model import dihedral_loss


@torch.no_grad()
def angular_mae(x_hat: torch.Tensor, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
    """
    Returns: mae_phi, mae_psi in radians
    """
    phi_pred = sincos_to_angle_torch(x_hat[..., 0], x_hat[..., 1])
    psi_pred = sincos_to_angle_torch(x_hat[..., 2], x_hat[..., 3])

    phi_true = sincos_to_angle_torch(x[..., 0], x[..., 1])
    psi_true = sincos_to_angle_torch(x[..., 2], x[..., 3])

    dphi = circular_diff(phi_pred, phi_true).abs()
    dpsi = circular_diff(psi_pred, psi_true).abs()

    if mask is not None:
        mask = mask.to(x.device)
        m = mask.unsqueeze(0).float()  # (1, N)
        denom = m.sum().clamp(min=1.0) * x.shape[0]
        mae_phi = (dphi * m).sum() / denom
        mae_psi = (dpsi * m).sum() / denom
    else:
        mae_phi = dphi.mean()
        mae_psi = dpsi.mean()

    return mae_phi.item(), mae_psi.item()


def _jacobian_penalty_hutchinson(
    model,
    x: torch.Tensor,
    mask: Optional[torch.Tensor],
    num_probes: int = 1,
) -> torch.Tensor:
    """
    Approximates ||d z / d x||_F^2 using Hutchinson estimator.
    Uses random probes v, computes grad_x <z, v>, then penalizes ||grad_x||^2.

    NOTE: This is a proxy for Jacobian norm; it is typically enough to reduce
    explosive gradients in downstream MD/PLUMED use.
    """
    # x must require grad for the penalty to connect to parameters
    if not x.requires_grad:
        raise RuntimeError("x.requires_grad must be True for jacobian penalty")

    penalty = 0.0
    B = x.shape[0]

    for _ in range(int(max(1, num_probes))):
        z = model.encode(x, mask)  # (B, latent_dim)
        v = torch.randn_like(z)
        scalar = (z * v).sum() / B

        grad_x = torch.autograd.grad(
            outputs=scalar,
            inputs=x,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        penalty = penalty + (grad_x ** 2).mean()

    return penalty / float(max(1, num_probes))


def train_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    device,
    mask,
    jacobian_lambda: float = 0.0,
    jacobian_num_probes: int = 1,
):
    model.train()
    total = 0.0

    jacobian_lambda = float(jacobian_lambda)

    for batch in loader:
        batch = batch.to(device)

        # Needed for Jacobian penalty: d/dx
        if jacobian_lambda > 0.0:
            batch.requires_grad_(True)

        x_hat, _ = model(batch, mask)
        loss_rec = dihedral_loss(x_hat, batch, mask)

        loss = loss_rec
        if jacobian_lambda > 0.0:
            loss_jac = _jacobian_penalty_hutchinson(
                model=model,
                x=batch,
                mask=mask,
                num_probes=jacobian_num_probes,
            )
            loss = loss + jacobian_lambda * loss_jac

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Keep the original parameter grad clipping as a secondary safety guard
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total += loss_rec.item()  # keep reporting reconstruction loss (stable comparison)

    return total / max(1, len(loader))


@torch.no_grad()
def validate(model, loader, device, mask):
    model.eval()
    total = 0.0
    for batch in loader:
        batch = batch.to(device)
        x_hat, _ = model(batch, mask)
        total += dihedral_loss(x_hat, batch, mask).item()
    return total / max(1, len(loader))


@torch.no_grad()
def validate_with_metrics(model, loader, device, mask):
    model.eval()
    total_loss = 0.0
    total_mae_phi = 0.0
    total_mae_psi = 0.0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        x_hat, _ = model(batch, mask)

        total_loss += dihedral_loss(x_hat, batch, mask).item()
        mae_phi, mae_psi = angular_mae(x_hat, batch, mask)
        total_mae_phi += mae_phi
        total_mae_psi += mae_psi

        n_batches += 1

    n_batches = max(1, n_batches)
    return (
        total_loss / n_batches,
        total_mae_phi / n_batches,
        total_mae_psi / n_batches,
    )


@torch.no_grad()
def extract_latents(model, loader, device, mask):
    model.eval()
    latents = []
    for batch in loader:
        batch = batch.to(device)
        z = model.encode(batch, mask)
        latents.append(z.cpu())
    return torch.cat(latents, dim=0).numpy()
