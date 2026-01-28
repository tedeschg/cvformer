import torch

from pkgs.utils import sincos_to_angle_torch, circular_diff
from pkgs.model import dihedral_loss
# -----------------------------
# Metrics (angular MAE)
# -----------------------------
@torch.no_grad()
def angular_mae(x_hat: torch.Tensor, x: torch.Tensor, mask: torch.Tensor | None = None):
    """
    Returns:
      mae_phi, mae_psi in radians
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

    return float(mae_phi), float(mae_psi)


# -----------------------------
# Train / Validate
# -----------------------------
def train_epoch(model, loader, optimizer, scheduler, device, mask):
    model.train()
    total = 0.0

    for batch in loader:
        batch = batch.to(device)

        x_hat, _ = model(batch, mask)
        loss = dihedral_loss(x_hat, batch, mask)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total += loss.item()

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

