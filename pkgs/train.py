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
# Contrastive loss (SimCLR-style)
# -----------------------------
def _normalize_z(z: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(z, dim=1, eps=1e-8)


def _nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    NT-Xent loss over a batch of positive pairs (z1[i], z2[i]).
    """
    if z1.size(0) <= 1:
        return torch.zeros((), device=z1.device)

    z1 = _normalize_z(z1)
    z2 = _normalize_z(z2)

    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    sim = torch.matmul(z, z.T) / max(temperature, 1e-8)  # (2B, 2B)

    # mask self-similarity
    mask = torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
    sim = sim.masked_fill(mask, float("-inf"))

    B = z1.size(0)
    targets = torch.arange(B, device=sim.device)
    targets = torch.cat([targets + B, targets], dim=0)  # positives indices

    return torch.nn.functional.cross_entropy(sim, targets)


def _augment_sincos(x: torch.Tensor, noise_std: float = 0.05) -> torch.Tensor:
    """
    Add small noise to sin/cos pairs and re-normalize to unit circle.
    x: (B, N, 4)
    """
    if noise_std <= 0:
        return x
    noise = torch.randn_like(x) * noise_std
    x_noisy = x + noise
    # re-normalize phi and psi pairs
    phi = x_noisy[..., 0:2]
    psi = x_noisy[..., 2:4]
    phi = phi / (phi.norm(p=2, dim=-1, keepdim=True) + 1e-8)
    psi = psi / (psi.norm(p=2, dim=-1, keepdim=True) + 1e-8)
    return torch.cat([phi, psi], dim=-1)


# -----------------------------
# Train / Validate
# -----------------------------
def train_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    device,
    mask,
    contrastive_weight: float = 0.0,
    contrastive_temp: float = 0.1,
    contrastive_noise: float = 0.05,
):
    model.train()
    total = 0.0

    for batch in loader:
        batch = batch.to(device)

        x_hat, _ = model(batch, mask)
        loss = dihedral_loss(x_hat, batch, mask)

        if contrastive_weight > 0.0:
            x1 = batch
            x2 = _augment_sincos(batch, noise_std=contrastive_noise)
            z1 = model.encode(x1, mask)
            z2 = model.encode(x2, mask)
            c_loss = _nt_xent_loss(z1, z2, temperature=contrastive_temp)
            loss = loss + contrastive_weight * c_loss

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


@torch.no_grad()
def extract_attention_weights(model, loader, device, mask):
    """
    Returns:
      W: (n_frames, n_tokens) attention pooling weights per frame (sum=1 per frame over valid tokens).
    """
    model.eval()
    all_w = []
    for batch in loader:
        batch = batch.to(device)
        _, w = model.encode_with_attention(batch, mask)
        all_w.append(w.cpu())
    return torch.cat(all_w, dim=0).numpy()
