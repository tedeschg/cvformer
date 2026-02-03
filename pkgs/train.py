import torch
import torch.nn.functional as F

from pkgs.utils import sincos_to_angle_torch, circular_diff
from pkgs.model import dihedral_loss, sample_prior


# -----------------------------
# Metrics (angular MAE)
# -----------------------------
@torch.no_grad()
def angular_mae(x_hat: torch.Tensor, x: torch.Tensor, mask: torch.Tensor | None = None):
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
# Vanilla AE Train / Validate
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


# -----------------------------
# WGAN-GP pieces
# -----------------------------
def _set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad_(flag)


def gradient_penalty(critic, z_real, z_fake, device, gp_center: float = 1.0):
    """
    WGAN-GP: penalty on ||∇_z D(z)||2 close to gp_center (default 1).
    """
    B = z_real.size(0)
    eps = torch.rand(B, 1, device=device).expand_as(z_real)

    z_hat = eps * z_real + (1.0 - eps) * z_fake
    z_hat.requires_grad_(True)

    d_hat = critic(z_hat)  # (B,)
    grad = torch.autograd.grad(
        outputs=d_hat,
        inputs=z_hat,
        grad_outputs=torch.ones_like(d_hat),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]  # (B, latent_dim)

    grad_norm = grad.norm(2, dim=1)  # (B,)
    gp = ((grad_norm - gp_center) ** 2).mean()
    return gp


# -----------------------------
# AAE Train epoch: WGAN-GP in latent
# -----------------------------
def train_epoch_aae_wgangp(
    ae,
    critic,
    loader,
    opt_ae,
    sch_ae,
    opt_critic,
    sch_critic,
    device,
    mask,
    prior_kind: str = "gaussian",
    lambda_adv: float = 0.2,
    n_critic: int = 5,
    lambda_gp: float = 10.0,
    freeze_decoder_on_adv: bool = True,
):
    """
    One epoch of AAE training with WGAN-GP in latent space.

    Per batch:
      A) Recon update
      B) Critic update(s): E[D(fake)] - E[D(real)] + lambda_gp*GP
      C) Encoder adv update: minimize -E[D(fake)]
    """
    ae.train()
    critic.train()

    total_recon = 0.0
    total_crit = 0.0
    total_gen = 0.0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        B = batch.size(0)

        # ---------------------
        # A) Reconstruction update
        # ---------------------
        x_hat, _ = ae(batch, mask)
        loss_recon = dihedral_loss(x_hat, batch, mask)

        opt_ae.zero_grad()
        loss_recon.backward()
        torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
        opt_ae.step()
        sch_ae.step()

        # ---------------------
        # B) Critic update(s)
        # ---------------------
        crit_loss_acc = 0.0
        for _ in range(max(1, int(n_critic))):
            z_real = sample_prior(B, ae.latent_dim, kind=prior_kind, device=device)
            with torch.no_grad():
                z_fake = ae.encode(batch, mask)

            d_real = critic(z_real).mean()
            d_fake = critic(z_fake).mean()

            gp = gradient_penalty(critic, z_real, z_fake, device=device)
            loss_critic = (d_fake - d_real) + (lambda_gp * gp)

            opt_critic.zero_grad()
            loss_critic.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            opt_critic.step()
            sch_critic.step()

            crit_loss_acc += float(loss_critic.item())

        loss_crit_mean = crit_loss_acc / max(1, int(n_critic))

        # ---------------------
        # C) Encoder adversarial update (fool critic)
        # ---------------------
        if freeze_decoder_on_adv:
            _set_requires_grad(ae.decoder, False)

        z_fake2 = ae.encode(batch, mask)
        loss_gen = -critic(z_fake2).mean()

        opt_ae.zero_grad()
        (lambda_adv * loss_gen).backward()
        torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
        opt_ae.step()
        sch_ae.step()

        if freeze_decoder_on_adv:
            _set_requires_grad(ae.decoder, True)

        total_recon += float(loss_recon.item())
        total_crit += float(loss_crit_mean)
        total_gen += float(loss_gen.item())
        n_batches += 1

    n_batches = max(1, n_batches)
    return (
        total_recon / n_batches,
        total_crit / n_batches,
        total_gen / n_batches,
    )


# -----------------------------
# Export helpers
# -----------------------------
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
    model.eval()
    all_w = []
    for batch in loader:
        batch = batch.to(device)
        _, w = model.encode_with_attention(batch, mask)
        all_w.append(w.cpu())
    return torch.cat(all_w, dim=0).numpy()
