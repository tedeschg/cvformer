# ==========================================================
# Dihedral Transformer Autoencoder
# Full, self-contained example
# ==========================================================
# - Input: phi/psi dihedral angles from MD (radians)
# - Representation: sin/cos
# - Model: Transformer encoder + CLS bottleneck + decoder
# - Positional encoding: standard sinusoidal
# ==========================================================

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import mdtraj as md

# ----------------------------------------------------------
# 1. Utilities: sin/cos conversion
# ----------------------------------------------------------

def angles_to_sincos(phi, psi):
    """
    phi, psi: (N_frames, N_res) in radians
    returns:  (N_frames, N_res, 4)
    """
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_psi = np.sin(psi)
    cos_psi = np.cos(psi)

    X = np.stack([sin_phi, cos_phi, sin_psi, cos_psi], axis=-1)
    return X


# ----------------------------------------------------------
# 2. Dataset
# ----------------------------------------------------------

class DihedralDataset(Dataset):
    def __init__(self, phi, psi):
        """
        phi, psi: numpy arrays (N_frames, N_res), radians
        """
        X = angles_to_sincos(phi, psi)
        self.X = torch.from_numpy(X).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]


# ----------------------------------------------------------
# 3. Sinusoidal positional encoding
# ----------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: (B, L, d_model)
        """
        return x + self.pe[: x.size(1)]

def extract_latent(model, dataloader, device):
    latents = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            _, z = model(batch)     # z: (B, latent_dim)
            latents.append(z.cpu())

    Z = torch.cat(latents, dim=0)
    return Z.numpy()

# ----------------------------------------------------------
# 4. Transformer autoencoder
# ----------------------------------------------------------

class DihedralTransformerAE(nn.Module):
    def __init__(
        self,
        n_residues,
        d_model=32,
        nhead=8,
        num_layers=4,
        dim_feedforward=32,
        dropout=0.1,
        latent_dim = 2,
    ):
        super().__init__()

        self.n_res = n_residues
        self.d_model = d_model

        # Input projection: (sin, cos, sin, cos) -> d_model
        self.input_proj = nn.Linear(4, d_model)

        # CLS token (learned)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.to_latent = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, latent_dim),
        )

        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Transformer decoder (parallel)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers)

        # Output projection back to sin/cos
        self.output_proj = nn.Linear(d_model, 4)

        self._init_parameters()

    def _init_parameters(self):
        nn.init.normal_(self.cls_token, std=0.02)

    # ----------------------
    # Encoder
    # ----------------------
    def encode(self, x):
        """
        x: (B, N_res, 4)
        returns: z (B, d_model)
        """
        B, N, _ = x.shape

        x = self.input_proj(x)              # (B, N, d_model)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)      # (B, N+1, d_model)

        x = self.pos_enc(x)
        h = self.encoder(x)

        cls_embed = h[:, 0]                 # (B, d_model)
        z = self.to_latent(cls_embed)       # (B, latent_dim)
        return z

    # ----------------------
    # Decoder
    # ----------------------
    def decode(self, z):
        """
        z: (B, d_model)
        returns: x_hat (B, N_res, 4)
        """
        B = z.size(0)

        cls_embed = self.from_latent(z)   # (B, d_model)

        x = cls_embed.unsqueeze(1).expand(B, self.n_res, self.d_model)
        x = self.pos_enc(x)

        h = self.decoder(x)
        x_hat = self.output_proj(h)
        return x_hat

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


# ----------------------------------------------------------
# 5. Loss function
# ----------------------------------------------------------

def dihedral_loss(x_hat, x, lambda_norm=0.1):
    """
    x_hat, x: (B, N, 4)
    """
    recon = ((x_hat - x) ** 2).mean()

    sin_phi, cos_phi = x_hat[..., 0], x_hat[..., 1]
    sin_psi, cos_psi = x_hat[..., 2], x_hat[..., 3]

    norm = (
        (sin_phi**2 + cos_phi**2 - 1) ** 2
        + (sin_psi**2 + cos_psi**2 - 1) ** 2
    ).mean()

    return recon + lambda_norm * norm


class FlattenEncoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        # x shape: [batch, features] where features = 4 * n_res
        B = x.size(0)
        x = x.view(B, encoder.n_res, 4)
        return self.encoder(x)

# ----------------------------------------------------------
# 6. Example training loop
# ----------------------------------------------------------

if __name__ == "__main__":
    traj = md.load("traj_skip100NoH.xtc", top="2JOF-0-proteinNoH.pdb")

    phi_idx, phi = md.compute_phi(traj)
    psi_idx, psi = md.compute_psi(traj)

    n_res = phi.shape[1]

    # Zero-fill terminals if desired
    #phi[:, 0] = 0.0
    #psi[:, -1] = 0.0

    dataset = DihedralDataset(phi, psi)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # ------------------------------------------------------
    # Model
    # ------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #device = "cpu"

    model = DihedralTransformerAE(
        n_residues=n_res,
        d_model=32,
        nhead=8,
        num_layers=4,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=2.0e-4)

    # ------------------------------------------------------
    # Training
    # ------------------------------------------------------
    n_epochs = 5000

    for epoch in range(n_epochs):
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device)

            x_hat, z = model(batch)
            loss = dihedral_loss(x_hat, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch:03d} | loss = {avg_loss:.6f}")

    # ------------------------------------------------------
    # After training: latent representations
    # ------------------------------------------------------
    with torch.no_grad():
        x0 = dataset[0].unsqueeze(0).to(device)
        _, z0 = model(x0)
        print("Latent vector shape:", z0.shape)

        loader2 = DataLoader(dataset, batch_size=1, shuffle=False)
        z = extract_latent(model, loader2, device)
        np.savetxt("latent.txt", z)

    encoder_state = {
        "input_proj": model.input_proj.state_dict(),
        "cls_token": model.cls_token,
        "pos_enc": model.pos_enc.state_dict(),
        "encoder": model.encoder.state_dict(),
        "to_latent": model.to_latent.state_dict(),
        "config": {
            "n_residues": model.n_res,
            "d_model": model.d_model,
            "latent_dim": model.to_latent[-1].out_features,
            "nhead": model.encoder.layers[0].self_attn.num_heads,
            "num_layers": len(model.encoder.layers),
        },
    }

    torch.save(encoder_state, "dihedral_encoder.pt")

    flat_encoder = FlattenEncoder(encoder)
    scripted_encoder = torch.jit.trace(flat_encoder, torch.zeros(1, 4*encoder.n_res))
    scripted_encoder.save("dihedral_encoder_plumed.pt")

