# ==========================================================
# Dihedral Transformer Autoencoder - IMPROVED VERSION
# ==========================================================
# - Input: phi/psi dihedral angles from MD (radians)
# - Representation: sin/cos
# - Model: Transformer encoder + pooling + decoder
# - Positional encoding: learnable
# - Proper validation, checkpointing, and logging
# ==========================================================

import math
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import mdtraj as md
from tqdm import tqdm


# Set seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# ----------------------------------------------------------
# 1. Utilities: sin/cos conversion
# ----------------------------------------------------------

def angles_to_sincos(phi, psi):
    """
    Convert dihedral angles to sin/cos representation.

    Args:
        phi, psi: (N_frames, N_res) in radians
    Returns:
        X: (N_frames, N_res, 4) [sin_phi, cos_phi, sin_psi, cos_psi]
    """
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_psi = np.sin(psi)
    cos_psi = np.cos(psi)

    X = np.stack([sin_phi, cos_phi, sin_psi, cos_psi], axis=-1)
    return X


def mask_undefined_angles(phi, psi):
    """
    Mask undefined terminal angles instead of zero-filling.

    Args:
        phi, psi: (N_frames, N_res) in radians
    Returns:
        phi, psi with NaN for undefined angles
        mask: boolean array (N_res,) indicating valid positions
    """
    # Create mask: True for valid positions
    mask = np.ones(phi.shape[1], dtype=bool)
    mask[0] = False  # First phi is undefined
    mask[-1] = False  # Last psi is undefined

    return phi, psi, mask


# ----------------------------------------------------------
# 2. Dataset
# ----------------------------------------------------------

class DihedralDataset(Dataset):
    def __init__(self, phi, psi, mask=None):
        """
        Args:
            phi, psi: numpy arrays (N_frames, N_res), radians
            mask: boolean array (N_res,) for valid positions
        """
        X = angles_to_sincos(phi, psi)
        self.X = torch.from_numpy(X).float()

        if mask is not None:
            self.mask = torch.from_numpy(mask).bool()
        else:
            self.mask = torch.ones(X.shape[1], dtype=torch.bool)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.mask


# ----------------------------------------------------------
# 3. Learnable positional encoding
# ----------------------------------------------------------

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x):
        """
        Args:
            x: (B, L, d_model)
        Returns:
            x + pe: (B, L, d_model)
        """
        return x + self.pe[:, :x.size(1), :]


# ----------------------------------------------------------
# 4. Transformer autoencoder (PROPERLY IMPLEMENTED)
# ----------------------------------------------------------

class DihedralTransformerAE(nn.Module):
    def __init__(
            self,
            n_residues,
            d_model=64,
            nhead=8,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=256,
            dropout=0.1,
            latent_dim=2,
    ):
        super().__init__()

        self.n_res = n_residues
        self.d_model = d_model
        self.latent_dim = latent_dim

        # Input projection: (sin, cos, sin, cos) -> d_model
        self.input_proj = nn.Linear(4, d_model)

        # Positional encoding
        self.pos_enc = LearnablePositionalEncoding(d_model, max_len=n_residues)

        # Layer normalization
        self.input_norm = nn.LayerNorm(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm for better training
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # Bottleneck: sequence -> latent
        self.to_latent = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.LayerNorm(dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, latent_dim),
        )

        # Expand: latent -> sequence initialization
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, dim_feedforward),
            nn.LayerNorm(dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        # Transformer decoder
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_decoder_layers)

        # Output projection back to sin/cos
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 4),
        )

        self._init_parameters()

    def _init_parameters(self):
        """Initialize parameters with Xavier/He initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, x, mask=None):
        """
        Encode input to latent representation.

        Args:
            x: (B, N_res, 4)
            mask: (N_res,) boolean mask for valid positions
        Returns:
            z: (B, latent_dim)
        """
        B, N, _ = x.shape

        # Project input
        x = self.input_proj(x)  # (B, N, d_model)
        x = self.pos_enc(x)
        x = self.input_norm(x)

        # Create padding mask for transformer
        src_key_padding_mask = None
        if mask is not None:
            # Invert mask: True means ignore
            src_key_padding_mask = ~mask.unsqueeze(0).expand(B, -1)

        # Transformer encoder
        h = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Global pooling (masked mean)
        if mask is not None:
            mask_expanded = mask.unsqueeze(0).unsqueeze(-1).float()
            h_masked = h * mask_expanded
            h_pooled = h_masked.sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            h_pooled = h.mean(dim=1)

        # Project to latent space
        z = self.to_latent(h_pooled)

        return z

    def decode(self, z):
        """
        Decode latent to output.

        Args:
            z: (B, latent_dim)
        Returns:
            x_hat: (B, N_res, 4)
        """
        B = z.size(0)

        # Expand latent to sequence
        h = self.from_latent(z)  # (B, d_model)
        h = h.unsqueeze(1).expand(B, self.n_res, self.d_model)

        # Add positional encoding
        h = self.pos_enc(h)

        # Transformer decoder
        h = self.decoder(h)

        # Project to output
        x_hat = self.output_proj(h)

        return x_hat

    def forward(self, x, mask=None):
        """
        Forward pass.

        Args:
            x: (B, N_res, 4)
            mask: (N_res,) boolean mask
        Returns:
            x_hat: (B, N_res, 4)
            z: (B, latent_dim)
        """
        z = self.encode(x, mask)
        x_hat = self.decode(z)
        return x_hat, z


# ----------------------------------------------------------
# 5. Loss function
# ----------------------------------------------------------

def dihedral_loss(x_hat, x, mask=None, lambda_norm=0.1):
    """
    Combined reconstruction and normalization loss.

    Args:
        x_hat, x: (B, N, 4)
        mask: (N,) boolean mask for valid positions
        lambda_norm: weight for normalization constraint
    Returns:
        total_loss: scalar
        recon_loss: scalar (for logging)
        norm_loss: scalar (for logging)
    """
    # Apply mask if provided
    if mask is not None:
        mask_expanded = mask.unsqueeze(0).unsqueeze(-1)
        x_hat_masked = x_hat * mask_expanded
        x_masked = x * mask_expanded
        n_valid = mask.sum()
    else:
        x_hat_masked = x_hat
        x_masked = x
        n_valid = x.shape[1]

    # Reconstruction loss
    recon = ((x_hat_masked - x_masked) ** 2).sum() / (n_valid * x.shape[0])

    # Normalization constraint: sin^2 + cos^2 = 1
    sin_phi, cos_phi = x_hat[..., 0], x_hat[..., 1]
    sin_psi, cos_psi = x_hat[..., 2], x_hat[..., 3]

    norm_phi = (sin_phi ** 2 + cos_phi ** 2 - 1) ** 2
    norm_psi = (sin_psi ** 2 + cos_psi ** 2 - 1) ** 2

    if mask is not None:
        norm = (norm_phi + norm_psi) * mask.unsqueeze(0)
        norm = norm.sum() / (n_valid * x.shape[0])
    else:
        norm = (norm_phi + norm_psi).mean()

    total_loss = recon + lambda_norm * norm

    return total_loss, recon, norm


# ----------------------------------------------------------
# 6. Training and validation
# ----------------------------------------------------------

def train_epoch(model, loader, optimizer, device, lambda_norm=0.1):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_norm = 0.0

    for batch, mask in loader:
        batch = batch.to(device)
        mask = mask.to(device)

        x_hat, z = model(batch, mask)
        loss, recon, norm = dihedral_loss(x_hat, batch, mask, lambda_norm)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon.item()
        total_norm += norm.item()

    n_batches = len(loader)
    return total_loss / n_batches, total_recon / n_batches, total_norm / n_batches


def validate(model, loader, device, lambda_norm=0.1):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_norm = 0.0

    with torch.no_grad():
        for batch, mask in loader:
            batch = batch.to(device)
            mask = mask.to(device)

            x_hat, z = model(batch, mask)
            loss, recon, norm = dihedral_loss(x_hat, batch, mask, lambda_norm)

            total_loss += loss.item()
            total_recon += recon.item()
            total_norm += norm.item()

    n_batches = len(loader)
    return total_loss / n_batches, total_recon / n_batches, total_norm / n_batches


def extract_latents(model, loader, device):
    """Extract latent representations."""
    model.eval()
    latents = []

    with torch.no_grad():
        for batch, mask in loader:
            batch = batch.to(device)
            mask = mask.to(device)

            z = model.encode(batch, mask)
            latents.append(z.cpu())

    return torch.cat(latents, dim=0).numpy()


# ----------------------------------------------------------
# 7. Main training script
# ----------------------------------------------------------

def main(args):
    set_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load trajectory
    print(f"Loading trajectory: {args.trajectory}")
    traj = md.load(args.trajectory, top=args.topology)
    print(f"Loaded {traj.n_frames} frames, {traj.n_residues} residues")

    # Compute dihedrals
    phi_idx, phi = md.compute_phi(traj)
    psi_idx, psi = md.compute_psi(traj)

    n_res = phi.shape[1]
    print(f"Computed dihedrals for {n_res} residues")

    # Mask undefined angles
    phi, psi, mask = mask_undefined_angles(phi, psi)
    print(f"Valid residues (masked): {mask.sum()}/{len(mask)}")

    # Create dataset
    dataset = DihedralDataset(phi, psi, mask)

    # Train/validation split
    train_size = int(args.train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Train set: {len(train_dataset)}, Val set: {len(val_dataset)}")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Model
    model = DihedralTransformerAE(
        n_residues=n_res,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        latent_dim=args.latent_dim,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=args.scheduler_patience,
        verbose=True,
    )

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0

    print("\nStarting training...")
    for epoch in range(args.epochs):
        # Train
        train_loss, train_recon, train_norm = train_epoch(
            model, train_loader, optimizer, device, args.lambda_norm
        )

        # Validate
        val_loss, val_recon, val_norm = validate(
            model, val_loader, device, args.lambda_norm
        )

        # Scheduler step
        scheduler.step(val_loss)

        # Logging
        if epoch % args.log_interval == 0:
            print(f"Epoch {epoch:04d} | "
                  f"Train Loss: {train_loss:.6f} (R: {train_recon:.6f}, N: {train_norm:.6f}) | "
                  f"Val Loss: {val_loss:.6f} (R: {val_recon:.6f}, N: {val_norm:.6f})")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': vars(args),
            }, output_dir / 'best_model.pt')

            print(f"  → Saved best model (val_loss: {val_loss:.6f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Load best model
    print("\nLoading best model...")
    checkpoint = torch.load(output_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Extract latents from full dataset
    print("Extracting latent representations...")
    full_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    latents = extract_latents(model, full_loader, device)

    np.savetxt(output_dir / 'latents.txt', latents)
    print(f"Saved latents to {output_dir / 'latents.txt'}")
    print(f"Latent shape: {latents.shape}")

    print("\nTraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Dihedral Transformer Autoencoder')

    # Data
    parser.add_argument('--trajectory', type=str, required=True,
                        help='Path to trajectory file (.xtc, .dcd, etc.)')
    parser.add_argument('--topology', type=str, required=True,
                        help='Path to topology file (.pdb, .gro, etc.)')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Output directory for results')

    # Model
    parser.add_argument('--d_model', type=int, default=64,
                        help='Model dimension')
    parser.add_argument('--nhead', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--num_encoder_layers', type=int, default=3,
                        help='Number of encoder layers')
    parser.add_argument('--num_decoder_layers', type=int, default=3,
                        help='Number of decoder layers')
    parser.add_argument('--dim_feedforward', type=int, default=256,
                        help='Feedforward dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--latent_dim', type=int, default=2,
                        help='Latent space dimension')

    # Training
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Maximum number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--lambda_norm', type=float, default=0.1,
                        help='Weight for normalization loss')
    parser.add_argument('--train_split', type=float, default=0.9,
                        help='Fraction of data for training')

    # Optimization
    parser.add_argument('--patience', type=int, default=100,
                        help='Early stopping patience')
    parser.add_argument('--scheduler_patience', type=int, default=20,
                        help='Learning rate scheduler patience')

    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loader workers')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Logging interval')

    args = parser.parse_args()
    main(args)

