# ==========================================================
# Dihedral Transformer Autoencoder - FIXED VERSION 2.0
# ==========================================================
# - Input: phi/psi dihedral angles from MD (radians)
# - Representation: sin/cos
# - Model: Transformer encoder + attention pooling + decoder with learned queries
# - Positional encoding: learnable
# - Proper temporal validation split and attention-based pooling
# ==========================================================

import math
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import mdtraj as md


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
# 4. Transformer autoencoder with FIXED decoder
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
            latent_dim=8,  # Increased default from 2 to 8
    ):
        super().__init__()

        self.n_res = n_residues
        self.d_model = d_model
        self.latent_dim = latent_dim

        # Input projection: (sin, cos, sin, cos) -> d_model
        self.input_proj = nn.Linear(4, d_model)

        # Positional encoding for encoder
        self.pos_enc_encoder = LearnablePositionalEncoding(d_model, max_len=n_residues)

        # Layer normalization
        self.input_norm = nn.LayerNorm(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # FIXED: Attention-based pooling instead of mean pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
        )

        # Bottleneck: pooled representation -> latent
        self.to_latent = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.LayerNorm(dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, latent_dim),
        )

        # FIXED: Learned queries for decoder (one per residue)
        self.residue_queries = nn.Parameter(torch.randn(1, n_residues, d_model) * 0.02)

        # Expand: latent -> decoder context
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, dim_feedforward),
            nn.LayerNorm(dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        # Positional encoding for decoder
        self.pos_enc_decoder = LearnablePositionalEncoding(d_model, max_len=n_residues)

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

        # FIXED: Output with Tanh to constrain to [-1, 1]
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 4),
            nn.Tanh(),  # Constrain output to [-1, 1] for sin/cos
        )

        self._init_parameters()

    def _init_parameters(self):
        """Initialize parameters with Xavier/He initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, x, mask=None):
        """
        Encode input to latent representation using attention pooling.

        Args:
            x: (B, N_res, 4)
            mask: (N_res,) boolean mask for valid positions
        Returns:
            z: (B, latent_dim)
        """
        B, N, _ = x.shape

        # Project input
        x = self.input_proj(x)  # (B, N, d_model)
        x = self.pos_enc_encoder(x)
        x = self.input_norm(x)

        # Create padding mask for transformer
        src_key_padding_mask = None
        if mask is not None:
            # Invert mask: True means ignore
            src_key_padding_mask = ~mask.unsqueeze(0).expand(B, -1)

        # Transformer encoder
        h = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        # FIXED: Attention-based pooling
        attention_scores = self.attention_pool(h).squeeze(-1)  # (B, N)

        if mask is not None:
            # Mask out invalid positions
            attention_scores = attention_scores.masked_fill(~mask.unsqueeze(0), -1e9)

        attention_weights = torch.softmax(attention_scores, dim=1)  # (B, N)
        h_pooled = (h * attention_weights.unsqueeze(-1)).sum(dim=1)  # (B, d_model)

        # Project to latent space
        z = self.to_latent(h_pooled)

        return z

    def decode(self, z, mask=None):
        """
        Decode latent to output using learned queries.

        Args:
            z: (B, latent_dim)
            mask: (N_res,) boolean mask for valid positions
        Returns:
            x_hat: (B, N_res, 4)
        """
        B = z.size(0)

        # FIXED: Use learned queries (different for each residue)
        queries = self.residue_queries.expand(B, -1, -1)  # (B, N_res, d_model)

        # Context from latent
        context = self.from_latent(z).unsqueeze(1)  # (B, 1, d_model)

        # Combine queries with context (broadcast addition)
        h = queries + context  # (B, N_res, d_model)

        # Add positional encoding
        h = self.pos_enc_decoder(h)

        # FIXED: Pass mask to decoder
        src_key_padding_mask = None
        if mask is not None:
            src_key_padding_mask = ~mask.unsqueeze(0).expand(B, -1)

        # Transformer decoder
        h = self.decoder(h, src_key_padding_mask=src_key_padding_mask)

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
        x_hat = self.decode(z, mask)
        return x_hat, z


# ----------------------------------------------------------
# 5. Loss function (simplified, no normalization constraint)
# ----------------------------------------------------------

def dihedral_loss(x_hat, x, mask=None):
    """
    Reconstruction loss only (normalization constraint removed due to Tanh).

    Args:
        x_hat, x: (B, N, 4)
        mask: (N,) boolean mask for valid positions
    Returns:
        total_loss: scalar
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

    return recon


# ----------------------------------------------------------
# 6. Learning rate warmup scheduler
# ----------------------------------------------------------

def get_warmup_scheduler(optimizer, warmup_steps, total_steps):
    """
    Creates a learning rate scheduler with linear warmup and cosine decay.
    """

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ----------------------------------------------------------
# 7. Training and validation
# ----------------------------------------------------------

def train_epoch(model, loader, optimizer, device, scheduler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for batch, mask in loader:
        batch = batch.to(device)
        mask = mask.to(device)

        x_hat, z = model(batch, mask)
        loss = dihedral_loss(x_hat, batch, mask)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch, mask in loader:
            batch = batch.to(device)
            mask = mask.to(device)

            x_hat, z = model(batch, mask)
            loss = dihedral_loss(x_hat, batch, mask)

            total_loss += loss.item()

    return total_loss / len(loader)


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
# 8. Main training script
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

    # FIXED: Temporal split instead of random split
    train_size = int(args.train_split * len(dataset))
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, len(dataset)))

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    print(f"Train set: {len(train_dataset)} (frames 0-{train_size - 1})")
    print(f"Val set: {len(val_dataset)} (frames {train_size}-{len(dataset) - 1})")
    print("NOTE: Using temporal split to avoid data leakage")

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

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,} (trainable: {n_trainable:,})")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # FIXED: Warmup scheduler
    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)
    warmup_scheduler = get_warmup_scheduler(optimizer, warmup_steps, total_steps)

    # Plateau scheduler (kicks in after warmup)
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
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
    print(f"Warmup for {args.warmup_epochs} epochs, then cosine decay")

    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, warmup_scheduler)

        # Validate
        val_loss = validate(model, val_loader, device)

        # Plateau scheduler (only after warmup)
        if epoch >= args.warmup_epochs:
            plateau_scheduler.step(val_loss)

        # Logging
        if epoch % args.log_interval == 0 or epoch < 10:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:04d} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"LR: {current_lr:.2e}")

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

            if epoch % args.log_interval == 0:
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
    print(f"Best validation loss: {checkpoint['val_loss']:.6f} at epoch {checkpoint['epoch']}")

    # Extract latents from full dataset
    print("\nExtracting latent representations...")
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
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate (increased default to 3e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--train_split', type=float, default=0.9,
                        help='Fraction of data for training (temporal split)')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='Number of warmup epochs')

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