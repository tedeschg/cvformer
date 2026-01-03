# CVFormer - Collective Variables from Transformer Autoencoder

Transformer-based autoencoder for extracting collective variables (CVs) from φ/ψ dihedral angles of MD trajectories, with PLUMED export support.

---

## Installation

```bash
mamba env create -f environment.yml
mamba activate cvformer
```

---

## Repository Structure

```
cvformer/
├── pkgs/                          # Modular package
│   ├── model.py                   # Model architecture
│   ├── train.py                   # Training loops and metrics
│   └── utils.py                   # Data utilities and preprocessing
├── testit.py                      # Main training script
├── environment.yml                # Conda/mamba dependencies
├── test9/                         # Example output
│   ├── dihedral_encoder.pt        # Saved encoder model
│   └── latent.txt                 # Extracted latents
├── .gitignore                     # Files to ignore
└── readme.md                      # This file
```

---

## Package Contents

### `pkgs/model.py`
Contains the Transformer Autoencoder architecture:

- **`DihedralTransformerAE`**: Main model class
  - Transformer encoder with attention pooling
  - Latent bottleneck (default 2D)
  - Transformer decoder with learned per-residue queries
  - Sin/cos normalization to unit circle

- **`SinusoidalPositionalEncoding`**: Sinusoidal positional encoding (non-learnable)

- **`dihedral_loss()`**: MSE loss on sin/cos representation with masking support

- **`WarmupCosineScheduler`**: Linear warmup + cosine decay scheduler

### `pkgs/train.py`
Functions for training, validation, and extraction:

- **`train_epoch()`**: Single epoch training with gradient clipping
- **`validate()`**: Simple validation (loss only)
- **`validate_with_metrics()`**: Validation with angular metrics (MAE φ/ψ)
- **`angular_mae()`**: Circular Mean Absolute Error for dihedral angles
- **`extract_latents()`**: Extract latent representations from full dataset

### `pkgs/utils.py`
Data preprocessing and management utilities:

- **`angles_to_sincos()`**: Convert angles → sin/cos representation
- **`sincos_to_angle_torch()`**: Convert sin/cos → angles (atan2)
- **`circular_diff()`**: Circular difference between angles (wrap [-π, π])
- **`compute_aligned_phi_psi()`**: Robust φ/ψ alignment by residue from MDTraj
- **`DihedralDataset`**: PyTorch Dataset for dihedral angles

### `testit.py`
Main integrated script for:

1. **Trajectory loading** with MDTraj
2. **Robust preprocessing** of φ/ψ angles (per-residue alignment)
3. **Training** with temporal split, early stopping, warmup+cosine scheduler
4. **Validation** with angular metrics (MAE)
5. **Latent extraction** from full dataset
6. **PLUMED EXPORT**:
   - `FlattenEncoder` class (wrapper for flat PLUMED input)
   - TorchScript tracing → `dihedral_encoder_plumed.pt`
   - JSON metadata (`plumed_info.json`) with residue and dimension info

---

## Usage

### Basic Training
```bash
python testit.py \
  --trajectory traj.xtc \
  --topology topol.pdb \
  --output_dir output \
  --latent_dim 2 \
  --epochs 1000 \
  --batch_size 64
```

### Generated Output
In the `output/` directory:

- **Model files**:
  - `best_model.pt` - Full checkpoint of best model
  - `dihedral_encoder_plumed.pt` - **Encoder for PLUMED** (TorchScript)
  - `plumed_info.json` - Metadata for PLUMED integration

- **Latents**:
  - `latents.npy` / `latents.txt` - Latent representations (n_frames × latent_dim)
  - `token_residue_ids.txt` - Mapping token → topology residue index

- **Training history**:
  - `train_losses.txt`, `val_losses.txt`
  - `mae_phi.txt`, `mae_psi.txt`
  - `config.txt` - Training configuration

---

## 🔧 PLUMED Integration

The `dihedral_encoder_plumed.pt` file can be used in PLUMED with the PyTorch module:

```plumed
# Compute φ/ψ angles
phi1: TORSION ATOMS=...
psi1: TORSION ATOMS=...
...

# Use encoder as CV (requires PLUMED-PyTorch interface)
# Input: flat vector [sin(φ₁), cos(φ₁), sin(ψ₁), cos(ψ₁), ...]
# Output: latent_dim CVs
```

The model expects:
- **Input shape**: `(batch, 4 × n_tokens)` flat array
- **Output shape**: `(batch, latent_dim)`
- Corresponding residues are listed in `token_residue_ids.txt`

---

## Key Features

✅ Robust per-residue φ/ψ alignment (handles terminals)
✅ Sin/cos representation with unit circle normalization
✅ Temporal split (prevents data leakage)
✅ Warmup + cosine decay scheduler
✅ Early stopping with patience
✅ Angular metrics (circular MAE)
✅ **TorchScript export for PLUMED**
✅ Gradient clipping, dropout, weight decay

---

## 🔬 Architecture

```
Input: (batch, n_residues, 4)  [sin φ, cos φ, sin ψ, cos ψ]
   ↓
Transformer Encoder (multi-head attention)
   ↓
Attention Pooling
   ↓
Latent Space (default 2D)
   ↓
Transformer Decoder (learned queries)
   ↓
Output: (batch, n_residues, 4)  [reconstructed]
```

---

## 📝 Main Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--d_model` | 64 | Embedding dimension |
| `--nhead` | 8 | Number of attention heads |
| `--num_encoder_layers` | 3 | Encoder layers |
| `--num_decoder_layers` | 3 | Decoder layers |
| `--latent_dim` | 2 | Latent space dimension |
| `--lr` | 3e-4 | Learning rate (peak) |
| `--warmup_epochs` | 10 | Warmup epochs |
| `--patience` | 100 | Early stopping patience |

---

## 🐛 Known Issues / TODO

- [ ] Test full PLUMED integration
- [ ] Add complete PLUMED input file example
- [ ] Support for multi-chain systems
- [ ] Variational autoencoder (VAE) variant

---

## 📧 Support

For bugs or feature requests, please open an issue on GitHub.
