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
├── pkgs/                          # Core package modules
│   ├── model.py                   # Transformer architecture & loss
│   ├── train.py                   # Training loops & metrics
│   ├── utils.py                   # Preprocessing & alignment
│   ├── plumed_export.py           # TorchScript export logic
│   └── make_plumed_file.py        # Utility to generate plumed.dat
├── testit.py                      # Main training & evaluation entry point
├── environment.yml                # Environment dependencies
├── analyze/                       # Analysis notebooks & plots
├── test02/                        # Example data & test runs
└── readme.md                      # This documentation
```

---

## Package Contents

### `pkgs/model.py`
Core architecture:
- **`DihedralTransformerAE`**: Transformer Autoencoder with attention pooling and sin/cos normalization.
- **`SinusoidalPositionalEncoding`**: Fixed positional embeddings.
- **`dihedral_loss()`**: Mask-aware MSE on unit-circle projections.
- **`WarmupCosineScheduler`**: Learning rate scheduling.

### `pkgs/plumed_export.py`
Handles model production:
- **`export_plumed_encoder()`**: Traces the encoder into TorchScript (`.pt`) and generates `plumed_info.json` with metadata (mask, residue IDs, input format).
- **`FlattenEncoderPlumed`**: Internal wrapper that ensures PLUMED inputs are correctly reshaped and masked.

### `pkgs/make_plumed_file.py`
Automation tool:
- Generates a ready-to-use `plumed.dat` by mapping training residues to the target MD topology.
- Supports METAD block generation using training latent ranges.

### `pkgs/utils.py` & `pkgs/train.py`
- **Robust alignment**: Ensures token $i$ consistently represents the same residue for both $\phi$ and $\psi$.
- **Circular Metrics**: Angular MAE for validation.
- **Temporal Split**: Default 90/10 split to prevent data leakage in MD trajectories.

---

## Usage

### 1. Training
```bash
python testit.py \
  --trajectory traj.xtc \
  --topology topol.pdb \
  --output_dir output \
  --latent_dim 2 \
  --epochs 1000
```

### 2. PLUMED Setup
After training, use the utility script to generate the `plumed.dat`:
```bash
python pkgs/make_plumed_file.py \
  --top training_top.pdb \
  --plumed_top md_top.pdb \
  --pt output/dihedral_encoder_plumed.pt \
  --info output/plumed_info.json \
  --latents output/latents.npy \
  --out plumed.dat
```

---

## Generated Output
In the `output/` directory:
- `best_model.pt`: Full training checkpoint.
- `dihedral_encoder_plumed.pt`: Optimized TorchScript model for PLUMED.
- `plumed_info.json`: Metadata (residue mapping, mask, notes).
- `latents.npy/txt`: Extracted CVs for the input trajectory.
- `token_residue_ids.txt`: Mapping of model tokens to topology residue indices.

---

## 🔧 PLUMED Integration

The exported model integrates with PLUMED via the `PYTORCH_MODEL` module. The input expected is a flat array of:
`[sin(φ₀), cos(φ₀), sin(ψ₀), cos(ψ₀), sin(φ₁), ...]`

The `make_plumed_file.py` script automates the creation of all necessary `TORSION` and `MATHEVAL` definitions required to feed the model correctly.

---

## Key Features
- ✅ **Mask-Aware Inference**: Automatically ignores residues without valid $\phi/\psi$ pairs (e.g., termini).
- ✅ **Unit Circle Projection**: Prevents numerical instability by normalizing sin/cos outputs.
- ✅ **Topology Mapping**: Safely maps training residue indices to different MD topologies.
- ✅ **Metadynamics Ready**: Calculates SIGMA and GRID parameters from training latent distributions.

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
