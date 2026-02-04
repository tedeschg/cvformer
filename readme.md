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
│   ├── train.py                   # Training loop & metrics
│   ├── utils.py                   # Preprocessing & alignment
│   ├── plumed_export.py           # TorchScript export for PLUMED
│   └── make_plumed_file.py        # Utility to generate plumed.dat
├── testit.py                      # Entry point for training/evaluation
├── environment.yml                # Environment dependencies
├── cvformer-test/analyze/         # Analysis notebooks & plots
├── cvformer-test/                 # Example data and test runs
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
Handles model export for PLUMED (two modes):
- **`export_plumed_encoder()`**: PLUMED provides a flat input vector `[sinφ0, cosφ0, sinψ0, cosψ0, ...]`. Exports the TorchScript model (`.pt`) and `plumed_info.json` with metadata (mask, residue IDs, input format).
- **`export_plumed_encoder_from_coords()`**: PLUMED provides ONLY atomic coordinates (`ATOMS=...`). The wrapper computes φ/ψ (sin/cos) internally from the topology, applies the mask, and feeds tokens to the model. Useful to avoid explicit `TORSION` blocks in `plumed.dat`.
- **Internal wrappers** ensure reshaping/masking consistent with training.

### `pkgs/make_plumed_file.py`
Automation tool:
- Generates a ready-to-use `plumed.dat` by mapping training residues to the target MD topology.
- Supports METAD block generation using training latent ranges.

### `pkgs/utils.py` & `pkgs/train.py`
- **Robust alignment**: Ensures token i consistently represents the same residue for both φ and ψ.
- **Circular metrics**: Angular MAE for validation.
- **Temporal split**: Default 90/10 split to prevent data leakage in MD trajectories.

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

### 2. Export for PLUMED
- Mode A (sin/cos input): use `export_plumed_encoder` during/after training.
- Mode B (coordinates only): use `export_plumed_encoder_from_coords` providing consistent topologies.

### 3. Generate plumed.dat
After the export, generate `plumed.dat` with:
```bash
python /home/tedeschg/prj/cvformer/pkgs/make_plumed_file.py \
  --top train.pdb \
  --traj train.xtc \
  --plumed_top npt.gro \
  --mode flat_sincos \
  --pt dihedral_encoder_plumed.pt \
  --info plumed_info.json \
  --latents latents.npy \
  --whole_entity0 1-272 \
  --out plumed_legacy.dat

python /home/tedeschg/prj/cvformer/pkgs/make_plumed_file.py \
  --mode coords \
  --pt dihedral_encoder_fromcoords_plumed.pt \
  --info plumed_info_fromcoords.json \
  --latents latents.npy \
  --whole_entity0 1-272 \
  --out plumed_coords.dat
```

---

## Generated Output
In the `output/` directory:
- `best_model.pt`: Full training checkpoint.
- `dihedral_encoder_plumed.pt`: TorchScript model for PLUMED (or the `fromcoords` variant).
- `plumed_info.json`: Metadata (residue map, mask, notes, input/output shapes).
- `latents.npy/txt`: CVs extracted for the input trajectory.
- `token_residue_ids.txt`: Mapping of model tokens to topology residue indices.
- Optional (if attention dump is enabled) `attn_w.npy`, `attn_w_mean.txt`: attention pooling weights for further analysis.

---

## 🔧 PLUMED Integration

Two integration options via the `PYTORCH_MODEL` module:
- Mode A — flat sin/cos input: `[sin(φ₀), cos(φ₀), sin(ψ₀), cos(ψ₀), ...]` (use `export_plumed_encoder`). Requires defining `TORSION` and `MATHEVAL` in `plumed.dat` to generate the flat array. The `make_plumed_file.py` tool automates these definitions.
- Mode B — coordinates only: PLUMED passes atomic coordinates and the TorchScript wrapper computes φ/ψ internally (use `export_plumed_encoder_from_coords`). This reduces verbosity and the risk of topology mismatch.

---

## Key Features
- ✅ **Mask-aware inference**: Automatically ignores residues without valid φ/ψ pairs (e.g., termini).
- ✅ **Unit-circle projection**: Prevents numerical instability by normalizing sin/cos outputs.
- ✅ **Topology mapping**: Safely maps training residue indices to different MD topologies.
- ✅ **Metadynamics-ready**: Computes SIGMA and GRID parameters from training latent distributions.

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

## 📊 Analysis: available notebooks

See `cvformer-test/analyze/` for ready-to-use examples:
- `analyze_latent.ipynb`: explores CV distributions (histograms, correlations, clustering) from `latents.npy`.
- `residue_importance.ipynb`: analyzes residue importance from attention weights (`attn_w.npy`) with robust statistics, state-wise clustering, and transition analysis.

A complete guide to the notebooks is available in `cvformer-test/analyze/readme_analyze.md`.

For a detailed technical comparison of the implementation changes in this branch compared to the original version, see `readme_implementation.md`.

---

## 📧 Support

For bugs or feature requests, please open an issue on GitHub.
