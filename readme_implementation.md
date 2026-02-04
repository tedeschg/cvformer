# Implementation Differences: `main` vs `guglielmo_implementation`

This document outlines the key differences between the original `main` branch and the improved `guglielmo_implementation` branch. The latter represents a significant refactoring focused on robustness, modularity, and advanced integration with PLUMED and analysis tools.

---

## 1. Architectural Changes: From Monolith to Modular
- **`main`**: A minimal, self-contained implementation. Most logic (model, dataset, training loop) was contained within a single `testit.py` file.
- **`guglielmo_implementation`**: Modularized into a package structure under `pkgs/`:
    - `pkgs/model.py`: Transformer architecture, unit-circle projection, and custom schedulers.
    - `pkgs/train.py`: Training loops, metric calculation, and latent/attention extraction.
    - `pkgs/utils.py`: Robust dihedral alignment and data loading.
    - `pkgs/plumed_export.py`: Advanced TorchScript export logic for PLUMED.
    - `pkgs/make_plumed_file.py`: Automation tool for generating `plumed.dat`.

---

## 2. Technical Enhancements

### Robust Dihedral Alignment
- **`main`**: Simple extraction of dihedrals.
- **`guglielmo_implementation`**: Implements `compute_aligned_phi_psi`. It uses MDTraj residue indices to ensure that token $i$ consistently represents the $\phi$ and $\psi$ angles of the *same* residue. It automatically generates a mask for residues missing one of the two angles (e.g., termini).

### Mask-Aware Inference
- The new implementation is fully **mask-aware**. The Transformer layers and the loss function (`dihedral_loss`) use a boolean mask to ignore invalid tokens (padding or incomplete residues) during both training and inference.

### Unit-Circle Projection
- Added a `_normalize_sincos_pairs` step in the model. This ensures that the reconstructed $\sin/\cos$ pairs for $\phi$ and $\psi$ are always projected onto the unit circle ($sin^2 + cos^2 = 1$), preventing numerical drift and improving stability.

### Improved Training Dynamics
- **Scheduler**: Replaced simple learning rate logic with a `WarmupCosineScheduler` that includes a warmup phase and a cosine decay, leading to better convergence.
- **Metrics**: Added circular angular Mean Absolute Error (MAE) for validation, providing a more physically meaningful interpretation of accuracy than raw MSE on $\sin/\cos$.

---

## 3. PLUMED Integration Evolution

| Feature | `main` | `guglielmo_implementation` |
|---------|--------|-----------------------------|
| **Export Mode** | Simple TorchScript | Multi-mode (Flat vs FromCoords) |
| **Flat Input** | Supported | Optimized (via `_FlattenEncoderPlumed`) |
| **Coord Input** | ❌ No | ✅ Supported (model computes dihedrals internally) |
| **Automation** | ❌ Manual `plumed.dat` | ✅ `make_plumed_file.py` tool |

The **"FromCoords"** mode is a major addition, allowing PLUMED to pass raw atomic coordinates to the model, which then computes the necessary dihedrals internally using embedded topology information.

---

## 4. Interpretability and Analysis
- **Attention Pooling**: The model now uses attention pooling to compress residue information into the latent space.
- **Weight Extraction**: Tools were added to extract per-frame attention weights (`attn_w.npy`), enabling the analysis of which residues are most important for defining the Collective Variables (CVs).
- **Notebooks**: A new suite of analysis notebooks was introduced in `cvformer-test/analyze/` to visualize the latent space and residue importance.

---

## 5. New Files Summary
- `pkgs/`: Core library modules.
- `cvformer-test/analyze/`: Advanced Jupyter notebooks for result interpretation.
- `environment.yml`: Standardized environment definition.
- `readme_analyze.md`: Guide for the analysis tools.
- `readme_implementation.md`: This document.
