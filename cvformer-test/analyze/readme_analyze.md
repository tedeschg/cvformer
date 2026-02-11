# Analysis notebooks guide

This document clearly and succinctly explains what the notebooks in `cvformer-test/analyze/` do and how to use them. The notebooks are designed to analyze results produced by CVFormer training: the latent space (`latents.npy`) and, if enabled, attention pooling weights (`attn_w.npy`).

---

## Common prerequisites
- Have an output directory from a run (e.g., `cvformer-test/test04/output/`) containing:
  - `latents.npy` (T × d): CVs for each frame.
  - `token_residue_ids.txt` (N): residue index associated with each token.
  - Optional for attention analysis: `attn_w.npy` (T × N) and `attn_w_mean.txt`.
- Set the `OUTPUT_DIR` variable in the notebook to the correct folder.

Tip: if the trajectory has a large stride or strong autocorrelation, prefer longer windows/parameters in methods that use block bootstrap.

---

## Notebook: `analyze_latent.ipynb`
Analyzes the distribution of CVs in `latents.npy` and produces plots and indicators useful to understand the latent manifold.

### Main sections
1. Configuration
   - Sets `OUTPUT_DIR` and reads `latents.npy`.
   - Applies basic normalizations (centering/scaling) when helpful for plots.

2. Data loading
   - Loads `latents.npy` (expected shape: T × d) and shows shape and summary stats (min, max, mean, std) for each latent dimension.

3. Quick exploration
   - 2D/3D scatter plots of the CVs (if `d ≥ 2/3`).
   - Histograms and densities for each latent dimension.
   - Time series of CVs to highlight stability, drift, or state jumps.

4. Optional clustering
   - Runs simple clustering (e.g., KMeans or GMM) on normalized CVs.
   - Estimates number of clusters via heuristics (silhouette/elbow) when present, or uses a predefined `n_clusters`.
   - Colors scatter plots by cluster assignment.

5. Output
   - Saved figures (scatters, histograms, CV timelines).
   - Optional file with per-frame cluster labels (useful for subsequent analyses).

### How to interpret
- Dense point clouds/clusters suggest metastable basins.
- CVs with very different variances may require rescaling for metadynamics.

---

## Notebook: `residue_importance.ipynb`
Performs a robust, MD-aware analysis of residue importance from attention pooling weights `w(t, i)`.

### Required files
- `attn_w.npy` (T × N): attention weights per frame (rows) and token/residue (columns).
- `token_residue_ids.txt` (N): token → residue index mapping.
- `latents.npy` (T × d): required for state-wise clustering and transition analysis.

### Main sections
1. Configuration
   - Sets `OUTPUT_DIR`, parameters for block bootstrap (`BLOCK_LEN`), and clustering choices (`CLUSTER_METHOD`, `N_CLUSTERS`).
   - Defines the rare-state threshold (e.g., `RARE_FRACTION_THRESHOLD`) and pre-entry windows (`PRE_ENTRY_WINDOW`, `REFRACTORY`).

2. Loading and sanity checks
   - Loads `attn_w.npy`, `token_residue_ids.txt`, `latents.npy`.
   - Checks shapes (T, N), cross-file consistency, and that each `W` row sums ≈ 1.
   - Normalizes CVs for clustering (z-score).

3. Global per-residue statistics
   - Computes, for each residue: mean, median, std, percentiles (5/25/75/95), IQR.
   - Derives an “episodic” score (p95/mean) to highlight rare spikes.
   - Exports `attn_residue_stats_global.csv`.

4. Interpretability metrics
   - Duty cycle: fraction of frames where a residue is “dominant” (above a threshold, typically 2× uniform attention).
   - Expected rank: average per-frame rank of the residue.
   - Cumulative contribution: how many residues explain 80–95% of mean attention.

5. Uncertainty via block bootstrap
   - Estimates 95% confidence intervals for mean/median accounting for MD autocorrelation.
   - Exports `attn_residue_stats_global_with_ci.csv`.

6. State-wise analysis (clustering in CVs)
   - Clusters CVs (KMeans or GMM) and identifies rare states (population < threshold).
   - Computes per-state per-residue statistics and saves per-state means (`attn_state_means.npy`).

7. Transition analysis towards rare states
   - Detects entries into rare clusters and, for each entry, computes mean attention in the pre-entry window.
   - Aggregates windows per residue and produces rankings of residues potentially involved in entries.
   - Exports summary CSVs (e.g., `attn_pre_entry_top_residues.csv`).

8. Outputs and figures
   - CSVs with global and per-state stats, NumPy arrays with per-state means, pre-entry rankings.
   - Figures: weight distributions, per-state heatmaps, duty-cycle timelines.

### How to interpret
- Residues with high mean/median and high duty cycle are “structural” candidates.
- Residues with a high episodic score may indicate local triggers or rare conformational events.
- In pre-entry to rare states, residues with elevated attention are good candidates for additional CVs or localized bias.

---

## Best practices
- Always verify consistency between `residue_ids` and the topology used in PLUMED.
- Choose `BLOCK_LEN` in line with the trajectory autocorrelation time (start testing 200–1000 frames).
- Fix `RANDOM_SEED` for reproducibility in clustering.

---

## Frequently asked questions (FAQ)
- I can’t find `attn_w.npy`: ensure training/saving enables attention weight dumping.
- There are too many/few “rare” states: adjust `N_CLUSTERS` or `RARE_FRACTION_THRESHOLD`.
- Confidence intervals are too wide: increase `BLOCK_LEN` or the trajectory length/stride.
