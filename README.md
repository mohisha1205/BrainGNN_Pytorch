# BrainGNN on ABIDE — ASD vs. Control Classification

A PyTorch Geometric implementation of **BrainGNN** for classifying autism spectrum
disorder (ASD) from resting-state fMRI connectivity, using the ABIDE I
preprocessed dataset (C-PAC pipeline, `filt_noglobal`, CC200 atlas).

The project is based on [xxxhois/BrainGNN_Pytorch](https://github.com/xxxhois/BrainGNN_Pytorch),
which reproduces the original BrainGNN paper (Li et al., *Medical Image Analysis*, 2021).
It has been updated to run on **Python 3.12 / NumPy 2 / PyG 2.7** (Google Colab), and
the dataset building, training and evaluation code has been rewritten.

## What's changed from the reference repo

| Area | Change |
|---|---|
| Compatibility | `imports/read_abide_stats_parall.py`: `from_numpy_matrix` → `from_numpy_array`, `to_scipy_sparse_matrix` → `to_scipy_sparse_array` (NetworkX 3.x) |
| Compatibility | `scripts/patch_deepdish.py`: patches deepdish for NumPy 2 (removed `np.unicode_`, `np.string_`, `np.object`, `np.ComplexWarning`) |
| Dataset | New `imports/build_abide_dataset.py`: builds a single `abide_graph_dataset.pt` directly from the `.mat` files |
| Training | `03-main.py` rewritten: nested split (5-fold stratified CV + stratified 20% validation split inside each fold), early stopping on validation accuracy, sensitivity/specificity, per-fold plots and a CSV of metrics |
| Visualisation | New `scripts/visualize_connectivity.py` for per-subject connectivity heatmaps and BOLD time series |
| Environment | `requirements.txt` updated to the versions used on Colab; outdated `environment.yml` (torch 1.13) removed |

## Graph construction

For each of the 1,035 subjects (`imports/build_abide_dataset.py`):

- **Nodes:** 200 ROIs (CC200 atlas)
- **Node features `x`:** the subject's 200×200 Pearson correlation matrix, z-scored with
  the mean and standard deviation taken over all subjects
- **Edges:** partial-correlation matrix → diagonal zeroed → |r| < 0.05 removed → top-10
  neighbours kept for each node → symmetrised → normalised by node degree
- **GDC:** Graph Diffusion Convolution (PPR, α = 0.05, threshold sparsification, average degree 8)
- **`pos`:** 200×200 identity (one-hot ROI identity, used by BrainGNN's ROI-aware conv)
- **Label `y`:** ASD = 1, typical control = 0 (from `DX_GROUP` in the phenotypic CSV)

## Project structure

```
BrainGNN_Pytorch/
├── 01-fetch_data.py              # Download ABIDE (nilearn) + compute connectivity .mat files
├── 02-process_data.py            # Original .h5 processing pipeline (optional, not used by 03-main.py)
├── 03-main.py                    # Train / evaluate BrainGNN with nested 5-fold CV
├── requirements.txt
├── TRAINING_TIPS.md              # Training notes from the reference repo (Chinese)
├── data/
│   └── subject_ID.txt            # 1,035 ABIDE subject IDs
├── imports/
│   ├── build_abide_dataset.py    # NEW: .mat → PyG graphs → data/processed/abide_graph_dataset.pt
│   ├── read_abide_stats_parall.py
│   ├── preprocess_data.py
│   ├── ABIDEDataset.py
│   ├── gdc.py
│   └── utils.py
├── net/
│   ├── braingnn.py               # BrainGNN network (Ra-GConv + R-pool)
│   ├── braingraphconv.py
│   ├── brainmsgpassing.py
│   └── inits.py
├── notebooks/
│   └── BrainGNN_ABIDE_colab.ipynb   # End-to-end Colab notebook (outputs cleared)
├── scripts/
│   ├── patch_deepdish.py         # NumPy-2 fix for deepdish
│   └── visualize_connectivity.py # Subject-level heatmaps & time series
└── results/
    └── README.md                 # Summary of CV results
```

Created by running the pipeline (git-ignored): `data/ABIDE_pcp/`, `data/processed/`,
`cv_models/`, `plots/`.

## Quick start

### 1. Install

```bash
git clone https://github.com/<your-username>/BrainGNN_Pytorch.git
cd BrainGNN_Pytorch
pip install -r requirements.txt
python scripts/patch_deepdish.py
```

`torch_scatter`, `torch_sparse`, `torch_cluster` and `torch_spline_conv` are compiled
against your installed `torch`. If a pip build fails, install the matching wheels from
`https://data.pyg.org/whl/torch-<version>+<cpu|cuXXX>.html`.

### 2. Download the data

```bash
python 01-fetch_data.py
```

This downloads the ABIDE I preprocessed ROI time series and the phenotypic CSV into
`data/ABIDE_pcp/`, and writes `*_cc200_correlation.mat` and
`*_cc200_partial_correlation.mat` for each subject.

### 3. Build the graph dataset

```bash
python imports/build_abide_dataset.py
# -> data/processed/abide_graph_dataset.pt  (1,035 graphs)
```

### 4. Train and evaluate

```bash
python 03-main.py --weightdecay 0.02 --ratio 0.5 --stepsize 10
```

Main arguments:

| Arg | Default | Meaning |
|---|---|---|
| `--n_epochs` | 150 | Max epochs per fold (early stopping patience = 12) |
| `--batchSize` | 16 | Batch size |
| `--lr` | 0.001 | Adam learning rate |
| `--stepsize` / `--gamma` | 20 / 0.5 | StepLR schedule |
| `--weightdecay` | 0.02 | L2 regularisation |
| `--ratio` | 0.2 | TopK pooling ratio |
| `--lamb0`…`--lamb5` | 1, 0.1 … | Loss weights: CE, unit-norm ×2, TopK ×2, group-consistency |
| `--val_size` | 0.2 | Validation fraction inside each training fold |

Outputs: `cv_models/fold{k}.pth`, `plots/fold{k}.png` (train/val accuracy curves),
`plots/fold_metrics.csv`.

### 5. (Optional) Visualise a subject

```bash
python scripts/visualize_connectivity.py --subject 50003 --save_dir plots/subject_50003
```

### On Google Colab

`notebooks/BrainGNN_ABIDE_colab.ipynb` is the original development notebook (outputs
cleared). Its first cell lists the shorter set of commands to use with this repo.

## Current results

5-fold stratified CV, 1,035 subjects (530 controls / 505 ASD):

| Metric | Mean ± Std |
|---|---|
| Accuracy | 0.577 ± 0.022 |
| Sensitivity (ASD) | 0.586 ± 0.091 |
| Specificity (TC) | 0.568 ± 0.100 |

Per-fold numbers are in [`results/README.md`](results/README.md). These results are
still well below the ~70% reported for BrainGNN on ABIDE. Tuning is in progress.

## Credits

- BrainGNN: X. Li *et al.*, "BrainGNN: Interpretable Brain Graph Neural Network for fMRI
  Analysis", *Medical Image Analysis* 74 (2021). Original code: https://github.com/xxlya/BrainGNN_Pytorch
- Reference fork: https://github.com/xxxhois/BrainGNN_Pytorch
- Data fetching code adapted from Kunda *et al.* / Parisot *et al.* (GPL-3.0, see file headers)
- ABIDE Preprocessed Connectomes Project: http://preprocessed-connectome-project.org/abide/
