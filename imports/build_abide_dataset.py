import argparse
import os
import numpy as np
import torch
from torch_geometric.data import Data
from scipy.io import loadmat
import pandas as pd
from tqdm import tqdm

from gdc import GDC


# Paths are resolved relative to the repo root, so this works on Colab
# (/content/BrainGNN_Pytorch) or any local clone.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(REPO_ROOT, "data", "ABIDE_pcp", "cpac", "filt_noglobal")
PHENO_CSV = os.path.join(REPO_ROOT, "data", "ABIDE_pcp", "Phenotypic_V1_0b_preprocessed1.csv")
# SAVE_PATH = os.path.join(REPO_ROOT, "data", "processed", "abide_graph_dataset.pt")
PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")


def dataset_path(atlas):
    """One dataset file per atlas, e.g. data/processed/abide_graph_dataset_cc200.pt"""
    return os.path.join(PROCESSED_DIR, f"abide_graph_dataset_{atlas}.pt")


def load_conn(sid, atlas, kind):
    """kind: 'correlation' or 'partial_correlation'"""
    path = os.path.join(DATA_ROOT, str(sid), f"{sid}_{atlas}_{kind}.mat")
    mat = loadmat(path)["connectivity"]
    # Guard against NaN/inf (e.g. an all-zero ROI time series in some atlases)
    return np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)

K = 10
THRESH = 0.05


def sparsify(A):
    np.fill_diagonal(A, 0)

    # remove weak correlations
    A[np.abs(A) < THRESH] = 0

    N = A.shape[0]
    mask = np.zeros_like(A)

    for i in range(N):
        idx = np.argsort(np.abs(A[i]))[::-1][:K]
        mask[i, idx] = 1

    A = A * mask
    A = np.maximum(A, A.T)

    return A


def normalize_adj(A):
    deg = np.sum(np.abs(A), axis=1)
    deg_inv_sqrt = 1.0 / np.sqrt(deg + 1e-6)
    return deg_inv_sqrt[:, None] * A * deg_inv_sqrt[None, :]


def build_dataset(atlas):

    save_path = dataset_path(atlas)
    print(f"Atlas: {atlas}")

    pheno = pd.read_csv(PHENO_CSV)

    label_map = {
        row.SUB_ID: 0 if row.DX_GROUP == 2 else 1
        for _, row in pheno.iterrows()
    }

    subjects = [
        int(s) for s in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, s)) and s.isdigit()
    ]
    subjects = sorted(subjects)

        # Keep only subjects that have both matrices for this atlas
    have = [
        sid for sid in subjects
        if all(os.path.exists(os.path.join(DATA_ROOT, str(sid), f"{sid}_{atlas}_{k}.mat"))
               for k in ("correlation", "partial_correlation"))
    ]
    if len(have) < len(subjects):
        print(f"WARNING: {len(subjects) - len(have)} subjects have no {atlas} matrices and are skipped "
              f"(run: python 01-fetch_data.py --atlas {atlas})")
    subjects = have
    if not subjects:
        raise SystemExit(f"No {atlas} connectivity matrices found in {DATA_ROOT}")

    # -------- PASS 1 : GLOBAL FEATURE NORMALIZATION --------

    all_features = []

    for sid in tqdm(subjects, desc="Collecting features"):
        corr = load_conn(sid, atlas, "correlation")
        all_features.append(corr)

    all_features = np.vstack(all_features)

    global_mean = all_features.mean(axis=0, keepdims=True)
    global_std = all_features.std(axis=0, keepdims=True) + 1e-6

    print("Global normalization stats computed")

    # -------- GDC TRANSFORM (CONTROLLED SETTINGS) --------

    gdc = GDC(
        self_loop_weight=1,
        normalization_in='sym',
        normalization_out='col',
        diffusion_kwargs=dict(method='ppr', alpha=0.05),
        sparsification_kwargs=dict(method='threshold', avg_degree=8),
        exact=True
    )

    dataset = []
    edge_counts = []

    # -------- PASS 2 : BUILD GRAPHS --------

    for sid in tqdm(subjects, desc="Building graphs"):

        pcorr = load_conn(sid, atlas, "partial_correlation")
        corr = load_conn(sid, atlas, "correlation")        

        pcorr = sparsify(pcorr)
        pcorr = normalize_adj(pcorr)

        row, col = np.nonzero(pcorr)

        edge_index = torch.tensor(np.stack([row, col]), dtype=torch.long)
        edge_attr = torch.tensor(pcorr[row, col], dtype=torch.float)

        X = (corr - global_mean) / global_std

        y = label_map[sid]

        n_roi = corr.shape[0]          # 200 for cc200, 392 for cc400, 111 for ho, ...
        pos = torch.eye(n_roi)

        data = Data(
            x=torch.tensor(X, dtype=torch.float),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([y], dtype=torch.long),
            pos=pos
        )

        # -------- APPLY CONTROLLED GDC --------
        data = gdc(data)

        dataset.append(data)
        edge_counts.append(data.edge_index.shape[1])

    print("\nDataset built.")
    print("ROIs (nodes) per graph:", dataset[0].num_nodes)
    print("Average edges per graph:", np.mean(edge_counts))

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    torch.save(dataset, save_path)

    print("Saved dataset at:", save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas", default="cc200",
                        help="Atlas used in 01-fetch_data.py: cc200, cc400, ho, aal, ez, tt, dosenbach160")
    args = parser.parse_args()
    build_dataset(args.atlas)    