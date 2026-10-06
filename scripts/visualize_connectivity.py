"""
Quick-look visualisations for one ABIDE subject (cc200 atlas):

  1. Pearson correlation matrix        (<sid>_cc200_correlation.mat)
  2. Partial correlation matrix        (<sid>_cc200_partial_correlation.mat)
  3. BOLD time series, first 3 ROIs    (*_rois_cc200.1D)
  4. Correlation matrix recomputed from the .1D time series

Usage (from the repo root):
    python scripts/visualize_connectivity.py --subject 50003
    python scripts/visualize_connectivity.py --subject 50003 --save_dir ./plots/subject_50003
"""
import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import seaborn as sns

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(REPO_ROOT, "data", "ABIDE_pcp", "cpac", "filt_noglobal")

parser = argparse.ArgumentParser()
parser.add_argument("--subject", type=str, default="50003")
parser.add_argument("--atlas", type=str, default="cc200")
parser.add_argument("--save_dir", type=str, default=None,
                    help="If given, figures are saved here instead of shown.")
opt = parser.parse_args()

subj_dir = os.path.join(DATA_ROOT, opt.subject)
if opt.save_dir:
    os.makedirs(opt.save_dir, exist_ok=True)


def finish(name):
    if opt.save_dir:
        plt.savefig(os.path.join(opt.save_dir, name), dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def heatmap(matrix, title, name):
    plt.figure(figsize=(10, 8))
    # coolwarm: strong negative correlations blue, strong positive red
    sns.heatmap(matrix, cmap="coolwarm", center=0, vmin=-1, vmax=1)
    plt.title(title)
    plt.xlabel("Brain Region (ROI)")
    plt.ylabel("Brain Region (ROI)")
    finish(name)


# 1-2. Connectivity matrices stored in .mat files (key: 'connectivity')
for kind, label in [("correlation", "Correlation"), ("partial_correlation", "Partial Correlation")]:
    path = os.path.join(subj_dir, f"{opt.subject}_{opt.atlas}_{kind}.mat")
    mat = scipy.io.loadmat(path)
    heatmap(mat["connectivity"], f"{label} Matrix - subject {opt.subject}", f"{kind}.png")

# 3-4. ROI time series (.1D: rows = timepoints, columns = ROIs)
ts_file = glob.glob(os.path.join(subj_dir, f"*_rois_{opt.atlas}.1D"))[0]
time_series = np.loadtxt(ts_file)
print(f"Time series shape: {time_series.shape}  (timepoints x ROIs)")

plt.figure(figsize=(12, 4))
plt.plot(time_series[:, :3])
plt.title("BOLD Signal Over Time (First 3 Brain Regions)")
plt.xlabel("Time (Scans)")
plt.ylabel("Signal")
finish("bold_timeseries.png")

heatmap(np.corrcoef(time_series.T),
        "Correlation Matrix recomputed from .1D time series",
        "correlation_from_1D.png")
