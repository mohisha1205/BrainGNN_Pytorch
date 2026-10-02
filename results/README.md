# Results

All runs: ABIDE I, CC200, 1,035 subjects (505 ASD / 530 TC), stratified 5-fold CV,
GDC graphs from `imports/build_abide_dataset.py`. Raw `plots/` and `cv_models/` are git-ignored.

## Final configuration

    python 03-main.py --weightdecay 0.02 --ratio 0.5 --stepsize 10 --patience 12 --lamb1 0 --lamb2 0

(`--lamb1 0 --lamb2 0` reproduces the reported run exactly: it was trained before the
unit-loss fix, when those two terms were always zero.)

| Fold | Accuracy (%) | Sensitivity (%) | Specificity (%) |
|------|------|------|------|
| 1 | 62.32 | 62.38 | 62.26 |
| 2 | 62.32 | 70.30 | 54.72 |
| 3 | 65.70 | 74.26 | 57.55 |
| 4 | 59.90 | 57.43 | 62.26 |
| 5 | 64.25 | 63.37 | 65.09 |
| **Mean** | **62.90** | **65.54** | **60.38** |
| Std | 1.97 | 5.99 | 3.73 |

## Ablations

| Configuration | Accuracy (mean ± std) |
|---|---|
| Original defaults (lr 0.01, wd 0.005, FC 512, no early stopping) | ~51.2% |
| FC bottleneck 512 (otherwise final config) | 57.7 ± 2.2% |
| Weight decay 0.005 (otherwise final config) | 57.4 ± 4.0% |
| **Final config (FC 16, wd 0.02)** | **62.9 ± 2.0%** |