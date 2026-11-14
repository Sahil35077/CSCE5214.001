# Pulsar Identification — Scientific & Engineering Audit Report

**Generated:** 2026-11-13 17:55:31 UTC

## 1. Dataset Provenance & Invariants

| Check | Result |
| --- | --- |
| Source | `local_csv:HTRU_2.csv` |
| Shape `(17898, 9)` | **PASS** — `17898 × 9` |
| Missing values | **PASS** — `0` |
| Duplicate rows | **PASS** — `0` |
| Class 0 (Non-Pulsar) | **PASS** — `16259` (90.84%) |
| Class 1 (Pulsar) | **PASS** — `1639` (9.16%) |
| Target leakage into X | **PASS** — X has 8 features only |

> **Paper distinction:** Official UCI HTRU2 = 17,898 rows / 0 missing. Abdullah et al. (2023) used a 12,528-row Kaggle train split (~70%) with ~3.1% injected missingness + MICE. Dual-track via `--simulate-paper-missingness` (IterativeImputer fit on **train only** after split).

## 2. Train/Test Split & Zero Leakage

| Check | Result |
| --- | --- |
| Train size | `14318` (expected 14318) |
| Test size | `3580` (expected 3580) |
| Index overlap | `0` (**PASS** if 0) |
| StandardScaler | **PASS** — encapsulated in `sklearn.pipeline.Pipeline`; fit on train / CV folds only |
| MICE (optional track) | **PASS** — `IterativeImputer.fit(X_train)` then `transform(X_test)` |

## 3. Model Implementation

| Model | Spec | Status |
| --- | --- | --- |
| MARS | degree=1 hinge pairs `max(0,x-t)`, `max(0,t-x)`; GCV prune nprune; logistic head; CV-safe state reset | best=`{'clf__nprune': 16}` |
| KNN | Euclidean on standardized features; k-grid incl. paper k=15 | best=`{'clf__n_neighbors': 21}` |
| ANN | MLPClassifier + Pipeline scaler; max_iter=2000; deterministic seed | best=`{'clf__activation': 'relu', 'clf__alpha': 0.0001, 'clf__hidden_layer_sizes': [64, 32], 'clf__solver': 'adam'}` |

### CV PR-AUC (tuning objective)

- **MARS**: best PR-AUC = `0.9300` (quick=False)
- **KNN**: best PR-AUC = `0.8962` (quick=False)
- **ANN**: best PR-AUC = `0.9290` (quick=False)

## 4. Holdout Metrics

**Primary (proposal):** `pos_label=1` Pulsar — see `metrics_pos1_pulsar.csv` and columns `pulsar_pos1__*` in `paper_comparison.csv`.

**Paper-aligned:** `pos_label=0` Non-Pulsar (caret default) — see `metrics_pos0_nonpulsar_paper_aligned.csv` and columns `nonpulsar_pos0__*`.

### Pulsar-class metrics (pos=1)

| model | metric_family | accuracy | precision | recall | f1 | pr_auc | roc_auc | balanced_accuracy | specificity | mcc | kappa | pulsar_recall_from_cm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MARS | PRIMARY_pos_label_1_Pulsar | 0.97933 | 0.929054 | 0.838415 | 0.88141 | 0.931495 | 0.977479 | 0.915979 | 0.993542 | 0.871497 | 0.870121 | 0.838415 |
| KNN | PRIMARY_pos_label_1_Pulsar | 0.981006 | 0.954545 | 0.832317 | 0.889251 | 0.908824 | 0.958974 | 0.91416 | 0.996002 | 0.881384 | 0.878916 | 0.832317 |
| ANN | PRIMARY_pos_label_1_Pulsar | 0.980726 | 0.92459 | 0.859756 | 0.890995 | 0.929868 | 0.974043 | 0.926342 | 0.992927 | 0.881139 | 0.880439 | 0.859756 |

### McNemar pairwise tests

| model_a | model_b | n_a_correct_b_wrong | n_a_wrong_b_correct | statistic | pvalue | method |
| --- | --- | --- | --- | --- | --- | --- |
| MARS | KNN | 11 | 17 | 0.892857 | 0.344704 | corrected |
| MARS | ANN | 6 | 11 | 6 | 0.332306 | exact |
| KNN | ANN | 13 | 12 | 0 | 1 | corrected |

## 5. Robustness

- Perturbations applied to **X_test only**; models/scalers never refit on noise.
- Noise: $\epsilon_j \sim \mathcal{N}(0, (\sigma_{\mathrm{level}} \cdot \sigma_{j,\mathrm{train}})^2)$, $\sigma_{\mathrm{level}} \in \{0.00,0.05,0.10,0.15,0.20,0.30,0.50\}$.
- Physical bounds: `std_*` features clipped to ≥ 0 after noise; kurtosis/skewness/means left as raw unbounded statistics.
- Experiment B: top-3 (Excess Kurt. IP, Skewness IP, Mean DM-SNR) vs bottom-5.

## 6. Figure DPI Verification (target = 300)

| Figure | DPI | Status | Note |
| --- | --- | --- | --- |
| `00_class_balance.png` | `(299.9994, 299.9994)` | **PASS** | ok |
| `01_correlation_heatmap.png` | `(299.9994, 299.9994)` | **PASS** | ok |
| `02_feature_boxplots.png` | `(299.9994, 299.9994)` | **PASS** | ok |
| `03_feature_kde_distributions.png` | `(299.9994, 299.9994)` | **PASS** | ok |
| `04_cv_hyperparameter_tuning.png` | `(299.9994, 299.9994)` | **PASS** | ok |
| `05_feature_importance_comparison.png` | `(299.9994, 299.9994)` | **PASS** | ok |
| `06_confusion_matrices.png` | `(299.9994, 299.9994)` | **PASS** | ok |
| `07_pr_and_roc_curves.png` | `(299.9994, 299.9994)` | **PASS** | ok |
| `08_robustness_degradation.png` | `(299.9994, 299.9994)` | **PASS** | ok |
| `09_targeted_feature_perturbation.png` | `(299.9994, 299.9994)` | **PASS** | ok |

**Overall DPI gate:** PASS.

## 7. Scientific Takeaways (Presentation)

1. Always state **17,898 vs 12,528** when quoting Abdullah et al. (2023) tables.
2. Quote **Pulsar (class 1) Recall / F1 / PR-AUC**, not caret’s class-0 precision/recall (~0.98/0.99), which inflate apparent performance.
3. Corrected paper pulsar recalls from their confusion matrices: MARS 81.30%, KNN 79.13%, ANN 84.35%.
4. Robustness curves show which model degrades slowest under $\sigma_{\mathrm{level}}$ noise — the proposal’s core extension.
5. MARS is a pure NumPy/SciPy sklearn estimator (no broken `py-earth` / GIS `pyearth`).

## 8. Reproducibility

- Seed: `42`
- Quick mode: `False`
- Elapsed seconds: `86.56141495704651`
- Commands: `pytest -v` · `python main.py` · `python main.py --quick`
