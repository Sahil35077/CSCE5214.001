# Identification and Estimation of Pulsars upon Statistical Analysis

**CSCE 5214 Final Project** — reproducible Python reproduction and extension of Abdullah et al. (2023 IEEE NMITCON) on the official **UCI HTRU2** dataset.

## Critical design choices (read before comparing to the paper)

| Topic | This project (proposal) | 2023 IEEE paper |
| --- | --- | --- |
| Dataset | Official UCI HTRU2 **17,898** rows, **0** missing | Kaggle train split **12,528** rows (~70%) with **~3.1%** injected missingness + MICE |
| Positive class | **`pos_label=1` (Pulsar)** for primary metrics | R `caret::confusionMatrix` defaulted to **Positive Class : 0** (non-pulsar) → inflated ~0.98 precision / ~0.99 recall |
| Corrected paper pulsar recall | Logged in `paper_baselines_reference.csv` | MARS 81.30%, KNN 79.13%, ANN 84.35% (from their CMs) |
| MARS | Pure NumPy/SciPy `MARSClassifier` (degree=1, nprune=14) | R `earth` package — **do not** install GIS `pyearth` or broken legacy `py-earth` on Py3.11+ |

Dual-track: `python main.py --simulate-paper-missingness` approximates the paper’s 12,528-row / MICE variant for sensitivity checks. Both `pos_label=1` and `pos_label=0` metrics are written to `results/metrics/paper_comparison.csv`.

## Quick start

```bash
cd pulsar-identification
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Unit tests (downloads/caches HTRU2 on first run)
pytest -q

# Full pipeline (30-fold RepeatedStratifiedKFold — can take a while)
python main.py

# Fast smoke / demo run
python main.py --quick
```

## CLI flags

| Flag | Meaning |
| --- | --- |
| `--quick` | 2-fold CV, fewer MC seeds (CI / presentation smoke) |
| `--simulate-paper-missingness` | 12,528-row + ~3.1% NaNs + IterativeImputer (MICE) |
| `--skip-robustness` | Skip Experiments A–C |
| `--eda-only` | Figures/tables only |
| `--force-download` | Re-fetch UCI data |

## Layout

```
pulsar-identification/
├── configs/experiment_config.yaml
├── data/raw/  data/processed/
├── src/   # data_loader, eda, preprocessing, models/{mars,knn,ann}, evaluate, robustness
├── notebooks/demo_walkthrough.ipynb
├── results/{figures,metrics,models}/
├── tests/test_pipeline.py
└── main.py
```

## Models

1. **MARS** — custom hinge basis (`max(0,x−t)`, `max(0,t−x)`), degree=1, GCV prune to `nprune=14`, logistic head  
2. **KNN** — `StandardScaler` + Euclidean `KNeighborsClassifier`, k grid including paper optimum **k=15**  
3. **ANN** — `MLPClassifier` with hidden-size / activation / α grid  

Evaluation: confusion matrices, Accuracy / Precision / Recall / F1 / PR-AUC / ROC-AUC / Balanced Accuracy / Specificity / MCC / Kappa, McNemar tests, PR & ROC overlays.

## Robustness extension

- **A** — Gaussian feature noise σ_level ∈ {0, 0.05, …, 0.50}, 20 MC seeds  
- **B** — Top-feature vs non-top perturbation (excess kurtosis IP, skewness IP, mean DM-SNR)  
- **C** — PR-optimal thresholds under noise  

See `results/figures/robustness_curves.png` and `results/metrics/robustness_results.csv`.

## Data source

- UCI ML Repository ID **372** (Lyon, 2015) via `ucimlrepo`, with CSV / zip fallback  
- Columns: 8 profile/DM-SNR features + `target_class` (0=non-pulsar, 1=pulsar)

Paper R reference (appendix): https://github.com/devika1402/Identification-of-Pulsar-star

## Citation

Abdullah et al., “A Machine Learning Approach for the Identification and Estimation of Pulsars upon Statistical Analysis,” IEEE NMITCON 2023.
