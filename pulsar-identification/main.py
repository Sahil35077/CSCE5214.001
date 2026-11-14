#!/usr/bin/env python3
"""
CSCE 5214 – Identification and Estimation of Pulsars upon Statistical Analysis

Single CLI entry point for the full reproducible pipeline.

Usage:
  python main.py                  # full protocol (30-fold RepeatedStratifiedKFold)
  python main.py --quick          # smoke / demo mode (2-fold CV, fewer MC seeds)
  python main.py --simulate-paper-missingness
  python main.py --skip-robustness
  python main.py --eda-only
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import matplotlib

matplotlib.use("Agg")

from src.audit import generate_audit_report
from src.data_loader import load_htru2, validate_htru2
from src.eda import run_eda
from src.evaluate import evaluate_models
from src.models.ann import tune_ann
from src.models.knn import tune_knn
from src.models.mars import tune_mars
from src.preprocessing import prepare_data
from src.robustness import run_robustness
from src.utils import ensure_dirs, load_config, save_json, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pulsar identification pipeline")
    p.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "experiment_config.yaml"),
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Fast smoke mode: reduced CV folds / MC seeds",
    )
    p.add_argument(
        "--simulate-paper-missingness",
        action="store_true",
        help="Dual-track: 12,528-row / ~3.1%% missingness + MICE (paper variant)",
    )
    p.add_argument("--skip-eda", action="store_true")
    p.add_argument("--skip-robustness", action="store_true")
    p.add_argument("--eda-only", action="store_true")
    p.add_argument("--force-download", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    # Surface ConvergenceWarning etc. during audit runs
    warnings.filterwarnings("default")

    config = load_config(args.config)
    set_seed(config["seed"])
    paths = ensure_dirs(config)

    print("=" * 72)
    print("CSCE 5214 Pulsar Identification Pipeline")
    print("Official UCI HTRU2 (17,898 rows) | MARS / KNN / ANN")
    print("=" * 72)
    if args.quick:
        print("[mode] QUICK smoke run (reduced CV / MC)")
    if args.simulate_paper_missingness:
        print("[mode] Paper missingness track (12,528 rows + MICE)")

    t0 = time.time()

    # ---- Data ----
    df = load_htru2(config, force_download=args.force_download)
    report = validate_htru2(df, config, silent=True)
    save_json(report, paths["processed_dir"] / "validation_report.json")
    if not report.get("ok", False):
        print("[error] Validation failed:", report.get("errors"))
        return 1
    print(
        f"[data] OK – {report['n_rows']} rows | "
        f"class0={report['n_class_0']} class1={report['n_class_1']} | "
        f"NaNs={report['n_nan']} dups={report.get('n_duplicates', 0)}"
    )
    print(f"[data] {report['paper_vs_uci_note']}")

    # ---- EDA ----
    if not args.skip_eda:
        run_eda(df, config)
    if args.eda_only:
        print("[done] EDA only.")
        return 0

    # ---- Preprocess ----
    prepared = prepare_data(
        df, config, simulate_paper=args.simulate_paper_missingness
    )
    X_train, X_test = prepared["X_train"], prepared["X_test"]
    y_train, y_test = prepared["y_train"], prepared["y_test"]
    print(
        f"[split] train={len(X_train)} test={len(X_test)} "
        f"(stratified {1 - config['split']['test_size']:.0%}/"
        f"{config['split']['test_size']:.0%})"
    )

    # ---- Tune & fit models ----
    print("[train] Tuning MARS ...")
    mars_model, mars_details = tune_mars(X_train, y_train, config, quick=args.quick)
    print(
        f"  best={mars_details['best_params']} "
        f"PR-AUC={mars_details['best_score_pr_auc']:.4f}"
    )

    print("[train] Tuning KNN ...")
    knn_model, knn_details = tune_knn(X_train, y_train, config, quick=args.quick)
    print(
        f"  best k={knn_details['selected_k']} "
        f"(paper k={knn_details['paper_optimal_k']}) "
        f"PR-AUC={knn_details['best_score_pr_auc']:.4f}"
    )

    print("[train] Tuning ANN ...")
    ann_model, ann_details = tune_ann(X_train, y_train, config, quick=args.quick)
    print(
        f"  best={ann_details['best_params']} "
        f"PR-AUC={ann_details['best_score_pr_auc']:.4f}"
    )

    models = {"MARS": mars_model, "KNN": knn_model, "ANN": ann_model}

    for name, model in models.items():
        out = paths["models_dir"] / f"{name.lower()}.joblib"
        joblib.dump(model, out)
        print(f"[save] {out}")

    save_json(
        {"mars": mars_details, "knn": knn_details, "ann": ann_details},
        paths["metrics_dir"] / "cv_tuning_details.json",
    )

    # ---- Evaluate ----
    print("[eval] Holdout evaluation (pos_label=1 primary, pos_label=0 paper-aligned) ...")
    comparison = evaluate_models(
        models,
        X_test,
        y_test,
        X_train=X_train,
        y_train=y_train,
        config=config,
        mars_details=mars_details,
        knn_details=knn_details,
        ann_details=ann_details,
    )
    show_cols = [
        c
        for c in [
            "model",
            "pulsar_pos1__accuracy",
            "pulsar_pos1__precision",
            "pulsar_pos1__recall",
            "pulsar_pos1__f1",
            "pulsar_pos1__pr_auc",
            "nonpulsar_pos0__precision",
            "nonpulsar_pos0__recall",
            "pulsar_pos1__pulsar_recall_from_cm",
        ]
        if c in comparison.columns
    ]
    print(comparison[show_cols].to_string(index=False))

    # ---- Robustness ----
    if not args.skip_robustness:
        print("[robustness] Running Experiments A–C ...")
        run_robustness(
            models, X_train, X_test, y_test, config=config, quick=args.quick
        )

    # ---- Manifest + Audit ----
    elapsed = time.time() - t0
    manifest = {
        "project": "CSCE 5214 Pulsar Identification",
        "seed": config["seed"],
        "quick": args.quick,
        "simulate_paper_missingness": args.simulate_paper_missingness,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "validation": report,
        "best_params": {
            "mars": mars_details["best_params"],
            "knn": knn_details["best_params"],
            "ann": ann_details["best_params"],
        },
        "elapsed_seconds": elapsed,
        "dataset_distinction": {
            "ours": "Official UCI HTRU2 17,898 rows, 0 missing",
            "paper": "Kaggle train split 12,528 rows (~70%), ~3.1% missing + MICE",
            "pos_label_note": (
                "Paper caret tables used Positive Class : 0 (non-pulsar). "
                "We log both pos_label=1 (proposal) and pos_label=0 (paper-aligned)."
            ),
        },
    }
    save_json(manifest, paths["metrics_dir"] / "experiment_manifest.json")
    save_json(manifest, ROOT / "results" / "experiment_manifest.json")
    generate_audit_report(config=config, manifest=manifest)

    print("=" * 72)
    print(f"Done in {elapsed / 60:.1f} min. Artifacts under results/")
    print("  metrics/paper_comparison.csv")
    print("  figures/08_robustness_degradation.png")
    print("  AUDIT_REPORT.md")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
