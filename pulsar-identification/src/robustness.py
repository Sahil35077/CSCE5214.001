"""
Controlled Gaussian feature perturbation & covariate-noise stress tests.

PHYSICAL PERTURBATION POLICY
----------------------------
Perturbations are applied in *raw feature space* on X_test only:
  ε_j ~ N(0, (σ_level · σ_{j,train})²)

We do NOT refit scalers or models on noisy data. Pipelines transform the
perturbed raw features with the *training-fit* StandardScaler at predict time.

Physical bounds:
  - std_* features are non-negative in nature. After additive Gaussian noise we
    optionally clip them to ≥ 0 (clip_nonnegative_std=True) so noisy std
    features remain physically plausible. Kurtosis/skewness/means are
    unbounded signed statistics and are left unclipped (raw feature-space
    perturbation). Both modes are documented in robustness_meta.json.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    recall_score,
)

from src.data_loader import FEATURE_NAMES
from src.plotting import MODEL_COLORS, apply_pub_style, despine, model_color, save_figure
from src.utils import PROJECT_ROOT, load_config, save_json

# Indices of features that are physically non-negative (standard deviations)
_STD_FEATURE_IDX = [
    FEATURE_NAMES.index("std_integrated_profile"),
    FEATURE_NAMES.index("std_dm_snr_curve"),
]


def _as_array(X) -> np.ndarray:
    if hasattr(X, "values"):
        return np.asarray(X.values, dtype=float)
    return np.asarray(X, dtype=float)


def add_gaussian_noise(
    X: np.ndarray,
    train_std: np.ndarray,
    sigma_level: float,
    rng: np.random.Generator,
    clip_nonnegative_std: bool = True,
) -> np.ndarray:
    """
    ε ~ N(0, (σ_level · σ_j_train)²) per feature on a copy of X_test.

    Never mutates the original array. Optionally clips std_* columns to ≥ 0.
    """
    X_out = X.copy()
    if sigma_level == 0.0:
        return X_out
    noise = rng.normal(0.0, sigma_level * train_std, size=X.shape)
    X_out = X_out + noise
    if clip_nonnegative_std:
        for j in _STD_FEATURE_IDX:
            X_out[:, j] = np.maximum(X_out[:, j], 0.0)
    return X_out


def optimal_threshold_from_pr(y_true, y_proba) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    if len(thresholds) == 0:
        return 0.5
    f1 = (
        2
        * precision[:-1]
        * recall[:-1]
        / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    )
    return float(thresholds[int(np.argmax(f1))])


def _eval_at_threshold(y_true, y_proba, threshold: float) -> dict[str, float]:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
    }


def _to_model_input(X_noisy: np.ndarray, X_test_template):
    if hasattr(X_test_template, "columns"):
        return pd.DataFrame(X_noisy, columns=X_test_template.columns)
    return X_noisy


def experiment_a_gaussian_noise(
    models: dict[str, Any],
    X_test,
    y_test,
    train_std: np.ndarray,
    sigma_levels: list[float],
    n_monte_carlo: int,
    seed: int,
    thresholds: dict[str, float] | None = None,
    clip_nonnegative_std: bool = True,
) -> pd.DataFrame:
    """Experiment A: uniform Gaussian noise on all 8 features — X_test only."""
    X = _as_array(X_test)
    y = np.asarray(y_test).ravel()
    rows = []
    thresholds = thresholds or {name: 0.5 for name in models}

    for name, model in models.items():
        thr = thresholds.get(name, 0.5)
        for sigma in sigma_levels:
            metrics_mc = []
            for mc in range(n_monte_carlo):
                rng = np.random.default_rng(seed + mc * 1009 + int(sigma * 1000))
                X_noisy = add_gaussian_noise(
                    X, train_std, sigma, rng, clip_nonnegative_std=clip_nonnegative_std
                )
                X_in = _to_model_input(X_noisy, X_test)
                proba = model.predict_proba(X_in)[:, 1]
                metrics_mc.append(_eval_at_threshold(y, proba, thr))
            df_mc = pd.DataFrame(metrics_mc)
            for metric in ["accuracy", "f1", "recall", "pr_auc"]:
                mean = float(df_mc[metric].mean())
                std = float(df_mc[metric].std(ddof=1)) if len(df_mc) > 1 else 0.0
                se = std / np.sqrt(len(df_mc)) if len(df_mc) > 1 else 0.0
                rows.append(
                    {
                        "experiment": "A_gaussian_noise",
                        "model": name,
                        "sigma_level": sigma,
                        "metric": metric,
                        "mean": mean,
                        "std": std,
                        "ci95_lo": mean - 1.96 * se,
                        "ci95_hi": mean + 1.96 * se,
                        "n_mc": n_monte_carlo,
                        "clip_nonnegative_std": clip_nonnegative_std,
                    }
                )
    return pd.DataFrame(rows)


def experiment_b_high_leverage(
    models: dict[str, Any],
    X_test,
    y_test,
    train_std: np.ndarray,
    feature_names: list[str],
    top_features: list[str],
    sigma_levels: list[float],
    n_monte_carlo: int,
    seed: int,
    clip_nonnegative_std: bool = True,
) -> pd.DataFrame:
    """Experiment B: top-3 predictive features vs remaining (bottom-5) features."""
    X = _as_array(X_test)
    y = np.asarray(y_test).ravel()
    top_idx = [feature_names.index(f) for f in top_features if f in feature_names]
    non_top_idx = [i for i in range(len(feature_names)) if i not in top_idx]
    rows = []

    for group_name, idxs in [
        ("top3_features", top_idx),
        ("bottom5_features", non_top_idx),
    ]:
        mask = np.zeros(len(feature_names), dtype=bool)
        mask[idxs] = True
        for name, model in models.items():
            for sigma in sigma_levels:
                vals = []
                for mc in range(n_monte_carlo):
                    rng = np.random.default_rng(seed + 17 + mc * 911 + int(sigma * 1000))
                    X_noisy = X.copy()
                    if sigma > 0:
                        noise = rng.normal(0.0, sigma * train_std, size=X.shape)
                        X_noisy[:, mask] = X[:, mask] + noise[:, mask]
                        if clip_nonnegative_std:
                            for j in _STD_FEATURE_IDX:
                                if mask[j]:
                                    X_noisy[:, j] = np.maximum(X_noisy[:, j], 0.0)
                    X_in = _to_model_input(X_noisy, X_test)
                    proba = model.predict_proba(X_in)[:, 1]
                    pred = (proba >= 0.5).astype(int)
                    vals.append(
                        {
                            "accuracy": float(accuracy_score(y, pred)),
                            "f1": float(f1_score(y, pred, pos_label=1, zero_division=0)),
                            "recall": float(
                                recall_score(y, pred, pos_label=1, zero_division=0)
                            ),
                            "pr_auc": float(average_precision_score(y, proba)),
                        }
                    )
                df_mc = pd.DataFrame(vals)
                for metric in ["accuracy", "f1", "recall", "pr_auc"]:
                    rows.append(
                        {
                            "experiment": "B_high_leverage",
                            "feature_group": group_name,
                            "model": name,
                            "sigma_level": sigma,
                            "metric": metric,
                            "mean": float(df_mc[metric].mean()),
                            "std": float(df_mc[metric].std(ddof=1))
                            if len(df_mc) > 1
                            else 0.0,
                            "n_mc": n_monte_carlo,
                        }
                    )
    return pd.DataFrame(rows)


def experiment_c_threshold_stress(
    models: dict[str, Any],
    X_test,
    y_test,
    train_std: np.ndarray,
    sigma_levels: list[float],
    seed: int,
    clip_nonnegative_std: bool = True,
) -> tuple[pd.DataFrame, dict[str, float]]:
    X = _as_array(X_test)
    y = np.asarray(y_test).ravel()
    thresholds = {}
    rows = []

    for name, model in models.items():
        X_in = _to_model_input(X, X_test)
        proba_clean = model.predict_proba(X_in)[:, 1]
        thr = optimal_threshold_from_pr(y, proba_clean)
        thresholds[name] = thr
        for sigma in sigma_levels:
            rng = np.random.default_rng(seed + 42 + int(sigma * 1000))
            X_noisy = add_gaussian_noise(
                X, train_std, sigma, rng, clip_nonnegative_std=clip_nonnegative_std
            )
            X_n = _to_model_input(X_noisy, X_test)
            proba = model.predict_proba(X_n)[:, 1]
            for use_thr, label in [(0.5, "fixed_0.5"), (thr, "pr_optimal")]:
                m = _eval_at_threshold(y, proba, use_thr)
                rows.append(
                    {
                        "experiment": "C_threshold_stress",
                        "model": name,
                        "sigma_level": sigma,
                        "threshold_mode": label,
                        "threshold": use_thr,
                        **m,
                    }
                )
    return pd.DataFrame(rows), thresholds


def plot_robustness_curves(results_a: pd.DataFrame, out_path: Path) -> None:
    apply_pub_style()
    metrics = ["f1", "pr_auc", "recall", "accuracy"]
    titles = [
        "F1-Score (Pulsar)",
        "PR-AUC",
        "Recall (Pulsar)",
        "Accuracy",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))
    axes = axes.ravel()
    models = [m for m in ["MARS", "KNN", "ANN"] if m in set(results_a["model"])]

    for ax, metric, title in zip(axes, metrics, titles):
        sub = results_a[results_a["metric"] == metric]
        for model in models:
            msub = sub[sub["model"] == model].sort_values("sigma_level")
            color = model_color(model)
            ax.plot(
                msub["sigma_level"],
                msub["mean"],
                "o-",
                label=model,
                color=color,
                lw=2.4,
                markersize=7,
            )
            ax.fill_between(
                msub["sigma_level"],
                msub["ci95_lo"],
                msub["ci95_hi"],
                alpha=0.22,
                color=color,
            )
        ax.set_xlabel(r"Perturbation level  $\sigma_{\mathrm{level}}$")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(frameon=True)
        despine(ax)

    fig.suptitle(
        r"Robustness Degradation under Gaussian Feature Noise"
        "\n"
        r"$X_{\mathrm{test}}$ only  ·  $\epsilon_j \sim \mathcal{N}(0,(\sigma_{\mathrm{level}}\cdot\sigma_{j,\mathrm{train}})^2)$"
        r"  ·  95% CI ribbons  ·  $N_{\mathrm{test}}=3{,}580$",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    save_figure(fig, out_path)


def plot_targeted_perturbation(results_b: pd.DataFrame, out_path: Path) -> None:
    """09: Accuracy & F1 drop — top-3 vs bottom-5 feature perturbation."""
    apply_pub_style()
    # Use a representative high-noise level for bar comparison + line across sigma
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: F1 vs sigma for top3 vs bottom5
    ax = axes[0]
    sub = results_b[results_b["metric"] == "f1"]
    for model in ["MARS", "KNN", "ANN"]:
        for group, ls in [("top3_features", "-"), ("bottom5_features", "--")]:
            g = sub[(sub["model"] == model) & (sub["feature_group"] == group)].sort_values(
                "sigma_level"
            )
            if g.empty:
                continue
            label = f"{model} / {'top-3' if 'top' in group else 'bottom-5'}"
            ax.plot(
                g["sigma_level"],
                g["mean"],
                ls,
                marker="o",
                color=model_color(model),
                lw=2.2,
                label=label,
                markersize=5,
            )
    ax.set_xlabel(r"$\sigma_{\mathrm{level}}$")
    ax.set_ylabel("F1 (Pulsar)")
    ax.set_title("F1 under Targeted Perturbation")
    ax.legend(fontsize=7, ncol=2)
    despine(ax)

    # Right: ΔAccuracy and ΔF1 at σ=0.30 (drop from clean)
    ax = axes[1]
    focus = 0.30
    available = sorted(results_b["sigma_level"].unique())
    if focus not in available:
        focus = available[-1] if available else 0.0
    rows = []
    for model in ["MARS", "KNN", "ANN"]:
        for metric in ["accuracy", "f1"]:
            for group in ["top3_features", "bottom5_features"]:
                clean = results_b[
                    (results_b["model"] == model)
                    & (results_b["feature_group"] == group)
                    & (results_b["metric"] == metric)
                    & (results_b["sigma_level"] == 0.0)
                ]
                noisy = results_b[
                    (results_b["model"] == model)
                    & (results_b["feature_group"] == group)
                    & (results_b["metric"] == metric)
                    & (results_b["sigma_level"] == focus)
                ]
                if clean.empty or noisy.empty:
                    continue
                drop = float(clean["mean"].iloc[0] - noisy["mean"].iloc[0])
                rows.append(
                    {
                        "model": model,
                        "metric": metric,
                        "group": "top-3" if "top" in group else "bottom-5",
                        "drop": drop,
                    }
                )
    if rows:
        rdf = pd.DataFrame(rows)
        metrics_present = rdf["metric"].unique()
        x = np.arange(len(["MARS", "KNN", "ANN"]))
        width = 0.18
        offsets = {
            ("accuracy", "top-3"): -1.5 * width,
            ("accuracy", "bottom-5"): -0.5 * width,
            ("f1", "top-3"): 0.5 * width,
            ("f1", "bottom-5"): 1.5 * width,
        }
        for metric in ["accuracy", "f1"]:
            for group in ["top-3", "bottom-5"]:
                vals = []
                for model in ["MARS", "KNN", "ANN"]:
                    hit = rdf[
                        (rdf["model"] == model)
                        & (rdf["metric"] == metric)
                        & (rdf["group"] == group)
                    ]
                    vals.append(float(hit["drop"].iloc[0]) if not hit.empty else 0.0)
                ax.bar(
                    x + offsets[(metric, group)],
                    vals,
                    width,
                    label=f"Δ{metric} {group}",
                    color=MODEL_COLORS["MARS"]
                    if metric == "f1" and group == "top-3"
                    else MODEL_COLORS["KNN"]
                    if metric == "f1"
                    else MODEL_COLORS["ANN"]
                    if group == "top-3"
                    else CLASS0_FALLBACK(),
                    alpha=0.85 if group == "top-3" else 0.55,
                )
                for xi, v in zip(x + offsets[(metric, group)], vals):
                    ax.text(xi, v + 0.002, f"{v:.3f}", ha="center", fontsize=7, rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels(["MARS", "KNN", "ANN"])
        ax.set_ylabel(f"Metric drop at σ={focus}")
        ax.set_title(f"Accuracy & F1 Drop (σ={focus})")
        ax.legend(fontsize=7, ncol=2)
        despine(ax)

    fig.suptitle(
        "Targeted Feature Perturbation: Top-3 Predictive vs Bottom-5 Features\n"
        "Top-3: Excess Kurt. IP, Skewness IP, Mean DM-SNR",
        fontsize=13,
        fontweight="bold",
        y=1.03,
    )
    fig.tight_layout()
    save_figure(fig, out_path)


def CLASS0_FALLBACK() -> str:
    return "#2B5B84"


def run_robustness(
    models: dict[str, Any],
    X_train,
    X_test,
    y_test,
    config: dict[str, Any] | None = None,
    quick: bool = False,
) -> pd.DataFrame:
    """Run Experiments A–C and save tables + 300 DPI plots."""
    if config is None:
        config = load_config()
    rob = config["robustness"]
    seed = config["seed"]
    figures_dir = PROJECT_ROOT / config["paths"]["figures_dir"]
    metrics_dir = PROJECT_ROOT / config["paths"]["metrics_dir"]
    apply_pub_style()

    sigma_levels = list(rob["sigma_levels"])
    n_mc = int(rob["n_monte_carlo"])
    if quick:
        sigma_levels = [0.0, 0.1, 0.3]
        n_mc = 3

    # Train-set feature std ONLY — never touch X_train values / never refit on noise
    train_std = _as_array(X_train).std(axis=0)
    train_std = np.where(train_std < 1e-12, 1.0, train_std)

    clip = True
    _, thresholds = experiment_c_threshold_stress(
        models, X_test, y_test, train_std, [0.0], seed, clip_nonnegative_std=clip
    )

    res_a = experiment_a_gaussian_noise(
        models, X_test, y_test, train_std, sigma_levels, n_mc, seed, thresholds, clip
    )
    res_b = experiment_b_high_leverage(
        models,
        X_test,
        y_test,
        train_std,
        FEATURE_NAMES,
        list(rob["top_features"]),
        sigma_levels,
        n_mc,
        seed,
        clip,
    )
    res_c, thresholds = experiment_c_threshold_stress(
        models, X_test, y_test, train_std, sigma_levels, seed, clip
    )

    res_a.to_csv(metrics_dir / "robustness_results.csv", index=False)
    res_b.to_csv(metrics_dir / "robustness_high_leverage.csv", index=False)
    res_c.to_csv(metrics_dir / "robustness_threshold_stress.csv", index=False)
    save_json(
        {
            "thresholds_pr_optimal": thresholds,
            "sigma_levels": sigma_levels,
            "n_mc": n_mc,
            "perturbation_policy": (
                "Raw feature-space Gaussian noise on X_test only; "
                "models/scalers never refit on noisy data. "
                "std_* features clipped to ≥0 after noise for physical plausibility; "
                "kurtosis/skewness/means left unbounded."
            ),
            "clip_nonnegative_std": clip,
            "top_features": list(rob["top_features"]),
            "train_feature_std": {
                f: float(s) for f, s in zip(FEATURE_NAMES, train_std)
            },
        },
        metrics_dir / "robustness_meta.json",
    )
    plot_robustness_curves(res_a, figures_dir / "08_robustness_degradation.png")
    plot_targeted_perturbation(res_b, figures_dir / "09_targeted_feature_perturbation.png")

    print(f"[robustness] Saved results → {metrics_dir / 'robustness_results.csv'}")
    return res_a
