"""Holdout evaluation, dual pos_label metrics, McNemar tests, 300 DPI figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from statsmodels.stats.contingency_tables import mcnemar

from src.data_loader import FEATURE_NAMES
from src.plotting import (
    CLASS0,
    CLASS1,
    FEATURE_SHORT_LABELS,
    MODEL_COLORS,
    PREVALENCE,
    apply_pub_style,
    despine,
    model_color,
    save_figure,
)
from src.utils import PROJECT_ROOT, load_config, save_json


def _specificity(y_true, y_pred, pos_label: int = 1) -> float:
    """Specificity relative to the stated positive label."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    if pos_label == 1:
        # Spec = TN/(TN+FP) when Pulsar is positive
        return float(tn / (tn + fp)) if (tn + fp) else 0.0
    # When Non-Pulsar (0) is positive: specificity = recall of the other class
    return float(tp / (tp + fn)) if (tp + fn) else 0.0


def compute_metrics(y_true, y_pred, y_proba, pos_label: int = 1) -> dict[str, float]:
    """
    Compute classification metrics for a chosen positive class.

    Primary project metrics MUST use pos_label=1 (Pulsar).
    pos_label=0 mirrors R caret Tables II/IV/VI (Positive Class : 0).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    y_proba = np.asarray(y_proba).ravel()

    if pos_label == 0:
        y_true_m = 1 - y_true
        y_pred_m = 1 - y_pred
        y_proba_m = 1.0 - y_proba
    else:
        y_true_m = y_true
        y_pred_m = y_pred
        y_proba_m = y_proba

    return {
        "pos_label": int(pos_label),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true_m, y_pred_m, pos_label=1, zero_division=0)
        ),
        "recall": float(recall_score(y_true_m, y_pred_m, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true_m, y_pred_m, pos_label=1, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true_m, y_proba_m)),
        "roc_auc": float(roc_auc_score(y_true_m, y_proba_m)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "specificity": _specificity(y_true, y_pred, pos_label=pos_label),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def confusion_counts(y_true, y_pred) -> dict[str, int]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}


def mcnemar_test(y_true, y_pred_a, y_pred_b) -> dict[str, float]:
    """
    Pairwise McNemar test on holdout predictions.

    Contingency of discordant pairs:
      b = A correct & B wrong
      c = A wrong & B correct
    Uses exact binomial test when b+c < 25, else continuity-corrected chi-square.
    """
    y_true = np.asarray(y_true).ravel()
    a_correct = np.asarray(y_pred_a).ravel() == y_true
    b_correct = np.asarray(y_pred_b).ravel() == y_true
    b = int(np.sum(a_correct & ~b_correct))
    c = int(np.sum(~a_correct & b_correct))
    # statsmodels expects 2x2; only discordant cells matter for McNemar
    table = np.array([[0, b], [c, 0]])
    if (b + c) < 25:
        result = mcnemar(table, exact=True)
        method = "exact"
    else:
        result = mcnemar(table, exact=False, correction=True)
        method = "corrected"
    return {
        "n_a_correct_b_wrong": b,
        "n_a_wrong_b_correct": c,
        "statistic": float(result.statistic),
        "pvalue": float(result.pvalue),
        "method": method,
    }


def plot_cv_tuning(
    mars_details: dict,
    knn_details: dict,
    ann_details: dict | None,
    figures_dir: Path,
) -> None:
    """
    Properly arranged CV tuning figure:

        ┌──────────────┬──────────────┐
        │     MARS     │     KNN      │
        └──────────────┴──────────────┘
        ┌─────────────────────────────┐
        │            ANN              │
        └─────────────────────────────┘
    """
    apply_pub_style()
    fig = plt.figure(figsize=(12.5, 9.4))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.2],
        hspace=0.42,
        wspace=0.28,
        left=0.09,
        right=0.97,
        top=0.90,
        bottom=0.14,
    )
    ax_mars = fig.add_subplot(gs[0, 0])
    ax_knn = fig.add_subplot(gs[0, 1])
    ax_ann = fig.add_subplot(gs[1, :])

    # ---- MARS ----
    if mars_details and "cv_results" in mars_details:
        params = mars_details["cv_results"]["params"]
        xs = np.array([p["clf__nprune"] for p in params], dtype=float)
        scores = np.array(mars_details["cv_results"]["mean_test_score"], dtype=float)
        stds = np.array(mars_details["cv_results"]["std_test_score"], dtype=float)
        order = np.argsort(xs)
        xs, scores, stds = xs[order], scores[order], stds[order]
        color = MODEL_COLORS["MARS"]
        ax_mars.fill_between(xs, scores - stds, scores + stds, color=color, alpha=0.18, zorder=1)
        ax_mars.plot(xs, scores, "o-", color=color, lw=2.6, markersize=8, zorder=3, label="CV PR-AUC")
        best = mars_details.get("best_params", {}).get("clf__nprune", 14)
        best_score = float(scores[list(xs).index(best)]) if best in xs else float(scores[-1])
        ax_mars.axvline(14, color="#94a3b8", ls="--", lw=1.5, zorder=2, label="Paper nprune=14")
        ax_mars.scatter(
            [best], [best_score], marker="*", s=340, c="#FBBF24",
            edgecolors="#0f172a", linewidths=0.8, zorder=5, label=f"Selected nprune={best}",
        )
        ax_mars.annotate(
            f"{best_score:.3f}",
            xy=(best, best_score),
            xytext=(-18, 12),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color=color,
        )
        ax_mars.set_ylim(
            max(0.85, float(scores.min() - stds.max()) - 0.01),
            min(1.0, float(scores.max() + stds.max()) + 0.025),
        )
        ax_mars.set_xticks(xs.astype(int))
    ax_mars.set_xlabel("nprune", fontsize=11, fontweight="bold")
    ax_mars.set_ylabel("CV PR-AUC (mean ± 1 SD)", fontsize=11, fontweight="bold")
    ax_mars.set_title("(a)  MARS", color=MODEL_COLORS["MARS"], fontsize=13, fontweight="bold", loc="left")
    ax_mars.legend(fontsize=8, frameon=True, fancybox=False, edgecolor="#e2e8f0", loc="lower right")
    despine(ax_mars)

    # ---- KNN (PR-AUC primary; Accuracy as secondary panel inset note) ----
    if knn_details and "cv_results" in knn_details:
        params = knn_details["cv_results"]["params"]
        ks = np.array([p["clf__n_neighbors"] for p in params], dtype=int)
        order = np.argsort(ks)
        ks = ks[order]
        pr = np.array(knn_details["cv_results"]["mean_test_score_pr_auc"], dtype=float)[order]
        pr_std = np.array(
            knn_details["cv_results"].get(
                "std_test_score_pr_auc",
                [0.0] * len(pr),
            ),
            dtype=float,
        )[order]
        acc = np.array(knn_details["cv_results"]["mean_test_score_accuracy"], dtype=float)[order]
        paper_k = int(knn_details.get("paper_optimal_k", 15))
        sel = int(knn_details.get("selected_k", paper_k))
        color = MODEL_COLORS["KNN"]

        ax_knn.fill_between(ks, pr - pr_std, pr + pr_std, color=color, alpha=0.18, zorder=1)
        ax_knn.plot(ks, pr, "o-", color=color, lw=2.6, markersize=7, zorder=3, label="CV PR-AUC")
        ax_knn.axvline(paper_k, color="#94a3b8", ls="--", lw=1.5, zorder=2, label=f"Paper k={paper_k}")
        if sel in list(ks):
            si = list(ks).index(sel)
            ax_knn.scatter(
                [sel], [pr[si]], marker="*", s=340, c="#FBBF24",
                edgecolors="#0f172a", linewidths=0.8, zorder=5, label=f"Selected k={sel}",
            )
            ax_knn.annotate(
                f"{pr[si]:.3f}",
                xy=(sel, pr[si]),
                xytext=(-22, 12),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                color=color,
            )
        # Accuracy callout (avoid twin-axis clutter)
        if paper_k in list(ks) and sel in list(ks):
            ax_knn.text(
                0.03,
                0.97,
                f"Accuracy @ k={paper_k}: {acc[list(ks).index(paper_k)]:.3f}\n"
                f"Accuracy @ k={sel}: {acc[list(ks).index(sel)]:.3f}",
                transform=ax_knn.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="#334155",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#f8fafc", edgecolor="#e2e8f0"),
            )
        ax_knn.set_ylim(0.70, 0.94)
        ax_knn.set_xticks(ks)
    ax_knn.set_xlabel("k (n_neighbors)", fontsize=11, fontweight="bold")
    ax_knn.set_ylabel("CV PR-AUC (mean ± 1 SD)", fontsize=11, fontweight="bold")
    ax_knn.set_title("(b)  KNN", color=MODEL_COLORS["KNN"], fontsize=13, fontweight="bold", loc="left")
    ax_knn.legend(fontsize=8, frameon=True, fancybox=False, edgecolor="#e2e8f0", loc="center right")
    despine(ax_knn)

    # ---- ANN (full-width vertical bars — readable labels) ----
    if ann_details and "cv_results" in ann_details:
        params = ann_details["cv_results"]["params"]
        scores = np.array(ann_details["cv_results"]["mean_test_score"], dtype=float)
        stds = np.array(ann_details["cv_results"]["std_test_score"], dtype=float)

        def _short_label(p: dict) -> str:
            h = p.get("clf__hidden_layer_sizes", "?")
            if isinstance(h, (list, tuple)):
                h = "–".join(str(x) for x in h)
            a = str(p.get("clf__activation", ""))
            al = p.get("clf__alpha", "")
            al_s = f"{al:g}" if isinstance(al, float) else str(al)
            return f"{h}\n{a}, α={al_s}"

        order = np.argsort(scores)[::-1][:8]
        labels = [_short_label(params[i]) for i in order]
        sc = scores[order]
        sd = stds[order]
        x = np.arange(len(order))
        colors = [MODEL_COLORS["ANN"] if i == 0 else "#FDBA74" for i in range(len(order))]
        edges = ["#0f172a" if i == 0 else "white" for i in range(len(order))]

        ax_ann.bar(
            x,
            sc,
            yerr=sd,
            color=colors,
            edgecolor=edges,
            linewidth=1.2,
            width=0.72,
            ecolor="#64748b",
            capsize=3,
            zorder=3,
        )
        for xi, (s, e) in enumerate(zip(sc, sd)):
            ax_ann.text(
                xi,
                s + e + 0.0012,
                f"{s:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )
        ax_ann.set_xticks(x)
        ax_ann.set_xticklabels(labels, fontsize=8)
        lo = float(sc.min() - sd.max()) - 0.006
        hi = float(sc.max() + sd.max()) + 0.012
        ax_ann.set_ylim(max(0.90, lo), min(0.955, hi))
        ax_ann.axhline(sc[0], color="#FBBF24", ls=":", lw=1.6, label=f"Best = {sc[0]:.3f}")
        ax_ann.legend(fontsize=9, loc="upper right", frameon=True, fancybox=False, edgecolor="#e2e8f0")
    ax_ann.set_ylabel("CV PR-AUC (zoomed · mean ± 1 SD)", fontsize=11, fontweight="bold")
    ax_ann.set_xlabel(
        "Architecture · activation · L2 α   (ranked best → worst)",
        fontsize=11,
        fontweight="bold",
        labelpad=10,
    )
    ax_ann.set_title("(c)  ANN", color=MODEL_COLORS["ANN"], fontsize=13, fontweight="bold", loc="left")
    despine(ax_ann)

    fig.suptitle(
        "Cross-Validation Hyperparameter Tuning  ·  RepeatedStratifiedKFold (10×3 = 30 folds)",
        fontsize=14,
        fontweight="bold",
        y=0.965,
    )
    fig.text(
        0.5,
        0.025,
        "Selection criterion = average precision (PR-AUC).  "
        "Stars = CV-selected settings; dashed lines = paper defaults (nprune=14, k=15).",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#64748b",
    )
    save_figure(fig, figures_dir / "04_cv_hyperparameter_tuning.png")


def plot_feature_importance(
    importances: dict[str, np.ndarray],
    feature_names: list[str],
    figures_dir: Path,
) -> None:
    """
    Grouped horizontal bars — one shared ranking, three model colors —
    for direct apples-to-apples comparison (replaces 3 separate lollipops).
    """
    apply_pub_style()
    model_names = list(importances.keys())
    short = np.array([FEATURE_SHORT_LABELS.get(f, f) for f in feature_names])

    # Normalize each model to relative share in [0,1]
    mat = []
    for name in model_names:
        imp = np.asarray(importances[name], dtype=float)
        imp = np.maximum(imp, 0.0)  # clip tiny negatives from perm. noise
        s = imp.sum()
        mat.append(imp / s if s > 0 else imp)
    mat = np.vstack(mat)  # (n_models, n_features)

    # Rank features by mean importance (descending → index 0 at top of plot)
    mean_imp = mat.mean(axis=0)
    order = np.argsort(mean_imp)[::-1]
    short_o = short[order]
    mat_o = mat[:, order]

    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    n_feat = len(feature_names)
    n_models = len(model_names)
    y = np.arange(n_feat)
    height = 0.24
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * height * 1.15

    # Highlight dominant feature row (y=0)
    ax.axhspan(-0.48, 0.48, color="#FEF3C7", alpha=0.55, zorder=0)

    for m_i, name in enumerate(model_names):
        vals = mat_o[m_i]
        bars = ax.barh(
            y + offsets[m_i],
            vals,
            height=height,
            color=model_color(name),
            edgecolor="white",
            linewidth=0.8,
            label=name,
            zorder=3,
        )
        for bar, v in zip(bars, vals):
            if v >= 0.05:
                ax.text(
                    v + 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:.2f}",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color=model_color(name),
                )

    ax.text(
        0.98,
        0.0,
        "★ Dominant\nacross models",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="center",
        fontsize=8,
        color="#92400E",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFBEB", edgecolor="#F59E0B", linewidth=1.0),
        zorder=5,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(short_o, fontsize=10)
    ax.invert_yaxis()  # y=0 (most important) at top
    ax.set_xlabel("Relative permutation importance (share of total)", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1.08)
    ax.legend(
        title="Model",
        fontsize=10,
        title_fontsize=10,
        frameon=True,
        fancybox=False,
        edgecolor="#e2e8f0",
        loc="lower right",
    )
    ax.set_title(
        "Permutation Feature Importance — MARS vs KNN vs ANN",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    fig.text(
        0.5,
        0.02,
        r"$N_{\mathrm{test}} = 3{,}580$  ·  scoring = average precision  ·  "
        "bars = within-model normalized share  ·  features ranked by mean importance",
        ha="center",
        fontsize=9,
        color="#64748b",
    )
    despine(ax)
    fig.tight_layout(rect=(0.02, 0.05, 0.98, 0.95))
    save_figure(fig, figures_dir / "05_feature_importance_comparison.png")


def plot_confusion_matrices_panel(
    pred_dict: dict[str, np.ndarray],
    y_true: np.ndarray,
    figures_dir: Path,
) -> None:
    """
    Publication confusion-matrix panel.

    Colored by *row-normalized rates* (so class imbalance does not wash out
    the Pulsar row), annotated with TN/FP/FN/TP tags + counts + %, and a
    compact metric footer under each model.
    """
    apply_pub_style()
    names = list(pred_dict.keys())
    n_test = int(len(y_true))
    n_pos = int(np.sum(y_true == 1))
    n_neg = n_test - n_pos

    # Equal-width panels + dedicated colorbar axis (avoids squeezing ANN)
    fig = plt.figure(figsize=(14.5, 6.6))
    gs = fig.add_gridspec(
        1,
        len(names) + 1,
        width_ratios=[1] * len(names) + [0.06],
        wspace=0.32,
        left=0.07,
        right=0.95,
        top=0.82,
        bottom=0.20,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(len(names))]
    cax = fig.add_subplot(gs[0, -1])

    cell_tags = np.array([["TN", "FP"], ["FN", "TP"]])
    last_hm = None

    for idx, (ax, name) in enumerate(zip(axes, names)):
        y_pred = pred_dict[name]
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        cm_rate = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

        annot = np.empty((2, 2), dtype=object)
        for i in range(2):
            for j in range(2):
                annot[i, j] = (
                    f"{cell_tags[i, j]}\n"
                    f"{cm[i, j]:,}\n"
                    f"{100 * cm_rate[i, j]:.1f}%"
                )

        last_hm = sns.heatmap(
            cm_rate,
            annot=annot,
            fmt="",
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
            cbar=False,
            ax=ax,
            square=True,
            linewidths=3.0,
            linecolor="white",
            xticklabels=["Non-Pulsar (0)", "Pulsar (1)"],
            yticklabels=["Non-Pulsar (0)", "Pulsar (1)"] if idx == 0 else ["", ""],
            annot_kws={"size": 12, "weight": "bold", "va": "center", "linespacing": 1.35},
        )

        for text, rate in zip(ax.texts, cm_rate.ravel()):
            text.set_color("white" if rate >= 0.55 else "#0f172a")

        acc = (tn + tp) / n_test
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0

        ax.set_title(name, color=model_color(name), fontsize=15, fontweight="bold", pad=12)
        ax.set_xlabel("Predicted", fontsize=10, fontweight="bold", labelpad=10)
        ax.set_ylabel("Actual" if idx == 0 else "", fontsize=10, fontweight="bold", labelpad=8)
        ax.tick_params(axis="both", labelsize=9, length=0)
        plt.setp(ax.get_xticklabels(), rotation=0)
        plt.setp(ax.get_yticklabels(), rotation=0, va="center")

        footer = (
            f"Acc {acc:.3f}  ·  Prec {prec:.3f}  ·  Rec {rec:.3f}\n"
            f"F1 {f1:.3f}  ·  Spec {spec:.3f}  ·  FP={fp}  FN={fn}"
        )
        ax.text(
            0.5,
            -0.30,
            footer,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            color="#334155",
            bbox=dict(
                boxstyle="round,pad=0.45",
                facecolor="#f8fafc",
                edgecolor=model_color(name),
                linewidth=1.5,
            ),
        )

    fig.colorbar(last_hm.collections[0], cax=cax, label="Row rate")

    fig.suptitle(
        "Holdout Confusion Matrices — Counts, Row Rates & Pulsar Metrics",
        fontsize=15,
        fontweight="bold",
        y=0.97,
    )
    fig.text(
        0.5,
        0.915,
        rf"$N_{{\mathrm{{test}}}} = {n_test:,}$  "
        rf"(Non-Pulsar $= {n_neg:,}$, Pulsar $= {n_pos:,}$)  ·  "
        r"Cell color = row-normalized rate  ·  Positive class = Pulsar (1)",
        ha="center",
        va="top",
        fontsize=9.5,
        color="#475569",
    )
    save_figure(fig, figures_dir / "06_confusion_matrices.png")


def plot_pr_roc_curves(
    y_true,
    proba_dict: dict[str, np.ndarray],
    figures_dir: Path,
) -> None:
    apply_pub_style()
    y_true = np.asarray(y_true).ravel()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for name, proba in proba_dict.items():
        p, r, _ = precision_recall_curve(y_true, proba)
        ap = average_precision_score(y_true, proba)
        ax.plot(r, p, lw=2.4, color=model_color(name), label=f"{name} (PR-AUC={ap:.3f})")
    ax.axhline(PREVALENCE, color="#94a3b8", ls="--", lw=1.5,
               label=f"prevalence baseline ({PREVALENCE:.4f})")
    ax.set_xlabel("Recall (Pulsar, pos=1)")
    ax.set_ylabel("Precision (Pulsar, pos=1)")
    ax.set_title("Precision–Recall Curves")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", fontsize=9)
    despine(ax)

    ax = axes[1]
    for name, proba in proba_dict.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc = roc_auc_score(y_true, proba)
        ax.plot(fpr, tpr, lw=2.4, color=model_color(name), label=f"{name} (ROC-AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.3, alpha=0.5, label="random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Pulsar Recall)")
    ax.set_title("ROC Curves")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=9)
    despine(ax)

    fig.suptitle(
        r"Holdout Discrimination  ·  $N_{\mathrm{test}} = 3{,}580$  ·  pos_label = 1 (Pulsar)",
        fontsize=14,
        fontweight="bold",
        y=1.03,
    )
    fig.tight_layout()
    save_figure(fig, figures_dir / "07_pr_and_roc_curves.png")


def evaluate_models(
    models: dict[str, Any],
    X_test,
    y_test,
    X_train=None,
    y_train=None,
    config: dict[str, Any] | None = None,
    mars_details: dict | None = None,
    knn_details: dict | None = None,
    ann_details: dict | None = None,
) -> pd.DataFrame:
    """Evaluate all models; log pos_label=1 (primary) AND pos_label=0 (paper-aligned)."""
    if config is None:
        config = load_config()
    figures_dir = PROJECT_ROOT / config["paths"]["figures_dir"]
    metrics_dir = PROJECT_ROOT / config["paths"]["metrics_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    apply_pub_style()

    y_test = np.asarray(y_test).ravel()
    rows_pos1 = []
    rows_pos0 = []
    cm_rows = []
    proba_dict = {}
    pred_dict = {}
    importances = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        proba_dict[name] = y_proba
        pred_dict[name] = y_pred

        m1 = compute_metrics(y_test, y_pred, y_proba, pos_label=1)
        m0 = compute_metrics(y_test, y_pred, y_proba, pos_label=0)
        cm = confusion_counts(y_test, y_pred)

        rows_pos1.append({"model": name, "metric_family": "PRIMARY_pos_label_1_Pulsar", **{
            k: v for k, v in m1.items() if k != "pos_label"
        }, "pulsar_recall_from_cm": cm["TP"] / (cm["TP"] + cm["FN"]) if (cm["TP"] + cm["FN"]) else 0.0})
        rows_pos0.append({"model": name, "metric_family": "PAPER_ALIGNED_pos_label_0_NonPulsar", **{
            k: v for k, v in m0.items() if k != "pos_label"
        }})
        cm_rows.append({"model": name, **cm})

        # Permutation importance
        try:
            X_imp, y_imp = X_test, y_test
            if len(X_test) > 2000:
                rng = np.random.default_rng(config["seed"])
                idx = rng.choice(len(X_test), size=2000, replace=False)
                X_imp = X_test.iloc[idx] if hasattr(X_test, "iloc") else X_test[idx]
                y_imp = y_test[idx]
            r = permutation_importance(
                model, X_imp, y_imp, n_repeats=10, random_state=config["seed"],
                scoring="average_precision", n_jobs=-1,
            )
            importances[name] = r.importances_mean
        except Exception as exc:  # noqa: BLE001
            print(f"[evaluate] permutation importance failed for {name}: {exc}")
            clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
            if hasattr(clf, "variable_importance"):
                importances[name] = clf.variable_importance()

    # Wide comparison table (explicit column prefixes — zero confusion)
    wide_rows = []
    for r1, r0, cm in zip(rows_pos1, rows_pos0, cm_rows):
        wide = {"model": r1["model"]}
        for k, v in r1.items():
            if k in ("model", "metric_family"):
                continue
            wide[f"pulsar_pos1__{k}"] = v
        for k, v in r0.items():
            if k in ("model", "metric_family"):
                continue
            wide[f"nonpulsar_pos0__{k}"] = v
        wide.update({f"cm__{k}": v for k, v in cm.items() if k != "model"})
        wide_rows.append(wide)

    comparison = pd.DataFrame(wide_rows)
    comparison.to_csv(metrics_dir / "paper_comparison.csv", index=False)

    # Also save long-form separated tables for the report
    pd.DataFrame(rows_pos1).to_csv(metrics_dir / "metrics_pos1_pulsar.csv", index=False)
    pd.DataFrame(rows_pos0).to_csv(metrics_dir / "metrics_pos0_nonpulsar_paper_aligned.csv", index=False)
    pd.DataFrame(cm_rows).to_csv(metrics_dir / "confusion_matrices.csv", index=False)

    baselines = config.get("paper_baselines", {})
    baseline_rows = []
    for model_name, b in baselines.items():
        baseline_rows.append({
            "model": model_name.upper(),
            "source": "Abdullah et al. 2023 (caret Positive Class : 0)",
            **b,
            "dataset_note": "Paper: 12,528-row Kaggle + MICE; Ours: official 17,898 UCI",
        })
    if baseline_rows:
        pd.DataFrame(baseline_rows).to_csv(
            metrics_dir / "paper_baselines_reference.csv", index=False
        )

    # McNemar pairwise
    names = list(pred_dict.keys())
    mcnemar_rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            res = mcnemar_test(y_test, pred_dict[names[i]], pred_dict[names[j]])
            mcnemar_rows.append({"model_a": names[i], "model_b": names[j], **res})
    pd.DataFrame(mcnemar_rows).to_csv(metrics_dir / "mcnemar_tests.csv", index=False)

    plot_pr_roc_curves(y_test, proba_dict, figures_dir)
    plot_cv_tuning(mars_details or {}, knn_details or {}, ann_details, figures_dir)
    plot_confusion_matrices_panel(pred_dict, y_test, figures_dir)
    if importances:
        plot_feature_importance(importances, FEATURE_NAMES, figures_dir)
        pd.DataFrame(importances, index=FEATURE_NAMES).to_csv(
            metrics_dir / "feature_importance.csv"
        )

    save_json(
        {
            "n_test": int(len(y_test)),
            "models": list(models.keys()),
            "primary_metrics": "pos_label=1 (Pulsar) — columns pulsar_pos1__*",
            "paper_aligned_metrics": (
                "pos_label=0 (Non-Pulsar) — columns nonpulsar_pos0__* "
                "mirror caret Tables II/IV/VI"
            ),
            "prevalence": PREVALENCE,
        },
        metrics_dir / "evaluation_meta.json",
    )
    print(f"[evaluate] Saved paper_comparison.csv → {metrics_dir}")
    return comparison
