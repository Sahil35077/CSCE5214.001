"""Exploratory & statistical analysis – publication-grade 300 DPI figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src.data_loader import FEATURE_NAMES, TARGET_NAME
from src.plotting import (
    CLASS0,
    CLASS1,
    FEATURE_SHORT_LABELS,
    apply_pub_style,
    despine,
    save_figure,
)
from src.utils import PROJECT_ROOT, load_config


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in list(FEATURE_NAMES) + [TARGET_NAME]:
        s = df[col]
        rows.append(
            {
                "feature": col,
                "count": int(s.count()),
                "missing": int(s.isna().sum()),
                "mean": float(s.mean()),
                "std": float(s.std()),
                "min": float(s.min()),
                "25%": float(s.quantile(0.25)),
                "50%": float(s.quantile(0.50)),
                "75%": float(s.quantile(0.75)),
                "max": float(s.max()),
                "skew": float(stats.skew(s.dropna())),
                "kurtosis": float(stats.kurtosis(s.dropna())),
            }
        )
    return pd.DataFrame(rows)


def plot_correlation_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    apply_pub_style()
    cols = FEATURE_NAMES + [TARGET_NAME]
    corr = df[cols].corr(method="pearson")
    labels = [FEATURE_SHORT_LABELS.get(c, c) for c in FEATURE_NAMES] + ["Target"]

    fig, ax = plt.subplots(figsize=(10, 8.5))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"size": 9, "weight": "bold"},
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_title(
        "Pearson Correlation Heatmap — HTRU2 Features + Target\n"
        r"$N = 17{,}898$ (Official UCI)",
        pad=12,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    save_figure(fig, out_path)


def plot_boxplots_by_class(df: pd.DataFrame, out_path: Path) -> None:
    apply_pub_style()
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5))
    axes = axes.ravel()
    for i, feat in enumerate(FEATURE_NAMES):
        ax = axes[i]
        sns.boxplot(
            data=df,
            x=TARGET_NAME,
            y=feat,
            ax=ax,
            hue=TARGET_NAME,
            palette={0: CLASS0, 1: CLASS1},
            legend=False,
            width=0.55,
            fliersize=2,
            linewidth=1.2,
        )
        # Light strip for raincloud effect
        sns.stripplot(
            data=df.sample(n=min(2000, len(df)), random_state=42),
            x=TARGET_NAME,
            y=feat,
            ax=ax,
            hue=TARGET_NAME,
            palette={0: CLASS0, 1: CLASS1},
            legend=False,
            size=1.5,
            alpha=0.25,
            jitter=0.25,
        )
        for cls in (0, 1):
            med = df.loc[df[TARGET_NAME] == cls, feat].median()
            ax.text(
                cls,
                med,
                f"{med:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color="black",
            )
        ax.set_title(FEATURE_SHORT_LABELS[feat], fontsize=11, fontweight="bold")
        ax.set_xlabel("Class")
        ax.set_ylabel("")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["0\nNon-Pulsar", "1\nPulsar"], fontsize=8)
        despine(ax)
    fig.suptitle(
        "Feature Distributions by Class (boxplot + strip)\n"
        r"$N = 17{,}898$  |  Class 0 = Non-Pulsar (RFI)  ·  Class 1 = Pulsar",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    save_figure(fig, out_path)


def plot_kde_by_class(df: pd.DataFrame, out_path: Path) -> None:
    apply_pub_style()
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5))
    axes = axes.ravel()
    for i, feat in enumerate(FEATURE_NAMES):
        ax = axes[i]
        for cls, color, label in [
            (0, CLASS0, "Non-Pulsar (0)"),
            (1, CLASS1, "Pulsar (1)"),
        ]:
            subset = df.loc[df[TARGET_NAME] == cls, feat].dropna()
            sns.kdeplot(
                subset,
                ax=ax,
                color=color,
                label=label,
                fill=True,
                alpha=0.45,
                linewidth=2.2,
            )
            ax.axvline(
                subset.mean(),
                color=color,
                linestyle="--",
                linewidth=1.4,
                alpha=0.85,
            )
        ax.set_title(FEATURE_SHORT_LABELS[feat], fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Density" if i % 4 == 0 else "")
        if i == 0:
            ax.legend(frameon=True, fancybox=False, edgecolor="#e2e8f0")
        despine(ax)
    fig.suptitle(
        "KDE Densities by Class (dashed lines = class means)\n"
        r"$N = 17{,}898$",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    save_figure(fig, out_path)


def plot_class_balance(df: pd.DataFrame, out_path: Path) -> None:
    """Clear two-bar class balance chart — labels never collide with the title."""
    apply_pub_style()
    counts = df[TARGET_NAME].value_counts().sort_index()
    labels = ["Non-Pulsar (0)", "Pulsar (1)"]
    colors = [CLASS0, CLASS1]
    n_total = int(len(df))
    prevalence = 100.0 * counts.loc[1] / n_total

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    x = np.arange(len(labels))
    bars = ax.bar(
        x,
        counts.values,
        color=colors,
        width=0.52,
        edgecolor="white",
        linewidth=1.5,
        zorder=3,
    )

    ymax = float(counts.max())
    # Extra headroom so value labels sit clearly below the title block
    ax.set_ylim(0, ymax * 1.28)

    for i, (b, v) in enumerate(zip(bars, counts.values)):
        pct = 100.0 * v / n_total
        # Tall majority bar: put count inside near the top (white); short bar: above
        if v > ymax * 0.45:
            ax.text(
                b.get_x() + b.get_width() / 2,
                v * 0.92,
                f"{v:,}",
                ha="center",
                va="top",
                fontsize=13,
                fontweight="bold",
                color="white",
                zorder=4,
            )
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + ymax * 0.035,
                f"{pct:.2f}% of sample",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color=colors[i],
                zorder=4,
            )
        else:
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + ymax * 0.04,
                f"{v:,}\n{pct:.2f}% of sample",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
                color=colors[i],
                zorder=4,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_ylabel("Count", fontsize=11, fontweight="bold")
    ax.set_xlabel("")

    # Title on axes; metadata sits above in its own band (no collision with bar labels)
    ax.set_title("HTRU2 Class Balance", fontsize=15, fontweight="bold", pad=22)
    fig.suptitle(
        rf"Official UCI  ·  $N = {n_total:,}$  ·  "
        rf"Prevalence $= {prevalence:.2f}\%$  ·  "
        rf"Imbalance ≈ ${counts.loc[0]/counts.loc[1]:.2f}$:1",
        fontsize=10,
        color="#475569",
        y=0.98,
    )

    ax.set_xlim(-0.6, 1.6)
    despine(ax)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.92))
    save_figure(fig, out_path)


def run_eda(df: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Generate all EDA tables and publication figures (01–03 + class balance)."""
    if config is None:
        config = load_config()
    figures_dir = PROJECT_ROOT / config["paths"]["figures_dir"]
    metrics_dir = PROJECT_ROOT / config["paths"]["metrics_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    apply_pub_style()

    summary = summary_statistics(df)
    summary.to_csv(metrics_dir / "eda_summary.csv", index=False)

    plot_class_balance(df, figures_dir / "00_class_balance.png")
    plot_correlation_heatmap(df, figures_dir / "01_correlation_heatmap.png")
    plot_boxplots_by_class(df, figures_dir / "02_feature_boxplots.png")
    plot_kde_by_class(df, figures_dir / "03_feature_kde_distributions.png")

    print(f"[eda] Saved summary → {metrics_dir / 'eda_summary.csv'}")
    print(f"[eda] Figures (300 DPI) → {figures_dir}")
    return summary
