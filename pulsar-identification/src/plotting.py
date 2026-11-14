"""Publication-grade plotting style: 300 DPI, vibrant colorblind-safe palette."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Cohesive colorblind-safe palette ----
CLASS0 = "#2B5B84"  # Deep Slate Blue — Non-Pulsar / RFI
CLASS1 = "#E63946"  # Vibrant Coral — Pulsar
CLASS0_LIGHT = "#3B82F6"
CLASS1_LIGHT = "#F97316"

MODEL_COLORS = {
    "MARS": "#0D9488",  # Vivid Teal / Emerald
    "KNN": "#6366F1",  # Royal Indigo / Violet
    "ANN": "#D97706",  # Vibrant Amber
}

DPI = 300
PREVALENCE = 1639 / 17898  # ≈ 0.0916


def apply_pub_style() -> None:
    """Set global rcParams for Nature/IEEE-style figures."""
    import warnings

    warnings.filterwarnings(
        "ignore",
        category=PendingDeprecationWarning,
        module=r"seaborn\..*",
    )
    warnings.filterwarnings(
        "ignore",
        category=mpl.MatplotlibDeprecationWarning,
        module=r"seaborn\..*",
    )
    mpl.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.labelweight": "bold",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "axes.axisbelow": True,
        }
    )
    sns.set_style("whitegrid", {"axes.grid": True, "grid.linestyle": "--"})


def save_figure(fig: plt.Figure, path: Path | str, **kwargs: Any) -> Path:
    """Save figure at 300 DPI with white background and tight bounding box."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        **kwargs,
    )
    plt.close(fig)
    # Guarantee PNG DPI metadata is readable by PIL (some backends omit it)
    try:
        from PIL import Image

        with Image.open(path) as im:
            im.load()
            im.save(path, dpi=(DPI, DPI))
    except Exception:
        pass
    return path


def despine(ax) -> None:
    sns.despine(ax=ax)


def model_color(name: str) -> str:
    return MODEL_COLORS.get(name, "#334155")


FEATURE_SHORT_LABELS = {
    "mean_integrated_profile": "Mean IP",
    "std_integrated_profile": "Std IP",
    "excess_kurtosis_integrated_profile": "Excess Kurt. IP",
    "skewness_integrated_profile": "Skewness IP",
    "mean_dm_snr_curve": "Mean DM-SNR",
    "std_dm_snr_curve": "Std DM-SNR",
    "excess_kurtosis_dm_snr_curve": "Excess Kurt. DM-SNR",
    "skewness_dm_snr_curve": "Skewness DM-SNR",
}

FEATURE_DESCRIPTIONS = {
    "mean_integrated_profile": "Mean of the integrated pulse profile",
    "std_integrated_profile": "Standard deviation of the integrated pulse profile",
    "excess_kurtosis_integrated_profile": "Excess kurtosis of the integrated pulse profile",
    "skewness_integrated_profile": "Skewness of the integrated pulse profile",
    "mean_dm_snr_curve": "Mean of the DM-SNR curve",
    "std_dm_snr_curve": "Standard deviation of the DM-SNR curve",
    "excess_kurtosis_dm_snr_curve": "Excess kurtosis of the DM-SNR curve",
    "skewness_dm_snr_curve": "Skewness of the DM-SNR curve",
}
