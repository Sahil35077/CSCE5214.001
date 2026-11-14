"""Fetch and validate the official UCI HTRU2 dataset (id=372)."""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from src.plotting import FEATURE_DESCRIPTIONS
from src.utils import PROJECT_ROOT, load_config

FEATURE_NAMES = [
    "mean_integrated_profile",
    "std_integrated_profile",
    "excess_kurtosis_integrated_profile",
    "skewness_integrated_profile",
    "mean_dm_snr_curve",
    "std_dm_snr_curve",
    "excess_kurtosis_dm_snr_curve",
    "skewness_dm_snr_curve",
]
TARGET_NAME = "target_class"
COLUMN_NAMES = FEATURE_NAMES + [TARGET_NAME]

# Physical meanings (4 integrated profile + 4 DM-SNR) — must match Lyon (2015) HTRU2
assert list(FEATURE_DESCRIPTIONS.keys()) == FEATURE_NAMES

UCI_CSV_URLS = [
    "https://archive.ics.uci.edu/static/public/372/htru2.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00372/HTRU2.zip",
]


def _assign_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure exact 9-column schema regardless of source headers."""
    if df.shape[1] == 9:
        out = df.copy()
        out.columns = COLUMN_NAMES
    elif set(FEATURE_NAMES).issubset(set(df.columns)) and (
        TARGET_NAME in df.columns or "class" in df.columns
    ):
        out = df.copy()
        if TARGET_NAME not in out.columns and "class" in out.columns:
            out = out.rename(columns={"class": TARGET_NAME})
        out = out[COLUMN_NAMES]
    else:
        raise ValueError(
            f"Unexpected schema with shape {df.shape} and columns {list(df.columns)}"
        )
    for c in FEATURE_NAMES:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out[TARGET_NAME] = pd.to_numeric(out[TARGET_NAME], errors="coerce").astype(int)
    # Refuse silent truncation / extra columns
    if out.shape[1] != 9:
        raise ValueError(f"Schema must have exactly 9 columns, got {out.shape}")
    return out.reset_index(drop=True)


def _load_from_ucimlrepo() -> tuple[pd.DataFrame | None, str]:
    try:
        from ucimlrepo import fetch_ucirepo

        htru2 = fetch_ucirepo(id=372)
        X = htru2.data.features
        y = htru2.data.targets
        if X is None or y is None:
            return None, "ucimlrepo returned empty features/targets"
        if len(X) != len(y):
            raise ValueError(f"Feature/target length mismatch: {len(X)} vs {len(y)}")
        df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
        if df.shape[1] == 9:
            df.columns = COLUMN_NAMES
        return _assign_schema(df), "ucimlrepo:id=372"
    except Exception as exc:  # noqa: BLE001
        print(f"[data_loader] ucimlrepo fetch failed: {exc}")
        return None, f"ucimlrepo_failed:{exc}"


def _load_from_local_csv(raw_dir: Path) -> tuple[pd.DataFrame | None, str]:
    candidates = [
        raw_dir / "HTRU_2.csv",
        raw_dir / "HTRU2.csv",
        raw_dir / "htru2.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path, header=None)
            if df.shape[1] == 9:
                return _assign_schema(df), f"local_csv:{path.name}"
            df = pd.read_csv(path)
            return _assign_schema(df), f"local_csv_header:{path.name}"
    return None, "no_local_csv"


def _download_uci_zip(raw_dir: Path) -> tuple[pd.DataFrame | None, str]:
    import io
    import zipfile

    raw_dir.mkdir(parents=True, exist_ok=True)
    for url in UCI_CSV_URLS:
        try:
            print(f"[data_loader] Downloading from {url} ...")
            with urllib.request.urlopen(url, timeout=90) as resp:
                data = resp.read()
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                # Prefer HTRU_2.csv specifically (avoid Readme / other files)
                csv_names = [
                    n
                    for n in zf.namelist()
                    if n.lower().endswith("htru_2.csv") or n.lower().endswith("htru2.csv")
                ]
                if not csv_names:
                    csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                if not csv_names:
                    continue
                with zf.open(csv_names[0]) as f:
                    df = pd.read_csv(f, header=None)
                if df.shape != (17898, 9):
                    raise ValueError(
                        f"Downloaded CSV shape {df.shape} != (17898, 9) — refusing silent truncation"
                    )
                out_csv = raw_dir / "HTRU_2.csv"
                df.to_csv(out_csv, index=False, header=False)
                print(f"[data_loader] Cached CSV → {out_csv}")
                return _assign_schema(df), f"uci_zip:{url}"
        except Exception as exc:  # noqa: BLE001
            print(f"[data_loader] Download failed ({url}): {exc}")
    return None, "download_failed"


def load_htru2(
    config: dict[str, Any] | None = None,
    raw_dir: Path | None = None,
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Load official UCI HTRU2 (17,898 rows).

    Priority: local cache → ucimlrepo → direct UCI zip download.
    Always validates exact shape before returning.
    """
    if config is None:
        config = load_config()
    if raw_dir is None:
        raw_dir = PROJECT_ROOT / config["paths"]["raw_dir"]
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    df = None
    source = "unknown"
    if not force_download:
        df, source = _load_from_local_csv(raw_dir)
    if df is None:
        df, source = _load_from_ucimlrepo()
        if df is not None:
            cache = raw_dir / "HTRU_2.csv"
            df.to_csv(cache, index=False, header=False)
            print(f"[data_loader] Cached via ucimlrepo → {cache}")
    if df is None:
        df, source = _download_uci_zip(raw_dir)
    if df is None:
        raise RuntimeError(
            "Could not load HTRU2. Place HTRU_2.csv (no header) in data/raw/ "
            "or install ucimlrepo and ensure network access."
        )

    # Hard invariants — never return silently truncated / corrupted data
    assert_dataset_invariants(df, config, source=source)
    df.attrs["provenance"] = source
    df.attrs["_invariants_checked"] = True
    return df


def assert_dataset_invariants(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
    source: str = "unknown",
) -> dict[str, Any]:
    """Print and assert exact UCI HTRU2 invariants. Raises AssertionError on failure."""
    if config is None:
        config = load_config()
    data_cfg = config["data"]

    n0 = int((df[TARGET_NAME] == 0).sum())
    n1 = int((df[TARGET_NAME] == 1).sum())
    n_nan = int(df.isna().sum().sum())
    n_dup = int(df.duplicated().sum())
    prevalence = n1 / len(df) if len(df) else 0.0

    print("[data_loader] === UCI HTRU2 Invariant Check ===")
    print(f"  provenance     : {source}")
    print(f"  shape          : {df.shape}  (expected (17898, 9))")
    print(f"  missing values : {n_nan}     (expected 0)")
    print(f"  duplicate rows : {n_dup}     (expected 0)")
    print(f"  class 0        : {n0} ({100 * n0 / len(df):.2f}%)  expected 16259 (90.84%)")
    print(f"  class 1        : {n1} ({100 * n1 / len(df):.2f}%)  expected 1639  (9.16%)")
    print("  features (X excludes target — zero target leakage):")
    for i, name in enumerate(FEATURE_NAMES):
        print(f"    [{i}] {name}: {FEATURE_DESCRIPTIONS[name]}")

    assert df.shape == (17898, 9), f"shape {df.shape} != (17898, 9)"
    assert list(df.columns) == COLUMN_NAMES, f"columns {list(df.columns)}"
    assert n_nan == 0, f"expected 0 NaNs, found {n_nan}"
    assert n_dup == 0, f"expected 0 duplicate rows, found {n_dup}"
    assert n0 == 16259, f"class0={n0}"
    assert n1 == 1639, f"class1={n1}"
    assert abs(prevalence - 0.091574) < 1e-5
    # X must never contain the target
    X = df[FEATURE_NAMES]
    assert TARGET_NAME not in X.columns
    assert X.shape == (17898, 8)

    return {
        "ok": True,
        "errors": [],
        "warnings": [],
        "provenance": source,
        "n_rows": 17898,
        "n_features": 8,
        "n_class_0": n0,
        "n_class_1": n1,
        "prevalence": prevalence,
        "prevalence_pct": round(100 * prevalence, 2),
        "class0_pct": round(100 * n0 / len(df), 2),
        "n_nan": n_nan,
        "n_duplicates": n_dup,
        "feature_descriptions": dict(FEATURE_DESCRIPTIONS),
        "paper_vs_uci_note": (
            f"Official UCI HTRU2 has {data_cfg['n_rows']} rows with 0 missing values. "
            f"The 2023 IEEE paper used a Kaggle train split of {data_cfg['paper_n_rows']} "
            f"rows (~70%) with ~{data_cfg['paper_missing_fraction']*100:.1f}% artificially "
            "injected missing values imputed via MICE."
        ),
    }


def validate_htru2(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
    *,
    silent: bool = False,
) -> dict[str, Any]:
    """Validate exact dataset invariants (non-raising wrapper used by main/tests)."""
    source = df.attrs.get("provenance", "unknown") if hasattr(df, "attrs") else "unknown"
    # Avoid duplicate console spam when load_htru2 already asserted
    if silent and df.attrs.get("_invariants_checked"):
        return {
            "ok": True,
            "errors": [],
            "warnings": [],
            "provenance": source,
            "n_rows": len(df),
            "n_features": 8,
            "n_class_0": int((df[TARGET_NAME] == 0).sum()),
            "n_class_1": int((df[TARGET_NAME] == 1).sum()),
            "prevalence": float((df[TARGET_NAME] == 1).mean()),
            "n_nan": int(df.isna().sum().sum()),
            "n_duplicates": int(df.duplicated().sum()),
            "paper_vs_uci_note": (
                "Official UCI HTRU2 has 17898 rows with 0 missing values. "
                "The 2023 IEEE paper used a Kaggle train split of 12528 rows (~70%) "
                "with ~3.1% artificially injected missing values imputed via MICE."
            ),
        }
    try:
        report = assert_dataset_invariants(df, config, source=source)
        df.attrs["_invariants_checked"] = True
        return report
    except AssertionError as exc:
        return {
            "ok": False,
            "errors": [str(exc)],
            "warnings": [],
            "n_rows": len(df),
            "n_nan": int(df.isna().sum().sum()),
            "n_duplicates": int(df.duplicated().sum()),
            "n_class_0": int((df[TARGET_NAME] == 0).sum()) if TARGET_NAME in df.columns else None,
            "n_class_1": int((df[TARGET_NAME] == 1).sum()) if TARGET_NAME in df.columns else None,
            "paper_vs_uci_note": str(exc),
        }


def get_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) with guaranteed zero target leakage into X."""
    X = df[FEATURE_NAMES].copy()
    y = df[TARGET_NAME].copy()
    assert TARGET_NAME not in X.columns
    assert X.shape[1] == 8
    return X, y


if __name__ == "__main__":
    cfg = load_config()
    data = load_htru2(cfg)
    print(validate_htru2(data, cfg))
    print(data.head())
