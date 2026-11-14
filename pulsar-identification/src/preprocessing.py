"""Train/test split, scaling, optional paper-missingness simulation with MICE.

CRITICAL LEAKAGE RULES
----------------------
1. StandardScaler is NEVER fit on the full dataset or test set. All model
   pipelines wrap StandardScaler inside sklearn.pipeline.Pipeline so μ,σ are
   fit only on each training fold / X_train.
2. For --simulate-paper-missingness: NaNs are injected on the 12,528-row
   subset, THEN stratified split, THEN IterativeImputer is fit on X_train
   only and applied to X_test (no test-set leakage into MICE).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import FEATURE_NAMES, TARGET_NAME
from src.utils import PROJECT_ROOT, load_config, save_json


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Fixed stratified 80/20 split (14,318 / 3,580 on full UCI)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    # Zero index overlap
    assert set(X_train.index).isdisjoint(set(X_test.index)), "Train/test index overlap!"
    return X_train, X_test, y_train, y_test


def make_scaler_pipeline(estimator) -> Pipeline:
    """Wrap any estimator with StandardScaler (fit on train / CV folds only)."""
    return Pipeline([("scaler", StandardScaler()), ("clf", estimator)])


def save_splits(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    processed_dir: Path,
) -> None:
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    train = X_train.copy()
    train[TARGET_NAME] = y_train.values
    test = X_test.copy()
    test[TARGET_NAME] = y_test.values
    train.to_csv(processed_dir / "train.csv", index=False)
    test.to_csv(processed_dir / "test.csv", index=False)
    overlap = set(X_train.index) & set(X_test.index)
    meta = {
        "n_train": len(train),
        "n_test": len(test),
        "train_class_counts": {int(k): int(v) for k, v in y_train.value_counts().items()},
        "test_class_counts": {int(k): int(v) for k, v in y_test.value_counts().items()},
        "feature_names": FEATURE_NAMES,
        "index_overlap": len(overlap),
        "scaler_policy": (
            "StandardScaler is fit inside model Pipelines on training data / CV "
            "folds only — never on the full dataset or holdout test set."
        ),
    }
    save_json(meta, processed_dir / "split_meta.json")
    print(
        f"[preprocess] Split OK: train={len(train)} test={len(test)} "
        f"index_overlap={len(overlap)}"
    )


def make_paper_missing_subset(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Build the paper-like 12,528-row subset WITH NaNs injected (NOT yet imputed).

    Imputation is deferred until after train/test split so IterativeImputer
    sees training rows only.
    """
    if config is None:
        config = load_config()
    data_cfg = config["data"]
    rng = np.random.default_rng(random_state)

    target_n = int(data_cfg["paper_n_rows"])
    frac = target_n / len(df)
    subset, _ = train_test_split(
        df,
        train_size=min(frac, 0.999),
        random_state=random_state,
        stratify=df[TARGET_NAME],
    )
    subset = subset.reset_index(drop=True)
    if len(subset) > target_n:
        subset = subset.iloc[:target_n].copy()
    elif len(subset) < target_n:
        # Should be rare; top up without replacement from original
        need = target_n - len(subset)
        pool = df.sample(n=min(need * 3, len(df)), random_state=random_state)
        extra = pool.iloc[:need]
        subset = pd.concat([subset, extra], ignore_index=True).iloc[:target_n]

    miss_cols = list(data_cfg["paper_missing_columns"])
    n_cells = len(subset) * len(FEATURE_NAMES)
    n_missing_target = int(round(data_cfg["paper_missing_fraction"] * n_cells))
    per_col = n_missing_target // len(miss_cols)
    remainder_m = n_missing_target - per_col * len(miss_cols)

    injected = subset.copy()
    for i, col in enumerate(miss_cols):
        n_miss = per_col + (1 if i < remainder_m else 0)
        n_miss = min(n_miss, len(injected))
        idx = rng.choice(len(injected), size=n_miss, replace=False)
        injected.loc[idx, col] = np.nan

    actual_missing = int(injected[FEATURE_NAMES].isna().sum().sum())
    report = {
        "mode": "simulate_paper_missingness",
        "n_rows": len(injected),
        "n_missing_injected": actual_missing,
        "missing_fraction": actual_missing / n_cells,
        "missing_columns": miss_cols,
        "class_counts": {
            int(k): int(v) for k, v in injected[TARGET_NAME].value_counts().items()
        },
        "imputation_policy": (
            "IterativeImputer (MICE) is fit on X_train ONLY after the stratified "
            "split, then transform is applied to X_test — no test leakage."
        ),
        "note": (
            "Approximates the 2023 IEEE paper's Kaggle train split "
            "(12,528 rows, ~3.1% missingness). Primary results use official "
            "17,898-row UCI with zero missing values."
        ),
    }
    return injected, report


def impute_train_test_mice(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, IterativeImputer]:
    """Fit IterativeImputer on training features only; transform train and test."""
    imputer = IterativeImputer(
        max_iter=20,
        random_state=random_state,
        sample_posterior=False,
    )
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train[FEATURE_NAMES]),
        columns=FEATURE_NAMES,
        index=X_train.index,
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test[FEATURE_NAMES]),
        columns=FEATURE_NAMES,
        index=X_test.index,
    )
    assert not X_train_imp.isna().any().any()
    assert not X_test_imp.isna().any().any()
    return X_train_imp, X_test_imp, imputer


def prepare_data(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
    simulate_paper: bool = False,
) -> dict[str, Any]:
    """Full preprocessing: optional missingness track → stratified split → persist."""
    if config is None:
        config = load_config()
    seed = config["seed"]
    processed_dir = PROJECT_ROOT / config["paths"]["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    paper_report = None
    working = df.copy()
    if simulate_paper:
        working, paper_report = make_paper_missing_subset(working, config, seed)

    X = working[FEATURE_NAMES]
    y = working[TARGET_NAME]
    # Assert no target leakage
    assert TARGET_NAME not in X.columns

    X_train, X_test, y_train, y_test = stratified_split(
        X,
        y,
        test_size=config["split"]["test_size"],
        random_state=seed,
    )

    mice_imputer = None
    if simulate_paper:
        X_train, X_test, mice_imputer = impute_train_test_mice(
            X_train, X_test, random_state=seed
        )
        paper_report["n_train"] = len(X_train)
        paper_report["n_test"] = len(X_test)
        save_json(paper_report, processed_dir / "paper_missingness_report.json")
        print(
            f"[preprocess] MICE fit on train only "
            f"(train={len(X_train)}, test={len(X_test)})"
        )

    # Official UCI sizes
    if not simulate_paper:
        assert len(X_train) == 14318 and len(X_test) == 3580, (
            f"Unexpected split sizes: {len(X_train)}/{len(X_test)}"
        )

    save_splits(X_train, X_test, y_train, y_test, processed_dir)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": FEATURE_NAMES,
        "simulate_paper": simulate_paper,
        "paper_report": paper_report,
        "mice_imputer": mice_imputer,
        # Explicit: do NOT use this for model fitting; models use Pipeline scalers
        "scaler_note": "Use Pipeline StandardScaler — never fit a global scaler on all data",
    }
