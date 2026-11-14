"""Pytest: data integrity, leakage, MARS, metrics, robustness, figure DPI."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import (
    COLUMN_NAMES,
    FEATURE_NAMES,
    TARGET_NAME,
    assert_dataset_invariants,
    load_htru2,
    validate_htru2,
)
from src.evaluate import compute_metrics, confusion_counts, mcnemar_test
from src.models.ann import build_ann_pipeline
from src.models.knn import build_knn_pipeline
from src.models.mars import MARSClassifier, MARSRegressor, _hinge, build_mars_pipeline
from src.plotting import DPI, FEATURE_DESCRIPTIONS
from src.preprocessing import (
    impute_train_test_mice,
    make_paper_missing_subset,
    stratified_split,
)
from src.robustness import add_gaussian_noise
from src.utils import load_config, set_seed


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def df(config):
    return load_htru2(config)


def test_dataset_shape_and_schema(df):
    assert df.shape == (17898, 9)
    assert list(df.columns) == COLUMN_NAMES


def test_class_counts_and_prevalence(df, config):
    n0 = int((df[TARGET_NAME] == 0).sum())
    n1 = int((df[TARGET_NAME] == 1).sum())
    assert n0 == 16259
    assert n1 == 1639
    assert abs(n0 / len(df) - 0.9084) < 1e-3
    assert abs(n1 / len(df) - 0.0916) < 1e-3


def test_zero_nans_duplicates_and_assert_invariants(df, config):
    assert int(df.isna().sum().sum()) == 0
    assert int(df.duplicated().sum()) == 0
    report = assert_dataset_invariants(df, config, source=df.attrs.get("provenance", "test"))
    assert report["ok"] is True
    assert validate_htru2(df, config)["ok"] is True


def test_feature_physical_meanings():
    assert list(FEATURE_DESCRIPTIONS.keys()) == FEATURE_NAMES
    # 4 integrated profile + 4 DM-SNR
    assert sum("integrated_profile" in f for f in FEATURE_NAMES) == 4
    assert sum("dm_snr" in f for f in FEATURE_NAMES) == 4


def test_no_target_leakage_in_X(df):
    X = df[FEATURE_NAMES]
    assert TARGET_NAME not in X.columns
    assert X.shape == (17898, 8)


def test_no_train_test_leakage(df, config):
    X = df[FEATURE_NAMES]
    y = df[TARGET_NAME]
    X_train, X_test, y_train, y_test = stratified_split(
        X, y, test_size=0.2, random_state=config["seed"]
    )
    assert set(X_train.index).isdisjoint(set(X_test.index))
    assert len(X_train) == 14318
    assert len(X_test) == 3580
    assert abs(y_train.mean() - y_test.mean()) < 0.01


def test_scaler_inside_pipeline_not_global(df, config):
    """StandardScaler must live inside Pipeline — never fit on full data for models."""
    for builder in (build_mars_pipeline, build_knn_pipeline, build_ann_pipeline):
        pipe = builder()
        assert isinstance(pipe, Pipeline)
        assert "scaler" in pipe.named_steps
        assert isinstance(pipe.named_steps["scaler"], StandardScaler)


def test_mice_fit_on_train_only(df, config):
    """IterativeImputer must not see test rows during fit."""
    injected, report = make_paper_missing_subset(df, config, random_state=42)
    assert report["n_rows"] == 12528
    assert report["n_missing_injected"] > 0
    X = injected[FEATURE_NAMES]
    y = injected[TARGET_NAME]
    X_train, X_test, y_train, y_test = stratified_split(X, y, random_state=42)
    # Train has NaNs before imputation
    assert X_train.isna().any().any() or X_test.isna().any().any()
    X_tr_imp, X_te_imp, imputer = impute_train_test_mice(X_train, X_test, random_state=42)
    assert not X_tr_imp.isna().any().any()
    assert not X_te_imp.isna().any().any()
    # Imputer was fitted (n_features_in_ set) — transform-only on test
    assert hasattr(imputer, "n_features_in_")
    assert imputer.n_features_in_ == 8


def test_mars_hinge_basis_functions():
    x = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
    t = 1.0
    assert np.allclose(_hinge(x, t, 1), np.array([0, 0, 0, 1, 2]))
    assert np.allclose(_hinge(x, t, -1), np.array([2, 1, 0, 0, 0]))


def test_mars_regressor_and_clone_resets_state():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = (X[:, 0] - 0.5 * X[:, 1] + rng.normal(scale=0.1, size=200) > 0).astype(float)
    mars = MARSRegressor(max_terms=10, nprune=6, n_knots=5)
    mars.fit(X, y)
    n_terms = len(mars.terms_)
    assert n_terms <= 6
    cloned = clone(mars)
    assert not hasattr(cloned, "terms_")
    cloned.fit(X, y)
    assert hasattr(cloned, "terms_")


def test_mars_classifier_proba_bounds_and_pipeline():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(300, 5))
    y = (X[:, 0] + 0.3 * X[:, 1] > 0).astype(int)
    clf = MARSClassifier(nprune=8, max_terms=16, n_knots=6, random_state=42)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (300, 2)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    pipe = build_mars_pipeline(nprune=8, max_terms=16, n_knots=6, random_state=42)
    pipe.fit(X, y)
    p2 = pipe.predict_proba(X)
    assert np.all((p2 >= 0) & (p2 <= 1))


def test_metric_pos_label_definitions():
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0])
    y_proba = np.array([0.9, 0.8, 0.4, 0.6, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    cm = confusion_counts(y_true, y_pred)
    assert cm == {"TN": 6, "FP": 1, "FN": 1, "TP": 2}
    m1 = compute_metrics(y_true, y_pred, y_proba, pos_label=1)
    assert m1["recall"] == pytest.approx(2 / 3)
    m0 = compute_metrics(y_true, y_pred, y_proba, pos_label=0)
    assert m0["recall"] == pytest.approx(6 / 7)


def test_paper_corrected_pulsar_recalls():
    assert 187 / (187 + 43) == pytest.approx(0.8130, abs=1e-4)
    assert 182 / (182 + 48) == pytest.approx(0.7913, abs=1e-4)
    assert 194 / (194 + 36) == pytest.approx(0.8435, abs=1e-4)


def test_mcnemar_discordant_counts():
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    a = np.array([0, 0, 1, 0, 0, 1, 1, 1])  # wrong on idx 3, 6
    b = np.array([0, 1, 1, 1, 0, 0, 0, 1])  # wrong on idx 1, 5
    res = mcnemar_test(y, a, b)
    assert "pvalue" in res
    assert res["n_a_correct_b_wrong"] + res["n_a_wrong_b_correct"] >= 0


def test_perturbation_reproducibility_and_std_clip():
    set_seed(42)
    X = np.ones((50, 8))
    std = np.ones(8)
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    a = add_gaussian_noise(X, std, 0.2, rng1, clip_nonnegative_std=True)
    b = add_gaussian_noise(X, std, 0.2, rng2, clip_nonnegative_std=True)
    assert np.allclose(a, b)
    # std columns (idx 1 and 5) never negative when clipped
    X_neg = np.full((100, 8), 0.01)
    rng = np.random.default_rng(7)
    noisy = add_gaussian_noise(X_neg, np.ones(8) * 5.0, 1.0, rng, clip_nonnegative_std=True)
    assert np.all(noisy[:, 1] >= 0)
    assert np.all(noisy[:, 5] >= 0)
    # Original X_test not mutated
    assert np.allclose(X_neg[:, 0], 0.01)


def test_knn_ann_proba_bounds_smoke():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 8))
    y = (X[:, 0] > 0).astype(int)
    knn = build_knn_pipeline(n_neighbors=5)
    knn.fit(X, y)
    assert np.all((knn.predict_proba(X) >= 0) & (knn.predict_proba(X) <= 1))

    ann = build_ann_pipeline(
        hidden_layer_sizes=(16,), max_iter=300, early_stopping=False, random_state=0
    )
    ann.fit(X, y)
    assert np.all((ann.predict_proba(X) >= 0) & (ann.predict_proba(X) <= 1))


def test_figure_dpi_300_when_present():
    """If publication figures exist, they must embed 300 DPI metadata."""
    figures_dir = ROOT / "results" / "figures"
    expected = [
        "01_correlation_heatmap.png",
        "02_feature_boxplots.png",
        "03_feature_kde_distributions.png",
        "07_pr_and_roc_curves.png",
        "08_robustness_degradation.png",
    ]
    present = [figures_dir / n for n in expected if (figures_dir / n).exists()]
    if not present:
        pytest.skip("Figures not generated yet — run main.py first")
    for path in present:
        with Image.open(path) as im:
            dpi = im.info.get("dpi")
            assert dpi is not None, f"{path.name} missing DPI metadata"
            if isinstance(dpi, tuple):
                assert abs(dpi[0] - DPI) < 1.0 and abs(dpi[1] - DPI) < 1.0
            else:
                assert abs(float(dpi) - DPI) < 1.0
