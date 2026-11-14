"""
Scikit-learn compatible MARSClassifier (pure NumPy/SciPy).

Implements Multivariate Adaptive Regression Splines with:
  - Forward pass: piecewise-linear hinge pairs max(0, x_j - t), max(0, t - x_j)
  - degree=1 (additive model, matching Abdullah et al. 2023)
  - Backward GCV pruning down to nprune / max_terms
  - Ridge-regularized logistic regression head for binary classification

Do NOT use legacy py-earth Cython builds or the unrelated GIS package `pyearth`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


@dataclass
class _HingeTerm:
    feature: int
    knot: float
    direction: int  # +1 => max(0, x - t), -1 => max(0, t - x)


def _hinge(x: np.ndarray, knot: float, direction: int) -> np.ndarray:
    if direction > 0:
        return np.maximum(0.0, x - knot)
    return np.maximum(0.0, knot - x)


def _gcv(rss: float, n: int, n_params: int, d: float = 3.0) -> float:
    """Friedman's GCV = (RSS / n) / (1 - d * n_params / n)^2."""
    if n <= 0:
        return np.inf
    penalty = 1.0 - (n_params * d) / n
    if penalty <= 1e-12:
        return np.inf
    return (rss / n) / (penalty ** 2)


class MARSRegressor(BaseEstimator):
    """Additive (degree=1) MARS with forward hinge search + backward GCV pruning."""

    def __init__(
        self,
        max_terms: int = 30,
        nprune: int = 14,
        n_knots: int = 12,
        degree: int = 1,
        gcv_penalty: float = 3.0,
    ):
        if degree != 1:
            raise ValueError("Only degree=1 (additive) is implemented.")
        self.max_terms = max_terms
        self.nprune = nprune
        self.n_knots = n_knots
        self.degree = degree
        self.gcv_penalty = gcv_penalty

    def _candidate_knots(self, x: np.ndarray) -> np.ndarray:
        xs = np.unique(x)
        if len(xs) <= 2:
            return xs
        qs = np.linspace(0.1, 0.9, self.n_knots)
        return np.unique(np.quantile(x, qs))

    def _design(self, X: np.ndarray, terms: list[_HingeTerm]) -> np.ndarray:
        n = X.shape[0]
        cols = [np.ones(n, dtype=float)]
        for term in terms:
            cols.append(_hinge(X[:, term.feature], term.knot, term.direction))
        return np.column_stack(cols)

    def _fit_ols(self, B: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        lam = 1e-8
        btb = B.T @ B + lam * np.eye(B.shape[1])
        try:
            coef = np.linalg.solve(btb, B.T @ y)
        except np.linalg.LinAlgError:
            coef = np.linalg.lstsq(B, y, rcond=None)[0]
        resid = y - B @ coef
        return coef, float(resid @ resid)

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        self.n_features_in_ = p
        # Reset fitted state so sklearn clone()/CV folds never reuse prior terms
        for attr in (
            "terms_",
            "forward_terms_",
            "coef_",
            "rss_",
            "intercept_",
            "gcv_path_",
            "selected_gcv_",
        ):
            if hasattr(self, attr):
                delattr(self, attr)

        # Precompute candidate hinge columns: list of (feature, knot, dir, column)
        candidates: list[tuple[int, float, int, np.ndarray]] = []
        for j in range(p):
            for t in self._candidate_knots(X[:, j]):
                t = float(t)
                candidates.append((j, t, 1, _hinge(X[:, j], t, 1)))
                candidates.append((j, t, -1, _hinge(X[:, j], t, -1)))

        terms: list[_HingeTerm] = []
        used = set()  # (feature, knot, direction)

        # Forward pass: add best hinge pair (both directions at same knot) each iteration
        # Group candidates by (feature, knot)
        knot_groups: dict[tuple[int, float], list[tuple[int, np.ndarray]]] = {}
        for feat, knot, direction, col in candidates:
            knot_groups.setdefault((feat, knot), []).append((direction, col))

        B = np.ones((n, 1), dtype=float)
        _, rss_cur = self._fit_ols(B, y)

        while len(terms) + 2 <= self.max_terms:
            best_rss = rss_cur
            best_key = None
            best_cols = None
            best_dirs = None

            for (feat, knot), dir_cols in knot_groups.items():
                if any((feat, knot, d) in used for d, _ in dir_cols):
                    continue
                if len(dir_cols) < 2:
                    continue
                # Expect both +1 and -1
                col_p = next(c for d, c in dir_cols if d == 1)
                col_m = next(c for d, c in dir_cols if d == -1)
                B_try = np.column_stack([B, col_p, col_m])
                _, rss = self._fit_ols(B_try, y)
                if rss < best_rss - 1e-10:
                    best_rss = rss
                    best_key = (feat, knot)
                    best_cols = (col_p, col_m)
                    best_dirs = (1, -1)

            if best_key is None:
                break

            feat, knot = best_key
            for d, col in zip(best_dirs, best_cols):
                terms.append(_HingeTerm(feature=feat, knot=knot, direction=d))
                used.add((feat, knot, d))
                B = np.column_stack([B, col])
            rss_cur = best_rss

        self.forward_terms_ = list(terms)

        # Backward GCV pruning path
        current = list(terms)
        path: list[tuple[list[_HingeTerm], float, float]] = []

        B_full = self._design(X, current)
        _, rss = self._fit_ols(B_full, y)
        gcv = _gcv(rss, n, B_full.shape[1], self.gcv_penalty)
        path.append((list(current), rss, gcv))

        while current:
            best_gcv = np.inf
            best_idx = -1
            best_rss_local = np.inf
            for i in range(len(current)):
                trial = current[:i] + current[i + 1 :]
                B_t = self._design(X, trial)
                _, rss_t = self._fit_ols(B_t, y)
                gcv_t = _gcv(rss_t, n, B_t.shape[1], self.gcv_penalty)
                if gcv_t < best_gcv:
                    best_gcv = gcv_t
                    best_idx = i
                    best_rss_local = rss_t
            current.pop(best_idx)
            path.append((list(current), float(best_rss_local), float(best_gcv)))

        eligible = [(t, r, g) for (t, r, g) in path if len(t) <= self.nprune]
        if not eligible:
            eligible = [path[-1]]
        best = min(eligible, key=lambda z: z[2])
        self.terms_ = best[0]
        self.gcv_path_ = [(len(t), g) for (t, _, g) in path]
        self.selected_gcv_ = best[2]

        B_final = self._design(X, self.terms_)
        self.coef_, self.rss_ = self._fit_ols(B_final, y)
        self.intercept_ = float(self.coef_[0])
        return self

    def transform(self, X) -> np.ndarray:
        check_is_fitted(self, "terms_")
        X = check_array(X, dtype=float)
        return self._design(X, self.terms_)

    def predict(self, X) -> np.ndarray:
        return self.transform(X) @ self.coef_

    def variable_importance(self) -> np.ndarray:
        check_is_fitted(self, "terms_")
        imp = np.zeros(self.n_features_in_, dtype=float)
        for i, term in enumerate(self.terms_):
            imp[term.feature] += abs(float(self.coef_[i + 1]))
        s = imp.sum()
        if s > 0:
            imp /= s
        return imp


class MARSClassifier(BaseEstimator, ClassifierMixin):
    """MARS basis expansion + LogisticRegression (ridge) classification head."""

    def __init__(
        self,
        nprune: int = 14,
        max_terms: int = 30,
        n_knots: int = 15,
        degree: int = 1,
        C: float = 1.0,
        max_iter: int = 1000,
        gcv_penalty: float = 3.0,
        random_state: int = 42,
    ):
        self.nprune = nprune
        self.max_terms = max_terms
        self.n_knots = n_knots
        self.degree = degree
        self.C = C
        self.max_iter = max_iter
        self.gcv_penalty = gcv_penalty
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("MARSClassifier currently supports binary classification.")
        # Fresh sub-estimators every fit (CV-safe clone semantics)
        for attr in ("mars_", "clf_"):
            if hasattr(self, attr):
                delattr(self, attr)

        self.mars_ = MARSRegressor(
            max_terms=self.max_terms,
            nprune=self.nprune,
            n_knots=self.n_knots,
            degree=self.degree,
            gcv_penalty=self.gcv_penalty,
        )
        self.mars_.fit(X, y.astype(float))
        B = self.mars_.transform(X)

        self.clf_ = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver="lbfgs",
            random_state=self.random_state,
        )
        self.clf_.fit(B[:, 1:], y)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "clf_")
        B = self.mars_.transform(X)
        proba = self.clf_.predict_proba(B[:, 1:])
        # Numerical safety: clip to [0, 1] and renormalize
        proba = np.clip(proba, 0.0, 1.0)
        proba = proba / proba.sum(axis=1, keepdims=True)
        return proba

    def predict(self, X):
        check_is_fitted(self, "clf_")
        B = self.mars_.transform(X)
        return self.clf_.predict(B[:, 1:])

    def decision_function(self, X):
        check_is_fitted(self, "clf_")
        B = self.mars_.transform(X)
        return self.clf_.decision_function(B[:, 1:])

    def variable_importance(self) -> np.ndarray:
        return self.mars_.variable_importance()


def build_mars_pipeline(nprune: int = 14, **kwargs) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", MARSClassifier(nprune=nprune, **kwargs)),
        ]
    )


def tune_mars(
    X,
    y,
    config: dict[str, Any],
    quick: bool = False,
) -> tuple[Pipeline, dict[str, Any]]:
    """Tune nprune via RepeatedStratifiedKFold; return best pipeline + CV details."""
    mars_cfg = config["mars"]
    cv_cfg = config["cv"]
    seed = config["seed"]

    pipe = build_mars_pipeline(
        nprune=mars_cfg["nprune"],
        max_terms=mars_cfg.get("max_terms", 30),
        n_knots=mars_cfg.get("n_knots", 15),
        degree=mars_cfg.get("degree", 1),
        C=mars_cfg.get("C", 1.0),
        max_iter=mars_cfg.get("max_iter", 1000),
        random_state=seed,
    )
    param_grid = {"clf__nprune": list(mars_cfg["nprune_grid"])}

    if quick:
        # Fast path for CI / smoke tests
        cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=seed)
        param_grid = {"clf__nprune": [mars_cfg["nprune"], 8]}
    else:
        cv = RepeatedStratifiedKFold(
            n_splits=cv_cfg["n_splits"],
            n_repeats=cv_cfg["n_repeats"],
            random_state=seed,
        )

    gs = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=cv,
        scoring="average_precision",
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )
    gs.fit(X, y)

    details = {
        "best_params": gs.best_params_,
        "best_score_pr_auc": float(gs.best_score_),
        "cv_results": {
            "params": [dict(p) for p in gs.cv_results_["params"]],
            "mean_test_score": [float(x) for x in gs.cv_results_["mean_test_score"]],
            "std_test_score": [float(x) for x in gs.cv_results_["std_test_score"]],
        },
        "nprune_grid": list(param_grid["clf__nprune"]),
        "quick": quick,
    }
    return gs.best_estimator_, details
