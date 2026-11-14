"""K-Nearest Neighbors pipeline and hyperparameter tuning."""

from __future__ import annotations

from typing import Any

from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_knn_pipeline(n_neighbors: int = 15, metric: str = "euclidean") -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                KNeighborsClassifier(
                    n_neighbors=n_neighbors,
                    metric=metric,
                    weights="uniform",
                    n_jobs=-1,
                ),
            ),
        ]
    )


def tune_knn(
    X,
    y,
    config: dict[str, Any],
    quick: bool = False,
) -> tuple[Pipeline, dict[str, Any]]:
    knn_cfg = config["knn"]
    cv_cfg = config["cv"]
    seed = config["seed"]

    pipe = build_knn_pipeline(
        n_neighbors=knn_cfg.get("paper_optimal_k", 15),
        metric=knn_cfg.get("metric", "euclidean"),
    )
    grid = list(knn_cfg["n_neighbors_grid"])
    if quick:
        grid = [5, knn_cfg.get("paper_optimal_k", 15)]
        cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=seed)
    else:
        cv = RepeatedStratifiedKFold(
            n_splits=cv_cfg["n_splits"],
            n_repeats=cv_cfg["n_repeats"],
            random_state=seed,
        )

    gs = GridSearchCV(
        pipe,
        param_grid={"clf__n_neighbors": grid},
        cv=cv,
        scoring="average_precision",
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )
    # Also collect accuracy for Fig. 8 style plots via a second pass of cv_results
    # Primary refit score is PR-AUC (proposal); we additionally score accuracy.
    gs.fit(X, y)

    # Accuracy curve across k (separate lightweight GridSearch for plotting)
    gs_acc = GridSearchCV(
        pipe,
        param_grid={"clf__n_neighbors": grid},
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        refit=False,
    )
    gs_acc.fit(X, y)

    details = {
        "best_params": gs.best_params_,
        "best_score_pr_auc": float(gs.best_score_),
        "paper_optimal_k": knn_cfg.get("paper_optimal_k", 15),
        "selected_k": int(gs.best_params_["clf__n_neighbors"]),
        "cv_results": {
            "params": [dict(p) for p in gs.cv_results_["params"]],
            "mean_test_score_pr_auc": [float(x) for x in gs.cv_results_["mean_test_score"]],
            "std_test_score_pr_auc": [float(x) for x in gs.cv_results_["std_test_score"]],
            "mean_test_score_accuracy": [
                float(x) for x in gs_acc.cv_results_["mean_test_score"]
            ],
            "std_test_score_accuracy": [
                float(x) for x in gs_acc.cv_results_["std_test_score"]
            ],
        },
        "n_neighbors_grid": grid,
        "quick": quick,
    }
    return gs.best_estimator_, details
