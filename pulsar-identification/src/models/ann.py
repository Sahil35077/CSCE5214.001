"""Artificial Neural Network (MLPClassifier) matching the paper's nnet setup."""

from __future__ import annotations

from typing import Any

from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_ann_pipeline(
    hidden_layer_sizes=(32,),
    activation: str = "relu",
    solver: str = "adam",
    alpha: float = 0.0001,
    max_iter: int = 2000,
    early_stopping: bool = True,
    validation_fraction: float = 0.1,
    learning_rate_init: float = 0.001,
    random_state: int = 42,
) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=tuple(hidden_layer_sizes),
                    activation=activation,
                    solver=solver,
                    alpha=alpha,
                    max_iter=max_iter,
                    early_stopping=early_stopping,
                    validation_fraction=validation_fraction,
                    learning_rate_init=learning_rate_init,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _serialize_param(value: Any) -> Any:
    return list(value) if isinstance(value, tuple) else value


def tune_ann(
    X,
    y,
    config: dict[str, Any],
    quick: bool = False,
) -> tuple[Pipeline, dict[str, Any]]:
    ann_cfg = config["ann"]
    cv_cfg = config["cv"]
    seed = config["seed"]

    pipe = build_ann_pipeline(
        max_iter=ann_cfg.get("max_iter", 500),
        early_stopping=ann_cfg.get("early_stopping", True),
        validation_fraction=ann_cfg.get("validation_fraction", 0.1),
        learning_rate_init=ann_cfg.get("learning_rate_init", 0.001),
        random_state=seed,
    )

    hidden = [tuple(h) for h in ann_cfg["hidden_layer_sizes_grid"]]
    activations = list(ann_cfg.get("activation_grid", ["relu"]))
    alphas = list(ann_cfg.get("alpha_grid", [0.0001]))
    solvers = list(ann_cfg.get("solver_grid", ["adam"]))

    if quick:
        hidden = [(32,)]
        activations = ["relu"]
        alphas = [0.0001]
        solvers = ["adam"]
        cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=seed)
        pipe.set_params(clf__max_iter=200)
    else:
        cv = RepeatedStratifiedKFold(
            n_splits=cv_cfg["n_splits"],
            n_repeats=cv_cfg["n_repeats"],
            random_state=seed,
        )

    param_grid = {
        "clf__hidden_layer_sizes": hidden,
        "clf__activation": activations,
        "clf__alpha": alphas,
        "clf__solver": solvers,
    }

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

    best_params = {k: _serialize_param(v) for k, v in gs.best_params_.items()}
    param_list = [
        {kk: _serialize_param(vv) for kk, vv in p.items()}
        for p in gs.cv_results_["params"]
    ]

    details = {
        "best_params": best_params,
        "best_score_pr_auc": float(gs.best_score_),
        "cv_results": {
            "params": param_list,
            "mean_test_score": [float(x) for x in gs.cv_results_["mean_test_score"]],
            "std_test_score": [float(x) for x in gs.cv_results_["std_test_score"]],
        },
        "quick": quick,
    }
    return gs.best_estimator_, details
