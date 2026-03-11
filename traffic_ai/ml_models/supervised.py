from __future__ import annotations

from dataclasses import dataclass
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neural_network import MLPClassifier

from traffic_ai.data_pipeline.preprocessing import PreparedDataset
from traffic_ai.utils.io_utils import save_model, write_dataframe

try:
    from xgboost import XGBClassifier  # type: ignore
except Exception:  # pragma: no cover - optional dependency fallback
    XGBClassifier = None


class TimeSeriesPhaseClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self) -> None:
        self.model = LogisticRegression(max_iter=1500)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TimeSeriesPhaseClassifier":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


@dataclass(slots=True)
class ModelEvaluation:
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    cv_best_score: float


class SupervisedModelSuite:
    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.models: dict[str, Any] = self._build_models()

    def train_and_evaluate(
        self,
        dataset: PreparedDataset,
        model_dir: Path,
        cv_folds: int = 3,
        quick_mode: bool = False,
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        model_dir.mkdir(parents=True, exist_ok=True)
        evaluations: list[ModelEvaluation] = []
        trained_models: dict[str, Any] = {}

        X_train = dataset.X_train
        y_train = dataset.y_train
        if quick_mode and len(X_train) > 2000:
            X_train = X_train[-2000:]
            y_train = y_train[-2000:]

        for name, model in self.models.items():
            if quick_mode:
                tuned_model = self._quick_fit_model(name, model, X_train, y_train)
                cv_score = float("nan")
            else:
                tuned_model, cv_score = self._fit_with_search(
                    name=name,
                    model=model,
                    X=X_train,
                    y=y_train,
                    cv_folds=cv_folds,
                )
            preds = tuned_model.predict(dataset.X_test)
            eval_row = ModelEvaluation(
                model_name=name,
                accuracy=float(accuracy_score(dataset.y_test, preds)),
                precision=float(precision_score(dataset.y_test, preds, zero_division=0)),
                recall=float(recall_score(dataset.y_test, preds, zero_division=0)),
                f1=float(f1_score(dataset.y_test, preds, zero_division=0)),
                cv_best_score=float(cv_score),
            )
            evaluations.append(eval_row)
            trained_models[name] = tuned_model
            save_model(tuned_model, model_dir / f"{name}.joblib")

        eval_df = pd.DataFrame([asdict(row) for row in evaluations]).sort_values(
            "f1", ascending=False
        )
        write_dataframe(eval_df, model_dir.parent / "results" / "supervised_model_metrics.csv")
        return trained_models, eval_df

    @staticmethod
    def _quick_fit_model(name: str, model: Any, X: np.ndarray, y: np.ndarray) -> Any:
        if name == "random_forest" and hasattr(model, "set_params"):
            model.set_params(n_estimators=40, max_depth=8)
        if name == "gradient_boosting" and hasattr(model, "set_params"):
            model.set_params(n_estimators=50, learning_rate=0.08, max_depth=2)
        if name == "xgboost" and hasattr(model, "set_params"):
            model.set_params(n_estimators=40, max_depth=4)
        if name == "mlp" and hasattr(model, "set_params"):
            model.set_params(hidden_layer_sizes=(32, 32), max_iter=60)
        model.fit(X, y)
        return model

    def _fit_with_search(
        self,
        name: str,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int,
    ) -> tuple[Any, float]:
        search_space = self._search_space(name)
        cv = TimeSeriesSplit(n_splits=max(2, cv_folds))
        search = GridSearchCV(
            estimator=model,
            param_grid=search_space,
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X, y)
        return search.best_estimator_, float(search.best_score_)

    def _build_models(self) -> dict[str, Any]:
        models: dict[str, Any] = {
            "random_forest": RandomForestClassifier(
                random_state=self.seed, n_estimators=160, max_depth=14
            ),
            "gradient_boosting": GradientBoostingClassifier(random_state=self.seed),
            "mlp": MLPClassifier(
                hidden_layer_sizes=(64, 64),
                max_iter=260,
                random_state=self.seed,
                early_stopping=True,
            ),
            "timeseries": TimeSeriesPhaseClassifier(),
        }
        if XGBClassifier is not None:
            models["xgboost"] = XGBClassifier(
                random_state=self.seed,
                n_estimators=160,
                learning_rate=0.06,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
            )
        else:
            models["xgboost"] = GradientBoostingClassifier(random_state=self.seed)
        return models

    @staticmethod
    def _search_space(name: str) -> dict[str, list[Any]]:
        if name == "random_forest":
            return {
                "n_estimators": [200, 300],
                "max_depth": [10, 14, 18],
                "min_samples_leaf": [1, 2],
            }
        if name == "gradient_boosting":
            return {
                "n_estimators": [100, 200],
                "learning_rate": [0.03, 0.06, 0.1],
                "max_depth": [2, 3],
            }
        if name == "xgboost":
            return {
                "n_estimators": [180, 260],
                "learning_rate": [0.04, 0.08],
                "max_depth": [4, 6],
            }
        if name == "mlp":
            return {
                "hidden_layer_sizes": [(64, 64), (128, 64)],
                "alpha": [0.0001, 0.001],
            }
        return {}
