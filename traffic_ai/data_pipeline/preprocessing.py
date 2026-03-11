from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass(slots=True)
class PreparedDataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    metadata: dict[str, Any]


class DatasetPreprocessor:
    def __init__(
        self,
        timestamp_col: str = "timestamp",
        target_col: str = "optimal_phase",
        normalize: bool = True,
    ) -> None:
        self.timestamp_col = timestamp_col
        self.target_col = target_col
        self.normalize = normalize
        self.scaler = StandardScaler()

    def prepare(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> PreparedDataset:
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("train/val/test ratios must sum to 1.0")

        ordered = df.sort_values(self.timestamp_col).reset_index(drop=True)
        y = ordered[self.target_col].astype(int).to_numpy()
        X_df = ordered.drop(columns=[self.target_col])

        # Keep timestamp for metadata but exclude from model matrix.
        timestamp = pd.to_datetime(X_df[self.timestamp_col], errors="coerce")
        X_df = X_df.drop(columns=[self.timestamp_col])
        X_df = pd.get_dummies(X_df, drop_first=False)
        feature_names = list(X_df.columns)
        X = X_df.to_numpy(dtype=np.float64)

        if len(X) < 6:
            raise RuntimeError(
                "Insufficient samples after preprocessing. Enable public/Kaggle data or synthetic augmentation."
            )

        train_end = max(1, int(len(X) * train_ratio))
        val_end = max(train_end + 1, int(len(X) * (train_ratio + val_ratio)))
        val_end = min(val_end, len(X) - 1)

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        if self.normalize:
            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)
            X_test = self.scaler.transform(X_test)

        metadata = {
            "n_samples": int(len(X)),
            "train_samples": int(len(X_train)),
            "val_samples": int(len(X_val)),
            "test_samples": int(len(X_test)),
            "start_timestamp": str(timestamp.min()),
            "end_timestamp": str(timestamp.max()),
            "class_balance": {
                "phase_0": float(np.mean(y == 0)),
                "phase_1": float(np.mean(y == 1)),
            },
        }
        return PreparedDataset(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_names,
            metadata=metadata,
        )

    def random_train_test_split(
        self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 42
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

    def save_scaler(self, path: str | Path) -> Path:
        import joblib

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, target)
        return target
