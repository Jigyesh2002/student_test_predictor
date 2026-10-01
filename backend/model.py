"""
model.py — Data loading, feature engineering, and ML model training.
Features: study_hours_per_day, sleep_hours
Target:   exam_score
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data",
                         "enhanced_student_habits_performance_dataset.csv")

FEATURES = ["study_hours_per_day", "sleep_hours"]
TARGET = "exam_score"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_and_prepare() -> Tuple[pd.DataFrame, pd.Series,
                                 pd.DataFrame, pd.Series,
                                 pd.DataFrame]:
    """Load CSV, select columns, drop nulls, split 80/20."""
    df = pd.read_csv(_CSV_PATH)
    df = df[FEATURES + [TARGET]].dropna()

    X = df[FEATURES]
    y = df[TARGET].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, df


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = math.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)}


# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------

class StudentScoreModel:
    """Trains Linear Regression and Random Forest; exposes predict()."""

    def __init__(self) -> None:
        X_train, X_test, y_train, y_test, self.df = load_and_prepare()

        # Linear Regression
        self.lr = LinearRegression()
        self.lr.fit(X_train, y_train)
        self.lr_metrics = evaluate(self.lr, X_test, y_test)

        # Random Forest
        self.rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        self.rf.fit(X_train, y_train)
        self.rf_metrics = evaluate(self.rf, X_test, y_test)

        # Compute residual std for LR (used for prediction intervals)
        lr_resid = y_train - self.lr.predict(X_train)
        self._lr_resid_std = float(np.std(lr_resid))

        # Compute residual std for RF
        rf_resid = y_train - self.rf.predict(X_train)
        self._rf_resid_std = float(np.std(rf_resid))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self, study_hours: float, sleep_hours: float, model_name: str = "linear"
    ) -> Dict[str, float]:
        """Return predicted score + 95 % prediction interval (clipped 0–100)."""
        X = pd.DataFrame([[study_hours, sleep_hours]], columns=FEATURES)

        if model_name == "random_forest":
            score = float(self.rf.predict(X)[0])
            std = self._rf_resid_std
        else:
            score = float(self.lr.predict(X)[0])
            std = self._lr_resid_std

        z95 = 1.96
        lower = score - z95 * std
        upper = score + z95 * std

        return {
            "predicted_score": round(min(max(score, 0), 100), 2),
            "lower_bound": round(min(max(lower, 0), 100), 2),
            "upper_bound": round(min(max(upper, 0), 100), 2),
        }

    def metrics(self) -> Dict[str, Dict[str, float]]:
        return {
            "linear_regression": self.lr_metrics,
            "random_forest": self.rf_metrics,
        }

    def summary_stats(self) -> Dict[str, Dict[str, float]]:
        """Descriptive statistics for the three key columns."""
        desc = self.df[FEATURES + [TARGET]].describe().round(3)
        return desc.to_dict()

    def correlation(self) -> Dict[str, Dict[str, float]]:
        """Pearson correlation matrix for the three key columns."""
        corr = self.df[FEATURES + [TARGET]].corr().round(4)
        return corr.to_dict()

    def scatter_data(self, max_rows: int = 2000) -> List[Dict]:
        """Return a sample of rows for scatter/3D visualisation."""
        df = self.df.sample(min(max_rows, len(self.df)), random_state=42)
        return df[FEATURES + [TARGET]].rename(
            columns={
                "study_hours_per_day": "study_hours",
                "sleep_hours": "sleep_hours",
                "exam_score": "exam_score",
            }
        ).to_dict(orient="records")


@lru_cache(maxsize=1)
def get_model() -> StudentScoreModel:
    """Module-level singleton — trained once, shared everywhere."""
    return StudentScoreModel()
