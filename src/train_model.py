"""FootballIQ Machine Learning Training and Evaluation Module.

Implements regression and classification workflows using Scikit-learn:
- Regression: Predicts player overall match rating (ss_rating) using
  Linear Regression, Decision Tree Regressor, and Random Forest Regressor.
  Evaluated on R² Score, MAE, and RMSE.
- Classification: Predicts whether a player is an Elite/High-Impact Performer
  (rating >= 7.0) using Logistic Regression, Decision Tree Classifier,
  and Random Forest Classifier. Evaluated on Accuracy, Precision, Recall, and F1 Score.
- Prevents data leakage by fitting preprocessors strictly on the training partition.
- Persists trained models to disk with joblib.
- Records predictions to CSV and saves model evaluation metadata to MongoDB (ml_results).
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.load_data import get_mongo_client
from src.preprocess import FootballDataPreprocessor


def evaluate_regression_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """Train and evaluate Linear Regression, Decision Tree, and Random Forest Regressors."""
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=5, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        ),
    }

    results: Dict[str, Dict[str, float]] = {}
    fitted_models: Dict[str, Any] = {}
    best_model_name = ""
    best_r2 = -float("inf")

    print("\n--- Regression Task: Player Overall Rating (ss_rating) Prediction ---")
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        r2 = float(r2_score(y_test, preds))
        mae = float(mean_absolute_error(y_test, preds))
        rmse = float(root_mean_squared_error(y_test, preds))

        results[name] = {
            "r2_score": round(r2, 4),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
        }
        fitted_models[name] = model

        print(f"   {name:<26} -> R²: {r2:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name

    print(f"   [BEST] Regression Model: {best_model_name} (R² = {best_r2:.4f})")
    return results, fitted_models, best_model_name


