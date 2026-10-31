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


def evaluate_classification_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """Train and evaluate Logistic Regression, Decision Tree, and Random Forest Classifiers."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree Classifier": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        ),
    }

    results: Dict[str, Dict[str, float]] = {}
    fitted_models: Dict[str, Any] = {}
    best_model_name = ""
    best_f1 = -float("inf")

    print("\n--- Classification Task: High-Impact Performer (Rating >= 7.0) ---")
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = float(accuracy_score(y_test, preds))
        prec = float(precision_score(y_test, preds, zero_division=0))
        rec = float(recall_score(y_test, preds, zero_division=0))
        f1 = float(f1_score(y_test, preds, zero_division=0))

        results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
        }
        fitted_models[name] = model

        print(f"   {name:<26} -> Acc: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name

    print(f"   [BEST] Classification Model: {best_model_name} (F1 = {best_f1:.4f})")
    return results, fitted_models, best_model_name


def extract_feature_importances(
    model: Any, feature_names: List[str], top_k: int = 10
) -> List[Dict[str, Any]]:
    """Extract top feature weights or importances from a trained model."""
    importances: List[Tuple[str, float]] = []

    if hasattr(model, "feature_importances_"):
        raw_vals = model.feature_importances_
        importances = [(feature_names[i], float(raw_vals[i])) for i in range(len(feature_names))]
    elif hasattr(model, "coef_"):
        raw_vals = np.abs(model.coef_).ravel()
        importances = [(feature_names[i], float(raw_vals[i])) for i in range(len(feature_names))]

    importances.sort(key=lambda x: x[1], reverse=True)
    return [{"feature": feat, "importance": round(val, 4)} for feat, val in importances[:top_k]]


def log_results_to_mongodb(
    evaluation_summary: Dict[str, Any]
) -> bool:
    """Save model evaluation metrics and run metadata to the MongoDB ml_results collection."""
    client = None
    try:
        client, db = get_mongo_client(timeout_ms=3000)
        ml_col = db["ml_results"]

        doc = {
            "_id": f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "target_regression": "ss_rating",
