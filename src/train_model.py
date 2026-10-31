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
            "target_classification": "is_high_performer",
            "sample_counts": evaluation_summary["sample_counts"],
            "regression_metrics": evaluation_summary["regression"],
            "classification_metrics": evaluation_summary["classification"],
            "best_models": evaluation_summary["best_models"],
            "top_features": evaluation_summary["top_features"],
        }
        ml_col.insert_one(doc)
        print("   Saved ML evaluation results to MongoDB collection 'ml_results'.")
        return True
    except Exception as e:
        print(f"   ℹ️  Note: Could not log to MongoDB ml_results collection: {e}")
        return False
    finally:
        if client:
            client.close()


def train_and_evaluate(
    csv_path: Optional[Path] = None,
    models_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute complete end-to-end Machine Learning pipeline with strict leak prevention."""
    if csv_path is None:
        csv_path = PROJECT_ROOT / "data/top5-football-dataset-master/data/master/player_seasons.csv"
    if models_dir is None:
        models_dir = PROJECT_ROOT / "models"
    if output_dir is None:
        output_dir = PROJECT_ROOT / "output"

    models_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(" FOOTBALLIQ MACHINE LEARNING TRAINING & EVALUATION PIPELINE")
    print("=" * 70)

    # 1. Load raw dataset
    print(f"\n[INFO] Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)

    # 2. Filter eligible rows
    preprocessor = FootballDataPreprocessor(min_minutes=450, rating_threshold=7.0)
    eligible_df = preprocessor.prepare_data(df)

    stratify_target = (eligible_df["ss_rating"] >= 7.0).astype(int)

    # 3. Train/Test Split (80/20) with stratification
    print("[INFO] Partitioning dataset into Train (80%) and Test (20%) sets...")
    train_df, test_df = train_test_split(
        eligible_df, test_size=0.2, random_state=42, stratify=stratify_target
    )

    # 4. Fit preprocessor ONLY on training set (Leakage Prevention)
    print("[INFO] Fitting preprocessor exclusively on training partition to avoid target leakage...")
    preprocessor.fit(train_df)
    preprocessor.save(models_dir / "preprocessor.joblib")

    X_train, y_reg_train, y_clf_train, meta_train = preprocessor.transform(train_df)
    X_test, y_reg_test, y_clf_test, meta_test = preprocessor.transform(test_df)

    feature_names = preprocessor.feature_columns_
    print(f"   Training samples: {len(X_train):,} | Testing samples: {len(X_test):,}")
    print(f"   Features count: {len(feature_names)}")

    # 5. Train & Evaluate Regression Models
    reg_results, reg_models, best_reg_name = evaluate_regression_models(
        X_train, X_test, y_reg_train, y_reg_test
    )

    # 6. Train & Evaluate Classification Models
    clf_results, clf_models, best_clf_name = evaluate_classification_models(
        X_train, X_test, y_clf_train, y_clf_test
    )

    # 7. Extract Feature Importances from best models
    top_features = extract_feature_importances(reg_models[best_reg_name], feature_names, top_k=8)
    print(f"\n[INFO] Top 8 Feature Importances ({best_reg_name}):")
    for item in top_features:
        print(f"   - {item['feature']:<30}: {item['importance']:.4f}")

    # 8. Save Best Models to Disk
    best_reg_path = models_dir / "best_regression_model.joblib"
    best_clf_path = models_dir / "best_classification_model.joblib"
    joblib.dump(reg_models[best_reg_name], best_reg_path)
    joblib.dump(clf_models[best_clf_name], best_clf_path)
    print(f"\n[SAVED] Best regression model to: {best_reg_path}")
    print(f"[SAVED] Best classification model to: {best_clf_path}")

    # 9. Generate & Save Predictions Table
    test_predictions_df = meta_test.copy()
    test_predictions_df["actual_rating"] = np.round(y_reg_test, 2)
    test_predictions_df["predicted_rating_lr"] = np.round(reg_models["Linear Regression"].predict(X_test), 2)
    test_predictions_df["predicted_rating_rf"] = np.round(reg_models["Random Forest Regressor"].predict(X_test), 2)
    test_predictions_df["actual_high_performer"] = y_clf_test
    test_predictions_df["pred_high_performer_logreg"] = clf_models["Logistic Regression"].predict(X_test)
    test_predictions_df["pred_high_performer_rf"] = clf_models["Random Forest Classifier"].predict(X_test)

    preds_csv_path = output_dir / "ml_test_predictions.csv"
    test_predictions_df.to_csv(preds_csv_path, index=False)
    print(f"[SAVED] Test predictions table ({len(test_predictions_df):,} rows) to: {preds_csv_path}")

    # 10. Save Evaluation Results Summary
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sample_counts": {
            "total_eligible": len(eligible_df),
            "train": len(X_train),
            "test": len(X_test),
        },
        "regression": reg_results,
        "classification": clf_results,
        "best_models": {
            "regression": best_reg_name,
            "classification": best_clf_name,
        },
        "top_features": top_features,
    }

    summary_json_path = output_dir / "ml_evaluation_results.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVED] Evaluation metrics JSON to: {summary_json_path}")

    # 11. Log to MongoDB ml_results collection if reachable
    log_results_to_mongodb(summary)

    print("\n" + "=" * 70)
    print(" MACHINE LEARNING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    train_and_evaluate()

