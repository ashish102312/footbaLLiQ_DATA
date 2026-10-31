"""FootballIQ Data Preprocessing Module.

Prepares raw football player-season data for machine learning modeling:
- Validates data integrity, shapes, and null distributions
- Imputes missing technical and tactical statistics using median strategies
- Filters noisy sample records (players with insufficient match exposure)
- Enforces strict target leakage prevention
- Normalizes numerical features via StandardScaler
- One-hot encodes categorical attributes (league, position group)
- Persists and loads reusable preprocessors using joblib
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def simplify_position(pos: Any) -> str:
    """Normalize multi-position designations (e.g. 'MF,FW' -> 'MF', 'DF,MF' -> 'DF')."""
    if pd.isna(pos) or not pos:
        return "MF"
    pos_str = str(pos).strip()
    primary = pos_str.split(",")[0].strip().upper()
    if primary in {"FW", "MF", "DF", "GK"}:
        return primary
    return "MF"


class FootballDataPreprocessor:
    """Reusable, leakage-free preprocessing pipeline for FootballIQ analytics and ML."""

    NUMERICAL_FEATURES: List[str] = [
        # Attacking & Shooting
        "fb_gls",
        "fb_ast",
        "fb_sh",
        "fb_sot",
        "us_xg",
        "us_xa",
        "us_key_passes",
        "np_xg_p90",
        "xa_p90",
        # Passing & Possession
        "ss_accuratePassesPercentage",
        "ss_keyPasses",
        "ss_touches",
        "ss_successfulDribblesPercentage",
        "ss_possessionLost",
        "ss_bigChancesCreated",
        # Defensive & Duels
        "ss_groundDuelsWonPercentage",
        "ss_aerialDuelsWonPercentage",
        "ss_interceptions",
        "ss_clearances",
        "tackles_won_p90",
        "interceptions_p90",
        # Physical & Experience
        "age",
        "playing_time_min",
    ]

    CATEGORICAL_FEATURES: List[str] = [
        "league",
        "pos_group",
    ]

    # Explicit list of columns that must NEVER be used as features to prevent data leakage
    LEAKAGE_COLUMNS: List[str] = [
        "ss_rating",
        "ss_totalRating",
        "ss_countRating",
        "ss_id",
        "ss_player_id",
        "fifa_id",
    ]

    def __init__(self, min_minutes: int = 450, rating_threshold: float = 7.0):
        self.min_minutes = min_minutes
        self.rating_threshold = rating_threshold
        self.medians_: Dict[str, float] = {}
        self.scaler_: Optional[StandardScaler] = None
        self.feature_columns_: List[str] = []
        self.fitted_: bool = False

    def validate_raw_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive data validation checks on the dataset."""
        total_rows, total_cols = df.shape
        pk_dupes = int(df.duplicated(["league", "season", "team", "player"]).sum()) if all(
            c in df.columns for c in ["league", "season", "team", "player"]
        ) else 0

        null_counts = df.isnull().sum()
        top_nulls = {col: int(cnt) for col, cnt in null_counts.sort_values(ascending=False).head(10).items()}

        seasons = sorted(df["season_label"].unique().tolist()) if "season_label" in df.columns else []
        leagues = sorted(df["league"].unique().tolist()) if "league" in df.columns else []

        return {
            "total_rows": total_rows,
            "total_columns": total_cols,
            "primary_key_duplicates": pk_dupes,
            "seasons": seasons,
            "leagues": leagues,
            "top_missing_columns": top_nulls,
        }

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter eligible records and generate necessary feature columns."""
        data = df.copy()

        # Filter out records with insufficient minutes (e.g. cameo appearances with unrepresentative stats)
        if "playing_time_min" in data.columns:
            data = data[data["playing_time_min"] >= self.min_minutes]

        # Require target column ss_rating to be present
        if "ss_rating" in data.columns:
            data = data[data["ss_rating"].notna()]

        # Generate simplified position group
        if "pos" in data.columns and "pos_group" not in data.columns:
            data["pos_group"] = data["pos"].apply(simplify_position)

        return data.reset_index(drop=True)

    def fit(self, df: pd.DataFrame) -> "FootballDataPreprocessor":
        """Fit imputation medians and standard scaler exclusively on training dataset."""
        cleaned = self.prepare_data(df)

        # 1. Compute and store medians for numerical columns
        for col in self.NUMERICAL_FEATURES:
            if col in cleaned.columns and cleaned[col].notna().any():
                self.medians_[col] = float(cleaned[col].median())
            else:
                self.medians_[col] = 0.0

        # Build feature DataFrame
        X_num = pd.DataFrame(index=cleaned.index)
        for col in self.NUMERICAL_FEATURES:
            if col in cleaned.columns:
                X_num[col] = cleaned[col].fillna(self.medians_[col])
            else:
                X_num[col] = self.medians_[col]

        # 2. One-hot encode categorical features
        cat_df = cleaned[self.CATEGORICAL_FEATURES] if all(c in cleaned.columns for c in self.CATEGORICAL_FEATURES) else pd.DataFrame(index=cleaned.index)
        X_cat = pd.get_dummies(cat_df, drop_first=True, dtype=float)

        X = pd.concat([X_num, X_cat], axis=1)
        self.feature_columns_ = X.columns.tolist()

        # 3. Fit StandardScaler on all features
        self.scaler_ = StandardScaler()
        self.scaler_.fit(X)
        self.fitted_ = True

        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        """Transform dataset using fitted preprocessor into ML-ready numpy arrays.

        Returns:
            X_scaled: Transformed feature matrix
            y_reg: Continuous regression target (ss_rating)
            y_clf: Binary classification target (1 if ss_rating >= threshold else 0)
            metadata: Player metadata (player, team, league, season_label)
        """
        if not self.fitted_ or self.scaler_ is None:
            raise RuntimeError("Preprocessor has not been fitted yet. Call fit() first.")

        cleaned = self.prepare_data(df)

        # Fill numerical features using stored medians
        X_num = pd.DataFrame(index=cleaned.index)
        for col in self.NUMERICAL_FEATURES:
            if col in cleaned.columns:
                X_num[col] = cleaned[col].fillna(self.medians_.get(col, 0.0))
            else:
                X_num[col] = self.medians_.get(col, 0.0)

        # One-hot encode categorical features and align to fitted schema
        cat_df = cleaned[self.CATEGORICAL_FEATURES] if all(c in cleaned.columns for c in self.CATEGORICAL_FEATURES) else pd.DataFrame(index=cleaned.index)
        X_cat = pd.get_dummies(cat_df, drop_first=True, dtype=float)

        X = pd.concat([X_num, X_cat], axis=1)

        # Reindex to ensure identical feature columns as during fit
        X = X.reindex(columns=self.feature_columns_, fill_value=0.0)

        X_scaled = self.scaler_.transform(X)

        # Target variables
        y_reg = cleaned["ss_rating"].to_numpy(dtype=float)
        y_clf = (cleaned["ss_rating"] >= self.rating_threshold).astype(int).to_numpy()

        metadata_cols = [c for c in ["player", "team", "league", "season_label", "pos", "age", "playing_time_min"] if c in cleaned.columns]
        metadata = cleaned[metadata_cols].copy()

        return X_scaled, y_reg, y_clf, metadata

    def fit_transform(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        """Fit preprocessor on dataset and immediately transform it."""
        self.fit(df)
        return self.transform(df)

    def save(self, filepath: Path) -> None:
        """Persist preprocessor pipeline to disk using joblib."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: Path) -> "FootballDataPreprocessor":
        """Load persisted preprocessor pipeline from disk."""
        if not filepath.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found at {filepath}")
        return joblib.load(filepath)


def run_preprocessing_pipeline(
    csv_path: Optional[Path] = None,
    save_path: Optional[Path] = None,
) -> Tuple[FootballDataPreprocessor, Dict[str, Any]]:
    """Execute preprocessing workflow, validate data, fit preprocessor, and save artifacts."""
    if csv_path is None:
        csv_path = PROJECT_ROOT / "data/top5-football-dataset-master/data/master/player_seasons.csv"
    if save_path is None:
        save_path = PROJECT_ROOT / "models/preprocessor.joblib"

    print(f"\n[INFO] Running Preprocessing Pipeline on {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)

    preprocessor = FootballDataPreprocessor(min_minutes=450, rating_threshold=7.0)
    validation_info = preprocessor.validate_raw_data(df)

    print(f"   Total Records: {validation_info['total_rows']:,} | Columns: {validation_info['total_columns']}")
    print(f"   Primary Key Duplicates: {validation_info['primary_key_duplicates']}")
    print(f"   Seasons: {', '.join(validation_info['seasons'])}")
    print(f"   Leagues: {len(validation_info['leagues'])} leagues")

    X_scaled, y_reg, y_clf, meta = preprocessor.fit_transform(df)
    preprocessor.save(save_path)

    print(f"   Filtered eligible player-seasons (min >= 450 mins): {len(y_reg):,}")
    print(f"   Feature matrix shape: {X_scaled.shape}")
    print(f"   High-performer class distribution: {np.mean(y_clf) * 100:.1f}% positive (Rating >= 7.0)")
    print(f"   Saved preprocessor artifact to: {save_path}")

    return preprocessor, {
        "validation": validation_info,
        "sample_count": len(y_reg),
        "feature_count": X_scaled.shape[1],
        "feature_names": preprocessor.feature_columns_,
    }


if __name__ == "__main__":
    run_preprocessing_pipeline()

