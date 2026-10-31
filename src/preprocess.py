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
