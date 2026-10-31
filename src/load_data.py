"""FootballIQ Data Ingestion & MongoDB Loader Module.

Handles dataset validation, sanitization (NaN/NumPy type conversions),
connection to MongoDB (local or Atlas), idempotent bulk insertion into
collections (player_season_stats, teams, leagues), and index creation.
"""

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pymongo import ASCENDING, DESCENDING, MongoClient, ReplaceOne
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, PyMongoError


def load_env_config(env_path: Optional[Path] = None) -> Dict[str, str]:
    """Parse key-value pairs from .env file into a dictionary without external dependencies."""
    config: Dict[str, str] = {}
    if env_path is None:
        env_path = Path(__file__).resolve().parent.parent / ".env"

    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    config[key] = val

    # Environment variables override .env file
    for k in ["MONGODB_URI", "MONGODB_DATABASE"]:
        if k in os.environ:
            config[k] = os.environ[k]

    # Provide safe defaults if not configured
    config.setdefault("MONGODB_URI", "mongodb://localhost:27017/")
    config.setdefault("MONGODB_DATABASE", "footballiq")
    return config


def get_mongo_client(
    uri: Optional[str] = None, timeout_ms: int = 5000
) -> Tuple[MongoClient, Database]:
    """Connect to MongoDB instance and return client and database handles.

    Supports both local instances and MongoDB Atlas connection URIs.
    """
    config = load_env_config()
    target_uri = uri or config.get("MONGODB_URI", "mongodb://localhost:27017/")
    db_name = config.get("MONGODB_DATABASE", "footballiq")

    client: MongoClient = MongoClient(
        target_uri,
        serverSelectionTimeoutMS=timeout_ms,
        connectTimeoutMS=timeout_ms,
        socketTimeoutMS=timeout_ms,
    )
    db = client[db_name]
    return client, db


def test_db_connection(uri: Optional[str] = None) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """Test connectivity to MongoDB and return status, message, and server info."""
    client = None
    try:
        client, db = get_mongo_client(uri, timeout_ms=3000)
        # Force a network roundtrip to test connection
        server_info = client.server_info()
        version = server_info.get("version", "unknown")
        db_name = db.name
        collections = db.list_collection_names()
        client.close()
        return (
            True,
            f"Successfully connected to MongoDB v{version} (Database: '{db_name}'). Existing collections: {collections}",
            server_info,
        )
    except ConnectionFailure as cf:
        return (
            False,
            f"MongoDB Connection Failure: Could not reach MongoDB server. Details: {cf}",
            None,
        )
    except Exception as e:
        return (
            False,
            f"MongoDB Error: {e}",
            None,
        )
    finally:
        if client:
            client.close()


def sanitize_value(val: Any) -> Any:
    """Recursively convert NaN, inf, and numpy data types into BSON-compatible Python types."""
    if val is None:
        return None
    if isinstance(val, (float, np.floating)):
        if math.isnan(val) or math.isinf(val):
            return None
        return float(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, np.ndarray):
        return [sanitize_value(x) for x in val.tolist()]
    if isinstance(val, dict):
        return {str(k): sanitize_value(v) for k, v in val.items()}
    if isinstance(val, list):
        return [sanitize_value(x) for x in val]
    if pd.isna(val):
        return None
    return val


def prepare_player_documents(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Transform raw player seasons DataFrame into sanitized MongoDB documents with deterministic _id."""
    documents: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        doc: Dict[str, Any] = {}
        for col in df.columns:
            val = row[col]
            doc[col] = sanitize_value(val)

        # Build compound deterministic primary key
        # Primary key: player_team_season_league (guaranteed 0 duplicates)
        player_clean = str(doc.get("player", "unknown")).replace(" ", "_")
        team_clean = str(doc.get("team", "unknown")).replace(" ", "_")
        season_clean = str(doc.get("season", "0"))
        league_clean = str(doc.get("league", "unknown")).replace(" ", "_")

        doc["_id"] = f"{player_clean}__{team_clean}__{season_clean}__{league_clean}"
        documents.append(doc)

    return documents


def prepare_team_documents(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Aggregate player records into team-level summary documents for the teams collection."""
    grouped = df.groupby(["team", "league", "season", "season_label"], as_index=False)
    team_docs: List[Dict[str, Any]] = []

    for (team, league, season, season_label), group in grouped:
