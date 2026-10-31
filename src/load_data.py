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
        top_scorer_row = group.loc[group["fb_gls"].idxmax()] if not group.empty else None
        top_scorer_name = top_scorer_row["player"] if top_scorer_row is not None else None
        top_scorer_goals = int(top_scorer_row["fb_gls"]) if top_scorer_row is not None else 0

        doc: Dict[str, Any] = {
            "_id": f"{str(team).replace(' ', '_')}__{season}__{str(league).replace(' ', '_')}",
            "team": str(team),
            "league": str(league),
            "season": int(season),
            "season_label": str(season_label),
            "squad_size": int(len(group)),
            "total_goals": int(group["fb_gls"].sum()),
            "total_assists": int(group["fb_ast"].sum()),
            "total_shots": int(group["fb_sh"].sum()) if "fb_sh" in group.columns else 0,
            "total_shots_on_target": int(group["fb_sot"].sum()) if "fb_sot" in group.columns else 0,
            "total_xg": round(float(group["us_xg"].sum()), 2) if "us_xg" in group.columns and group["us_xg"].notna().any() else 0.0,
            "total_xa": round(float(group["us_xa"].sum()), 2) if "us_xa" in group.columns and group["us_xa"].notna().any() else 0.0,
            "avg_squad_age": round(float(group["age"].mean()), 2) if "age" in group.columns and group["age"].notna().any() else None,
            "total_market_value_eur": round(float(group["fifa_value_eur"].sum()), 2) if "fifa_value_eur" in group.columns and group["fifa_value_eur"].notna().any() else 0.0,
            "top_scorer": {
                "player": top_scorer_name,
                "goals": top_scorer_goals,
            },
        }
        team_docs.append(sanitize_value(doc))

    return team_docs


def prepare_league_documents(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Aggregate player records into league-level summary documents for the leagues collection."""
    grouped = df.groupby("league")
    league_docs: List[Dict[str, Any]] = []

    country_map = {
        "ENG-Premier League": ("England", "ENG"),
        "ESP-La Liga": ("Spain", "ESP"),
        "FRA-Ligue 1": ("France", "FRA"),
        "GER-Bundesliga": ("Germany", "GER"),
        "ITA-Serie A": ("Italy", "ITA"),
    }

    for league, group in grouped:
        country, code = country_map.get(str(league), ("Europe", "EUR"))
        doc: Dict[str, Any] = {
            "_id": str(league),
            "league_name": str(league),
            "country": country,
            "country_code": code,
            "seasons": sorted([str(s) for s in group["season_label"].unique().tolist()]),
            "total_teams": int(group["team"].nunique()),
            "total_players": int(group["player"].nunique()),
            "total_player_seasons": int(len(group)),
            "total_goals": int(group["fb_gls"].sum()),
            "total_assists": int(group["fb_ast"].sum()),
            "total_xg": round(float(group["us_xg"].sum()), 2) if "us_xg" in group.columns and group["us_xg"].notna().any() else 0.0,
            "avg_player_age": round(float(group["age"].mean()), 2) if "age" in group.columns and group["age"].notna().any() else None,
        }
        league_docs.append(sanitize_value(doc))

    return league_docs


