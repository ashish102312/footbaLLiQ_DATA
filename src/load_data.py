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


def create_indexes(db: Database) -> Dict[str, List[str]]:
    """Create optimized indexes on all collections for high-performance querying."""
    indexes_created: Dict[str, List[str]] = {}

    # 1. Indexes for player_season_stats
    player_col: Collection = db["player_season_stats"]
    p_idx = [
        player_col.create_index([("player", ASCENDING)], name="idx_player"),
        player_col.create_index([("team", ASCENDING)], name="idx_team"),
        player_col.create_index([("league", ASCENDING)], name="idx_league"),
        player_col.create_index([("season", ASCENDING)], name="idx_season"),
        player_col.create_index([("league", ASCENDING), ("season", ASCENDING)], name="idx_league_season"),
        player_col.create_index([("player", ASCENDING), ("season", ASCENDING)], name="idx_player_season"),
        player_col.create_index([("fb_gls", DESCENDING)], name="idx_goals_desc"),
        player_col.create_index([("fb_ast", DESCENDING)], name="idx_assists_desc"),
        player_col.create_index([("ss_rating", DESCENDING)], name="idx_ss_rating_desc"),
        player_col.create_index([("fifa_value_eur", DESCENDING)], name="idx_fifa_value_desc"),
    ]
    indexes_created["player_season_stats"] = p_idx

    # 2. Indexes for teams
    teams_col: Collection = db["teams"]
    t_idx = [
        teams_col.create_index([("team", ASCENDING)], name="idx_team_name"),
        teams_col.create_index([("league", ASCENDING)], name="idx_team_league"),
        teams_col.create_index([("season", ASCENDING)], name="idx_team_season"),
        teams_col.create_index([("league", ASCENDING), ("season", ASCENDING)], name="idx_team_league_season"),
    ]
    indexes_created["teams"] = t_idx

    # 3. Indexes for leagues
    leagues_col: Collection = db["leagues"]
    l_idx = [
        leagues_col.create_index([("league_name", ASCENDING)], name="idx_league_name"),
        leagues_col.create_index([("country_code", ASCENDING)], name="idx_country_code"),
    ]
    indexes_created["leagues"] = l_idx

    # 4. Indexes for ml_results
    ml_col: Collection = db["ml_results"]
    m_idx = [
        ml_col.create_index([("task", ASCENDING)], name="idx_ml_task"),
        ml_col.create_index([("model_name", ASCENDING)], name="idx_ml_model"),
        ml_col.create_index([("created_at", DESCENDING)], name="idx_ml_created_at"),
    ]
    indexes_created["ml_results"] = m_idx

    return indexes_created


def bulk_upsert(collection: Collection, documents: List[Dict[str, Any]], batch_size: int = 1000) -> int:
    """Perform batched upsert using ReplaceOne to prevent duplicate insertions."""
    if not documents:
        return 0

    total_upserted = 0
    total_docs = len(documents)

    for i in range(0, total_docs, batch_size):
        batch = documents[i : i + batch_size]
        operations = [ReplaceOne({"_id": doc["_id"]}, doc, upsert=True) for doc in batch]
        result = collection.bulk_write(operations, ordered=False)
        total_upserted += result.upserted_count + result.modified_count + result.matched_count

    return total_upserted


def import_csv_to_mongodb(
    csv_path: Optional[Path] = None, uri: Optional[str] = None
) -> Dict[str, Any]:
    """Execute complete ingestion pipeline: read CSV, clean, prepare, and bulk upsert to MongoDB."""
    if csv_path is None:
        csv_path = Path(__file__).resolve().parent.parent / "data/top5-football-dataset-master/data/master/player_seasons.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Primary dataset not found at expected path: {csv_path}")

    print(f"\n[INFO] Reading dataset from {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    raw_rows, raw_cols = df.shape
    print(f"       Successfully loaded {raw_rows:,} records with {raw_cols} columns.")

    # Connect to MongoDB
    print("\n[INFO] Connecting to MongoDB...")
    client, db = get_mongo_client(uri)
    try:
        # Verify connection
        client.admin.command("ping")
        print(f"       Connected to MongoDB database: '{db.name}'")
    except Exception as e:
        client.close()
        raise ConnectionFailure(f"Failed to connect to MongoDB: {e}")

    # Prepare Documents
    print("\n[INFO] Transforming records for MongoDB storage...")
    player_docs = prepare_player_documents(df)
    team_docs = prepare_team_documents(df)
    league_docs = prepare_league_documents(df)

    print(f"       Prepared {len(player_docs):,} player documents.")
    print(f"       Prepared {len(team_docs):,} team summary documents.")
    print(f"       Prepared {len(league_docs):,} league documents.")

    # Bulk Upsert
    print("\n[INFO] Performing bulk upserts into MongoDB...")
    p_count = bulk_upsert(db["player_season_stats"], player_docs)
    t_count = bulk_upsert(db["teams"], team_docs)
    l_count = bulk_upsert(db["leagues"], league_docs)

    # Create Indexes
    print("\n[INFO] Creating database indexes for query optimization...")
    indexes = create_indexes(db)
    for col_name, idx_list in indexes.items():
        print(f"       Created {len(idx_list)} indexes on '{col_name}'.")

    # Verify Counts
    actual_p_count = db["player_season_stats"].count_documents({})
    actual_t_count = db["teams"].count_documents({})
    actual_l_count = db["leagues"].count_documents({})
    actual_m_count = db["ml_results"].count_documents({})

    print("\n[SUCCESS] Document Count Verification in MongoDB:")
    print(f"   - 'player_season_stats': {actual_p_count:,} documents")
    print(f"   - 'teams':               {actual_t_count:,} documents")
    print(f"   - 'leagues':             {actual_l_count:,} documents")
    print(f"   - 'ml_results':          {actual_m_count:,} documents")

    client.close()

    return {
        "status": "success",
        "csv_path": str(csv_path),
        "counts": {
            "player_season_stats": actual_p_count,
            "teams": actual_t_count,
            "leagues": actual_l_count,
            "ml_results": actual_m_count,
        },
    }


if __name__ == "__main__":
    import sys

    print("=" * 60)
    print(" FootballIQ - Data Ingestion & MongoDB Loader")
    print("=" * 60)

    is_ok, msg, _ = test_db_connection()
    print(f"Connection Status: {'ONLINE' if is_ok else 'OFFLINE'}")
    print(msg)

    if not is_ok:
        print("\n[WARNING] MongoDB is currently offline or unreachable.")
        print("Please check your .env configuration or start your local MongoDB daemon:")
        print("  brew services start mongodb-community  (macOS)")
        print("  mongod --config /opt/homebrew/etc/mongod.conf")
        sys.exit(1)

    import_csv_to_mongodb()

