"""FootballIQ MongoDB Analytics & Query Module.

Implements PyMongo operations:
- CRUD: find, find_one, update_one/update_many, delete_one, insert_one
- Projections, sorting, limiting
- Comparison operators ($gt, $gte, $lt, $lte, $ne)
- Logical operators ($and, $or, $nor)
- Membership operators ($in, $nin)
- Distinct values
- Aggregation pipelines ($match, $group, $sort, $project, $lookup, $unwind)

Practical queries:
- Top goal scorers & assist providers
- Best players by goals per 90
- League-wise player statistics
- Team-wise performance summaries
- Player comparisons
- Young player scouting analysis
- League-wise tactical averages
- Season-over-season player progressions
- Relational joins via $lookup
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pymongo.collection import Collection
from pymongo.database import Database

from src.load_data import get_mongo_client


def get_distinct_metadata(db: Database) -> Dict[str, List[Any]]:
    """Retrieve distinct leagues, seasons, and positions using MongoDB distinct()."""
    player_col = db["player_season_stats"]
    return {
        "leagues": sorted(player_col.distinct("league")),
        "seasons": sorted(player_col.distinct("season_label")),
        "positions": sorted([p for p in player_col.distinct("pos") if p]),
        "total_players": len(player_col.distinct("player")),
        "total_teams": len(player_col.distinct("team")),
    }


def find_player_profile(db: Database, player_name: str) -> List[Dict[str, Any]]:
    """Find player records across seasons using regex find() and projection."""
    player_col = db["player_season_stats"]
    projection = {
        "_id": 1,
        "player": 1,
        "team": 1,
        "league": 1,
        "season_label": 1,
        "pos": 1,
        "age": 1,
        "playing_time_mp": 1,
        "playing_time_min": 1,
        "fb_gls": 1,
        "fb_ast": 1,
        "us_xg": 1,
        "us_xa": 1,
        "ss_rating": 1,
        "fifa_value_eur": 1,
    }

    # Case-insensitive substring match
    query = {"player": {"$regex": player_name, "$options": "i"}}
    cursor = player_col.find(query, projection).sort("season", 1)
    return list(cursor)


def get_top_goal_scorers(
    db: Database,
    league: Optional[str] = None,
    season_label: Optional[str] = None,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Retrieve top goal scorers using find(), comparison filter, sort(), and limit()."""
    player_col = db["player_season_stats"]
    query: Dict[str, Any] = {"fb_gls": {"$gt": 0}}

    if league:
        query["league"] = league
    if season_label:
        query["season_label"] = season_label

    projection = {
        "player": 1,
        "team": 1,
        "league": 1,
        "season_label": 1,
        "pos": 1,
        "playing_time_min": 1,
        "fb_gls": 1,
        "fb_ast": 1,
        "us_xg": 1,
        "ss_rating": 1,
        "_id": 0,
    }

    cursor = player_col.find(query, projection).sort("fb_gls", -1).limit(limit)
    return list(cursor)


def get_top_assist_providers(
    db: Database,
    league: Optional[str] = None,
    season_label: Optional[str] = None,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Retrieve top assist providers using find(), sort(), and limit()."""
    player_col = db["player_season_stats"]
    query: Dict[str, Any] = {"fb_ast": {"$gt": 0}}

    if league:
        query["league"] = league
    if season_label:
        query["season_label"] = season_label

    projection = {
        "player": 1,
        "team": 1,
        "league": 1,
        "season_label": 1,
        "pos": 1,
        "playing_time_min": 1,
        "fb_ast": 1,
        "fb_gls": 1,
        "us_xa": 1,
        "ss_rating": 1,
        "_id": 0,
    }

