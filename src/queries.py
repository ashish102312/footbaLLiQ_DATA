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

    cursor = player_col.find(query, projection).sort("fb_ast", -1).limit(limit)
    return list(cursor)


def get_best_goals_per_90(
    db: Database, min_minutes: int = 900, limit: int = 10
) -> List[Dict[str, Any]]:
    """Calculate best goals per 90 using MongoDB Aggregation Pipeline ($match, $project, $sort, $limit)."""
    player_col = db["player_season_stats"]

    pipeline = [
        # Filter players with significant minutes and positive goals
        {
            "$match": {
                "playing_time_min": {"$gte": min_minutes},
                "fb_gls": {"$gt": 0},
            }
        },
        # Compute goals_p90 = (fb_gls / (playing_time_min / 90.0))
        {
            "$project": {
                "_id": 0,
                "player": 1,
                "team": 1,
                "league": 1,
                "season_label": 1,
                "age": 1,
                "pos": 1,
                "playing_time_min": 1,
                "fb_gls": 1,
                "us_xg": 1,
                "goals_p90": {
                    "$round": [
                        {
                            "$multiply": [
                                {"$divide": ["$fb_gls", "$playing_time_min"]},
                                90.0,
                            ]
                        },
                        2,
                    ]
                },
            }
        },
        {"$sort": {"goals_p90": -1}},
        {"$limit": limit},
    ]

    return list(player_col.aggregate(pipeline))


def get_league_player_statistics(db: Database) -> List[Dict[str, Any]]:
    """Aggregate league-wide player statistics using $group, $sort, and $project."""
    player_col = db["player_season_stats"]

    pipeline = [
        {
            "$group": {
                "_id": "$league",
                "total_player_seasons": {"$sum": 1},
                "unique_players": {"$addToSet": "$player"},
                "total_goals": {"$sum": "$fb_gls"},
                "total_assists": {"$sum": "$fb_ast"},
                "avg_player_age": {"$avg": "$age"},
                "avg_match_rating": {"$avg": "$ss_rating"},
                "total_market_value": {"$sum": "$fifa_value_eur"},
            }
        },
        {
            "$project": {
                "_id": 0,
                "league": "$_id",
                "total_player_seasons": 1,
                "unique_players_count": {"$size": "$unique_players"},
                "total_goals": 1,
                "total_assists": 1,
                "avg_goals_per_player": {
                    "$round": [
                        {"$divide": ["$total_goals", "$total_player_seasons"]},
                        2,
                    ]
                },
                "avg_player_age": {"$round": ["$avg_player_age", 1]},
                "avg_match_rating": {"$round": ["$avg_match_rating", 2]},
                "total_market_value_eur": "$total_market_value",
            }
        },
        {"$sort": {"total_goals": -1}},
    ]

    return list(player_col.aggregate(pipeline))


def get_team_performance_summary(
    db: Database, league: Optional[str] = None, limit: int = 15
) -> List[Dict[str, Any]]:
    """Query team-level summary documents from the teams collection."""
    teams_col = db["teams"]
    query: Dict[str, Any] = {}
    if league:
        query["league"] = league

    projection = {
        "_id": 0,
        "team": 1,
        "league": 1,
        "season_label": 1,
        "squad_size": 1,
        "total_goals": 1,
        "total_assists": 1,
        "total_xg": 1,
        "avg_squad_age": 1,
        "total_market_value_eur": 1,
        "top_scorer": 1,
    }

    cursor = teams_col.find(query, projection).sort("total_goals", -1).limit(limit)
    return list(cursor)


def compare_two_players(
    db: Database, player1_name: str, player2_name: str
) -> Dict[str, Any]:
    """Side-by-side player comparison using $in operator and projection."""
    player_col = db["player_season_stats"]

    # Match both players using $or with regex
    query = {
        "$or": [
            {"player": {"$regex": player1_name, "$options": "i"}},
            {"player": {"$regex": player2_name, "$options": "i"}},
        ]
    }

    projection = {
        "_id": 0,
        "player": 1,
        "team": 1,
        "league": 1,
        "season_label": 1,
        "pos": 1,
        "age": 1,
        "playing_time_min": 1,
        "fb_gls": 1,
        "fb_ast": 1,
        "us_xg": 1,
        "us_xa": 1,
        "ss_rating": 1,
        "ss_accuratePassesPercentage": 1,
        "ss_totalDuelsWonPercentage": 1,
        "fifa_value_eur": 1,
    }

    records = list(player_col.find(query, projection).sort("season_label", -1))
    return {
        "searched": [player1_name, player2_name],
        "records": records,
    }


def get_young_talents_scouting(
    db: Database,
    max_age: int = 21,
    min_minutes: int = 450,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Scout young talents using comparison ($lte, $gte) and logical operators ($and, $or)."""
    player_col = db["player_season_stats"]

    # Criteria: Young (age <= max_age), active (min >= min_minutes), and either high rating or positive xG
    query = {
        "$and": [
            {"age": {"$lte": max_age}},
            {"playing_time_min": {"$gte": min_minutes}},
            {
                "$or": [
                    {"ss_rating": {"$gte": 6.8}},
                    {"us_xg": {"$gte": 2.0}},
                    {"fb_gls": {"$gte": 3}},
                ]
            },
        ]
    }

    projection = {
        "_id": 0,
        "player": 1,
        "team": 1,
        "league": 1,
        "season_label": 1,
        "age": 1,
        "pos": 1,
        "playing_time_min": 1,
        "fb_gls": 1,
        "fb_ast": 1,
        "us_xg": 1,
        "ss_rating": 1,
        "fifa_value_eur": 1,
    }

    cursor = player_col.find(query, projection).sort([("ss_rating", -1), ("fb_gls", -1)]).limit(limit)
    return list(cursor)


def filter_by_leagues_and_positions(
    db: Database,
    target_leagues: List[str],
    excluded_positions: List[str],
    min_rating: float = 7.0,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Demonstrate $in and $nin operators to filter players across leagues and positions."""
    player_col = db["player_season_stats"]

    query = {
        "league": {"$in": target_leagues},
        "pos": {"$nin": excluded_positions},
        "ss_rating": {"$gte": min_rating},
        "playing_time_min": {"$gte": 450},
    }

    projection = {
        "_id": 0,
        "player": 1,
        "team": 1,
        "league": 1,
        "season_label": 1,
        "pos": 1,
        "ss_rating": 1,
        "fb_gls": 1,
        "fb_ast": 1,
        "playing_time_min": 1,
    }

    return list(player_col.find(query, projection).sort("ss_rating", -1).limit(limit))


def get_player_season_comparison(db: Database, player_name: str) -> List[Dict[str, Any]]:
    """Track player progression across seasons (e.g., 2024/25 vs 2025/26)."""
    player_col = db["player_season_stats"]

    query = {"player": {"$regex": f"^{player_name}$", "$options": "i"}}
    projection = {
        "_id": 0,
        "player": 1,
        "team": 1,
        "league": 1,
        "season_label": 1,
        "playing_time_min": 1,
        "fb_gls": 1,
        "fb_ast": 1,
        "us_xg": 1,
        "ss_rating": 1,
        "fifa_value_eur": 1,
    }

    return list(player_col.find(query, projection).sort("season_label", 1))


def get_player_with_team_lookup(
    db: Database, min_goals: int = 10, limit: int = 5
) -> List[Dict[str, Any]]:
    """Demonstrate relational join using $lookup, $unwind, and $project between collections.

    Joins player_season_stats with teams collection to compute player's share of team goals.
    """
    player_col = db["player_season_stats"]

    pipeline = [
        {"$match": {"fb_gls": {"$gte": min_goals}}},
        # Join with teams collection matching team and season
        {
            "$lookup": {
                "from": "teams",
                "let": {"p_team": "$team", "p_season": "$season"},
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {
                                "$and": [
                                    {"$eq": ["$team", "$$p_team"]},
                                    {"$eq": ["$season", "$$p_season"]},
                                ]
                            }
                        }
                    }
                ],
                "as": "team_info",
            }
        },
        {"$unwind": "$team_info"},
        {
            "$project": {
                "_id": 0,
                "player": 1,
                "team": 1,
                "league": 1,
                "season_label": 1,
                "player_goals": "$fb_gls",
                "team_total_goals": "$team_info.total_goals",
                "goal_share_pct": {
                    "$cond": [
                        {"$gt": ["$team_info.total_goals", 0]},
                        {
                            "$round": [
                                {
                                    "$multiply": [
                                        {"$divide": ["$fb_gls", "$team_info.total_goals"]},
                                        100.0,
                                    ]
                                },
                                1,
                            ]
                        },
                        0.0,
                    ]
                },
            }
        },
        {"$sort": {"player_goals": -1}},
