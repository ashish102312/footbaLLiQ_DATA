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
        {"$limit": limit},
    ]

    return list(player_col.aggregate(pipeline))


def demonstrate_updates_and_deletions(db: Database) -> Dict[str, Any]:
    """Demonstrate MongoDB update_one, update_many, delete_one, and temporary document lifecycle."""
    player_col = db["player_season_stats"]

    # 1. Update one document: Add a scouting tag to a top performer
    target = player_col.find_one({"fb_gls": {"$gte": 15}})
    if not target:
        target = player_col.find_one()

    target_id = target["_id"]
    player_name = target["player"]

    # Update operation using $set and $currentDate
    update_res = player_col.update_one(
        {"_id": target_id},
        {
            "$set": {
                "scouting_status": "Scouted - High Priority Target",
                "scout_priority": 1,
            },
            "$currentDate": {"scouted_at": True},
        },
    )

    # Verify update
    updated_doc = player_col.find_one({"_id": target_id}, {"player": 1, "scouting_status": 1, "scout_priority": 1, "scouted_at": 1})

    # Cleanup updated field using $unset to preserve dataset purity
    cleanup_res = player_col.update_one(
        {"_id": target_id},
        {"$unset": {"scouting_status": "", "scout_priority": "", "scouted_at": ""}},
    )

    # 2. Demonstration of insert_one and delete_one lifecycle
    temp_col = db["ml_results"]
    temp_doc = {
        "_id": "test_scouting_probe_001",
        "task": "scouting_probe",
        "model_name": "TemporaryTest",
        "status": "pending_validation",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    insert_res = temp_col.insert_one(temp_doc)
    doc_exists_before = temp_col.find_one({"_id": "test_scouting_probe_001"}) is not None
    delete_res = temp_col.delete_one({"_id": "test_scouting_probe_001"})
    doc_exists_after = temp_col.find_one({"_id": "test_scouting_probe_001"}) is not None

    return {
        "update_test": {
            "player": player_name,
            "target_id": target_id,
            "modified_count": update_res.modified_count,
            "verified_status": updated_doc.get("scouting_status") if updated_doc else None,
            "cleaned_up": cleanup_res.modified_count == 1,
        },
        "delete_test": {
            "inserted_id": str(insert_res.inserted_id),
            "doc_existed_before_delete": doc_exists_before,
            "deleted_count": delete_res.deleted_count,
            "doc_exists_after_delete": doc_exists_after,
        },
    }


def run_all_queries(db: Database) -> None:
    """Execute and display formatted results for all implemented analytics queries."""
    print("\n" + "=" * 70)
    print(" FOOTBALLIQ MONGODB ANALYTICS & INTELLIGENCE REPORT")
    print("=" * 70)

    # 1. Distinct Metadata
    print("\n[QUERY 1] Database Metadata (distinct values):")
    meta = get_distinct_metadata(db)
    print(f"   Leagues ({len(meta['leagues'])}): {', '.join(meta['leagues'])}")
    print(f"   Seasons ({len(meta['seasons'])}): {', '.join(meta['seasons'])}")
    print(f"   Positions ({len(meta['positions'])}): {', '.join(meta['positions'])}")
    print(f"   Unique Players: {meta['total_players']:,} | Unique Teams: {meta['total_teams']:,}")

    # 2. Top Goal Scorers
    print("\n[QUERY 2] Top 5 Goal Scorers Across All Leagues (find + sort + limit):")
    scorers = get_top_goal_scorers(db, limit=5)
    for i, s in enumerate(scorers, 1):
        xg_str = f"xG: {s.get('us_xg', 0.0):.2f}" if s.get("us_xg") is not None else "xG: N/A"
        print(f"   {i}. {s['player']} ({s['team']} - {s['league']}): {s['fb_gls']} goals in {s['playing_time_min']:,} mins ({xg_str})")

    # 3. Top Assist Providers
    print("\n[QUERY 3] Top 5 Assist Providers Across All Leagues:")
    assisters = get_top_assist_providers(db, limit=5)
    for i, a in enumerate(assisters, 1):
        xa_str = f"xA: {a.get('us_xa', 0.0):.2f}" if a.get("us_xa") is not None else "xA: N/A"
        print(f"   {i}. {a['player']} ({a['team']} - {a['league']}): {a['fb_ast']} assists ({xa_str})")

    # 4. Best Goals per 90 (Aggregation: $match, $project, $sort, $limit)
    print("\n[QUERY 4] Best Goals per 90 Minutes (Min. 900 minutes played):")
    p90_leaders = get_best_goals_per_90(db, min_minutes=900, limit=5)
    for i, p in enumerate(p90_leaders, 1):
        print(f"   {i}. {p['player']} ({p['team']}): {p['goals_p90']:.2f} G/90 ({p['fb_gls']} goals in {p['playing_time_min']} mins)")

    # 5. League-wise Player Statistics ($group aggregation)
    print("\n[QUERY 5] League Performance Breakdown (Aggregation: $group):")
    league_stats = get_league_player_statistics(db)
    print(f"   {'League':<22} {'Player-Seasons':<16} {'Total Goals':<14} {'Avg Goals/Player':<18} {'Avg Age':<10}")
    print("   " + "-" * 78)
    for ls in league_stats:
        print(f"   {ls['league']:<22} {ls['total_player_seasons']:<16} {ls['total_goals']:<14} {ls['avg_goals_per_player']:<18} {ls['avg_player_age']:<10}")

    # 6. Team Performance Summaries
    print("\n[QUERY 6] Top 5 Teams by Goals Scored (teams collection):")
    teams = get_team_performance_summary(db, limit=5)
    for i, t in enumerate(teams, 1):
        scorer = t.get("top_scorer", {})
        scorer_info = f"Top Scorer: {scorer.get('player')} ({scorer.get('goals')}G)" if scorer else ""
        print(f"   {i}. {t['team']} ({t['league']} - {t['season_label']}): {t['total_goals']} goals, {t['total_assists']} assists | {scorer_info}")

    # 7. Player Comparison
    print("\n[QUERY 7] Head-to-Head Comparison: Erling Haaland vs Robert Lewandowski:")
    comp = compare_two_players(db, "Erling Haaland", "Robert Lewandowski")
    for r in comp["records"]:
        val_str = f"EUR {r.get('fifa_value_eur', 0):,}" if r.get("fifa_value_eur") else "N/A"
        rating_str = f"{r.get('ss_rating', 0.0):.2f}" if r.get("ss_rating") else "N/A"
        print(f"   - {r['player']} ({r['team']} | {r['season_label']}): {r['fb_gls']} G, {r['fb_ast']} A, Rating: {rating_str}, Value: {val_str}")

    # 8. Young Player Scouting (Comparison & Logical Operators)
    print("\n[QUERY 8] Young Talent Scouting Insights (Age <= 21, Min. 450 minutes):")
    youngsters = get_young_talents_scouting(db, max_age=21, min_minutes=450, limit=5)
    for i, y in enumerate(youngsters, 1):
        rating_str = f"Rating: {y.get('ss_rating', 0.0):.2f}" if y.get("ss_rating") else ""
        val_str = f"Value: EUR {y.get('fifa_value_eur', 0):,}" if y.get("fifa_value_eur") else ""
        print(f"   {i}. {y['player']} ({y['team']}, Age: {int(y['age']) if y['age'] else 'N/A'}, Pos: {y['pos']}): {y['fb_gls']}G, {y['fb_ast']}A | {rating_str} | {val_str}")

    # 9. Relational Join ($lookup and $unwind)
    print("\n[QUERY 9] Relational Aggregation ($lookup & $unwind: Player Share of Team Goals):")
    shares = get_player_with_team_lookup(db, min_goals=12, limit=5)
    for i, row in enumerate(shares, 1):
        print(f"   {i}. {row['player']} ({row['team']}): {row['player_goals']} of team's {row['team_total_goals']} goals ({row['goal_share_pct']}% team goal share)")

    # 10. CRUD Lifecycle Test
    print("\n[QUERY 10] CRUD Operations (update_one, $unset, insert_one, delete_one):")
    crud_res = demonstrate_updates_and_deletions(db)
    print(f"   Update Test: Tagged '{crud_res['update_test']['player']}' -> Verified: '{crud_res['update_test']['verified_status']}' -> Cleaned: {crud_res['update_test']['cleaned_up']}")
    print(f"   Delete Test: Inserted doc '{crud_res['delete_test']['inserted_id']}' -> Existed: {crud_res['delete_test']['doc_existed_before_delete']} -> Deleted: {crud_res['delete_test']['deleted_count']} -> Post-delete Exists: {crud_res['delete_test']['doc_exists_after_delete']}")
    print("=" * 70)


if __name__ == "__main__":
    client = None
    try:
        client, db = get_mongo_client()
        client.admin.command("ping")
        run_all_queries(db)
    except Exception as e:
        print(f"Error executing queries: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if client:
            client.close()
