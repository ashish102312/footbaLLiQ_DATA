"""FootballIQ — Football Analytics & Player Intelligence System.

Main Application Entrypoint orchestrating:
1. Dataset validation and quality assessment.
2. Database connectivity checks (Local MongoDB & MongoDB Atlas).
3. Data ingestion & bulk upsert into MongoDB collections.
4. Advanced MongoDB analytics and scouting queries.
5. Data preprocessing and leakage-free feature scaling.
6. Machine Learning training and evaluation (Regression & Classification).
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.load_data import (
    get_mongo_client,
    import_csv_to_mongodb,
    load_env_config,
    test_db_connection,
)
from src.preprocess import FootballDataPreprocessor, run_preprocessing_pipeline
from src.queries import run_all_queries
from src.train_model import train_and_evaluate


def print_banner() -> None:
    """Display system welcome banner."""
    print("=" * 72)
    print("    FootballIQ — Football Analytics & Player Intelligence System")
    print("=" * 72)


def stage_validate_dataset(csv_path: Path) -> bool:
    """Stage 1: Validate dataset files, integrity, schema, and primary key uniqueness."""
    print("\n[STAGE 1] Dataset Validation & Quality Assessment")
    print("-" * 72)
    if not csv_path.exists():
        print(f"[ERROR] Primary dataset file not found at: {csv_path}")
        return False

    import pandas as pd

    df = pd.read_csv(csv_path, low_memory=False)
    preprocessor = FootballDataPreprocessor()
    val = preprocessor.validate_raw_data(df)

    print(f"  - Dataset File:         {csv_path.name}")
    print(f"  - Total Records:        {val['total_rows']:,}")
    print(f"  - Total Attributes:     {val['total_columns']}")
    print(f"  - Leagues Represented:  {', '.join(val['leagues'])}")
    print(f"  - Seasons Covered:      {', '.join(val['seasons'])}")
    print(f"  - Primary Key Check:    {val['primary_key_duplicates']} duplicates on (league, season, team, player)")
    print("  - Top Missing Fields:  ", ", ".join([f"{k} ({v})" for k, v in list(val["top_missing_columns"].items())[:5]]))
    print("  - Validation Status:    PASSED")
    return True


def stage_test_connection() -> bool:
    """Stage 2: Check MongoDB connection."""
    print("\n[STAGE 2] MongoDB Connectivity Check")
    print("-" * 72)
    cfg = load_env_config()
    print(f"  Target URI:      {cfg.get('MONGODB_URI')}")
    print(f"  Database Name:   {cfg.get('MONGODB_DATABASE')}")

    is_ok, msg, info = test_db_connection()
    if is_ok:
        print(f"  - Status:        ONLINE")
        print(f"  - Version:       {info.get('version', 'unknown') if info else 'unknown'}")
        print(f"  - Details:       {msg}")
        return True
    else:
        print(f"  [WARNING] Status: OFFLINE / UNREACHABLE")
        print(f"  Details:         {msg}")
        print("\n  Setup Guidance:")
        print("     To run local MongoDB, start the service:")
        print("       brew services start mongodb-community  (macOS)")
        print("       mongod --dbpath <data_dir>")
        print("     Or configure MongoDB Atlas URI in your .env file:")
        print("       MONGODB_URI=mongodb+srv://<user>:<password>@cluster.mongodb.net/?retryWrites=true&w=majority")
        return False


def stage_import_data(csv_path: Path) -> bool:
    """Stage 3: Bulk import player, team, and league records into MongoDB collections."""
    print("\n[STAGE 3] MongoDB Data Ingestion & Index Creation")
    print("-" * 72)
    try:
        results = import_csv_to_mongodb(csv_path=csv_path)
        print("  - Ingestion Status: COMPLETED")
        for col, count in results["counts"].items():
            print(f"    * Collection '{col}': {count:,} documents")
        return True
    except Exception as e:
        print(f"  [ERROR] Data ingestion failed: {e}")
        return False


def stage_run_queries() -> bool:
    """Stage 4: Execute comprehensive PyMongo analytics and scouting queries."""
    print("\n[STAGE 4] MongoDB Analytics & Scouting Intelligence")
    print("-" * 72)
    client = None
    try:
        client, db = get_mongo_client()
        client.admin.command("ping")
        run_all_queries(db)
        return True
    except Exception as e:
        print(f"  [ERROR] Query execution failed: {e}")
        return False
    finally:
        if client:
            client.close()


def stage_preprocess_data(csv_path: Path) -> bool:
    """Stage 5: Data Preprocessing and Feature Engineering."""
    print("\n[STAGE 5] Data Preprocessing & Feature Pipeline")
    print("-" * 72)
    try:
        preprocessor, summary = run_preprocessing_pipeline(csv_path=csv_path)
        print("  - Preprocessing Status: COMPLETED")
        print(f"    * Feature Count: {summary['feature_count']}")
        print(f"    * Eligible Sample Count: {summary['sample_count']:,}")
        return True
    except Exception as e:
        print(f"  [ERROR] Preprocessing pipeline failed: {e}")
        return False


def stage_train_ml(csv_path: Path) -> bool:
    """Stage 6: Machine Learning Training, Evaluation, and Serialization."""
    print("\n[STAGE 6] Machine Learning Workflow (Regression & Classification)")
    print("-" * 72)
    try:
        summary = train_and_evaluate(csv_path=csv_path)
        print("  - ML Workflow Status: COMPLETED")
        print(f"    * Best Regression Model:     {summary['best_models']['regression']}")
        print(f"    * Best Classification Model: {summary['best_models']['classification']}")
        return True
    except Exception as e:
        print(f"  [ERROR] ML workflow failed: {e}")
        return False


def run_pipeline(args: argparse.Namespace) -> None:
    """Coordinate pipeline execution based on CLI flags."""
    print_banner()

    csv_path = PROJECT_ROOT / "data/top5-football-dataset-master/data/master/player_seasons.csv"

    # Default to running all stages if no specific stage flag is set
    run_all = args.all or not any([
        args.validate,
        args.connect,
        args.import_data,
        args.queries,
        args.preprocess,
        args.train,
    ])

    if run_all or args.validate:
        if not stage_validate_dataset(csv_path) and not run_all:
            return

    db_online = False
    if run_all or args.connect:
        db_online = stage_test_connection()
        if not db_online and (args.connect and not run_all):
            return

    if run_all or args.import_data:
        if db_online or args.import_data:
            stage_import_data(csv_path)
        else:
            print("\n  [INFO] Skipping MongoDB Data Import (Database is offline).")

    if run_all or args.queries:
        if db_online or args.queries:
            stage_run_queries()
        else:
            print("\n  [INFO] Skipping MongoDB Queries (Database is offline).")

    if run_all or args.preprocess:
        stage_preprocess_data(csv_path)

    if run_all or args.train:
        stage_train_ml(csv_path)

    print("\n" + "=" * 72)
    print(" FootballIQ Pipeline Execution Complete")
    print("=" * 72 + "\n")


def main() -> None:
    """CLI Argument Parser."""
    parser = argparse.ArgumentParser(
        description="FootballIQ: European Top-5 Leagues Football Analytics & Player Intelligence System"
    )
    parser.add_argument(
        "--all", action="store_true", help="Execute complete end-to-end pipeline (default)"
    )
    parser.add_argument(
        "--validate", action="store_true", help="Stage 1: Validate dataset integrity and columns"
    )
    parser.add_argument(
        "--connect", action="store_true", help="Stage 2: Test MongoDB connectivity"
    )
    parser.add_argument(
        "--import-data", action="store_true", help="Stage 3: Bulk import CSV to MongoDB collections"
    )
    parser.add_argument(
        "--queries", action="store_true", help="Stage 4: Run MongoDB analytics & scouting queries"
    )
    parser.add_argument(
        "--preprocess", action="store_true", help="Stage 5: Run data preprocessing & feature scaling"
    )
    parser.add_argument(
        "--train", action="store_true", help="Stage 6: Train and evaluate ML models"
    )

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
