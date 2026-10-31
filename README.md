# FootballIQ — Football Analytics & Player Intelligence System

FootballIQ is a modular, high-performance football analytics and player intelligence platform built with **Python**, **MongoDB (PyMongo)**, and **Scikit-learn**. The system ingests, models, and analyzes multi-source performance data from Europe's top-5 domestic leagues (Premier League, La Liga, Serie A, Bundesliga, and Ligue 1) across the 2024/25 and 2025/26 seasons.

FootballIQ bridges the gap between raw match event aggregates and scouting intelligence, supporting advanced MongoDB query pipelines, positional benchmarking, and machine learning models for match performance rating prediction and high-impact performer classification.

---

## Table of Contents
1. [Project Objectives](#project-objectives)
2. [Key Features](#key-features)
3. [System Architecture](#system-architecture)
4. [Technology Stack](#technology-stack)
5. [Dataset Description & Origin](#dataset-description--origin)
6. [Installation & Setup](#installation--setup)
7. [MongoDB Configuration](#mongodb-configuration)
8. [Execution Guide](#execution-guide)
9. [MongoDB Analytics & Query Implementations](#mongodb-analytics--query-implementations)
10. [Machine Learning Methodology & Evaluation](#machine-learning-methodology--evaluation)
11. [Actual Evaluation Results](#actual-evaluation-results)
12. [Future Scope & Indian Football Roadmap](#future-scope--indian-football-roadmap)
13. [Academic & Project Integrity Notes](#academic--project-integrity-notes)

---

## Project Objectives

1. **Multi-Source Data Ingestion**: Clean, validate, and store comprehensive player-season records combining traditional counting stats, expected metrics ($xG$, $xA$), passing accuracy, and physical estimates.
2. **Document-Oriented Database Modeling**: Leverage MongoDB and PyMongo to store structured and semi-structured football data with bulk operations and query optimization.
3. **Advanced Scouting Analytics**: Provide queries utilizing comparison, logical, aggregation, and relational (`$lookup`, `$unwind`) operators.
4. **Leakage-Free Machine Learning**: Design a machine learning workflow that predicts player match performance ratings (`ss_rating`) and classifies high-impact performers without target leakage.
5. **Academic Rigor**: Implement modular, explainable code suitable for a B.Tech CSE capstone/major project.

---

## Key Features

- **Idempotent Data Ingestion**: Bulk upserts (`bulk_write` with `ReplaceOne(upsert=True)`) ensuring re-runs do not create duplicate records.
- **Relational Joins in Document Stores**: MongoDB Aggregation `$lookup` and `$unwind` joining player statistics with team-level summary documents to compute metrics such as player goal contribution share.
- **Scouting Pipelines**: Automated identification of young talents (U-21) outperforming their expected goals ($xG$) or match rating benchmarks.
- **Dual ML Paradigms**:
  - **Regression**: Predict continuous match performance ratings (`ss_rating`) using Linear Regression, Decision Tree Regressor, and Random Forest Regressor.
  - **Classification**: Identify elite/high-impact performers (rating $\ge$ 7.0) using Logistic Regression, Decision Tree Classifier, and Random Forest Classifier.
- **Artifact Serialization**: Scalers and models persisted via `joblib`, test predictions saved to CSV, and evaluation logs recorded directly in MongoDB.

---

## System Architecture

```
FootballIQ Architecture
├── Data Layer
│   ├── Raw Master CSV (player_seasons.csv: 5,693 records x 220 columns)
│   └── Multi-Source Harmonization (FBref + Understat + SofaScore + EA FC)
├── Database Layer (MongoDB / PyMongo)
│   ├── Database: footballiq
│   ├── Collections:
│   │   ├── player_season_stats (5,693 documents, 10 indexes)
│   │   ├── teams (192 documents, 4 indexes)
│   │   ├── leagues (5 documents, 2 indexes)
│   │   └── ml_results (Run metadata, evaluation metrics, feature weights)
├── Analytics & Query Engine (src/queries.py)
│   ├── CRUD Operations (find, find_one, update_one, delete_one)
│   ├── Aggregation Pipelines ($match, $group, $sort, $project, $lookup, $unwind)
│   └── Scouting Intelligence (Per-90, Under-21 filters, League comparisons)
├── Machine Learning Pipeline (src/preprocess.py, src/train_model.py)
│   ├── Stratified 80/20 Train/Test Split
│   ├── Median Imputation & Position Grouping
│   ├── Target Leakage Shielding (Excludes ss_totalRating, ss_countRating)
│   ├── StandardScaler Normalization
│   └── Comparative Model Evaluation (Regression & Classification)
└── Orchestration (main.py)
    └── CLI Flag Controller (--validate, --connect, --import-data, --queries, --preprocess, --train, --all)
```

---

## Technology Stack

- **Core Runtime**: Python 3.10+ (Tested on Python 3.14)
- **Database Engine**: MongoDB Community Server 8.x / MongoDB Atlas
- **Database Driver**: PyMongo 4.x
- **Data Engineering**: Pandas 3.x, NumPy 2.x
- **Machine Learning**: Scikit-learn 1.9+, Joblib
- **Data Visualization**: Matplotlib

---

## Dataset Description & Origin

The primary dataset is located at:
`data/top5-football-dataset-master/data/master/player_seasons.csv`

### Dataset Specifications:
- **Total Records**: 5,693 player-seasons
- **Unique Players**: 3,591 individual athletes
- **Total Columns**: 220 attributes
- **Leagues Represented**:
  - `ENG-Premier League` (England)
  - `ESP-La Liga` (Spain)
  - `FRA-Ligue 1` (France)
  - `GER-Bundesliga` (Germany)
  - `ITA-Serie A` (Italy)
- **Seasons**: 2024/25 (`2425`) and 2025/26 (`2526`)
- **Primary Key**: `(league, season, team, player)` — Verified with **0 duplicate rows**.

### Data Provenance Blocks:
| Prefix | Source | Description |
|---|---|---|
| *(None)* | Identity & Metadata | League, season, team, player, nation, position, age, minutes |
| `fb_` | FBref / Sports Reference | Goals, assists, shots, shots on target, cards |
| `us_` | Understat | Expected goals ($xG$), non-penalty $xG$, expected assists ($xA$), $xG$ chain, $xG$ buildup |
| `ss_` | SofaScore | Duels (aerial/ground/total), accurate passes, touches, tackles, clearances, match rating (`ss_rating`) |
| `fifa_` | SoFIFA / EA FC | Market valuation (`fifa_value_eur`), contract end, pace, physical scouted attributes |
| `*_p90` | Derived | Per-90 standardized rates |

---

## Installation & Setup

### 1. Clone & Navigate to Repository
```bash
git clone <repository_url>
cd Football
```

### 2. Set Up Python Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## MongoDB Configuration

FootballIQ supports both **local MongoDB installations** and **MongoDB Atlas** cloud clusters.

### Environment Configuration (`.env`)
Create or edit `.env` in the root folder:

```env
# Local MongoDB connection (Default)
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DATABASE=footballiq

# Alternatively, MongoDB Atlas connection:
# MONGODB_URI=mongodb+srv://<username>:<password>@cluster0.example.mongodb.net/?retryWrites=true&w=majority
# MONGODB_DATABASE=footballiq
```

### Starting Local MongoDB:
- **macOS (Homebrew)**:
  ```bash
  brew services start mongodb-community
  ```
- **Linux (systemd)**:
  ```bash
  sudo systemctl start mongod
  ```
- **Manual Binary**:
  ```bash
  mongod --dbpath /path/to/data/db
  ```

---

## Execution Guide

The entire system is orchestrated through `main.py` with flexible CLI flags:

### Run Complete End-to-End Pipeline
```bash
python main.py --all
```
*(Executes validation, database connection, ingestion, queries, preprocessing, and ML training).*

### Run Individual Stages:
- **Validate Dataset Quality**:
  ```bash
  python main.py --validate
  ```
- **Test Database Connection**:
  ```bash
  python main.py --connect
  ```
- **Ingest CSV Data into MongoDB**:
  ```bash
  python main.py --import-data
  ```
- **Execute MongoDB Analytics Queries**:
  ```bash
  python main.py --queries
  ```
- **Run Feature Preprocessing & Scaling**:
  ```bash
  python main.py --preprocess
  ```
- **Train and Evaluate ML Models**:
  ```bash
  python main.py --train
  ```

---

## MongoDB Analytics & Query Implementations

All queries are implemented in `src/queries.py` and satisfy the practical database requirements:

1. **`find()` and `find_one()` with Projections**:
   - `find_player_profile(db, player_name)` searches player records across seasons using regex and projects relevant metrics.
2. **Sorting and Limiting**:
   - `get_top_goal_scorers()` and `get_top_assist_providers()` sort descending by `fb_gls` and `fb_ast` with limits.
3. **Comparison Operators (`$gte`, `$lte`, `$gt`)**:
   - Filter players with minimum playing time (`playing_time_min >= 900`) or maximum age (`age <= 21`).
4. **Logical Operators (`$and`, `$or`)**:
   - `get_young_talents_scouting()` filters players where `age <= 21` **AND** (`ss_rating >= 6.8` **OR** `us_xg >= 2.0`).
5. **Membership Operators (`$in`, `$nin`)**:
   - `filter_by_leagues_and_positions()` queries across specific leagues while excluding non-applicable positions (e.g. Goalkeepers).
6. **Aggregation Pipeline (`$group`, `$project`, `$sort`)**:
   - Calculates league-wide goal totals, goals per player, average age, and average match ratings.
7. **Relational Aggregation (`$lookup` & `$unwind`)**:
   - Joins `player_season_stats` with `teams` to compute an individual player's contribution percentage to their club's total league goals.
8. **CRUD Life-Cycle Demonstration**:
   - Demonstrates `update_one` with `$set` and `$currentDate`, cleans up via `$unset`, and tests `insert_one` followed by `delete_one`.

---

## Machine Learning Methodology & Evaluation

### 1. Problem Formulation
- **Regression Task**: Predict player match performance rating (`ss_rating`, continuous float between ~5.7 and 8.2) based on underlying technical, passing, physical, and defensive metrics.
- **Classification Task**: Classify whether a player is an "Elite/High-Impact Performer" (`is_high_performer = 1` if `ss_rating >= 7.0`, 0 otherwise).

### 2. Feature Selection & Engineering
- **Attacking**: `fb_gls`, `fb_ast`, `fb_sh`, `fb_sot`, `us_xg`, `us_xa`, `us_key_passes`, `np_xg_p90`, `xa_p90`
- **Possession & Passing**: `ss_accuratePassesPercentage`, `ss_keyPasses`, `ss_touches`, `ss_successfulDribblesPercentage`, `ss_possessionLost`, `ss_bigChancesCreated`
- **Defensive & Duels**: `ss_groundDuelsWonPercentage`, `ss_aerialDuelsWonPercentage`, `ss_interceptions`, `ss_clearances`, `tackles_won_p90`, `interceptions_p90`
- **Contextual & Physical**: `age`, `playing_time_min`
- **Categorical (One-Hot Encoded)**: `pos_group` (FW, MF, DF, GK), `league`

### 3. Leakage Prevention
To guarantee academic integrity and prevent synthetic overfitting:
- `ss_totalRating` and `ss_countRating` are **strictly excluded**, as `ss_rating = ss_totalRating / ss_countRating`.
- Imputation medians and `StandardScaler` transformations are **fitted strictly on the 80% training set**, and transformed on the 20% test set.

---

## Actual Evaluation Results

Results generated directly from project execution:

### Dataset Scope
- Eligible Player-Seasons ($\ge$ 450 minutes played): **3,838**
- Training Samples: **3,070** (80%)
- Testing Samples: **768** (20%)
- Class Distribution: **22.5%** Positive (`ss_rating >= 7.0`)

### Regression Performance (Target: `ss_rating`)
| Model | $R^2$ Score | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) |
|---|:---:|:---:|:---:|
| **Linear Regression (Best)** | **0.6530** | **0.0984** | **0.1316** |
| Random Forest Regressor | 0.6430 | 0.1040 | 0.1335 |
| Decision Tree Regressor | 0.4838 | 0.1257 | 0.1605 |

### Classification Performance (Target: `is_high_performer` $\ge$ 7.0)
| Model | Accuracy | Precision | Recall | F1 Score |
|---|:---:|:---:|:---:|:---:|
| **Logistic Regression (Best)** | **85.29%** | **71.85%** | **56.40%** | **0.6319** |
| Random Forest Classifier | 84.77% | 70.07% | 55.81% | 0.6214 |
| Decision Tree Classifier | 82.55% | 64.62% | 48.84% | 0.5563 |

### Top Predictive Feature Importances
From the best regression model:
1. `ss_touches` (+0.1557) — High match involvement and volume of touches
2. `fb_gls` (+0.1117) — Direct goal output
3. `pos_group_GK` (+0.1030) — Baseline position rating offset
4. `playing_time_min` (+0.0931) — Consistent starting selection
5. `ss_groundDuelsWonPercentage` (+0.0466) — Physical duel dominance
6. `us_xg` (+0.0393) — High-quality chance generation
7. `xa_p90` (+0.0357) — Sustained playmaking rate per 90
8. `ss_keyPasses` (+0.0297) — Direct shot-creating passes

---

## Future Scope & Indian Football Roadmap

While FootballIQ currently models Europe's top-5 domestic leagues (England, Spain, Germany, Italy, France), its architecture is designed for domain adaptation:

- **Indian Super League (ISL) Ingestion**: Developing dedicated scrapers and ingestion pipelines for ISL and I-League match event data.
- **AIFF Grassroots Scouting**: Integrating domestic youth tournaments (Reliance Foundation Development League, Santosh Trophy) to identify emerging Indian talent.
- **Shot-Map & Pass-Network Integration**: Utilizing event-level coordinate data for spatial creation maps.
- **Transfer Value Estimation**: Adapting valuation algorithms to Indian domestic transfer and salary cap structures.

*(Note: Indian football integration is part of the future development roadmap and is not claimed as currently implemented).*

---

## Academic & Project Integrity Notes

- **No Mock Data**: All statistics and rankings originate from the genuine 5,693-row dataset.
- **Credentials Security**: No hardcoded credentials; database URIs are loaded via `.env`.
- **Reproducibility**: All random states are seeded (`random_state=42`), models and preprocessors are serialized, and predictions are stored in `output/ml_test_predictions.csv`.
