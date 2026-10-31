# FootballIQ: Academic Project Technical Report
**Football Analytics & Player Intelligence System**  
*Degree Program: B.Tech Computer Science and Engineering*  
*Specialization: Data Engineering & Applied Machine Learning*

---

## 1. Introduction

In modern football (soccer), scouting, squad recruitment, and tactical decision-making have shifted from subjective visual assessment to objective, data-driven intelligence. Top clubs evaluate players not merely by conventional metrics (e.g., goals and assists), but through multidimensional underlying indices such as Expected Goals ($xG$), Expected Assists ($xA$), passing accuracy under pressure, duel win rates, and standardized per-90 rates.

However, accessing and integrating these disparate data sources presents significant technical challenges. Leading public repositories often specialize in isolated domains (e.g., Understat for $xG$ models, SofaScore for granular defensive and passing statistics, FBref for match summaries, and EA Sports FC for scouted valuations). 

**FootballIQ** addresses these challenges by implementing an integrated, end-to-end player intelligence platform. The system combines:
1. A multi-source dataset covering the top five European domestic leagues across two seasons (2024/25 and 2025/26).
2. A document-oriented database using **MongoDB** and **PyMongo** for high-throughput ingestion, schema flexibility, and aggregation pipelines.
3. A leakage-free data preprocessing and feature engineering workflow.
4. Machine Learning models implemented in **Scikit-learn** for continuous player rating prediction (Regression) and elite performer classification.

---

## 2. Project Objectives

The primary objectives of FootballIQ are:
1. **Data Ingestion & Integrity**: Ingest, validate, and clean 5,693 player-season records across 220 attributes without discarding valuable tactical dimensions or corrupting data types.
2. **Document Database Modeling**: Design and implement four MongoDB collections (`player_season_stats`, `teams`, `leagues`, `ml_results`) supporting compound primary keys, indexes, and bulk write operations.
3. **Advanced Analytics Queries**: Implement a suite of PyMongo queries demonstrating CRUD operations, comparison/logical operators, multi-stage aggregation pipelines (`$match`, `$group`, `$sort`, `$project`), and relational document joins (`$lookup`, `$unwind`).
4. **Leakage-Free Machine Learning Workflow**: Train and benchmark four model families (Linear Regression, Decision Tree, Random Forest, Logistic Regression) on real performance targets while strictly preventing target and temporal data leakage.
5. **Reproducibility & Serialization**: Serialize trained models, preprocessors, evaluation summaries, and predictions to disk and database.

---

## 3. Dataset Description & Provenance

### 3.1 Overview
The system utilizes a curated master dataset compiled from four prominent football repositories:
- **Location**: `data/top5-football-dataset-master/data/master/player_seasons.csv`
- **Total Records**: 5,693 player-seasons
- **Unique Athletes**: 3,591 distinct players
- **Total Feature Attributes**: 220 columns
- **Temporal Scope**: Seasons 2024/25 (`2425`) and 2025/26 (`2526`)
- **Domestic Leagues**:
  1. English Premier League (`ENG-Premier League`)
  2. Spanish La Liga (`ESP-La Liga`)
  3. German Bundesliga (`GER-Bundesliga`)
  4. Italian Serie A (`ITA-Serie A`)
  5. French Ligue 1 (`FRA-Ligue 1`)

### 3.2 Data Lineage and Column Blocks
Columns in the dataset carry explicit source prefixes to ensure provenance:
- **Identity & Playing Time** (No prefix): `league`, `season`, `season_label`, `team`, `player`, `nation`, `pos`, `age`, `playing_time_min`, `playing_time_mp`.
- **FBref Basics (`fb_`)**: Traditional counting statistics including goals (`fb_gls`), assists (`fb_ast`), shots (`fb_sh`), shots on target (`fb_sot`), cards (`fb_crdy`, `fb_crdr`).
- **Understat Advanced Metrics (`us_`)**: Mathematical models including Expected Goals (`us_xg`), Non-Penalty Expected Goals (`us_np_xg`), Expected Assists (`us_xa`), $xG$ Chain (`us_xg_chain`), and $xG$ Buildup (`us_xg_buildup`).
- **SofaScore Detailed Tactical Aggregates (`ss_`)**: Granular action data including ground duels won percentage (`ss_groundDuelsWonPercentage`), aerial duels won percentage (`ss_aerialDuelsWonPercentage`), touches (`ss_touches`), clearances (`ss_clearances`), interceptions (`ss_interceptions`), key passes (`ss_keyPasses`), and average match rating (`ss_rating`).
- **EA FC / SoFIFA Ratings (`fifa_`)**: Scouted market values (`fifa_value_eur`), wages, contract expiration dates, and physical ratings (pace, sprint speed, strength).
- **Normalized Rates (`*_p90`)**: Per-90 minute conversions for fair cross-player comparisons regardless of playing time differences.

### 3.3 Data Quality & Primary Key Verification
A structural integrity audit verified that the compound tuple:
$$\text{Primary Key} = (\text{league}, \text{season}, \text{team}, \text{player})$$
yields exactly **0 duplicate records** across the entire 5,693-row table.

---

## 4. MongoDB & PyMongo Implementation

### 4.1 Database Architecture
The database is named `footballiq` and contains four dedicated collections:
1. `player_season_stats`: Stores all 5,693 player-season documents preserving all 220 attributes.
2. `teams`: Aggregated club-level documents (192 records) summarizing squad size, total goals, assists, cumulative $xG$, average squad age, total market valuation, and top goal scorer.
3. `leagues`: League-level documents (5 records) detailing participating clubs, player pools, goal production rates, and average player age.
4. `ml_results`: Audit collection storing timestamps, model configurations, evaluation metrics, and feature importance rankings.

### 4.2 Data Sanitization & Idempotent Ingestion
To ensure compatibility between Pandas and MongoDB BSON serialization, a recursive sanitization pipeline (`src/load_data.py`) converts:
- `np.nan`, `pd.NA`, `inf`, and `-inf` $\rightarrow$ `None` (mapped to BSON `null`).
- NumPy integer types (`np.int64`, `np.int32`) $\rightarrow$ native Python `int`.
- NumPy floating point types (`np.float64`) $\rightarrow$ native Python `float`.
- NumPy boolean types (`np.bool_`) $\rightarrow$ native Python `bool`.

A deterministic primary key is synthesized for every document:
$$\_id = \text{player}\_\_\text{team}\_\_\text{season}\_\_\text{league}$$

Ingestion utilizes PyMongo's `bulk_write` API with `ReplaceOne({"_id": doc["_id"]}, doc, upsert=True)`. This guarantees idempotency: re-running ingestion does not generate duplicate documents or throw primary key collision exceptions.

### 4.3 Database Index Optimization
To support low-latency querying, the following B-tree indexes were constructed:
- **`player_season_stats`**:
  - Single-field: `player`, `team`, `league`, `season`, `fb_gls` (DESC), `fb_ast` (DESC), `ss_rating` (DESC), `fifa_value_eur` (DESC).
  - Compound: `(league, season)`, `(player, season)`, `(team, season)`.
- **`teams`**: `team`, `league`, `season`, `(league, season)`.
- **`leagues`**: `league_name`, `country_code`.
- **`ml_results`**: `task`, `model_name`, `created_at` (DESC).

---

## 5. MongoDB Practice & Analytics Query Suite

The query module (`src/queries.py`) demonstrates complete mastery of MongoDB operations:

### 5.1 Query Operations Implemented
- **Basic Retrieval**: `find()` and `find_one()` using projections to restrict network transfer to necessary fields.
- **Sorting & Limiting**: Ranked leaderboards for goals and assists using `.sort([("field", -1)]).limit(N)`.
- **Comparison Operators**: Filtering by playing time (`$gte: 900`), age thresholds (`$lte: 21`), and metric cutoffs (`$gt: 0`).
- **Logical Operators**: Compound scouting filters combining `$and` with `$or` conditions (e.g., high rating or high $xG$).
- **Membership Operators**: Inclusion and exclusion queries using `$in` (target leagues) and `$nin` (excluding goalkeepers).
- **CRUD Life-Cycle**: Safe demonstration of `update_one` with `$set` and `$currentDate`, verification of modified state, field removal via `$unset`, and temporary document insertion and deletion (`insert_one`, `delete_one`).

### 5.2 Multi-Stage Aggregation Pipelines
1. **Goals Per 90 Minutes**:
   - `$match`: `playing_time_min >= 900` and `fb_gls > 0`.
   - `$project`: Calculates $\text{goals\_p90} = \text{round}((\text{fb\_gls} / \text{playing\_time\_min}) \times 90.0, 2)$.
   - `$sort` & `$limit`: Extracts the highest-efficiency goalscorers.
2. **League Comparative Breakdown**:
   - `$group`: Groups by `$league`, computing total player-seasons, total goals, total assists, average player age, average match rating, and cumulative market valuation.
   - `$project`: Standardizes ratios such as average goals per player.
3. **Cross-Collection Relational Join (`$lookup` & `$unwind`)**:
   - Matches players with $\ge 12$ goals in `player_season_stats`.
   - Performs a `$lookup` into `teams` matching both `team` and `season`.
   - Deconstructs the array using `$unwind`.
   - Computes each star player's exact percentage contribution to their club's total league goals.

---

## 6. Data Preprocessing & Leakage Prevention

### 6.1 Sample Filtering
Raw sports datasets contain fringe squad members with cameo appearances (e.g., 5 minutes played). Including such records distorts analytical per-90 rates and introduces severe statistical noise. FootballIQ filters records to players with $\ge 450$ minutes played (~5 full 90-minute matches), yielding **3,838 high-integrity player-seasons**.

### 6.2 Feature Space Selection
Thirty input features were selected across five analytical domains:
1. **Attacking Output**: `fb_gls`, `fb_ast`, `fb_sh`, `fb_sot`, `us_xg`, `us_xa`, `us_key_passes`, `np_xg_p90`, `xa_p90`.
2. **Possession & Playmaking**: `ss_accuratePassesPercentage`, `ss_keyPasses`, `ss_touches`, `ss_successfulDribblesPercentage`, `ss_possessionLost`, `ss_bigChancesCreated`.
3. **Defensive Rigor & Duels**: `ss_groundDuelsWonPercentage`, `ss_aerialDuelsWonPercentage`, `ss_interceptions`, `ss_clearances`, `tackles_won_p90`, `interceptions_p90`.
4. **Physical & Volume Attributes**: `age`, `playing_time_min`.
5. **Categorical Context**: `pos_group` (One-hot encoded: FW, MF, DF, GK), `league` (One-hot encoded).

### 6.3 Strict Target Leakage Shielding
In predictive modeling, target leakage occurs when features contain direct mathematical derivatives of the target variable. In SofaScore's schema:
$$\text{ss\_rating} = \frac{\text{ss\_totalRating}}{\text{ss\_countRating}}$$
Using `ss_totalRating` or `ss_countRating` as input would result in trivial, 100% artificial accuracy. The preprocessor explicitly excludes these columns, ensuring models learn genuine underlying physical, technical, and tactical relationships.

### 6.4 Leak-Free Scaling
To prevent train-to-test data leakage:
- Imputation medians and `StandardScaler` transformations are **fitted strictly on the 80% training partition**.
- The 20% test partition is transformed using the parameters learned from the training set.

---

## 7. Machine Learning Methodology & Evaluation

Two distinct machine learning tasks were formulated and evaluated:

### 7.1 Regression Task
- **Target**: `ss_rating` (Continuous overall match rating, range: 5.70 to 8.16).
- **Candidate Models**:
  1. Ordinary Least Squares Linear Regression
  2. Decision Tree Regressor (`max_depth=5`)
  3. Random Forest Regressor (`n_estimators=100`, `max_depth=10`, `random_state=42`)
- **Evaluation Metrics**: Coefficient of Determination ($R^2$), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).

### 7.2 Classification Task
- **Target**: `is_high_performer` (Binary: 1 if `ss_rating >= 7.0`, 0 otherwise). Class balance: 22.5% positive.
- **Candidate Models**:
  1. Logistic Regression (`max_iter=1000`, `random_state=42`)
  2. Decision Tree Classifier (`max_depth=5`)
  3. Random Forest Classifier (`n_estimators=100`, `max_depth=10`, `random_state=42`)
- **Evaluation Metrics**: Accuracy, Precision, Recall, and F1 Score (evaluated on the positive minority class).

---

## 8. Actual Execution Results

All results reported below were generated directly by executing the FootballIQ pipeline on the dataset:

### 8.1 Database Verification Counts
- `player_season_stats`: **5,693** documents
- `teams`: **192** documents
- `leagues`: **5** documents
- `ml_results`: Logged run records

### 8.2 Top Statistical Leaders Identified
- **Top Goalscorer**: Harry Kane (Bayern Munich, Bundesliga) — 36 goals, $xG$ 29.58.
- **Top Assist Provider**: Bruno Fernandes (Manchester United, Premier League) — 21 assists, $xA$ 17.76.
- **Goals Per 90 Leader** ($\ge 900$ mins): Harry Kane (1.36 G/90), Alexander Sørloth (1.15 G/90), Patrik Schick (1.12 G/90).
- **Top Young Talent (U-21)**: Lamine Yamal (Barcelona, Age 18) — 16 goals, 11 assists, Rating 7.93, Valuation €147M.
- **Highest Team Goal Share**: Kylian Mbappé scored 31 of Real Madrid's 78 league goals (**39.7%** team goal share).

### 8.3 Machine Learning Benchmark

#### Regression Model Comparison (Target: `ss_rating`)
| Model | $R^2$ Score | MAE | RMSE |
|---|:---:|:---:|:---:|
| **Linear Regression (Best)** | **0.6530** | **0.0984** | **0.1316** |
| Random Forest Regressor | 0.6430 | 0.1040 | 0.1335 |
| Decision Tree Regressor | 0.4838 | 0.1257 | 0.1605 |

*Analysis*: Linear Regression achieved the strongest performance with an $R^2$ of **0.6530** and a very low MAE of **0.0984** on a 10-point scale. This indicates that player match ratings are strongly and linearly driven by aggregate touch volume, duels won, and goal involvement.

#### Classification Model Comparison (Target: `is_high_performer >= 7.0`)
| Model | Accuracy | Precision | Recall | F1 Score |
|---|:---:|:---:|:---:|:---:|
| **Logistic Regression (Best)** | **85.29%** | **71.85%** | **56.40%** | **0.6319** |
| Random Forest Classifier | 84.77% | 70.07% | 55.81% | 0.6214 |
| Decision Tree Classifier | 82.55% | 64.62% | 48.84% | 0.5563 |

*Analysis*: Logistic Regression achieved an overall classification accuracy of **85.29%** and the highest F1 score (**0.6319**), successfully separating elite performers from average squad contributors with high precision (**71.85%**).

#### Feature Importance Analysis
Top positive regression coefficients:
1. `ss_touches` (+0.1557) — High ball involvement and team centrality.
2. `fb_gls` (+0.1117) — Direct goal output.
3. `pos_group_GK` (+0.1030) — Baseline position rating compensation.
4. `playing_time_min` (+0.0931) — Starting status and managerial confidence.
5. `ss_groundDuelsWonPercentage` (+0.0466) — Physical ball retention and winning 50-50 duels.
6. `us_xg` (+0.0393) — Accumulation of high-probability scoring chances.
7. `xa_p90` (+0.0357) — Standardized playmaking rate.
8. `ss_keyPasses` (+0.0297) — Shot-creating opportunities created for teammates.

---

## 9. Conclusion

The FootballIQ project successfully developed, deployed, and verified an end-to-end player analytics and intelligence system. By integrating data engineering with applied machine learning:
1. Ingested 5,693 multi-source records into MongoDB using idempotent bulk upserts.
2. Created high-performance compound indexes and executed queries spanning filtering, relational aggregation, and CRUD life-cycles.
3. Designed a leakage-free preprocessing pipeline and trained four standard model families.
4. Achieved an $R^2$ of **0.6530** (Linear Regression) and an F1 score of **0.6319** (Logistic Regression) on genuine European top-5 league data.

The modular structure, configuration-driven connection layer, and separation of concerns provide a production-ready template for sports analytics applications.

---

## 10. Future Scope & Indian Football Vision

While the current release focuses on Europe's top-5 domestic leagues, the long-term vision of FootballIQ is to provide analytical intelligence to **Indian football**:

1. **Indian Super League (ISL) Ingestion**: Integrate event-level match data from ISL clubs (e.g., Mohun Bagan SG, Mumbai City FC, Kerala Blasters) to analyze domestic player performance and tactical progression.
2. **Grassroots & Youth Scouting (AIFF)**: Ingest match reports and tracking data from the Reliance Foundation Development League (RFDL) and Santosh Trophy to identify emerging U-19 and U-21 Indian talent.
3. **Foreign Recruit Profiling**: Assist ISL clubs in scouting international players by evaluating how their European or Asian league metrics map to Indian pitch conditions.
4. **Spatial Analytics**: Incorporate $x, y$ coordinate data to generate automated shot maps and pass-origin heatmaps.

*(Note: Indian football support is part of the future development vision and has not been claimed as active in the current European data release).*

