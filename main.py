# import sqlite3
# from pathlib import Path;

# db_path = Path("data/database.sqlite")
# if not db_path.exists():
#     print("Database not FOUND!")
#     raise SystemExit(1)

# connection = sqlite3.connect(str(db_path))

# tables = connection.execute("""
#     SELECT name
#     FROM sqlite_master
#     WHERE type = 'table'
#     ORDER BY name;
# """).fetchall()

# print("DATABASE is connected ^ ")

# for(table_name,)in tables:
#     count = connection.execute(
#         f'SELECT COUNT(*) from "{table_name}"'
#     ).fetchone()[0]

#     columns = connection.execute(
#         f'PRAGMA table_info("{table_name}")'
#     ).fetchall()

#     print(f"\n Table: {table_name}")
#     print(f" Rows: {count}")
#     print(f"columns: ",[column[1] for column in columns])

# connection.close()

# import sqlite3
# import pandas as pd
# from pathlib import Path

# # Path to our downloaded football database
# DB_PATH = Path("data/database.sqlite")

# # Connect to SQLite
# connection = sqlite3.connect(DB_PATH)

# # Extract 5 actual player attribute records
# query = """
# SELECT
#     player_api_id,
#     player_fifa_api_id,
#     date,
#     overall_rating,
#     potential,
#     preferred_foot,
#     finishing,
#     short_passing,
#     acceleration,
#     sprint_speed
# FROM Player_Attributes
# LIMIT 5;
# """

# df = pd.read_sql_query(query, connection)

# connection.close()

# # Display the records
# print("\n===== ACTUAL PLAYER RECORDS =====")
# print(df.to_string(index=False))
# import sqlite3
# import pandas as pd
# from pathlib import Path

# # Path to our downloaded football database
# DB_PATH = Path("data/database.sqlite")

# # Connect to SQLite
# connection = sqlite3.connect(DB_PATH)

# # Extract 5 actual player attribute records
# query = """
# SELECT
#     player_api_id,
#     player_fifa_api_id,
#     date,
#     overall_rating,
#     potential,
#     preferred_foot,
#     finishing,
#     short_passing,
#     acceleration,
#     sprint_speed
# FROM Player_Attributes
# LIMIT 5;
# """

# df = pd.read_sql_query(query, connection)

# connection.close()

# # Display the records
# print("\n===== ACTUAL PLAYER RECORDS =====")
# print(df.to_string(index=False))

# print("\n===== DATASET INFORMATION =====")
# print("Number of records:", len(df))
# print("Columns:", df.columns.tolist())

# print("\n===== MISSING VALUES =====")
# print(df.isnull().sum())


# print("\n===== DATASET INFORMATION =====")
# print("Number of records:", len(df))
# print("Columns:", df.columns.tolist())

# print("\n===== MISSING VALUES =====")
# print(df.isnull().sum())


import pandas as pd
from pathlib import Path

CSV_PATH = Path(
    "data/top5-football-dataset-master/data/master/player_seasons.csv"
)

df = pd.read_csv(CSV_PATH, low_memory=False)

print("\n===== DATASET OVERVIEW =====")
print("Total records:", len(df))
print("Total columns:", len(df.columns))

print("\n===== COLUMN NAMES =====")
for column in df.columns:
    print(column)

print("\n===== SAMPLE PLAYER RECORDS =====")
columns_to_show = [
    col for col in [
        "season", "season_label", "team", "player",
        "nation", "pos", "age", "minutes",
        "fb_Gls", "fb_Ast", "fb_Sh", "fb_SoT",
        "us_xG", "us_xA"
    ]
    if col in df.columns
]

print(df[columns_to_show].head(10).to_string(index=False))

print("\n===== MISSING VALUES: TOP 15 =====")
print(df.isnull().sum().sort_values(ascending=False).head(15))

print("\n===== DUPLICATE ROWS =====")
print(df.duplicated().sum())
