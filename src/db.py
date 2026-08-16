"""Load data from a CSV file into a SQLite database."""

import sqlite3
from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/cs-training.csv")
DB_PATH = Path("data/credit.db")


def load_data():
    """Read data from a CSV file and return it as a pandas DataFrame."""
    return pd.read_csv(DATA_PATH, index_col=0)


def create_database(df):
    """Create a SQLite database and load the DataFrame into it."""

    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql("borrowers", conn, if_exists="replace", index_label="id")

    print(f"Database created at {DB_PATH} with table 'borrowers'.")


if __name__ == "__main__":
    df = load_data()
    print(df.shape)
    create_database(df)