"""
DataFlow360 — Database Loading
===============================
Purpose : Loads the validated, processed Parquet data into a local SQLite
          database. This makes it queryable for the Streamlit dashboard
          or any downstream analytics.

What this script does:
  1. Connects to SQLite (creates data/processed/battery_data.db)
  2. Creates the 'battery_cycles' table if it doesn't exist
  3. Loads the processed Parquet file into pandas
  4. Inserts/Replaces the data into the SQLite table
  5. Verifies the row count

Run with:
    python -m src.database.battery_database
"""

import sqlite3
import pandas as pd
from pathlib import Path
from contextlib import contextmanager

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ==============================================================================
# Constants
# ==============================================================================

PROCESSED_DATA_DIR = Path("data/processed")
PROCESSED_FILE     = PROCESSED_DATA_DIR / "battery_cycles_processed.parquet"
DB_FILE            = PROCESSED_DATA_DIR / "battery_data.db"
TABLE_NAME         = "battery_cycles"

# ==============================================================================
# Database Connections
# ==============================================================================

@contextmanager
def get_db_connection():
    """Context manager for SQLite connection to ensure it closes properly."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    try:
        yield conn
    finally:
        conn.close()

# ==============================================================================
# Table Creation
# ==============================================================================

def create_table() -> None:
    """Creates the target table with appropriate schema."""
    logger.info(f"Creating table '{TABLE_NAME}' in {DB_FILE.name}...")
    
    # We define schema explicitly to match our pandas types
    create_sql = f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        battery_id TEXT NOT NULL,
        raw_cycle_index INTEGER,
        discharge_cycle INTEGER NOT NULL,
        capacity_ah REAL,
        rated_capacity_ah REAL,
        soh_pct REAL,
        end_voltage_v REAL,
        avg_temperature_c REAL,
        ambient_temperature_c REAL,
        is_end_of_life BOOLEAN,
        capacity_fade_pct REAL,
        soh_capped_pct REAL,
        rolling_soh_7 REAL,
        soh_delta REAL,
        is_near_eol BOOLEAN,
        life_fraction REAL,
        cycles_to_eol INTEGER,
        PRIMARY KEY (battery_id, discharge_cycle)
    )
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(create_sql)
        conn.commit()

# ==============================================================================
# Data Loading
# ==============================================================================

def load_data_to_db(df: pd.DataFrame) -> int:
    """
    Loads DataFrame into SQLite table using pandas.to_sql.
    Uses if_exists='replace' to overwrite existing data for idempotency.
    """
    logger.info(f"Loading {len(df)} rows into {TABLE_NAME}...")
    
    with get_db_connection() as conn:
        # We use replace here so running the pipeline multiple times doesn't duplicate data.
        # In a real warehouse, you might use 'append' or an UPSERT logic.
        df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
        
        # Verify
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
        count = cursor.fetchone()[0]
        
    logger.info(f"Successfully loaded {count} rows into database.")
    return count

# ==============================================================================
# Main Orchestrator
# ==============================================================================

def run_database_load() -> bool:
    """
    Reads the processed parquet file and loads it into SQLite.
    Returns True if successful.
    """
    logger.info("=" * 60)
    logger.info("DataFlow360 — Database Loading Started")
    logger.info("=" * 60)

    if not PROCESSED_FILE.exists():
        logger.error(
            f"Processed file not found: {PROCESSED_FILE}\n"
            "Run transformation and validation first."
        )
        return False

    df = pd.read_parquet(PROCESSED_FILE)
    
    create_table()
    rows_loaded = load_data_to_db(df)
    
    logger.info("=" * 60)
    logger.info("Database Loading Complete")
    logger.info(f"  Database : {DB_FILE}")
    logger.info(f"  Table    : {TABLE_NAME}")
    logger.info(f"  Rows     : {rows_loaded}")
    logger.info("=" * 60)
    
    return True

if __name__ == "__main__":
    run_database_load()
