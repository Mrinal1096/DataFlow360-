import sqlite3
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List

app = FastAPI(title="DataFlow360 Dashboard API")

DB_FILE = Path("data/processed/battery_data.db")
STATIC_DIR = Path("src/dashboard/static")

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

@app.get("/")
def read_root():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/api/batteries")
def get_batteries():
    """Returns a list of all battery IDs."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT battery_id FROM battery_cycles ORDER BY battery_id")
        rows = cursor.fetchall()
        return [row["battery_id"] for row in rows]

@app.get("/api/battery/{battery_id}")
def get_battery_data(battery_id: str):
    """Returns cycle data for a specific battery."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT discharge_cycle, capacity_ah, soh_pct, rolling_soh_7, capacity_fade_pct, is_near_eol, is_end_of_life
            FROM battery_cycles 
            WHERE battery_id = ? 
            ORDER BY discharge_cycle
            """, 
            (battery_id,)
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

@app.get("/api/summary")
def get_summary():
    """Returns aggregate summary metrics for all batteries."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT 
                battery_id, 
                MAX(discharge_cycle) as total_cycles,
                MIN(soh_pct) as final_soh,
                MAX(capacity_fade_pct) as max_fade
            FROM battery_cycles 
            GROUP BY battery_id
            """
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.dashboard.app:app", host="127.0.0.1", port=8000, reload=True)
