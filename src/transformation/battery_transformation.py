"""
DataFlow360 — Battery Data Transformation
==========================================
Purpose : Loads raw discharge cycle parquets, cleans anomalies,
          engineers features, and writes a processed dataset ready
          for validation and ML.

What this script does:
  1. Load    — reads all battery parquets from data/raw/
  2. Combine — merges them into one DataFrame
  3. Clean   — caps SOH at 100% (sensor over-reads), drops nulls
  4. Engineer— adds derived features useful for ML & dashboards
  5. Save    — writes data/processed/battery_cycles_processed.parquet

Feature Engineering:
  capacity_fade_pct     : % capacity lost since the battery's first cycle
  soh_capped_pct        : SOH clamped to [0, 100] — removes sensor artifacts
  rolling_soh_7         : 7-cycle centred rolling average of SOH (smoothed trend)
  soh_delta             : cycle-over-cycle SOH change (negative = degrading)
  cycles_to_eol         : cycles remaining until EOL (SOH < 80%) — NaN after EOL
  is_near_eol           : True when SOH < 85% (early warning flag)
  life_fraction         : discharge_cycle / total_cycles for this battery (0–1)

Run with:
    python -m src.transformation.battery_transformation
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ==============================================================================
# Constants
# ==============================================================================

RAW_DATA_DIR       = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

BATTERY_IDS          = ["B0005", "B0006", "B0007", "B0018"]
RATED_CAPACITY_AH    = 2.0
EOL_SOH_THRESHOLD    = 80.0   # industry-standard end-of-life
NEAR_EOL_THRESHOLD   = 85.0   # early-warning flag
ROLLING_WINDOW       = 7      # cycles for rolling average


# ==============================================================================
# STEP 1 — Load
# ==============================================================================

def load_raw_data() -> pd.DataFrame:
    """
    Loads all battery raw parquet files and concatenates them into one DataFrame.

    Returns:
        Combined DataFrame with all batteries' discharge cycles.

    Raises:
        FileNotFoundError: If no parquet files are found in data/raw/.
    """
    logger.info("Loading raw parquet files...")
    frames = []

    for battery_id in BATTERY_IDS:
        path = RAW_DATA_DIR / f"{battery_id}_raw_cycles.parquet"
        if not path.exists():
            logger.warning(f"  Missing file for {battery_id}: {path} — skipping")
            continue
        df = pd.read_parquet(path)
        frames.append(df)
        logger.info(f"  Loaded {battery_id}: {len(df)} rows from {path.name}")

    if not frames:
        raise FileNotFoundError(
            "No raw parquet files found in data/raw/. "
            "Run the ingestion pipeline first:\n"
            "  python -m src.ingestion.battery_ingestion"
        )

    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"Combined dataset: {len(combined)} total rows across {len(frames)} batteries")
    return combined


# ==============================================================================
# STEP 2 — Clean
# ==============================================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies data quality fixes to the raw combined DataFrame.

    Issues addressed:
      - SOH > 100%: B0006 starts at 101.8% — a sensor over-read on the first
        few cycles. We cap SOH at 100.0 (the physical maximum for a new battery).
      - Negative capacity: physically impossible — rows dropped.
      - Null capacity or SOH: rows dropped (can't compute health metrics).

    Args:
        df: Raw combined DataFrame.

    Returns:
        Cleaned DataFrame.
    """
    initial_rows = len(df)
    logger.info(f"Cleaning data ({initial_rows} rows)...")

    # Drop rows where capacity or SOH is null — can't do anything with them
    df = df.dropna(subset=["capacity_ah", "soh_pct"])
    dropped_null = initial_rows - len(df)
    if dropped_null:
        logger.warning(f"  Dropped {dropped_null} rows with null capacity/SOH")

    # Drop physically impossible negative capacity
    neg_mask = df["capacity_ah"] < 0
    if neg_mask.any():
        logger.warning(f"  Dropped {neg_mask.sum()} rows with negative capacity")
        df = df[~neg_mask]

    # Cap SOH at 100% — sensor over-reads are above physical max
    over_100 = (df["soh_pct"] > 100.0).sum()
    if over_100:
        logger.info(
            f"  Capping {over_100} SOH readings above 100% "
            f"(max was {df['soh_pct'].max():.2f}%) — likely sensor artifact"
        )
    df["soh_pct"] = df["soh_pct"].clip(upper=100.0)

    logger.info(f"  Clean complete: {len(df)} rows remaining")
    return df.reset_index(drop=True)


# ==============================================================================
# STEP 3 — Feature Engineering
# ==============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds derived features to the cleaned DataFrame.

    All features are computed per-battery (grouped by battery_id) so that
    rolling windows and relative metrics don't bleed across different cells.

    Features added:
        capacity_fade_pct  : % capacity lost relative to cycle 1 of this battery
        soh_capped_pct     : alias for the already-capped soh_pct (explicit name)
        rolling_soh_7      : centred 7-cycle rolling mean of SOH (smoothed trend)
        soh_delta          : cycle-over-cycle change in SOH (negative = degrading)
        is_near_eol        : True when SOH < 85% (early warning)
        life_fraction      : discharge_cycle / max discharge cycle (0.0 → 1.0)
        cycles_to_eol      : estimated cycles until SOH drops below 80%

    Args:
        df: Cleaned DataFrame (output of clean_data).

    Returns:
        DataFrame with new feature columns appended.
    """
    logger.info("Engineering features...")
    result_frames = []

    for battery_id, group in df.groupby("battery_id", sort=False):
        g = group.sort_values("discharge_cycle").copy()

        # --- capacity_fade_pct ---
        # How much capacity has been lost since the very first discharge?
        first_capacity = g["capacity_ah"].iloc[0]
        g["capacity_fade_pct"] = round(
            ((first_capacity - g["capacity_ah"]) / first_capacity) * 100, 4
        )

        # --- soh_capped_pct ---
        # Rename the already-capped value to make its meaning explicit in downstream code
        g["soh_capped_pct"] = g["soh_pct"]

        # --- rolling_soh_7 ---
        # Smoothed SOH trend — 7-cycle centred rolling mean
        # min_periods=1 ensures we don't get NaN at the edges
        g["rolling_soh_7"] = (
            g["soh_pct"]
            .rolling(window=ROLLING_WINDOW, center=True, min_periods=1)
            .mean()
            .round(4)
        )

        # --- soh_delta ---
        # How much did SOH change from the previous cycle?
        # Positive = recovered (rare), negative = degraded
        g["soh_delta"] = g["soh_pct"].diff().round(4)

        # --- is_near_eol ---
        # Early-warning flag: True when within 5% SOH of end-of-life
        g["is_near_eol"] = g["soh_pct"] < NEAR_EOL_THRESHOLD

        # --- life_fraction ---
        # Where in the battery's life is this cycle? 0.0 = new, 1.0 = last cycle
        max_cycle = g["discharge_cycle"].max()
        g["life_fraction"] = (g["discharge_cycle"] / max_cycle).round(4)

        # --- cycles_to_eol ---
        # How many cycles left until end-of-life (SOH < 80%)?
        # Once EOL is reached this becomes 0; before EOL, count forward to first EOL cycle.
        eol_cycles = g.loc[g["soh_pct"] < EOL_SOH_THRESHOLD, "discharge_cycle"]
        if eol_cycles.empty:
            # Battery never reached EOL in the dataset
            g["cycles_to_eol"] = np.nan
        else:
            first_eol_cycle = eol_cycles.iloc[0]
            g["cycles_to_eol"] = (first_eol_cycle - g["discharge_cycle"]).clip(lower=0)

        logger.info(
            f"  {battery_id}: fade={g['capacity_fade_pct'].max():.1f}% | "
            f"near-EOL at cycle {g.loc[g['is_near_eol'], 'discharge_cycle'].min() if g['is_near_eol'].any() else 'N/A'}"
        )

        result_frames.append(g)

    engineered = pd.concat(result_frames, ignore_index=True)
    logger.info(f"Feature engineering complete — {len(engineered.columns)} columns total")
    return engineered


# ==============================================================================
# STEP 4 — Save
# ==============================================================================

def save_processed_data(df: pd.DataFrame) -> Path:
    """
    Saves the fully processed DataFrame as a Parquet file in data/processed/.

    The processed file is the single source of truth for downstream steps:
    validation, database loading, and the dashboard.

    Args:
        df: Engineered DataFrame.

    Returns:
        Path to the saved file.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / "battery_cycles_processed.parquet"
    df.to_parquet(output_path, index=False)
    size_kb = output_path.stat().st_size / 1024
    logger.info(f"Saved processed data: {output_path} ({size_kb:.1f} KB, {len(df)} rows)")
    return output_path


# ==============================================================================
# Main Orchestrator
# ==============================================================================

def run_transformation() -> pd.DataFrame:
    """
    Orchestrates the full transformation pipeline:
        Load → Combine → Clean → Engineer Features → Save

    Returns:
        The final processed DataFrame (also saved to disk).
    """
    logger.info("=" * 60)
    logger.info("DataFlow360 — Transformation Pipeline Started")
    logger.info("=" * 60)

    df_raw        = load_raw_data()
    df_clean      = clean_data(df_raw)
    df_engineered = engineer_features(df_clean)
    output_path   = save_processed_data(df_engineered)

    # --- Summary ---
    logger.info("=" * 60)
    logger.info("Transformation Pipeline Complete")
    logger.info(f"  Output : {output_path}")
    logger.info(f"  Rows   : {len(df_engineered)}")
    logger.info(f"  Columns: {list(df_engineered.columns)}")
    logger.info(f"  SOH range (all batteries): "
                f"{df_engineered['soh_pct'].min():.1f}% → {df_engineered['soh_pct'].max():.1f}%")
    logger.info("=" * 60)

    return df_engineered


# Allow running directly: python -m src.transformation.battery_transformation
if __name__ == "__main__":
    run_transformation()
