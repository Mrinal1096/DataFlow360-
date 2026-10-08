"""
DataFlow360 — NASA Battery Dataset Ingestion
=============================================
Purpose : Downloads the NASA Li-Ion Battery Aging Dataset and saves
          raw discharge cycle data as Parquet files in data/raw/.

Dataset : NASA PCoE Battery Data Set
          4 Li-Ion 18650 cells: B0005, B0006, B0007, B0018
          Each cell cycled hundreds of times until end-of-life.

What this script does:
  1. Downloads the NASA battery zip file
  2. Extracts the .mat files
  3. Parses each battery's discharge cycles
  4. Calculates State of Health (SOH) per cycle
  5. Saves structured data as Parquet files

Run with:
    python -m src.ingestion.battery_ingestion
"""

import os
import zipfile

import numpy as np
import pandas as pd
import requests
import scipy.io as sio
from dotenv import load_dotenv
from pathlib import Path

from src.utils.logger import get_logger

# Load environment variables from .env
load_dotenv()

# Get our named logger — all messages from this file show "ingestion.battery_ingestion"
logger = get_logger(__name__)


# ==============================================================================
# Constants — values that don't change during a run
# ==============================================================================

# Which battery cells to ingest from the NASA dataset
BATTERY_IDS = ["B0005", "B0006", "B0007", "B0018"]

# A fresh Li-Ion 18650 cell has a rated capacity of 2.0 Amp-hours
RATED_CAPACITY_AH = 2.0

# A battery is considered "end-of-life" when SOH drops below 80%
# This is the industry-standard threshold
EOL_SOH_THRESHOLD = 80.0

# Where raw data is stored (never modified after saving)
RAW_DATA_DIR = Path("data/raw")

# The zip file will be saved here
ZIP_PATH = RAW_DATA_DIR / "nasa_battery_dataset.zip"

# Download URL — can be overridden via .env if needed
NASA_BATTERY_URL = os.getenv(
    "NASA_BATTERY_URL",
    "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip",
)


# ==============================================================================
# STEP 1 — Download
# ==============================================================================

def download_dataset(url: str, save_path: Path) -> bool:
    """
    Downloads the NASA battery dataset zip file with progress logging.

    If the file already exists locally, download is skipped.
    This is called 'idempotent' behaviour — safe to run multiple times.

    Args:
        url      : URL to download the zip from
        save_path: Where to save the file on disk

    Returns:
        True if file is available (downloaded or already existed)
        False if download failed
    """
    # Skip download if file already exists
    if save_path.exists():
        logger.info(f"Dataset already exists locally: {save_path} — skipping download")
        return True

    logger.info(f"Starting download from: {url}")
    logger.info("This may take a few minutes depending on your connection speed...")

    try:
        # stream=True means we download in chunks instead of all at once
        # This prevents running out of memory for large files
        response = requests.get(url, stream=True, timeout=180)

        # raise_for_status() throws an error if server returned 404, 500, etc.
        response.raise_for_status()

        total_bytes = int(response.headers.get("content-length", 0))
        downloaded_bytes = 0
        chunk_size = 8192  # Download 8KB at a time

        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                file.write(chunk)
                downloaded_bytes += len(chunk)

                # Log progress every 10MB so the user knows it's still running
                if total_bytes > 0 and downloaded_bytes % (10 * 1024 * 1024) < chunk_size:
                    percent = (downloaded_bytes / total_bytes) * 100
                    mb_done = downloaded_bytes / 1024 / 1024
                    logger.info(f"  Downloading... {percent:.1f}% ({mb_done:.0f} MB)")

        final_size_mb = downloaded_bytes / 1024 / 1024
        logger.info(f"Download complete: {save_path} ({final_size_mb:.1f} MB)")
        return True

    except requests.exceptions.ConnectionError:
        logger.error("Download failed: No internet connection or server unreachable.")
        logger.error("Please check your internet connection and try again.")
        return False
    except requests.exceptions.Timeout:
        logger.error("Download timed out after 180 seconds.")
        logger.error("Try again, or download manually from: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/")
        return False
    except requests.exceptions.HTTPError as error:
        logger.error(f"Server returned an error: {error}")
        return False
    except OSError as error:
        logger.error(f"Could not write file to disk: {error}")
        return False


# ==============================================================================
# STEP 2 — Extract
# ==============================================================================

def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    """
    Extracts the downloaded zip file into the raw data directory.

    The NASA dataset is a zip-of-zips: the outer zip contains several
    inner zip files, each holding the actual .mat files for different
    battery cells. This function extracts the outer zip first, then
    finds and extracts all inner zips automatically.

    Args:
        zip_path  : Path to the downloaded zip file
        extract_to: Directory to extract files into

    Returns:
        True if extraction succeeded, False otherwise
    """
    logger.info(f"Extracting: {zip_path.name}")

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info(f"Outer zip extracted — files saved to: {extract_to}")
    except zipfile.BadZipFile:
        logger.error("The downloaded file appears to be corrupted.")
        logger.error(f"Please delete '{zip_path}' and run the pipeline again.")
        return False
    except Exception as error:
        logger.error(f"Unexpected error during extraction: {error}")
        return False

    # The NASA dataset is a zip-of-zips — find and extract all inner zips
    # Exclude the outer zip itself (zip_path) to avoid re-processing it
    inner_zips = [z for z in extract_to.rglob("*.zip") if z.resolve() != zip_path.resolve()]
    if inner_zips:
        logger.info(f"Found {len(inner_zips)} inner zip(s) — extracting each...")
        for inner_zip in sorted(inner_zips):
            logger.info(f"  Extracting inner zip: {inner_zip.name}")
            try:
                with zipfile.ZipFile(inner_zip, "r") as zip_ref:
                    zip_ref.extractall(inner_zip.parent)
                logger.info(f"  Done: {inner_zip.name}")
            except zipfile.BadZipFile:
                logger.warning(f"  Skipping corrupted inner zip: {inner_zip.name}")
            except Exception as error:
                logger.warning(f"  Could not extract {inner_zip.name}: {error}")

    logger.info("All extraction complete.")
    return True


# ==============================================================================
# STEP 3 — Parse
# ==============================================================================

def calculate_soh(capacity_ah: float, rated_capacity_ah: float) -> float:
    """
    Calculates State of Health (SOH) as a percentage.

    SOH tells us how healthy a battery is compared to when it was new.

    Formula:
        SOH (%) = (Current Capacity / Rated Capacity) x 100

    Examples:
        Fresh battery : SOH = (2.0 / 2.0) x 100 = 100%
        Degraded      : SOH = (1.7 / 2.0) x 100 = 85%
        End of life   : SOH = (1.6 / 2.0) x 100 = 80%  ← threshold

    Args:
        capacity_ah      : Measured discharge capacity (Amp-hours)
        rated_capacity_ah: Nominal capacity when battery was new

    Returns:
        SOH as a float percentage (e.g., 85.3)
    """
    if rated_capacity_ah <= 0 or np.isnan(capacity_ah):
        return np.nan
    return round((capacity_ah / rated_capacity_ah) * 100, 2)


def parse_battery_mat_file(mat_path: Path, battery_id: str) -> pd.DataFrame:
    """
    Parses a NASA battery .mat file and extracts discharge cycle data.

    The NASA .mat file stores data in a deeply nested structure.
    We use scipy.io.loadmat() with simplify_cells=True to flatten it.

    After flattening, the structure looks like:
        mat[battery_id]['cycle'][i]['type']           → 'charge' or 'discharge'
        mat[battery_id]['cycle'][i]['data']['Capacity'] → float (Amp-hours)
        mat[battery_id]['cycle'][i]['data']['Voltage_measured'] → array

    We only extract DISCHARGE cycles because:
        - Discharge cycles contain capacity measurements (needed for SOH)
        - Charge cycles don't tell us about battery health directly

    Args:
        mat_path  : Path to the .mat file
        battery_id: Battery name (e.g., 'B0005')

    Returns:
        DataFrame with one row per discharge cycle
    """
    logger.info(f"  Parsing {mat_path.name} for battery {battery_id}...")

    # Load the .mat file — simplify_cells=True flattens MATLAB's nested arrays
    try:
        mat = sio.loadmat(str(mat_path), simplify_cells=True)
    except Exception as error:
        logger.error(f"  Cannot read .mat file {mat_path.name}: {error}")
        raise

    # Get the battery's data block
    battery_data = mat.get(battery_id)
    if battery_data is None:
        raise KeyError(f"Battery '{battery_id}' not found in {mat_path.name}")

    cycles = battery_data["cycle"]
    records = []
    discharge_count = 0

    for raw_cycle_index, cycle in enumerate(cycles):
        # Each cycle is either 'charge', 'discharge', or 'impedance'
        cycle_type = str(cycle.get("type", "")).strip()

        if cycle_type != "discharge":
            continue  # Skip non-discharge cycles

        discharge_count += 1
        data = cycle.get("data", {})

        try:
            # Capacity: total charge delivered during this discharge (Amp-hours)
            capacity = float(data.get("Capacity", np.nan))

            # Temperature: average across the full discharge
            temp_array = data.get("Temperature_measured", [])
            avg_temp = float(np.mean(temp_array)) if len(temp_array) > 0 else np.nan

            # Voltage: end-of-discharge voltage (low voltage = more degraded)
            voltage_array = data.get("Voltage_measured", [])
            end_voltage = float(voltage_array[-1]) if len(voltage_array) > 0 else np.nan

            # Ambient temperature set by the lab
            ambient_temp = float(cycle.get("ambient_temperature", np.nan))

            # Calculate State of Health for this cycle
            soh = calculate_soh(capacity, RATED_CAPACITY_AH)

            records.append({
                "battery_id":            battery_id,
                "raw_cycle_index":       raw_cycle_index + 1,
                "discharge_cycle":       discharge_count,
                "capacity_ah":           round(capacity, 6),
                "rated_capacity_ah":     RATED_CAPACITY_AH,
                "soh_pct":               soh,
                "end_voltage_v":         round(end_voltage, 4),
                "avg_temperature_c":     round(avg_temp, 4),
                "ambient_temperature_c": round(ambient_temp, 2),
                "is_end_of_life":        soh < EOL_SOH_THRESHOLD,
            })

        except (TypeError, ValueError, IndexError) as error:
            logger.warning(f"  Skipping cycle {raw_cycle_index+1}: {error}")
            continue

    df = pd.DataFrame(records)

    if df.empty:
        logger.warning(f"  No discharge cycles found for {battery_id}")
    else:
        logger.info(
            f"  {battery_id}: {len(df)} discharge cycles | "
            f"SOH: {df['soh_pct'].max():.1f}% → {df['soh_pct'].min():.1f}%"
        )

    return df


# ==============================================================================
# STEP 4 — Save
# ==============================================================================

def save_raw_data(df: pd.DataFrame, battery_id: str) -> Path:
    """
    Saves parsed battery data as a Parquet file in data/raw/.

    Why Parquet instead of CSV?
    - Parquet is a columnar format — much faster to read specific columns
    - Automatically compressed — smaller file size
    - Preserves data types — no risk of pandas reading numbers as strings
    - Industry standard for Data Engineering pipelines

    The raw data is NEVER modified. If we need to transform it,
    we create a new file in data/processed/ (Phase 3).

    Args:
        df        : Parsed battery cycle DataFrame
        battery_id: Battery name — used in the filename

    Returns:
        Path to the saved file
    """
    output_path = RAW_DATA_DIR / f"{battery_id}_raw_cycles.parquet"
    df.to_parquet(output_path, index=False)
    size_kb = output_path.stat().st_size / 1024
    logger.info(f"  Saved: {output_path} ({size_kb:.1f} KB, {len(df)} rows)")
    return output_path


# ==============================================================================
# Main Orchestrator
# ==============================================================================

def run_ingestion() -> None:
    """
    Orchestrates the full ingestion pipeline:
        Download → Extract → Parse → Save

    Each step is logged and errors are handled gracefully.
    If one battery fails, others continue processing.
    """
    logger.info("=" * 60)
    logger.info("DataFlow360 — Battery Ingestion Pipeline Started")
    logger.info("=" * 60)

    # --- Download ---
    if not download_dataset(NASA_BATTERY_URL, ZIP_PATH):
        logger.error("Pipeline stopped: download failed.")
        return

    # --- Extract ---
    if not extract_zip(ZIP_PATH, RAW_DATA_DIR):
        logger.error("Pipeline stopped: extraction failed.")
        return

    # --- Parse and Save each battery ---
    successful_batteries = []
    failed_batteries = []

    for battery_id in BATTERY_IDS:
        # Search recursively — the zip may extract into a subfolder
        mat_files = list(RAW_DATA_DIR.rglob(f"{battery_id}.mat"))

        if not mat_files:
            logger.warning(f"No .mat file found for {battery_id} — skipping")
            failed_batteries.append(battery_id)
            continue

        mat_path = mat_files[0]
        logger.info(f"Processing {battery_id} from: {mat_path}")

        try:
            df = parse_battery_mat_file(mat_path, battery_id)
            if not df.empty:
                save_raw_data(df, battery_id)
                successful_batteries.append(battery_id)
            else:
                failed_batteries.append(battery_id)
        except Exception as error:
            logger.error(f"Failed to process {battery_id}: {error}")
            failed_batteries.append(battery_id)

    # --- Final Summary ---
    logger.info("=" * 60)
    logger.info("Ingestion Pipeline Complete")
    logger.info(f"  Successful : {len(successful_batteries)} batteries — {', '.join(successful_batteries) or 'none'}")
    if failed_batteries:
        logger.warning(f"  Failed     : {len(failed_batteries)} batteries — {', '.join(failed_batteries)}")
    logger.info("=" * 60)


# Allow running directly: python -m src.ingestion.battery_ingestion
if __name__ == "__main__":
    run_ingestion()
