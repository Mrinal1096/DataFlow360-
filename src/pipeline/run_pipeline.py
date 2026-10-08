"""
DataFlow360 — Main Pipeline Orchestrator
=========================================
Purpose : Runs all phases of the data pipeline sequentially:
          Ingestion -> Transformation -> Validation -> Database

          If any phase fails, the pipeline halts to prevent bad data
          from propagating downstream.

Run with:
    python -m src.pipeline.run_pipeline
"""

import sys
import time

from src.utils.logger import get_logger

# Import the runner from each module
from src.ingestion.battery_ingestion import run_ingestion
from src.transformation.battery_transformation import run_transformation
from src.validation.battery_validation import run_validation
from src.database.battery_database import run_database_load

logger = get_logger(__name__)

def main():
    logger.info("*" * 60)
    logger.info("DataFlow360 — Full Pipeline Execution Started")
    logger.info("*" * 60)
    
    start_time = time.time()

    try:
        # ---------------------------------------------------------
        # Phase 1: Ingestion
        # ---------------------------------------------------------
        logger.info("\n--- Starting Phase 1: Ingestion ---")
        run_ingestion()
        
        # ---------------------------------------------------------
        # Phase 2: Transformation
        # ---------------------------------------------------------
        logger.info("\n--- Starting Phase 2: Transformation ---")
        # Transformation returns the DataFrame, but we don't strictly need it in memory here
        df_processed = run_transformation()
        if df_processed is None or df_processed.empty:
            logger.error("Transformation returned empty data. Halting pipeline.")
            sys.exit(1)

        # ---------------------------------------------------------
        # Phase 3: Validation
        # ---------------------------------------------------------
        logger.info("\n--- Starting Phase 3: Validation ---")
        is_valid = run_validation()
        if not is_valid:
            logger.error("Validation failed. Halting pipeline to protect database.")
            sys.exit(1)

        # ---------------------------------------------------------
        # Phase 4: Database Loading
        # ---------------------------------------------------------
        logger.info("\n--- Starting Phase 4: Database Loading ---")
        db_success = run_database_load()
        if not db_success:
            logger.error("Database loading failed. Halting pipeline.")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Pipeline crashed with an unexpected error: {e}", exc_info=True)
        sys.exit(1)

    # ---------------------------------------------------------
    # Completion
    # ---------------------------------------------------------
    elapsed = time.time() - start_time
    logger.info("\n" + "*" * 60)
    logger.info(f"DataFlow360 — Full Pipeline Completed Successfully in {elapsed:.1f}s")
    logger.info("*" * 60)

if __name__ == "__main__":
    main()
