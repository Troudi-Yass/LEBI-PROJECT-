"""LEBI Project - Phase 2: ETL (Extract, Transform, Load)

Clean and prepare raw data for machine learning.

Usage:
    python run_etl.py
"""
from src.etl.data_cleaning import prepare_clean
from src.utils.config import RAW_CSV, CLEAN_CSV, get_logger

logger = get_logger("run_etl")

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("PHASE 2: ETL - DATA CLEANING")
    logger.info("=" * 60)
    
    try:
        logger.info("Loading raw data from: %s", RAW_CSV)
        df = prepare_clean(path_in=RAW_CSV, path_out=CLEAN_CSV)
        
        logger.info("✓ ETL completed successfully")
        logger.info("  - Input:  %s", RAW_CSV)
        logger.info("  - Output: %s", CLEAN_CSV)
        logger.info("  - Rows:   %d", len(df))
        logger.info("  - Columns: %s", list(df.columns))
    except Exception as e:
        logger.error("✗ ETL failed: %s", e)
        raise
    
    logger.info("=" * 60)
    logger.info("PHASE 2 COMPLETED")
    logger.info("=" * 60)
