"""Project configuration and utilities.

Provides file paths, constants and a logger factory used across modules.
"""
from pathlib import Path
import logging
import sys

# Base paths
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ENRICHED_DIR = DATA_DIR / "enriched"

# File paths
# Default raw CSV produced by Phase 1 (provided dataset)
RAW_CSV = RAW_DIR / "hellowork_final_sectors_data.csv"
# Cleaned CSV produced by Phase 2 (ETL)
CLEAN_CSV = PROCESSED_DIR / "hellowork_cleaned.csv"
# Enriched CSV produced by Phase 3 (ML)
ENRICHED_CSV = ENRICHED_DIR / "hellowork_ml_enriched.csv"

# Scraper settings
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
)
REQUESTS_TIMEOUT = 15

# Selenium settings (if used)
SELENIUM_DRIVER_PATH = None  # set to executable path if needed


def get_logger(name: str = "lebi") -> logging.Logger:
    """Create and return a configured logger.

    Args:
        name: Logger name.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def ensure_dirs() -> None:
    """Ensure expected data directories exist."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ENRICHED_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    get_logger().info("Created data directories if missing: %s %s %s", RAW_DIR, PROCESSED_DIR, ENRICHED_DIR)
