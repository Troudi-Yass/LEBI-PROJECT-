"""LEBI Project - Phase 4: Dashboard

Launch interactive Dash dashboard for data visualization.

Usage:
    python run_dashboard.py
    
Then open your browser to: http://127.0.0.1:8050/
"""
from src.dashboard.app import run
from src.utils.config import get_logger

logger = get_logger("run_dashboard")

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("PHASE 4: DASHBOARD")
    logger.info("=" * 60)
    
    try:
        run()
    except KeyboardInterrupt:
        logger.info("\n✓ Dashboard stopped by user.")
    except Exception as e:
        logger.error("✗ Dashboard failed: %s", e)
        raise
