"""LEBI Project - Phase 1: Web Scraping

Scrape job listings from HelloWork website.

Usage:
    python run_scraping.py
"""
from src.scraping.hellowork_scraper import scrape_listings
from src.utils.config import get_logger

logger = get_logger("run_scraping")

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("PHASE 1: WEB SCRAPING")
    logger.info("=" * 60)
    
    # Configure scraping parameters
    start_url = "https://www.hellowork.com/fr-fr/emploi/recherche.html"
    max_pages = 5  # Adjust as needed
    
    try:
        logger.info("Starting scraping from: %s", start_url)
        results = scrape_listings(start_url, pages=max_pages, save_csv=True)
        logger.info("✓ Scraping completed: %d jobs extracted", len(results))
        logger.info("✓ Raw data saved to data/raw/hellowork_final_sectors_data.csv")
    except Exception as e:
        logger.error("✗ Scraping failed: %s", e)
        raise
    
    logger.info("=" * 60)
    logger.info("PHASE 1 COMPLETED")
    logger.info("=" * 60)
