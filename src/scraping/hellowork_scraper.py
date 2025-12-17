"""Scraper for HelloWork job listings.

Sector-based scraping with Selenium for dynamic content.
Scrapes job offers across 26 professional sectors.

Example usage:
    python -m src.scraping.hellowork_scraper
"""
from typing import List, Dict, Optional
import re
import csv
import time
from urllib.parse import urljoin

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

from src.utils.config import RAW_CSV, USER_AGENT, get_logger, ensure_dirs

logger = get_logger("hellowork_scraper")

# Base search URL for student jobs
BASE_SEARCH_URL = "https://www.hellowork.com/fr-fr/emploi/recherche.html?k=job+%C3%A9tudiant&st=relevance"
MAX_PAGES_PER_SECTOR = 10

# Complete list of 26 sectors
SECTORS_LIST = [
    {"id": "Agri_peche", "name": "Agriculture • Pêche"},
    {"id": "BTP", "name": "BTP"},
    {"id": "Banq_assur_finan", "name": "Banque • Assurance • Finance"},
    {"id": "Distrib_commerce", "name": "Distribution • Commerce de gros"},
    {"id": "Enseign_forma", "name": "Enseignement • Formation"},
    {"id": "Immo", "name": "Immobilier"},
    {"id": "Ind_agro", "name": "Industrie Agro • alimentaire"},
    {"id": "Ind_auto_meca_nav", "name": "Industrie Auto • Meca • Navale"},
    {"id": "Ind_aero", "name": "Industrie Aéronautique • Aérospatial"},
    {"id": "Ind_manufact", "name": "Industrie Manufacturière"},
    {"id": "Ind_pharma_bio_chim", "name": "Industrie Pharmaceutique • Biotechn. • Chimie"},
    {"id": "Ind_petro", "name": "Industrie Pétrolière • Pétrochimie"},
    {"id": "Ind_hightech_telecom", "name": "Industrie high • tech • Telecom"},
    {"id": "Media_internet_com", "name": "Média • Internet • Communication"},
    {"id": "Resto", "name": "Restauration"},
    {"id": "Sante_social", "name": "Santé • Social • Association"},
    {"id": "Energie_envir", "name": "Secteur Energie • Environnement"},
    {"id": "Inform_SSII", "name": "Secteur informatique • ESN"},
    {"id": "Serv_public_autre", "name": "Service public autres"},
    {"id": "Serv_public_etat", "name": "Service public d'état"},
    {"id": "Serv_public_collec_terri", "name": "Service public des collectivités territoriales"},
    {"id": "Serv_public_hosp", "name": "Service public hospitalier"},
    {"id": "Serv_entreprise", "name": "Services aux Entreprises"},
    {"id": "Serv_pers_part", "name": "Services aux Personnes • Particuliers"},
    {"id": "Tourism_hotel_loisir", "name": "Tourisme • Hôtellerie • Loisirs"},
    {"id": "Transport_logist", "name": "Transport • Logistique"}
]


def setup_driver():
    """Setup Chrome WebDriver with appropriate options."""
    options = webdriver.ChromeOptions()
    # options.add_argument("--headless")  # Uncomment for background execution
    options.add_argument("--start-maximized")
    options.add_argument(f"user-agent={USER_AGENT}")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    return driver


def handle_cookies(driver):
    """Close cookies banner if present."""
    try:
        cookie_btn = WebDriverWait(driver, 4).until(
            EC.element_to_be_clickable((By.ID, "hw-cc-notice-continue-without-accepting-btn"))
        )
        cookie_btn.click()
        logger.info("Cookies handled.")
        time.sleep(1)
    except:
        logger.info("No cookies banner found.")


def scrape_job_details(driver, url: str, sector_name: str) -> Dict[str, Optional[str]]:
    """Scrape details from a specific job offer page.
    
    Args:
        driver: Selenium WebDriver instance
        url: Job offer URL
        sector_name: Name of the sector
        
    Returns:
        Dictionary with job details
    """
    driver.get(url)
    data = {
        "sector": sector_name,
        "job_title": "N/A",
        "company": "N/A",
        "location": "N/A",
        "contract_type": "N/A",
        "salary": "N/A",
        "description": "N/A",
        "publication_date": "N/A",
        "job_url": url
    }

    try:
        WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.TAG_NAME, "h1")))

        # 1. Job Title
        try:
            data["job_title"] = driver.find_element(By.CSS_SELECTOR, '[data-cy="jobTitle"]').text.strip()
        except: pass

        # 2. Company
        try:
            data["company"] = driver.find_element(By.CSS_SELECTOR, 'h1 a').text.strip()
        except: pass

        # 3. Location & Contract
        try:
            tags = driver.find_elements(By.CSS_SELECTOR, 'ul.tw-flex.tw-flex-wrap.tw-gap-3 li')
            if len(tags) > 0: data["location"] = tags[0].text.strip()
            if len(tags) > 1: data["contract_type"] = tags[1].text.strip()
        except: pass

        # 4. Salary
        try:
            data["salary"] = driver.find_element(By.CSS_SELECTOR, '[data-cy="salary-tag-button"]').text.strip()
        except: pass

        # 5. Description
        try:
            desc = driver.find_element(By.CSS_SELECTOR, '[data-truncate-text-target="content"]').text
            data["description"] = desc.replace("\n", " ").strip()
        except: pass

        # 6. Publication Date
        try:
            try:
                date_elem = driver.find_element(By.CSS_SELECTOR, '.tw-typo-xs.tw-text-grey-500')
                date_text = date_elem.text.strip()
                if date_text:
                    data["publication_date"] = date_text
            except:
                pass
        except: pass

    except Exception as e:
        logger.error("Error extracting details for %s: %s", url, e)

    return data


def scrape_sector_based(max_pages_per_sector: int = MAX_PAGES_PER_SECTOR, save_csv: bool = True) -> List[Dict]:
    """Scrape job listings across all sectors using Selenium.
    
    Args:
        max_pages_per_sector: Maximum pages to scrape per sector
        save_csv: Whether to save results to CSV
        
    Returns:
        List of job dictionaries
    """
    ensure_dirs()
    driver = setup_driver()
    all_results = []

    try:
        driver.get(BASE_SEARCH_URL)
        handle_cookies(driver)

        for sector in SECTORS_LIST:
            s_name = sector['name']
            s_id = sector['id']
            
            logger.info("=" * 60)
            logger.info("START SECTOR: %s (ID: %s)", s_name, s_id)
            logger.info("=" * 60)

            for page in range(1, max_pages_per_sector + 1):
                sector_url = f"{BASE_SEARCH_URL}&s={s_id}&p={page}"
                logger.info("Page %d | Sector URL: %s", page, sector_url)
                driver.get(sector_url)

                try:
                    WebDriverWait(driver, 6).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, 'a[data-cy="offerTitle"]'))
                    )
                except TimeoutException:
                    logger.info("No jobs on page %d, skipping rest of sector.", page)
                    break

                offer_elems = driver.find_elements(By.CSS_SELECTOR, 'a[data-cy="offerTitle"]')
                urls_to_visit = list(set([elem.get_attribute("href") for elem in offer_elems]))
                logger.info("Found %d unique jobs on this page.", len(urls_to_visit))

                for url in urls_to_visit:
                    job_data = scrape_job_details(driver, url, s_name)
                    all_results.append(job_data)
                    time.sleep(0.5)  # Politeness delay

            # Save progress after each sector
            if save_csv:
                progress_file = RAW_CSV.parent / "hellowork_progress.csv"
                pd.DataFrame(all_results).to_csv(progress_file, index=False, encoding='utf-8')
                logger.info("Progress saved: %d jobs so far", len(all_results))

    finally:
        if save_csv and all_results:
            _save_raw(all_results)
        driver.quit()
        logger.info("=" * 60)
        logger.info("SCRAPING COMPLETED: %d total jobs", len(all_results))
        logger.info("=" * 60)

    return all_results


def scrape_listings(start_url: str = BASE_SEARCH_URL, pages: int = 1, save_csv: bool = True) -> List[Dict[str, Optional[str]]]:
    """Legacy function for compatibility. Uses sector-based scraping.
    
    Args:
        start_url: Starting URL (ignored, uses BASE_SEARCH_URL)
        pages: Number of pages per sector
        save_csv: Whether to save to CSV
        
    Returns:
        List of scraped jobs
    """
    return scrape_sector_based(max_pages_per_sector=pages, save_csv=save_csv)


def _save_raw(items: List[Dict], path: Optional[str] = None) -> None:
    """Persist raw items to CSV at config.RAW_CSV or provided path."""
    ensure_dirs()
    target = RAW_CSV if path is None else path
    if not items:
        logger.warning("No items to save to %s", target)
        return
    fieldnames = [
        "job_title",
        "company",
        "location",
        "salary",
        "contract_type",
        "publication_date",
        "description",
        "job_url",
        "sector",
    ]
    try:
        with open(target, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for it in items:
                writer.writerow({k: (it.get(k) or "") for k in fieldnames})
        logger.info("Saved %d raw items to %s", len(items), target)
    except Exception as e:
        logger.error("Failed saving raw CSV: %s", e)


if __name__ == "__main__":
    # Example entrypoint: scrape first page of HelloWork offers
    start = "https://www.hellowork.com/emploi/"
    try:
        results = scrape_listings(start, pages=1)
        logger.info("Scraped %d items", len(results))
    except Exception as e:
        logger.exception("Scraping failed: %s", e)
