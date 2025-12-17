
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# --- Configuration ---
BASE_SEARCH_URL = "https://www.hellowork.com/fr-fr/emploi/recherche.html?k=job+%C3%A9tudiant&st=relevance"
MAX_PAGES_PER_SECTOR = 10

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
    options = webdriver.ChromeOptions()
    # options.add_argument("--headless")  # Décommenter pour exécution en arrière-plan
    options.add_argument("--start-maximized")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    return driver

def handle_cookies(driver):
    """Ferme la bannière de cookies si présente."""
    try:
        cookie_btn = WebDriverWait(driver, 4).until(
            EC.element_to_be_clickable((By.ID, "hw-cc-notice-continue-without-accepting-btn"))
        )
        cookie_btn.click()
        print("Cookies handled.")
        time.sleep(1)
    except:
        print("No cookies banner found.")

def scrape_job_details(driver, url, sector_name):
    """Scrape les détails d'une offre spécifique."""
    driver.get(url)
    data = {
        "Sector": sector_name,
        "Job_Title": "N/A",
        "Company": "N/A",
        "Location": "N/A",
        "Contract": "N/A",
        "Salary": "N/A",
        "Description": "N/A",
        "Publication_Date": "N/A",
        "URL": url
    }

    try:
        WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.TAG_NAME, "h1")))

        # 1. Job Title
        try:
            data["Job_Title"] = driver.find_element(By.CSS_SELECTOR, '[data-cy="jobTitle"]').text.strip()
        except: pass

        # 2. Company
        try:
            data["Company"] = driver.find_element(By.CSS_SELECTOR, 'h1 a').text.strip()
        except: pass

        # 3. Location & Contract
        try:
            tags = driver.find_elements(By.CSS_SELECTOR, 'ul.tw-flex.tw-flex-wrap.tw-gap-3 li')
            if len(tags) > 0: data["Location"] = tags[0].text.strip()
            if len(tags) > 1: data["Contract"] = tags[1].text.strip()
        except: pass

        # 4. Salary
        try:
            data["Salary"] = driver.find_element(By.CSS_SELECTOR, '[data-cy="salary-tag-button"]').text.strip()
        except: pass

        # 5. Description
        try:
            desc = driver.find_element(By.CSS_SELECTOR, '[data-truncate-text-target="content"]').text
            data["Description"] = desc.replace("\n", " ").strip()
        except: pass

        # 6. Publication Date
        try:
            # Strategy 1: Look for text containing "il y a" or "Publié" in small grey text
            # Based on inspection: classes include 'tw-typo-xs', 'tw-text-grey-500'
            try:
                # Try finding by class combination
                date_elem = driver.find_element(By.CSS_SELECTOR, '.tw-typo-xs.tw-text-grey-500')
                date_text = date_elem.text.strip()
                # Verify it contains date-related keywords
                if any(keyword in date_text.lower() for keyword in ['il y a', 'publié', 'jour', 'heure']):
                    data["Publication_Date"] = date_text
            except:
                # Strategy 2: Search all small text elements for date keywords
                try:
                    small_texts = driver.find_elements(By.CSS_SELECTOR, '.tw-typo-xs')
                    for elem in small_texts:
                        text = elem.text.strip()
                        if any(keyword in text.lower() for keyword in ['il y a', 'publié']):
                            data["Publication_Date"] = text
                            break
                except:
                    pass
        except: pass


    except Exception as e:
        print(f"Error extracting details for {url}: {e}")

    return data

def main():
    driver = setup_driver()
    all_results = []

    try:
        driver.get(BASE_SEARCH_URL)
        handle_cookies(driver)

        for sector in SECTORS_LIST:
            s_name = sector['name']
            s_id = sector['id']
            
            print(f"\n--- START SECTOR: {s_name} (ID: {s_id}) ---")

            for page in range(1, MAX_PAGES_PER_SECTOR + 1):
                sector_url = f"{BASE_SEARCH_URL}&s={s_id}&p={page}"
                print(f"Page {page} | Sector URL: {sector_url}")
                driver.get(sector_url)

                try:
                    WebDriverWait(driver, 6).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, 'a[data-cy="offerTitle"]'))
                    )
                except TimeoutException:
                    print(f"No jobs on page {page}, skipping sector.")
                    break

                offer_elems = driver.find_elements(By.CSS_SELECTOR, 'a[data-cy="offerTitle"]')
                urls_to_visit = list(set([elem.get_attribute("href") for elem in offer_elems]))
                print(f"Found {len(urls_to_visit)} jobs.")

                for url in urls_to_visit:
                    job_data = scrape_job_details(driver, url, s_name)
                    all_results.append(job_data)
                    time.sleep(0.5)

            # Sauvegarde intermédiaire
            pd.DataFrame(all_results).to_csv("hellowork_progress.csv", index=False, encoding='utf-8-sig')

    finally:
        # Sauvegarde finale
        df = pd.DataFrame(all_results)
        final_filename = "hellowork_final_sectors_data.csv"
        df.to_csv(final_filename, index=False, encoding='utf-8-sig')
        print(f"Scraping terminé. Fichier CSV : {final_filename} | Total jobs : {len(all_results)}")
        driver.quit()

if __name__ == "__main__":
    main()
