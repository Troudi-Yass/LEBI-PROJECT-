# LEBI Project – Job Offers Analysis Pipeline

## 🎯 Overview

The **LEBI Project** is a complete end-to-end data pipeline for scraping, cleaning, enriching, and visualizing job offers from HelloWork. It demonstrates professional data engineering and machine learning workflows.

### 🔄 Pipeline Phases

1. **Phase 1: Web Scraping** – Extract job data from HelloWork
2. **Phase 2: ETL** – Clean, standardize, and normalize data
3. **Phase 3: Machine Learning** – Cluster jobs and predict salary categories
4. **Phase 4: Dashboard** – Interactive visualization with Dash

---

## 📁 Project Structure

```
LEBI PROJECT/
├── data/                          # Data directory (gitignored)
│   ├── raw/                       # Phase 1 output: raw scraped data
│   ├── processed/                 # Phase 2 output: cleaned data
│   └── enriched/                  # Phase 3 output: ML-enriched data
│
├── src/                           # Source code (modular architecture)
│   ├── scraping/
│   │   └── hellowork_scraper.py   # Web scraper
│   ├── etl/
│   │   └── data_cleaning.py       # ETL pipeline
│   ├── ml/
│   │   ├── vectorization.py       # TF-IDF vectorization
│   │   ├── clustering.py          # KMeans clustering
│   │   └── classification.py      # Salary prediction
│   ├── dashboard/
│   │   └── app.py                 # Dash web application
│   └── utils/
│       └── config.py              # Configuration & logging
│
├── run_scraping.py                # Run Phase 1
├── run_etl.py                     # Run Phase 2
├── run_ml.py                      # Run Phase 3
├── run_dashboard.py               # Run Phase 4
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate it (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Pipeline

**Option A: Run All Phases Sequentially**

```bash
# Phase 1: Scraping (optional if data already exists)
python run_scraping.py

# Phase 2: ETL (required)
python run_etl.py

# Phase 3: Machine Learning (required)
python run_ml.py

# Phase 4: Dashboard (view results)
python run_dashboard.py
```

**Option B: Skip Scraping (Use Existing Data)**

If you already have `hellowork_final_sectors_data.csv` in the `data/raw/` folder:

```bash
python run_etl.py
python run_ml.py
python run_dashboard.py
```

### 3. View Dashboard

Open your browser to: **http://127.0.0.1:8050/**

---

## 📊 Data Flow

```
Raw CSV (data/raw/)
    ↓
[Phase 2: ETL]
    ↓
Cleaned CSV (data/processed/)
    ↓
[Phase 3: ML]
    ↓
Enriched CSV (data/enriched/)
    ↓
[Phase 4: Dashboard]
    ↓
Interactive Visualizations
```

---

## 🛠️ Key Features

### Phase 1: Web Scraping
- Requests + BeautifulSoup for fast parsing
- Selenium fallback for JavaScript-heavy pages
- Robust error handling and logging

### Phase 2: ETL
- Duplicate removal
- Salary normalization (handles ranges, units, hourly/yearly)
- Missing data handling
- TF-IDF keyword extraction
- Categorical encoding

### Phase 3: Machine Learning
- **Clustering**: KMeans with auto K-selection (silhouette score)
- **Classification**: Logistic Regression for salary prediction
- AUC scores and evaluation metrics

### Phase 4: Dashboard
- Interactive filters (sector, location, contract type, cluster, salary)
- Real-time visualizations:
  - Job distribution by sector
  - Salary distribution histogram
  - Cluster scatter plot
  - Top companies bar chart

---

## 📦 Module Documentation

### `src.utils.config`
Central configuration hub with:
- File paths (RAW_CSV, CLEAN_CSV, ENRICHED_CSV)
- Logging factory (`get_logger()`)
- Directory management (`ensure_dirs()`)

### `src.scraping.hellowork_scraper`
Web scraper with:
- `scrape_listings()` – Main scraping function
- `fetch_page()` – HTTP/Selenium page fetching
- Dynamic content detection

### `src.etl.data_cleaning`
ETL pipeline with:
- `load_raw()` – Load and standardize columns
- `clean_duplicates()` – Remove duplicates
- `normalize_salary()` – Parse salary strings
- `prepare_clean()` – Full pipeline execution

### `src.ml.vectorization`
- `build_tfidf_matrix()` – Create TF-IDF vectors from text

### `src.ml.clustering`
- `find_optimal_k()` – Auto-select K using silhouette score
- `apply_kmeans()` – Cluster jobs by description similarity

### `src.ml.classification`
- `prepare_labels()` – Create binary salary labels
- `train_logistic()` – Train classifier and return metrics

### `src.dashboard.app`
- `create_app()` – Build Dash layout and callbacks
- `run()` – Launch server

---

## 🎓 Educational Notes

> **⚠️ Note Pédagogique:** This scraping was done for educational purposes only, without intensive automation or data resale. The project demonstrates ETL, ML, and visualization concepts in an educational context.

---

## 📝 Requirements

- Python 3.8+
- See `requirements.txt` for full dependencies

---

## 🐛 Troubleshooting

**Issue: "Raw CSV not found"**
- Ensure `hellowork_final_sectors_data.csv` exists in `data/raw/`
- Or run `python run_scraping.py` to generate it

**Issue: "Module not found"**
- Activate virtual environment: `.venv\Scripts\activate`
- Install dependencies: `pip install -r requirements.txt`

**Issue: Dashboard won't start**
- Ensure Phase 2 and 3 completed successfully
- Check that enriched CSV exists in `data/enriched/`

---

## 👥 Contributors

LEBI Project Team

---

## 📄 License

Educational Project - Not for commercial use
