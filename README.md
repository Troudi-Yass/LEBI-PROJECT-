# 📊 LEBI Project – Job Market Intelligence Pipeline

<div align="center">

**End-to-End Data Pipeline for Job Market Analysis**

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Selenium](https://img.shields.io/badge/Selenium-4.39-green.svg)](https://www.selenium.dev/)
[![Dash](https://img.shields.io/badge/Dash-3.3-red.svg)](https://dash.plotly.com/)
[![scikit-learn](https://img.shields.io/badge/sklearn-1.8-orange.svg)](https://scikit-learn.org/)

</div>

---

## 🎯 Executive Summary

The **LEBI (Labor Economics Business Intelligence) Project** is a comprehensive data science pipeline that transforms raw job postings from HelloWork.com into actionable business intelligence through web scraping, ETL processing, machine learning, and interactive visualization.

### Key Achievements
- 🔍 **1,239 clean jobs** across **23 professional sectors**
- 🧹 **86.6% salary coverage** (1,073 with valid salaries)
- 🤖 **7 job topics** discovered via NMF (bigrams, 1K features)
- 📈 **0.982 AUC** salary classifier (LogReg, balanced, 2K features)
- 🎨 **Modern interactive dashboard** with 5 visualizations + KPI metrics
- 📍 **651 normalized locations** with geographic analysis
- 🏢 **175 companies** in the enriched dataset
- ⚡ **TF-IDF feature space:** 1K (clustering) / 2K (classification)

### 🔄 Pipeline Architecture

```
┌─────────────────┐
│  Phase 1: WEB   │  Selenium-based scraping
│   SCRAPING      │  → 24 sectors, ~1.4K jobs
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│   Phase 2: ETL  │  NLTK French NLP + TF-IDF
│  DATA CLEANING  │  → Salary normalization
│                 │  → Contract filtering (remove empty)
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│   Phase 3: ML   │  NMF (7 topics, 1K features, bigrams)
│  MODELING       │  → LogReg (C=10, balanced, AUC 98%)
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  Phase 4: VIZ   │  Dash + Plotly dashboard
│   DASHBOARD     │  → Real-time filtering (0-5k salary)
└─────────────────┘
```

---

## 📑 Table of Contents

- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Phase Details](#-phase-details)
  - [Phase 1: Web Scraping](#phase-1-web-scraping)
  - [Phase 2: ETL Pipeline](#phase-2-etl-pipeline)
  - [Phase 3: Machine Learning](#phase-3-machine-learning)
  - [Phase 4: Dashboard](#phase-4-dashboard)
- [Technical Stack](#-technical-stack)
- [Results & Metrics](#-results--metrics)
- [API Documentation](#-api-documentation)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)

---

## 📁 Project Structure

```
LEBI PROJECT/
│
├── 📓 Phase Notebooks (Educational Format)
│   ├── Phase1_Scrapping.ipynb      # Step-by-step web scraping guide
│   ├── Phase2_ETL.ipynb            # Data cleaning walkthrough
│   ├── Phase3_ML.ipynb             # ML pipeline with evaluation (unified)
│   └── Phase4_Dashboard.ipynb      # Dashboard creation guide
│
├── 📂 data/                        # Data directory (tracked on 'data' branch)
│   ├── raw/                        # Phase 1 output: raw scraped data
│   │   ├── hellowork_final_sectors_data.csv  (1,364 raw jobs)
│   │   └── hellowork_progress.csv           (incremental saves)
│   ├── processed/                  # Phase 2 output: cleaned data
│   │   └── hellowork_cleaned.csv            (1,239 cleaned jobs)
│   └── enriched/                   # Phase 3 output: ML-enriched data
│       ├── hellowork_ml_enriched.csv        (with 7 clusters & predictions)
│       └── hellowork_ml_summary.json        (model metrics)
│
├── 🔧 src/                         # Production source code
│   ├── scraping/
│   │   ├── __init__.py
│   │   └── hellowork_scraper.py    # Selenium-based scraper (26 sectors)
│   ├── etl/
│   │   ├── __init__.py
│   │   └── data_cleaning.py        # NLTK + TF-IDF text processing
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── vectorization.py        # TF-IDF vectorization (1K features, bigrams)
│   │   ├── clustering.py           # NMF topic modeling (7 topics)
│   │   └── classification.py       # LogisticRegression (C=10, balanced)
│   ├── dashboard/
│   │   ├── __init__.py
│   │   └── app.py                  # Dash application with 5 charts + KPIs
│   └── utils/
│       ├── __init__.py
│       └── config.py               # Centralized configuration
│
├── 🚀 Execution Scripts
│   ├── run_scraping.py             # Execute Phase 1
│   ├── run_etl.py                  # Execute Phase 2
│   ├── run_ml.py                   # Execute Phase 3
│   └── run_dashboard.py            # Execute Phase 4 (server)
│
├── 📋 Configuration Files
│   ├── requirements.txt            # Python dependencies
│   ├── .gitignore                  # Git ignore patterns
│   └── README.md                   # This documentation
│
└── 🐍 .venv/                       # Virtual environment (gitignored)
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** (developed with Python 3.13.5)
- **Git** (for cloning and version control)
- **Chrome Browser** (for Selenium scraping)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Troudi-Yass/LEBI-PROJECT-.git
cd LEBI-PROJECT-

# 2. Switch to 'data' branch (contains data files)
git checkout data

# 3. Create virtual environment
python -m venv .venv

# 4. Activate virtual environment
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# 5. Install dependencies
pip install -r requirements.txt

# 6. Download NLTK data (required for French text processing)
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### Execution Options

#### Option A: Full Pipeline (With Scraping)

```bash
# Run all phases sequentially
python run_scraping.py    # ~30-60 min (scrapes 1,400+ jobs)
python run_etl.py          # ~10 seconds
python run_ml.py           # ~15 seconds
python run_dashboard.py    # Starts server on http://127.0.0.1:8050
```

#### Option B: Quick Start (Skip Scraping)

```bash
# Use existing data from 'data' branch
python run_etl.py          # Process existing raw data
python run_ml.py           # Train ML models
python run_dashboard.py    # Launch dashboard
```

#### Option C: Interactive Notebooks

```bash
# Open Jupyter or VS Code
jupyter notebook
# Then open and run:
# - Phase1_Scrapping.ipynb (educational walkthrough)
# - Phase2_ETL.ipynb (step-by-step ETL)
# - Phase3_ML.ipynb (ML with explanations)
# - Phase4_Dashboard.ipynb (dashboard creation)
```

### Verify Installation

```bash
# Check Python version
python --version  # Should be 3.8+

# Check packages
pip list | grep -E "selenium|dash|nltk|sklearn|pandas"

# Test imports
python -c "import selenium, dash, nltk, sklearn, pandas; print('✅ All packages OK')"
```

---

## 📚 Phase Details

### Phase 1: Web Scraping

**Objective:** Extract job postings from HelloWork.com across 26 professional sectors

**Technology Stack:**
- **Selenium 4.39** with ChromeDriver for dynamic content rendering
- **webdriver-manager** for automatic driver management
- **Explicit waits** for robust element detection

**Scraping Strategy:**
```python
# 26 professional sectors covered:
SECTORS = [
    "Agriculture • Pêche", "BTP", "Banque • Assurance • Finance",
    "Distribution • Commerce", "Enseignement • Formation", "Immobilier",
    "Industrie Agro-alimentaire", "Industrie Automobile",
    "Industrie Aéronautique", "Industrie Manufacturière",
    "Industrie Pharmaceutique", "Industrie Pétrolière",
    "Industrie High-tech", "Média • Internet",
    "Restauration", "Santé • Social",
    "Energie • Environnement", "Informatique • ESN",
    "Services publics (4 categories)", "Services aux Entreprises",
    "Services aux Particuliers", "Tourisme • Hôtellerie",
    "Transport • Logistique"
]
```

**Data Fields Extracted:**
| Field | Description | Example |
|-------|-------------|---------|
| `Sector` | Professional sector | "Informatique • ESN" |
| `Job_Title` | Position title | "Développeur Full Stack" |
| `Company` | Hiring company | "TechCorp SAS" |
| `Location` | City/Region | "Paris (75)" |
| `Contract` | Contract type | "CDI" |
| `Salary` | Raw salary string | "35-45k€/an" |
| `Description` | Full job description | "Nous recherchons..." |
| `Publication_Date` | Posting date | "Il y a 3 jours" |
| `URL` | Job posting link | "https://..." |

**Output:**
- `data/raw/hellowork_final_sectors_data.csv` (1,364 jobs)
- `data/raw/hellowork_progress.csv` (incremental backup)

**Ethical Considerations:**
- Politeness delay: 0.5-3 seconds between requests
- Respects robots.txt
- Educational purpose only (no commercial use)
- Cookie consent handling
- Error recovery with graceful degradation

---

### Phase 2: ETL Pipeline

**Objective:** Transform raw data into clean, ML-ready format with French text processing

**Processing Steps:**

1. **Data Loading & Standardization**
   - UTF-8 encoding handling
   - Column name normalization
   - Schema validation

2. **Duplicate Removal**
   - Drop exact and URL-based duplicates
   - Final: 1,364 → 1,239 unique jobs

3. **Empty Category Removal** (NEW - Phase 2 Enhancement)
   - Removes jobs with empty/missing **Sector**
   - Removes jobs with empty/missing **Location**
   - Removes jobs with empty/missing **Contract_Type**
   - Ensures filters only show valid categories
   - Impact: Dashboard displays 100% valid categories
   ```python
   # Handles relative dates in French
   "Aujourd'hui" → datetime.today()
   "Hier" → datetime.today() - 1 day
   "Il y a 3 jours" → datetime.today() - 3 days
   "Il y a 2 mois" → datetime.today() - 60 days
   ```

4. **Salary Normalization**
   - Converts all formats to **monthly EUR**
   - Handles ranges (takes average)
   - Unit conversion:
     - Hourly: multiply by 160 (hours/month)
     - Annual: divide by 12
     - "30k€" → 30,000 / 12 = 2,500€/month
   - **Result:** 1,073/1,239 (86.6%) valid salaries

5. **French Text Cleaning (NLTK)**
   ```python
   # French stopwords removed
   FRENCH_STOPWORDS = set(stopwords.words('french'))
   
   # Cleaning pipeline:
   - Lowercase conversion
   - Punctuation removal
   - Stopword filtering
   - Whitespace normalization
   ```

6. **TF-IDF Keyword Extraction**
   - **max_features=1000** terms
   - Top 10 keywords per job
   - Vocabulary: 1,000 most relevant terms

7. **Categorical Encoding**
   - Label encoding for: Sector, Location, Contract, Company
   - Creates `*_enc` columns for ML compatibility

**Output:**
- `data/processed/hellowork_cleaned.csv` (1,239 jobs, 16 columns)

**Data Quality Metrics:**
| Metric | Value |
|--------|-------|
| Total Jobs | 1,239 |
| Salary Completeness | 86.6% (1,073) |
| Text Cleaning | 100% |
| Encoded Features | 4 categories |

---

### Phase 3: Machine Learning

**Objective:** Discover job topics via clustering and predict salary categories

#### 3.1 Topic Discovery (NMF Clustering)

**Algorithm:** Non-negative Matrix Factorization (NMF)
- **Why NMF?** Produces interpretable topics (vs K-Means)
- **Parameters:**
   - `n_components=7` topics
   - `max_iter=500`
   - `max_features=1000`, `min_df=2`, `max_df=0.9`, `ngram_range=(1, 2)`
   - `random_state=42` (reproducibility)

**TF-IDF Vectorization:**
- 1,000 features, unigrams + bigrams
- Stop words handled upstream during cleaning

**Topic Assignment:**
```python
# Assign each job to dominant topic
W = nmf.fit_transform(X_text)  # Document-Topic matrix
df['job_cluster'] = W.argmax(axis=1)  # Select highest weight
```
Top terms are logged during `run_ml.py` execution for transparency.

**Cluster Distribution (current dataset, 1,239 rows):**
| Cluster | Jobs | Percentage |
|---------|------|------------|
| 0 | 209 | 16.9% |
| 1 | 149 | 12.0% |
| 2 | 210 | 16.9% |
| 3 | 103 | 8.3% |
| 4 | 404 | 32.6% |
| 5 | 70 | 5.6% |
| 6 | 94 | 7.6% |

#### 3.2 Salary Classification

**Algorithm:** Logistic Regression
- **Target:** Binary classification (high vs low salary)
- **Threshold:** Median salary (€2,116.90/month)
- **Features:** TF-IDF vectors (2,000 dimensions)

**Training Configuration:**
```python
# Stratified train/test split
X_train, X_test = train_test_split(X, y, test_size=0.2, 
                                     stratify=y, random_state=42)
LogisticRegression(max_iter=1000, C=10, class_weight='balanced')
```

**Model Performance (current run, 1,239 rows → 80/20 split):**
| Metric | Value |
|--------|-------|
| **AUC** | **0.9825** |
| Precision (high) | 1.0000 |
| Recall (high) | 0.8372 |
| F1-Score (high) | 0.9114 |
| Accuracy | 0.9435 |

**Confusion Matrix (test set):**
```
        Predicted
        Low    High
Actual Low    162     0
   High     14    72
```

**Feature Importance (Top 15 TF-IDF Terms):**
```
développeur       ↑ HIGH  (+0.8234)  # Strong indicator of high salary
ingénieur         ↑ HIGH  (+0.7891)
senior            ↑ HIGH  (+0.6543)
manager           ↑ HIGH  (+0.6234)
data              ↑ HIGH  (+0.5876)
...
stage             ↓ LOW   (-0.7123)  # Strong indicator of low salary
apprentissage     ↓ LOW   (-0.6789)
junior            ↓ LOW   (-0.5432)
```

**Output:**
- `data/enriched/hellowork_ml_enriched.csv` (1,239 jobs + ML features, cleaned data)
- `data/enriched/hellowork_ml_summary.json` (model metrics)

**Run Command:**
```bash
python run_ml.py
```

---

### Phase 4: Dashboard (Enhanced UI/UX)

**Objective:** Interactive visualization of enriched job data with modern design

**Technology Stack:**
- **Dash 3.3** (Python web framework)
- **Plotly 6.5** (interactive graphs)
- **Flask 3.1** (web server)
- **Custom CSS** with professional color palette

**Latest Dashboard Enhancements (December 2025):**
✨ **Modern UI/UX Design:**
- Professional color scheme (navy blue, turquoise, emerald, coral)
- Card-based layout with shadows and rounded corners
- Sticky sidebar for persistent filter access
- KPI metrics cards showing real-time stats
- Responsive grid layout optimized for large datasets
- Custom hover tooltips with formatted currency values
- Smooth transitions and visual feedback
- Improved typography and spacing

**KPI Cards (Dashboard Header):**
```
┌──────────────┬──────────────┬─────────────┬──────────────┐
│ 📊 JOBS      │ 💰 AVG SAL   │ 🏢 SECTORS  │ 🏭 COMPANIES │
│   1,239      │   €2,106     │     23      │     175      │
└──────────────┴──────────────┴─────────────┴──────────────┘
```

**Interactive Filters (Sticky Sidebar):**
1. **🏢 Sector** (multi-select): 23 professional sectors
2. **📍 Location** (multi-select): 651 unique locations
3. **📝 Contract Type** (multi-select): 5 contract types
4. **🔢 Cluster ID** (numeric input): Topics 0-6
5. **💵 Salary Range** (dual slider): €0 - €20,000/month
6. **🔄 Reset Button**: Clear all filters instantly

#### 📈 Dashboard Visualizations (5 Charts)

**1. 📊 Job Distribution by Sector (Bar Chart - Top 15)**
- Shows job volume across top sectors
- Color-coded with value labels outside bars
- Blue gradient for visual appeal
- Hover: Exact job counts
- Sorted descending by volume
- Identifies high-demand industries

**2. 💰 Salary Distribution (Histogram + Statistics)**
- 40-bin histogram showing salary frequency
- **Mean line** (red dashed): €2,106/month
- **Median line** (orange dotted): €2,117/month
- Reveals salary patterns and outliers
- Range: €1 - €8,229/month
- Formatted hover tooltips with currency

**3. 📍 Top 20 Job Locations (Horizontal Bar Chart)** ⭐ NEW
- Geographic analysis of job market
- Teal gradient color scheme
- Shows which cities have most opportunities
- 651 total locations tracked
- Helps job seekers identify regional hotspots
- Text labels for easy reading

**4. 🎯 Cluster Analysis by Salary (Box Plots)** 🔄 IMPROVED
- X-axis: NMF Clusters (0-6)
- Y-axis: Monthly Salary (€)
- Shows quartiles, median, and outliers
- Color-coded by cluster
- Identifies high-value vs. entry-level topics
- Statistical summary at a glance

**5. 🏢 Top 10 Companies (Horizontal Bar Chart)**
- Highlights major employers recruiting
- Viridis color gradient
- Text labels show exact job counts
- Useful for job seeker company targeting
- Shows employer concentration

#### 🎛️ Real-Time Interactive Callbacks
All 6 charts update **simultaneously** when filters change:
```python
@app.callback(
   [Output('jobs-by-sector', 'figure'),
    Output('salary-dist', 'figure'),
    Output('jobs-by-location', 'figure'),
    Output('cluster-viz', 'figure'),
    Output('top-companies', 'figure')],
   [Input('sector-filter', 'value'),
    Input('location-filter', 'value'),
    Input('contract-filter', 'value'),
    Input('cluster-filter', 'value'),
    Input('salary-range', 'value')]
)
def update_all_charts(sectors, locations, contracts, cluster, salary):
   # Filter data and return 5 updated figures
```

**Run Dashboard:**
```bash
python run_dashboard.py
# Opens at http://127.0.0.1:8050/
```

**Performance:**
- Loads 1,239 clean jobs instantly
- Real-time filtering (<100ms response)
- Responsive on desktop and tablet
- Responsive layout (desktop/tablet)

---

## 🧰 Technical Stack

### Core Technologies

| Category | Technology | Version | Purpose |
|----------|-----------|---------|---------|
| **Language** | Python | 3.13.5 | Core development language |
| **Web Scraping** | Selenium | 4.39.0 | Dynamic content extraction |
| | webdriver-manager | 4.0.2 | ChromeDriver management |
| **Data Processing** | pandas | 2.3.3 | DataFrame operations |
| | numpy | 2.3.5 | Numerical computing |
| **NLP** | NLTK | 3.9.2 | French text processing |
| | scikit-learn (TF-IDF) | 1.8.0 | Feature extraction |
| **Machine Learning** | scikit-learn | 1.8.0 | NMF + LogReg models |
| | scipy | 1.16.3 | Sparse matrix operations |
| **Visualization** | Dash | 3.3.0 | Web dashboard framework |
| | Plotly | 6.5.0 | Interactive charts |
| | Flask | 3.1.2 | Web server |
| | matplotlib | 3.10.8 | Static plots (notebooks) |
| | seaborn | 0.13.2 | Statistical visualization |
| **Development** | Jupyter | - | Interactive notebooks |
| | tqdm | 4.67.1 | Progress bars |

### Architecture Patterns

**1. Modular Design**
```
src/
├── scraping/    # Extraction layer
├── etl/         # Transformation layer
├── ml/          # Model layer
├── dashboard/   # Presentation layer
└── utils/       # Cross-cutting concerns
```

**2. Configuration Management**
```python
# Centralized in src/utils/config.py
BASE_DIR = Path(__file__).parent.parent.parent
RAW_CSV = BASE_DIR / "data" / "raw" / "hellowork_final_sectors_data.csv"
CLEAN_CSV = BASE_DIR / "data" / "processed" / "hellowork_cleaned.csv"
ENRICHED_CSV = BASE_DIR / "data" / "enriched" / "hellowork_ml_enriched.csv"
```

**3. Logging Strategy**
```python
# Consistent logging across all modules
logger = get_logger("module_name")
logger.info("Processing started")
logger.warning("Missing data detected")
logger.error("Critical failure")
```

**4. Error Handling**
- Try-catch blocks for external API calls
- Graceful degradation for missing data
- Explicit waits for Selenium (no sleep())
- Validation at each pipeline stage

## 📊 Results & Metrics (Updated with Data Cleaning)

### Data Pipeline Metrics (UPDATED)

| Stage | Input | Output | Processing | Time |
|-------|-------|--------|------------|------|
| **Phase 1: Scraping** | - | 1,364 jobs | 26 sectors | ~45 min |
| **Phase 2: ETL** | 1,364 | 1,239 jobs | Deduplicate + normalize + clean | ~10 sec |
| **Phase 2b: Cleaning** | 1,239 | 1,239 jobs | Remove empty categories ✨ | <1 sec |
| **Phase 3: ML** | 1,239 | 1,239 enriched | NMF + LogReg | ~15 sec |
| **Phase 4: Dashboard** | 1,239 | 5 charts + KPIs | Visualization | <1 sec |

### Data Quality Report (After Cleaning)

**Data Snapshot (1,239 rows):**
- Salary coverage: 1,073 rows (86.6%)
- Unique sectors: 23 | Locations: 651 | Companies: 175 | Contract types: 5
- Salary stats (n=1,073): mean €2,106 | median €2,117 | min €1 | max €8,229 | std €514 | IQR €458

**Top Sectors (by job count):**
1. Services aux Personnes / Particuliers — 290
2. Enseignement / Formation — 251
3. Distribution / Commerce — 213
4. Services aux Entreprises — 125
5. Service public hospitalier — 55

### ML Model Performance

**NMF Clustering:**
- **Topics:** 7 (bigrams, 1K features)
- **Distribution (current dataset):** {0: 209, 1: 149, 2: 210, 3: 103, 4: 404, 5: 70, 6: 94}

**Logistic Regression (Salary Prediction):**
- **Split:** 80/20 stratified on 1,239 rows (2K TF-IDF features)
- **AUC:** 0.9825 | **Precision (high):** 1.00 | **Recall (high):** 0.84 | **F1 (high):** 0.91 | **Accuracy:** 0.94

**ROC Curve Analysis:**
```
True Positive Rate @ 10% FPR: 78.2%
True Positive Rate @ 20% FPR: 89.1%
Optimal Threshold: 0.43 (probability)
```

---

## 📖 API Documentation

### Module: `src.etl.data_cleaning`

#### `load_raw(path: Path) → pd.DataFrame`
Load raw CSV with automatic column standardization.

**Parameters:**
- `path` (Path): Path to raw CSV file

**Returns:**
- DataFrame with standardized column names

**Example:**
```python
from src.etl.data_cleaning import load_raw
df = load_raw(Path("data/raw/hellowork_final_sectors_data.csv"))
```

---

#### `normalize_salary(value: str) → float`
Convert salary string to monthly EUR value.

**Handles:**
- Ranges: "2000-3000€" → 2500.0
- Units: "30k€/an" → 2500.0
- Hourly: "15€/h" → 2400.0 (15 × 160)
- Estimates: "à partir de 2500€" → 2500.0

**Returns:**
- float: Monthly salary in EUR
- np.nan: If unparseable

**Example:**
```python
normalize_salary("35-45k€/an")  # → 3333.33
normalize_salary("2500€/mois")  # → 2500.0
```

---

#### `clean_text(text: str) → str`
Clean French text using NLTK.

**Process:**
1. Lowercase
2. Remove punctuation
3. Remove French stopwords
4. Normalize whitespace

**Example:**
```python
clean_text("Nous recherchons un développeur passionné!")
# → "recherchons développeur passionné"
```

---

#### `prepare_clean(path_in: Path, path_out: Path) → pd.DataFrame`
Full ETL pipeline execution.

**Steps:**
1. Load raw data
2. Remove duplicates
3. Parse French dates
4. Normalize salaries
5. Clean text
6. Extract TF-IDF keywords
7. Encode categoricals
8. Save cleaned CSV

**Example:**
```python
df_clean = prepare_clean(
    path_in=Path("data/raw/hellowork_final_sectors_data.csv"),
    path_out=Path("data/processed/hellowork_cleaned.csv")
)
```

---

### Module: `src.ml.clustering`

#### `apply_nmf(df: pd.DataFrame, text_col: str, n_components: int) → pd.DataFrame`
Apply NMF topic modeling to job descriptions.

**Parameters:**
- `df`: Input DataFrame
- `text_col`: Column with cleaned text (default: "description_clean")
- `n_components`: Number of topics (default: 5)

**Returns:**
- DataFrame with added `job_cluster` column (0 to n_components-1)

**Example:**
```python
from src.ml.clustering import apply_nmf
df_clustered = apply_nmf(df, text_col="description_clean", n_components=5)
```

---

### Module: `src.ml.classification`

#### `prepare_labels(df: pd.DataFrame, salary_col: str) → pd.DataFrame`
Create binary salary labels using median threshold.

**Returns:**
- DataFrame with added `high_salary` column (0/1)

---

#### `train_logistic(df: pd.DataFrame, text_col: str, label_col: str) → Tuple[LogisticRegression, dict]`
Train salary classifier.

**Returns:**
- Tuple of (trained_model, metrics_dict)

**Metrics Include:**
- `classification_report`: Precision, recall, F1
- `roc_auc`: AUC score

**Example:**
```python
from src.ml.classification import train_logistic
model, metrics = train_logistic(df, text_col="description_clean")
print(f"AUC: {metrics['roc_auc']:.4f}")
```

---

### Module: `src.dashboard.app`

#### `create_app(df: pd.DataFrame) → Dash`
Create Dash application with layout and callbacks.

**Parameters:**
- `df`: Enriched DataFrame from Phase 3

**Returns:**
- Configured Dash app instance

**Example:**
```python
from src.dashboard.app import create_app
import pandas as pd

df = pd.read_csv("data/enriched/hellowork_ml_enriched.csv")
app = create_app(df)
app.run(debug=True, port=8050)
```

---

## 🔧 Troubleshooting

### Common Issues

#### 🐛 Issue: "Raw CSV not found"

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/raw/hellowork_final_sectors_data.csv'
```

**Solutions:**
1. **Ensure you're on the 'data' branch:**
   ```bash
   git checkout data
   ```

2. **Or run the scraper to generate data:**
   ```bash
   python run_scraping.py
   ```

3. **Verify file exists:**
   ```bash
   ls data/raw/
   ```

---

#### 🐛 Issue: "Module not found: nltk/selenium/dash"

**Error:**
```
ModuleNotFoundError: No module named 'nltk'
```

**Solutions:**
1. **Activate virtual environment:**
   ```bash
   # Windows
   .venv\Scripts\activate
   
   # Linux/Mac
   source .venv/bin/activate
   ```

2. **Reinstall dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data:**
   ```bash
   python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
   ```

---

#### 🐛 Issue: "Dashboard won't start / Port already in use"

**Error:**
```
OSError: [Errno 48] Address already in use
```

**Solutions:**
1. **Kill existing process on port 8050:**
   ```bash
   # Windows
   netstat -ano | findstr :8050
   taskkill /PID <PID> /F
   
   # Linux/Mac
   lsof -ti:8050 | xargs kill -9
   ```

2. **Change port in run_dashboard.py:**
   ```python
   app.run(debug=True, port=8051)  # Use different port
   ```

---

#### 🐛 Issue: "ChromeDriver version mismatch"

**Error:**
```
selenium.common.exceptions.SessionNotCreatedException: Message: session not created: This version of ChromeDriver only supports Chrome version X
```

**Solutions:**
1. **Update Chrome browser to latest version**

2. **webdriver-manager handles this automatically, but if issues persist:**
   ```bash
   pip install --upgrade webdriver-manager
   ```

3. **Clear webdriver cache:**
   ```bash
   rm -rf ~/.wdm  # Linux/Mac
   del %USERPROFILE%\.wdm  # Windows
   ```

---

#### 🐛 Issue: "Salary column has NaN values"

**Symptoms:**
- Dashboard salary histogram is empty
- Classification model has poor performance

**Solutions:**
1. **Check ETL Phase 2 completion:**
   ```python
   import pandas as pd
   df = pd.read_csv("data/processed/hellowork_cleaned.csv")
   print(df["Salary_Monthly"].describe())
   ```

2. **Re-run ETL if necessary:**
   ```bash
   python run_etl.py
   ```

3. **Expected:** 86.6% completeness (1,073/1,239 valid salaries)

---

#### 🐛 Issue: "Jupyter notebook kernel won't start"

**Solutions:**
1. **Install ipykernel in virtual environment:**
   ```bash
   pip install ipykernel
   python -m ipykernel install --user --name=lebi_env
   ```

2. **Select kernel in VS Code/Jupyter:**
   - Kernel → Change Kernel → lebi_env

---

### Performance Optimization

#### Slow Scraping (Phase 1)
- **Reduce MAX_PAGES_PER_SECTOR** in `run_scraping.py`
- **Decrease sleep time** (but respect politeness)
- **Use headless mode** in Selenium options

#### Memory Issues
- **Process in chunks** for large datasets
- **Use iterators** instead of loading full CSV
- **Clear variables** with `del df` after use

#### Dashboard Loading Slow
- **Reduce data size** by filtering in preprocessing
- **Enable caching** in Dash callbacks
- **Use server-side filtering** instead of client-side

---

### Debugging Tips

1. **Enable verbose logging:**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Check data at each phase:**
   ```bash
   # After Phase 1
   python -c "import pandas as pd; print(pd.read_csv('data/raw/hellowork_final_sectors_data.csv').shape)"
   
   # After Phase 2
   python -c "import pandas as pd; print(pd.read_csv('data/processed/hellowork_cleaned.csv').shape)"
   
   # After Phase 3
   python -c "import pandas as pd; print(pd.read_csv('data/enriched/hellowork_ml_enriched.csv').shape)"
   ```

3. **Test individual modules:**
   ```python
   # Test ETL module
   from src.etl.data_cleaning import normalize_salary
   print(normalize_salary("35k€/an"))  # Should return 2916.67
   ```

---

## 🤝 Contributing

### Development Setup

1. **Fork the repository**
2. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make changes and test:**
   ```bash
   # Run tests (if available)
   pytest tests/
   
   # Run linters
   flake8 src/
   black src/
   ```

4. **Commit with conventional commits:**
   ```bash
   git commit -m "feat: add new clustering algorithm"
   git commit -m "fix: correct salary normalization bug"
   git commit -m "docs: update README with new examples"
   ```

5. **Push and create Pull Request**

### Code Standards

- **PEP 8** compliance for Python code
- **Type hints** for function signatures
- **Docstrings** for all public functions
- **Logging** instead of print statements
- **Error handling** for external calls

### Contribution Areas

- 🌐 **Multi-language support** (extend beyond French)
- 🔍 **Additional data sources** (LinkedIn, Indeed)
- 📊 **More visualizations** (skill networks, career paths)
- 🤖 **Advanced ML models** (BERT embeddings, deep learning)
- 🚀 **Deployment guides** (Docker, AWS, Heroku)

---

## 📚 References & Resources

### Academic Papers
- **NMF:** Lee & Seung (1999) - "Learning the parts of objects by non-negative matrix factorization"
- **TF-IDF:** Salton & Buckley (1988) - "Term-weighting approaches in automatic text retrieval"

### Documentation
- [Selenium Python Docs](https://selenium-python.readthedocs.io/)
- [NLTK Documentation](https://www.nltk.org/)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Dash Documentation](https://dash.plotly.com/)
- [Plotly Python](https://plotly.com/python/)

### Tutorials Used
- Web Scraping with Selenium: [RealPython Guide](https://realpython.com/modern-web-automation-with-python-and-selenium/)
- French NLP with NLTK: [NLTK Corpus](https://www.nltk.org/howto/corpus.html)
- Dash Dashboards: [Plotly Dash Tutorial](https://dash.plotly.com/tutorial)

---

## 📄 License & Ethics

### License
This project is developed for **educational purposes only**. Not for commercial use.

### Ethical Considerations

**Web Scraping Ethics:**
- ✅ **Politeness delay** (0.5-3 seconds) implemented
- ✅ **robots.txt** compliance
- ✅ **No aggressive scraping** (respects server resources)
- ✅ **Cookie consent** handled properly
- ✅ **Educational use only** (no commercial exploitation)

**Data Privacy:**
- ✅ All data is **publicly available** job postings
- ✅ No personal data scraped (email, phone, etc.)
- ✅ Company names are public information
- ✅ Data used for analysis only, not redistribution

**Academic Integrity:**
- ✅ Project demonstrates learning objectives
- ✅ Transparent methodology
- ✅ Reproducible research practices

---

## 👥 Authors & Contact

**LEBI Project Team**
- **GitHub:** [Troudi-Yass/LEBI-PROJECT-](https://github.com/Troudi-Yass/LEBI-PROJECT-)
- **Project Type:** Educational Data Science Pipeline
- **Institution:** Business Intelligence & Data Analytics Program

### Acknowledgments

Special thanks to:
- **HelloWork.com** for providing publicly accessible job data
- **Open-source communities** (Selenium, NLTK, scikit-learn, Dash)
- **Python Software Foundation** for the amazing ecosystem

---

## 🎓 Educational Notes

This project demonstrates key concepts in:

1. **Data Engineering**
   - ETL pipeline design
   - Data quality management
   - Schema standardization

2. **Natural Language Processing**
   - French text preprocessing
   - TF-IDF vectorization
   - Topic modeling with NMF

3. **Machine Learning**
   - Unsupervised learning (clustering)
   - Supervised learning (classification)
   - Model evaluation & cross-validation

4. **Data Visualization**
   - Interactive dashboards
   - Business intelligence reporting
   - Real-time filtering

5. **Software Engineering**
   - Modular architecture
   - Configuration management
   - Logging & error handling
   - Version control with Git

---

## 📈 Future Enhancements

### Planned Features

- [ ] **Real-time scraping** with scheduled updates (cron/airflow)
- [ ] **Skill extraction** using Named Entity Recognition (NER)
- [ ] **Salary prediction** using gradient boosting (XGBoost, LightGBM)
- [ ] **Deep learning** with BERT for French text embeddings
- [ ] **Graph analysis** of company-sector relationships
- [ ] **Time series forecasting** for job market trends
- [ ] **Deployment** to cloud (AWS, GCP, Heroku)
- [ ] **Dockerization** for reproducible environments
- [ ] **Unit tests** with pytest (target: 80% coverage)
- [ ] **CI/CD pipeline** with GitHub Actions

### Research Directions

- Multi-modal learning (text + structured features)
- Career path recommendation system
- Job-candidate matching algorithms
- Labor market economic indicators

---

<div align="center">

**⭐ If you find this project useful, please star the repository! ⭐**

Made with ❤️ for Data Science Education

[🔝 Back to Top](#-lebi-project--job-market-intelligence-pipeline)

</div>
