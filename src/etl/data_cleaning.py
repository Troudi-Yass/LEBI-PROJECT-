"""ETL and data preparation utilities.

Functions to load raw CSV, clean duplicates, handle missing values, normalize
salary fields, extract keywords via TF-IDF and encode categorical variables.
"""
from typing import Tuple
import re
import logging
import string
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from src.utils.config import RAW_CSV, CLEAN_CSV, get_logger, ensure_dirs

logger = get_logger("data_cleaning")

# Download NLTK resources (only once)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# Initialize French stopwords
FRENCH_STOPWORDS = set(stopwords.words('french'))


def load_raw(path: Path = RAW_CSV) -> pd.DataFrame:
    """Load raw CSV into a DataFrame.

    Args:
        path: Path to the raw CSV file.
    """
    ensure_dirs()
    try:
        df = pd.read_csv(path, encoding="utf-8")
        logger.info("Loaded raw data: %s rows", len(df))
        # standardize columns from various raw formats (e.g. provided HelloWork CSV)
        df = standardize_columns(df)
        return df
    except FileNotFoundError:
        logger.error("Raw CSV not found at %s", path)
        return pd.DataFrame()


def clean_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows and duplicates by URL if present."""
    if df.empty:
        return df
    before = len(df)
    df = df.drop_duplicates()
    if "job_url" in df.columns:
        df = df.drop_duplicates(subset=["job_url"])
    logger.info("Dropped duplicates: %d -> %d", before, len(df))
    return df


def parse_relative_date(date_str) -> pd.Timestamp:
    """Parse French relative dates (aujourd'hui, hier, il y a X jours/mois).
    
    Args:
        date_str: Date string to parse
        
    Returns:
        Parsed datetime or NaT if parsing fails
    """
    if pd.isna(date_str) or date_str == "N/A":
        return pd.NaT
    
    date_str = str(date_str).lower()
    today = datetime.today().date()
    
    try:
        if "hier" in date_str:
            return pd.Timestamp(today - timedelta(days=1))
        elif "aujourd'hui" in date_str:
            return pd.Timestamp(today)
        elif "il y a" in date_str:
            # Extract number
            nums = re.findall(r'\d+', date_str)
            if nums:
                val = int(nums[0])
                if "mois" in date_str:
                    return pd.Timestamp(today - timedelta(days=val*30))
                elif "jour" in date_str:
                    return pd.Timestamp(today - timedelta(days=val))
                elif "heure" in date_str:
                    return pd.Timestamp(today)  # Same day
                elif "minute" in date_str:
                    return pd.Timestamp(today)  # Same day
        # Fallback: try standard date parsing if it contains "/"
        return pd.to_datetime(date_str, dayfirst=True, errors='coerce')
    except:
        return pd.NaT


def clean_text(text: str) -> str:
    """Clean text using NLTK: lowercase, remove punctuation, stopwords.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    if pd.isna(text) or text == "Not specified":
        return ""
    
    text = text.lower()  # lowercase
    text = text.replace("\n", " ").strip()  # remove line breaks
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    tokens = [word for word in text.split() if word not in FRENCH_STOPWORDS]  # remove stopwords
    return " ".join(tokens)


def normalize_salary(value: str) -> float:
    """Normalize salary strings into monthly numeric value when possible.

    Returns NaN if salary cannot be parsed.
    Examples supported: '30k€', '30000 €/an', '2 500 €/mois', 'à partir de 45k€'
    """
    if not isinstance(value, str) or not value.strip():
        return np.nan
    s = value.lower()
    # normalize whitespace and non-breaking spaces
    s = s.replace("\u202f", " ").replace("\xa0", " ").replace("\u2009", " ")
    s = s.replace("à partir de", "").replace("estimation →", "").replace("estimation", "")
    s = s.replace('\u200b', '')

    # unify euro/hour/year mentions
    per_hour = "heure" in s or "/heure" in s or "h" in s and ("€/h" in s or "€ / h" in s)
    per_year = "an" in s or "/an" in s or "par an" in s or "annuel" in s
    per_month = "mois" in s or "/mois" in s or "par mois" in s

    # find numbers like 1.234,56 or 1234.56 or 1234
    num_matches = re.findall(r"\d+[\d\.\s]*[\,\.]?\d*", s)
    clean_nums = []
    for m in num_matches:
        m_clean = m.strip().replace(" ", "").replace("\u202f", "")
        # replace comma decimal with dot
        if m_clean.count(',') and m_clean.count('.') == 0:
            m_clean = m_clean.replace(',', '.')
        # remove grouping dots if any (e.g. 1.900)
        if m_clean.count('.') > 1:
            m_clean = m_clean.replace('.', '')
        # final remove non-numeric except dot
        m_clean = re.sub(r"[^0-9\.]", "", m_clean)
        try:
            clean_nums.append(float(m_clean))
        except Exception:
            continue

    if not clean_nums:
        return np.nan

    # If range provided, take mean
    if len(clean_nums) >= 2:
        val = float(sum(clean_nums[:2]) / 2.0)
    else:
        val = float(clean_nums[0])

    # detect 'k' multiplier
    if 'k' in s:
        val = val * 1000.0

    # Convert units to monthly
    if per_hour:
        # assume 160 working hours per month
        monthly = val * 160.0
    elif per_year:
        monthly = val / 12.0
    elif per_month:
        monthly = val
    else:
        # ambiguous: if value > 5000 assume yearly else monthly
        monthly = val / 12.0 if val > 5000 else val

    return float(monthly)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names from various raw inputs into expected schema.

    Maps common variants to: job_title, company, location, salary, contract_type,
    publication_date, description, job_url, sector
    """
    if df.empty:
        return df
    cols = {c: c for c in df.columns}
    mapping = {}
    for c in df.columns:
        cn = c.strip().lower()
        if cn in ("job_title", "job title", "title", "titre", "job") or 'job_title' in c or 'job title' in cn:
            mapping[c] = 'job_title'
        elif cn in ("company", "employer", "entreprise") or 'company' in c or 'entreprise' in cn:
            mapping[c] = 'company'
        elif cn in ("location", "lieu", "ville") or 'location' in c:
            mapping[c] = 'location'
        elif cn in ("salary", "salaire") or 'salary' in c or 'salaire' in cn:
            mapping[c] = 'salary'
        elif cn in ("contract", "contract_type", "type-contrat", "contrat") or 'contract' in c or 'contrat' in cn:
            mapping[c] = 'contract_type'
        elif cn in ("publication_date", "date", "posted", "publication"):
            mapping[c] = 'publication_date'
        elif cn in ("description", "desc", "description_offre") or 'description' in c:
            mapping[c] = 'description'
        elif cn in ("url", "job_url", "link") or 'url' in c or 'link' in c:
            mapping[c] = 'job_url'
        elif cn in ("sector", "secteur") or 'sector' in c or 'secteur' in cn:
            mapping[c] = 'sector'
        else:
            # leave other columns untouched
            mapping[c] = c

    df = df.rename(columns=mapping)
    # ensure all expected columns exist
    expected = ["job_title", "company", "location", "salary", "contract_type", "publication_date", "description", "job_url", "sector"]
    for e in expected:
        if e not in df.columns:
            df[e] = "" if e in ("job_title", "company", "location", "description", "sector", "contract_type") else np.nan

    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values with simple strategies.

    - Fill empty strings for text columns
    - Keep salary as NaN when not parsed
    """
    if df.empty:
        return df
    text_cols = [c for c in ["job_title", "company", "location", "description", "sector"] if c in df.columns]
    for c in text_cols:
        df[c] = df[c].fillna("")
    return df


def remove_empty_categories(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """Remove rows with empty/missing values in critical categorical columns.
    
    Args:
        df: Input DataFrame
        columns: List of columns to check. Defaults to ['sector', 'location', 'contract_type']
        
    Returns:
        DataFrame with rows having empty critical columns removed
    """
    if df.empty:
        return df
    
    if columns is None:
        columns = ['sector', 'location', 'contract_type']
    
    # Filter to only existing columns
    columns_to_check = [c for c in columns if c in df.columns]
    
    before = len(df)
    
    # Remove rows where any of the critical columns are empty/missing
    for col in columns_to_check:
        # Convert to string and strip whitespace
        df[col] = df[col].astype(str).str.strip()
        # Remove rows with empty strings or 'nan'
        df = df[(df[col] != '') & (df[col] != 'nan') & (df[col].notna())]
        logger.info("  - Removed empty '%s' values: %d rows remaining", col, len(df))
    
    removed = before - len(df)
    logger.info("✓ Removed rows with empty categories: %d -> %d (removed %d)", before, len(df), removed)
    
    return df


def remove_contracts_without_salary(df: pd.DataFrame, min_salary_jobs: int = 5) -> pd.DataFrame:
    """Remove contract types with insufficient salary data.
    
    Args:
        df: Input DataFrame
        min_salary_jobs: Minimum number of jobs with salary data to keep a contract type
        
    Returns:
        DataFrame with contract types having insufficient salary data removed
    """
    if df.empty or 'contract_type' not in df.columns or 'salary_monthly' not in df.columns:
        return df
    
    before = len(df)
    contracts_to_remove = []
    
    # Check salary data for each contract type
    for contract in df['contract_type'].unique():
        if pd.isna(contract) or contract == '':
            continue
        
        contract_subset = df[df['contract_type'] == contract]
        salary_count = contract_subset['salary_monthly'].notna().sum()
        
        # If less than min_salary_jobs with salary, mark for removal
        if salary_count < min_salary_jobs:
            contracts_to_remove.append(contract)
            logger.info(f"  - Removing '{contract}': only {salary_count} jobs with salary data")
    
    # Remove the marked contract types
    if contracts_to_remove:
        df = df[~df['contract_type'].isin(contracts_to_remove)]
        removed = before - len(df)
        logger.info(f"✓ Removed {len(contracts_to_remove)} contract types with insufficient salary: {before} -> {len(df)} (removed {removed})")
    
    return df


def extract_keywords_tfidf(df: pd.DataFrame, text_col: str = "description", top_k: int = 10) -> pd.DataFrame:
    """Compute TF-IDF and attach top keywords as a new column.

    Returns the DataFrame with an added `top_keywords` column (comma separated).
    """
    if df.empty or text_col not in df.columns:
        df["top_keywords"] = ""
        return df
    # sklearn only provides built-in 'english' stop word list. Use None for
    # compatibility; users can provide a French list if available later.
    vect = TfidfVectorizer(max_features=1000, stop_words=None)
    texts = df[text_col].fillna("").astype(str).tolist()
    X = vect.fit_transform(texts)
    feature_names = vect.get_feature_names_out()

    def top_terms(row):
        if row.nnz == 0:
            return ""
        scores = zip(row.indices, row.data)
        sorted_terms = sorted(scores, key=lambda x: -x[1])[:top_k]
        return ",".join(feature_names[i] for i, _ in sorted_terms)

    df["top_keywords"] = [top_terms(X[i]) for i in range(X.shape[0])]
    return df


def encode_categoricals(df: pd.DataFrame, cols: list = None) -> pd.DataFrame:
    """Encode categorical variables using simple label encoding (factorize).

    Args:
        df: Input dataframe.
        cols: List of columns to encode; if None common columns are used.
    """
    if df.empty:
        return df
    if cols is None:
        cols = [c for c in ["sector", "location", "contract_type", "company"] if c in df.columns]
    for c in cols:
        df[c + "_enc"] = pd.factorize(df[c].astype(str))[0]
    return df


def prepare_clean(path_in: Path = RAW_CSV, path_out: Path = CLEAN_CSV) -> pd.DataFrame:
    """Full ETL pipeline: load, clean, normalize and save a clean CSV for ML.

    Returns:
        Cleaned DataFrame.
    """
    df = load_raw(path_in)
    if df.empty:
        return df
    df = clean_duplicates(df)
    df = handle_missing(df)
    
    # Remove rows with empty critical categories (MAJOR CHANGE!)
    logger.info("Removing rows with empty categories...")
    df = remove_empty_categories(df, columns=['sector', 'location', 'contract_type'])
    
    # Parse publication dates (French relative dates)
    if "publication_date" in df.columns:
        logger.info("Parsing publication dates...")
        df["publication_date"] = df["publication_date"].apply(parse_relative_date)
        df["publication_date"] = pd.to_datetime(df["publication_date"])
        logger.info("Date range: %s to %s", df["publication_date"].min(), df["publication_date"].max())
    
    # Normalize salary to monthly numeric
    if "salary" in df.columns:
        df["salary_monthly"] = df["salary"].apply(normalize_salary)
    
    # Remove contract types with insufficient salary data (AFTER salary normalization!)
    logger.info("Removing contract types with insufficient salary data...")
    df = remove_contracts_without_salary(df, min_salary_jobs=5)
    
    # Clean text descriptions using NLTK
    if "description" in df.columns:
        logger.info("Cleaning text descriptions...")
        df["description_clean"] = df["description"].apply(clean_text)
    
    # Extract keywords from cleaned text
    text_col = "description_clean" if "description_clean" in df.columns else "description"
    df = extract_keywords_tfidf(df, text_col=text_col)
    
    # Encode categoricals
    df = encode_categoricals(df)

    try:
        ensure_dirs()
        df.to_csv(path_out, index=False, encoding="utf-8")
        logger.info("Saved cleaned data to %s (rows=%d)", path_out, len(df))
    except Exception as e:
        logger.error("Failed saving clean CSV: %s", e)

    return df


if __name__ == "__main__":
    prepare_clean()
