
# --- Basic Data Handling ---
import pandas as pd          # For loading, cleaning, and manipulating datasets
import numpy as np           # For numerical operations and handling NaNs
# --- File / Path Utilities ---
from pathlib import Path     # For filesystem-safe path handling
import os                    # For interacting with the filesystem
# --- Web/Regex/Text Processing ---
import re                    # For regular expressions (cleaning salaries, text)
import string                # For punctuation removal in text
import nltk                  # Natural Language Toolkit for text processing
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('stopwords')   # Download the stopwords corpus
nltk.download('punkt')       # Download tokenizer models
nltk.download('wordnet')     # Download WordNet lemmatizer data
# --- Machine Learning / Preprocessing ---
from sklearn.feature_extraction.text import TfidfVectorizer  # Convert text to numeric features
from sklearn.feature_extraction.text import CountVectorizer  # Bag-of-words representation
from sklearn.preprocessing import OneHotEncoder              # Encode categorical variables
from sklearn.preprocessing import LabelEncoder               # Label encoding if needed
from sklearn.preprocessing import StandardScaler             # Scale numeric features
from sklearn.model_selection import train_test_split         # Train/validation splits
# --- Date/Time Handling ---
import datetime              # For working with publication dates and job age
from datetime import timedelta
import random                # For mocking dates if missing

# --- Visualization & Progress ---
import matplotlib.pyplot as plt  # Quick exploratory plots
import seaborn as sns            # Statistical visualizations
from tqdm import tqdm            # Progress bars for loops

# --- Step 1: Load Raw Dataset ---
# Load the final scraped CSV from Phase 1
df = pd.read_csv("hellowork_final_sectors_data.csv", encoding='utf-8-sig')

# Inspect basic info
print("Dataset shape:", df.shape)
print("\nColumns:\n", df.columns)
print("\nFirst 5 rows:")
print(df.head())

# --- Step 2: Handle Missing Values and Duplicates ---

# 1. Remove duplicate rows
df = df.drop_duplicates()
print(f"After removing duplicates, dataset shape: {df.shape}")

# 2. Handle missing values
text_columns = ["Job_Title", "Company", "Location", "Contract", "Description"]
for col in text_columns:
    df[col] = df[col].fillna("Not specified")

# For Salary, leave as NaN for now (we'll process it in Step 3)
df['Salary'] = pd.to_numeric(df['Salary'].str.replace(r'[^\d]', '', regex=True), errors='coerce')

# --- DATE PARSING (Real Data) ---
# Parse extracted 'Publication_Date' extracting relative dates.

def parse_relative_date(date_str):
    if pd.isna(date_str) or date_str == "N/A":
        return np.nan
    
    date_str = str(date_str).lower()
    today = datetime.date.today()
    
    try:
        if "hier" in date_str:
            return today - timedelta(days=1)
        elif "aujourd'hui" in date_str:
            return today
        elif "il y a" in date_str:
            # Extract number
            nums = re.findall(r'\d+', date_str)
            if nums:
                val = int(nums[0])
                if "mois" in date_str:
                    return today - timedelta(days=val*30)
                elif "jour" in date_str:
                    return today - timedelta(days=val)
                elif "heure" in date_str:
                    return today # Same day
                elif "minute" in date_str:
                    return today # Same day
        # Fallback: try standard date parsing if it contains "/"
        return pd.to_datetime(date_str, dayfirst=True, errors='coerce')
    except:
        return np.nan

print("Parsing Publication Dates...")
if 'Publication_Date' in df.columns:
    df['Publication_Date'] = df['Publication_Date'].apply(parse_relative_date)
    df['Publication_Date'] = pd.to_datetime(df['Publication_Date'])
else:
    print("Warning: Publication_Date column missing in input CSV.")

print("\nDate range:", df['Publication_Date'].min(), "to", df['Publication_Date'].max())

# 3. Check remaining missing values
print("\nMissing values per column:")
print(df.isna().sum())

# --- Step 3: Standardize Salaries ---

def clean_salary(salary_str):
    """
    Extract numeric salary value and standardize it.
    Assumes:
    - Monthly salary if not specified
    - Average for ranges
    """
    if pd.isna(salary_str) or salary_str == "Not specified":
        return np.nan
    # Remove non-digit characters
    numbers = re.findall(r'\d+', salary_str.replace(" ", ""))
    numbers = [int(n) for n in numbers]
    if len(numbers) == 0:
        return np.nan
    elif len(numbers) == 1:
        return numbers[0]
    else:
        # If range, return average
        return sum(numbers)/len(numbers)

# Apply cleaning function
df['Salary_Clean'] = df['Salary'].astype(str).apply(clean_salary)

# Inspect cleaned salary column
print(df[['Salary', 'Salary_Clean']].head(10))

# --- Step 4: Text Preprocessing ---

# Download stopwords if not already
nltk.download('stopwords')
stop_words = set(stopwords.words('french'))  # Using French stopwords

def clean_text(text):
    text = text.lower()  # lowercase
    text = text.replace("\n", " ").strip()  # remove line breaks
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    tokens = [word for word in text.split() if word not in stop_words]  # remove stopwords
    return " ".join(tokens)

# Apply cleaning
df['Description_Clean'] = df['Description'].apply(clean_text)

# Inspect cleaned descriptions
print(df[['Description', 'Description_Clean']].head(5))

# --- Step 5: Encode Categorical Variables ---

from sklearn.preprocessing import LabelEncoder

# Columns to encode
categorical_cols = ['Contract', 'Location', 'Sector']

# Apply Label Encoding
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col + '_Encoded'] = le.fit_transform(df[col])
    le_dict[col] = le  # Save encoder for future inverse_transform if needed

# Inspect encoding
print(df[['Contract', 'Contract_Encoded', 'Location', 'Location_Encoded', 'Sector', 'Sector_Encoded']].head(5))

# --- Step 6: Feature Extraction from Job Descriptions ---

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=500,  # Keep top 500 words
                                   min_df=5,          # Ignore words appearing in <5 docs
                                   max_df=0.9)        # Ignore words appearing in >90% of docs

# Fit and transform the cleaned descriptions
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Description_Clean'])

# Convert to DataFrame for inspection
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

print(f"TF-IDF feature matrix shape: {tfidf_df.shape}")
print(tfidf_df.head())

# --- SAVE PROCESSED DATA ---
df.to_csv("hellowork_preprocessed.csv", index=False)
print("Data preprocessed and saved to hellowork_preprocessed.csv")
