"""Clustering utilities using TF-IDF + NMF topic modeling.

Provides functions to cluster job descriptions using Non-negative Matrix 
Factorization (NMF) for interpretable topic discovery.
"""
from typing import Optional
import logging
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.config import get_logger

logger = get_logger("clustering")


def apply_nmf(df: pd.DataFrame, text_col: str = "description_clean", 
              n_components: int = 5, max_features: int = 500) -> pd.DataFrame:
    """Apply NMF (Non-negative Matrix Factorization) to cluster jobs by topic.
    
    NMF discovers latent topics in job descriptions and assigns each job to the
    dominant topic. This is useful for grouping similar roles and understanding
    job market themes.
    
    Args:
        df: DataFrame with text descriptions
        text_col: Column containing cleaned text (defaults to 'description_clean')
        n_components: Number of topics/clusters to discover
        max_features: Maximum TF-IDF features
        
    Returns:
        DataFrame with added 'job_cluster' column (0 to n_components-1)
    """
    if df.empty or text_col not in df.columns:
        logger.warning("No data or text column '%s' missing, skipping NMF", text_col)
        df["job_cluster"] = -1
        return df
    
    texts = df[text_col].fillna("").astype(str)
    logger.info("Applying TF-IDF vectorization (max_features=%d)", max_features)
    tfidf = TfidfVectorizer(max_features=max_features, min_df=2, max_df=0.9, ngram_range=(1, 2))
    X_text = tfidf.fit_transform(texts)
    
    logger.info("Applying NMF with %d components", n_components)
    nmf = NMF(n_components=n_components, random_state=42, max_iter=500, init='nndsvda')
    W = nmf.fit_transform(X_text)  # Documents x Topics matrix
    
    # Assign each job to dominant topic
    df = df.copy()
    df["job_cluster"] = W.argmax(axis=1)
    
    # Log top terms for each topic for interpretability
    terms = tfidf.get_feature_names_out()
    logger.info("Topic interpretation (top 10 terms per topic):")
    for i in range(n_components):
        top_term_indices = nmf.components_[i].argsort()[-10:]
        top_terms = [terms[idx] for idx in top_term_indices]
        logger.info("  Topic %d: %s", i, ", ".join(top_terms))
    
    logger.info("NMF clustering complete. Cluster distribution:\n%s", 
                df["job_cluster"].value_counts().sort_index())
    
    return df
