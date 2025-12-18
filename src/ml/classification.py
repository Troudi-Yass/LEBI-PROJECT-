"""Classification module: simple binary classifier for high vs low salary.

Trains a Logistic Regression model on TF-IDF description features to predict
whether a job belongs to the 'high salary' class.
"""
from typing import Tuple
import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

from src.ml.vectorization import build_tfidf_matrix
from src.utils.config import get_logger

logger = get_logger("classification")


def prepare_labels(df: pd.DataFrame, salary_col: str = "salary_monthly") -> pd.DataFrame:
    """Create binary label `high_salary` using median threshold."""
    df = df.copy()
    if salary_col not in df.columns:
        df["high_salary"] = 0
        return df
    median = df[salary_col].median(skipna=True)
    df["high_salary"] = (df[salary_col] > median).astype(int)
    logger.info("Label median threshold = %s", median)
    return df


def train_logistic(df: pd.DataFrame, text_col: str = "description", label_col: str = "high_salary") -> Tuple[LogisticRegression, dict]:
    """Train logistic regression and return model and evaluation metrics."""
    if df.empty or text_col not in df.columns or label_col not in df.columns:
        raise ValueError("Insufficient data for training")
    texts = df[text_col].fillna("").astype(str).tolist()
    vect, X = build_tfidf_matrix(texts)
    y = df[label_col].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = LogisticRegression(max_iter=1000, C=10, solver='lbfgs', class_weight='balanced')
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    report = classification_report(y_test, preds, output_dict=True)
    auc = roc_auc_score(y_test, probs) if probs is not None else None
    metrics = {"classification_report": report, "roc_auc": auc}
    logger.info("Training complete. AUC: %s", auc)
    return model, metrics


def predict_and_attach(df: pd.DataFrame, model, vect, text_col: str = "description", out_col: str = "predicted_high_salary") -> pd.DataFrame:
    """Attach predictions to DataFrame and return it."""
    df = df.copy()
    texts = df[text_col].fillna("").astype(str).tolist()
    X = vect.transform(texts)
    preds = model.predict(X)
    df[out_col] = preds
    return df


if __name__ == "__main__":
    print("Classification module. Use prepare_labels() and train_logistic().")
