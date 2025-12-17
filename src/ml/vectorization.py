"""TF-IDF vectorization utilities for job descriptions."""
from typing import Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_matrix(texts, max_features: int = 2000) -> Tuple[TfidfVectorizer, object]:
    """Fit a TF-IDF vectorizer and return (vectorizer, matrix).

    Args:
        texts: Iterable of documents.
        max_features: Maximum vocabulary size.
    """
    # Use no built-in stop words to remain compatible with scikit-learn versions
    # that only provide an 'english' built-in list. A custom French stop-list
    # can be provided later if desired.
    vect = TfidfVectorizer(max_features=max_features, stop_words=None)
    X = vect.fit_transform([t if isinstance(t, str) else "" for t in texts])
    return vect, X


def load_vectorizer(path=None):
    """Placeholder for backwards compatibility (always returns None).
    
    Vectorizer is now computed on-the-fly rather than cached.
    """
    return None


if __name__ == "__main__":
    print("Vectorization module. Import and call build_tfidf_matrix().")
