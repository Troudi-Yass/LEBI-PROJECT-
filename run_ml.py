"""LEBI Project - Phase 3: Machine Learning

Apply NMF clustering and salary classification to enrich the dataset.

Usage:
    python run_ml.py
"""
import pandas as pd
from src.ml.clustering import apply_nmf
from src.ml.classification import prepare_labels, train_logistic
from src.ml.vectorization import build_tfidf_matrix
from src.utils.config import CLEAN_CSV, ENRICHED_CSV, get_logger, ensure_dirs

logger = get_logger("run_ml")

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("PHASE 3: MACHINE LEARNING")
    logger.info("=" * 60)
    
    try:
        # Load cleaned data
        logger.info("Loading cleaned data from: %s", CLEAN_CSV)
        df = pd.read_csv(CLEAN_CSV, encoding="utf-8")
        logger.info("✓ Loaded %d rows", len(df))
        
        # Step 1: NMF Topic Clustering
        logger.info("\n--- Step 1: NMF Topic Clustering ---")
        text_col = "description_clean" if "description_clean" in df.columns else "description"
        df = apply_nmf(df, text_col=text_col, n_components=7, max_features=1000)
        logger.info("✓ NMF clustering completed. Added 'job_cluster' column.")
        
        # Step 2: Classification (Salary Prediction)
        logger.info("\n--- Step 2: Classification (Salary Prediction) ---")
        df = prepare_labels(df, salary_col="salary_monthly")
        logger.info("✓ Labels prepared based on salary median.")
        
        # Filter rows with valid labels for training
        df_train = df[df["high_salary"].notna() & df[text_col].notna()].copy()
        
        if len(df_train) > 10:
            try:
                model, metrics = train_logistic(df_train, text_col=text_col, label_col="high_salary")
                logger.info("✓ Model trained. AUC: %.4f", metrics.get("roc_auc", 0))
                
                # Add predictions to full dataset
                vect, X = build_tfidf_matrix(df[text_col].fillna("").astype(str).tolist())
                preds = model.predict(X)
                df["predicted_high_salary"] = preds
                logger.info("✓ Predictions added to dataset.")
            except Exception as e:
                logger.warning("Classification failed: %s", e)
                df["predicted_high_salary"] = -1
        else:
            logger.warning("Insufficient data for classification. Skipping.")
            df["predicted_high_salary"] = -1
        
        # Step 3: Save enriched dataset
        logger.info("\n--- Step 3: Saving Enriched Data ---")
        ensure_dirs()
        df.to_csv(ENRICHED_CSV, index=False, encoding="utf-8")
        logger.info("✓ Enriched data saved to: %s", ENRICHED_CSV)
        logger.info("  - Rows: %d", len(df))
        logger.info("  - New columns: cluster, high_salary, predicted_high_salary")
        
    except Exception as e:
        logger.error("✗ ML pipeline failed: %s", e)
        raise
    
    logger.info("=" * 60)
    logger.info("PHASE 3 COMPLETED")
    logger.info("=" * 60)
