
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# ==========================================
# Phase 3 - Modélisation & Enrichissement Machine Learning
# ==========================================

print("--- 3.2 Chargement des données ---")
df = pd.read_csv("hellowork_preprocessed.csv")
print(f"Dimensions du dataset : {df.shape}")

# Ensure Date column is preserved/loaded as datetime
if 'Publication_Date' in df.columns:
    df['Publication_Date'] = pd.to_datetime(df['Publication_Date'])

# ==========================================
# 3.3 Clustering NLP (Descriptions d'offres)
# ==========================================

print("\n--- a) Vectorisation TF-IDF ---")
tfidf = TfidfVectorizer(max_features=500)
# On s'assure que Description_Clean est bien de type string
X_text = tfidf.fit_transform(df['Description_Clean'].astype(str))

print("\n--- b) Modélisation par NMF (Non-negative Matrix Factorization) ---")
# On choisit 5 thématiques (clusters)
nmf = NMF(n_components=5, random_state=42)

# W contient la matrice Documents x Topics
W = nmf.fit_transform(X_text)

# On assigne chaque offre au topic dominant (celui avec le score le plus élevé)
df['Job_Cluster'] = W.argmax(axis=1)

print("\n--- c) Interprétation des clusters (Thématiques) ---")
terms = tfidf.get_feature_names_out()
# Pour NMF, nmf.components_ contient la matrice Topics x Termes
for i in range(5):
    center_terms = nmf.components_[i].argsort()[-10:]
    print(f"Cluster {i}:", [terms[t] for t in center_terms])

# ==========================================
# 3.4 Classification - Niveau de salaire
# ==========================================

print("\n--- a) Création de la cible ---")
median_salary = df['Salary_Clean'].median()
df['High_Salary'] = (df['Salary_Clean'] >= median_salary).astype(int)

print("\n--- b) Entraînement du modèle ---")
features = df[['Sector_Encoded', 'Contract_Encoded']].fillna(0)
X_train, X_test, y_train, y_test = train_test_split(
    features, df['High_Salary'], test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print(classification_report(y_test, model.predict(X_test)))

print("\n--- c) Ajout des prédictions ---")
df['Salary_Class_Pred'] = model.predict(features)

# ==========================================
# 3.5 Sauvegarde dataset enrichi
# ==========================================
output_file = "hellowork_ml_enriched.csv"
df.to_csv(output_file, index=False)
print(f"\nDataset enrichi sauvegardé sous : {output_file}")
