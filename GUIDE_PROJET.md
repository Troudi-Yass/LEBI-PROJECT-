# 📘 Guide du Projet LEBI - Analyse du Marché de l'Emploi

Ce guide explique le fonctionnement technique et fonctionnel des 4 phases du projet.
Le projet est disponible en deux formats : **Notebooks Jupyter** (`.ipynb`) pour l'exploration et **Scripts Python** (`.py`) pour l'automatisation.

---

## 🏗️ Phase 1 : Extraction des Données (Web Scraping)
**Objectif** : Récupérer les offres d'emploi depuis le site **Hellowork**.

*   **Fichiers** : `LEBI_Scrapping.py` (ou `.ipynb`)
*   **Technologie** : `Selenium` (Chrome WebDriver)
*   **Fonctionnement** :
    1.  Le script parcourt une liste de **26 secteurs** d'activité.
    2.  Pour chaque secteur, il navigue sur les pages de résultats (jusqu'à 10 pages).
    3.  Il extrait : Titre, Entreprise, Localisation, Contrat, Salaire, Description, URL.
*   **Résultat** : Crée un fichier brut `hellowork_final_sectors_data.csv`.

---

## 🧹 Phase 2 : Préparation des Données (ETL)
**Objectif** : Nettoyer et structurer les données brutes pour le Machine Learning.

*   **Fichiers** : `LEBI_ETL.py` (ou `.ipynb`)
*   **Technologie** : `Pandas`, `NLTK`, `Scikit-learn`
*   **Traitements** :
    1.  **Nettoyage** : Suppression des doublons et des valeurs manquantes.
    2.  **Date (Simulation)** : Génération de dates de publication fictives (analyse temporelle) car non scrapées en Phase 1.
    3.  **Salaires** : Conversion des textes (ex: "30k-40k") en valeurs numériques (moyenne mensuelle).
    4.  **NLP** : Nettoyage du texte des descriptions (minuscules, retrait ponctuation/stopwords).
    5.  **Encodage** : Transformation des variables Catégorielles (Secteur, Contrat) en chiffres.
*   **Résultat** : Crée le fichier propre `hellowork_preprocessed.csv`.

---

## 🤖 Phase 3 : Modélisation & Enrichissement ML
**Objectif** : Créer de la valeur ajoutée grâce à l'Intelligence Artificielle.

*   **Fichiers** : `Phase3_Modelisation.py` (ou `.ipynb`)
*   **Technologie** : `Scikit-learn` (NMF, LogisticRegression)
*   **Algorithmes** :
    1.  **Clustering (NMF)** : Analyse les descriptions pour regrouper les offres en **5 thématiques métiers** (Topics). *Utilise NMF au lieu de KMeans pour une meilleure interprétation textuelle.*
    2.  **Classification (Logistic Regression)** : Prédit si une offre est "Haut Salaire" ou "Bas Salaire" en fonction du secteur et du contrat.
*   **Résultat** : Crée le fichier enrichi `hellowork_ml_enriched.csv` contenant les clusters et les prédictions.

---

## 📊 Phase 4 : Dashboard Interactif
**Objectif** : Visualiser les données et les résultats des modèles.

*   **Fichiers** : `Phase4_Dashboard.py` (ou `.ipynb`)
*   **Technologie** : `Dalsh` (Plotly)
*   **Fonctionnalités** :
    1.  **Filtres** : Sélection dynamique par secteur d'activité.
    2.  **Graphiques** :
        *   Répartition des Clusters Métiers (Histogramme).
        *   Classification Salariale (Barres).
        *   **NOUVEAU** : Analyse Temporelle (Courbe des offres par semaine).
*   **Accès** : Lancez le script et ouvrez http://127.0.0.1:8050/ dans votre navigateur.

---

## 🚀 Comment lancer le projet complet ?

**Option A (Recommandée) : Lancement Automatique**
Lancez tout le pipeline en une seule commande :
```bash
python main.py
```

**Option B : Lancement Manuel (étape par étape)**
Exécutez les commandes suivantes dans votre terminal, dans l'ordre :

```bash
# 1. Scraping (Long - peut être sauté si vous avez déjà les données)
python LEBI_Scrapping.py

# 2. Nettoyage et préparation (Inclus la génération de dates)
python LEBI_ETL.py

# 3. Intelligence Artificielle (Clustering & Classification)
python Phase3_Modelisation.py

# 4. Lancement de l'application Dashboard
python Phase4_Dashboard.py
```
