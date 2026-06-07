# 🔤 Letter Recognition — ML Classification

> Reconnaissance automatique des 26 lettres de l'alphabet à partir de caractéristiques statistiques d'images, avec **97.8% de précision** (SVM RBF).

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-UCI%20Letter%20Recognition-purple)](https://archive.ics.uci.edu/dataset/59/letter+recognition)

---

## 📋 Vue d'ensemble

Ce projet applique des techniques classiques de **machine learning supervisé** sur le dataset UCI *Letter Recognition* : 20 000 exemples de lettres (A–Z) encodées en 16 attributs statistiques (moments, densité de pixels, contours).

### Résultats

| Modèle | Accuracy | CV (5-fold) |
|--------|----------|-------------|
| k-NN (k=5) | 95.6% | 95.4% ± 0.3% |
| Random Forest (200 arbres) | 96.3% | 96.2% ± 0.2% |
| **SVM (RBF, C=10)** | **97.8%** | **97.7% ± 0.2%** |

---

## 🗂️ Structure du projet

```
letter-recognition/
├── data/
│   └── letter-recognition.csv     # Dataset UCI (à télécharger)
├── notebooks/
│   └── letter_recognition.ipynb   # Analyse complète & visualisations
├── src/
│   ├── train.py                   # Pipeline d'entraînement
│   ├── clustering.py              # PCA, t-SNE, KMeans
│   └── predict.py                 # Inférence & API de prédiction
├── outputs/                        # Modèles .pkl & graphiques générés
├── requirements.txt
└── README.md
```

---

## 🚀 Démarrage rapide

### 1. Installation

```bash
git clone https://github.com/VOTRE_USERNAME/letter-recognition.git
cd letter-recognition
pip install -r requirements.txt
```

### 2. Données

Télécharger le dataset depuis [UCI](https://archive.ics.uci.edu/dataset/59/letter+recognition) et placer le CSV dans `data/letter-recognition.csv`.

### 3. Entraîner un modèle

```bash
# SVM (meilleure précision)
python src/train.py --model svm --save

# Tous les modèles + comparaison
python src/train.py --model all --cv 5

# Avec optimisation des hyperparamètres
python src/train.py --model svm --tune --save
```

### 4. Analyser les données (PCA, t-SNE)

```bash
# Toutes les analyses
python src/clustering.py

# t-SNE uniquement (5000 exemples)
python src/clustering.py --method tsne --sample 5000

# PCA uniquement
python src/clustering.py --method pca
```

### 5. Prédire une lettre

```bash
# À partir de 16 features manuelles (ici : lettre A typique)
python src/predict.py --model outputs/model_svm.pkl \
  --features "4,4,5,6,3,7,4,6,5,8,7,6,3,7,2,6"

# Sur un fichier CSV
python src/predict.py --model outputs/model_svm.pkl \
  --input mon_fichier.csv --output predictions.csv

# Mode interactif
python src/predict.py --model outputs/model_svm.pkl
```

### 6. Notebook interactif

```bash
jupyter notebook notebooks/letter_recognition.ipynb
```

---

## 📊 Analyses réalisées

### Exploration (EDA)
- Distribution des 26 classes (quasi-équilibrée)
- Distributions des 16 features par lettre
- Heatmap de corrélation entre features

### Réduction de dimensionnalité
- **PCA** : 8 composantes capturent 95% de la variance
- **t-SNE** : clusters nets et séparés pour la majorité des lettres
- Identification des paires difficiles (I/J, B/D, G/C)

### Modélisation
- Pipeline sklearn avec `StandardScaler` (anti data-leakage)
- Validation croisée 5-fold stratifiée
- `GridSearchCV` pour l'optimisation des hyperparamètres SVM

### Évaluation
- Matrice de confusion 26×26
- Précision par classe
- Analyse des erreurs les plus fréquentes

---

## 🔍 Features du dataset

| Feature | Description |
|---------|-------------|
| `x-box`, `y-box` | Position de la boîte englobante |
| `width`, `high` | Dimensions de la boîte |
| `onpix` | Nombre de pixels allumés |
| `x-bar`, `y-bar` | Centroïdes horizontal / vertical |
| `x2bar`, `y2bar` | Moments du 2ᵉ ordre (variance) |
| `xybar` | Corrélation X×Y |
| `x2ybr`, `xy2br` | Moments croisés |
| `x-ege`, `y-ege` | Comptage de contours H / V |
| `xegvy`, `yegvx` | Corrélations contours-position |

---

## 💡 Points d'apprentissage clés

- **Pipeline sklearn** : encapsulation scaler + classifieur pour éviter le data leakage
- **Cross-validation stratifiée** : évaluation robuste sur données multi-classes
- **GridSearchCV** : optimisation systématique des hyperparamètres
- **PCA vs t-SNE** : exploration de la structure des données haute dimension
- **Analyse des erreurs** : interpréter les confusions pour guider les améliorations

---

## 📈 Pistes d'amélioration

- [ ] Features engineered (ratios, symétrie)
- [ ] MLP / réseau de neurones profond
- [ ] Stacking d'ensemble (SVM + RF + kNN)
- [ ] Analyse SHAP pour l'interprétabilité
- [ ] Interface web Flask/Streamlit

---

## 📚 Référence

Frey, P. W., & Slate, D. J. (1991). *Letter Recognition Using Holland-Style Adaptive Classifiers*. Machine Learning, 6(2), 161–182.

Dataset UCI : https://archive.ics.uci.edu/dataset/59/letter+recognition

---

## Licence

MIT — voir [LICENSE](LICENSE)
