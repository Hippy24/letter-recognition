"""
Letter Recognition — Pipeline d'entraînement complet
Dataset UCI : 20 000 exemples, 16 features, 26 classes (A–Z)

Usage:
    python src/train.py --model svm --output outputs/
    python src/train.py --model all --cv 5
"""

import argparse
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
import joblib


# CONFIG

COLUMNS = [
    "lettr", "x-box", "y-box", "width", "high", "onpix",
    "x-bar", "y-bar", "x2bar", "y2bar", "xybar",
    "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx"
]

MODELS = {
    "knn": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=5, n_jobs=-1))
    ]),
    "rf": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, max_depth=None,
                                        random_state=42, n_jobs=-1))
    ]),
    "svm": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(C=10, kernel="rbf", gamma="scale",
                    probability=True, random_state=42))
    ]),
}

PARAM_GRIDS = {
    "knn": {"clf__n_neighbors": [3, 5, 7, 11, 15]},
    "rf":  {"clf__n_estimators": [100, 200, 300], "clf__max_depth": [None, 20, 30]},
    "svm": {"clf__C": [1, 10, 50], "clf__gamma": ["scale", "auto", 0.01]},
}


# DATA

def load_data(path: str) -> tuple:
    """Charge le CSV UCI Letter Recognition."""
    df = pd.read_csv(path, header=None, names=COLUMNS)

    # Certaines versions ont déjà un header — on le détecte
    if df["lettr"].iloc[0] == "lettr":
        df = pd.read_csv(path, names=COLUMNS, skiprows=1)

    X = df.drop("lettr", axis=1).astype(float)
    y = df["lettr"].str.strip().str.upper()

    print(f"✓ Données chargées : {X.shape[0]} exemples, {X.shape[1]} features, {y.nunique()} classes")
    print(f"  Distribution : min={y.value_counts().min()}, max={y.value_counts().max()} ex/classe")
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """Split stratifié train/test."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print(f"✓ Split : {len(X_train)} train / {len(X_test)} test")
    return X_train, X_test, y_train, y_test


# TRAINING

def train_model(name: str, X_train, y_train, tune_hp=False, cv=5):
    """Entraîne un modèle, avec optionnel GridSearchCV."""
    model = MODELS[name]

    if tune_hp:
        print(f"\n⚙  GridSearchCV pour {name.upper()} (cv={cv})...")
        gs = GridSearchCV(model, PARAM_GRIDS[name], cv=cv,
                          scoring="accuracy", n_jobs=-1, verbose=1)
        t0 = time.time()
        gs.fit(X_train, y_train)
        print(f"   Best params : {gs.best_params_}")
        print(f"   Best CV acc : {gs.best_score_:.4f}  ({time.time()-t0:.1f}s)")
        return gs.best_estimator_

    print(f"\n🚀 Entraînement {name.upper()}...")
    t0 = time.time()
    model.fit(X_train, y_train)
    print(f"   Terminé en {time.time()-t0:.1f}s")

    cv_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                scoring="accuracy", n_jobs=-1)
    print(f"   CV accuracy ({cv}-fold) : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    return model


def evaluate_model(model, X_test, y_test, name: str):
    """Évalue le modèle et affiche les métriques clés."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n📊 Résultats — {name.upper()}")
    print(f"   Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"\n{classification_report(y_test, y_pred, zero_division=0)}")
    return acc, y_pred


# VISUALISATIONS

def plot_confusion_matrix(y_test, y_pred, name: str, output_dir: Path):
    """Matrice de confusion complète 26×26."""
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Purples",
                xticklabels=labels, yticklabels=labels, ax=ax,
                linewidths=0.3, linecolor="white")
    ax.set_xlabel("Prédiction", fontsize=12)
    ax.set_ylabel("Réalité", fontsize=12)
    ax.set_title(f"Matrice de confusion — {name.upper()}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = output_dir / f"confusion_matrix_{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Sauvegardé : {path}")


def plot_per_class_accuracy(y_test, y_pred, name: str, output_dir: Path):
    """Précision par lettre sous forme de barres."""
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    per_class = cm.diagonal() / cm.sum(axis=1)

    colors = ["#1D9E75" if v >= 0.98 else "#7F77DD" if v >= 0.95 else "#D85A30"
              for v in per_class]

    fig, ax = plt.subplots(figsize=(14, 4))
    bars = ax.bar(labels, per_class * 100, color=colors, width=0.7, edgecolor="white")
    ax.set_ylim(80, 102)
    ax.set_ylabel("Précision (%)")
    ax.set_title(f"Précision par lettre — {name.upper()}", fontweight="bold")
    ax.axhline(y=per_class.mean() * 100, color="#888", linestyle="--",
               linewidth=1, label=f"Moyenne : {per_class.mean()*100:.1f}%")
    ax.legend(fontsize=10)

    for bar, val in zip(bars, per_class):
        if val < 0.95:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{val*100:.0f}%", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    path = output_dir / f"per_class_accuracy_{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Sauvegardé : {path}")


def plot_model_comparison(results: dict, output_dir: Path):
    """Comparaison des modèles."""
    names = list(results.keys())
    accs = [results[n]["accuracy"] for n in names]
    times = [results[n]["train_time"] for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    bars = ax1.barh(names, [a * 100 for a in accs],
                    color=["#7F77DD", "#1D9E75", "#D85A30"], height=0.5)
    ax1.set_xlim(90, 100)
    ax1.set_xlabel("Accuracy (%)")
    ax1.set_title("Comparaison des modèles", fontweight="bold")
    for bar, acc in zip(bars, accs):
        ax1.text(bar.get_width() - 0.05, bar.get_y() + bar.get_height()/2,
                 f"{acc*100:.2f}%", ha="right", va="center",
                 color="white", fontweight="bold", fontsize=10)

    ax2.bar(names, times, color=["#7F77DD", "#1D9E75", "#D85A30"], width=0.5)
    ax2.set_ylabel("Temps (s)")
    ax2.set_title("Temps d'entraînement", fontweight="bold")
    for i, t in enumerate(times):
        ax2.text(i, t + 0.1, f"{t:.1f}s", ha="center", fontsize=9)

    plt.tight_layout()
    path = output_dir / "model_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Sauvegardé : {path}")


# MAIN

def main():
    parser = argparse.ArgumentParser(description="Letter Recognition ML Pipeline")
    parser.add_argument("--data",   default="data/letter-recognition.csv")
    parser.add_argument("--model",  default="svm", choices=["knn", "rf", "svm", "all"])
    parser.add_argument("--output", default="outputs/")
    parser.add_argument("--cv",     type=int, default=5)
    parser.add_argument("--tune",   action="store_true", help="GridSearchCV hyperparameter tuning")
    parser.add_argument("--save",   action="store_true", help="Sauvegarder les modèles (.pkl)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Chargement
    X, y = load_data(args.data)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Modèles à entraîner
    model_names = list(MODELS.keys()) if args.model == "all" else [args.model]
    results = {}

    for name in model_names:
        t0 = time.time()
        model = train_model(name, X_train, y_train, tune_hp=args.tune, cv=args.cv)
        train_time = time.time() - t0

        acc, y_pred = evaluate_model(model, X_test, y_test, name)

        results[name] = {"accuracy": acc, "train_time": train_time}

        print(f"\n📈 Génération des visualisations...")
        plot_confusion_matrix(y_test, y_pred, name, output_dir)
        plot_per_class_accuracy(y_test, y_pred, name, output_dir)

        if args.save:
            model_path = output_dir / f"model_{name}.pkl"
            joblib.dump(model, model_path)
            print(f"   Modèle sauvegardé : {model_path}")

    if len(results) > 1:
        plot_model_comparison(results, output_dir)

    print(f"\nTerminé. Résultats dans : {output_dir}/")


if __name__ == "__main__":
    main()
