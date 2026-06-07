"""
Letter Recognition — Analyse exploratoire & clustering
PCA + t-SNE pour visualiser la structure des données.

Usage:
    python src/clustering.py --data data/letter-recognition.csv
    python src/clustering.py --method tsne --perplexity 40
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ─── CONFIG ──────────────────────────────────────────────────────────────────

COLUMNS = [
    "lettr", "x-box", "y-box", "width", "high", "onpix",
    "x-bar", "y-bar", "x2bar", "y2bar", "xybar",
    "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx"
]

# Palette 26 couleurs distinctes
PALETTE = [
    "#e6194b","#3cb44b","#ffe119","#4363d8","#f58231","#911eb4","#42d4f4",
    "#f032e6","#bfef45","#fabed4","#469990","#dcbeff","#9A6324","#fffac8",
    "#800000","#aaffc3","#808000","#ffd8b1","#000075","#a9a9a9","#ffffff",
    "#000000","#4169E1","#DC143C","#00CED1","#FF8C00"
]

LETTER_TO_COLOR = {chr(65+i): PALETTE[i] for i in range(26)}


# ─── DATA ────────────────────────────────────────────────────────────────────

def load_and_preprocess(path: str, sample_n: int = 5000):
    """Charge + normalise les données, prend un échantillon pour t-SNE."""
    df = pd.read_csv(path, header=None, names=COLUMNS)
    if df["lettr"].iloc[0] == "lettr":
        df = pd.read_csv(path, names=COLUMNS, skiprows=1)

    df["lettr"] = df["lettr"].str.strip().str.upper()
    X = df.drop("lettr", axis=1).astype(float)
    y = df["lettr"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Échantillon stratifié pour t-SNE (lourd sur 20k pts)
    if sample_n and len(X_scaled) > sample_n:
        idx = (df.groupby("lettr").apply(lambda g: g.sample(
            min(len(g), sample_n // 26), random_state=42
        )).index.get_level_values(1))
        X_sample = X_scaled[idx]
        y_sample = y.iloc[idx].reset_index(drop=True)
    else:
        X_sample, y_sample = X_scaled, y.reset_index(drop=True)

    print(f"✓ Données normalisées : {X_sample.shape[0]} exemples sélectionnés")
    return X_scaled, y, X_sample, y_sample, df


# ─── PCA ─────────────────────────────────────────────────────────────────────

def run_pca(X_scaled, y, output_dir: Path):
    """PCA complète : variance expliquée + projection 2D/3D."""
    print("\n🔵 PCA en cours...")
    pca_full = PCA().fit(X_scaled)
    explained = pca_full.explained_variance_ratio_

    # Plot 1 — Variance expliquée
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.bar(range(1, 17), explained * 100, color="#7F77DD", alpha=0.85, edgecolor="white")
    ax1.set_xlabel("Composante principale")
    ax1.set_ylabel("Variance expliquée (%)")
    ax1.set_title("Variance expliquée par composante", fontweight="bold")
    ax1.set_xticks(range(1, 17))

    cumvar = np.cumsum(explained) * 100
    ax2.plot(range(1, 17), cumvar, "o-", color="#7F77DD", linewidth=2, markersize=6)
    ax2.axhline(y=95, color="#D85A30", linestyle="--", label="95%")
    ax2.axhline(y=99, color="#1D9E75", linestyle="--", label="99%")
    ax2.fill_between(range(1, 17), cumvar, alpha=0.15, color="#7F77DD")
    ax2.set_xlabel("Nombre de composantes")
    ax2.set_ylabel("Variance cumulée (%)")
    ax2.set_title("Variance cumulée", fontweight="bold")
    ax2.legend()
    ax2.set_ylim(0, 105)

    n95 = np.argmax(cumvar >= 95) + 1
    n99 = np.argmax(cumvar >= 99) + 1
    print(f"   {n95} composantes → 95% de variance")
    print(f"   {n99} composantes → 99% de variance")

    plt.tight_layout()
    fig.savefig(output_dir / "pca_variance.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 2 — Projection 2D (PC1 vs PC2)
    pca2 = PCA(n_components=2)
    X_pca = pca2.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(12, 9))
    for letter in sorted(y.unique()):
        mask = y == letter
        ax.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=LETTER_TO_COLOR[letter], label=letter,
            alpha=0.35, s=8, linewidths=0
        )

    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}% variance)", fontsize=11)
    ax.set_title("Projection PCA 2D — 26 lettres", fontsize=14, fontweight="bold")

    legend = ax.legend(
        handles=[mpatches.Patch(color=LETTER_TO_COLOR[l], label=l)
                 for l in sorted(y.unique())],
        loc="upper right", ncol=3, fontsize=8,
        framealpha=0.9, title="Lettre"
    )
    plt.tight_layout()
    fig.savefig(output_dir / "pca_2d.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Sauvegardé : pca_variance.png, pca_2d.png")

    return X_pca, pca_full


# ─── t-SNE ───────────────────────────────────────────────────────────────────

def run_tsne(X_sample, y_sample, perplexity=40, output_dir: Path = None):
    """t-SNE 2D sur l'échantillon, avec annotation des centroïdes."""
    print(f"\n🟣 t-SNE (perplexity={perplexity})... [peut prendre 1-3 min]")

    tsne = TSNE(n_components=2, perplexity=perplexity,
                n_iter=1000, random_state=42, verbose=1)
    X_tsne = tsne.fit_transform(X_sample)

    fig, ax = plt.subplots(figsize=(13, 10))
    for letter in sorted(y_sample.unique()):
        mask = y_sample == letter
        ax.scatter(
            X_tsne[mask, 0], X_tsne[mask, 1],
            c=LETTER_TO_COLOR[letter],
            alpha=0.55, s=12, linewidths=0, label=letter
        )
        # Annoter le centroïde
        cx, cy = X_tsne[mask].mean(axis=0)
        ax.text(cx, cy, letter, fontsize=9, fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="none", alpha=0.75))

    ax.set_title(f"t-SNE 2D — {len(X_sample)} exemples, perplexity={perplexity}",
                 fontsize=14, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(handles=[mpatches.Patch(color=LETTER_TO_COLOR[l], label=l)
                        for l in sorted(y_sample.unique())],
              loc="lower right", ncol=4, fontsize=7, framealpha=0.9)

    plt.tight_layout()
    path = output_dir / "tsne_2d.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Sauvegardé : {path}")
    return X_tsne


# ─── FEATURE ANALYSIS ────────────────────────────────────────────────────────

def plot_feature_distributions(df, output_dir: Path):
    """Distributions des 16 features, par lettre sélectionnée."""
    features = [c for c in df.columns if c != "lettr"]
    highlight = ["A", "E", "I", "O", "U", "M", "W", "X"]

    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        ax = axes[i]
        for letter in highlight:
            vals = df[df["lettr"] == letter][feat]
            ax.hist(vals, bins=15, alpha=0.5,
                    label=letter, color=LETTER_TO_COLOR[letter], density=True)
        ax.set_title(feat, fontsize=10, fontweight="bold")
        ax.set_ylabel("Densité", fontsize=8)
        ax.tick_params(labelsize=7)

    axes[0].legend(fontsize=7, ncol=4)
    fig.suptitle("Distribution des features par lettre (sélection)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = output_dir / "feature_distributions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Sauvegardé : {path}")


def plot_correlation_heatmap(df, output_dir: Path):
    """Heatmap de corrélation entre les 16 features."""
    features = [c for c in df.columns if c != "lettr"]
    corr = df[features].astype(float).corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns_import = True
    try:
        import seaborn as sns
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdPu",
                    center=0, ax=ax, linewidths=0.5, annot_kws={"size": 8})
    except ImportError:
        im = ax.imshow(corr, cmap="RdPu", vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(features)))
        ax.set_yticks(range(len(features)))
        ax.set_xticklabels(features, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(features, fontsize=9)

    ax.set_title("Corrélation entre les features", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = output_dir / "correlation_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Sauvegardé : {path}")


def plot_kmeans_elbow(X_scaled, output_dir: Path):
    """Elbow method + silhouette pour choisir k dans KMeans."""
    print("\n🔶 KMeans clustering (elbow method)...")
    ks = range(5, 35, 5)
    inertias, silhouettes = [], []

    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=100)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels, sample_size=2000))
        print(f"   k={k:2d}  inertia={km.inertia_:>10.0f}  silhouette={silhouettes[-1]:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(ks, inertias, "o-", color="#7F77DD", linewidth=2, markersize=7)
    ax1.set_xlabel("Nombre de clusters k")
    ax1.set_ylabel("Inertie")
    ax1.set_title("Elbow method", fontweight="bold")
    ax1.axvline(x=26, color="#D85A30", linestyle="--", label="k=26 (référence)")
    ax1.legend()

    ax2.plot(ks, silhouettes, "o-", color="#1D9E75", linewidth=2, markersize=7)
    ax2.set_xlabel("Nombre de clusters k")
    ax2.set_ylabel("Silhouette score")
    ax2.set_title("Score silhouette", fontweight="bold")

    plt.tight_layout()
    path = output_dir / "kmeans_elbow.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Sauvegardé : {path}")


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Clustering & dimensionality reduction")
    parser.add_argument("--data",       default="data/letter-recognition.csv")
    parser.add_argument("--output",     default="outputs/")
    parser.add_argument("--method",     default="all", choices=["pca", "tsne", "cluster", "features", "all"])
    parser.add_argument("--sample",     type=int, default=5000, help="Nb exemples pour t-SNE")
    parser.add_argument("--perplexity", type=int, default=40)
    args = parser.parse_args()

    import seaborn  # vérifie dispo
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_scaled, y, X_sample, y_sample, df = load_and_preprocess(args.data, args.sample)

    if args.method in ("pca", "all"):
        run_pca(X_scaled, y, output_dir)

    if args.method in ("tsne", "all"):
        run_tsne(X_sample, y_sample, args.perplexity, output_dir)

    if args.method in ("features", "all"):
        print("\n📐 Distributions & corrélations...")
        plot_feature_distributions(df, output_dir)
        plot_correlation_heatmap(df, output_dir)

    if args.method in ("cluster", "all"):
        plot_kmeans_elbow(X_scaled, output_dir)

    print(f"\n✅ Terminé. Tous les graphiques dans : {output_dir}/")


if __name__ == "__main__":
    main()
