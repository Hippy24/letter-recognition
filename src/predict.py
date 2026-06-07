"""
Letter Recognition — Inférence & API de prédiction
Charge un modèle sauvegardé et prédit sur de nouvelles données.

Usage:
    python src/predict.py --model outputs/model_svm.pkl --input sample.csv
    python src/predict.py --model outputs/model_svm.pkl --features "4,4,5,6,3,7,4,6,5,8,7,6,3,7,2,6"
"""

import argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path


FEATURE_NAMES = [
    "x-box", "y-box", "width", "high", "onpix",
    "x-bar", "y-bar", "x2bar", "y2bar", "xybar",
    "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx"
]


def load_model(model_path: str):
    """Charge un modèle sklearn depuis un fichier .pkl."""
    model = joblib.load(model_path)
    print(f"✓ Modèle chargé : {model_path}")
    return model


def predict_from_features(model, feature_string: str) -> dict:
    """
    Prédit une lettre à partir d'une chaîne de 16 valeurs séparées par des virgules.

    Args:
        model: modèle sklearn chargé
        feature_string: ex. "4,4,5,6,3,7,4,6,5,8,7,6,3,7,2,6"

    Returns:
        dict avec la lettre prédite et le top-5
    """
    values = [float(v.strip()) for v in feature_string.split(",")]
    if len(values) != 16:
        raise ValueError(f"16 features attendues, {len(values)} reçues")

    X = np.array(values).reshape(1, -1)
    letter = model.predict(X)[0]

    result = {"prediction": letter, "features": dict(zip(FEATURE_NAMES, values))}

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        classes = model.classes_
        top5_idx = np.argsort(proba)[::-1][:5]
        result["top5"] = [
            {"letter": classes[i], "confidence": round(float(proba[i]), 4)}
            for i in top5_idx
        ]
        result["confidence"] = round(float(proba[top5_idx[0]]), 4)

    return result


def predict_from_csv(model, csv_path: str) -> pd.DataFrame:
    """Prédit sur un CSV de features (sans colonne lettre)."""
    df = pd.read_csv(csv_path)
    if "lettr" in df.columns:
        df = df.drop("lettr", axis=1)

    X = df[FEATURE_NAMES].values
    predictions = model.predict(X)

    result_df = df.copy()
    result_df["predicted_letter"] = predictions

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        result_df["confidence"] = proba.max(axis=1).round(4)

    return result_df


def print_prediction(result: dict):
    """Affiche joliment le résultat d'une prédiction."""
    print(f"\n{'='*40}")
    print(f"  Prédiction : {result['prediction']}")
    if "confidence" in result:
        print(f"  Confiance  : {result['confidence']*100:.1f}%")
    if "top5" in result:
        print("\n  Top 5 :")
        for rank, item in enumerate(result["top5"], 1):
            bar = "█" * int(item["confidence"] * 30)
            print(f"    {rank}. {item['letter']}  {bar:<30} {item['confidence']*100:.1f}%")
    print(f"{'='*40}")


def main():
    parser = argparse.ArgumentParser(description="Letter Recognition — Inférence")
    parser.add_argument("--model",    required=True, help="Chemin vers le .pkl")
    parser.add_argument("--input",    help="CSV de features à prédire")
    parser.add_argument("--features", help="16 valeurs séparées par des virgules")
    parser.add_argument("--output",   help="Sauvegarder les prédictions dans un CSV")
    args = parser.parse_args()

    model = load_model(args.model)

    if args.features:
        result = predict_from_features(model, args.features)
        print_prediction(result)

    elif args.input:
        df_pred = predict_from_csv(model, args.input)
        print(f"\n✓ {len(df_pred)} prédictions effectuées")
        print(df_pred.head(10).to_string())

        if args.output:
            df_pred.to_csv(args.output, index=False)
            print(f"\n✓ Résultats sauvegardés : {args.output}")

    else:
        # Mode interactif
        print("\nMode interactif — entrez 16 valeurs séparées par des virgules")
        print(f"Features attendues : {', '.join(FEATURE_NAMES)}")
        print("(Ctrl+C pour quitter)\n")

        while True:
            try:
                raw = input("Features > ").strip()
                if raw:
                    result = predict_from_features(model, raw)
                    print_prediction(result)
            except (KeyboardInterrupt, EOFError):
                print("\nAu revoir !")
                break
            except ValueError as e:
                print(f"Erreur : {e}")


if __name__ == "__main__":
    main()
