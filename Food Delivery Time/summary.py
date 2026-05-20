"""Summary script for the Food Delivery Time model training pipeline.

This module trains the baseline models defined in models.py, then computes
and prints a concise evaluation summary for each model.
"""

import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

from data_processing import load_and_preprocess
from models import train_and_evaluate


def build_summary(results: dict) -> pd.DataFrame:
    """Build a summary DataFrame from trained model results."""
    meta = results.get("_meta", {})
    y_test = meta.get("y_test")
    if y_test is None:
        raise ValueError("Results dict does not contain test labels under '_meta'.")

    rows = []
    for name, result in results.items():
        if name.startswith("_"):
            continue

        preds = result["preds"]
        prob = result["prob"]
        acc = accuracy_score(y_test, preds)
        roc_auc = roc_auc_score(y_test, prob)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

        rows.append(
            {
                "Model": name,
                "Accuracy": acc,
                "AUC": roc_auc,
                "True Positives": int(tp),
                "False Positives": int(fp),
                "True Negatives": int(tn),
                "False Negatives": int(fn),
            }
        )

    return pd.DataFrame(rows)


def print_summary(summary: pd.DataFrame) -> None:
    """Print the model evaluation summary to the console."""
    print("\n=== Model Summary ===")
    print(summary.to_string(index=False))


def save_summary(summary: pd.DataFrame, path: str = "model_summary.csv") -> None:
    """Save the model summary to a CSV file."""
    summary.to_csv(path, index=False)
    print(f"Saved summary to {path}")


if __name__ == "__main__":
    X, y = load_and_preprocess()
    results = train_and_evaluate(X, y)
    summary_df = build_summary(results)
    print_summary(summary_df)
    save_summary(summary_df)
