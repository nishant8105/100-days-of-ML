# models.py
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
import matplotlib.pyplot as plt
import seaborn as sns

def train_and_evaluate(X, y):
    """Trains Naive Bayes, KNN and Decision Tree classifiers.

    Returns a dictionary of fitted models and evaluation metrics.
    """
    # split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = {}

    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    preds_nb = nb.predict(X_test)
    results["Naive Bayes"] = {
        "model": nb,
        "preds": preds_nb,
        "prob": nb.predict_proba(X_test)[:, 1],
    }

    # KNN with GridSearch
    knn_grid = GridSearchCV(
        KNeighborsClassifier(),
        {"n_neighbors": range(1, 21)},
        cv=5,
    )
    knn_grid.fit(X_train, y_train)
    preds_knn = knn_grid.best_estimator_.predict(X_test)
    results["KNN"] = {
        "model": knn_grid.best_estimator_,
        "preds": preds_knn,
        "prob": knn_grid.best_estimator_.predict_proba(X_test)[:, 1],
    }

    # Decision Tree with GridSearch
    dt_grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        {"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10]},
        cv=5,
    )
    dt_grid.fit(X_train, y_train)
    preds_dt = dt_grid.best_estimator_.predict(X_test)
    results["Decision Tree"] = {
        "model": dt_grid.best_estimator_,
        "preds": preds_dt,
        "prob": dt_grid.best_estimator_.predict_proba(X_test)[:, 1],
    }

    # Evaluation
    for name, r in results.items():
        acc = accuracy_score(y_test, r["preds"])
        cm = confusion_matrix(y_test, r["preds"])
        cr = classification_report(y_test, r["preds"])
        fpr, tpr, _ = roc_curve(y_test, r["prob"])
        roc_auc = auc(fpr, tpr)

        print(f"\n=== {name} ===")
        print("Accuracy:", acc)
        print("Confusion Matrix:\n", cm)
        print("Classification Report:\n", cr)
        print("AUC:", roc_auc)

    # Summary DataFrame
    summary = {
        "Model": [],
        "Accuracy": [],
    }
    for name, r in results.items():
        summary["Model"].append(name)
        summary["Accuracy"].append(accuracy_score(y_test, r["preds"]))

    print("\nOverall Accuracy Comparison")
    print(pd.DataFrame(summary))

    # Store metadata for summary generation
    results["_meta"] = {
        "y_test": y_test,
    }

    return results
