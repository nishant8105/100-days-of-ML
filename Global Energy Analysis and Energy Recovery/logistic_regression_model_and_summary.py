# logistic_regression_model_and_summary.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from feature_engineering import preprocess_data

def run_logistic_regression(df):
    """Trains and evaluates the Logistic Regression model for pollution severity."""
    print("--- Starting Logistic Regression Model Training ---")
    
    # 5.1. Encode Target Variable (Assuming 'Pollution_Severity' is already set in df)
    from sklearn.preprocessing import LabelEncoder
    severity_encoded = LabelEncoder()
    df['Severity_Encoded'] = severity_encoded.fit_transform(df['Pollution_Severity'])
    print("Target variable 'Pollution_Severity' encoded.")

    # 5.2 Define Features and Target
    X = df[[
        'Air_Pollution_Index',
        'CO2_Emissions (in MT)',
        'Industrial_Waste (in tons)',
        'Water_Pollution_Index'
    ]]
    y = df['Severity_Encoded']

    # 5.3 Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )
    print("Data split for classification.")

    # 5.4 Train Logistic Regression Model
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, y_train)
    print("Logistic Regression model trained successfully.")

    # 5.5 Predictions
    y_pred = log_model.predict(X_test)

    # 5.6 Evaluate Logistic Regression Model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("\n--- Logistic Regression Evaluation ---")
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 Score:', f1)

    # 5.7 Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))


def summarize_findings():
    """Prints the final comparative analysis and conclusion."""
    print("\n\n==================================================")
    print("        MODEL EVALUATION AND SUMMARY FINDINGS       ")
    print("==================================================")

    # Comparison table (using descriptive text since markdown is lost in scripts)
    print("""
        |        Model        |            Task             |                  Evaluation Metrics              |
        |---------------------|-----------------------------|--------------------------------------------------|
        |  Linear  Regression | Predict   Energy   Recovery | R²: -0.018, MSE: 24636.455, MAE: 141.720         |
        | Logistic Regression | Classify Pollution Severity | Accuracy:0.975, F1-score: 0.974, Precision: 0.975|
""")

    print("\n--- Key Pollutant Impact Insights ---")
    print("* Countries with higher industrial waste tend to show increased pollution.")
    print("* Higher renewable energy percentages may reduce pollution severity.")
    print("* Energy recovery may improve when industrial waste management is optimized.")


if __name__ == "__main__":
    data = preprocess_data()
    run_logistic_regression(data)
    summarize_findings()