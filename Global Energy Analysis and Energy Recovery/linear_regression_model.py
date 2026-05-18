# linear_regression_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from feature_engineering import preprocess_data

def run_linear_regression(df):
    """Trains and evaluates the Linear Regression model to predict Energy Recovery."""
    print("--- Starting Linear Regression Model Training ---")
    
    # 4.1. select features and target
    feature_cols = [
        'Air_Pollution_Index',
        'Water_Pollution_Index',
        'Soil_Pollution_Index',
        'Industrial_Waste (in tons)',
        'CO2_Emissions (in MT)',
        'Renewable_Energy (%)'
    ]
    
    X = df[feature_cols]
    y = df['Energy_Recovered (in GWh)']

    # 4.2 Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )
    print("Data split into training and testing sets.")

    # 4.3 Train Linear Regression Model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    print("Linear Regression model trained successfully.")

    # 4.4 Predictions
    y_pred = lr_model.predict(X_test)

    # 4.5. Evaluate Model
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Linear Regression Evaluation ---")
    print('Mean Squared Error (MSE):', mse)
    print('Mean Absolute Error (MAE):', mae)
    print('R2 Score:', r2)

    # 4.6. Visualize Predictions
    indices = np.arange(len(y_test))
    plt.figure(figsize=(10, 7))
    plt.scatter(indices, y_test, c='red', label="Actual Values", alpha=0.7)
    plt.scatter(indices, y_pred, c='green', label="Predicted Values", alpha=0.7)
    plt.xlabel("Data Point Index")
    plt.ylabel("Energy recovery value (GWh)")
    plt.title("Actual vs Predicted Energy Recovery")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    data = preprocess_data()
    run_linear_regression(data)