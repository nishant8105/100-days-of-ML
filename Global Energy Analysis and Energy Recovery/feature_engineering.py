# feature_engineering.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from data_loading_and_eda import run_eda

def preprocess_data(df : pd.DataFrame = run_eda()):
    """Performs Feature Engineering and scaling necessary for modeling."""
    print("--- Starting Feature Engineering ---")
    
    # 3.1 Create Yearly Trend Features (Example)
    year_trend = df.groupby('Year')['Energy_Recovered (in GWh)'].mean()
    print("\nAverage Energy Recovered over the Years:")
    print(year_trend)

    # 3.2 Calculate Energy Consumption per capita
    df['Calculated_Energy_Per_Capita'] = (
        df['Energy_Recovered (in GWh)'] /
        df['Population (in millions)']
    )
    print("\n'Calculated_Energy_Per_Capita' created.")

    # 3.3 Create Pollution Severity Category
    def pollution_category(x):
        if pd.isna(x): return 'Low' # Handle potential NaNs if they were present
        elif x < 100:
            return 'Low'
        elif x < 200:
            return 'Medium'
        else:
            return 'High'

    # Using Air_Pollution_Index for severity calculation as in the original notebook
    df['Pollution_Severity'] = df['Air_Pollution_Index'].apply(pollution_category)
    print("'Pollution_Severity' column created based on Air_Pollution_Index.")


    # 3.4 Feature Scaling (For Pollution Indices)
    pollution_cols = [
        'Air_Pollution_Index', 
        'Water_Pollution_Index', 
        'Soil_Pollution_Index'
    ]

    scaler = StandardScaler()
    df[pollution_cols] = scaler.fit_transform(df[pollution_cols])
    print("Pollution indices scaled using StandardScaler.")


    # Re-calculate Severity after scaling (as per original notebook logic)
    # Note: This is a simplified re-application of the rule based on median, assuming Air_Pollution_Index is used.
    median_air = df['Air_Pollution_Index'].median()
    df['Pollution_Severity'] = np.where(
        df['Air_Pollution_Index'] > median_air,
        'High',
        'Low'
    )
    print("Final 'Pollution_Severity' column set based on scaled Air Pollution Index.")

    return df


if __name__ == "__main__":
    preprocess_data()