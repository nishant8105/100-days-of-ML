# data_loading_and_eda.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(file_path="./Global_Pollution_Analysis.csv"):
    """Loads the dataset and performs initial Exploratory Data Analysis."""
    print("--- 1. Data Import and Preprocessing ---")
    
    # Load dataset
    try:
        df = pd.read_csv(file_path)
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")

    # Check dataset information
    shape = df.shape
    print(f"\nDataset Shape: Rows={shape[0]}, Columns={shape[1]}")
    
    # Basic info check (optional, depends on the user running it)
    # print("\nData Info:")
    # df.info() 

    # Handle missing values
    null_sum = df.isnull().sum().sum()
    print(f"\nTotal Null Values: {null_sum}")
    
    # Remove Duplicates
    duplicate_count = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicate_count}")

    # 1.6 Encode Categorical variables (Country)
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    df['Country_Encoded'] = encoder.fit_transform(df['Country'])
    print("\nSuccessfully encoded 'Country' into 'Country_Encoded'.")

    # 2. Exploratory Data Analysis (EDA)
    print("\n--- 2. Descriptive Statistics ---")
    print(df.describe().T)

    return df


def run_correlation_analysis(df:pd.DataFrame):
    """Performs and visualizes the correlation matrix."""
    print("\n--- 3. Correlation Analysis ---")
    # Select only numeric columns for correlation
    correlation_matrix = df.corr(numeric_only=True)

    plt.figure(figsize=(14, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix of Pollution Factors")
    plt.show()
    
    # Note: Detailed interpretations are left as comments/markdown for the user to keep in mind.
    print("\n--- EDA Complete. Correlation Heatmap displayed. ---")
    return df

if __name__ == "__main__":
    # This function returns the processed DataFrame, which subsequent scripts will use.
    data = run_eda()
    final_df = run_correlation_analysis(data)