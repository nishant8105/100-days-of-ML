# data_processing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_and_preprocess(csv_path: str = "./Food_Delivery_Time_Prediction.csv") -> pd.DataFrame:
    """Load the CSV and perform all feature engineering steps.

    Returns a DataFrame ready for modeling (features scaled, target added).
    """
    df = pd.read_csv(csv_path)

    # ---- Feature Engineering: parse coordinates & haversine distance -----
    def parse_coord(coord_str):
        return tuple(map(float, coord_str.strip("()").split(",")))

    df[["cust_lat", "cust_long"]] = df["Customer_Location"].apply(parse_coord).apply(pd.Series)
    df[["rest_lat", "rest_long"]] = df["Restaurant_Location"].apply(parse_coord).apply(pd.Series)
    df.drop(columns=["Customer_Location", "Restaurant_Location"], inplace=True)

    from math import radians, sin, cos, sqrt, atan2

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c

    df["distance_km"] = df.apply(
        lambda row: haversine(row["rest_lat"], row["rest_long"], row["cust_lat"], row["cust_long"]),
        axis=1,
    )

    # ---- Target Variable -----
    median_time = df["Delivery_Time"].median()
    df["Delivery_status"] = (df["Delivery_Time"] > median_time).astype(int)

    # ---- Encoding categorical columns -----
    cat_cols = df.select_dtypes(include="object").columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    # ---- Feature selection & scaling -----
    X = df.drop(["Delivery_Time", "Delivery_status"], axis=1)
    y = df["Delivery_status"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return pd.DataFrame(X_scaled, columns=X.columns), y
