# main.py

"""Entry point that ties together data processing and model training.

Run with: python main.py
"""

import pandas as pd
from data_processing import load_and_preprocess
from models import train_and_evaluate

if __name__ == "__main__":
    X, y = load_and_preprocess()
    results = train_and_evaluate(X, y)
