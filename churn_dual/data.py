import os
import pandas as pd
import numpy as np


def load_data(uploaded=None, default="Customer Churn new.csv"):
    if uploaded is not None:
        return pd.read_csv(uploaded)
    if os.path.exists(default):
        return pd.read_csv(default)
    return _sample()


def _sample(n=1500):
    np.random.seed(42)
    df = pd.DataFrame({
        "CreditScore": np.random.randint(300, 850, n),
        "Geography": np.random.choice(["France", "Spain", "Germany"], n),
        "Gender": np.random.choice(["Male", "Female"], n),
        "Age": np.random.randint(18, 80, n),
        "Tenure": np.random.randint(0, 11, n),
        "Balance": np.random.uniform(0, 200000, n).round(2),
        "EstimatedSalary": np.random.uniform(10000, 150000, n).round(2),
    })
    df["Exited"] = np.random.binomial(1, 0.2, n)
    return df
