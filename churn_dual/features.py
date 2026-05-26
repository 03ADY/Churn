"""Feature prep — drop IDs and align live-scoring columns."""

import pandas as pd

DROP_COLS = ("RowNumber", "CustomerId", "Surname", "Customerid", "customer_id")


def prepare_churn_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in DROP_COLS:
        if col in out.columns:
            out = out.drop(columns=col)
    return out


LIVE_FEATURE_DEFAULTS = {
    "CreditScore": 650,
    "Geography": "France",
    "Gender": "Male",
    "Age": 40,
    "Tenure": 3,
    "Balance": 50000.0,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 75000.0,
}
