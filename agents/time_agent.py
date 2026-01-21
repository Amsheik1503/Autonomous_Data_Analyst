# agents/time_agent.py

import pandas as pd
import numpy as np

def time_agent(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds time-based exogenous features for forecasting.
    Responsibility:
    - Detect datetime column
    - Clean invalid dates
    - Add calendar & cyclical features
    """

    df = df.copy()

    # --------------------------------------------------
    # Identify date column
    # --------------------------------------------------
    date_cols = [c for c in df.columns if "date" in c.lower()]

    if not date_cols:
        print("⚠️ Time Agent: No date column found — skipping time features")
        return df

    if len(date_cols) > 1:
        print(f"⚠️ Time Agent: Multiple date columns detected {date_cols}. Using '{date_cols[0]}'")

    date_col = date_cols[0]
    print(f"⏱ Time Agent: Using date column → {date_col}")

    df[date_col] = pd.to_datetime(
        df[date_col],
        format="mixed",
        dayfirst=True,
        errors="coerce"
    )

    # --------------------------------------------------
    # Drop invalid dates
    # --------------------------------------------------
    before = len(df)
    df = df.dropna(subset=[date_col])
    after = len(df)

    if before != after:
        print(f"⏱ Time Agent: Dropped {before - after} rows due to invalid dates")

    df = df.sort_values(date_col).reset_index(drop=True)

    # --------------------------------------------------
    # Calendar features
    # --------------------------------------------------
    df["day"] = df[date_col].dt.day
    df["month"] = df[date_col].dt.month
    df["year"] = df[date_col].dt.year
    df["dayofweek"] = df[date_col].dt.dayofweek
    df["weekofyear"] = df[date_col].dt.isocalendar().week.astype(int)
    df["quarter"] = df[date_col].dt.quarter
    df["dayofyear"] = df[date_col].dt.dayofyear

    # --------------------------------------------------
    # Cyclical encoding (seasonality)
    # --------------------------------------------------
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["dayofyear_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["dayofyear_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)

    print("⏱ Time Agent: Time-based and cyclical features added")

    return df
