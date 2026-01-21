# agents/feature_agent.py

import pandas as pd
from typing import List, Tuple

def feature_agent(
    df: pd.DataFrame,
    target_col: str,
    lags: List[int] = [1, 3, 7, 14, 30],
    rolling_windows: List[int] = [7, 14]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, List[str]]:
    """
    Creates time-series safe lag and rolling features.

    Responsibilities:
    - Generate past-only lag features
    - Generate rolling statistics (shifted)
    - Prevent target leakage
    """

    df = df.copy()

    if target_col not in df.columns:
        raise ValueError(f"Feature Agent Error: '{target_col}' not found")

    # --------------------------------------------------
    # Ensure time order
    # --------------------------------------------------
    date_cols = [c for c in df.columns if "date" in c.lower()]
    if date_cols:
        df = df.sort_values(date_cols[0]).reset_index(drop=True)

    # --------------------------------------------------
    # Lag features (PAST ONLY)
    # --------------------------------------------------
    for lag in lags:
        df[f"{target_col}_lag{lag}"] = df[target_col].shift(lag)

    # --------------------------------------------------
    # Rolling statistics (shifted to avoid leakage)
    # --------------------------------------------------
    for window in rolling_windows:
        df[f"{target_col}_roll_mean_{window}"] = (
            df[target_col].shift(1).rolling(window=window).mean()
        )
        df[f"{target_col}_roll_std_{window}"] = (
            df[target_col].shift(1).rolling(window=window).std()
        )

    # --------------------------------------------------
    # Drop rows with NaNs from lag/rolling creation
    # --------------------------------------------------
    df = df.dropna().reset_index(drop=True)

    # --------------------------------------------------
    # Select ML-safe numeric features
    # --------------------------------------------------
    exclude_cols = {target_col}
    ml_features = [
        c for c in df.columns
        if c not in exclude_cols
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    X = df[ml_features]
    y = df[target_col]

    print(f"\n🧩 Feature Agent: Created {len(ml_features)} features")
    print("🧩 Feature Agent: Feature list:")
    print(ml_features)

    return df, X, y, ml_features
