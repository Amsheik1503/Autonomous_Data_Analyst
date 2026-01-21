import pandas as pd
import numpy as np

def outlier_agent(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Handles outliers using IQR capping for numeric targets only.
    """

    if target_col not in df.columns:
        raise ValueError(f"❌ Target column '{target_col}' not found")

    # --------------------------------------------------
    # Ensure numeric target
    # --------------------------------------------------
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print(f"⚠️ Outlier Agent skipped → '{target_col}' is non-numeric")
        return df

    print("\n🚨 Outlier Agent: Handling extreme values")

    series = df[target_col].dropna()

    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    before_min, before_max = series.min(), series.max()

    df[target_col] = np.clip(df[target_col], lower, upper)

    after_min, after_max = df[target_col].min(), df[target_col].max()

    print(
        f"Outlier capping applied | "
        f"Before → min: {before_min:.2f}, max: {before_max:.2f} | "
        f"After → min: {after_min:.2f}, max: {after_max:.2f}"
    )

    return df
