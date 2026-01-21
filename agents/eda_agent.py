# agents/eda_agent.py

import pandas as pd

def eda_agent(data_path: str) -> pd.DataFrame:
    """
    Performs basic exploratory data analysis.
    Responsibility:
    - Load dataset
    - Inspect schema
    - Check missing values
    - Print summary statistics
    """

    print("\n🔍 EDA Agent: Starting exploratory analysis")

    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ EDA Agent Error: File not found → {data_path}")
    except Exception as e:
        raise RuntimeError(f"❌ EDA Agent Error while loading data: {e}")

    if df.empty:
        raise ValueError("❌ EDA Agent Error: Dataset is empty")

    print(f"\n📊 Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    print("\n📌 Schema:")
    print(df.dtypes)

    print("\n📌 Missing Values:")
    print(df.isna().sum())

    print("\n📌 Basic Statistics:")
    print(df.describe(include="all"))

    return df
