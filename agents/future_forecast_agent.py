# agents/future_forecast_agent.py

import pandas as pd
import numpy as np


def future_forecast_agent(
    model,
    df: pd.DataFrame,
    target_col: str,
    forecast_days: int,
    feature_cols: list
):
    """
    Recursive multi-step forecasting agent for ML models only.
    Assumes lag and rolling features already exist.
    """

    print("\n🔮 Future Forecast Agent (ML)")

    df = df.copy().reset_index(drop=True)
    forecasts = []

    # --------------------------------------------------
    # Identify date column
    # --------------------------------------------------
    date_cols = [c for c in df.columns if "date" in c.lower()]
    date_col = date_cols[0] if date_cols else None

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        last_date = df[date_col].iloc[-1]

    # --------------------------------------------------
    # Identify lag features dynamically
    # --------------------------------------------------
    lag_features = sorted(
        [c for c in feature_cols if c.startswith(f"{target_col}_lag")],
        key=lambda x: int(x.split("lag")[-1])
    )

    if not lag_features:
        raise ValueError("❌ Future Forecast Agent: No lag features found")

    max_lag = len(lag_features)

    # --------------------------------------------------
    # Start from last known row
    # --------------------------------------------------
    last_row = df.iloc[-1].copy()

    for step in range(1, forecast_days + 1):

        # -------- ML input (2D, numeric only) --------
        X_next = (
            last_row[feature_cols]
            .astype(float)
            .values
            .reshape(1, -1)
        )

        y_pred = float(model.predict(X_next)[0])

        forecasts.append({
            "Day": step,
            "Forecast": y_pred
        })

        # -----------------------------
        # Update lag features
        # -----------------------------
        for lag in range(max_lag, 1, -1):
            last_row[f"{target_col}_lag{lag}"] = last_row[f"{target_col}_lag{lag - 1}"]

        last_row[f"{target_col}_lag1"] = y_pred

        # -----------------------------
        # Update rolling features
        # -----------------------------
        lag_values = [
            last_row[f"{target_col}_lag{i}"]
            for i in range(1, max_lag + 1)
        ]

        mean_col = f"{target_col}_roll_mean_{max_lag}"
        std_col = f"{target_col}_roll_std_{max_lag}"

        if mean_col in last_row:
            last_row[mean_col] = np.mean(lag_values)
        if std_col in last_row:
            last_row[std_col] = np.std(lag_values)

        # -----------------------------
        # Update time features
        # -----------------------------
        if date_col:
            next_date = last_date + pd.Timedelta(days=1)
            last_date = next_date
            last_row[date_col] = next_date

            if "day" in last_row:
                last_row["day"] = next_date.day
            if "month" in last_row:
                last_row["month"] = next_date.month
            if "year" in last_row:
                last_row["year"] = next_date.year
            if "dayofweek" in last_row:
                last_row["dayofweek"] = next_date.dayofweek
            if "weekofyear" in last_row:
                last_row["weekofyear"] = int(next_date.isocalendar().week)
            if "quarter" in last_row:
                last_row["quarter"] = next_date.quarter
            if "dayofyear" in last_row:
                last_row["dayofyear"] = next_date.dayofyear

            if "month_sin" in last_row:
                last_row["month_sin"] = np.sin(2 * np.pi * last_row["month"] / 12)
                last_row["month_cos"] = np.cos(2 * np.pi * last_row["month"] / 12)

            if "dayofyear_sin" in last_row:
                last_row["dayofyear_sin"] = np.sin(
                    2 * np.pi * last_row["dayofyear"] / 365
                )
                last_row["dayofyear_cos"] = np.cos(
                    2 * np.pi * last_row["dayofyear"] / 365
                )

    return pd.DataFrame(forecasts)
