# planner_agent.py
import pandas as pd
import numpy as np

from agents.eda_agent import eda_agent
from agents.time_agent import time_agent
from agents.outlier_agent import outlier_agent
from agents.feature_agent import feature_agent
from agents.model_comparison_agent import model_comparison_agent
from agents.dl_agent import dl_agent
from agents.ml_dl_comparison_agent import ml_dl_comparison_agent
from agents.future_forecast_agent import future_forecast_agent
from agents.uncertainty_agent import add_forecast_uncertainty
from agents.sarimax_agent import sarimax_agent
from agents.weather_insight_agent import weather_insight_agent


# --------------------------------------------------
# Helper checks
# --------------------------------------------------
def has_datetime(df: pd.DataFrame) -> bool:
    return any("date" in c.lower() for c in df.columns)

def is_time_series(df: pd.DataFrame) -> bool:
    return has_datetime(df) and len(df) >= 60


# --------------------------------------------------
# PLANNER AGENT
# --------------------------------------------------
def planner_agent(
    data_path: str,
    target_col: str,
    forecast_days: int = 30,
    run_sarimax: bool = True
) -> dict:


    print("\n🚀 AUTONOMOUS DATA ANALYST STARTED")

    # ==================================================
    # 1️⃣ EDA
    # ==================================================
    df = eda_agent(data_path)
    if df.empty:
        raise ValueError("❌ Empty dataset after EDA")

    # ==================================================
    # 2️⃣ Time Features
    # ==================================================
    if is_time_series(df):
        print("\n⏱ Time-series detected → Time Agent")
        df = time_agent(df)
    else:
        print("⏭ No time-series structure detected")

    # --------------------------------------------------
    # Validate target column
    # --------------------------------------------------
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        raise ValueError(
            f"❌ Target column '{target_col}' must be numeric for forecasting"
        )


    # ==================================================
    # 3️⃣ Outlier Handling
    # ==================================================
    df = outlier_agent(df, target_col)

    # ==================================================
    # 4️⃣ Feature Engineering
    # ==================================================
    df_fe, X, y, ml_features = feature_agent(df, target_col)

    # Remove date columns
    ml_features = [c for c in ml_features if "date" not in c.lower()]
    X = X[ml_features].dropna()
    y = y.loc[X.index]

    print("\n✅ ML Features:", X.columns.tolist())

    # ==================================================
    # 5️⃣ ML Model Comparison
    # ==================================================
    print("\n🤖 Running ML models...")
    best_ml_name, best_ml_model, ml_results, train_residuals = model_comparison_agent(X, y)

    # ==================================================
    # 6️⃣ DL Model Comparison
    # ==================================================
    print("\n🧠 Running DL models...")
    best_dl_name, best_dl_model, dl_results = dl_agent(df, target_col)

    # ==================================================
    # 7️⃣ ML vs DL Decision
    # ==================================================
    decision = ml_dl_comparison_agent(
        best_ml_model=best_ml_model,
        best_ml_rmse=ml_results.loc[best_ml_name, "RMSE"],
        best_dl_model=best_dl_model,
        best_dl_rmse=dl_results.loc[best_dl_name, "RMSE"]
    )

    final_model = decision["selected_model"]
    final_type = decision["selected_type"]

    print(f"\n🏆 FINAL MODEL SELECTED → {final_type}")

    # ==================================================
    # 8️⃣ SARIMAX Baseline (always computed)
    # ==================================================
    print("\n📉 Running SARIMAX baseline...")
    sarimax_features = [
        "Temp_Max", "Temp_Min", "Precipitation_Sum",
        "Windspeed_Max", "Windgusts_Max", "Sunshine_Duration",
        "month_sin", "month_cos", "dayofyear_sin", "dayofyear_cos"
    ]

    # ==================================================
    # 8️⃣ SARIMAX Baseline (Robust)
    # ==================================================
    print("\n📉 Running SARIMAX baseline...")

    available_exog = [c for c in sarimax_features if c in df.columns]

    if available_exog:
        print(f"📌 SARIMAX using exogenous variables: {available_exog}")
    else:
        print("⚠️ SARIMAX running without exogenous variables")

    sarimax_forecast = sarimax_agent(
        df=df,
        target_col=target_col,
        exog_cols=available_exog if available_exog else None,
        forecast_days=forecast_days
        )


    # ==================================================
    # 9️⃣ Forecasting (CORRECT LOGIC)
    # ==================================================
    print(f"\n📈 Forecasting next {forecast_days} days...")

    if final_type == "ML":
        forecast = future_forecast_agent(
            model=final_model,
            df=df_fe,
            target_col=target_col,
            forecast_days=forecast_days,
            feature_cols=ml_features
        )

        forecast = add_forecast_uncertainty(
            model=final_model,
            forecast_df=forecast,
            residuals=train_residuals,
            n_simulations=500,
            ci=0.95
        )

    elif final_type == "DL":
        print("⚠️ DL selected → Using SARIMAX forecast")
        forecast = sarimax_forecast

    else:
        raise ValueError("Unknown model type selected")

    # ==================================================
    # 🔟 Weather / Domain Intelligence
    # ==================================================
    insight = weather_insight_agent(forecast)

    print("\n🌡 Weather Insight:")
    print(insight)

    print("\n✅ AUTONOMOUS PIPELINE COMPLETED SUCCESSFULLY")

    return {
        "final_model_type": final_type,
        "selected_model": final_model,
        "ml_results": ml_results,
        "dl_results": dl_results,
        "forecast": forecast,
        "sarimax_forecast": sarimax_forecast,
        "insight": insight
    }
