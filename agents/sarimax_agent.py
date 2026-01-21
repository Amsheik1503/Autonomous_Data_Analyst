import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def sarimax_agent(df, target_col, forecast_days=30, exog_cols=None, order=(1,1,1), seasonal_order=(0,1,1,12)):
    """
    SARIMAX Forecasting Agent

    Parameters:
    - df           : DataFrame with historical data
    - target_col   : Column to forecast
    - forecast_days: Number of days to forecast
    - exog_cols    : List of exogenous variable columns (optional)
    - order        : SARIMAX order (p,d,q)
    - seasonal_order: Seasonal order (P,D,Q,s)
    
    Returns:
    - forecast_df  : DataFrame with forecast for `forecast_days`
    """

    df = df.copy().reset_index(drop=True)

    # Prepare endogenous and exogenous data
    y = df[target_col]
    if exog_cols:
       exog = df[exog_cols].copy()
    else:
         exog = None


    # Fit SARIMAX
    model = SARIMAX(
        endog=y,
        exog=exog,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    model_fit = model.fit(disp=False)

    # Prepare exog for forecast if exists
    if exog_cols:
        last_exog = df[exog_cols].iloc[-forecast_days:].copy()
        # If forecast_days > available exog, repeat last row
        if len(last_exog) < forecast_days:
            repeats = forecast_days - len(last_exog)
            last_row = last_exog.iloc[[-1]]
            last_exog = pd.concat([last_exog, pd.concat([last_row]*repeats, ignore_index=True)], ignore_index=True)
        exog_forecast = last_exog
    else:
        exog_forecast = None

    # Forecast
    y_forecast = model_fit.get_forecast(steps=forecast_days, exog=exog_forecast)
    forecast_values = y_forecast.predicted_mean.values

    # Confidence intervals
    conf_int = y_forecast.conf_int(alpha=0.05) if hasattr(y_forecast, "conf_int") else None

    forecast_df = pd.DataFrame({
        "Day": range(1, forecast_days + 1),
        "Forecast": forecast_values
    })

    if conf_int is not None:
        forecast_df["Lower_CI"] = conf_int.iloc[:,0].values
        forecast_df["Upper_CI"] = conf_int.iloc[:,1].values

    return forecast_df
