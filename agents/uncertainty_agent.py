# agents/uncertainty_agent.py

import numpy as np
import pandas as pd


def add_forecast_uncertainty(
    model,
    forecast_df: pd.DataFrame,
    residuals: np.ndarray,
    forecast_col: str = "Forecast",
    n_simulations: int = 500,
    ci: float = 0.95,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Adds confidence intervals to ML forecasts using residual bootstrap.

    Parameters:
    - forecast_df : DataFrame containing forecasted values
    - residuals   : residuals from training predictions
    - forecast_col: column name in forecast_df to use
    - n_simulations : number of bootstrap simulations
    - ci          : confidence level (e.g., 0.95)
    - random_state: for reproducibility

    Returns:
    - forecast_df with 'Lower_CI' and 'Upper_CI' columns
    """

    np.random.seed(random_state)

    if forecast_col not in forecast_df.columns:
        raise ValueError(f"Column '{forecast_col}' not found in forecast_df")

    forecasts = forecast_df[forecast_col].astype(float).values
    residuals = residuals.astype(float)

    simulated_paths = []
    for _ in range(n_simulations):
        noise = np.random.choice(residuals, size=len(forecasts), replace=True)
        simulated_paths.append(forecasts + noise)

    simulated_paths = np.array(simulated_paths)

    lower_q = (1 - ci) / 2
    upper_q = 1 - lower_q

    forecast_df["Lower_CI"] = np.quantile(simulated_paths, lower_q, axis=0)
    forecast_df["Upper_CI"] = np.quantile(simulated_paths, upper_q, axis=0)

    return forecast_df
