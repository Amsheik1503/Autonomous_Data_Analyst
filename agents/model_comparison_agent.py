# agents/model_comparison_agent.py

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.base import clone

# Optional XGBoost
try:
    from xgboost import XGBRegressor
    xgb_available = True
except ImportError:
    xgb_available = False


def model_comparison_agent(X: pd.DataFrame, y: pd.Series, n_splits: int = 5):
    """
    Compare multiple ML models on time-series data.
    Returns: best_model_name, best_model, results_df, train_residuals
    """
    print("\n🤖 Model Comparison Agent: Comparing ML models...")

    # Time-series train/test split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Models
    models = {
        "Linear": LinearRegression(),
        "RandomForest": RandomForestRegressor(random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42)
    }

    params = {
        "Linear": {},
        "RandomForest": {"n_estimators": [100, 200], "max_depth": [5, 10]},
        "GradientBoosting": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]}
    }

    if xgb_available:
        models["XGBoost"] = XGBRegressor(objective="reg:squarederror", random_state=42)
        params["XGBoost"] = {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3,5]}

    results = {}
    fitted_models = {}

    # CV with TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits)

    for name, model in models.items():
        print(f"\n🔹 Training {name}")

        grid = GridSearchCV(model, params[name], cv=tscv,
                            scoring="neg_root_mean_squared_error", n_jobs=-1)
        grid.fit(X_train, y_train)

        best_estimator = grid.best_estimator_
        preds = best_estimator.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = np.mean(np.abs(y_test - preds))
        r2 = 1 - np.sum((y_test - preds) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)

        results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2}
        fitted_models[name] = best_estimator

        print(f"{name} → RMSE: {rmse:.3f}, MAE: {mae:.3f}, R2: {r2:.3f}")

    results_df = pd.DataFrame(results).T.sort_values("RMSE")
    best_model_name = results_df.index[0]
    best_model = fitted_models[best_model_name]

    # Compute residuals for selected model (training only)
    train_preds = best_model.predict(X_train)
    train_residuals = y_train - train_preds

    print(f"\n🏆 Best ML Model: {best_model_name}")
    return best_model_name, best_model, results_df, train_residuals
