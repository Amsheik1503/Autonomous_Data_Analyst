def weather_insight_agent(forecast_df):
    """
    Generates domain-level insight from forecast DataFrame
    """

    # Ensure correct column
    if "Forecast" not in forecast_df.columns:
        raise ValueError("Forecast DataFrame must contain 'Forecast' column")

    temps = forecast_df["Forecast"].astype(float)

    avg_temp = temps.mean()
    max_temp = temps.max()
    min_temp = temps.min()
    trend = "increasing" if temps.iloc[-1] > temps.iloc[0] else "decreasing"

    insight = (
        f"Average temperature over forecast period: {avg_temp:.2f}°C\n"
        f"Maximum expected temperature: {max_temp:.2f}°C\n"
        f"Minimum expected temperature: {min_temp:.2f}°C\n"
        f"Overall temperature trend appears to be {trend}."
    )

    return insight
