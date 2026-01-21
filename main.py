from planner_agent import planner_agent

def main():
    # Run the autonomous pipeline
    results = planner_agent(
        data_path="datasets/India_weather_data.csv",
        target_col="Temp_Mean",
        forecast_days=30
    )

    # Optional: Print summary
    print("\n📊 Pipeline Summary:")
    print("Final Model Type:", results["final_model_type"])
    print("ML Results:\n", results["ml_results"])
    print("DL Results:\n", results["dl_results"])
    print("Forecast (next 30 days):\n", results["forecast"])
    print("Weather Insight:\n", results["insight"])

if __name__ == "__main__":
    main()
