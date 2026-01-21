import streamlit as st
import pandas as pd
import tempfile
import os
import matplotlib.pyplot as plt

from planner_agent import planner_agent

# --------------------------------------------------
# Page config (MUST be first Streamlit command)
# --------------------------------------------------
st.set_page_config(
    page_title="Autonomous Data Analyst",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# Title
# --------------------------------------------------
st.title("🤖 Autonomous Data Analyst")
st.caption(
    "Upload a dataset and automatically perform EDA, feature engineering, "
    "ML/DL model selection, forecasting, and domain insights."
)

st.divider()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("⚙️ Configuration")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

forecast_days = st.sidebar.slider(
    "Forecast horizon (days)",
    min_value=7,
    max_value=90,
    value=30
)

run_sarimax = st.sidebar.checkbox(
    "Run SARIMAX baseline",
    value=True,
    help="Disable if dataset lacks weather/exogenous variables"
)

run_button = st.sidebar.button("🚀 Run Analysis")

# --------------------------------------------------
# Main logic
# --------------------------------------------------
if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    target_col = st.selectbox(
        "🎯 Select target column to forecast",
        options=df.columns
    )

    if run_button:

        with st.spinner("Running autonomous pipeline..."):

            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                df.to_csv(tmp.name, index=False)
                temp_path = tmp.name

            try:
                results = planner_agent(
                    data_path=temp_path,
                    target_col=target_col,
                    forecast_days=forecast_days,
                    run_sarimax=run_sarimax
                )
            except Exception as e:
                st.error("❌ Pipeline failed")
                st.exception(e)
                os.unlink(temp_path)
                st.stop()

            os.unlink(temp_path)

        st.success("✅ Analysis completed successfully!")

        # --------------------------------------------------
        # Final model
        # --------------------------------------------------
        st.divider()
        st.subheader("🏆 Final Model Selection")
        st.write("**Selected Model Type:**", results["final_model_type"])

        # --------------------------------------------------
        # ML results
        # --------------------------------------------------
        st.subheader("🤖 ML Model Performance")
        st.dataframe(
            results["ml_results"].sort_values("RMSE"),
            use_container_width=True
        )

        # --------------------------------------------------
        # DL results
        # --------------------------------------------------
        st.subheader("🧠 DL Model Performance")
        st.dataframe(
            results["dl_results"].sort_values("RMSE"),
            use_container_width=True
        )

        # --------------------------------------------------
        # Forecast table
        # --------------------------------------------------
        st.subheader("📈 Forecast (Next Days)")
        forecast_df = results["forecast"]
        st.dataframe(forecast_df, use_container_width=True)

        # --------------------------------------------------
        # Forecast plot (Matplotlib)
        # --------------------------------------------------
        st.subheader("📉 Forecast Visualization")

        fig, ax = plt.subplots(figsize=(10, 4))

        # Forecast line
        ax.plot(
            forecast_df["Day"],
            forecast_df["Forecast"],
            label="Forecast",
            linewidth=2
        )       

        # Confidence Interval (safe)
        if {"Lower_CI", "Upper_CI"}.issubset(forecast_df.columns):
            ax.fill_between(
                forecast_df["Day"],
                forecast_df["Lower_CI"],
                forecast_df["Upper_CI"],
                alpha=0.25,
                label="Confidence Interval"
            )

        ax.set_xlabel("Day")
        ax.set_ylabel(target_col)
        ax.set_title("Future Forecast")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)

        # --------------------------------------------------
        # Insight
        # --------------------------------------------------
        st.subheader("🌡 Domain Insight")
        st.info(results["insight"])

else:
    st.info("⬅️ Upload a CSV file to begin analysis.")
