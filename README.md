# 🤖 Autonomous Data Analyst (Agentic AI Project)

An end‑to‑end **Agentic AI system** that automatically performs **EDA, feature engineering, ML/DL model comparison, time‑series forecasting, uncertainty estimation, and domain insights** from a user‑uploaded dataset.

The system is designed as a **multi‑agent architecture**, coordinated by a central **Planner Agent**, and exposed through a **Streamlit application**.

---

## 🚀 Key Capabilities

* Automatic dataset inspection (EDA)
* Time‑series detection and feature generation
* Outlier handling
* Lag & rolling feature engineering
* Machine Learning model comparison
* Deep Learning sequence model comparison
* Intelligent ML vs DL model selection
* Statistical baseline via SARIMAX
* Multi‑step future forecasting
* Forecast uncertainty (confidence intervals)
* Domain / weather insight generation
* Fully interactive Streamlit UI

---

## 🧠 Architecture Overview

The project follows an **Agent‑based modular design**:

```
Autonomous_Data_Analyst/
│
├── agents/
│   ├── eda_agent.py
│   ├── time_agent.py
│   ├── outlier_agent.py
│   ├── feature_agent.py
│   ├── model_comparison_agent.py
│   ├── dl_agent.py
│   ├── ml_dl_comparison_agent.py
│   ├── future_forecast_agent.py
│   ├── uncertainty_agent.py
│   ├── sarimax_agent.py
│   └── weather_insight_agent.py
│
├── planner_agent.py
├── streamlit_app.py
├── main.py
├── requirements.txt
└── README.md
```

The **Planner Agent** orchestrates all agents dynamically based on data characteristics and model performance.

---

## 📸 Application Screenshots

### Streamlit Application Interface
![Streamlit UI](screenshots/streamlit_home.png)

### ML vs DL Model Comparison
![ML DL Table](screenshots/ml_dl_table.png)

### Forecast Visualization
![Forecast Plot](screenshots/forecast_plot.png)

### Forecast with Confidence Intervals
![Forecast CI](screenshots/forecast_with_ci.png)


---
## 🧩 Agents Used in Final Pipeline (11 Agents)

### 1️⃣ EDA Agent (`eda_agent`)

**Purpose:**

* Loads dataset
* Prints schema, missing values, and descriptive statistics

**Why it matters:**
Establishes a clean diagnostic baseline before modeling.

---

### 2️⃣ Time Agent (`time_agent`)

**Purpose:**

* Detects datetime columns
* Generates calendar and cyclical features (sin/cos)

**Why it matters:**
Prevents flat forecasts by injecting seasonality and temporal structure.

---

### 3️⃣ Outlier Agent (`outlier_agent`)

**Purpose:**

* Detects and caps/removes extreme target values using IQR

**Why it matters:**
Improves model stability and prevents distortion of loss metrics.

---

### 4️⃣ Feature Agent (`feature_agent`)

**Purpose:**

* Creates lag features
* Creates rolling mean and rolling standard deviation features
* Selects numeric ML‑ready features

**Why it matters:**
Transforms raw time‑series data into supervised learning format.

---

### 5️⃣ ML Model Comparison Agent (`model_comparison_agent`)

**Models Compared:**

* Linear Regression
* Random Forest
* Gradient Boosting
* XGBoost (if available)

**Metrics:**

* RMSE (primary)
* MAE
* R²

**Why it matters:**
Uses time‑aware validation to avoid leakage and select the best ML model.

---

### 6️⃣ DL Agent (`dl_agent`)

**Models Compared:**

* LSTM
* Bi‑LSTM
* GRU
* CNN‑LSTM

**Approach:**

* Sequence creation
* Scaling
* Early stopping

**Why it matters:**
Captures non‑linear temporal dependencies missed by classical ML.

---

### 7️⃣ ML vs DL Decision Agent (`ml_dl_comparison_agent`)

**Purpose:**

* Compares best ML vs best DL model using RMSE
* Selects the globally optimal approach

**Design Choice:**
RMSE prioritized over R² to avoid misleadingly high scores in time‑series.

---

### 8️⃣ SARIMAX Agent (`sarimax_agent`)

**Purpose:**

* Provides a statistical baseline forecast
* Supports exogenous variables (weather features)

**Why it matters:**
Acts as a transparent, interpretable benchmark for ML/DL forecasts.

---

### 9️⃣ Future Forecast Agent (`future_forecast_agent`)

**Purpose:**

* Performs recursive multi‑step forecasting
* Dynamically updates lag, rolling, and time features

**Why it matters:**
Enables real‑world future prediction beyond test data.

---

### 🔟 Uncertainty Agent (`add_forecast_uncertainty`)

**Purpose:**

* Adds confidence intervals using residual bootstrapping

**Why it matters:**
Transforms point forecasts into probabilistic forecasts.

---

### 1️⃣1️⃣ Weather / Domain Insight Agent (`weather_insight_agent`)

**Purpose:**

* Converts numeric forecasts into human‑readable insights

**Why it matters:**
Bridges the gap between data science output and business understanding.

---

## 📊 Evaluation Philosophy

* **RMSE is the primary decision metric** (scale‑aware, penalty on large errors)
* **R² is reported but not trusted alone** (often inflated in time‑series)
* DL models may show lower R² but superior RMSE — this is expected and valid

---

## 🖥 Streamlit Application

The Streamlit UI allows:

* Uploading any CSV dataset
* Selecting target variable
* Configuring forecast horizon
* Enabling/disabling SARIMAX baseline
* Viewing:

  * ML/DL comparison tables
  * Forecast values with confidence intervals
  * Domain insights

Run locally:

```bash
streamlit run streamlit_app.py
```

---

## 📦 Installation

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🎯 Project Outcomes

* Demonstrates **Agentic AI system design**
* Combines **ML, DL, and statistical modeling** in one pipeline
* Production‑ready Streamlit deployment

---

## 🧭 Future Enhancements

* Automated hyperparameter optimization (Optuna)
* Probabilistic DL forecasting (Quantile / Bayesian LSTM)
* Model registry & persistence
* Cloud deployment

---

## 👤 Author

**[AMSHEIK S]**
MSc Statistics | Data Analyst | Agentic AI Systems
