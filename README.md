# 🦟 Dengue Early-Warning System (Dengue EWS)

> **AI-powered, ward-level dengue risk forecasting for proactive public health intervention.**

**EDU AI Engineering Hackathon 2026** · Preliminary Round  
**Challenge:** AI for Urban Dengue Outbreak

---

## 🌍 Overview

Dengue Early-Warning System (Dengue EWS) is an offline-first machine learning prototype designed to forecast dengue risk at the ward level **2–4 weeks in advance**.

By combining historical weather patterns with ward-specific vulnerability indicators, the system predicts future dengue cases and classifies wards into four risk levels:

🟢 **Low** · 🟡 **Medium** · 🟠 **High** · 🔴 **Critical**

The goal is to help public health officials identify vulnerable areas earlier and prioritize preventive interventions such as larvicide application, mosquito control, waste management, and targeted spraying.

### Coverage

- **70 wards**
- **5 urban areas**
- **10 years of weekly data (2016–2025)**

| City | Coverage |
|---|---|
| Dhaka North | Included |
| Dhaka South | Included |
| Chattogram | Included |
| Gazipur | Included |
| Narayanganj | Included |

---

## ✨ Key Features

- 🔮 **2–4 Week Forecasting** — Predict dengue cases several weeks ahead.
- 🗺️ **Ward-Level Risk Assessment** — Identify risk across individual wards.
- 🌦️ **Weather-Based Intelligence** — Analyze rainfall, temperature, and humidity patterns.
- 📊 **Interactive Dashboard** — Explore forecasts, ward rankings, and risk heatmaps.
- 📈 **Prediction Intervals** — Display an estimated 80% forecast interval.
- 🔍 **Explainable Predictions** — Understand which factors contribute to each prediction.
- 🏙️ **City-Wide Monitoring** — Compare wards across five urban areas.
- ⚡ **Offline Inference** — No weather API, LLM, or internet connection required at runtime.

---

## 🚀 Quick Start

Get the dashboard running locally in approximately 1–3 minutes.

### 1. Clone the repository

```bash
git clone https://github.com/souravd25/dengue-ews.git
cd dengue-ews
```

### 2. Create a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Windows (Git Bash):**

```bash
python -m venv .venv
source .venv/Scripts/activate
```

**Linux / macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The dashboard will open in your browser:

**http://localhost:8501**

Pre-trained models are included in the `models/` directory, so the application can run without retraining.

---

## 🧠 How It Works

The system transforms historical weather and ward-level information into future dengue risk predictions through three main stages.

### 1. Multi-Week Feature Engineering

Implemented in `src/features.py`.

The feature pipeline extracts temporal patterns and ward-specific characteristics to capture the delayed relationship between weather and dengue outbreaks.

**Weather features:**
- Average temperature (`avg_temp_c`)
- Rainfall (`rainfall_mm`)
- Relative humidity (`relative_humidity_pct`)

**Temporal features:**
- Rainfall lags from 1 to 8 weeks
- Rolling rainfall totals over 4 and 8 weeks
- Temperature lags of 2 and 4 weeks
- Humidity lags of 2 and 4 weeks
- Four-week rolling temperature and humidity averages
- Seasonal patterns

**Ward vulnerability features:**
- Waterlogging risk
- Population density
- Sanitation score

The pipeline uses ward-specific lag calculations to ensure historical data from one ward never contaminates another ward's features.

### 2. Machine Learning Prediction

Implemented in `src/train.py`.

The system uses **LightGBM with a Poisson objective**, which is suitable for modeling count-based data such as weekly dengue cases.

Separate models are trained for each forecasting horizon:

| Forecast Horizon | Prediction |
|---|---|
| 2 weeks | Dengue cases 2 weeks ahead |
| 3 weeks | Dengue cases 3 weeks ahead |
| 4 weeks | Dengue cases 4 weeks ahead |

Two additional quantile models generate prediction intervals. These intervals are calibrated using the validation year to achieve approximately 80% coverage.

Predicted case counts are converted into risk categories using thresholds derived from the 60th, 85th, and 95th percentiles of training-period weekly case distributions.

### 3. Interactive Dashboard

Implemented in `app.py` using Streamlit.

Users can explore:

- City and ward selection
- Forecast horizon (2, 3, or 4 weeks)
- Forecast origin week
- Predicted dengue cases
- Risk category
- 80% prediction interval
- Actual cases, when available
- Ward-level risk rankings
- Ward-by-week risk heatmaps
- Model validation and performance metrics

### 🔎 Prediction Explainability

The dashboard provides LightGBM feature contributions grouped into meaningful categories:

| Driver Group | Examples |
|---|---|
| 🌧️ Rainfall | Rainfall lags and rolling totals |
| 🌡️ Temperature | Temperature lags and averages |
| 💧 Humidity | Humidity lags and averages |
| 🏙️ Drainage | Waterlogging risk |
| 👥 Density | Population density |
| 🧹 Sanitation | Sanitation score |
| 📅 Seasonality | Seasonal patterns |

---

## 📊 Model Evaluation

### Evaluation Strategy

The model is evaluated using a **chronological split without random shuffling**, preserving the time-series structure of the data.

| Dataset | Period | Purpose |
|---|---|---|
| Training | 2016–2022 | Model fitting, feature development, and comparison |
| Validation | 2023 | Model selection, tuning, and interval calibration |
| Test | 2024–2025 | Final performance evaluation |

The final models are trained using the training and validation data, while the test period is reserved for final evaluation.

### Test Results (2024–2025)

| Horizon | MAE | Seasonal Baseline MAE | Mean Baseline MAE | RMSE | Tier Accuracy | Within 1 Tier | 80% Coverage |
|---|---:|---:|---:|---:|---:|---:|
| **2 Weeks** | 1.01 | 1.27 | 3.12 | 2.01 | 82.9% | 99.3% | 82.9% |
| **3 Weeks** | 1.04 | 1.28 | 3.13 | 2.00 | 81.4% | 99.2% | 83.1% |
| **4 Weeks** | 1.25 | 1.29 | 3.13 | 2.42 | 78.1% | 98.6% | 82.4% |

### Model Comparison (Validation Set)

Models were compared using Mean Absolute Error (MAE). Lower values indicate better forecasting performance.

| Model | 2 Weeks | 3 Weeks | 4 Weeks | Approx. Training Time |
|---|---:|---:|---:|---|
| Poisson GLM | 1.280 | 1.364 | 1.721 | < 1 sec |
| Random Forest | 1.225 | 1.221 | 1.500 | ~50 sec |
| **LightGBM (Poisson)** | **1.227** | **1.222** | **1.416** | **~2 sec** |

LightGBM was selected because it:

- Performs competitively at 2- and 3-week forecasting horizons.
- Achieves the lowest MAE at the 4-week horizon among the evaluated models.
- Trains significantly faster than Random Forest in the observed comparison.
- Supports quantile prediction and feature contribution analysis.

---

## 📉 Limitations

The prototype has several limitations that should be considered:

- Weather and ward vulnerability features provide most of the predictive signal.
- Temperature and humidity lag features contribute relatively small improvements.
- Forecast performance decreases at the 4-week horizon.
- Historical dengue case counts are not used as input features.
- The system uses a provided historical dataset rather than live surveillance feeds.
- Predictions are intended for decision support, not as a substitute for epidemiological expertise.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data processing and feature engineering |
| NumPy | Numerical operations |
| Scikit-learn | Baselines, metrics, and model comparison |
| LightGBM | Forecasting model |
| Joblib | Model serialization |
| Streamlit | Interactive dashboard |
| Plotly | Data visualization |
| Pytest | Automated testing |

---

## 📁 Repository Structure

```text
dengue-ews/
│
├── app.py                       # Streamlit dashboard
├── requirements.txt               # Runtime dependencies
├── requirements-dev.txt            # Development dependencies
│
├── data/
│   └── dengue.csv                # Weekly ward-level dataset
│
├── models/
│   ├── models.joblib              # Serialized trained models
│   ├── meta.json                  # Features, thresholds, metadata
│   ├── metrics.csv                # Final evaluation metrics
│   ├── model_comparison.csv        # Model comparison results
│   └── tuning_results.csv          # Hyperparameter tuning results
│
├── src/
│   ├── features.py                # Lag and rolling features
│   ├── train.py                   # Training and evaluation pipeline
│   ├── compare_models.py           # Model comparison
│   └── tune.py                    # Hyperparameter tuning
│
├── tests/
│   └── test_lags.py                # Feature alignment and leakage tests
│
├── ARCHITECTURE.md                # System architecture and design
└── README.md
```

---

## ⚙️ Optional Commands

### Retrain the Models

```bash
python -m src.train
```

Retrains the forecasting pipeline and exports updated model artifacts.

### Run Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
```

Tests verify that lag features are ward-specific, correctly aligned, and based only on past observations.

### Compare Models

```bash
python -m src.compare_models
```

Compares Poisson GLM, Random Forest, and LightGBM on the validation period.

### Tune Hyperparameters

```bash
python -m src.tune
```

Runs the hyperparameter search using the validation dataset.

---

## 💻 Compute & API Costs

The system is designed to minimize runtime complexity and external dependencies.

- **Zero runtime API calls**
- **No LLM inference**
- **No live weather API dependency**
- **Local CSV-based data processing**
- **Serialized models loaded once using `st.cache_resource`**
- **Feature calculations cached using `st.cache_data`**
- **Vectorized inference across wards**
- **CPU-based training in approximately 30 seconds**

The dashboard can generate predictions locally without relying on paid cloud services.

---

## 🤖 AI Assistance & Transparency

AI tools were used during development to support code generation, understanding, documentation, and prototyping.

**AI assistant used:**
- **Claude (Anthropic)** — Assisted with generating the initial project structure, feature pipeline, training scripts, dashboard, automated tests, and documentation drafts.

All generated code and materials were reviewed, tested, and modified by the team.

### Team AI Usage

| Team Member | AI Assistance |
|---|---|
| Sourav Das | Understanding lag features, model comparison, testing, and ML pipeline development |
| Roxana Rahman Othoi | Dashboard development, layout, and visualization assistance |
| Simi Saha | Documentation, repository organization, and demo preparation |

> AI was used as a development assistant. The team remains responsible for the implementation, verification, and final results.

**Runtime AI/API Usage:** None. The system makes no external API or LLM calls during prediction.

---

## 👥 Team

| Member | GitHub | Contribution |
|---|---|---|
| **Sourav Das** | [@souravd25](https://github.com/souravd25) | ML pipeline and forecasting model |
| **Roxana Rahman Othoi** | [@rahman-othoi](https://github.com/rahman-othoi) | Dashboard development and video |
| **Simi Saha** | [@sattyajitsaha94-web](https://github.com/sattyajitsaha94-web) | Documentation, repository, and demo |

---

## 🎥 Demo

https://drive.google.com/file/d/1caXXMBpDNxtGQ4E_wsirgzrGU1y3pBrM/view?usp=sharing

---

## 🎯 Project Goal

> **Predict dengue risk before cases surge — helping cities move from reactive response to proactive prevention.**

Dengue EWS aims to turn historical weather patterns and urban vulnerability data into actionable early warnings, enabling better-informed and more targeted public health decisions.

---

*Built for EDU AI Engineering Hackathon 2026.*