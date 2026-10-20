<img width="1024" height="1536" alt="image" src="https://github.com/user-attachments/assets/8511e37b-d617-46f3-a947-0046a87e2dec" /># Semiconductor Chip Failure Prediction

## Problem Statement

Predict semiconductor chip failures using manufacturing
and testing parameters.

## Project Architecture

```text
┌──────────────────────────────┐
│     Semiconductor Chip Data  │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│           Data EDA           │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│      Data Preprocessing      │
│  • Missing Value Handling    │
│  • Feature Engineering       │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│       ColumnTransformer      │
│  • Numerical Pipeline        │
│  • Ordinal Pipeline          │
│  • Nominal Pipeline          │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│      Logistic Regression     │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│       Prediction Model       │
│         best_model.pkl       │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│     Streamlit Application    │
└──────────────────────────────┘
```

## Machine Learning Workflow

1. **Data EDA:** Explore the semiconductor chip dataset using statistical analysis and visualizations.
2. **Data Preprocessing:** Handle missing values and perform feature engineering.
3. **Feature Transformation:** Use `ColumnTransformer` to process numerical, ordinal, and nominal features with separate pipelines.
4. **Model Training:** Train a Logistic Regression classification model.
5. **Model Serialization:** Save the trained model as `best_model.pkl`.
6. **Deployment:** Build an interactive web application using Streamlit for semiconductor chip predictions.






## Dataset Information

Features:
- Temperature
- Voltage
- Frequency
- Current
- Power Consumption
- Leakage Current
- Test Duration
- Defect Severity
- Manufacturing Process

Target:
- Failure / Non-Failure

## Exploratory Data Analysis

View the complete EDA notebook here:

[EDA Notebook](../Notebooks/01_preprocessing_EDA.ipynb)



## Data Preprocessing

- Missing Value Handling
- Feature Scaling
- Ordinal Encoding
- One Hot Encoding
- ColumnTransformer

## Machine Learning Pipeline

- SimpleImputer
- StandardScaler
- OrdinalEncoder
- OneHotEncoder
- ColumnTransformer
- Logistic Regression

## Model Evaluation

![Heatmap](../images/correlation_heatmap.png)


A



## Streamlit Application

![Streamlit_UI](../images/app_UI.png)




## How to Run

pip install -r requirements.txt

streamlit run streamlit_app.py
