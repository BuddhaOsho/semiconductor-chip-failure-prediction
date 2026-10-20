import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/best_model.pkl")

# Page Configuration
st.set_page_config(
    page_title="Semiconductor Chip Failure Prediction",
    page_icon="🔬",
    layout="wide"
)

# Title
st.title("🔬 Semiconductor Chip Failure Prediction")

st.markdown("""
Predict whether a semiconductor chip is likely to **Fail** or **Pass**
based on manufacturing parameters.
""")

# Sidebar
st.sidebar.header("Project Information")

st.sidebar.info("""
Model: Logistic Regression

Business Goal:
Detect potentially defective semiconductor chips before production deployment.

Target:
0 = Pass
1 = Fail
""")

# Input Form
st.header("Enter Manufacturing Parameters")

col1, col2 = st.columns(2)

with col1:

    temperature_c = st.number_input(
        "Temperature (°C)",
        min_value=0.0,
        value=50.0
    )

    voltage_v = st.number_input(
        "Voltage (V)",
        min_value=0.0,
        value=1.10
    )

    frequency_mhz = st.number_input(
        "Frequency (MHz)",
        min_value=0.0,
        value=2500.0
    )

    current_a = st.number_input(
        "Current (A)",
        min_value=0.0,
        value=0.30
    )

    power_consumption_mw = st.number_input(
        "Power Consumption (mW)",
        min_value=0.0,
        value=300.0
    )

with col2:

    leakage_current_ua = st.number_input(
        "Leakage Current (µA)",
        min_value=0.0,
        value=2.0
    )

    test_duration_sec = st.number_input(
        "Test Duration (Sec)",
        min_value=0.0,
        value=30.0
    )

    manufacturing_process = st.selectbox(
        "Manufacturing Process",
        [
            "FinFET",
            "FD-SOI",
            "GAA"
        ]
    )

    defect_rate = st.number_input(
        "Defect Rate",
        min_value=0.0,
        value=0.05
    )

    defect_severity = st.selectbox(
        "Defect Severity",
        [
            "Low",
            "Medium",
            "High"
        ]
    )

# Prediction Button
if st.button("Predict Failure Risk"):

    input_df = pd.DataFrame({
        "temperature_c": [temperature_c],
        "voltage_v": [voltage_v],
        "frequency_mhz": [frequency_mhz],
        "current_a": [current_a],
        "power_consumption_mw": [power_consumption_mw],
        "leakage_current_ua": [leakage_current_ua],
        "test_duration_sec": [test_duration_sec],
        "manufacturing_process": [manufacturing_process],
        "defect_rate": [defect_rate],
        "defect_severity": [defect_severity]
    })

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error(
            f"⚠️ CHIP FAILURE PREDICTED\n\n"
            f"Failure Probability: {probability:.2%}"
        )

    else:

        st.success(
            f"✅ CHIP PASSED\n\n"
            f"Failure Probability: {probability:.2%}"
        )

    st.subheader("Input Summary")

    st.dataframe(input_df)