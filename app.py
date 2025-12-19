import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the model and column structure
model = joblib.load('xgb_fraud_model.pkl')
model_columns = joblib.load('model_columns.pkl')

st.title("🛡️ Transaction Fraud Detector")
st.write("Input transaction details below to check for fraud risk.")

# Create two columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    transaction_hour = st.slider("Transaction Hour (0-23)", 0, 23, 12)
    device_trust = st.number_input("Device Trust Score", 0.0, 1.0, 0.5)
    amount = st.number_input("Transaction Amount", 0.0, 10000.0, 50.0)

with col2:
    foreign = st.selectbox("Foreign Transaction?", ["No", "Yes"])
    mismatch = st.selectbox("Location Mismatch?", ["No", "Yes"])
    velocity = st.number_input("Velocity (Last 24h)", 0, 50, 1)

# Prediction Button
if st.button("Analyze Transaction"):
    # 1. Prepare the input dictionary
    input_data = {
        'transaction_hour': transaction_hour,
        'device_trust_score': device_trust,
        'amount': amount,
        'foreign_transaction': 1 if foreign == "Yes" else 0,
        'location_mismatch': 1 if mismatch == "Yes" else 0,
        'velocity_last_24h': velocity
    }
    
    # 2. Convert to DataFrame and align with training columns
    input_df = pd.DataFrame([input_data])
    # Ensure all missing dummy columns are added as 0
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    
    # 3. Make Prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    # 4. Display Results
    if prediction == 1:
        st.error(f"🚨 High Risk Detected! (Probability: {probability:.2%})")
    else:
        st.success(f"✅ Transaction Appears Safe. (Probability: {probability:.2%})")