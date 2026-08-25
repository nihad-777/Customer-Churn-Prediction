import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model and scaler
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_cols = joblib.load('feature_columns.pkl')

st.title("E-commerce Customer Churn Predictor")
st.write("Enter customer details to predict the likelihood of them leaving.")

# Create input fields for the most important features we found in EDA
# Note: In a full app, you'd include all features, but let's start with the top ones
col1, col2 = st.columns(2)

with col1:
    calls = st.number_input("Customer Service Calls", min_value=0, max_value=20, value=5)
    abandon_rate = st.slider("Cart Abandonment Rate (%)", 0.0, 100.0, 50.0)
    session_dur = st.number_input("Avg Session Duration (min)", 1.0, 100.0, 25.0)

with col2:
    ltv = st.number_input("Lifetime Value ($)", 0.0, 10000.0, 1200.0)
    logins = st.number_input("Login Frequency (per month)", 0, 30, 10)
    membership = st.number_input("Membership Years", 0.0, 10.0, 2.5)

if st.button("Predict Churn Risk"):
    # 1. Create a dummy row with all 0s for all features
    input_data = pd.DataFrame(0, index=[0], columns=feature_cols)
    
    # 2. Fill in the values the user provided
    input_data['Customer_Service_Calls'] = calls
    input_data['Cart_Abandonment_Rate'] = abandon_rate
    input_data['Session_Duration_Avg'] = session_dur
    input_data['Lifetime_Value'] = ltv
    input_data['Login_Frequency'] = logins
    input_data['Membership_Years'] = membership
    
    # 3. Scale the data
    input_scaled = scaler.transform(input_data)
    
    # 4. Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    st.write(f"Debug - Raw Probability of Churn: {probability}")
    
    # 5. Show Results
    if prediction == 1:
        st.error(f"High Risk! Churn Probability: {probability:.2%}")
        st.write("Recommendation: Send a retention discount or follow up via email.")
    else:
        st.success(f"Low Risk. Churn Probability: {probability:.2%}")
        st.write("Recommendation: Customer is loyal. Consider upselling premium products.")