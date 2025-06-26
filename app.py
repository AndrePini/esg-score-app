
import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("esg_model.pkl")
encoder_industry = joblib.load("encoder_industry.pkl")
encoder_exchange = joblib.load("encoder_exchange.pkl")
encoder_currency = joblib.load("encoder_currency.pkl")

# Hero Section with Custom Styling
st.markdown(
    """
    <div style="background-color:#1E293B;padding:50px;border-radius:10px">
        <h1 style="color:#FFFFFF;text-align:center">📊 ESG Score Predictor</h1>
        <p style="color:#CCCCCC;text-align:center;font-size:18px">
            Predict a company’s Environmental, Social, and Governance score instantly using AI.
        </p>
        <div style="text-align:center">
            <a href="#predict" style="background-color:#FFC107;padding:10px 20px;color:#000000;
                text-decoration:none;border-radius:5px;font-weight:bold;margin-right:10px;">
                Get a Prediction
            </a>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Divider
st.markdown("<hr id='predict'>", unsafe_allow_html=True)

# Input Section
st.subheader("Enter Company Information")
industry = st.selectbox("Industry", encoder_industry.classes_)
exchange = st.selectbox("Exchange", encoder_exchange.classes_)
currency = st.selectbox("Currency", encoder_currency.classes_)

# Predict Button
if st.button("Predict ESG Score"):
    input_df = pd.DataFrame([{
        'industry': encoder_industry.transform([industry])[0],
        'exchange': encoder_exchange.transform([exchange])[0],
        'currency': encoder_currency.transform([currency])[0]
    }])

    prediction = model.predict(input_df)[0]
    st.success(f"Predicted ESG Score: {prediction:.2f}")
