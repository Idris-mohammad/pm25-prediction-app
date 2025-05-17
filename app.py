import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime,time
import matplotlib.pyplot as plt

# Load model and feature list
with open("model.pkl", "rb") as f:
    model, features_used = pickle.load(f)

st.set_page_config(page_title="PM2.5 Predictor", layout="centered")
st.title("🌫️ Air Quality PM2.5 Prediction App")

st.markdown("Enter the pollutant levels to predict PM2.5:")

# Input sliders for features
co = st.slider("CO (Carbon Monoxide)", 0.0, 5000.0, 1000.0)
no = st.slider("NO (Nitric Oxide)", 0.0, 500.0, 10.0)
no2 = st.slider("NO₂ (Nitrogen Dioxide)", 0.0, 500.0, 20.0)
o3 = st.slider("O₃ (Ozone)", 0.0, 800.0, 50.0)
so2 = st.slider("SO₂ (Sulfur Dioxide)", 0.0, 600.0, 20.0)
pm10 = st.slider("PM10 (Particulate Matter 10)", 0.0, 2000.0, 100.0)
nh3 = st.slider("NH₃ (Ammonia)", 0.0, 300.0, 10.0)

# Create input dataframe
input_data = {
    "co": [co],
    "no": [no],
    "no2": [no2],
    "o3": [o3],
    "so2": [so2],
    "pm10": [pm10],
    "nh3": [nh3]
}
input_df = pd.DataFrame(input_data)

# User-selectable date and time
selected_date = st.date_input("📅 Select Date", datetime.now().date())
selected_time = st.time_input("⏰ Select Time", datetime.now().time())
selected_datetime = datetime.combine(selected_date, selected_time)

# Add datetime features
input_df["month"] = selected_datetime.month
input_df["day"] = selected_datetime.day
input_df["hour"] = selected_datetime.hour
input_df["dayofweek"] = selected_datetime.weekday()
input_df["is_weekend"] = 1 if selected_datetime.weekday() >= 5 else 0

# Reorder columns to match training
input_df = input_df[features_used]

# Display input table
st.subheader("📋 Input Summary")
st.dataframe(input_df)

# Store predictions
if "history" not in st.session_state:
    st.session_state.history = []

# Predict button
if st.button("🔮 Predict PM2.5"):
    prediction = model.predict(input_df)[0]
    st.success(f"### Predicted PM2.5 Level: {prediction:.2f}")
    # Save to history
    entry = input_df.copy()
    entry["predicted_pm2_5"] = prediction
    st.session_state.history.append(entry)

# Plot prediction history
if st.session_state.history:
    hist_df = pd.concat(st.session_state.history, ignore_index=True)

    st.subheader("📈 Prediction History")
    st.dataframe(hist_df)

    # Line chart of predictions
    fig, ax = plt.subplots()
    ax.plot(hist_df.index + 1, hist_df["predicted_pm2_5"], marker="o")
    ax.set_title("Predicted PM2.5 Over Time")
    ax.set_xlabel("Prediction Number")
    ax.set_ylabel("PM2.5 Level")
    st.pyplot(fig, use_container_width=True)

    # Download CSV
    csv = hist_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Prediction History as CSV",
        data=csv,
        file_name="pm2_5_predictions.csv",
        mime="text/csv"
    )
