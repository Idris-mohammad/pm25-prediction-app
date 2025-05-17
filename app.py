import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import os
from datetime import datetime
import matplotlib.pyplot as plt

st.set_page_config(page_title="PM2.5 Predictor", layout="centered")  # ✅ Move this here

MODEL_URL = "https://drive.google.com/uc?id=1ByXdSW_f4ne9qLRcXB7osFbIMynXuu2V"
MODEL_PATH = "model_compressed.pkl"

# Now it's safe to use Streamlit decorators or functions
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            response = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
    return joblib.load(MODEL_PATH)

# Load model
model, features_used = load_model()

# ...rest of your code continues
