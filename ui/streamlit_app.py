"""Streamlit UI for FraudWatch. Run the API first, then:
  pip install -r requirements-ui.txt
  streamlit run ui/streamlit_app.py    # calls the API at FRAUDWATCH_API (default http://localhost:8000)
The API also serves a minimal built-in page at / so the demo works without Streamlit.
"""
import os
import requests
import streamlit as st

API = os.environ.get("FRAUDWATCH_API", "http://localhost:8000")
st.title("FraudWatch — fraud scoring demo (synthetic data)")
try:
    model = requests.get(f"{API}/api/model", timeout=5).json()
    st.info(f"Serving model v{model['model_version']} ({model['kind']}), threshold {model['threshold']}")
except Exception as e:
    st.error(f"API not reachable at {API}: {e}"); st.stop()

amount = st.number_input("Amount", min_value=0.01, value=240.0)
hour = st.slider("Hour", 0, 23, 2)
merchant = st.selectbox("Merchant", ["grocery", "fuel", "restaurant", "retail", "travel", "online_market", "crypto_kiosk", "pharmacy"])
channel = st.selectbox("Channel", ["card_present", "online", "mobile_wallet", "atm"])
device = st.selectbox("Device", ["card", "phone_known", "phone_new", "browser_known", "browser_new"])
country_match = st.selectbox("Country match", [1, 0])
velocity = st.number_input("Velocity (tx in 24h)", min_value=0, value=3)
ratio = st.number_input("Amount / customer median", min_value=0.0, value=3.1)

if st.button("Score transaction"):
    payload = {"amount": amount, "hour": hour, "day_of_week": 2, "is_weekend": 0,
               "merchant_category": merchant, "channel": channel, "device_type": device,
               "country_match": country_match, "velocity_24h": int(velocity),
               "days_since_prev_tx": 0.5, "amount_ratio_to_customer_median": ratio,
               "customer_age_days": 120}
    r = requests.post(f"{API}/api/predict", json=payload, timeout=10)
    st.json(r.json())
