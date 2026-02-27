import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

@st.cache_resource
def load_interval():
    with open("interval.json", "r") as f:
        return json.load(f)

model = load_model()
interval_info = load_interval()

# Use MAE as a simple, robust interval half-width by default
DEFAULT_HALF_WIDTH = float(interval_info.get("mae", 70000))

st.title("🏠 House Price Prediction (King County)")
st.write("Enter house features to predict a **price range**.")

st.sidebar.header("Input Features")

# --- Inputs (match your feature set) ---
bedrooms = st.sidebar.number_input("Bedrooms", min_value=0, max_value=10, value=3)
bathrooms = st.sidebar.number_input("Bathrooms", min_value=0.0, max_value=10.0, value=2.0, step=0.25)

sqft_living = st.sidebar.number_input("Sqft Living", min_value=100, max_value=20000, value=1800, step=50)
sqft_lot = st.sidebar.number_input("Sqft Lot", min_value=100, max_value=200000, value=5000, step=100)
floors = st.sidebar.number_input("Floors", min_value=1.0, max_value=4.0, value=1.0, step=0.5)

waterfront = st.sidebar.selectbox("Waterfront", [0, 1], index=0)
view = st.sidebar.selectbox("View (0-4)", [0, 1, 2, 3, 4], index=0)
condition = st.sidebar.selectbox("Condition (1-5)", [1, 2, 3, 4, 5], index=2)
grade = st.sidebar.selectbox("Grade (1-13)", list(range(1, 14)), index=7)

sqft_basement = st.sidebar.number_input("Sqft Basement", min_value=0, max_value=10000, value=0, step=50)

lat = st.sidebar.number_input("Latitude", value=47.5112, format="%.6f")
long = st.sidebar.number_input("Longitude", value=-122.2570, format="%.6f")

sqft_living15 = st.sidebar.number_input("Sqft Living15", min_value=100, max_value=20000, value=2000, step=50)

basement_flag = st.sidebar.selectbox("Basement Flag", [0, 1], index=0)
renovated_flag = st.sidebar.selectbox("Renovated Flag", [0, 1], index=0)
house_age = st.sidebar.number_input("House Age", min_value=0, max_value=200, value=30, step=1)

# Interval settings
st.sidebar.header("Prediction Range")
half_width = st.sidebar.number_input(
    "Half-width ($) used for range",
    min_value=5000,
    max_value=500000,
    value=int(DEFAULT_HALF_WIDTH),
    step=1000
)
range_method = st.sidebar.selectbox(
    "Range method",
    ["± MAE (simple)", "± 1.5×MAE (wider)", "± 2×MAE (conservative)"],
    index=0
)

mult = {"± MAE (simple)": 1.0, "± 1.5×MAE (wider)": 1.5, "± 2×MAE (conservative)": 2.0}[range_method]
half_width = half_width * mult

# --- Build input row ---
# IMPORTANT: This must match your TRAINING dataframe columns (same names).
# If you log-transformed some X columns in your df already, the app should feed those logged values.
# Below assumes your df already contains logged versions stored under the same column names.
input_df = pd.DataFrame([{
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "sqft_living": sqft_living,
    "sqft_lot": sqft_lot,
    "floors": floors,
    "waterfront": waterfront,
    "view": view,
    "condition": condition,
    "grade": grade,
    "sqft_basement": sqft_basement,
    "lat": lat,
    "long": long,
    "sqft_living15": sqft_living15,
    "basement_flag": basement_flag,
    "renovated_flag": renovated_flag,
    "house_age": house_age
}])

# If during training you applied log transforms to these X columns *in the dataframe*,
# do the same here so the model sees the same distribution.
LOG_FEATURES = ["sqft_living", "sqft_lot", "sqft_basement", "sqft_living15"]
for col in LOG_FEATURES:
    input_df[col] = np.log1p(input_df[col])

# Keep flags categorical if your pipeline expects them as categories
for col in ["basement_flag", "renovated_flag"]:
    input_df[col] = input_df[col].astype("category")

st.subheader("Your inputs")
st.dataframe(input_df, use_container_width=True)

if st.button("Predict Price Range"):
    # Model predicts log(price)
    pred_log = float(model.predict(input_df)[0])
    pred_price = float(np.expm1(pred_log))

    low = max(0.0, pred_price - half_width)
    high = pred_price + half_width

    st.success(f"Estimated Price: **${pred_price:,.0f}**")
    st.info(f"Expected Range: **${low:,.0f} – ${high:,.0f}**")

    st.caption(
        "Range is residual-based using your stored MAE (or chosen half-width). "
        "For tighter, more correct intervals, consider quantile models."
    )