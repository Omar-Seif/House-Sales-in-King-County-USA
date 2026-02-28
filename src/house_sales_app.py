import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, date

import folium
from streamlit_folium import st_folium


st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load("../model/model.joblib")


@st.cache_resource
def load_interval():
    with open("../json/interval.json", "r") as f:
        return json.load(f)


@st.cache_resource
def load_geo():
    clusterer = joblib.load("../model/geo_clusterer.joblib")
    with open("../json/geo_meta.json", "r") as f:
        meta = json.load(f)
    return clusterer, meta


model = load_model()
interval_info = load_interval()
DEFAULT_MAE = float(interval_info.get("mae", 65000))

geo_clusterer, geo_meta = load_geo()
bounds = geo_meta["bounds"]
clusters = geo_meta["clusters"]


def format_money(x: float) -> str:
    return f"${x:,.0f}"


def in_bounds(lat: float, lon: float) -> bool:
    return (bounds["min_lat"] <= lat <= bounds["max_lat"]) and (bounds["min_long"] <= lon <= bounds["max_long"])


def clamp_to_bounds(lat: float, lon: float):
    lat = min(max(lat, bounds["min_lat"]), bounds["max_lat"])
    lon = min(max(lon, bounds["min_long"]), bounds["max_long"])
    return lat, lon


def cluster_id_for(lat: float, lon: float) -> int:
    return int(geo_clusterer.predict(np.array([[lat, lon]], dtype=float))[0])


st.title("🏠 King County House Price Predictor")
st.caption("Realistic UI: neighborhood dropdown + map click + derived features + **price range**.")


# ---------------- Sidebar (ONLY range controls) ----------------
st.sidebar.header("Prediction Range")

range_method = st.sidebar.selectbox(
    "Range method",
    ["± MAE (simple)", "± 1.5×MAE (wider)", "± 2×MAE (conservative)"],
    index=0,
)
mult = {"± MAE (simple)": 1.0, "± 1.5×MAE (wider)": 1.5, "± 2×MAE (conservative)": 2.0}[range_method]

base_half_width = st.sidebar.number_input(
    "Base half-width ($) (default = your test MAE)",
    min_value=5000,
    max_value=500000,
    value=int(DEFAULT_MAE),
    step=1000,
)

half_width = base_half_width * mult
st.sidebar.markdown("---")
st.sidebar.write(f"**Final half-width:** {format_money(half_width)}")


# ---------------- Session State defaults ----------------
# Default to cluster 0 centroid
if "cluster_id" not in st.session_state:
    st.session_state.cluster_id = clusters[0]["id"]

if "lat" not in st.session_state or "long" not in st.session_state:
    c0 = clusters[st.session_state.cluster_id]
    st.session_state.lat = c0["centroid_lat"]
    st.session_state.long = c0["centroid_long"]

# Ensure current point stays in bounds
st.session_state.lat, st.session_state.long = clamp_to_bounds(st.session_state.lat, st.session_state.long)


# ---------------- Main UI Layout ----------------
left, right = st.columns([1.15, 1])

with left:
    st.subheader("1) Choose neighborhood + location")

    # Neighborhood dropdown
    options = {f'{c["name"]} (#{c["id"]})': c["id"] for c in clusters}
    # Keep selected label stable
    current_label = None
    for label, cid in options.items():
        if cid == st.session_state.cluster_id:
            current_label = label
            break

    chosen_label = st.selectbox("Neighborhood (clustered from lat/long)", list(options.keys()),
                                index=list(options.keys()).index(current_label))
    chosen_cluster_id = options[chosen_label]

    # If neighborhood changed, recenter to centroid
    if chosen_cluster_id != st.session_state.cluster_id:
        st.session_state.cluster_id = chosen_cluster_id
        cc = clusters[chosen_cluster_id]
        st.session_state.lat = cc["centroid_lat"]
        st.session_state.long = cc["centroid_long"]

    st.caption("Tip: you can also click the map; the app will auto-assign the nearest neighborhood cluster.")

    # --- Map with bounds restriction ---
    m = folium.Map(location=[st.session_state.lat, st.session_state.long], zoom_start=11, max_bounds=True)

    # Fit map to dataset bounds and draw a rectangle to show scope
    sw = [bounds["min_lat"], bounds["min_long"]]
    ne = [bounds["max_lat"], bounds["max_long"]]
    m.fit_bounds([sw, ne])
    folium.Rectangle(bounds=[sw, ne], color="blue", fill=False).add_to(m)

    folium.Marker(
        [st.session_state.lat, st.session_state.long],
        tooltip="Selected location",
    ).add_to(m)

    map_data = st_folium(m, height=420, width=None)

    # Handle click
    if map_data and map_data.get("last_clicked"):
        clicked_lat = float(map_data["last_clicked"]["lat"])
        clicked_lon = float(map_data["last_clicked"]["lng"])

        if in_bounds(clicked_lat, clicked_lon):
            st.session_state.lat = clicked_lat
            st.session_state.long = clicked_lon
            st.session_state.cluster_id = cluster_id_for(clicked_lat, clicked_lon)
        else:
            st.warning("That point is outside King County (dataset scope). Please click inside the blue rectangle.")

    st.write(f"📍 Selected: **lat={st.session_state.lat:.6f}**, **long={st.session_state.long:.6f}**")
    st.write(f"🏘️ Auto neighborhood: **Neighborhood {st.session_state.cluster_id + 1}**")

    st.subheader("2) House details")

    c1, c2, c3 = st.columns(3)

    with c1:
        bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=3)
        bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=10.0, value=2.0, step=0.25)
        floors = st.number_input("Floors", min_value=1.0, max_value=4.0, value=1.0, step=0.5)

    with c2:
        sqft_living = st.number_input("Living area (sqft)", min_value=200, max_value=20000, value=1800, step=50)
        sqft_lot = st.number_input("Lot size (sqft)", min_value=200, max_value=200000, value=5000, step=100)
        sqft_basement = st.number_input("Basement (sqft)", min_value=0, max_value=10000, value=0, step=50)

    with c3:
        sqft_living15 = st.number_input("Nearby avg living area (sqft_living15)", min_value=200, max_value=20000, value=2000, step=50)

        built_date = st.date_input("Year built (pick any date in that year)", value=date(1990, 1, 1))
        current_year = datetime.now().year
        house_age = max(0, current_year - built_date.year)
        st.write(f"Derived **house_age**: **{house_age}** years")

    st.subheader("3) Property attributes")

    a1, a2, a3 = st.columns(3)

    with a1:
        waterfront = st.checkbox("Waterfront", value=False)
        basement_flag = st.checkbox("Has basement", value=(sqft_basement > 0))
        renovated_flag = st.checkbox("Renovated", value=False)

    with a2:
        view = st.slider("View rating (0–4)", min_value=0, max_value=4, value=0)
        condition = st.slider("Condition (1–5)", min_value=1, max_value=5, value=3)

    with a3:
        grade = st.slider("Grade (1–13)", min_value=1, max_value=13, value=8)


with right:
    st.subheader("Summary + Prediction")

    # RAW inputs (human readable, no logs)
    input_raw = pd.DataFrame([{
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "sqft_living": sqft_living,
        "sqft_lot": sqft_lot,
        "floors": floors,
        "waterfront": int(waterfront),
        "view": view,
        "condition": condition,
        "grade": grade,
        "sqft_basement": sqft_basement,
        "lat": float(st.session_state.lat),
        "long": float(st.session_state.long),
        "sqft_living15": sqft_living15,
        "basement_flag": int(basement_flag),
        "renovated_flag": int(renovated_flag),
        "house_age": house_age,
    }])

    st.write("**Your inputs (raw values)**")
    st.dataframe(input_raw, use_container_width=True)

    # MODEL inputs (apply same transforms used during training)
    input_model = input_raw.copy()

    # Apply log1p to sqft features if you transformed them before training
    LOG_FEATURES = ["sqft_living", "sqft_lot", "sqft_basement", "sqft_living15"]
    for col in LOG_FEATURES:
        input_model[col] = np.log1p(input_model[col])

    # Flags categorical (match your pipeline setup)
    for col in ["basement_flag", "renovated_flag"]:
        input_model[col] = input_model[col].astype("category")

    st.markdown("---")

    if st.button("Predict Price Range", type="primary"):
        pred_log = float(model.predict(input_model)[0])
        pred_price = float(np.expm1(pred_log))

        low = max(0.0, pred_price - half_width)
        high = pred_price + half_width

        st.success(f"Estimated Price: **{format_money(pred_price)}**")
        st.info(f"Expected Range: **{format_money(low)} – {format_money(high)}**")
        st.caption("Range is MAE-based. For a more realistic interval, use quantile models (10th/90th).")