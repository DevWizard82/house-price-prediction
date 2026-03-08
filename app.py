import streamlit as st
import pandas as pd
import joblib
import folium
import numpy as np

from streamlit_folium import st_folium
from folium.plugins import HeatMap


# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="California House Price Predictor",
    layout="wide"
)

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------

model = joblib.load("models/model.pkl")


# ------------------------------------------------
# SESSION STATE
# ------------------------------------------------

if "prediction" not in st.session_state:
    st.session_state.prediction = None


# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------

st.sidebar.title("🏠 California Housing ML App")

st.sidebar.markdown("""
This application predicts **California housing prices** using a Machine Learning model.

### Model
Random Forest Regressor

### Dataset
California Housing Dataset

### Instructions
1. Click a location on the map  
2. Enter housing characteristics  
3. Click **Predict House Price**
""")


# ------------------------------------------------
# TITLE
# ------------------------------------------------

st.title("🏡 California House Price Predictor")

st.markdown(
"""
Estimate housing prices using geographic location and housing characteristics.
"""
)

st.divider()


# ------------------------------------------------
# CALIFORNIA BOUNDS
# ------------------------------------------------

CA_BOUNDS = [
    [32.0, -124.5],
    [42.0, -114.0]
]


# ------------------------------------------------
# LOCATION MAP
# ------------------------------------------------

st.subheader("📍 Select Property Location")

base_map = folium.Map(
    location=[36.5, -119.5],
    zoom_start=6,
    min_zoom=5,
    max_bounds=True
)

base_map.fit_bounds(CA_BOUNDS)

# California bounding box
folium.Rectangle(
    bounds=CA_BOUNDS,
    color="blue",
    fill=False
).add_to(base_map)

map_data = st_folium(base_map, width=900, height=450)

# default location (Los Angeles)
latitude = 34.05
longitude = -118.25

if map_data["last_clicked"] is not None:

    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]

    if 32 <= lat <= 42 and -124.5 <= lon <= -114:
        latitude = lat
        longitude = lon
    else:
        st.warning("⚠️ Please select a location inside California.")

st.write(f"Selected Coordinates → **Lat {latitude:.4f}, Lon {longitude:.4f}**")

st.divider()


# ------------------------------------------------
# INPUT GRID
# ------------------------------------------------

st.subheader("🏘 Housing Characteristics")

col1, col2, col3 = st.columns(3)

with col1:
    housing_median_age = st.number_input("House Age", 1, 100, 20)

with col2:
    median_income = st.number_input("Median Income", 1.0, 15.0, 4.0)

with col3:
    ocean_proximity = st.selectbox(
        "Ocean Proximity",
        ["<1H OCEAN","INLAND","NEAR OCEAN","NEAR BAY","ISLAND"]
    )

col4, col5, col6 = st.columns(3)

with col4:
    total_rooms = st.number_input("Total Rooms", 100, 10000, 2000)

with col5:
    total_bedrooms = st.number_input("Total Bedrooms", 50, 3000, 500)

with col6:
    households = st.number_input("Households", 100, 3000, 400)

col7, col8 = st.columns(2)

with col7:
    population = st.number_input("Population", 100, 5000, 1000)


# ------------------------------------------------
# FEATURE ENGINEERING
# ------------------------------------------------

rooms_per_household = total_rooms / households
bedrooms_per_room = total_bedrooms / total_rooms
population_per_household = population / households

data = pd.DataFrame({
    "longitude":[longitude],
    "latitude":[latitude],
    "housing_median_age":[housing_median_age],
    "total_rooms":[total_rooms],
    "total_bedrooms":[total_bedrooms],
    "population":[population],
    "households":[households],
    "median_income":[median_income],
    "ocean_proximity":[ocean_proximity],
    "rooms_per_household":[rooms_per_household],
    "bedrooms_per_room":[bedrooms_per_room],
    "population_per_household":[population_per_household]
})


st.divider()


# ------------------------------------------------
# PREDICTION BUTTON
# ------------------------------------------------

if st.button("🔮 Predict House Price", use_container_width=True):

    with st.spinner("Predicting house price..."):

        prediction = model.predict(data)[0]

        st.session_state.prediction = prediction


# ------------------------------------------------
# DISPLAY PREDICTION
# ------------------------------------------------

if st.session_state.prediction is not None:

    st.success(
        f"🏡 Estimated House Price: **${st.session_state.prediction:,.0f}**"
    )

    st.subheader("📍 Predicted Property Location")

    pred_map = folium.Map(
        location=[latitude, longitude],
        zoom_start=8
    )

    folium.Marker(
        [latitude, longitude],
        tooltip=f"Predicted Price: ${st.session_state.prediction:,.0f}",
        icon=folium.Icon(color="green", icon="home")
    ).add_to(pred_map)

    st_folium(pred_map, width=900, height=400)


