import json
import urllib.request
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

st.set_page_config(page_title="TerraSpectra Crop Disease GIS", layout="wide")
st.title("TerraSpectra: Hyperspectral Crop Stress Dashboard")
st.caption("3D-CNN Spatial-Spectral Disease Detection (64×64×16 PCA Cube)")

PLACES = {
    "Narakoduru, Guntur (Andhra Pradesh)": {"lat": 16.2450, "lon": 80.5180},
    "Shamirpet, Medchal-Malkajgiri (Telangana)": {"lat": 17.5940, "lon": 78.5718},
    "Walajabad, Kanchipuram (Tamil Nadu)": {"lat": 12.7925, "lon": 79.8210},
}

CROPS = ["Chilli", "Cotton", "Paddy", "Maize", "Sugarcane", "Groundnut"]
ACRES_OPTIONS = [10, 25, 50, 100, 250, 500]


@st.cache_data
def load_farm_data():
    # Try fetching live from FastAPI first; fallback to local JSON if offline
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/api/predictions", timeout=2) as res:
            records = json.loads(res.read().decode())
            source = "Live FastAPI Endpoint (/api/predictions)"
    except Exception:
        with open("data/farm_predictions.json", "r") as f:
            records = json.load(f)
            source = "Local File (data/farm_predictions.json)"
    return pd.DataFrame(records), source


raw_df, data_source = load_farm_data()

# Sidebar controls: Threshold first, followed by Place, Crop, and Acres dropdowns
st.sidebar.header("Model & Farm Controls")
threshold = st.sidebar.slider("Stress Probability Threshold", 0.10, 0.90, 0.50, 0.05)

selected_place = st.sidebar.selectbox("Place", list(PLACES.keys()))
selected_crop = st.sidebar.selectbox("Crop", CROPS)
selected_acres = st.sidebar.selectbox("Acres", ACRES_OPTIONS, index=2)

st.sidebar.info(f"Data Source:\n{data_source}")

# Shift grid coordinates relative to the selected Place
place_coords = PLACES[selected_place]
df = raw_df.copy()
lat_offset = place_coords["lat"] - df["lat"].min()
lon_offset = place_coords["lon"] - df["lon"].min()
df["lat"] = (df["lat"] + lat_offset).round(6)
df["lon"] = (df["lon"] + lon_offset).round(6)

# Recalculate status dynamically based on slider
df["dynamic_status"] = df["stress_score"].apply(
    lambda s: "Stressed" if s >= threshold else "Healthy"
)
df["color"] = df["dynamic_status"].apply(
    lambda st_val: [220, 53, 69, 200] if st_val == "Stressed" else [40, 167, 69, 140]
)
df["elevation"] = df["stress_score"] * 120

# Top KPI Metrics Row (scaled to selected Acres)
total_pixels = len(df)
stressed_pixels = int((df["dynamic_status"] == "Stressed").sum())
stress_ratio = stressed_pixels / total_pixels
stress_pct = round(stress_ratio * 100, 2)

stressed_acres = round(selected_acres * stress_ratio, 2)
healthy_acres = round(selected_acres - stressed_acres, 2)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Monitored Farm", f"{selected_acres} Acres", selected_crop)
col2.metric("Healthy Area", f"{healthy_acres} Acres")
col3.metric("Stressed Area", f"{stressed_acres} Acres")
col4.metric("Affected Percentage", f"{stress_pct}%")

# Dual Tabs: PyDeck 3D Map + Plotly Data Science Charts
tab1, tab2 = st.tabs(["3D Geospatial Map (PyDeck)", "Data Science Analysis (Plotly)"])

with tab1:
    st.subheader(f"3D Crop Stress Map — {selected_place} ({selected_crop})")
    view_state = pdk.ViewState(
        latitude=df["lat"].mean(),
        longitude=df["lon"].mean(),
        zoom=14.2,
        pitch=45,
    )
    layer = pdk.Layer(
        "ColumnLayer",
        data=df,
        get_position=["lon", "lat"],
        get_elevation="elevation",
        elevation_scale=1,
        radius=18,
        get_fill_color="color",
        pickable=True,
        auto_highlight=True,
    )
    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={
                "text": f"Crop: {selected_crop}\nStatus: {{dynamic_status}}\nStress Score: {{stress_score}}\nLat: {{lat}}, Lon: {{lon}}"
            },
        )
    )

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("2D Farm Grid Stress Heatmap")
        fig_map = px.scatter(
            df,
            x="lon",
            y="lat",
            color="dynamic_status",
            color_discrete_map={"Healthy": "#28a745", "Stressed": "#dc3545"},
            hover_data=["stress_score"],
            title=f"{selected_crop} Field Grid ({selected_acres} Acres) — {selected_place}",
        )
        fig_map.update_traces(marker=dict(size=7, symbol="square"))
        st.plotly_chart(fig_map, width="stretch")

    with c2:
        st.subheader("3D-CNN Stress Probability Distribution")
        fig_hist = px.histogram(
            df,
            x="stress_score",
            nbins=25,
            color="dynamic_status",
            color_discrete_map={"Healthy": "#28a745", "Stressed": "#dc3545"},
            title=f"{selected_crop} Confidence Score Distribution",
        )
        fig_hist.add_vline(x=threshold, line_dash="dash", line_color="black")
        st.plotly_chart(fig_hist, width="stretch")