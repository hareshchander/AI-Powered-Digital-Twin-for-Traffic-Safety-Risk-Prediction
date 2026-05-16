import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from datetime import datetime

from data_generator import TrafficDataGenerator
from model_training import train_and_save_models
from llm_analysis import generate_incident_analysis
from utils import prepare_features

# --- Streamlit Config ---
st.set_page_config(page_title="Traffic Digital Twin", layout="wide", page_icon="🚦")

# --- Initialize Models ---
MODELS_DIR = "models"
if not os.path.exists(MODELS_DIR) or not os.path.exists("synthetic_traffic_data.csv"):
    with st.spinner("Initializing system and training models..."):
        generator = TrafficDataGenerator(num_records=5000)
        df = generator.generate_data()
        df.to_csv("synthetic_traffic_data.csv", index=False)
        train_and_save_models()

@st.cache_resource
def load_models():
    coll_model = joblib.load(os.path.join(MODELS_DIR, 'collision_model.pkl'))
    cong_model = joblib.load(os.path.join(MODELS_DIR, 'congestion_model.pkl'))
    risk_model = joblib.load(os.path.join(MODELS_DIR, 'risk_model.pkl'))
    anomaly_model = joblib.load(os.path.join(MODELS_DIR, 'anomaly_model.pkl'))
    return coll_model, cong_model, risk_model, anomaly_model

coll_model, cong_model, risk_model, anomaly_model = load_models()

# --- Load Historical Data ---
@st.cache_data
def load_historical_data():
    return pd.read_csv("synthetic_traffic_data.csv")

hist_data = load_historical_data()

# --- Sidebar Controls ---
st.sidebar.title("Cyber-Physical Controls")
st.sidebar.markdown("Adjust physical layer parameters to simulate real-time cyber layer response.")

sim_speed = st.sidebar.slider("Avg Vehicle Speed (km/h)", 0, 150, 80)
sim_flow = st.sidebar.slider("Traffic Flow (veh/hr)", 100, 6000, 2500)
sim_occupancy = st.sidebar.slider("Lane Occupancy (%)", 0.0, 1.0, 0.4)
sim_weather = st.sidebar.selectbox("Weather Condition", ['Clear', 'Rain', 'Fog', 'Snow'])
sim_braking = st.sidebar.slider("Sudden Braking Events", 0, 50, 5)

# Simulate current state DataFrame
current_state = pd.DataFrame([{
    'timestamp': datetime.now(),
    'traffic_flow_rate': sim_flow,
    'lane_occupancy_ratio': sim_occupancy,
    'vehicle_speed_avg': sim_speed,
    'weather_condition': sim_weather,
    'sudden_braking_events': sim_braking
}])

current_features = prepare_features(current_state)
# Ensure columns match training
for col in coll_model.feature_names_in_:
    if col not in current_features.columns:
         current_features[col] = 0
current_features = current_features[coll_model.feature_names_in_]

# --- Predictions ---
pred_collision = coll_model.predict(current_features)[0]
pred_congestion = cong_model.predict(current_features)[0]
pred_risk = risk_model.predict(current_features)[0]
anomaly_score = anomaly_model.predict(current_features)[0] # -1 for anomaly, 1 for normal

# --- Main Dashboard ---
st.title("🚦 AI-Powered Digital Twin for Traffic Safety")
st.markdown("""
This dashboard represents the **Cyber Layer** of a Cyber-Physical Transportation System. 
It ingests real-time **Physical Layer** data, predicts safety risks using Machine Learning, 
and provides incident analysis via Large Language Models.
""")

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Speed", f"{sim_speed} km/h", f"{sim_speed - 80} from avg" if sim_speed != 80 else "0")
col2.metric("Traffic Flow", f"{sim_flow} v/h", f"{sim_flow - 2000} from avg" if sim_flow != 2000 else "0")
col3.metric("Lane Occupancy", f"{sim_occupancy * 100:.1f}%")
anomaly_status = "Normal" if anomaly_score == 1 else "ANOMALY DETECTED"
col4.metric("System Status", anomaly_status, delta_color="inverse" if anomaly_score == -1 else "normal")

st.divider()

# --- Digital Twin Visualization (Plotly) ---
st.subheader("🌐 Digital Twin Environment State")
vis_col1, vis_col2 = st.columns([2, 1])

with vis_col1:
    # 3D visualization or complex 2D using plotly to simulate the twin concept
    # Here we show a risk scatter based on historical context highlighting current state
    fig = px.scatter(
        hist_data.sample(500), 
        x="traffic_flow_rate", 
        y="vehicle_speed_avg", 
        color="risk_category",
        size="lane_occupancy_ratio",
        hover_data=['weather_condition'],
        color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'},
        title="Physical Layer System Space (Historical Context)"
    )
    # Add current state marker
    fig.add_trace(go.Scatter(
        x=[sim_flow], y=[sim_speed], 
        mode='markers+text', 
        marker=dict(color='black', size=20, symbol='star'),
        text=["Current State"], textposition="top center",
        name="Current State"
    ))
    st.plotly_chart(fig, use_container_width=True)

with vis_col2:
    st.markdown("### Risk Prediction Engine")
    
    # Gauges for probability
    fig_coll = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = pred_collision * 100,
        title = {'text': "Collision Probability (%)"},
        gauge = {'axis': {'range': [None, 100]},
                 'bar': {'color': "darkred" if pred_collision > 0.6 else "orange" if pred_collision > 0.3 else "green"}}
    ))
    fig_coll.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_coll, use_container_width=True)

    fig_cong = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = pred_congestion * 100,
        title = {'text': "Congestion Level (%)"},
        gauge = {'axis': {'range': [None, 100]},
                 'bar': {'color': "darkred" if pred_congestion > 0.8 else "orange" if pred_congestion > 0.5 else "green"}}
    ))
    fig_cong.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_cong, use_container_width=True)
    
    risk_color = "red" if pred_risk == 'High' else "orange" if pred_risk == 'Medium' else "green"
    st.markdown(f"### Predicted Risk Category: <span style='color:{risk_color}'>{pred_risk}</span>", unsafe_allow_html=True)

st.divider()

# --- LLM Incident Analysis ---
st.subheader("🧠 LLM-Based Incident Analysis & Mitigation")

context_data = {
    'speed': sim_speed,
    'occupancy': sim_occupancy,
    'flow': sim_flow,
    'weather': sim_weather,
    'braking': sim_braking,
    'collision_prob': pred_collision,
    'congestion': pred_congestion,
    'risk': pred_risk
}

with st.spinner("Generating automated safety analysis..."):
    llm_report = generate_incident_analysis(context_data)
    st.markdown(llm_report)

st.divider()

# --- Adaptive Signal Recommendation ---
st.subheader("🚥 Cyber-to-Physical Actuation: Signal Recommendation")
st.markdown("Based on current system state, the digital twin recommends the following physical interventions:")

if pred_congestion > 0.7:
    st.info("🔄 **Increase Main Arterial Green Time:** Extend Phase A by 15 seconds to flush congestion.")
elif pred_risk == 'High':
    st.error("🛑 **Implement Safety Protocols:** All-red clearance interval extended by 2 seconds. Reduce speed limits on Variable Message Signs.")
else:
    st.success("✅ **Normal Operations:** Maintain current actuated signal timing plan.")

# --- Data Upload/Download ---
st.sidebar.divider()
st.sidebar.markdown("### Data Management")
csv = hist_data.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="Download Synthetic Dataset",
    data=csv,
    file_name='traffic_digital_twin_data.csv',
    mime='text/csv',
)
uploaded_file = st.sidebar.file_uploader("Upload custom traffic CSV", type=['csv'])
if uploaded_file is not None:
    st.sidebar.success("Custom data loaded! (Simulation mode disabled)")
    # Logic to switch to historical analysis would go here.
