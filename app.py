# app.py
"""
EV Innovate - Final Streamlit app (integrated with your Linear Regression model)
Features:
- Login screen (demo)
- Navigation: Home | Analytics | AI Assistant
- Upload dataset
- Sidebar filters + Multi-Policy Simulator
- Model integration (Linear Regression + scaler)
- Prediction shown as Metric Card
- avg_cost_ev default = dataset mean (editable)
- total_vehicles_registered default = dataset mean (not editable)
- AI Assistant (TF-IDF retrieval)
- Dark/Light theme toggle
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import plotly.express as px

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="EV Innovate – Analytics & Forecasting",
                   page_icon="⚡", layout="wide")

MODEL_PATH = "models/ev_policy_best_model.pkl"
SCALER_PATH = "models/scaler.pkl"

# ----------------------------
# Load model & scaler
# ----------------------------
@st.cache_resource
def load_model_and_scaler():
    model = None
    scaler = None
    msg = ""
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            msg += f"Loaded model from {MODEL_PATH}. "
        except Exception as e:
            msg += f"Failed to load model: {e}. "
    else:
        msg += f"No model file at {MODEL_PATH}. "
    if os.path.exists(SCALER_PATH):
        try:
            scaler = joblib.load(SCALER_PATH)
            msg += f"Loaded scaler from {SCALER_PATH}."
        except Exception as e:
            msg += f"Failed to load scaler: {e}."
    else:
        msg += f"No scaler file at {SCALER_PATH}."
    return model, scaler, msg

model, scaler, model_status = load_model_and_scaler()

# ----------------------------
# Session state
# ----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True
if "df" not in st.session_state:
    st.session_state.df = None
if "tfidf" not in st.session_state:
    st.session_state.tfidf = None
if "tfidf_matrix" not in st.session_state:
    st.session_state.tfidf_matrix = None
if "corpus" not in st.session_state:
    st.session_state.corpus = None

# ----------------------------
# CSS themes
# ----------------------------
LIGHT_CSS = """
<style>
body { background-color: #f6f8fa; color: #0b1220; }
.header-title { font-size: 36px; font-weight: 800; color: #0c8f34; margin-bottom: 0; }
.header-sub { font-size: 16px; color: #566170; margin-top: 0; }
.upload-box { border: 2px dashed #dbeafe; padding: 28px; border-radius: 12px; background:white; }
.card { background: white; padding: 16px; border-radius: 10px; border:1px solid #e6edf3; }
.metric { font-size:28px; font-weight:700; margin-bottom:6px; }
.small { color:#6b7280; }
</style>
"""

DARK_CSS = """
<style>
body { background-color: #0b0f14; color: #e6eef6; }
.header-title { font-size: 36px; font-weight: 800; color: #66e07f; margin-bottom: 0; }
.header-sub { font-size: 16px; color: #a9b6c2; margin-top: 0; }
.upload-box { border: 2px dashed #15324b; padding: 28px; border-radius: 12px; background:#071018; }
.card { background: #06141a; padding: 16px; border-radius: 10px; border:1px solid #12323f; }
.metric { font-size:28px; font-weight:700; margin-bottom:6px; color:#e6f4ea; }
.small { color:#9fb0bf; }
</style>
"""

# ----------------------------
# LOGIN
# ----------------------------
def login_screen():
    st.sidebar.markdown("## 🔒 Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if username == "admin" and password == "password":
            st.session_state.logged_in = True
            st.sidebar.success("Logged in as admin")
        else:
            st.sidebar.error("Invalid credentials (demo: admin / password)")

# ----------------------------
# Navigation header
# ----------------------------
def nav_header():
    cols = st.columns([1, 4, 1])
    with cols[1]:
        st.markdown('<p class="header-title">Data Analytics & Forecasting</p>', unsafe_allow_html=True)
        st.markdown('<p class="header-sub">Upload EV data to generate insights, simulations and predictions</p>', unsafe_allow_html=True)
    # theme toggle
    theme = st.checkbox("Dark Theme", value=st.session_state.dark_mode)
    st.session_state.dark_mode = theme

# ----------------------------
# Upload
# ----------------------------
def upload_area():
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload dataset (CSV or Excel)", type=["csv", "xlsx"])
    st.markdown('</div>', unsafe_allow_html=True)
    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            st.session_state.df = df
            st.success(f"Loaded `{uploaded.name}` — rows: {df.shape[0]}, cols: {df.shape[1]}")
        except Exception as e:
            st.error(f"Failed to load file: {e}")

# ----------------------------
# Sidebar filters + simulator sliders
# ----------------------------
def sidebar_controls():
    st.sidebar.header("Filters & Policy Simulator")
    df = st.session_state.df
    country = None
    year = None
    if df is not None:
        if "country" in df.columns:
            countries = ["All"] + sorted(df["country"].dropna().unique().tolist())
            country = st.sidebar.selectbox("Country", countries)
        if "year" in df.columns:
            years = ["All"] + sorted(df["year"].dropna().unique().tolist())
            year = st.sidebar.selectbox("Year", years)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Policy Simulator (multi-policy)")
    incentive = st.sidebar.slider("Government Incentive (USD)", 0, 20000, 3000, step=500)
    charging_add = st.sidebar.slider("Add Charging Stations", 0, 5000, 100, step=10)
    tax_change = st.sidebar.slider("Tax Incentive (%)", -50, 50, 0, step=1)
    fuel_price = st.sidebar.slider("Fuel Price (USD per liter)", 0.2, 5.0, 1.5, step=0.1)
    elec_price = st.sidebar.slider("Electricity Price (USD/kWh)", 0.05, 1.0, 0.15, step=0.01)
    st.sidebar.markdown("---")
    st.sidebar.caption("Prediction uses loaded model + scaler.")
    return {"country": country, "year": year, "incentive": incentive,
            "charging_add": charging_add, "tax_change": tax_change,
            "fuel_price": fuel_price, "elec_price": elec_price}

# ----------------------------
# AI assistant: build TF-IDF index
# ----------------------------
def build_tfidf(df):
    # create simple corpus from rows
    corpus = []
    for _, row in df.iterrows():
        parts = []
        for c in df.columns:
            parts.append(f"{c}: {row[c]}")
        corpus.append(" | ".join(parts))
    tfidf = TfidfVectorizer(stop_words="english", max_features=3000)
    mat = tfidf.fit_transform(corpus)
    return tfidf, mat, corpus

def ai_query(q, top_k=3):
    if st.session_state.df is None:
        return "Upload dataset first."
    if st.session_state.tfidf is None:
        tfidf, mat, corpus = build_tfidf(st.session_state.df)
        st.session_state.tfidf = tfidf
        st.session_state.tfidf_matrix = mat
        st.session_state.corpus = corpus
    tfidf = st.session_state.tfidf
    mat = st.session_state.tfidf_matrix
    corpus = st.session_state.corpus
    qv = tfidf.transform([q])
    sims = linear_kernel(qv, mat).flatten()
    idx = sims.argsort()[:-top_k-1:-1]
    results = []
    for i in idx:
        if sims[i] > 0:
            results.append(f"Score {sims[i]:.2f} → {corpus[i]}")
    return "\n\n".join(results) if results else "No relevant rows found."

# ----------------------------
# Prediction function (uses exact feature order)
# ----------------------------
MODEL_FEATURES = [
    "total_vehicles_registered",
    "ev_percentage_share",
    "charging_stations_count",
    "avg_cost_ev",
    "avg_cost_gasoline_vehicle",
    "gov_incentive_amount",
    "co2_emissions_per_vehicle",
    "fuel_price_per_liter",
    "electricity_price_per_kWh"
]

def predict_from_inputs(inputs: dict):
    if model is None or scaler is None:
        return None, "Model or scaler not loaded."
    # Build row in correct order
    row = {k: inputs.get(k, 0) for k in MODEL_FEATURES}
    X = pd.DataFrame([row])[MODEL_FEATURES]
    try:
        Xs = scaler.transform(X)
    except Exception as e:
        return None, f"Scaler transform failed: {e}"
    try:
        pred = model.predict(Xs)[0]
        return pred, None
    except Exception as e:
        return None, f"Model predict failed: {e}"

# ----------------------------
# UI cards
# ----------------------------
def overview_cards(df):
    total_ev = int(df['ev_vehicles_registered'].sum()) if 'ev_vehicles_registered' in df.columns else int(df.shape[0]*10)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric">{total_ev:,}</div>', unsafe_allow_html=True)
        st.markdown('<div class="small">Total EVs Registered</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric">{int(df.shape[0])}</div>', unsafe_allow_html=True)
        st.markdown('<div class="small">Dataset Rows</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        avg_cost = int(df['avg_cost_ev'].mean()) if 'avg_cost_ev' in df.columns else 0
        st.markdown(f'<div class="metric">${avg_cost:,}</div>', unsafe_allow_html=True)
        st.markdown('<div class="small">Avg EV Cost</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric">{int(df["charging_stations_count"].sum()) if "charging_stations_count" in df.columns else 0:,}</div>', unsafe_allow_html=True)
        st.markdown('<div class="small">Total Charging Stations</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# Pages
# ----------------------------
def page_home():
    nav_header()
    if st.session_state.dark_mode:
        st.markdown(DARK_CSS, unsafe_allow_html=True)
    else:
        st.markdown(LIGHT_CSS, unsafe_allow_html=True)
    st.markdown("---")
    upload_area()
    st.markdown("---")
    if st.session_state.df is not None:
        df = st.session_state.df
        overview_cards(df)
        st.markdown("---")
        st.subheader("EV Time Series (sample numeric column)")
        numeric = df.select_dtypes(include=np.number)
        if not numeric.empty:
            first_num = numeric.columns[0]
            fig = px.line(numeric.reset_index(), y=first_num)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Upload dataset with numeric columns to visualize.")
    else:
        st.info("Upload dataset on Home to begin. Use admin/password to login for model features.")

def page_analytics():
    nav_header()
    if st.session_state.dark_mode:
        st.markdown(DARK_CSS, unsafe_allow_html=True)
    else:
        st.markdown(LIGHT_CSS, unsafe_allow_html=True)
    st.markdown("---")
    if st.session_state.df is None:
        st.warning("Upload dataset on Home first.")
        return
    df = st.session_state.df.copy()
    filters = sidebar_controls()
    st.markdown("---")
    # apply filters
    if filters["country"] and filters["country"] != "All" and "country" in df.columns:
        df = df[df["country"] == filters["country"]]
    if filters["year"] and filters["year"] != "All" and "year" in df.columns:
        df = df[df["year"] == filters["year"]]
    overview_cards(df)
    st.markdown("---")
    with st.expander("Preview data (first 10 rows)"):
        st.dataframe(df.head(10))
    with st.expander("Summary & correlation"):
        st.write(df.describe(include='all'))

        st.subheader("Correlation with avg_cost_ev")

        numeric_df = df.select_dtypes(include=[np.number])

        if 'avg_cost_ev' in numeric_df.columns:
        st.write(numeric_df.corr()['avg_cost_ev'].sort_values(ascending=False))
        else:
        st.info("avg_cost_ev is not numeric or is missing from numeric columns.")
    st.markdown("---")
    st.subheader("Policy Simulator — Multi-Policy")
    st.write("Adjust policy levers (sidebar) then refine here and run prediction.")
    # defaults from df means
    total_vehicles_default = int(df['total_vehicles_registered'].mean()) if 'total_vehicles_registered' in df.columns else 100000
    avg_cost_ev_default = float(df['avg_cost_ev'].mean()) if 'avg_cost_ev' in df.columns else 30000.0

    # show user-adjustable controls for the two fields:
    st.markdown("### Base inputs (you can adjust avg_cost_ev):")
    st.write(f"**total_vehicles_registered** will default to dataset mean: {total_vehicles_default} (not editable)")
    avg_cost_ev_val = st.number_input("avg_cost_ev (you may override default)", value=avg_cost_ev_default, step=500.0)

    # assemble final input dict (use sidebar sliders too)
    input_for_model = {
        "total_vehicles_registered": total_vehicles_default,  # chosen: default average
        "ev_percentage_share": df['ev_percentage_share'].mean() if 'ev_percentage_share' in df.columns else 1.0,
        "charging_stations_count": (df['charging_stations_count'].mean() + filters["charging_add"]) if 'charging_stations_count' in df.columns else filters["charging_add"],
        "avg_cost_ev": avg_cost_ev_val,
        "avg_cost_gasoline_vehicle": df['avg_cost_gasoline_vehicle'].mean() if 'avg_cost_gasoline_vehicle' in df.columns else 20000,
        "gov_incentive_amount": filters["incentive"],
        "co2_emissions_per_vehicle": df['co2_emissions_per_vehicle'].mean() if 'co2_emissions_per_vehicle' in df.columns else 120,
        "fuel_price_per_liter": filters["fuel_price"],
        "electricity_price_per_kWh": filters["elec_price"]
    }

    st.write("**Preview input used for prediction:**")
    st.json(input_for_model)

    if st.button("🔮 Run Simulation & Predict EV Registrations"):
        pred, err = predict_from_inputs(input_for_model)
        if err:
            st.error(err)
        else:
            st.success("Prediction complete")
            st.metric(label="Estimated EV Registrations (annual)", value=f"{int(pred):,}")
            # baseline (no incentive)
            baseline = input_for_model.copy()
            baseline["gov_incentive_amount"] = 0
            baseline_pred, _ = predict_from_inputs(baseline)
            series = pd.DataFrame({
                "Scenario": ["Baseline (0 incentive)", "Current Simulation"],
                "Predicted": [int(baseline_pred) if baseline_pred else 0, int(pred)]
            })
            fig = px.bar(series, x="Scenario", y="Predicted", text="Predicted")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    # scatter if columns present
    if 'charging_stations_count' in df.columns and 'ev_vehicles_registered' in df.columns:
        fig = px.scatter(df, x='charging_stations_count', y='ev_vehicles_registered', trendline="ols")
        st.plotly_chart(fig, use_container_width=True)

def page_ai_assistant():
    nav_header()
    if st.session_state.dark_mode:
        st.markdown(DARK_CSS, unsafe_allow_html=True)
    else:
        st.markdown(LIGHT_CSS, unsafe_allow_html=True)
    st.markdown("---")
    st.header("AI Assistant — Dataset Retriever")
    st.write("Ask questions and the assistant will return matching dataset rows and context.")
    if st.session_state.df is None:
        st.warning("Upload dataset on Home first.")
        return
    q = st.text_input("Ask a question about your dataset")
    k = st.slider("Results to return", 1, 6, 3)
    if st.button("Ask"):
        with st.spinner("Searching..."):
            resp = ai_query(q, top_k=k)
        st.markdown("**Assistant Response:**")
        st.code(resp)

# ----------------------------
# Main
# ----------------------------
def main():
    # login
    if not st.session_state.logged_in:
        login_screen()

    # navigation menu
    menu = ["Home", "Analytics", "AI Assistant"]
    choice = st.sidebar.selectbox("Navigation", menu)

    # theme CSS
    if st.session_state.dark_mode:
        st.markdown(DARK_CSS, unsafe_allow_html=True)
    else:
        st.markdown(LIGHT_CSS, unsafe_allow_html=True)

    # model status in sidebar
    st.sidebar.markdown("## Model Status")
    st.sidebar.write(model_status)
    if model is None or scaler is None:
        st.sidebar.warning("Model or scaler not loaded. Place files in /models or run training.")
    else:
        st.sidebar.success("Model & Scaler loaded ✓")

    # route pages
    if choice == "Home":
        page_home()
    elif choice == "Analytics":
        page_analytics()
    elif choice == "AI Assistant":
        page_ai_assistant()

if __name__ == "__main__":
    main()
