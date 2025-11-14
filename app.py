# app.py
"""
EV Innovate - Full-featured Streamlit dashboard (Option C: Green EV Theme)
Features:
 - Top navigation (Home, Analytics, AI Assistant, Simulator, Settings)
 - Login (demo)
 - Upload CSV/XLSX
 - Hybrid Analytics: EV-specific + dynamic analytics
 - Dataset health, missing heatmap, column profiling
 - Policy Simulator (multi-policy) + prediction using Linear Regression
 - TF-IDF based AI Assistant (dataset retriever) with chat UI
 - Dark/Light theme toggle
Notes:
 - Place model/scaler in ./models/ev_policy_best_model.pkl and ./models/scaler.pkl
 - Required libs: streamlit, pandas, numpy, scikit-learn, joblib, plotly, matplotlib,
   seaborn, sklearn, statsmodels (if using trendline), scipy pinned per instructions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io
from typing import Tuple, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import plotly.express as px
import matplotlib.pyplot as plt

# ----------------------------
# Config & paths
# ----------------------------
st.set_page_config(page_title="EV Innovate – Dashboard",
                   page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")

MODEL_PATH = "models/ev_policy_best_model.pkl"
SCALER_PATH = "models/scaler.pkl"

# ----------------------------
# Theme CSS (Green Gradient)
# ----------------------------
BASE_CSS = """
<style>
:root {
  --card-radius: 12px;
  --glass: rgba(255,255,255,0.03);
}
body { font-family: "Inter", sans-serif; }
.header {
  background: linear-gradient(90deg, #0f9d58, #34a853);
  color: white;
  padding: 18px 28px;
  border-radius: 12px;
  box-shadow: 0 6px 18px rgba(10,10,10,0.08);
}
.header-title { font-size: 28px; font-weight:700; margin:0; }
.header-sub { margin:0; opacity:0.95; }
.upload-box {
  border: 2px dashed rgba(255,255,255,0.08);
  padding: 28px;
  border-radius: 12px;
  text-align: center;
  background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
}
.card {
  background: white;
  padding: 14px;
  border-radius: var(--card-radius);
  box-shadow: 0 6px 18px rgba(12,16,20,0.04);
}
.dark .card {
  background: #081217;
  color: #e6eef6;
  border: 1px solid rgba(255,255,255,0.03);
}
.kpi {
  font-size: 26px;
  font-weight: 800;
  margin-bottom: 6px;
}
.kpi-sub { color: #6b7280; font-size:12px; }
.small { color:#6b7280; font-size:12px; }
.footer { color:#a0a8b5; font-size:12px; margin-top:18px; }
</style>
"""

DARK_OVERRIDES = """
<style>
body { background: #071018; color: #e6eef6; }
</style>
"""

# ----------------------------
# Session state defaults
# ----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "df" not in st.session_state:
    st.session_state.df = None
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True
if "tfidf" not in st.session_state:
    st.session_state.tfidf = None
if "tfidf_matrix" not in st.session_state:
    st.session_state.tfidf_matrix = None
if "corpus" not in st.session_state:
    st.session_state.corpus = None

# ----------------------------
# Utility: load model + scaler
# ----------------------------
@st.cache_resource
def load_model_and_scaler() -> Tuple[object, object, str]:
    model = None
    scaler = None
    msg = ""
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            msg += "Model loaded. "
        except Exception as e:
            msg += f"Failed to load model: {e}. "
    else:
        msg += f"No model at {MODEL_PATH}. "
    if os.path.exists(SCALER_PATH):
        try:
            scaler = joblib.load(SCALER_PATH)
            msg += "Scaler loaded."
        except Exception as e:
            msg += f"Failed to load scaler: {e}."
    else:
        msg += f"No scaler at {SCALER_PATH}."
    return model, scaler, msg

model, scaler, model_status = load_model_and_scaler()

# ----------------------------
# Small helpers
# ----------------------------
def safe_display_df(df: pd.DataFrame, n=5):
    """Return safe truncated df for display (avoid huge output)."""
    return df.head(n)

def guess_year_column(df: pd.DataFrame):
    for c in df.columns:
        if "year" in c.lower() or "date" in c.lower():
            return c
    return None

def ev_likeness_score(df: pd.DataFrame) -> float:
    ev_keywords = {"ev", "electric", "charging", "co2", "incentive", "stations", "vehicles", "percentage", "registered"}
    score = 0
    for col in df.columns:
        for kw in ev_keywords:
            if kw in col.lower():
                score += 1
    # normalized to percent of columns
    return round((score / max(1, len(df.columns))) * 100, 2)

def build_tfidf_index(df: pd.DataFrame):
    corpus = []
    for _, row in df.iterrows():
        parts = []
        for c in df.columns:
            try:
                parts.append(f"{c}: {row[c]}")
            except Exception:
                parts.append(f"{c}: ")
        corpus.append(" | ".join(parts))
    tfidf = TfidfVectorizer(stop_words="english", max_features=3000)
    mat = tfidf.fit_transform(corpus)
    return tfidf, mat, corpus

def ai_query(q: str, top_k: int = 3) -> str:
    if st.session_state.df is None:
        return "Please upload a dataset first."
    if st.session_state.tfidf is None:
        tfidf, mat, corpus = build_tfidf_index(st.session_state.df)
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
    return "\n\n".join(results) if results else "No similar rows found."

# ----------------------------
# Prediction util (uses exact model features)
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

def predict_with_model(input_map: Dict[str, float]):
    if model is None or scaler is None:
        return None, "Model or scaler not present."
    # Build row with exact order
    row = {k: input_map.get(k, 0.0) for k in MODEL_FEATURES}
    X = pd.DataFrame([row])[MODEL_FEATURES]
    # Ensure numeric dtype
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    try:
        Xs = scaler.transform(X)
    except Exception as e:
        return None, f"Scaler transform error: {e}"
    try:
        pred = model.predict(Xs)[0]
        return float(pred), None
    except Exception as e:
        return None, f"Model prediction error: {e}"

# ----------------------------
# Dataset health & profiling helpers
# ----------------------------
def dataset_health_cards(df: pd.DataFrame):
    total_rows = int(len(df))
    total_cols = int(len(df.columns))
    missing = int(df.isnull().sum().sum())
    dupes = int(df.duplicated().sum())
    numeric_count = int(len(df.select_dtypes(include=[np.number]).columns))
    categorical_count = int(len(df.select_dtypes(include=['object', 'category']).columns))
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", total_rows)
    c2.metric("Columns", total_cols)
    c3.metric("Missing values", missing)
    c4, c5 = st.columns(2)
    c4.metric("Duplicate rows", dupes)
    c5.metric("Numeric columns", numeric_count)
    st.caption(f"Categorical columns: {categorical_count}")

def missing_heatmap(df: pd.DataFrame):
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(10, 3))
    sns.heatmap(df.isnull(), cbar=False, cmap="Blues", ax=ax)
    ax.set_xlabel("")
    st.pyplot(fig)

def column_profile(df: pd.DataFrame):
    info = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "missing_%": [round(df[c].isnull().mean() * 100, 2) for c in df.columns],
        "unique": [int(df[c].nunique(dropna=True)) for c in df.columns]
    })
    st.dataframe(info, use_container_width=True)

def suggested_visualizations(df: pd.DataFrame):
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    st.subheader("Suggested visualizations")
    if len(numeric) >= 1:
        col = numeric[0]
        st.write(f"Distribution of **{col}**")
        fig = px.histogram(df, x=col, nbins=30)
        st.plotly_chart(fig, use_container_width=True)
    if len(numeric) >= 2:
        st.write(f"Scatter: **{numeric[0]}** vs **{numeric[1]}**")
        fig = px.scatter(df, x=numeric[0], y=numeric[1])
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# UI: Login & header
# ----------------------------
def login_box():
    st.sidebar.markdown("## 🔒 Login (demo)")
    u = st.sidebar.text_input("Username")
    p = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if u == "admin" and p == "password":
            st.session_state.logged_in = True
            st.sidebar.success("Logged in as admin")
        else:
            st.sidebar.error("Invalid credentials (demo: admin/password)")

def top_header():
    st.markdown(BASE_CSS, unsafe_allow_html=True)
    if st.session_state.dark_mode:
        st.markdown(DARK_OVERRIDES, unsafe_allow_html=True)
        st.markdown('<div class="header"><div style="display:flex;justify-content:space-between;align-items:center;"><div><p class="header-title">EV Innovate</p><p class="header-sub">Green Policy Simulator & EV Analytics</p></div><div style="text-align:right"><small>Model status:</small><br><strong>'+model_status+'</strong></div></div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="header"><div style="display:flex;justify-content:space-between;align-items:center;"><div><p class="header-title">EV Innovate</p><p class="header-sub">Green Policy Simulator & EV Analytics</p></div><div style="text-align:right"><small>Model status:</small><br><strong>'+model_status+'</strong></div></div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# ----------------------------
# Upload area
# ----------------------------
def upload_area():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Upload dataset (CSV or Excel)")
    uploaded = st.file_uploader("", type=["csv", "xlsx"])
    if uploaded is not None:
        try:
            if str(uploaded.name).lower().endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            st.session_state.df = df
            st.success(f"Loaded `{uploaded.name}` — rows: {df.shape[0]}, cols: {df.shape[1]}")
        except Exception as e:
            st.error(f"Failed to load file: {e}")
    else:
        st.info("No file uploaded. Use sample dataset or upload your own CSV/Excel.")
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# Sidebar: theme toggle + help
# ----------------------------
def sidebar_area():
    st.sidebar.markdown("## Controls")
    st.sidebar.checkbox("Dark theme", value=st.session_state.dark_mode, key="theme_toggle")
    st.session_state.dark_mode = st.session_state.theme_toggle
    login_box()
    with st.sidebar.expander("❓ How to use"):
        st.write("""
- Upload any CSV/Excel dataset.
- If your data contains EV-specific columns the EV dashboard activates.
- Use the Policy Simulator to create scenarios and predict EV registrations.
- Visit AI Assistant to ask data questions (retrieval-based).
""")

# ----------------------------
# Hybrid analytics page (complete)
# ----------------------------
def analytics_page():
    if st.session_state.df is None:
        st.info("Upload a dataset on Home to view analytics.")
        return

    df = st.session_state.df.copy()
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Dataset Health Overview")
    dataset_health_cards(df)

    # EV-likeness and quick stats
    score = ev_likeness_score(df)
    st.info(f"EV-likeness score: **{score}%**")

    # column profile & missing heatmap collapsible
    with st.expander("Column profile & missing values"):
        column_profile(df)
        if df.isnull().sum().sum() > 0:
            missing_heatmap(df)
        else:
            st.success("No missing values detected.")

    # Detect EV dataset
    ev_required = {
        "total_vehicles_registered",
        "ev_vehicles_registered",
        "ev_percentage_share",
        "charging_stations_count",
        "avg_cost_ev",
        "avg_cost_gasoline_vehicle",
        "gov_incentive_amount",
        "co2_emissions_per_vehicle",
        "fuel_price_per_liter",
        "electricity_price_per_kWh"
    }
    df_cols = set([c.lower() for c in df.columns])
    is_ev = all(req in df_cols for req in [c.lower() for c in ev_required])

    if is_ev:
        st.success("EV dataset detected — showing EV-specific analytics")
        # Quick KPIs
        try:
            overview_kpis(df)
        except Exception:
            # fallback
            st.write("KPIs not available due to missing columns.")
        # correlation w/ avg_cost_ev (numeric only)
        st.markdown("---")
        st.subheader("Correlation with avg_cost_ev")
        numeric = df.select_dtypes(include=[np.number])
        if "avg_cost_ev" in numeric.columns:
            st.write(numeric.corr()["avg_cost_ev"].sort_values(ascending=False))
        else:
            st.info("avg_cost_ev not numeric or missing.")

        # scatter plot
        st.markdown("---")
        st.subheader("Charging Stations vs EV Registrations")
        if "charging_stations_count" in df.columns and "ev_vehicles_registered" in df.columns:
            fig = px.scatter(df, x="charging_stations_count", y="ev_vehicles_registered", trendline="ols")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Required columns for this plot are missing.")
    else:
        st.warning("Non-EV dataset detected — switching to generic analytics mode.")
        st.markdown("---")
        st.subheader("Dataset Summary")
        st.write(df.describe(include="all"))

        numeric_cols = df.select_dtypes(include=[np.number])
        if numeric_cols.shape[1] >= 2:
            st.subheader("Numeric Correlation Matrix")
            st.write(numeric_cols.corr())
        else:
            st.info("Not enough numeric columns for correlation matrix.")

        st.markdown("---")
        suggested_visualizations(df)
    st.markdown('</div>', unsafe_allow_html=True)

# helper to show KPIs (EV)
def overview_kpis(df):
    st.markdown('<div style="display:flex;gap:14px;">', unsafe_allow_html=True)
    # Total EVs Registered
    if "ev_vehicles_registered" in df.columns:
        total_ev = int(df["ev_vehicles_registered"].sum())
    else:
        total_ev = int(df.shape[0] * 10)
    # Average EV cost
    avg_cost = int(df["avg_cost_ev"].mean()) if "avg_cost_ev" in df.columns else 0
    # Charging stations
    stations = int(df["charging_stations_count"].sum()) if "charging_stations_count" in df.columns else 0
    # Incentives average
    incentive = int(df["gov_incentive_amount"].mean()) if "gov_incentive_amount" in df.columns else 0

    st.markdown(f'<div class="card"><div class="kpi">{total_ev:,}</div><div class="kpi-sub">Total EVs Registered</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="card"><div class="kpi">${avg_cost:,}</div><div class="kpi-sub">Avg EV Cost</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="card"><div class="kpi">{stations:,}</div><div class="kpi-sub">Charging Stations</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="card"><div class="kpi">${incentive:,}</div><div class="kpi-sub">Avg Gov Incentive</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# Policy Simulator page
# ----------------------------
def simulator_page():
    if st.session_state.df is None:
        st.info("Upload dataset on Home to use the Policy Simulator.")
        return

    df = st.session_state.df.copy()
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.title("Policy Simulator — Multi-Policy Scenario")

    # sidebar-like controls in main area for better UI
    st.subheader("Adjust policy levers")
    c1, c2 = st.columns(2)
    with c1:
        incentive = st.number_input("Government Incentive (USD)", min_value=0, max_value=50000, value=int(df["gov_incentive_amount"].mean() if "gov_incentive_amount" in df.columns else 3000), step=500)
        add_stations = st.number_input("Add Charging Stations", min_value=0, max_value=20000, value=100, step=10)
    with c2:
        tax = st.slider("Tax Incentive (%)", -50, 100, 0)
        fuel_price = st.number_input("Fuel Price (USD per liter)", value=float(df["fuel_price_per_liter"].mean() if "fuel_price_per_liter" in df.columns else 1.5))
        elec_price = st.number_input("Electricity Price (USD/kWh)", value=float(df["electricity_price_per_kWh"].mean() if "electricity_price_per_kWh" in df.columns else 0.15))

    # defaults for model-required fields
    total_vehicles_default = int(df["total_vehicles_registered"].mean()) if "total_vehicles_registered" in df.columns else 100000
    ev_pct_default = float(df["ev_percentage_share"].mean()) if "ev_percentage_share" in df.columns else 1.0
    avg_cost_ev_default = float(df["avg_cost_ev"].mean()) if "avg_cost_ev" in df.columns else 30000.0
    gas_cost_default = float(df["avg_cost_gasoline_vehicle"].mean()) if "avg_cost_gasoline_vehicle" in df.columns else 25000.0
    co2_default = float(df["co2_emissions_per_vehicle"].mean()) if "co2_emissions_per_vehicle" in df.columns else 120.0
    charging_base = float(df["charging_stations_count"].mean()) if "charging_stations_count" in df.columns else 1000.0

    st.markdown("---")
    st.subheader("Model Input Preview (editable)")
    input_map = {
        "total_vehicles_registered": total_vehicles_default,
        "ev_percentage_share": ev_pct_default,
        "charging_stations_count": charging_base + add_stations,
        "avg_cost_ev": st.number_input("avg_cost_ev (editable)", value=avg_cost_ev_default, step=500.0),
        "avg_cost_gasoline_vehicle": gas_cost_default,
        "gov_incentive_amount": incentive,
        "co2_emissions_per_vehicle": co2_default,
        "fuel_price_per_liter": fuel_price,
        "electricity_price_per_kWh": elec_price
    }

    st.json(input_map)

    if st.button("🔮 Run Simulation & Predict EV Registrations"):
        pred, err = predict_with_model(input_map)
        if err:
            st.error(err)
        else:
            st.success("Prediction complete")
            st.metric(label="Estimated EV Registrations (annual)", value=f"{int(pred):,}")
            # Compare baseline with zero incentive
            baseline_map = input_map.copy()
            baseline_map["gov_incentive_amount"] = 0
            baseline_pred, _ = predict_with_model(baseline_map)
            scenario_df = pd.DataFrame({
                "Scenario": ["Baseline (0 incentive)", "Simulation"],
                "Predicted": [int(baseline_pred) if baseline_pred else 0, int(pred)]
            })
            fig = px.bar(scenario_df, x="Scenario", y="Predicted", text="Predicted")
            st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# AI Assistant page (chat style)
# ----------------------------
def ai_assistant_page():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("AI Assistant — Dataset Retriever (TF-IDF)")

    if st.session_state.df is None:
        st.info("Upload dataset on Home to use the assistant.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # simple chat UI
    query = st.text_input("Ask a question about your dataset (e.g., 'highest EV registrations')")
    k = st.slider("Results to return", 1, 6, 3)
    if st.button("Ask"):
        with st.spinner("Searching dataset..."):
            resp = ai_query(query, top_k=k)
        st.markdown("**Assistant response:**")
        st.code(resp)
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# Home page
# ----------------------------
def home_page():
    top_header()
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Upload or load sample dataset")
    upload_area()
    st.markdown("---")
    if st.session_state.df is not None:
        st.subheader("Quick Preview")
        st.dataframe(safe_display_df(st.session_state.df, n=6), use_container_width=True)
        st.markdown("---")
        st.caption("Use the navigation (top-left) to access Analytics, AI Assistant and Simulator.")
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# Settings page
# ----------------------------
def settings_page():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Settings")
    st.write("Adjust app preferences")
    st.checkbox("Dark mode", value=st.session_state.dark_mode, key="ui_dark")
    st.session_state.dark_mode = st.session_state.ui_dark
    st.markdown("Model files should be placed in `./models/` folder.")
    st.write(model_status)
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# Main layout & navigation
# ----------------------------
def main():
    # header + sidebar
    top_header()
    sidebar_area()

    # top navigation simulated via sidebar selectbox for responsiveness
    pages = ["Home", "Analytics", "Simulator", "AI Assistant", "Settings"]
    choice = st.sidebar.selectbox("Navigation", pages, index=0)

    # route
    if choice == "Home":
        home_page()
    elif choice == "Analytics":
        analytics_page()
    elif choice == "Simulator":
        simulator_page()
    elif choice == "AI Assistant":
        ai_assistant_page()
    elif choice == "Settings":
        settings_page()
    else:
        st.info("Select a page from the sidebar.")

    # footer
    st.markdown("<div class='footer'>Built with ❤️ — EV Innovate</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
