# app.py
# ============================================================
#  EV INNOVATE PRO - FULL (Corrected Groq Client import)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
# Use Client import for groq
from groq import Client
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# Streamlit Page Setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="EV Innovate Pro – Green Policy Simulator",
    page_icon="⚡",
    layout="wide",
)

# ------------------------------------------------------------
# GLOBAL THEME CSS
# ------------------------------------------------------------
BASE_CSS = """
<style>
body {
    font-family: 'Inter', sans-serif;
}
.header-bar {
    background: linear-gradient(90deg, #0f9d58, #34a853);
    padding: 22px;
    border-radius: 18px;
    color: white;
    font-weight: 700;
    font-size: 26px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    margin-bottom: 20px;
}
.glass-card {
    background: rgba(255,255,255,0.12);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    padding: 20px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.15);
    margin-bottom: 20px;
    box-shadow: 0 12px 28px rgba(0,0,0,0.15);
}
.kpi-card {
    padding: 20px;
    text-align: center;
    border-radius: 18px;
    background: rgba(255,255,255,0.15);
    color: white;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    box-shadow: 0 10px 25px rgba(0,0,0,0.2);
}
.kpi-number {
    font-size: 34px;
    font-weight: 800;
}
.kpi-label {
    font-size: 13px;
    opacity: 0.85;
}
.chat-bubble-user {
    background: rgba(52,168,83,0.35);
    padding: 14px;
    border-radius: 14px;
    margin: 8px 0;
    width: 80%;
    margin-left: 20%;
    color: white;
    font-weight: 500;
    backdrop-filter: blur(14px);
}
.chat-bubble-bot {
    background: rgba(255,255,255,0.10);
    padding: 14px;
    border-radius: 14px;
    margin: 8px 0;
    width: 80%;
    margin-right: 20%;
    color: white;
    font-weight: 500;
    border: 1px solid rgba(255,255,255,0.18);
    backdrop-filter: blur(14px);
}
</style>
"""
st.markdown(BASE_CSS, unsafe_allow_html=True)

# ------------------------------------------------------------
# Session state defaults
# ------------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "dark" not in st.session_state:
    st.session_state.dark = False

# ------------------------------------------------------------
# Model & scaler loader
# ------------------------------------------------------------
MODEL_PATH = "models/ev_policy_best_model.pkl"
SCALER_PATH = "models/scaler.pkl"

@st.cache_resource
def load_model_and_scaler():
    model = scaler = None
    msg = ""
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            msg += "Model loaded. "
        except Exception as e:
            msg += f"Model error: {e}. "
    if os.path.exists(SCALER_PATH):
        try:
            scaler = joblib.load(SCALER_PATH)
            msg += "Scaler loaded."
        except Exception as e:
            msg += f"Scaler error: {e}. "
    return model, scaler, msg

model, scaler, model_status = load_model_and_scaler()

# ------------------------------------------------------------
# Groq LLM call (using Client)
# ------------------------------------------------------------
def call_llm_groq(query, context=""):
    """
    Uses groq.Client to call the LLM.
    Expects GROQ_API_KEY environment variable to be set.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "❌ GROQ_API_KEY missing in environment variables."

    try:
        client = Client(api_key=api_key)
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an EV policy expert and data analyst. "
                        "Use the dataset context to generate meaningful insights."
                    )
                },
                {
                    "role": "user",
                    "content": f"Question: {query}\n\nDataset Context:\n{context}"
                }
            ],
            temperature=0.25
        )
        # Response shape: response.choices[0].message["content"]
        return response.choices[0].message["content"]
    except Exception as e:
        return f"LLM Error: {e}"

# ------------------------------------------------------------
# TF-IDF retriever (RAG)
# ------------------------------------------------------------
def build_tfidf_index(df):
    corpus = []
    for _, row in df.iterrows():
        # build a text row safely
        parts = []
        for c in df.columns:
            try:
                parts.append(f"{c}: {row[c]}")
            except Exception:
                parts.append(f"{c}: ")
        corpus.append(" | ".join(parts))
    vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix, corpus

@st.cache_resource
def get_tfidf(df):
    return build_tfidf_index(df)

def ai_retrieve(df, query, top_k=3):
    vectorizer, matrix, corpus = get_tfidf(df)
    q_vec = vectorizer.transform([query])
    sims = linear_kernel(q_vec, matrix).flatten()
    top_idx = sims.argsort()[:-top_k-1:-1]
    retrieved = "\n\n".join([corpus[i] for i in top_idx])
    return retrieved

# ------------------------------------------------------------
# EV detection & prediction
# ------------------------------------------------------------
EV_COLUMNS = {
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

def is_ev_dataset(df):
    df_cols = set([c.lower() for c in df.columns])
    return all(c.lower() in df_cols for c in EV_COLUMNS)

PRED_FEATURES = list(EV_COLUMNS)

def predict_ev_registration(inputs):
    if model is None or scaler is None:
        return None, "Model or scaler missing."
    try:
        X = pd.DataFrame([inputs])[PRED_FEATURES]
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        return float(pred), None
    except Exception as e:
        return None, str(e)

# ------------------------------------------------------------
# UI: login, header, upload, pages
# ------------------------------------------------------------
def login_page():
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🔐 Login to Continue")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")
    if st.button("Login"):
        if user == "admin" and pwd == "password":
            st.session_state.logged_in = True
            st.success("Login successful!")
        else:
            st.error("Invalid credentials.")
    st.markdown("</div>", unsafe_allow_html=True)

def top_header():
    st.markdown(
        f"""
        <div class="header-bar">
            EV Innovate Pro — Green Policy Simulator
            <div style='font-size:14px; font-weight:400; opacity:0.85;'>
                {model_status}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("## 🌗 Theme")
    # use checkbox, toggle may not exist
    st.session_state.dark = st.sidebar.checkbox("Enable Dark Mode", value=False)
    if st.session_state.dark:
        st.markdown(
            """
            <style>
            body { background-color: #0b1116 !important; color: #e8f0f7 !important; }
            </style>
            """,
            unsafe_allow_html=True,
        )

def upload_dataset():
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📂 Upload Dataset")
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    if uploaded:
        try:
            if uploaded.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            st.session_state.df = df
            st.success(f"Uploaded: `{uploaded.name}` — {df.shape[0]} rows, {df.shape[1]} cols")
        except Exception as e:
            st.error(f"Failed to read file: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

def home_page():
    top_header()
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🏠 Home — Welcome to EV Innovate Pro")
    st.write("""
    - Upload a dataset to begin analysis
    - Access EV-specific and generic analytics
    - Simulate policy scenarios
    - Interact with the AI Assistant
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    upload_dataset()
    if st.session_state.df is not None:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("👀 Dataset Preview (first 7 rows)")
        try:
            st.dataframe(st.session_state.df.head(7), use_container_width=True)
        except Exception:
            st.write(st.session_state.df.head(7))
        st.markdown("</div>", unsafe_allow_html=True)

def ev_kpi_cards(df):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        val = int(df["ev_vehicles_registered"].sum()) if "ev_vehicles_registered" in df else 0
        st.markdown(f"<div class='kpi-card'><div class='kpi-number'>{val:,}</div><div class='kpi-label'>Total EV Registrations</div></div>", unsafe_allow_html=True)
    with col2:
        cost = float(df["avg_cost_ev"].mean()) if "avg_cost_ev" in df else 0
        st.markdown(f"<div class='kpi-card'><div class='kpi-number'>${cost:,.0f}</div><div class='kpi-label'>Avg EV Cost</div></div>", unsafe_allow_html=True)
    with col3:
        stns = int(df["charging_stations_count"].sum()) if "charging_stations_count" in df else 0
        st.markdown(f"<div class='kpi-card'><div class='kpi-number'>{stns:,}</div><div class='kpi-label'>Charging Stations</div></div>", unsafe_allow_html=True)
    with col4:
        inc = float(df["gov_incentive_amount"].mean()) if "gov_incentive_amount" in df else 0
        st.markdown(f"<div class='kpi-card'><div class='kpi-number'>${inc:,.0f}</div><div class='kpi-label'>Avg Incentive</div></div>", unsafe_allow_html=True)

def analytics_page():
    top_header()
    if st.session_state.df is None:
        st.warning("Please upload a dataset first.")
        return
    df = st.session_state.df.copy()
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📊 Dataset Summary")
    st.write(df.describe(include="all"))
    st.markdown("</div>", unsafe_allow_html=True)
    if is_ev_dataset(df):
        st.success("EV dataset detected — Showing EV-specific analytics")
        ev_kpi_cards(df)
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("📈 Charging Stations vs EV Registrations")
        if "charging_stations_count" in df.columns and "ev_vehicles_registered" in df.columns:
            try:
                fig = px.scatter(df, x="charging_stations_count", y="ev_vehicles_registered", trendline="ols")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                fig = px.scatter(df, x="charging_stations_count", y="ev_vehicles_registered")
                st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("📉 Correlation Heatmap")
        num = df.select_dtypes(include=[np.number])
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(num.corr(), cmap="Greens", annot=False, ax=ax)
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("Generic dataset detected.")
        num = df.select_dtypes(include=[np.number])
        if num.shape[1] == 0:
            st.info("No numeric columns to visualize.")
            return
        col = num.columns[0]
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("📈 Histogram")
        fig = px.histogram(df, x=col)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

def simulator_page():
    top_header()
    if st.session_state.df is None:
        st.warning("Upload dataset first.")
        return
    df = st.session_state.df.copy()
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🛠️ EV Policy Simulator — Multi-Policy Model")
    defaults = {c: float(df[c].mean()) if c in df.columns else 0 for c in PRED_FEATURES}
    sim_inputs = {}
    for k in PRED_FEATURES:
        sim_inputs[k] = st.number_input(k, value=defaults[k])
    if st.button("🔮 Predict EV Registrations"):
        pred, err = predict_ev_registration(sim_inputs)
        if err:
            st.error(err)
        else:
            st.success("Prediction Successful!")
            st.metric("Estimated EV Registrations", f"{int(pred):,}")
    st.markdown("</div>", unsafe_allow_html=True)

def ai_assistant_page():
    top_header()
    if st.session_state.df is None:
        st.warning("Please upload a dataset first.")
        return
    df = st.session_state.df
    st.subheader("🤖 AI Assistant (Glass UI + Groq LLM)")
    # Chat history
    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"<div class='chat-bubble-user'>{msg}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bubble-bot'>{msg}</div>", unsafe_allow_html=True)
    query = st.text_input("Ask something about your dataset:")
    if st.button("Ask"):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            context = ai_retrieve(df, query, top_k=3)
            answer = call_llm_groq(query, context)
            st.session_state.chat_history.append(("user", query))
            st.session_state.chat_history.append(("bot", answer))
            st.markdown(f"<div class='chat-bubble-user'>{query}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='chat-bubble-bot'>{answer}</div>", unsafe_allow_html=True)
            st.subheader("📄 Dataset Context")
            st.code(context)
    if st.button("Clear Chat"):
        st.session_state.chat_history = []

def settings_page():
    top_header()
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("⚙️ Settings")
    st.write("- Toggle theme from the sidebar")
    st.write("- Model status:")
    st.code(model_status)
    st.markdown("</div>", unsafe_allow_html=True)

def navigation_menu():
    menu = st.sidebar.radio("📌 Navigation", ["Home", "Analytics", "Simulator", "AI Assistant", "Settings"])
    return menu

def main():
    if not st.session_state.logged_in:
        login_page()
        return
    page = navigation_menu()
    if page == "Home":
        home_page()
    elif page == "Analytics":
        analytics_page()
    elif page == "Simulator":
        simulator_page()
    elif page == "AI Assistant":
        ai_assistant_page()
    elif page == "Settings":
        settings_page()
    st.markdown("<div style='text-align:center; margin-top:30px; opacity:0.6;'>EV Innovate Pro © 2025 — Powered by Streamlit + Groq LLM</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
