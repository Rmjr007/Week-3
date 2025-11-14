# app.py
# ============================================================
#  EV INNOVATE PRO — FINAL RENDER-STABLE VERSION (GROQ FIXED)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# *** FIXED GROQ IMPORT ***
from groq import Client

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# Streamlit Setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="EV Innovate Pro — Policy Simulator",
    page_icon="⚡",
    layout="wide",
)

# ------------------------------------------------------------
# Glassmorphism CSS
# ------------------------------------------------------------
st.markdown("""
<style>
body { font-family: 'Inter', sans-serif; }
.header-bar {
    background: linear-gradient(90deg, #0f9d58, #34a853);
    padding: 22px; border-radius: 18px; color: white;
    font-weight: 700; font-size: 26px; margin-bottom: 25px;
}
.glass-card {
    background: rgba(255,255,255,0.12);
    padding: 20px; border-radius: 18px;
    backdrop-filter: blur(18px);
    margin-bottom: 22px;
}
.chat-bubble-user {
    background: rgba(52,168,83,0.35);
    padding: 14px; border-radius: 14px; margin: 10px 0;
    width: 80%; margin-left: 20%; color: white;
}
.chat-bubble-bot {
    background: rgba(255,255,255,0.12);
    padding: 14px; border-radius: 14px; margin: 10px 0;
    width: 80%; margin-right: 20%; color: white;
}
.kpi-card {
    background: rgba(255,255,255,0.15);
    padding: 22px; border-radius: 18px; text-align: center;
    color: white;
}
.kpi-number { font-size: 34px; font-weight: 800; }
.kpi-label { opacity: 0.7; font-size: 13px; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Session State
# ------------------------------------------------------------
if "df" not in st.session_state: st.session_state.df = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "logged_in" not in st.session_state: st.session_state.logged_in = False

# ------------------------------------------------------------
# Load Model & Scaler
# ------------------------------------------------------------
MODEL_PATH = "models/ev_policy_best_model.pkl"
SCALER_PATH = "models/scaler.pkl"

@st.cache_resource
def load_model_scaler():
    model, scaler, msg = None, None, ""

    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            msg += "Model loaded. "
        except Exception as e:
            msg += f"Model load error: {e}. "

    if os.path.exists(SCALER_PATH):
        try:
            scaler = joblib.load(SCALER_PATH)
            msg += "Scaler loaded."
        except Exception as e:
            msg += f"Scaler load error: {e}. "

    return model, scaler, msg

model, scaler, model_status = load_model_scaler()

# ------------------------------------------------------------
# FIXED Groq LLM (NO PROXIES)
# ------------------------------------------------------------
def call_llm_groq(query, context=""):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "❌ GROQ_API_KEY missing."

    try:
        client = Client(api_key=api_key)    # FIXED — no proxies

        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", 
                 "content": "You are an EV policy expert. Use the dataset context to answer."},
                {"role": "user", 
                 "content": f"User Question: {query}\n\nDataset Context:\n{context}"}
            ],
            temperature=0.25
        )

        return response.choices[0].message["content"]

    except Exception as e:
        return f"LLM Error: {e}"

# ------------------------------------------------------------
# TF-IDF Retriever
# ------------------------------------------------------------
def build_tfidf(df):
    corpus = [" | ".join([f"{c}: {row[c]}" for c in df.columns]) for _, row in df.iterrows()]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix, corpus

@st.cache_resource
def get_tfidf(df):
    return build_tfidf(df)

def ai_retrieve(df, query, top_k=3):
    vectorizer, matrix, corpus = get_tfidf(df)
    q_vec = vectorizer.transform([query])
    sims = linear_kernel(q_vec, matrix).flatten()
    top_idx = sims.argsort()[:-top_k-1:-1]
    return "\n\n".join([corpus[i] for i in top_idx])

# ------------------------------------------------------------
# EV Feature Definitions (Target Removed)
# ------------------------------------------------------------
PRED_FEATURES = [
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

def is_ev_dataset(df):
    df_cols = set([c.lower() for c in df.columns])
    return all(f.lower() in df_cols for f in PRED_FEATURES)

# ------------------------------------------------------------
# Prediction Function
# ------------------------------------------------------------
def predict_ev(inputs):
    if model is None or scaler is None:
        return None, "Model or scaler missing."

    try:
        X = pd.DataFrame([inputs])[PRED_FEATURES]
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        return float(pred), None
    except Exception as e:
        return None, str(e)

# ------------------------------------------------------------
# LOGIN PAGE
# ------------------------------------------------------------
def login_page():
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🔐 Login")

    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if user == "admin" and pwd == "password":
            st.session_state.logged_in = True
            st.success("Logged in.")
        else:
            st.error("Invalid credentials.")

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------
def header():
    st.markdown(
        f"""
        <div class="header-bar">
            EV Innovate Pro — Green Policy Simulator
            <div style="opacity:0.75; font-size:14px;">{model_status}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ------------------------------------------------------------
# HOME PAGE
# ------------------------------------------------------------
def home_page():
    header()
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📂 Upload Dataset")

    file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

    if file:
        df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
        st.session_state.df = df
        st.success("Dataset Uploaded.")
        st.dataframe(df.head())

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# KPI CARDS
# ------------------------------------------------------------
def ev_kpi(df):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"<div class='kpi-card'><div class='kpi-number'>{df['ev_percentage_share'].mean():.1f}%</div><div class='kpi-label'>EV Share</div></div>", unsafe_allow_html=True)

    with col2:
        st.markdown(f"<div class='kpi-card'><div class='kpi-number'>{df['charging_stations_count'].sum():,}</div><div class='kpi-label'>Charging Stations</div></div>", unsafe_allow_html=True)

    with col3:
        st.markdown(f"<div class='kpi-card'><div class='kpi-number'>${df['avg_cost_ev'].mean():,.0f}</div><div class='kpi-label'>Avg EV Cost</div></div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# ANALYTICS
# ------------------------------------------------------------
def analytics_page():
    header()

    if st.session_state.df is None:
        st.warning("Upload dataset first.")
        return

    df = st.session_state.df

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📊 Dataset Summary")
    st.write(df.describe(include="all"))
    st.markdown("</div>", unsafe_allow_html=True)

    if is_ev_dataset(df):
        st.success("EV Dataset Detected.")
        ev_kpi(df)

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("🔌 Charging Stations vs EV Share")
        fig = px.scatter(df, x="charging_stations_count", y="ev_percentage_share", trendline="ols")
        st.plotly_chart(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.info("Generic dataset detected.")

# ------------------------------------------------------------
# SIMULATOR
# ------------------------------------------------------------
def simulator_page():
    header()

    if st.session_state.df is None:
        st.warning("Upload dataset first.")
        return

    df = st.session_state.df

    defaults = {f: float(df[f].mean()) if f in df else 0 for f in PRED_FEATURES}

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🔧 Multi-Policy EV Simulator")

    inputs = {}
    for f in PRED_FEATURES:
        inputs[f] = st.number_input(f, value=defaults[f])

    if st.button("Predict"):
        pred, err = predict_ev(inputs)
        if err:
            st.error(err)
        else:
            st.success("Prediction Successful!")
            st.metric("Estimated EV Registrations", f"{int(pred):,}")

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# AI ASSISTANT
# ------------------------------------------------------------
def ai_page():
    header()

    if st.session_state.df is None:
        st.warning("Upload dataset first.")
        return

    df = st.session_state.df

    st.subheader("🤖 AI Assistant (Groq + RAG)")

    for role, msg in st.session_state.chat_history:
        bubble = "chat-bubble-user" if role == "user" else "chat-bubble-bot"
        st.markdown(f"<div class='{bubble}'>{msg}</div>", unsafe_allow_html=True)

    question = st.text_input("Ask a question about your dataset:")

    if st.button("Ask"):
        ctx = ai_retrieve(df, question)
        answer = call_llm_groq(question, ctx)

        st.session_state.chat_history.append(("user", question))
        st.session_state.chat_history.append(("bot", answer))
        st.experimental_rerun()

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.experimental_rerun()

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------
def settings_page():
    header()
    st.subheader("⚙ Settings")
    st.code(model_status)

# ------------------------------------------------------------
# NAVIGATION
# ------------------------------------------------------------
def navigation():
    return st.sidebar.radio("📌 Menu", ["Home", "Analytics", "Simulator", "AI Assistant", "Settings"])

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    if not st.session_state.logged_in:
        login_page()
        return

    page = navigation()

    if page == "Home": home_page()
    elif page == "Analytics": analytics_page()
    elif page == "Simulator": simulator_page()
    elif page == "AI Assistant": ai_page()
    elif page == "Settings": settings_page()

if __name__ == "__main__":
    main()
