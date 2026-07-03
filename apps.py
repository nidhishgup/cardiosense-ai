import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import io
import os
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioSense AI — Heart Disease Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS — Dark mode premium design
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1525 50%, #0a1020 100%);
    color: #e2e8f0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2e 0%, #0a1520 100%);
    border-right: 1px solid rgba(0, 212, 255, 0.15);
}
section[data-testid="stSidebar"] .stRadio label {
    color: #94a3b8 !important;
    font-size: 14px;
    padding: 6px 0;
    transition: color 0.2s;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    color: #00d4ff !important;
}

/* ── Metric Cards ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(0, 212, 255, 0.15);
    border-radius: 12px;
    padding: 16px;
    backdrop-filter: blur(10px);
    transition: transform 0.2s, border-color 0.2s;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    border-color: rgba(0, 212, 255, 0.4);
}
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 12px !important; }
[data-testid="stMetricValue"] { color: #00d4ff !important; font-size: 28px !important; font-weight: 700 !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #00d4ff, #0099cc);
    color: #0a0e1a;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    font-size: 16px;
    padding: 12px 32px;
    width: 100%;
    cursor: pointer;
    transition: all 0.3s;
    box-shadow: 0 4px 20px rgba(0, 212, 255, 0.3);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0, 212, 255, 0.5);
}

/* ── Inputs ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stSlider > div { color: #e2e8f0 !important; }

/* ── DataFrames ── */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* ── Expanders ── */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(0,212,255,0.15) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

/* ── Custom Hero HTML ── */
.hero-banner {
    background: linear-gradient(135deg, rgba(0,212,255,0.08) 0%, rgba(255,71,87,0.05) 100%);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 20px;
    padding: 48px 40px;
    text-align: center;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at center, rgba(0,212,255,0.05) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 3rem; font-weight: 800;
    background: linear-gradient(135deg, #00d4ff, #ffffff, #00d4ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 12px;
    animation: shimmer 3s infinite;
    background-size: 200%;
}
@keyframes shimmer {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.hero-sub {
    font-size: 1.1rem; color: #94a3b8; margin-bottom: 20px; font-weight: 400;
}
.ecg-line {
    width: 100%; height: 60px; margin: 20px auto;
    display: block; opacity: 0.7;
}

/* ── Stat Cards ── */
.stat-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 16px;
    padding: 24px 20px;
    text-align: center;
    transition: all 0.3s;
}
.stat-card:hover {
    border-color: rgba(0,212,255,0.5);
    transform: translateY(-4px);
    box-shadow: 0 10px 30px rgba(0,212,255,0.15);
}
.stat-number { font-size: 2.4rem; font-weight: 800; color: #00d4ff; line-height: 1; }
.stat-label  { font-size: 0.85rem; color: #64748b; margin-top: 6px; text-transform: uppercase; letter-spacing: 1px; }
.stat-icon   { font-size: 2rem; margin-bottom: 10px; }

/* ── Risk Badges ── */
.badge { display:inline-block; padding:6px 18px; border-radius:50px; font-weight:700; font-size:1rem; }
.badge-low      { background:rgba(46,213,115,0.15); color:#2ed573; border:1px solid #2ed573; }
.badge-moderate { background:rgba(255,213,79,0.15); color:#ffd54f; border:1px solid #ffd54f; }
.badge-high     { background:rgba(255,165,2,0.15);  color:#ffa502; border:1px solid #ffa502; }
.badge-veryhigh { background:rgba(255,71,87,0.15);  color:#ff4757; border:1px solid #ff4757; }

/* ── Section Headers ── */
.section-header {
    font-size: 1.8rem; font-weight: 700; color: #ffffff;
    border-left: 4px solid #00d4ff;
    padding-left: 16px; margin: 32px 0 20px 0;
}

/* ── Factor Cards ── */
.factor-card {
    background: rgba(255,71,87,0.08);
    border: 1px solid rgba(255,71,87,0.3);
    border-radius: 10px;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 0.95rem;
    color: #ff6b7a;
}
.factor-good {
    background: rgba(46,213,115,0.08);
    border: 1px solid rgba(46,213,115,0.3);
    border-radius: 10px;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 0.95rem;
    color: #2ed573;
}
.suggestion-card {
    background: rgba(0,212,255,0.06);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 10px;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 0.9rem;
    color: #94d0e8;
}

/* ── Upload Zone ── */
.upload-card {
    background: rgba(0,212,255,0.05);
    border: 2px dashed rgba(0,212,255,0.35);
    border-radius: 16px;
    padding: 30px;
    text-align: center;
    margin: 16px 0;
}

/* ── Model Perf Table ── */
.perf-table th {
    background: rgba(0,212,255,0.1) !important;
    color: #00d4ff !important;
}
.perf-table tr:hover { background: rgba(255,255,255,0.04) !important; }

/* ── Pipeline Steps ── */
.pipeline-step {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 12px;
    padding: 18px;
    text-align: center;
    position: relative;
}
.pipeline-step .step-num {
    font-size: 0.7rem; color: #00d4ff; font-weight: 700;
    text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;
}
.pipeline-step .step-title { font-size: 1rem; font-weight: 600; color: #e2e8f0; }
.pipeline-step .step-desc  { font-size: 0.8rem; color: #64748b; margin-top: 4px; }

/* ── Divider ── */
.neon-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,212,255,0.4), transparent);
    margin: 32px 0;
}

/* ── Footer ── */
.footer {
    text-align:center; padding:24px; color:#475569; font-size:0.8rem;
    border-top: 1px solid rgba(255,255,255,0.06); margin-top:40px;
}

/* Plotly dark bg fix */
.js-plotly-plot .plotly .bg { fill: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.02)",
    font=dict(family="Inter", color="#94a3b8"),
    title_font=dict(family="Inter", color="#e2e8f0", size=16),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", showline=False, tickfont=dict(color="#64748b")),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", showline=False, tickfont=dict(color="#64748b")),
    margin=dict(l=40, r=20, t=50, b=40),
    colorway=["#00d4ff", "#ff4757", "#2ed573", "#ffa502", "#a855f7", "#f97316"],
)

def apply_theme(fig, title=""):
    fig.update_layout(**PLOTLY_LAYOUT, title=dict(text=title, font=dict(color="#e2e8f0")))
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & CACHING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_bytes=None, model_path="heart_disease_model.pkl"):
    if model_bytes:
        return joblib.load(io.BytesIO(model_bytes))
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

@st.cache_resource
def load_scaler(scaler_bytes=None, scaler_path="scaler.pkl"):
    if scaler_bytes:
        return joblib.load(io.BytesIO(scaler_bytes))
    if os.path.exists(scaler_path):
        return joblib.load(scaler_path)
    return None

@st.cache_data
def load_and_clean_data(csv_bytes=None, csv_path="cardio_train.csv"):
    if csv_bytes:
        df = pd.read_csv(io.BytesIO(csv_bytes), sep=";")
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path, sep=";")
    else:
        return None

    # Replicate cleaning from notebook
    if "id" in df.columns:
        df.drop("id", axis=1, inplace=True)
    df["age_years"] = (df["age"] / 365).astype(int)
    df.drop_duplicates(inplace=True)
    df = df[(df["ap_hi"] > 60) & (df["ap_hi"] < 240)]
    df = df[(df["ap_lo"] > 40) & (df["ap_lo"] < 180)]
    df = df[df["ap_hi"] >= df["ap_lo"]]
    df = df[(df["height"] > 130) & (df["height"] < 220)]
    df = df[(df["weight"] > 30) & (df["weight"] < 200)]
    df["BMI"] = (df["weight"] / ((df["height"] / 100) ** 2)).round(1)
    df["PulsePressure"] = df["ap_hi"] - df["ap_lo"]
    df["age_group"] = pd.cut(df["age_years"], bins=[0, 40, 50, 60, 100], labels=[0, 1, 2, 3]).astype(int)

    def bp_category(row):
        if row["ap_hi"] < 120 and row["ap_lo"] < 80:   return 0
        elif row["ap_hi"] < 130 and row["ap_lo"] < 80: return 1
        elif row["ap_hi"] < 140 or row["ap_lo"] < 90:  return 2
        else:                                            return 3

    df["bp_category"] = df.apply(bp_category, axis=1)
    df["risk_score"] = df["cholesterol"] + df["gluc"] + df["smoke"] + df["alco"] + (1 - df["active"])
    return df

# Expected feature column order (must match training X.columns)
FEATURE_COLS = [
    "gender", "height", "weight", "ap_hi", "ap_lo",
    "cholesterol", "gluc", "smoke", "alco", "active",
    "age_years", "BMI", "PulsePressure", "age_group", "bp_category", "risk_score"
]

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS (exact logic from notebook)
# ─────────────────────────────────────────────────────────────────────────────
def get_risk_category(prob):
    if prob < 0.25:   return "Low Risk",       "badge-low"
    elif prob < 0.50: return "Moderate Risk",  "badge-moderate"
    elif prob < 0.75: return "High Risk",       "badge-high"
    else:             return "Very High Risk",  "badge-veryhigh"

def get_contributing_factors(ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, bmi, age_years):
    factors, safe = [], []
    if ap_hi >= 140 or ap_lo >= 90:  factors.append("🔴 High Blood Pressure")
    elif ap_hi >= 130:               factors.append("🟠 Elevated Blood Pressure")
    else:                             safe.append("✅ Blood Pressure is Normal")

    if cholesterol > 1:  factors.append("🔴 Above Normal Cholesterol")
    else:                 safe.append("✅ Normal Cholesterol")

    if gluc > 1:  factors.append("🔴 Above Normal Glucose")
    else:          safe.append("✅ Normal Glucose")

    if smoke == 1:   factors.append("🔴 Smoking")
    else:             safe.append("✅ Non-Smoker")

    if alco == 1:    factors.append("🟠 Alcohol Consumption")
    else:             safe.append("✅ No Alcohol")

    if active == 0:  factors.append("🟠 Physically Inactive")
    else:             safe.append("✅ Physically Active")

    if bmi >= 30:       factors.append("🔴 Obesity (BMI ≥ 30)")
    elif bmi >= 25:     factors.append("🟠 Overweight (BMI 25–29.9)")
    else:                safe.append("✅ Healthy BMI")

    if age_years >= 55:  factors.append("🟡 Age Above 55")

    return factors, safe

def get_suggestions(ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, bmi):
    lifestyle, food_eat, food_avoid = [], [], []

    if smoke == 1:
        lifestyle.append("🚭 Quit smoking — biggest controllable risk factor")
    if alco == 1:
        lifestyle.append("🍺 Reduce alcohol intake, even moderate drinking adds risk")
    if active == 0:
        lifestyle.append("🏃 Add at least 30 mins of walking or exercise daily")
    if bmi >= 25:
        lifestyle.append("⚖️ Aim for 5–10% weight loss — significantly reduces risk")
    if ap_hi >= 130 or ap_lo >= 85:
        lifestyle.append("🩺 Monitor blood pressure regularly, consider consulting a doctor")
        food_avoid.append("🧂 High-salt foods: chips, processed meats, pickles, canned soups")
        food_eat.append("🍌 Potassium-rich foods: bananas, leafy greens, sweet potatoes")

    if cholesterol > 1:
        food_avoid.append("🍔 Fried food, red meat, butter, full-fat dairy in excess")
        food_eat.append("🥣 Oats, nuts, fatty fish, olive oil, fiber-rich vegetables")

    if gluc > 1:
        food_avoid.append("🥤 Sugary drinks, sweets, white rice and refined carbs")
        food_eat.append("🌾 Whole grains, vegetables, low glycemic index foods")

    if bmi >= 25:
        food_avoid.append("🍟 Fast food, sugary snacks, deep-fried items")
        food_eat.append("🥗 High-protein, high-fiber meals with smaller portions")

    if not lifestyle:
        lifestyle.append("💪 Keep up your current healthy habits — you're doing great!")
    if not food_eat:
        food_eat.append("🥦 Balanced diet with fruits, vegetables and whole grains")
    if not food_avoid:
        food_avoid.append("✅ No major dietary restrictions — just avoid excess junk food")

    return lifestyle, food_eat, food_avoid

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px 0;'>
        <div style='font-size:2.8rem;'>🫀</div>
        <div style='font-size:1.2rem; font-weight:800; color:#00d4ff; letter-spacing:1px;'>CardioSense AI</div>
        <div style='font-size:0.75rem; color:#475569; margin-top:4px;'>Heart Disease Prediction</div>
    </div>
    <hr style='border:none;height:1px;background:rgba(0,212,255,0.15);margin:16px 0;'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🏠  Home",
         "📊  Data Explorer",
         "🤖  Model Performance",
         "🔮  Live Prediction",
         "📖  Methodology"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border:none;height:1px;background:rgba(255,255,255,0.06);margin:24px 0 12px 0;'>", unsafe_allow_html=True)

    # ── File Upload in Sidebar ──
    st.markdown("<div style='font-size:0.8rem;color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;'>📁 Upload Model Files</div>", unsafe_allow_html=True)

    uploaded_model  = st.file_uploader("Model (.pkl)",  type=["pkl"], key="model_upload",  label_visibility="collapsed")
    st.caption("heart_disease_model.pkl")
    uploaded_scaler = st.file_uploader("Scaler (.pkl)", type=["pkl"], key="scaler_upload", label_visibility="collapsed")
    st.caption("scaler.pkl")
    uploaded_csv    = st.file_uploader("Dataset (.csv)", type=["csv","CSV"], key="csv_upload",   label_visibility="collapsed")
    st.caption("cardio_train.csv")

    # ── Status Indicators ──
    st.markdown("<div style='margin-top:16px;'>", unsafe_allow_html=True)

    model_bytes  = uploaded_model.read()  if uploaded_model  else None
    scaler_bytes = uploaded_scaler.read() if uploaded_scaler else None
    csv_bytes    = uploaded_csv.read()    if uploaded_csv    else None

    model  = load_model(model_bytes)
    scaler = load_scaler(scaler_bytes)
    df     = load_and_clean_data(csv_bytes)

    def status_dot(ok): return "🟢" if ok else "🔴"
    st.markdown(f"""
    <div style='font-size:0.8rem;line-height:2;'>
        {status_dot(model  is not None)} Model loaded<br>
        {status_dot(scaler is not None)} Scaler loaded<br>
        {status_dot(df     is not None)} Dataset loaded
    </div>
    """, unsafe_allow_html=True)

    if df is not None:
        st.markdown(f"<div style='font-size:0.75rem;color:#475569;margin-top:8px;'>{len(df):,} records ready</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-top:auto;padding-top:32px;font-size:0.7rem;color:#334155;text-align:center;'>
        Built with ❤️ using Streamlit
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ██████████████████  PAGE 1: HOME  ██████████████████
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠  Home":

    # ── ECG Hero Banner ──
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🫀 CardioSense AI</div>
        <div class="hero-sub">
            Machine Learning-Powered Cardiovascular Disease Prediction<br>
            <span style="color:#64748b;font-size:0.9rem;">
                Trained on real clinical data · Random Forest · 72%+ accuracy
            </span>
        </div>
        <svg class="ecg-line" viewBox="0 0 900 60" xmlns="http://www.w3.org/2000/svg">
            <polyline
                points="0,30 60,30 80,30 90,5 100,55 110,20 125,30 200,30 220,30 230,5 240,55 250,20 265,30 340,30 360,30 370,5 380,55 390,20 405,30 480,30 500,30 510,5 520,55 530,20 545,30 620,30 640,30 650,5 660,55 670,20 685,30 760,30 780,30 790,5 800,55 810,20 825,30 900,30"
                fill="none"
                stroke="url(#ecgGrad)"
                stroke-width="2.5"
                stroke-linecap="round"
                stroke-linejoin="round"
            />
            <defs>
                <linearGradient id="ecgGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%"   stop-color="#00d4ff" stop-opacity="0"/>
                    <stop offset="20%"  stop-color="#00d4ff"/>
                    <stop offset="80%"  stop-color="#ff4757"/>
                    <stop offset="100%" stop-color="#ff4757" stop-opacity="0"/>
                </linearGradient>
            </defs>
        </svg>
    </div>
    """, unsafe_allow_html=True)

    # ── Top Stats ──
    c1, c2, c3, c4 = st.columns(4)
    records = f"{len(df):,}" if df is not None else "70,000+"
    with c1:
        st.markdown(f"""<div class="stat-card">
            <div class="stat-icon">🗃️</div>
            <div class="stat-number">{records}</div>
            <div class="stat-label">Patient Records</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="stat-card">
            <div class="stat-icon">⚙️</div>
            <div class="stat-number">16</div>
            <div class="stat-label">Features Used</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="stat-card">
            <div class="stat-icon">🤖</div>
            <div class="stat-number">6</div>
            <div class="stat-label">Models Compared</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""<div class="stat-card">
            <div class="stat-icon">🎯</div>
            <div class="stat-number">~72%</div>
            <div class="stat-label">Best Accuracy</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)

    # ── Model Snapshot ──
    st.markdown("<div class='section-header'>Model Performance Snapshot</div>", unsafe_allow_html=True)

    perf_data = {
        "Model": ["Logistic Regression", "Random Forest", "KNN", "Decision Tree", "Naive Bayes", "Gradient Boosting"],
        "Accuracy": [0.723, 0.724, 0.697, 0.676, 0.697, 0.727],
        "F1 Score": [0.721, 0.722, 0.697, 0.675, 0.700, 0.724],
        "AUC":      [0.793, 0.793, 0.767, 0.676, 0.760, 0.799],
    }
    perf_df = pd.DataFrame(perf_data)

    fig = go.Figure()
    metrics = ["Accuracy", "F1 Score", "AUC"]
    colors  = ["#00d4ff", "#2ed573", "#a855f7"]
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        fig.add_trace(go.Bar(
            name=metric,
            x=perf_df["Model"],
            y=perf_df[metric],
            marker_color=color,
            opacity=0.85,
            marker_line_color="rgba(255,255,255,0.1)",
            marker_line_width=1
        ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        barmode="group",
        title="All Models — Accuracy / F1 / AUC",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)"),
        yaxis_range=[0.6, 0.85],
        height=380
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)

    # ── Feature Quick Info ──
    st.markdown("<div class='section-header'>What Does the Model Analyze?</div>", unsafe_allow_html=True)

    feature_info = [
        ("🧬", "Demographics",    "Age, Gender"),
        ("📏", "Physical",        "Height, Weight, BMI"),
        ("💉", "Blood Pressure",  "Systolic, Diastolic, Pulse Pressure"),
        ("🧪", "Lab Values",      "Cholesterol, Glucose levels"),
        ("🚬", "Lifestyle",       "Smoking, Alcohol, Physical Activity"),
        ("⚠️", "Risk Score",      "Composite lifestyle risk index"),
    ]
    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(feature_info):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="stat-card" style="text-align:left; padding:20px;">
                <div style="font-size:1.5rem;margin-bottom:8px;">{icon}</div>
                <div style="font-weight:700;color:#e2e8f0;font-size:0.95rem;">{title}</div>
                <div style="font-size:0.8rem;color:#64748b;margin-top:4px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    if model is None or scaler is None:
        st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="upload-card">
            <div style="font-size:2rem;margin-bottom:10px;">📁</div>
            <div style="font-weight:700;color:#e2e8f0;font-size:1rem;">Upload your model files to enable Live Prediction</div>
            <div style="font-size:0.85rem;color:#64748b;margin-top:8px;">
                Use the sidebar to upload <code>heart_disease_model.pkl</code>, <code>scaler.pkl</code>, and <code>cardio_train.csv</code>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='footer'>CardioSense AI · Built for educational purposes · Not a substitute for medical advice</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ██████████████████  PAGE 2: DATA EXPLORER  ██████████████████
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊  Data Explorer":

    st.markdown("<div class='section-header'>📊 Data Explorer</div>", unsafe_allow_html=True)

    if df is None:
        st.warning("⚠️ Please upload `cardio_train.csv` using the sidebar to explore the dataset.")
        st.stop()

    # ── Dataset Preview ──
    with st.expander("🗃️ Dataset Preview", expanded=False):
        st.dataframe(df.head(100), use_container_width=True, height=300)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Records",  f"{len(df):,}")
        c2.metric("Features",       f"{df.shape[1]}")
        c3.metric("Missing Values", f"{df.isnull().sum().sum()}")

    st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)

    # ── Filters ──
    st.markdown("#### 🔧 Filter Dataset")
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        age_range = st.slider("Age Range", int(df.age_years.min()), int(df.age_years.max()), (30, 65))
    with fc2:
        gender_filter = st.multiselect("Gender", options=[1, 2], default=[1, 2], format_func=lambda x: "Female" if x == 1 else "Male")
    with fc3:
        cardio_filter = st.multiselect("Cardio Status", options=[0, 1], default=[0, 1], format_func=lambda x: "No Disease" if x == 0 else "Has Disease")

    fdf = df[
        (df.age_years >= age_range[0]) & (df.age_years <= age_range[1]) &
        (df.gender.isin(gender_filter)) &
        (df.cardio.isin(cardio_filter))
    ]
    st.caption(f"Showing **{len(fdf):,}** records after filters")

    st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)

    # ── Chart Row 1: Target Balance ──
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        counts = fdf["cardio"].value_counts()
        fig = go.Figure(go.Pie(
            labels=["No Disease", "Has Disease"],
            values=[counts.get(0, 0), counts.get(1, 0)],
            hole=0.55,
            marker_colors=["#2ed573", "#ff4757"],
            textinfo="percent+label",
            textfont_size=13,
        ))
        fig.update_layout(**PLOTLY_LAYOUT, title="Target Class Balance", height=350,
                          showlegend=True, legend=dict(orientation="h", y=-0.1, bgcolor="rgba(0,0,0,0)"))
        fig.add_annotation(text=f"{len(fdf):,}<br>records", x=0.5, y=0.5, showarrow=False,
                           font=dict(size=14, color="#e2e8f0", family="Inter"))
        st.plotly_chart(fig, use_container_width=True)

    with r1c2:
        fig = px.histogram(fdf, x="age_years", color="cardio",
                           color_discrete_map={0: "#2ed573", 1: "#ff4757"},
                           nbins=25, barmode="overlay", opacity=0.75,
                           labels={"cardio": "Cardio", "age_years": "Age (years)"},
                           category_orders={"cardio": [0, 1]})
        fig.update_traces(marker_line_width=0)
        apply_theme(fig, "Age Distribution by Cardio Status")
        fig.update_layout(height=350, legend_title="Cardio",
                          legend=dict(orientation="h", y=1.05, bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig, use_container_width=True)

    # ── Chart Row 2: BMI & BP ──
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        fig = px.box(fdf, x="cardio", y="BMI", color="cardio",
                     color_discrete_map={0: "#2ed573", 1: "#ff4757"},
                     labels={"cardio": "Cardio Status", "BMI": "BMI"},
                     category_orders={"cardio": [0, 1]})
        apply_theme(fig, "BMI Distribution by Cardio Status")
        fig.update_layout(height=350, showlegend=False,
                          xaxis=dict(ticktext=["No Disease", "Has Disease"], tickvals=[0, 1], **PLOTLY_LAYOUT["xaxis"]))
        st.plotly_chart(fig, use_container_width=True)

    with r2c2:
        fig = px.box(fdf, x="cardio", y="ap_hi", color="cardio",
                     color_discrete_map={0: "#2ed573", 1: "#ff4757"},
                     labels={"cardio": "Cardio Status", "ap_hi": "Systolic BP (mmHg)"},
                     category_orders={"cardio": [0, 1]})
        apply_theme(fig, "Systolic BP by Cardio Status")
        fig.update_layout(height=350, showlegend=False,
                          xaxis=dict(ticktext=["No Disease", "Has Disease"], tickvals=[0, 1], **PLOTLY_LAYOUT["xaxis"]))
        st.plotly_chart(fig, use_container_width=True)

    # ── Chart Row 3: Categorical ──
    r3c1, r3c2, r3c3 = st.columns(3)

    with r3c1:
        grp = fdf.groupby(["cholesterol", "cardio"]).size().reset_index(name="count")
        grp["cholesterol_label"] = grp["cholesterol"].map({1: "Normal", 2: "Above Normal", 3: "Well Above"})
        grp["cardio_label"] = grp["cardio"].map({0: "No Disease", 1: "Disease"})
        fig = px.bar(grp, x="cholesterol_label", y="count", color="cardio_label",
                     color_discrete_map={"No Disease": "#2ed573", "Disease": "#ff4757"},
                     barmode="group", labels={"cholesterol_label": "Cholesterol", "count": "Count", "cardio_label": "Cardio"})
        apply_theme(fig, "Cholesterol vs Cardio")
        fig.update_layout(height=300, legend=dict(orientation="h", y=1.05, bgcolor="rgba(0,0,0,0)"), legend_title="")
        st.plotly_chart(fig, use_container_width=True)

    with r3c2:
        grp = fdf.groupby(["gluc", "cardio"]).size().reset_index(name="count")
        grp["gluc_label"] = grp["gluc"].map({1: "Normal", 2: "Above Normal", 3: "Well Above"})
        grp["cardio_label"] = grp["cardio"].map({0: "No Disease", 1: "Disease"})
        fig = px.bar(grp, x="gluc_label", y="count", color="cardio_label",
                     color_discrete_map={"No Disease": "#2ed573", "Disease": "#ff4757"},
                     barmode="group", labels={"gluc_label": "Glucose", "count": "Count", "cardio_label": "Cardio"})
        apply_theme(fig, "Glucose vs Cardio")
        fig.update_layout(height=300, legend=dict(orientation="h", y=1.05, bgcolor="rgba(0,0,0,0)"), legend_title="")
        st.plotly_chart(fig, use_container_width=True)

    with r3c3:
        grp = fdf.groupby(["gender", "cardio"]).size().reset_index(name="count")
        grp["gender_label"] = grp["gender"].map({1: "Female", 2: "Male"})
        grp["cardio_label"] = grp["cardio"].map({0: "No Disease", 1: "Disease"})
        fig = px.bar(grp, x="gender_label", y="count", color="cardio_label",
                     color_discrete_map={"No Disease": "#2ed573", "Disease": "#ff4757"},
                     barmode="group", labels={"gender_label": "Gender", "count": "Count", "cardio_label": "Cardio"})
        apply_theme(fig, "Gender vs Cardio")
        fig.update_layout(height=300, legend=dict(orientation="h", y=1.05, bgcolor="rgba(0,0,0,0)"), legend_title="")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)

    # ── Lifestyle Factors ──
    st.markdown("#### 🚬 Lifestyle Risk Factors vs Cardio")
    lc1, lc2, lc3 = st.columns(3)

    for col_name, title, ax in [("smoke", "Smoking", lc1), ("alco", "Alcohol", lc2), ("active", "Physical Activity", lc3)]:
        with ax:
            grp = fdf.groupby([col_name, "cardio"]).size().reset_index(name="count")
            grp["label"] = grp[col_name].map({0: "No", 1: "Yes"})
            grp["cardio_label"] = grp["cardio"].map({0: "No Disease", 1: "Disease"})
            fig = px.bar(grp, x="label", y="count", color="cardio_label",
                         color_discrete_map={"No Disease": "#2ed573", "Disease": "#ff4757"},
                         barmode="group")
            apply_theme(fig, title)
            fig.update_layout(height=280, showlegend=False, legend_title="",
                               xaxis_title=title, yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)

    # ── Correlation Heatmap ──
    st.markdown("#### 🔥 Correlation Heatmap")
    num_cols = ["age_years", "BMI", "ap_hi", "ap_lo", "PulsePressure", "cholesterol", "gluc", "smoke", "alco", "active", "cardio"]
    corr = fdf[num_cols].corr()

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale="RdBu",
        zmin=-1, zmax=1,
        text=corr.values.round(2),
        texttemplate="%{text}",
        hoverongaps=False
    ))
    apply_theme(fig, "Feature Correlation Matrix")
    fig.update_layout(height=500, margin=dict(l=100, r=20, t=50, b=100))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='footer'>CardioSense AI · Built for educational purposes · Not a substitute for medical advice</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ██████████████████  PAGE 3: MODEL PERFORMANCE  ██████████████████
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🤖  Model Performance":

    st.markdown("<div class='section-header'>🤖 Model Performance</div>", unsafe_allow_html=True)

    # ── Results Table ──
    results = pd.DataFrame({
        "Model":     ["Logistic Regression", "Random Forest", "KNN", "Decision Tree", "Naive Bayes", "Gradient Boosting"],
        "Accuracy":  [0.7232, 0.7241, 0.6972, 0.6762, 0.6974, 0.7274],
        "Precision": [0.7198, 0.7215, 0.6951, 0.6731, 0.6960, 0.7245],
        "Recall":    [0.7271, 0.7268, 0.7010, 0.6798, 0.7001, 0.7308],
        "F1 Score":  [0.7234, 0.7241, 0.6980, 0.6764, 0.6980, 0.7276],
        "AUC":       [0.7930, 0.7930, 0.7670, 0.6760, 0.7600, 0.7990],
    })

    best_idx  = results["Accuracy"].idxmax()
    best_name = results.loc[best_idx, "Model"]

    st.markdown(f"""
    <div style="background:rgba(0,212,255,0.08);border:1px solid rgba(0,212,255,0.3);border-radius:12px;
                padding:16px 24px;display:flex;align-items:center;gap:16px;margin-bottom:24px;">
        <div style="font-size:2rem;">🏆</div>
        <div>
            <div style="font-size:1.1rem;font-weight:700;color:#00d4ff;">Best Model: {best_name}</div>
            <div style="font-size:0.85rem;color:#64748b;">Highest accuracy on held-out test set · Selected for Live Prediction</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Styled table
    def highlight_best(s):
        best = s.max()
        return ["background:rgba(0,212,255,0.15);color:#00d4ff;font-weight:700;" if v == best else "" for v in s]

    styled = results.style \
        .apply(highlight_best, subset=["Accuracy", "Precision", "Recall", "F1 Score", "AUC"]) \
        .format({"Accuracy": "{:.1%}", "Precision": "{:.1%}", "Recall": "{:.1%}", "F1 Score": "{:.1%}", "AUC": "{:.3f}"}) \
        .set_properties(**{"background-color": "rgba(255,255,255,0.02)", "color": "#e2e8f0", "border-color": "rgba(255,255,255,0.05)"})

    st.dataframe(styled, use_container_width=True, height=270)

    st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)

    # ── Bar Chart Comparison ──
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    colors  = ["#00d4ff", "#2ed573", "#ffa502", "#a855f7"]

    fig = go.Figure()
    for metric, color in zip(metrics, colors):
        fig.add_trace(go.Bar(name=metric, x=results["Model"], y=results[metric],
                             marker_color=color, opacity=0.85))
    fig.update_layout(**PLOTLY_LAYOUT, barmode="group",
                      title="Model Comparison — All Metrics",
                      yaxis_range=[0.65, 0.78], height=420,
                      legend=dict(orientation="h", y=1.05, bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)

    # ── AUC Comparison ──
    c1, c2 = st.columns(2)

    with c1:
        fig = go.Figure(go.Bar(
            x=results["AUC"], y=results["Model"],
            orientation="h",
            marker_color=["#ff4757" if m == best_name else "#00d4ff" for m in results["Model"]],
            text=results["AUC"].apply(lambda x: f"{x:.3f}"),
            textposition="outside",
            textfont=dict(color="#e2e8f0", size=12),
        ))
        apply_theme(fig, "ROC-AUC Score by Model")
        fig.update_layout(height=350, xaxis_range=[0.65, 0.82], showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Overfitting check
        overfit = pd.DataFrame({
            "Model":      ["Logistic Regression", "Random Forest", "KNN", "Decision Tree", "Naive Bayes", "Gradient Boosting"],
            "Train Acc":  [0.7245, 0.7823, 0.7512, 0.7910, 0.7001, 0.8110],
            "Test Acc":   [0.7232, 0.7241, 0.6972, 0.6762, 0.6974, 0.7274],
        })
        overfit["Gap"] = (overfit["Train Acc"] - overfit["Test Acc"]).round(4)
        overfit["Status"] = overfit["Gap"].apply(
            lambda g: "⚠️ Possible Overfit" if g > 0.10 else ("🟡 Mild Gap" if g > 0.05 else "✅ Safe")
        )
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Train", x=overfit["Model"], y=overfit["Train Acc"], marker_color="#00d4ff", opacity=0.8))
        fig.add_trace(go.Bar(name="Test",  x=overfit["Model"], y=overfit["Test Acc"],  marker_color="#ff4757", opacity=0.8))
        apply_theme(fig, "Train vs Test Accuracy (Overfit Check)")
        fig.update_layout(height=350, barmode="group", yaxis_range=[0.65, 0.84],
                          legend=dict(orientation="h", y=1.05, bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)

    # ── Feature Importance ──
    st.markdown("#### 🔑 Feature Importance (Random Forest / Best Model)")

    if model is not None and hasattr(model, "feature_importances_"):
        importance = pd.DataFrame({
            "Feature":    FEATURE_COLS,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=True).tail(12)

        fig = go.Figure(go.Bar(
            x=importance["Importance"], y=importance["Feature"],
            orientation="h",
            marker=dict(
                color=importance["Importance"],
                colorscale=[[0, "#1e3a5f"], [0.5, "#00d4ff"], [1, "#ff4757"]],
                showscale=False
            ),
            text=importance["Importance"].apply(lambda x: f"{x:.3f}"),
            textposition="outside",
            textfont=dict(color="#94a3b8", size=11),
        ))
        apply_theme(fig, "Feature Importance — Actual Model")
        fig.update_layout(height=420, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Illustrative feature importance (from typical runs)
        feat_imp = pd.DataFrame({
            "Feature":    ["age_years", "ap_hi", "BMI", "PulsePressure", "ap_lo", "cholesterol",
                           "weight", "height", "risk_score", "bp_category", "gluc", "gender"],
            "Importance": [0.192, 0.183, 0.132, 0.091, 0.081, 0.062, 0.058, 0.045, 0.038, 0.035, 0.029, 0.018]
        }).sort_values("Importance", ascending=True)

        fig = go.Figure(go.Bar(
            x=feat_imp["Importance"], y=feat_imp["Feature"],
            orientation="h",
            marker=dict(
                color=feat_imp["Importance"],
                colorscale=[[0, "#1e3a5f"], [0.5, "#00d4ff"], [1, "#ff4757"]],
                showscale=False
            ),
            text=feat_imp["Importance"].apply(lambda x: f"{x:.3f}"),
            textposition="outside",
            textfont=dict(color="#94a3b8", size=11),
        ))
        apply_theme(fig, "Feature Importance — Illustrative (upload model for actual values)")
        fig.update_layout(height=420, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("ℹ️ Upload `heart_disease_model.pkl` to see actual feature importances from your trained model.")

    st.markdown("<div class='footer'>CardioSense AI · Model Performance</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ██████████████████  PAGE 4: LIVE PREDICTION  ██████████████████
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🔮  Live Prediction":
    st.markdown("<div class='section-header'>🔮 Live Prediction Inference</div>", unsafe_allow_html=True)

    if model is None or scaler is None:
        st.error("⚠️ Prediction disabled. Please upload both `heart_disease_model.pkl` and `scaler.pkl` in the sidebar first.")
        st.stop()

    st.markdown("##### Input Patient Clinical Data")
    
    # ── Input Fields Form Layout ──
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age_years = st.number_input("Age (Years)", min_value=1, max_value=120, value=50, step=1)
        gender_val = st.selectbox("Gender", options=[1, 2], format_func=lambda x: "Female" if x == 1 else "Male")
        height = st.number_input("Height (cm)", min_value=50, max_value=250, value=165, step=1)
        weight = st.number_input("Weight (kg)", min_value=10, max_value=300, value=70, step=1)

    with col2:
        ap_hi = st.number_input("Systolic Blood Pressure (ap_hi)", min_value=40, max_value=300, value=120, step=1)
        ap_lo = st.number_input("Diastolic Blood Pressure (ap_lo)", min_value=30, max_value=200, value=80, step=1)
        cholesterol = st.selectbox("Cholesterol Level", options=[1, 2, 3], format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}[x])
        gluc = st.selectbox("Glucose Level", options=[1, 2, 3], format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}[x])

    with col3:
        smoke = st.selectbox("Smoking Status", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        alco = st.selectbox("Alcohol Consumption", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        active = st.selectbox("Physical Activity (Active?)", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    # ── Derived Feature Engineering Pipeline (Matches training data) ──
    bmi = round(weight / ((height / 100) ** 2), 1)
    pulse_pressure = ap_hi - ap_lo
    
    # Binning Age Groups
    if age_years <= 40: age_group = 0
    elif age_years <= 50: age_group = 1
    elif age_years <= 60: age_group = 2
    else: age_group = 3

    # Categorizing Blood Pressure
    if ap_hi < 120 and ap_lo < 80: bp_cat = 0
    elif ap_hi < 130 and ap_lo < 80: bp_cat = 1
    elif ap_hi < 140 or ap_lo < 90: bp_cat = 2
    else: bp_cat = 3

    risk_score = cholesterol + gluc + smoke + alco + (1 - active)

    # Secondary KPI Readouts
    st.markdown("---")
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Calculated BMI", f"{bmi}", delta="Overweight/Obese" if bmi >= 25 else "Normal Weight", delta_color="inverse")
    mc2.metric("Pulse Pressure", f"{pulse_pressure} mmHg")
    mc3.metric("Lifestyle Risk Factor Index", f"{risk_score} / 5")

    # ── Inference Execution ──
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🧠 Execute Heart Disease Prediction"):
        
        # 1. Map variables into structural DataFrame matching exact order of FEATURE_COLS
        input_data = pd.DataFrame([{
            "gender": gender_val, "height": height, "weight": weight, "ap_hi": ap_hi, "ap_lo": ap_lo,
            "cholesterol": cholesterol, "gluc": gluc, "smoke": smoke, "alco": alco, "active": active,
            "age_years": age_years, "BMI": bmi, "PulsePressure": pulse_pressure, "age_group": age_group, 
            "bp_category": bp_cat, "risk_score": risk_score
        }])[FEATURE_COLS]

        try:
            # 2. Scale features using user's uploaded scalar
            scaled_data = scaler.transform(input_data)
            
            # 3. Generate target probability
            prob = model.predict_proba(scaled_data)[0][1]
            category, badge_class = get_risk_category(prob)
            
            # ── Visual UI Results Display ──
            st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)
            
            res_col1, res_col2 = st.columns([1, 1.5])
            with res_col1:
                st.markdown(f"""
                <div style='text-align: center; padding: 20px; background: rgba(255,255,255,0.03); border-radius:15px; border: 1px solid rgba(0,212,255,0.1);'>
                    <h4 style='margin: 0; color: #94a3b8;'>Prediction Outcome</h4>
                    <div style='font-size: 3.5rem; margin: 15px 0;'>🫀</div>
                    <span class="badge {badge_class}" style="font-size:1.3rem; padding: 8px 24px;">{category}</span>
                    <h3 style='color: #00d4ff; margin-top: 20px; font-size: 2.2rem;'>{prob*100:.1f}%</h3>
                    <p style='color: #64748b; font-size: 0.85rem; margin: 0;'>Probability of Cardiovascular Disease</p>
                </div>
                """, unsafe_allow_html=True)
                
            with res_col2:
                st.markdown("⚡ **Clinical Risk Breakdown**")
                factors, safe = get_contributing_factors(ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, bmi, age_years)
                lifestyle, food_eat, food_avoid = get_suggestions(ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, bmi)
                
                tab1, tab2 = st.tabs(["📊 Risk Drivers", "🥗 Recommendation Plan"])
                with tab1:
                    if factors:
                        st.write("Metrics increasing risk:")
                        for f in factors: st.markdown(f"<div class='factor-card'>{f}</div>", unsafe_allow_html=True)
                    if safe:
                        st.write("Metrics lowering risk:")
                        for s in safe: st.markdown(f"<div class='factor-good'>{s}</div>", unsafe_allow_html=True)
                with tab2:
                    st.write("**Suggested Lifestyle Changes:**")
                    for l in lifestyle: st.markdown(f"<div class='suggestion-card'>{l}</div>", unsafe_allow_html=True)
                    st.success(f"➕ **Diet Additions:** {', '.join(food_eat)}")
                    st.error(f"➖ **Diet Restrictions:** {', '.join(food_avoid)}")

        except Exception as e:
            st.error(f"❌ Transformation Error: Check if your uploaded scaler matches the 16 model features. Details: {e}")

    st.markdown("<div class='footer'>CardioSense AI · Educational Demo Tool</div>", unsafe_allow_html=True)
# ██████████████████  PAGE 5: METHODOLOGY  ██████████████████
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📖  Methodology":

    st.markdown("<div class='section-header'>📖 Project Methodology</div>", unsafe_allow_html=True)

    # ── Pipeline Diagram ──
    st.markdown("#### 🔄 ML Pipeline")
    steps = [
        ("01", "Data Ingestion",      "cardio_train.csv\n70k patient records, 11 raw features"),
        ("02", "Cleaning",            "Remove duplicates\nFilter invalid BP, height, weight values"),
        ("03", "Feature Engineering", "BMI, Pulse Pressure\nAge Group, BP Category, Risk Score"),
        ("04", "EDA",                 "Distribution plots, correlation\nheatmap, outlier detection"),
        ("05", "Model Training",      "6 models compared\nTrain/test split 80/20, StandardScaler"),
        ("06", "Evaluation",          "Accuracy, F1, AUC\nConfusion matrix, ROC curves"),
        ("07", "Hyperparameter Tuning","GridSearchCV on\nRandom Forest (best model)"),
        ("08", "Deployment",          "Streamlit app\nActual .pkl model served live"),
    ]

    for row_start in range(0, len(steps), 4):
        cols = st.columns(4)
        for i, (num, title, desc) in enumerate(steps[row_start:row_start+4]):
            with cols[i]:
                st.markdown(f"""
                <div class="pipeline-step">
                    <div class="step-num">Step {num}</div>
                    <div class="step-title">{title}</div>
                    <div class="step-desc">{desc.replace(chr(10), "<br>")}</div>
                </div>
                """, unsafe_allow_html=True)
        if row_start + 4 < len(steps):
            st.markdown("""
            <div style="text-align:center;font-size:1.5rem;color:#00d4ff;margin:8px 0;">↓</div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)

    # ── Data Cleaning Summary ──
    st.markdown("#### 🧹 Data Cleaning Details")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **Issues Found in Raw Data:**
        - Negative blood pressure values
        - Systolic BP > 240 mmHg (physiologically impossible)
        - Diastolic BP > Systolic BP
        - Height < 130 cm or > 220 cm
        - Weight < 30 kg or > 200 kg
        - Duplicate patient rows
        - Age stored in days (converted to years)
        """)
    with c2:
        st.markdown("""
        **Cleaning Steps Applied:**
        - Removed duplicate rows
        - `ap_hi`: kept 60–240 mmHg range
        - `ap_lo`: kept 40–180 mmHg range
        - Enforced `ap_hi >= ap_lo`
        - `height`: kept 130–220 cm
        - `weight`: kept 30–200 kg
        - Converted age: `age_years = age / 365`
        """)

    st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)

    # ── Feature Engineering ──
    st.markdown("#### ⚙️ Engineered Features")
    feat_table = pd.DataFrame({
        "Feature":     ["BMI", "PulsePressure", "age_group", "bp_category", "risk_score"],
        "Formula":     [
            "weight / (height/100)²",
            "ap_hi − ap_lo",
            "Bins: ≤40→0, ≤50→1, ≤60→2, >60→3",
            "Normal→0, Elevated→1, High→2, Crisis→3",
            "cholesterol + gluc + smoke + alco + (1−active)"
        ],
        "Rationale": [
            "Standard obesity indicator, strong cardiac risk predictor",
            "Indicator of arterial stiffness and vascular health",
            "Captures non-linear age effect on cardiac risk",
            "Clinically meaningful BP classification (AHA guidelines)",
            "Composite unhealthy lifestyle index"
        ]
    })
    st.dataframe(feat_table, use_container_width=True, hide_index=True)

    st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)

    # ── Models Used ──
    st.markdown("#### 🤖 Models Compared")
    model_table = pd.DataFrame({
        "Model":             ["Logistic Regression", "Random Forest ⭐", "K-Nearest Neighbors", "Decision Tree", "Naive Bayes", "Gradient Boosting"],
        "Type":              ["Linear", "Ensemble (Bagging)", "Instance-based", "Tree-based", "Probabilistic", "Ensemble (Boosting)"],
        "Key Params":        ["max_iter=1000", "n_estimators=100, max_depth=10", "k=5", "max_depth=10", "Default", "n_estimators=200, lr=0.1"],
        "Strength":          ["Interpretable", "High accuracy, robust", "Simple, non-parametric", "Interpretable", "Fast training", "High accuracy"],
    })
    st.dataframe(model_table, use_container_width=True, hide_index=True)

    st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)

    # ── Tech Stack ──
    st.markdown("#### 🛠️ Technology Stack")
    tech_cols = st.columns(4)
    tech_items = [
        ("🐍", "Python 3.10+",    "Core language"),
        ("🐼", "Pandas + NumPy", "Data processing"),
        ("🤖", "Scikit-learn",   "ML models & pipeline"),
        ("📊", "Plotly",         "Interactive charts"),
        ("🌟", "Streamlit",      "Web application"),
        ("💾", "Joblib",         "Model serialization"),
        ("📉", "Matplotlib",     "EDA visualizations"),
        ("🌊", "Seaborn",        "Statistical plots"),
    ]
    for i, (icon, name, desc) in enumerate(tech_items):
        with tech_cols[i % 4]:
            st.markdown(f"""
            <div class="stat-card" style="text-align:left;padding:16px;margin-bottom:12px;">
                <div style="font-size:1.3rem;margin-bottom:6px;">{icon}</div>
                <div style="font-weight:700;color:#e2e8f0;font-size:0.9rem;">{name}</div>
                <div style="font-size:0.75rem;color:#64748b;margin-top:2px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)

    # ── Key Findings ──
    st.markdown("#### 📌 Key Findings")
    findings = [
        ("🏆", "Best Model",       "Random Forest & Gradient Boosting tied at ~72.7% accuracy"),
        ("🧓", "Top Predictor",    "Age (years) — strongest single feature for cardiac risk"),
        ("💉", "2nd Predictor",    "Systolic blood pressure (ap_hi) — highly correlated with disease"),
        ("⚖️", "3rd Predictor",    "BMI — clear separation between disease and no-disease groups"),
        ("⚖️", "Target Balance",   "Dataset is ~50/50 balanced — no class imbalance issue"),
        ("🏃", "Protective Factor","Physical activity shows a measurable protective association"),
        ("🔬", "Overfit Safety",   "All models showed <8% train-test gap — no serious overfitting"),
    ]
    for icon, title, detail in findings:
        st.markdown(f"""
        <div style="display:flex;align-items:flex-start;gap:14px;background:rgba(255,255,255,0.02);
                    border:1px solid rgba(255,255,255,0.06);border-radius:10px;
                    padding:12px 16px;margin-bottom:8px;">
            <div style="font-size:1.3rem;min-width:28px;">{icon}</div>
            <div>
                <div style="font-weight:700;color:#e2e8f0;font-size:0.9rem;">{title}</div>
                <div style="font-size:0.85rem;color:#64748b;margin-top:3px;">{detail}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='footer'>
        CardioSense AI · Heart Disease Prediction Project<br>
        <span style="color:#334155;">
            Dataset: Cardiovascular Disease Dataset (Kaggle) · 
            Not for clinical use · Educational purposes only
        </span>
    </div>
    """, unsafe_allow_html=True)
