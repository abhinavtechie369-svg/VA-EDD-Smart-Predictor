# ============================================================
# VA-EDD AI LAB - GOD MODE VISUALIZATION VERSION
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import random
import time

from scipy.interpolate import griddata

from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

# ============================================================
# UI
# ============================================================

st.set_page_config(page_title="VA-EDD AI LAB", layout="wide")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #020617, #0f172a);
    color: white;
}
h1, h2, h3 {
    color: #38bdf8;
}
</style>
""", unsafe_allow_html=True)

st.title(" VA-EDD AI LAB ")

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_resource
def load_models():
    df = pd.read_csv("data/va_edd_dataset.csv")

    df['IP_TON'] = df['IP'] * df['TON']
    df['TON_TOFF'] = df['TON'] * df['TOFF']
    df['IP_VA'] = df['IP'] * df['VA']

    X = df[['IP','TON','TOFF','TR','VA','IP_TON','TON_TOFF','IP_VA']]
    y = df[['MRR','SR']]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ridge = MultiOutputRegressor(Ridge()).fit(X_scaled, y)
    rf = MultiOutputRegressor(RandomForestRegressor()).fit(X_scaled, y)

    return df, scaler, ridge, rf

df, scaler, ridge_model, rf_model = load_models()

# ============================================================
# MODEL
# ============================================================

model_choice = st.sidebar.selectbox("Model", ["Ridge", "Random Forest"])
model = ridge_model if model_choice == "Ridge" else rf_model

# ============================================================
# INPUT
# ============================================================

IP = st.sidebar.slider("IP", 9, 15, 12)
TON = st.sidebar.slider("TON", 60, 120, 90)
TOFF = st.sidebar.slider("TOFF", 15, 90, 45)
TR = st.sidebar.slider("TR", 800, 1200, 1000)
VA = st.sidebar.slider("VA", 8, 12, 10)

# ============================================================
# PREDICT
# ============================================================

def predict(IP, TON, TOFF, TR, VA):
    data = pd.DataFrame([{
        'IP': IP,
        'TON': TON,
        'TOFF': TOFF,
        'TR': TR,
        'VA': VA,
        'IP_TON': IP * TON,
        'TON_TOFF': TON * TOFF,
        'IP_VA': IP * VA
    }])

    scaled = scaler.transform(data)
    pred = model.predict(scaled)
    return pred[0][0], pred[0][1]

MRR_pred, SR_pred = predict(IP, TON, TOFF, TR, VA)

col1, col2 = st.columns(2)
col1.metric("MRR", f"{MRR_pred:.2f}")
col2.metric("SR", f"{SR_pred:.2f}")

# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    " Basic",
    " 2D",
    " 3D ",
    " 4D",
    " Simulation",
    " Optimization"
])

# ============================================================
# BASIC
# ============================================================

with tab1:
    st.plotly_chart(px.scatter(df, x='IP', y='MRR', color='MRR', template="plotly_dark"))
    st.plotly_chart(px.imshow(df.select_dtypes(include=np.number).corr(), text_auto=True))

# ============================================================
# 2D
# ============================================================

with tab2:
    cols = df.select_dtypes(include=np.number).columns.tolist()

    x = st.selectbox("X-axis", cols)
    y = st.selectbox("Y-axis", cols)

    fig = px.scatter(df, x=x, y=y, color='MRR', template="plotly_dark")
    st.plotly_chart(fig)

# ============================================================
# 3D GOD MODE
# ============================================================

# ============================================================
# 3D GOD MODE (ROBUST VERSION - NEVER FAILS)
# ============================================================

from scipy.interpolate import griddata

with tab3:
    st.subheader(" 3D Smart Visualization")

    cols = df.select_dtypes(include=np.number).columns.tolist()

    x_col = st.selectbox("X-axis", cols, key="x3d")
    y_col = st.selectbox("Y-axis", cols, key="y3d")
    z_col = st.selectbox("Z-axis", cols, key="z3d")

    x = df[x_col].values
    y = df[y_col].values
    z = df[z_col].values

    # Create grid
    xi = np.linspace(x.min(), x.max(), 50)
    yi = np.linspace(y.min(), y.max(), 50)
    xi, yi = np.meshgrid(xi, yi)

    zi = None

    # STEP 1: Try cubic
    try:
        zi = griddata((x, y), z, (xi, yi), method='cubic')
        method_used = "Cubic Interpolation"
    except:
        zi = None

    # STEP 2: fallback linear
    if zi is None or np.isnan(zi).sum() > 0:
        try:
            zi = griddata((x, y), z, (xi, yi), method='linear')
            method_used = "Linear Interpolation"
        except:
            zi = None

    # STEP 3: FINAL fallback → scatter
    if zi is None or np.isnan(zi).sum() > 0:

        st.warning(" Interpolation failed → showing Scatter Plot")

        fig = go.Figure(data=[go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(size=5, color=z, colorscale='Turbo')
        )])

    else:
        st.success(f"✔ Using {method_used}")

        fig = go.Figure()

        # Smooth surface
        fig.add_trace(go.Surface(
            x=xi,
            y=yi,
            z=zi,
            colorscale='Turbo',
            opacity=0.9
        ))

        # Real points
        fig.add_trace(go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(size=4, color='white'),
            name="Real Data"
        ))

    fig.update_layout(
        template="plotly_dark",
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col
        )
    )

    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 4D
# ============================================================

with tab4:
    st.markdown("""
    ### 4D Visualization

    - X → TON  
    - Y → IP  
    - Z → MRR  
    - Color → SR  
    """)

    fig = go.Figure(data=[go.Scatter3d(
        x=df['TON'],
        y=df['IP'],
        z=df['MRR'],
        mode='markers',
        marker=dict(size=5, color=df['SR'], colorscale='Turbo')
    )])

    fig.update_layout(template="plotly_dark")

    st.plotly_chart(fig)

# ============================================================
# SIMULATION
# ============================================================

with tab5:
    run = st.button("▶ Run Simulation")
    chart = st.empty()

    if run:
        for ton in np.linspace(60,120,10):
            ip_vals = np.linspace(9,15,10)

            mrr_vals = [predict(ip, ton, TOFF, TR, VA)[0] for ip in ip_vals]

            fig = px.line(x=ip_vals, y=mrr_vals, template="plotly_dark",
                          title=f"TON={round(ton,1)}")

            chart.plotly_chart(fig)
            time.sleep(0.2)

# ============================================================
# OPTIMIZATION
# ============================================================

with tab6:
    def fitness(ind):
        mrr, sr = predict(ind[0], ind[1], ind[2], TR, VA)
        return mrr - 0.5*sr

    pop = [[random.uniform(9,15), random.uniform(60,120), random.uniform(15,90)] for _ in range(10)]

    for _ in range(10):
        pop = sorted(pop, key=fitness, reverse=True)
        new = pop[:5]

        while len(new) < 10:
            p1, p2 = random.sample(pop[:5], 2)
            new.append([(p1[i]+p2[i])/2 for i in range(3)])

        pop = new

    best = max(pop, key=fitness)

    st.success("Optimal Parameters")
    st.write({
        "IP": round(best[0],2),
        "TON": round(best[1],2),
        "TOFF": round(best[2],2)
    })