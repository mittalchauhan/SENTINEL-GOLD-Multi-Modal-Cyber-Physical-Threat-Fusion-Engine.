import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import entropy
import json
import time
import os

# --- 1. CORE ARCHITECTURE ---
st.set_page_config(page_title="SENTINEL | EXECUTIVE", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #050505; color: #d4af37; font-family: 'Inter', sans-serif; }
    [data-testid="stHeader"] { display: none !important; }
    
    /* Executive Header */
    .exec-header {
        border-bottom: 1px solid #333; padding: 20px 0; margin-bottom: 30px;
        display: flex; justify-content: space-between; align-items: center;
    }
    
    /* Polished KPI Cards */
    .metric-card {
        background: #0e0e0e; border: 1px solid #1f1f1f;
        padding: 15px; border-radius: 8px; text-align: center;
        min-height: 100px; display: flex; flex-direction: column; justify-content: center;
    }
    .m-label { color: #8b8b8b; font-size: 0.65rem; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 5px; }
    .m-value { font-size: 1.4rem; font-weight: 700; }
    </style>
""", unsafe_allow_html=True)

# Fix: Data Loader
def load_data():
    if os.path.exists("business_stream.jsonl"):
        with open("business_stream.jsonl", "r") as f:
            return pd.DataFrame([json.loads(line) for line in f])
    return pd.DataFrame()

# Fix: Academic Math Block
def get_research_math(df):
    if len(df) < 5: return 0.0, 0.0
    scores = df['fused_score'].values
    # Entropy (H)
    counts = pd.Series(scores).value_counts(normalize=True)
    h_val = round(entropy(counts), 3)
    # Z-Score (Sigma)
    mu, sigma = np.mean(scores[:-1]), np.std(scores[:-1])
    z_val = round((scores[-1] - mu) / (sigma + 1e-9), 2)
    return h_val, z_val

def get_risk_analysis(df):
    if len(df) < 5: return "STABLE", 0.0
    recent_avg = df['fused_score'].tail(5).mean()
    slope = df['fused_score'].iloc[-1] - df['fused_score'].iloc[-5]
    forecast = df['fused_score'].iloc[-1] + (slope * 0.5)
    status = "CRITICAL" if recent_avg > 80 else "ELEVATED" if recent_avg > 50 else "STABLE"
    return status, round(forecast, 1)



# --- 3. THE HEADER (FIXED TOP POSITION) ---
st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid #222; margin-top: -60px;">
        <div style="font-size: 1.5rem; letter-spacing: 5px; font-weight: 200; color: #d4af37;">
            SENTINEL <span style="font-weight:900; color:#d4af37;">GOLD</span>
        </div>
        <div style="font-family: monospace; color: #238636; font-size: 0.85rem;">
            ● SYSTEM LIVE // RESEARCH_AUDIT // VECTOR_ID: 884-AX
        </div>
    </div>
    <br>
""", unsafe_allow_html=True)

placeholder = st.empty()

while True:
    df = load_data()
    with placeholder.container():
        if not df.empty:
            latest = df.iloc[-1]
            score = float(latest['fused_score'])
            status_text, pred_score = get_risk_analysis(df)
            h_val, z_val = get_research_math(df)
            uid = str(time.time()).replace(".","")

            # --- 4. KPI RIBBON (SIX METRICS IN ONE HORIZONTAL LINE) ---
            # Using 6 columns to keep everything on one line
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            
            with m1: st.markdown(f'<div class="metric-card"><p class="m-label">Live Threat</p><p class="m-value">{score}%</p></div>', unsafe_allow_html=True)
            
            d_color = "#ff4b4b" if "LOCKDOWN" in str(latest['rl_action']).upper() else "#d4af37"
            with m2: st.markdown(f'<div class="metric-card" style="border-top: 3px solid {d_color};"><p class="m-label">Decision</p><p class="m-value" style="color:{d_color}">{latest["rl_action"]}</p></div>', unsafe_allow_html=True)
            
            with m3: st.markdown(f'<div class="metric-card"><p class="m-label">Predictive</p><p class="m-value">{pred_score}%</p></div>', unsafe_allow_html=True)
            
            # --- THE NEW RESEARCH METRICS ---
            with m4: st.markdown(f'<div class="metric-card" style="border-top: 3px solid #7b2cbf;"><p class="m-label">Entropy (H)</p><p class="m-value">{h_val}</p></div>', unsafe_allow_html=True)
            with m5: st.markdown(f'<div class="metric-card" style="border-top: 3px solid #7b2cbf;"><p class="m-label">Z-Score (σ)</p><p class="m-value">{z_val}</p></div>', unsafe_allow_html=True)
            
            stat_color = "#ff4b4b" if abs(z_val) > 1.96 else "#238636"
            with m6: st.markdown(f'<div class="metric-card"><p class="m-label">Stat State</p><p class="m-value" style="color:{stat_color};">{"ANOMALY" if abs(z_val) > 1.96 else "NOMINAL"}</p></div>', unsafe_allow_html=True)

            st.markdown("---")

            # --- 5. LINE CHART & GAUGE ---
            col_chart, col_gauge = st.columns([2, 1])
            with col_chart:
                st.markdown("#####  THREAT VECTOR (LINE ANALYTICS)")
                st.write('  ')
                fig_line = go.Figure()
                fig_line.add_trace(go.Scatter(y=df['fused_score'], mode='lines+markers', line=dict(color='#d4af37', width=2), marker=dict(size=4, color='#fff')))
                fig_line.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350, margin=dict(l=0,r=0,t=10,b=0), xaxis_visible=False, yaxis_gridcolor="#1f1f1f")
                st.plotly_chart(fig_line, use_container_width=True, key=f"l_{uid}", config={'displayModeBar': False})

            with col_gauge:
                st.markdown("#####  CURRENT INTENSITY")
                fig_gauge = go.Figure(go.Indicator(mode = "gauge+number", value = score, gauge = {'axis': {'range': [0, 100], 'tickcolor': "#d4af37"}, 'bar': {'color': "#d4af37"}, 'steps': [{'range': [0, 100], 'color': '#111'}]}))
                fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "#d4af37"}, height=300)
                st.plotly_chart(fig_gauge, use_container_width=True, key=f"g_{uid}")

            # --- 6. REFINED AUDIT LOG (INTERACTIVE COLUMN GLOW) ---
            st.markdown("##### REAL-TIME AUDIT LOG (STATUS CODED)")

            # Copy the last 10 entries for processing
            log_df = df[['timestamp', 'fused_score', 'rl_action', 'llm_summary']].tail(10).iloc[::-1].copy()

            def apply_pro_style(s):
                # Base style: Clean grey for timestamp and fused_score data
                styles = ['color: #888; border-bottom: 1px solid #111;'] * len(s)
    
                # Logic for High Risk / Emergency (Red Highlights)
                if "LOCKDOWN" in str(s.rl_action).upper() or s.fused_score > 80:
                    # RL Action column becomes Bold Red
                    styles[2] = 'color: #ff4b4b; font-weight: bold; border-bottom: 1px solid #111;'
                    # LLM Summary column gets the Red Background Glow
                    styles[3] = 'background-color: rgba(255, 75, 75, 0.12); color: #ff4b4b; border-bottom: 1px solid #111;'
    
                # Logic for Elevated Risk (Gold Highlights)
                elif s.fused_score > 60:
                    styles[2] = 'color: #ffb700; border-bottom: 1px solid #111;'
                    styles[3] = 'background-color: rgba(212, 175, 55, 0.08); color: #d4af37; border-bottom: 1px solid #111;'
        
                return styles

            st.dataframe(log_df.style.apply(apply_pro_style, axis=1), use_container_width=True)

            # --- 7. THE NEURAL SIGNAL GRID (Purely Numerical / Visual) ---
            st.markdown("---")
            st.markdown("#####  SUBSYSTEM SIGNAL ARRAY (HEURISTIC NODES)")
            st.write(' ')
            
            # Generating 12 unique mathematical signals based on your score
            grid_cols = st.columns(6)
            signals = [
                ("VAR", np.var(df['fused_score'].tail(5))),
                ("DRIFT", z_val * 0.5),
                ("PEAK", df['fused_score'].max()),
                ("DECAY", h_val / 2),
                ("BIAS", (score - 50) / 10),
                ("FREQ", 1.5),
                ("RMS", np.sqrt(np.mean(df['fused_score']**2)) / 10),
                ("JIT", np.diff(df['fused_score'].tail(3)).mean() if len(df)>3 else 0),
                ("GAIN", score / pred_score if pred_score != 0 else 1),
                ("LOAD", len(df) % 100),
                ("SN-R", 44.2),
                ("VECT", score * z_val / 100)
            ]

            # Displaying them as high-tech "LED Nodes"
            for i, (label, val) in enumerate(signals):
                with grid_cols[i % 6]:
                    # Dynamic coloring based on value
                    sig_color = "#ff4b4b" if abs(val) > 1.5 else "#d4af37" if abs(val) > 0.5 else "#238636"
                    st.markdown(f"""
                        <div style="background: #0a0a0a; border: 1px solid #1a1a1a; padding: 10px; border-radius: 4px; text-align: center;">
                            <p style="margin:0; font-size:0.6rem; color:#555; letter-spacing:1px;">{label}</p>
                            <p style="margin:0; font-size:1.1rem; color:{sig_color}; font-family:monospace;">{abs(val):.2f}</p>
                            <div style="height:2px; width:100%; background:{sig_color}; opacity:0.3; margin-top:5px;"></div>
                        </div>
                    """, unsafe_allow_html=True)
        else:
            st.info(" INITIALIZING DATA VECTORS...")

    time.sleep(1.5)