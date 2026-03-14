import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.graph_objs as go
import time
from scipy.fft import fft


st.set_page_config(page_title="VIBRA-SHIELD Analytics", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    h1, h2, h3 { color: #4facfe; font-weight: 700; }
    .stMetric { background-color: #161b22; border-radius: 10px; padding: 15px; border: 1px solid #30363d; }
    .stButton>button { width: 100%; background: linear-gradient(45deg, #4facfe 0%, #00f2fe 100%); color: white; border: none; font-weight: bold; }
    .stAlert { border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ VIBRA-SHIELD: Structural Health Analytics")
st.markdown("#### *AI-Driven Early Warning System for Critical Infrastructure*")
st.divider()


def extract_features(data_window):
    features = {}
    features['Mean_Accel'] = np.mean(data_window)
    features['Signal_Std'] = np.std(data_window)
    features['RMS_Energy'] = np.sqrt(np.mean(data_window**2))
    features['Peak_Amp'] = np.max(data_window) - np.min(data_window)
    # Frequency Analysis via FFT
    yf = fft(data_window)
    features['Resonant_Freq_Idx'] = np.max(np.abs(yf[1:len(yf)//2])) 
    return features


with st.sidebar:
    st.header("Control Center")
    sim_speed = st.slider("Streaming Latency (Sec)", 0.1, 2.0, 0.5)
    st.divider()
    st.subheader("📁 Data Pipeline")
    training_file = st.file_uploader("1. Historical Training Data", type=["csv"], key="train")
    live_file = st.file_uploader("2. Live Telemetry Feed", type=["csv"], key="live")


if training_file is not None:
    df_train = pd.read_csv(training_file)
    
    if 'Sensor_Reading' in df_train.columns and 'Status' in df_train.columns:
        window_size = 50
        X_list, y_list = [], []
        
        for i in range(0, len(df_train) - window_size, window_size):
            window = df_train['Sensor_Reading'].iloc[i:i+window_size].values
            label = 1 if df_train['Status'].iloc[i:i+window_size].sum() > 0 else 0
            X_list.append(extract_features(window))
            y_list.append(label)
            
        X_df = pd.DataFrame(X_list)
        y_array = np.array(y_list)
        
        tab1, tab2 = st.tabs(["📊 Model Training", "🚀 Live Monitoring"])
        
        with tab1:
            st.subheader("Feature Engineering Preview")
            display_df = X_df.copy()
            display_df['Label'] = y_array
            # Updated width to 'stretch' for compatibility
            st.dataframe(display_df.head(10), width='stretch')
            
            X_fit, X_eval, y_fit, y_eval = train_test_split(X_df, y_array, test_size=0.3, random_state=42)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_fit, y_fit)
            preds = model.predict(X_eval)
            
            st.subheader("Performance Analytics")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Validation Accuracy", f"{accuracy_score(y_eval, preds)*100:.1f}%")
            m2.metric("Precision Score", f"{precision_score(y_eval, preds):.2f}")
            m3.metric("Recall (Sensitivity)", f"{recall_score(y_eval, preds):.2f}")
            m4.metric("F1 Diagnostic", f"{f1_score(y_eval, preds):.2f}")

            st.subheader("Critical Feature Importance")
            importance = pd.DataFrame({'Feature': X_df.columns, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
            fig_imp = go.Figure(go.Bar(x=importance['Importance'], y=importance['Feature'], orientation='h', marker_color='#4facfe'))
            fig_imp.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white"))
            st.plotly_chart(fig_imp, width='stretch')

        with tab2:
            if live_file is not None:
                st.subheader("Real-Time Structural Health Telemetry")
                df_live = pd.read_csv(live_file)
                
                if st.button("Initialize Live Stream"):
                    
                    col_rms, col_freq = st.columns(2)
                    rms_placeholder = col_rms.empty()
                    freq_placeholder = col_freq.empty()
                    status_placeholder = st.empty()
                    
                    hist_x = []
                    hist_rms = []
                    hist_freq = []
                    
                    for i in range(0, len(df_live) - window_size, window_size):
                        chunk = df_live['Sensor_Reading'].iloc[i:i+window_size].values
                        feats = extract_features(chunk)
                        pred = model.predict(pd.DataFrame([feats]))[0]
                        
                        hist_x.append(i)
                        hist_rms.append(feats['RMS_Energy'])
                        hist_freq.append(feats['Resonant_Freq_Idx'])
                        
                        # --- RMS Energy Chart ---
                        fig_rms = go.Figure(go.Scatter(x=hist_x, y=hist_rms, mode='lines+markers', line=dict(color='#00f2fe', width=3), name="RMS Energy"))
                        fig_rms.update_layout(title="Energy Profile (RMS)", template="plotly_dark", height=350, margin=dict(l=10, r=10, t=40, b=10))
                        rms_placeholder.plotly_chart(fig_rms, width='stretch')
                        
                        # --- Resonant Frequency Chart ---
                        fig_freq = go.Figure(go.Scatter(x=hist_x, y=hist_freq, mode='lines+markers', line=dict(color='#ff7e5f', width=3), name="Freq Index"))
                        fig_freq.update_layout(title="Frequency Profile (Resonance)", template="plotly_dark", height=350, margin=dict(l=10, r=10, t=40, b=10))
                        freq_placeholder.plotly_chart(fig_freq, width='stretch')
                        
                        if pred == 1:
                            status_placeholder.error("🚨 ANOMALY DETECTED: Structural integrity breach suspected. Analyzing frequency shift...")
                        else:
                            status_placeholder.success("✅ SYSTEM NOMINAL: Bridge dynamics within safety thresholds.")
                        
                        time.sleep(sim_speed)
                        if len(hist_x) > 20:
                            hist_x.pop(0)
                            hist_rms.pop(0)
                            hist_freq.pop(0)
            else:
                st.info("💡 Pro-Tip: Upload the 'live_telemetry.csv' in the sidebar to begin the simulation.")
    else:
        st.error("Invalid File Format: Ensure columns 'Sensor_Reading' and 'Status' are present.")
else:
    st.info("👋 Welcome! Please upload your historical training dataset to initialize the VIBRA-SHIELD AI engine.")
