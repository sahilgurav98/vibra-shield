import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.graph_objs as go
import time
from scipy.fft import fft

# --- UI Styling (Cyberpunk/Hacker Theme) ---
st.set_page_config(page_title="Bridge SHM System", layout="wide")
st.markdown("""
    <style>
    .stApp {
        background-color: #0d1117;
        color: #00ff00;
        font-family: 'Courier New', Courier, monospace;
    }
    h1, h2, h3 {
        color: #00ff00;
        text-shadow: 0 0 5px #00ff00;
    }
    .stButton>button {
        color: #0d1117;
        background-color: #00ff00;
        border: 1px solid #00ff00;
        box-shadow: 0 0 10px #00ff00;
    }
    .stMetric label {
        color: #00ff00 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🚧 Bridge Structural Health Monitoring [AI-CORE]")
st.markdown("### Real-Time Vibration Analysis & Damage Detection")

# --- Objective 2: Identify Damage-Sensitive Features ---
def extract_features(data_window):
    """
    Extracts time-domain and frequency-domain features from raw accelerometer data.
    """
    features = {}
    # Time-domain features
    features['Mean'] = np.mean(data_window)
    features['Std_Dev'] = np.std(data_window)
    features['RMS'] = np.sqrt(np.mean(data_window**2))
    features['Peak_to_Peak'] = np.max(data_window) - np.min(data_window)
    
    # Frequency-domain feature (Simplified Dominant Frequency via FFT)
    yf = fft(data_window)
    features['Dominant_Freq_Mag'] = np.max(np.abs(yf[1:len(yf)//2])) 
    
    return features

# --- Objective 1 & 3: Model Training & Evaluation ---
st.sidebar.header("🔧 System Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Training Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Data Loaded Successfully!")
    
    # Assuming CSV has 'Sensor_Reading' and 'Status' (0 for Normal, 1 for Damaged)
    if 'Sensor_Reading' in df.columns and 'Status' in df.columns:
        
        st.header("1. Data Processing & Feature Extraction")
        st.write("Extracting damage-sensitive features (RMS, Std Dev, Frequencies)...")
        
        # Simulate rolling window feature extraction for the dataset
        window_size = 50
        feature_list = []
        labels = []
        
        for i in range(0, len(df) - window_size, window_size):
            window = df['Sensor_Reading'].iloc[i:i+window_size].values
            label = df['Status'].iloc[i+window_size-1] # Take label at end of window
            feats = extract_features(window)
            feature_list.append(feats)
            labels.append(label)
            
        feature_df = pd.DataFrame(feature_list)
        X = feature_df
        y = np.array(labels)
        
        st.dataframe(feature_df.head())
        
        # Train Model
        st.header("2. Model Evaluation Metrics")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{accuracy_score(y_test, predictions):.2f}")
        col2.metric("Precision", f"{precision_score(y_test, predictions, zero_division=0):.2f}")
        col3.metric("Recall", f"{recall_score(y_test, predictions, zero_division=0):.2f}")
        col4.metric("F1-Score", f"{f1_score(y_test, predictions, zero_division=0):.2f}")
        
        # --- Objective 4 & 5: Real-Time Monitoring & Alerts ---
        st.header("3. Real-Time Monitoring Telemetry")
        st.markdown("Simulating live accelerometer data stream... (Enhancing Bridge Safety via predictive maintenance)")
        
        if st.button("Initialize Live Monitoring"):
            chart_placeholder = st.empty()
            alert_placeholder = st.empty()
            
            # Simulate a live stream using the test data
            live_data_y = []
            live_data_x = []
            
            for i, row in X_test.iterrows():
                live_data_x.append(i)
                live_data_y.append(row['RMS']) # Plotting RMS vibration
                
                # Plotly Chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=live_data_x, y=live_data_y, mode='lines', 
                                         line=dict(color='#00ff00', width=2), name="RMS Vibration"))
                fig.update_layout(paper_bgcolor='#0d1117', plot_bgcolor='#0d1117', 
                                  font=dict(color='#00ff00'), margin=dict(l=0, r=0, t=30, b=0),
                                  title="Live Sensor RMS Data")
                
                chart_placeholder.plotly_chart(fig, width="stretch")
                
                # Make prediction on current window
                current_features = pd.DataFrame([row])
                pred = model.predict(current_features)[0]
                
                if pred == 1:
                    alert_placeholder.error("🚨 CRITICAL ALERT: Anomalous Vibration Detected! Potential Structural Damage. Dispatching inspection team.")
                else:
                    alert_placeholder.success("✅ Status Nominal: Structural integrity verified.")
                
                time.sleep(2.0) # Simulate time delay for streaming
                
                # Keep only the last 20 points for the rolling chart
                if len(live_data_x) > 20:
                    live_data_x.pop(0)
                    live_data_y.pop(0)
                    
    else:
        st.error("Dataset must contain 'Sensor_Reading' and 'Status' columns.")
else:
    st.info("Awaiting dataset upload. Please upload training data via the sidebar to initialize the AI model.")