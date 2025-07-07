# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

# Streamlit UI setup
st.set_page_config(page_title="Network Anomaly Detector", layout="wide")
st.title("🛡️ Network Anomaly Detection")
st.markdown("Upload your dataset to detect anomalies using Isolation Forest.")

with st.sidebar:
    uploaded_file = st.file_uploader("Upload File", type=["txt", "csv"])
    contamination = st.slider("Contamination Rate", 0.01, 0.20, 0.05)
    n_estimators = st.slider("Number of Trees", 50, 300, 100, step=10)

if uploaded_file:
    columns = [
        "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
        "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
        "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
        "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
        "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
        "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"
    ]

    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            if df.shape[1] != len(columns):
                st.warning("CSV detected. Reassigning correct column names...")
                df.columns = columns
        else:
            df = pd.read_csv(uploaded_file, header=None, names=columns, sep=",")

        if df.shape[1] != len(columns):
            st.error("❌ Incorrect file format. Please upload the NSL-KDD KDDTrain+.txt with 43 columns.")
            st.stop()

        original_df = df.copy()
        df.drop(columns=["label", "difficulty"], inplace=True)

        for col in ['protocol_type', 'service', 'flag']:
            try:
                df[col] = LabelEncoder().fit_transform(df[col])
            except Exception as e:
                st.warning(f"⚠️ Could not encode `{col}`: {e}")
                df[col] = 0

        model = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42)
        df['anomaly'] = model.fit_predict(df)

        st.subheader("📊 Detection Summary")
        anomaly_counts = df['anomaly'].value_counts().rename({1: "Normal", -1: "Anomaly"})
        col1, col2, col3 = st.columns(3)
        col1.metric("Total", len(df))
        col2.metric("Normal", anomaly_counts.get("Normal", 0))
        col3.metric("Anomalies", anomaly_counts.get("Anomaly", 0))


        fig = px.histogram(df, x='anomaly', color='anomaly', title="Anomaly Distribution")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("🔬 Feature Correlation Heatmap"):
            fig2, ax = plt.subplots(figsize=(14, 6))
            sns.heatmap(df.drop(columns=['anomaly']).corr(), cmap='coolwarm', ax=ax)
            st.pyplot(fig2)

        st.subheader("🧠 Summary Insights")
        try:
            top_proto = original_df.loc[df['anomaly'] == -1, 'protocol_type'].mode()[0]
            top_service = original_df.loc[df['anomaly'] == -1, 'service'].mode()[0]
            bytes_ratio = (
                df[df['anomaly'] == -1]['src_bytes'].mean() /
                df[df['anomaly'] == 1]['src_bytes'].mean()
            )
            st.markdown(f"- Top protocol in anomalies: **{top_proto}**")
            st.markdown(f"- Top service in anomalies: **{top_service}**")
            st.markdown(f"- `src_bytes` is **{bytes_ratio:.2f}x** higher in anomalies.")
        except Exception as e:
            st.warning("Could not generate full summary.")

        st.subheader("🚨 Sample Anomalies")
        st.dataframe(df[df['anomaly'] == -1].head(10))

        try:
            csv = df[df['anomaly'] == -1].to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Anomaly Report", data=csv, file_name="anomaly_report.csv")
        except Exception as e:
            st.warning(f"Download failed: {e}")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
else:
    st.info(" Upload a valid file to begin.")
