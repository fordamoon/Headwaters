import streamlit as st
import pandas as pd
import subprocess
import os
import tempfile

st.set_page_config(page_title="Headwaters", layout="wide")
st.title("Headwaters")
st.caption("Marketing Mix Modeling for Tourism & Hospitality")

REQUIRED_COLUMNS = ["date", "tv_spend", "social_spend", "search_spend", "competitor_spend", "sales"]

st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head())

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    if st.button("Run MMM Model"):
        with st.spinner("Running model..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            result = subprocess.run(
                ["python", "mmm_demo.py", "--data", tmp_path, "--output", "output"],
                capture_output=True,
                text=True
            )
            os.unlink(tmp_path)

        if result.returncode != 0:
            st.error("Model failed to run.")
            st.code(result.stderr)
        else:
            st.success("Model complete")

            col1, col2 = st.columns(2)
            with col1:
                if os.path.exists("output/actual_vs_predicted.png"):
                    st.image("output/actual_vs_predicted.png", caption="Actual vs Predicted")
            with col2:
                if os.path.exists("output/channel_contributions.png"):
                    st.image("output/channel_contributions.png", caption="Average Channel Contributions")

            if result.stdout:
                st.subheader("Model Summary")
                st.code(result.stdout)
