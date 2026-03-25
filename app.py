import streamlit as st
import pandas as pd
import subprocess
import os
import tempfile

st.set_page_config(page_title="Headwaters", layout="wide")
st.title("Headwaters")
st.caption("Marketing Mix Modeling for Tourism & Hospitality")

REQUIRED_COLUMNS = ["date", "tv_spend", "social_spend", "search_spend", "competitor_spend", "sales"]

def run_model(data_path=None, output_dir="output"):
    cmd = ["python", "mmm_demo.py", "--output", output_dir]
    if data_path:
        cmd += ["--data", data_path]
    return subprocess.run(cmd, capture_output=True, text=True)

def show_charts(output_dir="output"):
    col1, col2 = st.columns(2)
    with col1:
        path = f"{output_dir}/actual_vs_predicted.png"
        if os.path.exists(path):
            st.image(path, caption="Actual vs Predicted")
    with col2:
        path = f"{output_dir}/channel_contributions.png"
        if os.path.exists(path):
            st.image(path, caption="Average Channel Contributions")

# ─────────────────────────────────────────────
# PATH 1 — Sample Data Demo
# ─────────────────────────────────────────────
st.subheader("See how it works")

if st.button("▶  See a live demo with sample data", type="primary", use_container_width=True):
    st.session_state.show_demo = True
    st.session_state.show_upload = False
    # Clear any prior demo results so it reruns fresh
    st.session_state.pop("demo_ok", None)

if st.session_state.get("show_demo"):
    if "demo_ok" not in st.session_state:
        with st.spinner("Running model on sample data..."):
            result = run_model(output_dir="output_demo")
        st.session_state.demo_ok = (result.returncode == 0)
        st.session_state.demo_stdout = result.stdout
        st.session_state.demo_stderr = result.stderr

    st.info(
        "**Demo Mode** — these results use synthetic sample data modeled on a "
        "typical New England inn. Upload your own data below to see your actual numbers."
    )

    if st.session_state.demo_ok:
        show_charts(output_dir="output_demo")
        if st.session_state.get("demo_stdout"):
            with st.expander("Model summary"):
                st.code(st.session_state.demo_stdout)
    else:
        st.error("Demo failed to run.")
        st.code(st.session_state.get("demo_stderr", ""))

st.divider()

# ─────────────────────────────────────────────
# PATH 2 — Upload My Own Data
# ─────────────────────────────────────────────
st.subheader("Analyze your own data")

if st.button("Upload my own data", use_container_width=True):
    st.session_state.show_upload = True

if st.session_state.get("show_upload"):
    st.markdown(
        """
**Your CSV needs these columns:**

| Column | Description |
|---|---|
| `date` | Week or month (e.g. 2024-01-07) |
| `tv_spend` | TV / streaming ad spend |
| `social_spend` | Facebook / Instagram spend |
| `search_spend` | Google / Bing spend |
| `competitor_spend` | Estimated competitor spend (0 if unknown) |
| `sales` | Bookings or revenue for that period |

Go back 6–12 months minimum. One full season is enough to start.
        """
    )

    with open("sample_mmm_data.csv", "rb") as f:
        st.download_button(
            label="Download CSV template",
            data=f,
            file_name="headwaters_template.csv",
            mime="text/csv"
        )

    uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head(), use_container_width=True)

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
            st.stop()

        if st.button("Run my analysis", type="primary"):
            with st.spinner("Running your analysis..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                result = run_model(data_path=tmp_path, output_dir="output_user")
                os.unlink(tmp_path)

            if result.returncode != 0:
                st.error("Analysis failed.")
                st.code(result.stderr)
            else:
                st.success("Analysis complete")
                st.info("**Your Data** — results based on your actual marketing spend and booking history.")
                show_charts(output_dir="output_user")
                if result.stdout:
                    with st.expander("Model summary"):
                        st.code(result.stdout)
