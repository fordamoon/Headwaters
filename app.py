import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import tempfile
import os

st.set_page_config(page_title="Headwaters", layout="wide")
st.title("Headwaters")
st.caption("Marketing Mix Modeling for Tourism & Hospitality")

REQUIRED_COLUMNS = ["date", "tv_spend", "social_spend", "search_spend", "competitor_spend", "sales"]


def adstock(series: np.ndarray, decay: float = 0.5, max_lag: int = 4) -> np.ndarray:
    result = np.zeros_like(series, dtype=float)
    for i in range(len(series)):
        result[i] = series[i]
        for j in range(1, min(i, max_lag) + 1):
            result[i] += decay**j * series[i - j]
    return result


def run_mmm(df: pd.DataFrame):
    """Fit a simple adstock + linear regression MMM. Returns (model, X, y, y_pred, summary)."""
    for col in ["tv_spend", "social_spend", "search_spend"]:
        df[f"{col}_adstock"] = adstock(df[col].values)

    X = df[["tv_spend_adstock", "social_spend_adstock", "search_spend_adstock", "competitor_spend"]]
    y = df["sales"]
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)

    lines = [
        f"R-squared: {r2_score(y, y_pred):.4f}",
        f"Model intercept: {model.intercept_:.2f}",
    ]
    for name, coef in zip(X.columns, model.coef_):
        lines.append(f"{name}: {coef:.4f}")
    summary = "\n".join(lines)

    return model, X, y, y_pred, summary


def show_charts(df: pd.DataFrame, model, X, y, y_pred):
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        ax.plot(df["date"], y, label="Actual")
        ax.plot(df["date"], y_pred, label="Predicted")
        ax.legend()
        ax.set_title("Actual vs Predicted Sales")
        fig.autofmt_xdate()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        contrib = (X * model.coef_).mean()
        fig, ax = plt.subplots()
        contrib.plot(kind="bar", ax=ax)
        ax.set_title("Average Channel Contributions")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


def generate_demo_data() -> pd.DataFrame:
    np.random.seed(42)
    weeks = pd.date_range(start="2023-01-01", periods=20, freq="W")
    df = pd.DataFrame({
        "date": weeks,
        "tv_spend": np.random.gamma(200, 300, len(weeks)),
        "social_spend": np.random.gamma(100, 150, len(weeks)),
        "search_spend": np.random.gamma(150, 100, len(weeks)),
        "competitor_spend": np.random.gamma(150, 200, len(weeks)),
    })
    df["sales"] = (
        0.1 * df["tv_spend"]
        + 0.15 * df["social_spend"]
        + 0.12 * df["search_spend"]
        - 0.08 * df["competitor_spend"]
        + np.random.normal(0, 5000, len(weeks))
        + 50000
    )
    return df


# ─────────────────────────────────────────────
# PATH 1 — Sample Data Demo
# ─────────────────────────────────────────────
st.subheader("See how it works")

if st.button("▶  See a live demo with sample data", type="primary", use_container_width=True):
    st.session_state.show_demo = True
    st.session_state.show_upload = False
    st.session_state.pop("demo_results", None)

if st.session_state.get("show_demo"):
    if "demo_results" not in st.session_state:
        with st.spinner("Running model on sample data..."):
            demo_df = generate_demo_data()
            model, X, y, y_pred, summary = run_mmm(demo_df)
            st.session_state.demo_results = (demo_df, model, X, y, y_pred, summary)

    st.info(
        "**Demo Mode** — these results use synthetic sample data modeled on a "
        "typical New England inn. Upload your own data below to see your actual numbers."
    )

    demo_df, model, X, y, y_pred, summary = st.session_state.demo_results
    show_charts(demo_df, model, X, y, y_pred)
    with st.expander("Model summary"):
        st.code(summary)

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
        df = pd.read_csv(uploaded_file, parse_dates=["date"])
        st.dataframe(df.head(), use_container_width=True)

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
            st.stop()

        if st.button("Run my analysis", type="primary"):
            with st.spinner("Running your analysis..."):
                model, X, y, y_pred, summary = run_mmm(df)

            st.success("Analysis complete")
            st.info("**Your Data** — results based on your actual marketing spend and booking history.")
            show_charts(df, model, X, y, y_pred)
            with st.expander("Model summary"):
                st.code(summary)
