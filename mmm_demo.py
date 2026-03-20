import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os

def adstock(series: np.ndarray, decay: float = 0.5, max_lag: int = 4) -> np.ndarray:
    result = np.zeros_like(series, dtype=float)
    for i in range(len(series)):
        result[i] = series[i]
        for j in range(1, min(i, max_lag) + 1):
            result[i] += decay**j * series[i-j]
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="CSV file path")
    parser.add_argument("--output", type=str, default="output", help="Output folder")
    args = parser.parse_args()

    if args.data:
        df = pd.read_csv(args.data, parse_dates=["date"])
    else:
        print("No data file supplied. Generating synthetic sample dataset...")
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
            0.1*df["tv_spend"] +
            0.15*df["social_spend"] +
            0.12*df["search_spend"] -
            0.08*df["competitor_spend"] +
            np.random.normal(0, 5000, len(weeks)) +
            50000
        )
        df.to_csv("sample_generated_data.csv", index=False)

    for col in ["tv_spend","social_spend","search_spend"]:
        df[f"{col}_adstock"] = adstock(df[col].values)

    X = df[["tv_spend_adstock","social_spend_adstock","search_spend_adstock","competitor_spend"]]
    y = df["sales"]
    model = LinearRegression().fit(X,y)
    y_pred = model.predict(X)

    print(f"R-squared: {r2_score(y,y_pred):.4f}")
    print(f"Model intercept: {model.intercept_:.2f}")
    for n,c in zip(X.columns, model.coef_):
        print(f"{n}: {c:.4f}")

    os.makedirs(args.output, exist_ok=True)
    plt.figure()
    plt.plot(df["date"], y, label="Actual")
    plt.plot(df["date"], y_pred, label="Predicted")
    plt.legend(); plt.title("Actual vs Predicted Sales")
    plt.savefig(f"{args.output}/actual_vs_predicted.png")

    plt.figure()
    contrib = (X * model.coef_).mean()
    contrib.plot(kind="bar"); plt.title("Average Channel Contributions")
    plt.savefig(f"{args.output}/channel_contributions.png")

if __name__ == "__main__":
    main()
