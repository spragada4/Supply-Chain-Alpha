# backtest/backtest.py
#
# Tests whether our combined signal predicts
# forward stock returns. This is the alpha question.
#
# Strategy logic:
#   When port stress is HIGH and NLP stress is RISING
#   → short the sector ETF (XLI = Industrials)
#   When both are LOW → long or flat
#
# We measure: Sharpe ratio, hit rate, max drawdown

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import os
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Load combined signals
# ─────────────────────────────────────────────

def load_signals(path: str = "data/combined_signals.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"]    = pd.to_datetime(df["date"])
    df["quarter"] = pd.to_datetime(df["quarter"])
    return df


# ─────────────────────────────────────────────
# Fetch stock prices for our companies + ETFs
# ─────────────────────────────────────────────

def fetch_prices() -> pd.DataFrame:
    print("Fetching stock prices...")

    tickers = {
        "Apple":          "AAPL",
        "Nike":           "NKE",
        "FedEx":          "FDX",
        "Ford":           "F",
        "General Motors": "GM",
        "Target":         "TGT",
        "Walmart":        "WMT",
        "UPS":            "UPS",
        "3M":             "MMM",
        # Sector ETFs for aggregate signal
        "XLI":            "XLI",   # Industrials ETF
        "XLY":            "XLY",   # Consumer Discretionary ETF
    }

    frames = {}
    for name, ticker in tickers.items():
        try:
            df = yf.download(
                ticker,
                start    = "2019-01-01",
                end      = "2024-12-31",
                progress = False
            )
            if not df.empty:
                frames[name] = df["Close"]
                print(f"  ✅ {ticker}")
        except Exception as e:
            print(f"  ⚠️  {ticker}: {e}")

    prices = pd.DataFrame(frames)
    prices.index = pd.to_datetime(prices.index)
    return prices


# ─────────────────────────────────────────────
# Compute quarterly forward returns
# for each company
# ─────────────────────────────────────────────

def compute_forward_returns(prices: pd.DataFrame) -> pd.DataFrame:
    # Resample to quarter-end prices
    quarterly_prices = prices.resample("QE").last()

    # Forward 1-quarter return
    forward_returns = quarterly_prices.pct_change(1).shift(-1)
    forward_returns.index = forward_returns.index.to_period("Q").to_timestamp()

    return forward_returns


# ─────────────────────────────────────────────
# Build signal: aggregate stress per quarter
# across all companies
# ─────────────────────────────────────────────

def build_aggregate_signal(signals_df: pd.DataFrame) -> pd.DataFrame:
    agg = signals_df.groupby("quarter").agg(
        nlp_stress_mean     = ("stress_score",       "mean"),
        nlp_stress_max      = ("stress_score",       "max"),
        port_stress         = ("port_stress_index",  "first"),
        port_stress_lag1q   = ("port_stress_lag1q",  "first"),
        keyword_density     = ("keyword_density",    "mean"),
    ).reset_index()

    agg = agg.sort_values("quarter").reset_index(drop=True)

    # Composite signal: blend NLP + port stress
    # Port stress lagged by 1Q (lead indicator)
    agg["composite_signal"] = (
        0.5 * agg["nlp_stress_mean"] +
        0.3 * agg["port_stress"].fillna(0) +
        0.2 * agg["port_stress_lag1q"].fillna(0)
    )

    # Signal change (momentum)
    agg["signal_change"] = agg["composite_signal"].diff()

    # Z-score of composite signal
    roll_mean = agg["composite_signal"].rolling(4).mean()
    roll_std  = agg["composite_signal"].rolling(4).std()
    agg["signal_zscore"] = (agg["composite_signal"] - roll_mean) / roll_std.replace(0, 1)

    return agg


# ─────────────────────────────────────────────
# Backtest: signal vs XLI/XLY forward returns
#
# Rules:
#   signal_zscore > 1.0  → SHORT (stress is high)
#   signal_zscore < -0.5 → LONG  (stress is low)
#   else                 → FLAT
# ─────────────────────────────────────────────

def run_backtest(
    agg_signal:      pd.DataFrame,
    forward_returns: pd.DataFrame,
    target:          str = "XLI"
) -> pd.DataFrame:

    print(f"\nRunning backtest on {target}...")

    # Align signal with forward returns
    bt = agg_signal.copy()
    bt = bt.set_index("quarter")

    # Get target returns
    if target in forward_returns.columns:
        bt["fwd_return"] = forward_returns[target].reindex(bt.index)
    else:
        print(f"  ⚠️  {target} not in prices, using XLI")
        bt["fwd_return"] = forward_returns["XLI"].reindex(bt.index)

    # Generate positions
    bt["position"] = 0  # flat
    bt.loc[bt["signal_zscore"] >  1.0,  "position"] = -1  # short
    bt.loc[bt["signal_zscore"] < -0.5,  "position"] =  1  # long

    # Strategy return = position * forward return
    bt["strategy_return"] = bt["position"] * bt["fwd_return"]
    bt["buy_hold_return"]  = bt["fwd_return"]

    # Cumulative returns
    bt["cum_strategy"]  = (1 + bt["strategy_return"].fillna(0)).cumprod()
    bt["cum_buy_hold"]  = (1 + bt["buy_hold_return"].fillna(0)).cumprod()

    return bt


# ─────────────────────────────────────────────
# Compute performance metrics
# ─────────────────────────────────────────────

def compute_metrics(bt: pd.DataFrame) -> dict:
    strat = bt["strategy_return"].dropna()
    bh    = bt["buy_hold_return"].dropna()

    def sharpe(returns, rf=0.04/4):  # 4% annual risk-free, quarterly
        excess = returns - rf
        return (excess.mean() / excess.std()) * np.sqrt(4) if excess.std() > 0 else 0

    def max_drawdown(cum_returns):
        rolling_max = cum_returns.cummax()
        drawdown    = (cum_returns - rolling_max) / rolling_max
        return drawdown.min()

    def hit_rate(returns):
        active = returns[returns != 0]
        return (active > 0).mean() if len(active) > 0 else 0

    trades = bt[bt["position"] != 0]

    metrics = {
        "strategy_sharpe":      round(sharpe(strat), 3),
        "buyhold_sharpe":       round(sharpe(bh), 3),
        "strategy_total_return":round((bt["cum_strategy"].iloc[-1] - 1) * 100, 2),
        "buyhold_total_return": round((bt["cum_buy_hold"].iloc[-1] - 1) * 100, 2),
        "strategy_max_drawdown":round(max_drawdown(bt["cum_strategy"]) * 100, 2),
        "buyhold_max_drawdown": round(max_drawdown(bt["cum_buy_hold"]) * 100, 2),
        "hit_rate":             round(hit_rate(strat) * 100, 2),
        "n_trades":             int((bt["position"] != 0).sum()),
        "n_short":              int((bt["position"] == -1).sum()),
        "n_long":               int((bt["position"] == 1).sum()),
    }

    return metrics


# ─────────────────────────────────────────────
# Plot backtest results
# ─────────────────────────────────────────────

def plot_backtest(bt: pd.DataFrame, metrics: dict, agg: pd.DataFrame):
    fig = make_subplots(
        rows        = 3,
        cols        = 1,
        shared_xaxes= True,
        subplot_titles = (
            "Cumulative Returns: Strategy vs Buy & Hold (XLI)",
            "Composite Stress Signal (Z-Score)",
            "Portfolio Position (-1=Short, 0=Flat, 1=Long)",
        ),
        vertical_spacing = 0.08,
        row_heights      = [0.5, 0.3, 0.2],
    )

    # Panel 1: Cumulative returns
    fig.add_trace(go.Scatter(
        x    = bt.index,
        y    = bt["cum_strategy"],
        name = "Strategy",
        line = dict(color="darkgreen", width=2.5),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x    = bt.index,
        y    = bt["cum_buy_hold"],
        name = "Buy & Hold XLI",
        line = dict(color="steelblue", width=2, dash="dash"),
    ), row=1, col=1)

    # Annotate metrics on chart
    fig.add_annotation(
        x    = 0.01, y = 0.97,
        xref = "paper", yref = "paper",
        text = (
            f"Strategy Sharpe: {metrics['strategy_sharpe']}<br>"
            f"B&H Sharpe: {metrics['buyhold_sharpe']}<br>"
            f"Strategy Return: {metrics['strategy_total_return']}%<br>"
            f"Hit Rate: {metrics['hit_rate']}%<br>"
            f"Max Drawdown: {metrics['strategy_max_drawdown']}%"
        ),
        showarrow  = False,
        align      = "left",
        bgcolor    = "white",
        bordercolor= "grey",
        font       = dict(size=11),
    )

    # Panel 2: Signal z-score
    colors_signal = [
        "crimson" if z > 1.0 else ("seagreen" if z < -0.5 else "grey")
        for z in agg["signal_zscore"].fillna(0)
    ]

    fig.add_trace(go.Bar(
        x      = agg["quarter"],
        y      = agg["signal_zscore"],
        name   = "Signal Z-Score",
        marker_color = colors_signal,
    ), row=2, col=1)

    fig.add_hline(y=1.0,  line_dash="dot", line_color="crimson",
                  annotation_text="Short threshold", row=2, col=1)
    fig.add_hline(y=-0.5, line_dash="dot", line_color="seagreen",
                  annotation_text="Long threshold",  row=2, col=1)

    # Panel 3: Position
    pos_colors = [
        "crimson" if p == -1 else ("seagreen" if p == 1 else "lightgrey")
        for p in bt["position"]
    ]

    fig.add_trace(go.Bar(
        x      = bt.index,
        y      = bt["position"],
        name   = "Position",
        marker_color = pos_colors,
    ), row=3, col=1)

    # COVID shading
    for row in [1, 2, 3]:
        fig.add_vrect(
            x0="2021-01-01", x1="2022-06-30",
            fillcolor="orange", opacity=0.07,
            line_width=0, row=row, col=1,
        )

    fig.update_layout(
        title    = "Supply Chain Alpha Strategy — Backtest (2019–2024)",
        height   = 750,
        template = "plotly_white",
        showlegend = True,
        legend   = dict(orientation="h", y=-0.05),
    )

    fig.write_html("output/backtest_results.html")
    print("✅ Saved: output/backtest_results.html")


# ─────────────────────────────────────────────
# Per-company signal vs forward return analysis
# ─────────────────────────────────────────────

def company_level_analysis(
    signals_df:      pd.DataFrame,
    forward_returns: pd.DataFrame
) -> pd.DataFrame:

    ticker_map = {
        "Apple": "Apple", "Nike": "Nike", "FedEx": "FedEx",
        "General Motors": "General Motors", "Target": "Target",
        "Walmart": "Walmart", "UPS": "UPS", "3M": "3M",
    }

    results = []
    for company, col in ticker_map.items():
        co_df = signals_df[signals_df["company"] == company].copy()
        co_df = co_df.set_index("quarter")

        if col not in forward_returns.columns:
            continue

        co_df["fwd_return"] = forward_returns[col].reindex(co_df.index)
        valid = co_df.dropna(subset=["stress_score", "fwd_return"])

        if len(valid) < 4:
            continue

        corr = valid["stress_score"].corr(valid["fwd_return"])
        results.append({
            "company":           company,
            "stress_vs_fwd_corr": round(corr, 4),
            "n_quarters":        len(valid),
            "mean_stress":       round(valid["stress_score"].mean(), 4),
        })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    # Load data
    signals_df = load_signals("data/combined_signals.csv")
    prices     = fetch_prices()

    # Forward returns
    fwd_returns = compute_forward_returns(prices)

    # Aggregate signal
    agg_signal = build_aggregate_signal(signals_df)

    print("\nAggregate signal by quarter:")
    print(agg_signal[[
        "quarter", "nlp_stress_mean", "port_stress",
        "composite_signal", "signal_zscore"
    ]].to_string())

    # Run backtest
    bt      = run_backtest(agg_signal, fwd_returns, target="XLI")
    metrics = compute_metrics(bt)

    print(f"\n{'='*50}")
    print("BACKTEST RESULTS")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k:<30}: {v}")

    # Plot
    plot_backtest(bt, metrics, agg_signal)

    # Company level
    co_analysis = company_level_analysis(signals_df, fwd_returns)
    print(f"\nPer-company stress vs forward return correlation:")
    print(co_analysis.sort_values("stress_vs_fwd_corr").to_string())
    co_analysis.to_csv("data/company_analysis.csv", index=False)

    # Open chart
    os.system("open output/backtest_results.html")
