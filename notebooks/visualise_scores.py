# notebooks/visualise_scores.py

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import os

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────

df = pd.read_csv("data/stress_scores.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["company", "date"])

# Remove Ford (zero scores = extraction failed)
df = df[df["stress_score"] > 0]

companies = df["company"].unique()

# ─────────────────────────────────────────────
# Plot 1: Stress score over time per company
# ─────────────────────────────────────────────

fig1 = go.Figure()

colors = px.colors.qualitative.Set2

for i, company in enumerate(companies):
    co_df = df[df["company"] == company]
    fig1.add_trace(go.Scatter(
        x          = co_df["date"],
        y          = co_df["stress_score"],
        mode       = "lines+markers",
        name       = company,
        line       = dict(color=colors[i % len(colors)], width=2),
        marker     = dict(size=6),
        hovertemplate = (
            f"<b>{company}</b><br>"
            "Date: %{x|%Y-%m}<br>"
            "Stress Score: %{y:.3f}<br>"
            "<extra></extra>"
        )
    ))

# Mark COVID supply chain crisis period
fig1.add_vrect(
    x0          = "2021-01-01",
    x1          = "2022-06-30",
    fillcolor   = "red",
    opacity     = 0.08,
    line_width  = 0,
    annotation_text     = "COVID Supply Chain Crisis",
    annotation_position = "top left",
    annotation_font_size = 11,
)

fig1.update_layout(
    title      = "Supply Chain Stress Score by Company (2019–2024)",
    xaxis_title = "Quarter",
    yaxis_title = "Stress Score (0 = no stress, 1 = max stress)",
    hovermode  = "x unified",
    template   = "plotly_white",
    height     = 550,
    legend     = dict(orientation="h", yanchor="bottom", y=-0.3),
)

fig1.write_html("output/stress_over_time.html")
print("✅ Saved: output/stress_over_time.html")


# ─────────────────────────────────────────────
# Plot 2: Heatmap — stress by company x quarter
# ─────────────────────────────────────────────

df["quarter"] = df["date"].dt.to_period("Q").astype(str)
pivot = df.pivot_table(
    index   = "company",
    columns = "quarter",
    values  = "stress_score",
    aggfunc = "mean"
)

fig2 = go.Figure(data=go.Heatmap(
    z              = pivot.values,
    x              = pivot.columns.tolist(),
    y              = pivot.index.tolist(),
    colorscale     = "RdYlGn_r",  # red = high stress, green = low
    hoverongaps    = False,
    hovertemplate  = (
        "Company: %{y}<br>"
        "Quarter: %{x}<br>"
        "Stress Score: %{z:.3f}<br>"
        "<extra></extra>"
    )
))

fig2.update_layout(
    title       = "Supply Chain Stress Heatmap (Red = High Stress)",
    xaxis_title = "Quarter",
    yaxis_title = "Company",
    height      = 400,
    xaxis       = dict(tickangle=-45),
)

fig2.write_html("output/stress_heatmap.html")
print("✅ Saved: output/stress_heatmap.html")


# ─────────────────────────────────────────────
# Plot 3: Sentiment breakdown — neg vs pos vs neutral
# ─────────────────────────────────────────────

avg_sentiment = df.groupby("company")[
    ["avg_negative", "avg_positive", "avg_neutral"]
].mean().reset_index()

fig3 = go.Figure()

fig3.add_trace(go.Bar(
    name  = "Negative",
    x     = avg_sentiment["company"],
    y     = avg_sentiment["avg_negative"],
    marker_color = "crimson"
))
fig3.add_trace(go.Bar(
    name  = "Positive",
    x     = avg_sentiment["company"],
    y     = avg_sentiment["avg_positive"],
    marker_color = "seagreen"
))
fig3.add_trace(go.Bar(
    name  = "Neutral",
    x     = avg_sentiment["company"],
    y     = avg_sentiment["avg_neutral"],
    marker_color = "steelblue"
))

fig3.update_layout(
    barmode     = "stack",
    title       = "Average Sentiment Breakdown by Company (Supply Chain Paragraphs)",
    xaxis_title = "Company",
    yaxis_title = "Proportion",
    template    = "plotly_white",
    height      = 450,
    legend      = dict(orientation="h", yanchor="bottom", y=-0.25),
)

fig3.write_html("output/sentiment_breakdown.html")
print("✅ Saved: output/sentiment_breakdown.html")


# ─────────────────────────────────────────────
# Plot 4: Aggregate market stress index
# Average stress across all companies per quarter
# This is our headline signal
# ─────────────────────────────────────────────

market_stress = df.groupby("date")["stress_score"].mean().reset_index()
market_stress.columns = ["date", "market_stress_index"]

fig4 = go.Figure()

fig4.add_trace(go.Scatter(
    x    = market_stress["date"],
    y    = market_stress["market_stress_index"],
    mode = "lines+markers",
    name = "Market Stress Index",
    line = dict(color="darkred", width=3),
    fill = "tozeroy",
    fillcolor = "rgba(180,0,0,0.1)",
))

fig4.add_vrect(
    x0        = "2021-01-01",
    x1        = "2022-06-30",
    fillcolor = "red",
    opacity   = 0.08,
    line_width = 0,
    annotation_text      = "COVID Crisis",
    annotation_position  = "top left",
)

fig4.update_layout(
    title       = "Aggregate Supply Chain Stress Index (All Companies)",
    xaxis_title = "Quarter",
    yaxis_title = "Average Stress Score",
    template    = "plotly_white",
    height      = 400,
)

fig4.write_html("output/market_stress_index.html")
print("✅ Saved: output/market_stress_index.html")


# ─────────────────────────────────────────────
# Print summary stats
# ─────────────────────────────────────────────

print("\n" + "="*50)
print("SIGNAL SUMMARY")
print("="*50)

peak_quarter = market_stress.loc[market_stress["market_stress_index"].idxmax()]
print(f"Peak stress quarter : {peak_quarter['date'].strftime('%Y-%m')} "
      f"(score: {peak_quarter['market_stress_index']:.4f})")

low_quarter = market_stress.loc[market_stress["market_stress_index"].idxmin()]
print(f"Lowest stress quarter: {low_quarter['date'].strftime('%Y-%m')} "
      f"(score: {low_quarter['market_stress_index']:.4f})")

print(f"\nMost stressed company overall : "
      f"{df.groupby('company')['stress_score'].mean().idxmax()}")
print(f"Least stressed company overall: "
      f"{df.groupby('company')['stress_score'].mean().idxmin()}")

print("\nOpen output/ folder to view the interactive charts.")
