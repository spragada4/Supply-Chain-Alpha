# Supply Chain Alpha

Extracting trading signals from SEC 10-Q filings and geospatial shipping data.

## What this does

1. **Scrapes** MD&A sections from 10-Q filings via SEC EDGAR API
2. **Scores** supply chain language using FinBERT financial NLP
3. **Builds** a Port Stress Index from shipping market proxies
4. **Backtests** a long/short signal against the XLI Industrials ETF

## Key results

- NLP stress score peaked Q4 2022 — matches documented supply chain crisis
- Apple stress score: -0.89 correlation with forward returns
- Strategy max drawdown: -16.3% vs -27.0% buy & hold
- Hit rate: 66.7% across 12 trades (2019–2024)

## Stack

Python · SEC EDGAR API · FinBERT (ProsusAI) · yfinance · GeoPandas · Plotly

## Structure

```
scraper/   — EDGAR 10-Q MD&A extraction
nlp/       — FinBERT supply chain stress scoring  
ports/     — Port Stress Index construction
backtest/  — Signal backtesting and performance metrics
output/    — Charts and research note
```

## Run it

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python3 scraper/edgar_scraper.py
python3 nlp/finbert_pipeline.py
python3 ports/port_stress.py
python3 backtest/backtest.py
```

## Results walkthrough

### NLP Signal
FinBERT scores supply chain paragraphs in each 10-Q MD&A section.
Stress score = 0.7 × negative sentiment + 0.3 × keyword density.

### Port Stress Index
Blends shipping ETF market data (BDRY, ZIM, MATX) with a calibrated
event index covering LA/LB backlog, Suez, Shanghai lockdowns, Red Sea.

### Backtest
Long/short XLI based on composite signal z-score.
Signal correctly identified 2021–2022 crisis and 2023 normalisation.

## Research note

See `output/research_note.md` for full methodology and results.

## Extending this

- Expand universe to 50–200 companies for stronger signal
- Replace calibrated port index with live AIS data (Spire, exactEarth)
- Add satellite footfall layer (RS Metrics) for retail companies
- Fine-tune FinBERT on supply chain specific language
- Add transaction cost model to backtest

## Author

Built as a portfolio project demonstrating geospatial + NLP alpha research.
Not investment advice.