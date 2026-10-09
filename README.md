# Minimum-Variance Portfolio Dashboard

An interactive dashboard that finds the lowest-risk mix of any stocks you type in, using daily prices from Yahoo Finance.

## What it shows

- **Efficient frontier:** random portfolios, the exact frontier, and the minimum-variance and max-Sharpe portfolios.
- **Weights, statistics and risk contribution:** what each portfolio holds, and where its risk comes from.
- **Correlation heatmap:** how closely each pair of stocks moves together.
- **Out-of-sample backtest:** weights re-estimated each month or quarter using only past data, with trading costs, compared against equal weight and a benchmark.
- **Drawdowns and weights over time:** how deep the losses went and how stable the allocation was.

Options include Ledoit-Wolf covariance shrinkage, a maximum weight per stock and short selling.

## Run it locally

```
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Check the maths

```
python verify_dashboard.py
```

This runs 29 checks against independent reference calculations and should end with `29/29 passed`.

*For education only, not investment advice. Weights are estimated from past data, and past performance doesn't guarantee future results.*
