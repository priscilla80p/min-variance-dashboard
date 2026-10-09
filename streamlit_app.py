"""
Minimum-Variance Portfolio Dashboard
Run with:  streamlit run streamlit_app.py
Requires:  pip install streamlit yfinance plotly scipy
"""
import html
import math
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize

# Force Streamlit's own dark theme so widgets match the black page. The theme is sent
# to the browser when the page first loads, so on the first run after the server starts
# we set it and reload the page once.
_THEME_NEEDS_RELOAD = False
try:
    _THEME_NEEDS_RELOAD = st.get_option("theme.primaryColor") != "#00e676"
    st._config.set_option("theme.base", "dark")
    st._config.set_option("theme.primaryColor", "#00e676")
    st._config.set_option("theme.backgroundColor", "#000000")
    st._config.set_option("theme.secondaryBackgroundColor", "#0b0b0e")
    st._config.set_option("theme.textColor", "#ecebf2")
except Exception:
    pass

st.set_page_config(page_title="Minimum-Variance Portfolio", page_icon="◆", layout="wide")
if _THEME_NEEDS_RELOAD:
    _reload = "<script>window.parent.location.reload()</script>"
    if hasattr(st, "iframe"):
        st.iframe(_reload, height=1)
    else:
        import streamlit.components.v1 as components
        components.html(_reload, height=0)
    st.stop()

TRADING_DAYS = 252

# Palette
BG, PANEL, LINE = "#000000", "#0b0b0e", "#1d1d24"
TEXT, MUTED = "#ecebf2", "#8a8a96"
GREEN, RED, VIOLET, GREY = "#00e676", "#ff5252", "#b388ff", "#6f6f7a"

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
    background: {BG} !important; color: {TEXT};
}}
[data-testid="stHeader"] {{ background: rgba(0,0,0,0.75) !important; }}
[data-testid="stSidebar"] {{ background: {PANEL} !important; border-right: 1px solid {LINE}; }}
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p,
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {{ color: {TEXT} !important; }}
h1, h2, h3, h4, p, label, li, .stMarkdown {{ font-family: 'Space Grotesk', sans-serif; }}
.block-container {{ padding-top: 3.2rem; }}

/* inputs */
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="input"] input,
[data-baseweb="select"] > div {{
    background: #141419 !important; color: {TEXT} !important; border-color: {LINE} !important;
}}
[data-testid="stTextInput"] input, [data-testid="stDateInput"] input,
[data-testid="stNumberInput"] input, div[data-baseweb="input"] > div {{
    background: #141419 !important; color: {TEXT} !important; -webkit-text-fill-color: {TEXT} !important;
}}
[data-testid="stNumberInput"] button {{ background: #141419 !important; color: {TEXT} !important; }}
[data-testid="stExpander"] details {{ background: {PANEL}; border: 1px solid {LINE}; }}
[data-testid="stExpander"] summary, [data-testid="stExpander"] p {{ color: {TEXT} !important; }}
.stDownloadButton button {{
    background: transparent !important; color: {GREEN} !important;
    border: 1px solid {GREEN} !important; border-radius: 999px; font-family: 'JetBrains Mono', monospace;
}}
.stDownloadButton button:hover {{ background: {GREEN} !important; color: #000 !important; }}

/* ticker tape */
.tape {{
    overflow: hidden; white-space: nowrap; background: #050507;
    border-top: 1px solid {LINE}; border-bottom: 1px solid {LINE};
    margin: 0 0 1.6rem 0; padding: 10px 0;
    mask-image: linear-gradient(90deg, transparent, #000 6%, #000 94%, transparent);
    -webkit-mask-image: linear-gradient(90deg, transparent, #000 6%, #000 94%, transparent);
}}
.track {{ display: inline-flex; animation: scroll var(--dur) linear infinite; }}
.tape:hover .track {{ animation-play-state: paused; }}
@keyframes scroll {{ from {{ transform: translateX(0); }} to {{ transform: translateX(-50%); }} }}
.tk {{ display: inline-flex; align-items: baseline; gap: 10px; padding: 0 28px;
       font-family: 'JetBrains Mono', monospace; font-size: 15px; border-right: 1px solid {LINE}; }}
.tk b {{ color: {TEXT}; letter-spacing: .06em; }}
.tk .px {{ color: {MUTED}; }}
.up {{ color: {GREEN}; }} .down {{ color: {RED}; }}

/* hero + sections */
.hero .t {{ font-family: 'Space Grotesk', sans-serif; line-height: 1.15; font-size: 2.6rem; font-weight: 700; margin: 0; letter-spacing: -0.02em; color: {TEXT}; }}
.hero .t .g {{ color: {GREEN}; }}
.hero p {{ color: {MUTED}; margin: .3rem 0 1.4rem 0; font-size: .95rem; }}
.sec {{ margin: 2.2rem 0 .6rem 0; }}
.sec h3 {{ margin: 0; font-size: 1.25rem; color: {TEXT}; font-weight: 600; }}
.sec h3::before {{ content: ""; display: inline-block; width: 8px; height: 8px; border-radius: 50%;
                   background: {GREEN}; margin-right: 10px; vertical-align: middle;
                   box-shadow: 0 0 10px {GREEN}; }}
.sec p {{ color: {MUTED}; margin: .35rem 0 0 18px; font-size: .9rem; max-width: 900px; }}

/* metric cards */
.cards {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }}
@media (max-width: 900px) {{ .cards {{ grid-template-columns: repeat(2, 1fr); }} }}
.card {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 14px; padding: 16px 18px; }}
.card .lbl {{ color: {MUTED}; font-size: .8rem; text-transform: uppercase; letter-spacing: .08em; }}
.card .val {{ font-family: 'JetBrains Mono', monospace; font-size: 1.7rem; color: {TEXT}; margin-top: 6px; }}
.card .sub {{ color: {MUTED}; font-size: .8rem; margin-top: 4px; }}

/* tables */
table.dt {{ width: 100%; border-collapse: collapse; font-family: 'JetBrains Mono', monospace; font-size: .85rem; }}
table.dt, table.dt th, table.dt td {{ border-left: none !important; border-right: none !important; white-space: nowrap; }}
table.dt th {{ color: {MUTED}; font-weight: 400; text-align: right; padding: 8px 10px;
              border-bottom: 1px solid {LINE}; font-family: 'Space Grotesk', sans-serif; }}
table.dt th:first-child, table.dt td:first-child {{ text-align: left; }}
table.dt td {{ color: {TEXT}; text-align: right; padding: 8px 10px; border-bottom: 1px solid #121217; }}
table.dt tr.port td {{ color: {GREEN}; border-top: 1px solid {LINE}; }}
table.dt tr:hover td {{ background: #0e0e12; }}
</style>
""", unsafe_allow_html=True)


def section(title, desc):
    st.markdown(f'<div class="sec"><h3>{title}</h3><p>{desc}</p></div>', unsafe_allow_html=True)


def dark(fig, height):
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, height=height,
        font=dict(family="Space Grotesk, sans-serif", color=TEXT, size=13),
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED)),
        hoverlabel=dict(bgcolor="#141419", bordercolor=LINE, font=dict(color=TEXT)))
    fig.update_xaxes(gridcolor="#16161c", zerolinecolor="#26262e", linecolor=LINE)
    fig.update_yaxes(gridcolor="#16161c", zerolinecolor="#26262e", linecolor=LINE)
    return fig


def html_table(df, fmt, port_rows=()):
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in [""] + list(df.columns))
    body = []
    for idx, row in df.iterrows():
        cells = []
        for c in df.columns:
            v = row[c]
            cells.append("<td>—</td>" if pd.isna(v) else f"<td>{fmt.get(c, '{:.4f}').format(v)}</td>")
        cls = ' class="port"' if idx in port_rows else ""
        body.append(f"<tr{cls}><td>{html.escape(str(idx))}</td>{''.join(cells)}</tr>")
    return f'<table class="dt"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def load_prices(tickers: tuple, start: date, end: date) -> pd.DataFrame:
    raw = yf.download(list(tickers), start=start, end=end,
                      auto_adjust=False, progress=False)
    if raw is None or raw.empty:
        return pd.DataFrame()
    prices = raw["Adj Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])
    return prices


def parse_tickers(text: str) -> list:
    parts = text.replace(";", ",").replace(" ", ",").split(",")
    seen, out = set(), []
    for p in parts:
        t = p.strip().upper()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


# ----------------------------------------------------------------------------
# Covariance estimation
# ----------------------------------------------------------------------------
def estimate_cov(log_ret: np.ndarray, method: str):
    """Annualised covariance. Returns (cov, shrinkage intensity or None)."""
    if method.startswith("Ledoit"):
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf().fit(log_ret)
            return lw.covariance_ * TRADING_DAYS, float(lw.shrinkage_)
        except ImportError:
            st.warning("scikit-learn isn't installed, so the sample covariance was used instead.")
    return np.cov(log_ret, rowvar=False) * TRADING_DAYS, None


# ----------------------------------------------------------------------------
# Optimisers (daily log returns; mean and covariance annualised)
# ----------------------------------------------------------------------------
def port_stats(w, mu, cov, rf):
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ cov @ w))
    sharpe = (ret - rf) / vol if vol > 0 else np.nan
    return ret, vol, sharpe


def _sum_to_one():
    return {"type": "eq", "fun": lambda w: w.sum() - 1}


def _bounds(n, allow_short, cap):
    return [(None, None)] * n if allow_short else [(0, cap)] * n


def _clean(w, allow_short):
    if not allow_short:
        w = np.clip(w, 0, None)
    return w / w.sum()


def min_variance(cov, allow_short, cap=1.0):
    n = len(cov)
    ones = np.ones(n)
    if allow_short:  # closed form: w = Σ⁻¹1 / 1'Σ⁻¹1
        return _clean(np.linalg.pinv(cov) @ ones, True)
    res = minimize(lambda w: w @ cov @ w, ones / n, method="SLSQP",
                   bounds=_bounds(n, False, cap), constraints=[_sum_to_one()],
                   options={"ftol": 1e-12, "maxiter": 1000})
    return _clean(res.x, False)


def max_sharpe(mu, cov, rf, allow_short, cap=1.0):
    n = len(cov)
    if allow_short:  # tangency portfolio: w ∝ Σ⁻¹(μ − rf)
        raw = np.linalg.pinv(cov) @ (mu - rf)
        if raw.sum() > 0:
            return _clean(raw, True)
        # no positive-Sharpe tangency exists (e.g. returns below the risk-free rate):
        # fall back to the best long-only portfolio rather than flipping the signs
        allow_short = False

    def neg_sharpe(w):
        r, v, _ = port_stats(w, mu, cov, rf)
        return -(r - rf) / v
    res = minimize(neg_sharpe, np.ones(n) / n, method="SLSQP",
                   bounds=_bounds(n, False, cap), constraints=[_sum_to_one()],
                   options={"ftol": 1e-12, "maxiter": 1000})
    return _clean(res.x, False)


def max_attainable_return(mu, cap):
    """Highest return a long-only, capped portfolio can reach (fill best names first)."""
    left, total = 1.0, 0.0
    for m in np.sort(mu)[::-1]:
        take = min(cap, left)
        total += take * m
        left -= take
        if left <= 1e-12:
            break
    return total


def efficient_frontier(mu, cov, allow_short, cap, w_minvar, points=40):
    n = len(cov)
    r_min = float(w_minvar @ mu)
    r_max = float(mu.max()) * 1.5 if allow_short else max_attainable_return(mu, cap)
    if r_max <= r_min:
        return pd.DataFrame(columns=["vol", "ret"])
    rows, x0 = [], w_minvar.copy()
    for target in np.linspace(r_min, r_max * 0.999, points):
        cons = [_sum_to_one(), {"type": "eq", "fun": lambda w, t=target: w @ mu - t}]
        res = minimize(lambda w: w @ cov @ w, x0, method="SLSQP",
                       bounds=_bounds(n, allow_short, cap), constraints=cons,
                       options={"ftol": 1e-12, "maxiter": 1000})
        if res.success:
            x0 = res.x
            rows.append((np.sqrt(res.x @ cov @ res.x), target))
    return pd.DataFrame(rows, columns=["vol", "ret"])


def random_portfolios(mu, cov, rf, n_sims, seed=42):
    rng = np.random.default_rng(seed)
    W = rng.dirichlet(np.ones(len(mu)), n_sims)  # long-only, sums to 1
    rets = W @ mu
    vols = np.sqrt(np.einsum("ij,jk,ik->i", W, cov, W))
    return pd.DataFrame({"ret": rets, "vol": vols, "sharpe": (rets - rf) / vols})


def risk_contribution(w, cov):
    """Share of portfolio variance coming from each asset (sums to 1)."""
    return w * (cov @ w) / (w @ cov @ w)


# ----------------------------------------------------------------------------
# Walk-forward (out-of-sample) backtest
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def walk_forward(log_ret: pd.DataFrame, lookback: int, freq: str, method: str,
                 allow_short: bool, cap: float, cost_bps: float):
    """Re-estimate the min-variance weights at each rebalance date using only the
    previous `lookback` days, then hold them (letting them drift) until the next one."""
    simple = np.exp(log_ret) - 1
    dates = log_ret.index
    period = dates.to_period("M" if freq == "Monthly" else "Q")
    rebal_idx = [i for i in range(lookback, len(dates)) if i == lookback or period[i] != period[i - 1]]
    if not rebal_idx:
        return None
    n = log_ret.shape[1]
    out = {}
    for name in ("Minimum variance", "Equal weight"):
        values, turnover, w_hist = [], [], []
        value, holdings = 1.0, np.zeros(n)
        for k, i0 in enumerate(rebal_idx):
            i1 = rebal_idx[k + 1] if k + 1 < len(rebal_idx) else len(dates)
            if name == "Minimum variance":
                cov, _ = estimate_cov(log_ret.values[i0 - lookback:i0], method)
                w = min_variance(cov, allow_short, cap)
            else:
                w = np.ones(n) / n
            if k > 0:
                value = holdings.sum()  # carry the gains/losses since the last rebalance
            drifted = holdings / holdings.sum() if k > 0 else np.zeros(n)
            to = np.abs(w - drifted).sum()
            turnover.append(to)
            value *= 1 - to * cost_bps / 10_000
            holdings = w * value
            w_hist.append(pd.Series(w, index=log_ret.columns, name=dates[i0]))
            for i in range(i0, i1):
                holdings = holdings * (1 + simple.values[i])
                values.append(holdings.sum())
        out[name] = {
            "values": pd.Series(values, index=dates[rebal_idx[0]:]),
            "turnover": float(np.mean(turnover[1:])) if len(turnover) > 1 else 0.0,
            "weights": pd.DataFrame(w_hist),
        }
    return out


def perf_metrics(values: pd.Series, rf: float, turnover=np.nan):
    v = pd.concat([pd.Series([1.0], index=[values.index[0] - pd.Timedelta(days=1)]), values])
    r = v.pct_change().dropna()
    years = len(r) / TRADING_DAYS
    cagr = v.iloc[-1] ** (1 / years) - 1
    vol = r.std() * np.sqrt(TRADING_DAYS)
    sharpe = (r.mean() * TRADING_DAYS - rf) / vol if vol > 0 else np.nan
    dd = v / v.cummax() - 1
    var95 = -np.quantile(r, 0.05)
    cvar95 = -r[r <= -var95].mean()
    return {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "Max DD": dd.min(),
            "CVaR 95%": cvar95, "Turnover": turnover}


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.header("Portfolio")
ticker_text = st.sidebar.text_input(
    "Tickers (comma or space separated)", "AAPL, CAT, MSFT, JNJ, XOM")
c1, c2 = st.sidebar.columns(2)
start = c1.date_input("Start", date(2015, 1, 1))
end = c2.date_input("End", date.today())
rf = st.sidebar.number_input("Risk-free rate (% p.a.)", 0.0, 20.0, 4.0, 0.25) / 100

st.sidebar.header("Optimiser")
cov_method = st.sidebar.radio(
    "Covariance estimate", ["Ledoit-Wolf shrinkage", "Sample"],
    help="Shrinkage pulls extreme covariance estimates towards a stable average, "
         "which usually gives steadier weights and lower risk out of sample.")
allow_short = st.sidebar.checkbox("Allow short selling", value=False)
tickers = parse_tickers(ticker_text)
min_cap = math.ceil(100 / max(len(tickers), 1))
cap = st.sidebar.slider("Max weight per stock (%)", min(min_cap, 100), 100, 100, 5,
                        disabled=allow_short,
                        help="Stops the optimiser piling into one stock. "
                             "Only applies when short selling is off.") / 100
n_sims = st.sidebar.slider("Random portfolios", 500, 20000, 5000, 500)

st.sidebar.header("Backtest")
bench = st.sidebar.text_input("Benchmark ticker", "SPY").strip().upper()
lookback = st.sidebar.slider("Lookback window (trading days)", 63, 756, 252, 21,
                             help="How much past data each rebalance uses to estimate risk.")
freq = st.sidebar.selectbox("Rebalance", ["Monthly", "Quarterly"], index=1)
cost_bps = st.sidebar.number_input("Trading cost (bps per $ traded)", 0.0, 100.0, 5.0, 1.0)

tape_slot = st.empty()  # filled once prices are loaded
st.markdown('<div class="hero"><div class="t">Minimum-<span class="g">Variance</span> Portfolio</div>'
            '<p>Type tickers in the sidebar. Everything below recalculates from Yahoo Finance prices.</p></div>',
            unsafe_allow_html=True)

if len(tickers) < 2:
    st.info("Enter at least two tickers in the sidebar.")
    st.stop()
if start >= end:
    st.error("Start date must be before end date.")
    st.stop()

with st.spinner(f"Downloading {', '.join(tickers)}…"):
    prices = load_prices(tuple(tickers), start, end)

if prices.empty:
    st.error("No data came back. Check the tickers and your internet connection.")
    st.stop()

missing = [t for t in tickers if t not in prices.columns or prices[t].dropna().empty]
if missing:
    st.warning(f"No price data for: {', '.join(missing)}. They were left out.")
prices = prices[[t for t in tickers if t not in missing]].dropna()

if prices.shape[1] < 2 or len(prices) < 30:
    st.error("Need at least two tickers with overlapping history (30+ trading days).")
    st.stop()

assets = list(prices.columns)
if not allow_short and cap * len(assets) < 1:
    cap = 1 / len(assets)

# ----------------------------------------------------------------------------
# Ticker tape
# ----------------------------------------------------------------------------
last, prev, first = prices.iloc[-1], prices.iloc[-2], prices.iloc[0]
items = []
for t in assets:
    day = last[t] / prev[t] - 1
    period_ret = last[t] / first[t] - 1
    cls, arrow = ("up", "▲") if day >= 0 else ("down", "▼")
    pcls = "up" if period_ret >= 0 else "down"
    items.append(
        f'<span class="tk"><b>{html.escape(t)}</b><span class="px">{last[t]:,.2f}</span>'
        f'<span class="{cls}">{arrow} {day:+.2%}</span>'
        f'<span class="px">period</span><span class="{pcls}">{period_ret:+.1%}</span></span>')
reps = max(2, math.ceil(10 / len(items)))
half = "".join(items * reps)
tape_slot.markdown(
    f'<div class="tape"><div class="track" style="--dur:{len(items) * reps * 4}s">{half}{half}</div></div>',
    unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Full-period optimisation
# ----------------------------------------------------------------------------
log_ret = np.log(prices / prices.shift(1)).dropna()
mu = log_ret.mean().values * TRADING_DAYS
cov_opt, shrink = estimate_cov(log_ret.values, cov_method)  # what the optimiser uses
cov, _ = estimate_cov(log_ret.values, "Sample")              # actual historical risk, used for reporting

w_mv = min_variance(cov_opt, allow_short, cap)
w_ms = max_sharpe(mu, cov_opt, rf, allow_short, cap)
w_eq = np.ones(len(assets)) / len(assets)
mv_ret, mv_vol, mv_sh = port_stats(w_mv, mu, cov, rf)
ms_ret, ms_vol, ms_sh = port_stats(w_ms, mu, cov, rf)
eq_ret, eq_vol, eq_sh = port_stats(w_eq, mu, cov, rf)
asset_vol = np.sqrt(np.diag(cov))
best_single = asset_vol.min()
best_name = assets[int(asset_vol.argmin())]

# ----------------------------------------------------------------------------
# Headline cards
# ----------------------------------------------------------------------------
cov_note = (f"Ledoit-Wolf shrinkage of {shrink:.0%}" if shrink is not None else "the sample covariance")
constraint_note = ("short selling allowed" if allow_short
                   else f"long-only, max {cap:.0%} per stock")
section("Snapshot",
        f"The minimum-variance portfolio built from {len(assets)} stocks over "
        f"{len(log_ret):,} trading days ({log_ret.index[0]:%b %Y} to {log_ret.index[-1]:%b %Y}), "
        f"using {cov_note}, {constraint_note}. Figures are annualised from daily log returns "
        "and estimated on this whole period. The backtest further down shows how it would have done in real time.")
diff = mv_vol - best_single
st.markdown(f"""
<div class="cards">
  <div class="card"><div class="lbl">Volatility</div><div class="val">{mv_vol:.2%}</div>
    <div class="sub">annualised standard deviation</div></div>
  <div class="card"><div class="lbl">Expected return</div><div class="val">{mv_ret:.2%}</div>
    <div class="sub">annualised mean log return</div></div>
  <div class="card"><div class="lbl">Sharpe ratio</div><div class="val">{mv_sh:.2f}</div>
    <div class="sub">risk-free rate {rf:.2%}</div></div>
  <div class="card"><div class="lbl">Risk vs. safest stock</div>
    <div class="val {'up' if diff <= 0 else 'down'}">{diff:+.2%}</div>
    <div class="sub">vs. {html.escape(best_name)} at {best_single:.2%}</div></div>
</div>""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Efficient frontier
# ----------------------------------------------------------------------------
section("Efficient frontier",
        "Each dot is a random long-only mix of your stocks, coloured by Sharpe ratio. The white line is the "
        "frontier, the lowest risk for each level of return under your constraints. The green star is the "
        "minimum-variance portfolio and the violet star is the max-Sharpe portfolio. All risk here is the "
        "actual historical volatility; with shrinkage on, the stars can sit a hair inside the line because "
        "their weights came from the shrunk estimate.")
sims = random_portfolios(mu, cov, rf, n_sims)
frontier = efficient_frontier(mu, cov, allow_short, cap, min_variance(cov, allow_short, cap))

fig = go.Figure()
fig.add_trace(go.Scattergl(
    x=sims.vol, y=sims.ret, mode="markers", name="Random portfolios",
    marker=dict(size=4, color=sims.sharpe, colorscale="Viridis", opacity=0.55,
                colorbar=dict(title="Sharpe", outlinewidth=0, tickfont=dict(color=MUTED))),
    hovertemplate="Vol %{x:.2%}<br>Return %{y:.2%}<extra></extra>"))
if not frontier.empty:
    fig.add_trace(go.Scatter(x=frontier.vol, y=frontier.ret, mode="lines",
                             name="Efficient frontier", line=dict(width=3, color=TEXT)))
fig.add_trace(go.Scatter(
    x=asset_vol, y=mu, mode="markers+text", text=assets, textposition="top center",
    textfont=dict(color=TEXT), name="Single stocks",
    marker=dict(size=10, symbol="diamond", color=GREY, line=dict(width=1, color=TEXT))))
fig.add_trace(go.Scatter(x=[mv_vol], y=[mv_ret], mode="markers", name="Minimum variance",
                         marker=dict(size=20, symbol="star", color=GREEN, line=dict(width=1, color="#000"))))
fig.add_trace(go.Scatter(x=[ms_vol], y=[ms_ret], mode="markers", name="Max Sharpe",
                         marker=dict(size=20, symbol="star", color=VIOLET, line=dict(width=1, color="#000"))))
fig.update_layout(xaxis_title="Annualised volatility", yaxis_title="Annualised return",
                  xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                  legend=dict(orientation="h", y=-0.18))
st.plotly_chart(dark(fig, 540), width="stretch")

# ----------------------------------------------------------------------------
# Weights + stats
# ----------------------------------------------------------------------------
left, right = st.columns([1, 1], gap="large")
with left:
    section("Portfolio weights",
            "How much of the portfolio goes into each stock. Minimum variance favours stable, "
            "lowly correlated names, while max Sharpe tilts towards stocks with the best return per unit of risk.")
    wdf = pd.DataFrame({"Asset": assets, "Minimum variance": w_mv, "Max Sharpe": w_ms})
    wlong = wdf.melt(id_vars="Asset", var_name="Portfolio", value_name="Weight")
    fw = px.bar(wlong, x="Asset", y="Weight", color="Portfolio", barmode="group",
                color_discrete_map={"Minimum variance": GREEN, "Max Sharpe": VIOLET})
    fw.update_layout(yaxis_tickformat=".0%", xaxis_title="", legend_title="",
                     legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(dark(fw, 420), width="stretch")

with right:
    section("Statistics",
            "Annualised return, volatility and Sharpe ratio for each stock and portfolio, "
            "with each stock's weight in the two optimised portfolios. The green rows are the three portfolios.")
    table = pd.DataFrame({
        "Return": mu, "Vol": asset_vol, "Sharpe": (mu - rf) / asset_vol,
        "Min-var": w_mv, "Max-Sharpe": w_ms}, index=assets)
    summary = pd.DataFrame(
        [[mv_ret, mv_vol, mv_sh, 1.0, np.nan], [ms_ret, ms_vol, ms_sh, np.nan, 1.0],
         [eq_ret, eq_vol, eq_sh, np.nan, np.nan]],
        index=["Min-variance", "Max Sharpe", "Equal weight"], columns=table.columns)
    full = pd.concat([table, summary])
    pct = "{:.2%}"
    st.markdown(html_table(full, {"Return": pct, "Vol": pct, "Sharpe": "{:.2f}",
                                  "Min-var": pct, "Max-Sharpe": pct},
                           port_rows=set(summary.index)), unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Risk contribution + correlation
# ----------------------------------------------------------------------------
left, right = st.columns([1, 1], gap="large")
with left:
    section("Risk contribution",
            "Where each portfolio's risk actually comes from. A stock's weight and its share of risk can be "
            "very different: equal weighting often lets one volatile stock dominate, while minimum variance spreads risk out.")
    rc = pd.DataFrame({"Asset": assets,
                       "Minimum variance": risk_contribution(w_mv, cov),
                       "Equal weight": risk_contribution(w_eq, cov)})
    rcl = rc.melt(id_vars="Asset", var_name="Portfolio", value_name="Share of risk")
    fr = px.bar(rcl, x="Asset", y="Share of risk", color="Portfolio", barmode="group",
                color_discrete_map={"Minimum variance": GREEN, "Equal weight": GREY})
    fr.update_layout(yaxis_tickformat=".0%", xaxis_title="", legend_title="",
                     legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(dark(fr, 440), width="stretch")

with right:
    section("Correlation",
            "How closely each pair of stocks moves together day to day. Blue (low or negative) pairs "
            "diversify each other the most, which is what lets the portfolio's risk fall below any single stock's.")
    fc = px.imshow(log_ret.corr(), text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fc.update_coloraxes(colorbar=dict(outlinewidth=0, tickfont=dict(color=MUTED)))
    st.plotly_chart(dark(fc, 440), width="stretch")

# ----------------------------------------------------------------------------
# Out-of-sample backtest
# ----------------------------------------------------------------------------
section("Out-of-sample backtest",
        f"A realistic test: every {freq.lower()[:-2]} the weights are re-estimated using only the previous "
        f"{lookback} trading days, as you would have had to in real time, then held until the next rebalance. "
        f"Each trade costs {cost_bps:g} bps. Compared with equal weight and {html.escape(bench) or 'no benchmark'}, held throughout.")

if len(log_ret) <= lookback + 21:
    st.info(f"Not enough history for a {lookback}-day lookback. Pick an earlier start date or a shorter lookback.")
else:
    with st.spinner("Running walk-forward backtest…"):
        bt = walk_forward(log_ret, lookback, freq, cov_method, allow_short, cap, cost_bps)

    curves = {"Minimum variance": bt["Minimum variance"]["values"],
              "Equal weight": bt["Equal weight"]["values"]}
    bt_start = curves["Minimum variance"].index[0]
    if bench:
        bpx = load_prices((bench,), start, end)
        if not bpx.empty and bench in bpx.columns and not bpx[bench].dropna().empty:
            b = bpx[bench].dropna()
            b = b.reindex(curves["Minimum variance"].index).ffill().dropna()
            prev_close = bpx[bench].dropna()
            prev_close = prev_close[prev_close.index < bt_start]
            base = prev_close.iloc[-1] if len(prev_close) else b.iloc[0]
            curves[bench] = b / base
        else:
            st.warning(f"Couldn't load benchmark {bench}.")

    rows = {}
    for name, series in curves.items():
        to = bt[name]["turnover"] if name in bt else np.nan
        rows[name] = perf_metrics(series, rf, to)
    metrics = pd.DataFrame(rows).T

    colors = {"Minimum variance": GREEN, "Equal weight": GREY, bench: VIOLET}
    fb = go.Figure()
    for name in reversed(list(curves)):  # min-variance drawn last, on top
        fb.add_trace(go.Scatter(
            x=curves[name].index, y=curves[name].values, name=name, mode="lines",
            line=dict(color=colors.get(name, TEXT), width=2.5 if name == "Minimum variance" else 1.6,
                      dash="dot" if name == "Equal weight" else None)))
    fb.update_layout(yaxis_title="Value of $1", xaxis_title="", legend=dict(orientation="h", y=-0.15),
                     yaxis_type="log")
    st.plotly_chart(dark(fb, 460), width="stretch")

    pct = "{:.2%}"
    st.markdown(html_table(metrics, {"CAGR": pct, "Vol": pct, "Sharpe": "{:.2f}", "Max DD": pct,
                                     "CVaR 95%": pct, "Turnover": pct},
                           port_rows={"Minimum variance"}), unsafe_allow_html=True)
    st.markdown(
        f'<p style="color:{MUTED};font-size:.82rem;margin-top:.6rem">'
        "CAGR is the compound annual growth rate. Max DD is the worst peak-to-trough fall. "
        "CVaR 95% is the average loss on the worst 5% of days. Turnover is the average share of the "
        "portfolio traded at each rebalance.</p>", unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="large")
    with left:
        section("Drawdowns",
                "How far each strategy sat below its previous high at every point. Shallower, shorter dips "
                "are what minimum variance is designed to deliver.")
        fd = go.Figure()
        for name in reversed(list(curves)):
            v = curves[name]
            fd.add_trace(go.Scatter(
                x=v.index, y=v / v.cummax() - 1, name=name, mode="lines",
                line=dict(color=colors.get(name, TEXT), width=2 if name == "Minimum variance" else 1.2),
                fill="tozeroy" if name == "Minimum variance" else None,
                fillcolor="rgba(0,230,118,0.12)"))
        fd.update_layout(yaxis_tickformat=".0%", xaxis_title="", legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(dark(fd, 420), width="stretch")

    with right:
        section("Weights over time",
                "The minimum-variance weights chosen at each rebalance in the backtest. Steady bands mean "
                "the allocation is stable; jumpy bands mean higher trading costs and less reliable estimates.")
        wh = bt["Minimum variance"]["weights"]
        fwt = go.Figure()
        palette = px.colors.qualitative.Bold
        for j, a in enumerate(wh.columns):
            fwt.add_trace(go.Scatter(x=wh.index, y=wh[a], name=a, mode="lines", stackgroup="w",
                                     line=dict(width=0.5, color=palette[j % len(palette)], shape="hv")))
        fwt.update_layout(yaxis_tickformat=".0%", xaxis_title="", legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(dark(fwt, 420), width="stretch")

# ----------------------------------------------------------------------------
# Covariance + download
# ----------------------------------------------------------------------------
section("Covariance matrix",
        "The annualised covariance of returns that the optimiser minimises over"
        + (", after Ledoit-Wolf shrinkage" if shrink is not None else "")
        + ". The diagonal holds each stock's variance.")
with st.expander("Show covariance matrix"):
    cov_df = pd.DataFrame(cov_opt, index=assets, columns=assets)
    st.markdown(html_table(cov_df, {c: "{:.5f}" for c in assets}), unsafe_allow_html=True)

st.write("")
st.download_button("↓ Download weights (CSV)",
                   wdf.to_csv(index=False).encode(), "portfolio_weights.csv", "text/csv")
