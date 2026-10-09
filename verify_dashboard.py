"""Run with:  python verify_dashboard.py   (keep it in the same folder as streamlit_app.py)
Independent verification of min_variance_dashboard.py.
Reference answers are computed with separate code (closed forms, brute-force grids,
hand-written loops) and compared with the dashboard's own functions."""
import re, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, streamlit as st
from scipy.optimize import minimize

import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "streamlit_app.py")).read()
ns = {"st": st, "np": np, "pd": pd, "TRADING_DAYS": 252, "minimize": minimize}
for name in ["estimate_cov", "port_stats", "_sum_to_one", "_bounds", "_clean", "min_variance",
             "max_sharpe", "max_attainable_return", "efficient_frontier", "risk_contribution",
             "walk_forward", "perf_metrics", "parse_tickers"]:
    m = re.search(rf"(@st.cache_data\([^)]*\)\n)?def {name}\(.*?(?=\n\n\n|\n# ---)", src, re.S)
    exec(re.sub(r"@st.cache_data\([^)]*\)\n", "", m.group(0)), ns)
D = type("D", (), ns)  # dot access

results = []
def check(label, ok, detail=""):
    results.append((label, bool(ok), detail))

# ---------------------------------------------------------------- 1. Notebook figures
S_daily = np.array([[0.000496, 0.000173], [0.000173, 0.000395]])  # df.cov() from the notebook
w = np.array([0.5, 0.5])
nb_std = np.sqrt(w @ S_daily @ w) * np.sqrt(250)
check("Notebook 50/50 vol reproduces (250-day basis)", abs(nb_std - 0.27803) < 2e-4, f"{nb_std:.5f} vs notebook 0.27804")
cov = S_daily * 252
_, vol, _ = D.port_stats(w, np.zeros(2), cov, 0)
check("Dashboard 50/50 vol = notebook × √(252/250)", abs(vol - nb_std * np.sqrt(252 / 250)) < 1e-12, f"{vol:.5f}")
a, b, c = S_daily[0, 0], S_daily[1, 1], S_daily[0, 1]
wA = (b - c) / (a + b - 2 * c)  # textbook two-asset min-variance formula
w_mv = D.min_variance(cov, False)
check("AAPL/CAT min-variance weight = textbook formula", abs(w_mv[0] - wA) < 1e-6, f"AAPL {w_mv[0]:.4f} vs {wA:.4f}")
grid = np.linspace(0, 1, 100001)
vg = grid**2 * a + (1 - grid)**2 * b + 2 * grid * (1 - grid) * c
check("…and = brute-force search over 100,001 weights", abs(grid[vg.argmin()] - w_mv[0]) < 1e-4, f"grid {grid[vg.argmin()]:.4f}")
nb_best = min(np.sqrt(np.array([x, 1 - x]) @ S_daily @ np.array([x, 1 - x])) * np.sqrt(250) for x in np.random.default_rng(0).random(500))
dash_best = np.sqrt(w_mv @ S_daily @ w_mv) * np.sqrt(250)
check("Optimiser beats the notebook's best of 500 random draws", dash_best <= nb_best + 1e-12, f"{dash_best:.5f} ≤ {nb_best:.5f}")

# ---------------------------------------------------------------- 2. Optimisers on random problems
rng = np.random.default_rng(42)
worst = {"mv_short": 0, "mv_long": 0, "ms": 0, "cap": 0}
for trial in range(30):
    n = rng.integers(3, 9)
    A = rng.normal(size=(n, n)); cov = (A @ A.T / n + np.diag(rng.uniform(0.01, 0.1, n))) * 0.1
    mu = rng.normal(0.08, 0.05, n)
    # closed form (short allowed)
    inv1 = np.linalg.solve(cov, np.ones(n)); ref = inv1 / inv1.sum()
    worst["mv_short"] = max(worst["mv_short"], np.abs(D.min_variance(cov, True) - ref).max())
    # long-only: compare variance with an independent optimiser started from many points
    wl = D.min_variance(cov, False)
    best = min((minimize(lambda x: x @ cov @ x, x0 / x0.sum(), method="trust-constr",
                         bounds=[(0, 1)] * n, constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1}]).fun
                for x0 in rng.random((3, n))))
    worst["mv_long"] = max(worst["mv_long"], (wl @ cov @ wl - best) / best)
    # max Sharpe (short allowed) vs tangency formula
    t = np.linalg.solve(cov, mu - 0.03); t /= t.sum()
    if t @ mu > 0.03:
        worst["ms"] = max(worst["ms"], np.abs(D.max_sharpe(mu, cov, 0.03, True) - t).max())
    # cap respected and sums to one
    cap = max(1 / n + 0.05, 0.3)
    wc = D.min_variance(cov, False, cap)
    worst["cap"] = max(worst["cap"], max(wc.max() - cap, abs(wc.sum() - 1), -wc.min()))
check("Min-variance with shorting = closed form Σ⁻¹1/1'Σ⁻¹1 (30 random problems)", worst["mv_short"] < 1e-9, f"max diff {worst['mv_short']:.1e}")
check("Long-only min-variance is the true minimum (vs independent solver)", worst["mv_long"] < 1e-5, f"worst relative gap {worst['mv_long']:.1e}")
check("Max-Sharpe with shorting = tangency formula Σ⁻¹(μ−rf)", worst["ms"] < 1e-9, f"max diff {worst['ms']:.1e}")
check("Weight cap respected, weights ≥ 0 and sum to 1", worst["cap"] < 1e-8, f"worst violation {worst['cap']:.1e}")

# long-only max Sharpe vs brute force (3 assets, simplex grid)
n = 3; A = rng.normal(size=(n, n)); cov = A @ A.T * 0.02 + np.eye(n) * 0.01; mu = np.array([0.06, 0.10, 0.14])
g = np.linspace(0, 1, 401); W = np.array([(x, y, 1 - x - y) for x in g for y in g if x + y <= 1 + 1e-12])
sh = (W @ mu - 0.02) / np.sqrt(np.einsum("ij,jk,ik->i", W, cov, W))
wms = D.max_sharpe(mu, cov, 0.02, False)
check("Long-only max-Sharpe ≥ best of 80k grid portfolios", (wms @ mu - 0.02) / np.sqrt(wms @ cov @ wms) >= sh.max() - 1e-6)

mu_low = np.array([0.01, 0.02, 0.015])
wf = D.max_sharpe(mu_low, cov, 0.20, True)
check("Max-Sharpe with shorting stays sensible when returns are below the risk-free rate", (wf >= -1e-12).all() and abs(wf.sum()-1) < 1e-9)
# frontier points are optimal: compare with the exact solution (enumerate held assets, solve KKT)
import itertools
def exact_min_vol(cov, mu, t):
    n=len(mu); best=np.inf
    for k in range(1,n+1):
        for Sx in itertools.combinations(range(n),k):
            Sx=list(Sx); C=cov[np.ix_(Sx,Sx)]; m=mu[Sx]
            K=np.zeros((k+2,k+2)); K[:k,:k]=2*C; K[:k,k]=K[k,:k]=1; K[:k,k+1]=K[k+1,:k]=m
            try: sol=np.linalg.solve(K, np.r_[np.zeros(k),1,t])
            except np.linalg.LinAlgError: continue
            if (sol[:k]>=-1e-10).all(): best=min(best, sol[:k]@C@sol[:k])
    return np.sqrt(best)
fr = D.efficient_frontier(mu, cov, False, 1.0, D.min_variance(cov, False))
gap = max((v - exact_min_vol(cov, mu, t)) / v for v, t in fr.values)
check("Every frontier point is the exact minimum risk for its return", gap < 1e-8 and len(fr) > 30, f"{len(fr)} points, worst gap {gap:.1e}")
check("Max attainable return with 40% cap = greedy fill", abs(D.max_attainable_return(mu, 0.4) - (0.4 * .14 + .4 * .10 + .2 * .06)) < 1e-12)

# ---------------------------------------------------------------- 3. Covariance + risk contribution
X = rng.normal(0, 0.01, (300, 5)) @ rng.normal(size=(5, 5))
S, _ = D.estimate_cov(X, "Sample")
check("Sample covariance = pandas .cov() × 252", np.allclose(S, pd.DataFrame(X).cov().values * 252))
L, d = D.estimate_cov(X, "Ledoit-Wolf shrinkage")
Xc = X - X.mean(0); T = len(X); Sm = Xc.T @ Xc / T; mu_ = np.trace(Sm) / 5
d2 = ((Sm - mu_ * np.eye(5))**2).sum()
b2 = min(d2, sum(((np.outer(x, x) - Sm)**2).sum() for x in Xc) / T**2)
ref = (b2 / d2) * mu_ * np.eye(5) + (1 - b2 / d2) * Sm
check("Ledoit-Wolf matches the formula from the 2004 paper", np.allclose(L, ref * 252) and abs(d - b2 / d2) < 1e-10, f"shrinkage {d:.3f}")
wv = rng.dirichlet(np.ones(5)); rc = D.risk_contribution(wv, S)
check("Risk contributions sum to 100%", abs(rc.sum() - 1) < 1e-12)
eps = 1e-6; num = np.array([(np.sqrt((wv + eps * np.eye(5)[i]) @ S @ (wv + eps * np.eye(5)[i])) - np.sqrt(wv @ S @ wv)) / eps for i in range(5)])
check("Risk contribution = weight × marginal risk (numerical derivative)", np.allclose(rc, wv * num / np.sqrt(wv @ S @ wv), atol=1e-4))
ws = D.min_variance(S, True); mrc = D.risk_contribution(ws, S) / ws
check("At unconstrained min-variance, every stock's marginal risk is equal", np.allclose(mrc, mrc[0]))

# ---------------------------------------------------------------- 4. Backtest
idx = pd.bdate_range("2012-01-02", "2019-12-31")
lr = pd.DataFrame(rng.normal(0.0004, 0.012, (len(idx), 4)) + rng.normal(0, 0.006, (len(idx), 1)), idx, list("ABCD"))
bt = D.walk_forward(lr, 252, "Quarterly", "Sample", False, 1.0, 0.0)
# hand-written reference: re-estimate with np.cov on the previous 252 rows, buy-and-hold between rebalances
simple = np.exp(lr.values) - 1; per = idx.to_period("Q"); v = 1.0; h = None; vals = []; ws_ref = []
for i in range(252, len(idx)):
    if i == 252 or per[i] != per[i - 1]:
        if h is not None: v = h.sum()
        C = np.cov(lr.values[i - 252:i], rowvar=False) * 252
        wref = D.min_variance(C, False); ws_ref.append(wref); h = wref * v
    h = h * (1 + simple[i]); vals.append(h.sum())
check("Walk-forward values = independent hand-written loop", np.allclose(vals, bt["Minimum variance"]["values"].values, rtol=1e-10))
check("Rebalance dates fall on the first trading day of each quarter",
      all(d == idx[(per == d.to_period("Q")).argmax()] for d in bt["Minimum variance"]["weights"].index[1:]))
# no look-ahead: scramble all data after a rebalance date; weights up to that date must not change
cut = bt["Minimum variance"]["weights"].index[10]
lr2 = lr.copy(); lr2.loc[lr2.index >= cut] = rng.normal(0, 0.05, lr2.loc[lr2.index >= cut].shape)
bt2 = D.walk_forward(lr2, 252, "Quarterly", "Sample", False, 1.0, 0.0)
check("No look-ahead: changing future data leaves past weights identical",
      np.allclose(bt["Minimum variance"]["weights"].iloc[:11].values, bt2["Minimum variance"]["weights"].iloc[:11].values))
check("…and the curve before that date is identical",
      np.allclose(bt["Minimum variance"]["values"][:cut].iloc[:-1], bt2["Minimum variance"]["values"][:cut].iloc[:-1]))
# trading costs: final value = no-cost value × Π(1 − turnover × cost)
bt_c = D.walk_forward(lr, 252, "Quarterly", "Sample", False, 1.0, 25.0)
check("Costs reduce the final value", bt_c["Minimum variance"]["values"].iloc[-1] < bt["Minimum variance"]["values"].iloc[-1])
ew = bt["Equal weight"]["values"]; ref_ew = []; v = 1.0; h = None
for i in range(252, len(idx)):
    if i == 252 or per[i] != per[i - 1]:
        if h is not None: v = h.sum()
        h = np.ones(4) / 4 * v
    h = h * (1 + simple[i]); ref_ew.append(h.sum())
check("Equal-weight backtest = hand-written loop", np.allclose(ref_ew, ew.values))

# ---------------------------------------------------------------- 5. Performance metrics
vals = pd.Series(np.cumprod(1 + rng.normal(0.0005, 0.01, 1260)), index=pd.bdate_range("2020-01-01", periods=1260))
m = D.perf_metrics(vals, 0.02)
r = pd.concat([pd.Series([1.0]), pd.Series(vals.values)]).pct_change().dropna().values
check("CAGR = (end value)^(252/days) − 1", abs(m["CAGR"] - (vals.iloc[-1] ** (252 / 1260) - 1)) < 1e-12)
check("Vol = daily std × √252", abs(m["Vol"] - r.std(ddof=1) * np.sqrt(252)) < 1e-12)
peak = np.maximum.accumulate(np.r_[1.0, vals.values]); check("Max drawdown = worst fall from running peak", abs(m["Max DD"] - (np.r_[1.0, vals.values] / peak - 1).min()) < 1e-12)
q = np.sort(r)[: int(np.ceil(0.05 * len(r)))]
check("CVaR 95% ≈ average of the worst 5% of days", abs(m["CVaR 95%"] + np.mean(q)) < 5e-4, f"{m['CVaR 95%']:.4%} vs {-np.mean(q):.4%}")
check("Ticker parsing handles commas, spaces, case, duplicates", D.parse_tickers("aapl, cat;msft  AAPL") == ["AAPL", "CAT", "MSFT"])

for label, ok, detail in results:
    print(("PASS " if ok else "FAIL ") + label + (f"  [{detail}]" if detail else ""))
print(f"\n{sum(ok for _, ok, _ in results)}/{len(results)} passed")
