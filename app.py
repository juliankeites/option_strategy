import numpy as np, pandas as pd, streamlit as st, plotly.graph_objects as go
from scipy.stats import norm

st.set_page_config(page_title="Option Strategy Payoff", layout="wide")

PRESETS = {"Jet fuel (USD/t)": 800.0, "Brent (USD/bbl)": 85.0, "ICE Gasoil (USD/t)": 750.0,
           "EUR/USD": 1.08, "GBP/USD": 1.30, "Custom": 100.0}

def price(kind, S, K, T, vol, r, model):
    if T <= 0 or vol <= 0:
        return max(S-K, 0) if kind == "Call" else max(K-S, 0)
    if model == "Black-76":
        d1 = (np.log(S/K) + 0.5*vol**2*T)/(vol*np.sqrt(T)); d2 = d1 - vol*np.sqrt(T); df = np.exp(-r*T)
        return df*(S*norm.cdf(d1)-K*norm.cdf(d2)) if kind == "Call" else df*(K*norm.cdf(-d2)-S*norm.cdf(-d1))
    d1 = (np.log(S/K) + (r+0.5*vol**2)*T)/(vol*np.sqrt(T)); d2 = d1 - vol*np.sqrt(T)
    return S*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2) if kind == "Call" else K*np.exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1)

def intrinsic(kind, S, K):
    return np.maximum(S-K, 0) if kind == "Call" else np.maximum(K-S, 0)

st.title("European Option Strategy – P&L Payoff Diagram")
c = st.columns(4)
name = c[0].selectbox("1) Underlying", list(PRESETS))
if name == "Custom":
    name = st.sidebar.text_input("Custom name", "My underlying")
spot = c[1].number_input("Underlying price", value=PRESETS.get(name, 100.0), min_value=0.0001, format="%.4f")
T = c[2].number_input("Time to expiry (years)", value=0.5, min_value=0.01, step=0.05)
r = c[3].number_input("Risk-free rate (%)", value=4.0, step=0.25)/100
model = st.sidebar.radio("Pricing model", ["Black-76", "Black-Scholes"], help="Black-76 for futures/swaps/forwards; Black-Scholes for spot.")
mult = st.sidebar.number_input("Contract size / notional multiplier", value=1.0, min_value=0.0001)
ccy = st.sidebar.text_input("Currency / unit label", "USD")
show_mtm = st.sidebar.checkbox("Show value today (pre-expiry) curve", False)

st.subheader("2–3) Build legs")
TEMPLATES = {  # (Direction, Type, strike % of spot, vol %)
 "Custom": [("Long","Call",100,30),("Short","Call",115,28)],
 "Long call": [("Long","Call",100,30)],
 "Call spread (2-way)": [("Long","Call",100,30),("Short","Call",115,28)],
 "Zero-cost collar (2-way)": [("Long","Call",105,30),("Short","Put",95,32)],
 "Three-way (call spread + short put)": [("Long","Call",100,30),("Short","Call",120,28),("Short","Put",90,33)],
 "Four-way (call spread + put spread)": [("Long","Call",100,30),("Short","Call",120,28),("Short","Put",90,33),("Long","Put",80,36)],
}
tpl = st.selectbox("Strategy template (editable afterwards)", list(TEMPLATES), index=5)
qty0 = st.number_input("Default quantity per leg", value=1000.0, min_value=0.0)
t = TEMPLATES[tpl]
n = int(st.number_input("Number of legs", 1, 8, len(t), key=f"n_{tpl}"))
rows0 = [(d, ty, round(spot*k/100, 2), v, qty0) for d, ty, k, v in t]
while len(rows0) < n: rows0.append(("Long", "Call", round(spot, 2), 30.0, qty0))
default = pd.DataFrame(rows0[:n], columns=["Direction","Type","Strike","Vol %","Qty"])
legs = st.data_editor(
    default, key=f"legs_{tpl}_{n}_{spot}_{qty0}", num_rows="fixed", use_container_width=True,
    column_config={"Direction": st.column_config.SelectboxColumn(options=["Long", "Short"]),
                   "Type": st.column_config.SelectboxColumn(options=["Call", "Put"]),
                   "Strike": st.column_config.NumberColumn(min_value=0.0001, format="%.4f"),
                   "Vol %": st.column_config.NumberColumn(min_value=0.1, format="%.2f"),
                   "Qty": st.column_config.NumberColumn(min_value=0.0, format="%.2f")})

rows, S_grid = [], np.linspace(spot*0.3, spot*1.9, 1200)
total_cost, payoff, mtm, call_net = 0.0, np.zeros_like(S_grid), np.zeros_like(S_grid), 0.0
for _, l in legs.iterrows():
    sign = 1 if l["Direction"] == "Long" else -1
    q = l["Qty"]*mult; v = l["Vol %"]/100
    p = price(l["Type"], spot, l["Strike"], T, v, r, model)
    total_cost += sign*p*q
    payoff += sign*intrinsic(l["Type"], S_grid, l["Strike"])*q
    mtm += sign*np.array([price(l["Type"], s, l["Strike"], T, v, r, model) for s in S_grid])*q
    if l["Type"] == "Call": call_net += sign*q
    rows.append({**l.to_dict(), "Unit premium": p, "Cash (+pay/-receive)": sign*p*q})
pnl = payoff - total_cost

def segments(x, y):
    xs, ys = [x[0]], [y[0]]
    for i in range(1, len(x)):
        if y[i-1]*y[i] < 0:
            xz = x[i-1] - y[i-1]*(x[i]-x[i-1])/(y[i]-y[i-1]); xs.append(xz); ys.append(0.0)
        xs.append(x[i]); ys.append(y[i])
    return np.array(xs), np.array(ys)
xs, ys = segments(S_grid, pnl)
fig = go.Figure()
fig.add_trace(go.Scatter(x=xs, y=np.where(ys >= 0, ys, 0), mode="lines", line=dict(width=0),
                         fill="tozeroy", fillcolor="rgba(0,160,0,0.35)", hoverinfo="skip", showlegend=False))
fig.add_trace(go.Scatter(x=xs, y=np.where(ys <= 0, ys, 0), mode="lines", line=dict(width=0),
                         fill="tozeroy", fillcolor="rgba(220,0,0,0.35)", hoverinfo="skip", showlegend=False))
for mask, col, nm in [(ys >= 0, "green", "Gain"), (ys <= 0, "red", "Loss")]:
    fig.add_trace(go.Scatter(x=xs, y=np.where(mask, ys, np.nan), mode="lines", line=dict(color=col, width=3), name=nm))
if show_mtm:
    fig.add_trace(go.Scatter(x=S_grid, y=mtm-total_cost, mode="lines", line=dict(color="grey", dash="dash"), name="Value today"))
fig.add_vline(x=spot, line_dash="dot", annotation_text=f"Spot {spot:g}")
fig.add_hline(y=0, line_color="black")
fig.update_layout(title=f"{name} – {tpl} – P&L at expiry", xaxis_title=f"{name} price at expiry",
                  yaxis_title=f"P&L ({ccy})", height=520, hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

def pnl_at(s):
    v = -total_cost
    for _, l in legs.iterrows():
        sign = 1 if l["Direction"] == "Long" else -1
        v += sign*float(intrinsic(l["Type"], s, l["Strike"]))*l["Qty"]*mult
    return v
up_unl, down_unl = call_net > 1e-12, call_net < -1e-12
mx = max(pnl.max(), pnl_at(0.0)); mn = min(pnl.min(), pnl_at(0.0))
max_gain_txt = "Unlimited" if up_unl else f"{mx:,.2f} {ccy}"
max_loss_txt = "Unlimited" if down_unl else f"{mn:,.2f} {ccy}"
be = sorted(set(round(x, 2) for x, y0, y1 in zip(xs[:-1], ys[:-1], ys[1:]) if y0 == 0 and y1 != 0))

st.subheader("4–6) Results")
m = st.columns(4)
m[0].metric("Net cost (+ pay / – receive)", f"{total_cost:,.2f} {ccy}")
m[1].metric("Max gain", max_gain_txt)
m[2].metric("Max loss", max_loss_txt)
m[3].metric("Breakevens", ", ".join(f"{b:,.2f}" for b in be) or "None")
if down_unl:
    st.error("Net short calls: loss is unlimited if price rises – flag for risk committee.")
st.dataframe(pd.DataFrame(rows).style.format({"Unit premium": "{:,.4f}", "Cash (+pay/-receive)": "{:,.2f}"}), use_container_width=True)
st.caption("Flat vol (no skew), European exercise, no transaction costs. Indicative only.")
