import calendar, datetime as dt
import numpy as np, pandas as pd, streamlit as st, plotly.graph_objects as go
from scipy.stats import norm

st.set_page_config(page_title="Option Strategy Payoff", layout="wide")

PRESETS = {"Jet fuel (USD/t)":1500, "Jet fuel (USD/bbl)": 196.0, "Brent (USD/bbl)": 105.0,
           "ICE Gasoil (USD/t)": 1400, "EUR/USD": 1.08, "GBP/USD": 1.30, "Custom": 100.0}

def price(kind, F, K, T, vol, r, model):
    if T <= 0 or vol <= 0:
        return max(F-K, 0) if kind == "Call" else max(K-F, 0)
    if model == "Black-76":
        d1 = (np.log(F/K) + 0.5*vol**2*T)/(vol*np.sqrt(T)); d2 = d1 - vol*np.sqrt(T); df = np.exp(-r*T)
        return df*(F*norm.cdf(d1)-K*norm.cdf(d2)) if kind == "Call" else df*(K*norm.cdf(-d2)-F*norm.cdf(-d1))
    d1 = (np.log(F/K) + (r+0.5*vol**2)*T)/(vol*np.sqrt(T)); d2 = d1 - vol*np.sqrt(T)
    return F*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2) if kind == "Call" else K*np.exp(-r*T)*norm.cdf(-d2)-F*norm.cdf(-d1)

def intrinsic(kind, S, K):
    return np.maximum(S-K, 0) if kind == "Call" else np.maximum(K-S, 0)

def add_months(d, m):
    y, mo = divmod(d.month - 1 + m, 12)
    return dt.date(d.year + y, mo + 1, 1)

def last_bday(first):
    d = dt.date(first.year, first.month, calendar.monthrange(first.year, first.month)[1])
    while d.weekday() > 4: d -= dt.timedelta(days=1)
    return d

def segments(x, y):
    xs, ys = [x[0]], [y[0]]
    for i in range(1, len(x)):
        if y[i-1]*y[i] < 0:
            xs.append(x[i-1] - y[i-1]*(x[i]-x[i-1])/(y[i]-y[i-1])); ys.append(0.0)
        xs.append(x[i]); ys.append(y[i])
    return np.array(xs), np.array(ys)

def red_green(fig, xs, ys, lw=3, fill=True, label=True):
    if fill:
        fig.add_trace(go.Scatter(x=xs, y=np.where(ys >= 0, ys, 0), mode="lines", line=dict(width=0), fill="tozeroy",
                                 fillcolor="rgba(0,160,0,0.35)", hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=xs, y=np.where(ys <= 0, ys, 0), mode="lines", line=dict(width=0), fill="tozeroy",
                                 fillcolor="rgba(220,0,0,0.35)", hoverinfo="skip", showlegend=False))
    for mask, col, nm in [(ys >= 0, "green", "Gain"), (ys <= 0, "red", "Loss")]:
        fig.add_trace(go.Scatter(x=xs, y=np.where(mask, ys, np.nan), mode="lines", line=dict(color=col, width=lw),
                                 name=nm if label else None, showlegend=label))

# ------------------------------------------------------------------ sidebar
sb = st.sidebar
sb.header("Market & expiry")
name = sb.selectbox("1) Underlying", list(PRESETS))
if name == "Custom":
    name = sb.text_input("Custom name", "My underlying")
spot = sb.number_input("Underlying / forward price", value=PRESETS.get(name, 100.0), min_value=0.0001, format="%.4f",
                       help="Flat forward curve assumed: forward at every expiry = this price.")
today = dt.date.today()
months = sb.slider("Months ahead", 1, 6, 3)
first = add_months(today, months)
last = dt.date(first.year, first.month, calendar.monthrange(first.year, first.month)[1])
expiry = sb.date_input(f"Option expiry date ({first:%B %Y})", value=last_bday(first), min_value=first, max_value=last,
                       help="Set the exchange expiry day for this product (e.g. gasoil and Brent options expire on different dates).")
T = max((expiry - today).days, 1)/365.0
sb.caption(f"Expiry {expiry:%a %d %b %Y} | {(expiry-today).days} days | T = {T:.3f} yrs")
vol_all = sb.number_input("Volatility for all strikes (%)", value=30.0, min_value=0.1, step=1.0,
                          help="Flat, no skew. Sets every leg; override per leg in the table.")
r = sb.number_input("Risk-free rate (%)", value=4.0, step=0.25)/100
model = sb.radio("Pricing model", ["Black-76", "Black-Scholes"], help="Black-76 for futures/swaps; Black-Scholes for spot.")
mult = sb.number_input("Contract size multiplier", value=1.0, min_value=0.0001)
ccy = sb.text_input("Currency label", "USD")
show_mtm = sb.checkbox("Show value-today curve on option chart", False)

sb.header("Physical exposure")
phys_side = sb.radio("Physical position", ["Buyer of fuel (cost rises with price)", "Seller / producer (revenue rises with price)", "None"])
phys_vol = sb.number_input("Physical volume (same unit as underlying)", value=1000.0, min_value=0.0)
phys_ref = sb.number_input("Physical reference / budget price", value=float(spot), min_value=0.0001, format="%.4f",
                           help="Physical cost is measured against this price.")

# ------------------------------------------------------------------ legs
st.title("European Option Strategy – P&L Payoff Diagram")
st.subheader("2–3) Build legs")
TEMPLATES = {
 "Custom": [("Long","Call",100),("Short","Call",115)],
 "Long call": [("Long","Call",100)],
 "Call spread (2-way)": [("Long","Call",100),("Short","Call",115)],
 "Zero-cost collar (2-way)": [("Long","Call",105),("Short","Put",95)],
 "Three-way (call spread + short put)": [("Long","Call",100),("Short","Call",120),("Short","Put",90)],
 "Four-way (call spread + put spread)": [("Long","Call",100),("Short","Call",120),("Short","Put",90),("Long","Put",80)],
}
tpl = st.selectbox("Strategy template (editable afterwards)", list(TEMPLATES), index=5)
qty0 = st.number_input("Default quantity per leg", value=1000.0, min_value=0.0)
t = TEMPLATES[tpl]
n = int(st.number_input("Number of legs", 1, 8, len(t), key=f"n_{tpl}"))
rows0 = [(d, ty, round(spot*k/100, 2), vol_all, qty0) for d, ty, k in t]
while len(rows0) < n: rows0.append(("Long", "Call", round(spot, 2), vol_all, qty0))
default = pd.DataFrame(rows0[:n], columns=["Direction","Type","Strike","Vol %","Qty"])
legs = st.data_editor(
    default, key=f"legs_{tpl}_{n}_{spot}_{qty0}_{vol_all}", num_rows="fixed", use_container_width=True,
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

def pnl_at(s):
    v = -total_cost
    for _, l in legs.iterrows():
        sign = 1 if l["Direction"] == "Long" else -1
        v += sign*float(intrinsic(l["Type"], s, l["Strike"]))*l["Qty"]*mult
    return v

# ------------------------------------------------------------------ chart 1
xs, ys = segments(S_grid, pnl)
fig = go.Figure()
red_green(fig, xs, ys)
if show_mtm:
    fig.add_trace(go.Scatter(x=S_grid, y=mtm-total_cost, mode="lines", line=dict(color="grey", dash="dash"), name="Value today"))
fig.add_vline(x=spot, line_dash="dot", annotation_text=f"Fwd {spot:g}")
fig.add_hline(y=0, line_color="black")
fig.update_layout(title=f"{name} – {tpl} – option P&L at expiry ({expiry:%d %b %Y})", xaxis_title=f"{name} price at expiry",
                  yaxis_title=f"P&L ({ccy})", height=480, hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

up_unl, down_unl = call_net > 1e-12, call_net < -1e-12
mx = max(pnl.max(), pnl_at(0.0)); mn = min(pnl.min(), pnl_at(0.0))
be = sorted(set(round(x, 2) for x, y0, y1 in zip(xs[:-1], ys[:-1], ys[1:]) if y0 == 0 and y1 != 0))
st.subheader("4–6) Option strategy results")
m = st.columns(4)
m[0].metric("Net cost (+ pay / – receive)", f"{total_cost:,.2f} {ccy}")
m[1].metric("Max gain", "Unlimited" if up_unl else f"{mx:,.2f} {ccy}")
m[2].metric("Max loss", "Unlimited" if down_unl else f"{mn:,.2f} {ccy}")
m[3].metric("Breakevens", ", ".join(f"{b:,.2f}" for b in be) or "None")
if down_unl:
    st.error("Net short calls: loss is unlimited if price rises – flag for risk committee.")
st.dataframe(pd.DataFrame(rows).style.format({"Unit premium": "{:,.4f}", "Cash (+pay/-receive)": "{:,.2f}"}), use_container_width=True)

# ------------------------------------------------------------------ chart 2: physical + options
st.subheader("7) Physical exposure + option strategy = net position")
sgn = {"Buyer of fuel (cost rises with price)": -1, "Seller / producer (revenue rises with price)": 1, "None": 0}[phys_side]
phys = sgn*(S_grid - phys_ref)*phys_vol
net = phys + pnl
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=S_grid, y=phys, mode="lines", line=dict(color="rgba(120,120,120,0.6)", width=2, dash="dash"),
                          name="Physical (lowlight)"))
fig2.add_trace(go.Scatter(x=S_grid, y=pnl, mode="lines", line=dict(color="rgba(70,130,200,0.55)", width=2),
                          name="Option strategy (lowlight)"))
xn, yn = segments(S_grid, net)
red_green(fig2, xn, yn, lw=4, fill=True)
fig2.add_vline(x=spot, line_dash="dot", annotation_text=f"Fwd {spot:g}")
fig2.add_hline(y=0, line_color="black")
fig2.update_layout(title="Net P&L vs physical reference price (negative = cost increase / loss)",
                   xaxis_title=f"{name} price at expiry", yaxis_title=f"P&L ({ccy})", height=520, hovermode="x unified")
st.plotly_chart(fig2, use_container_width=True)

nm = st.columns(3)
if up_unl or down_unl or sgn == 0:
    pass
nm[0].metric("Net worst case (grid 30%-190% of fwd)", f"{net.min():,.0f} {ccy}")
nm[1].metric("Net best case (grid)", f"{net.max():,.0f} {ccy}")
nm[2].metric("Physical-only worst (grid)", f"{phys.min():,.0f} {ccy}")
shocks = [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3]
sc = []
for s in shocks:
    P = spot*(1+s); ph = sgn*(P-phys_ref)*phys_vol; op = pnl_at(P)
    sc.append({"Price move": f"{s:+.0%}", "Price": P, "Physical P&L": ph, "Option P&L": op, "Net P&L": ph+op,
               "Effective unit price (buyer)": (phys_ref*phys_vol - ph - op)/phys_vol if phys_vol else np.nan})
st.dataframe(pd.DataFrame(sc).style.format({"Price": "{:,.2f}", "Physical P&L": "{:,.0f}", "Option P&L": "{:,.0f}",
             "Net P&L": "{:,.0f}", "Effective unit price (buyer)": "{:,.2f}"}), use_container_width=True)
st.caption("Flat forward curve, flat vol (no skew), European exercise, no transaction costs/basis/margining. "
           "Physical P&L = side x (price - reference) x volume; option P&L includes premium paid/received. Indicative only.")
