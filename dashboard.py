"""
Barrier Option Interactive Dashboard
=====================================
Runs Monte Carlo simulation and displays results in a browser dashboard.
No matplotlib needed — fully interactive via Plotly / Dash.

Run:
    python3 dashboard.py
    → http://127.0.0.1:8053
"""

from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
from scipy.stats import norm
from scipy.optimize import brentq

# ── Palette ───────────────────────────────────────────────────────────────────
BG    = "#0d1117"; CARD  = "#161b22"; BORDER = "#30363d"
GREEN = "#3fb950"; RED   = "#f85149"; BLUE   = "#58a6ff"
AMBER = "#d29922"; GREY  = "#8b949e"; WHITE  = "#e6edf3"
PURPLE= "#bc8cff"; FONT  = "Inter, sans-serif"

BASE = dict(paper_bgcolor=BG, plot_bgcolor=CARD,
            font=dict(family=FONT, color=WHITE, size=11),
            margin=dict(l=55, r=20, t=40, b=40),
            xaxis=dict(gridcolor=BORDER, linecolor=BORDER),
            yaxis=dict(gridcolor=BORDER, linecolor=BORDER),
            legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER))


# ── Pricing functions ─────────────────────────────────────────────────────────

def bs_call(S, K, T, r, sigma):
    if T <= 0: return max(S - K, 0)
    d1 = (np.log(S/K) + (r + .5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def barrier_price(S, K, B, T, r, sigma):
    if S >= B or T <= 0: return 0.0
    lam = r/sigma**2 + 0.5
    y1 = np.log(B**2/(S*K))/(sigma*np.sqrt(T)) + lam*sigma*np.sqrt(T)
    y2 = np.log(B/S)/(sigma*np.sqrt(T))         + lam*sigma*np.sqrt(T)
    x1 = np.log(S/K)/(sigma*np.sqrt(T))          + lam*sigma*np.sqrt(T)
    ref = (S*(B/S)**(2*lam)*(norm.cdf(y1)-norm.cdf(y2))
           - K*np.exp(-r*T)*(B/S)**(2*lam-2)
           *(norm.cdf(y1-sigma*np.sqrt(T))-norm.cdf(y2-sigma*np.sqrt(T))))
    return max(bs_call(S,K,T,r,sigma) - ref, 0)

def barrier_delta(S, K, B, T, r, sigma, h=0.01):
    return (barrier_price(S+h,K,B,T,r,sigma)-barrier_price(S-h,K,B,T,r,sigma))/(2*h)

def barrier_vega(S, K, B, T, r, sigma, h=0.001):
    return (barrier_price(S,K,B,T,r,sigma+h)-barrier_price(S,K,B,T,r,sigma-h))/(2*h)

def bs_vega(S, K, T, r, sigma):
    if T <= 0: return 0.0
    d1 = (np.log(S/K)+(r+.5*sigma**2)*T)/(sigma*np.sqrt(T))
    return S*norm.pdf(d1)*np.sqrt(T)


# ── Simulation ────────────────────────────────────────────────────────────────

def run_simulation(S0, K, B, T_days, r, sigma, n_paths, tc_rate, seed=42):
    np.random.seed(seed)
    dt = 1/252
    Z  = np.random.randn(n_paths, T_days)
    S  = np.zeros((n_paths, T_days+1)); S[:,0] = S0
    for t in range(T_days):
        S[:,t+1] = S[:,t]*np.exp((r-.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z[:,t])

    knocked = S.max(axis=1) >= B
    payoffs = np.where(knocked, 0, np.maximum(S[:,-1]-K, 0)) * np.exp(-r*T_days/252)

    # Hedging PnL for four strategies
    pnls = {}
    for rebal, use_vega, label in [
        (1,  False, "Daily Δ"),
        (5,  False, "Weekly Δ"),
        (1,  True,  "Daily Δ+V"),
        (5,  True,  "Weekly Δ+V"),
    ]:
        pnl = np.zeros(n_paths)
        for i in range(n_paths):
            path = S[i]; ko = False; port = barrier_price(S0,K,B,T_days/252,r,sigma)
            dh = 0.0; vn = 0.0
            for t in range(T_days):
                if path[t] >= B: ko=True; break
                T_rem = (T_days-t)/252
                if t % rebal == 0:
                    nd = barrier_delta(path[t],K,B,T_rem,r,sigma)
                    port -= (nd-dh)*path[t] + tc_rate*abs(nd-dh)*path[t]; dh=nd
                    if use_vega:
                        vb = barrier_vega(path[t],K,B,T_rem,r,sigma)
                        vv = bs_vega(path[t],K,T_rem,r,sigma)
                        nv = -vb/vv if abs(vv)>1e-6 else 0
                        pv = bs_call(path[t],K,T_rem,r,sigma)
                        port -= (nv-vn)*pv + tc_rate*abs(nv-vn)*pv; vn=nv
                port += dh*(path[t+1]-path[t])
                if use_vega and vn:
                    Tr2 = max((T_days-t-1)/252,1e-6)
                    port += vn*(bs_call(path[t+1],K,Tr2,r,sigma)-bs_call(path[t],K,max(T_rem,1e-6),r,sigma))
            if not ko:
                port += max(path[-1]-K,0) - dh*path[-1] - tc_rate*abs(dh)*path[-1]
            pnl[i] = port*np.exp(-r*T_days/252)
        pnls[label] = pnl

    return S, payoffs, knocked, pnls

def sensitivity_table(K, B_range, T_ann, r, sigma):
    return pd.DataFrame({
        "Barrier": B_range,
        "Price":   [round(barrier_price(100,K,b,T_ann,r,sigma),4) for b in B_range],
        "Delta":   [round(barrier_delta(100,K,b,T_ann,r,sigma),4) for b in B_range],
        "Vega":    [round(barrier_vega(100,K,b,T_ann,r,sigma),4)  for b in B_range],
    })


# ── Dash app ──────────────────────────────────────────────────────────────────

def kpi(label, value, color=WHITE):
    return dbc.Col(dbc.Card(dbc.CardBody([
        html.P(label, style={"fontSize":"0.68rem","color":GREY,
                              "textTransform":"uppercase","letterSpacing":"0.06em","marginBottom":2}),
        html.H4(value, style={"color":color,"fontWeight":700,"margin":0}),
    ]), style={"background":CARD,"border":f"1px solid {BORDER}","borderRadius":8}),
    xs=6, sm=4, md=2, className="mb-2")

app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY],
           title="Barrier Option Dashboard")

app.layout = dbc.Container(fluid=True,
    style={"backgroundColor":BG,"minHeight":"100vh","padding":"24px"},
    children=[
        dbc.Row([
            dbc.Col(html.H3("⚡ Up-and-Out Call Barrier Option",
                            style={"color":WHITE,"fontWeight":700})),
        ], className="mb-3"),

        # Controls
        dbc.Card(dbc.CardBody(dbc.Row([
            dbc.Col([html.Label("Barrier B", style={"color":GREY,"fontSize":"0.8rem"}),
                     dcc.Slider(id="sl-B", min=105, max=150, step=1, value=110,
                                marks={v:str(v) for v in range(105,155,5)},
                                tooltip={"placement":"bottom"})], md=4),
            dbc.Col([html.Label("Volatility σ", style={"color":GREY,"fontSize":"0.8rem"}),
                     dcc.Slider(id="sl-sig", min=0.10, max=0.50, step=0.01, value=0.20,
                                marks={v:f"{v:.0%}" for v in [.10,.20,.30,.40,.50]},
                                tooltip={"placement":"bottom"})], md=3),
            dbc.Col([html.Label("Paths", style={"color":GREY,"fontSize":"0.8rem"}),
                     dcc.Slider(id="sl-n", min=1000, max=10000, step=1000, value=3000,
                                marks={v:str(v) for v in [1000,3000,5000,10000]},
                                tooltip={"placement":"bottom"})], md=3),
            dbc.Col([html.Label("TC %", style={"color":GREY,"fontSize":"0.8rem"}),
                     dcc.Slider(id="sl-tc", min=0, max=0.005, step=0.0005, value=0.001,
                                marks={0:"0%",0.001:"0.1%",0.003:"0.3%",0.005:"0.5%"},
                                tooltip={"placement":"bottom"})], md=2),
        ])), style={"background":CARD,"border":f"1px solid {BORDER}","marginBottom":16}),

        dbc.Row(id="kpi-row", className="mb-3"),

        dbc.Row([
            dbc.Col(dcc.Graph(id="fig-paths",  config={"displayModeBar":False}), md=8),
            dbc.Col(dcc.Graph(id="fig-conv",   config={"displayModeBar":False}), md=4),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col(dcc.Graph(id="fig-pnl",    config={"displayModeBar":False}), md=7),
            dbc.Col(dcc.Graph(id="fig-std",    config={"displayModeBar":False}), md=5),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col(dcc.Graph(id="fig-payoff", config={"displayModeBar":False}), md=5),
            dbc.Col(dcc.Graph(id="fig-sens",   config={"displayModeBar":False}), md=7),
        ], className="mb-3"),

        dcc.Loading(html.Div(id="loading-state"), type="circle"),
    ])


@app.callback(
    Output("kpi-row",    "children"),
    Output("fig-paths",  "figure"),
    Output("fig-conv",   "figure"),
    Output("fig-pnl",    "figure"),
    Output("fig-std",    "figure"),
    Output("fig-payoff", "figure"),
    Output("fig-sens",   "figure"),
    Output("loading-state","children"),
    Input("sl-B",   "value"),
    Input("sl-sig", "value"),
    Input("sl-n",   "value"),
    Input("sl-tc",  "value"),
)
def update(B, sigma, n_paths, tc_rate):
    S0=100; K=100; T_days=21; r=0.02

    S, payoffs, knocked, pnls = run_simulation(
        S0, K, B, T_days, r, sigma, n_paths, tc_rate)

    T_ann = T_days/252
    p_analytical = barrier_price(S0,K,B,T_ann,r,sigma)
    p_vanilla    = bs_call(S0,K,T_ann,r,sigma)
    mc_price     = payoffs.mean()
    ko_rate      = knocked.mean()
    discount     = 1 - p_analytical/p_vanilla if p_vanilla > 0 else 0
    delta        = barrier_delta(S0,K,B,T_ann,r,sigma)
    vega         = barrier_vega(S0,K,B,T_ann,r,sigma)

    colors_strat = {"Daily Δ":BLUE,"Weekly Δ":GREEN,"Daily Δ+V":AMBER,"Weekly Δ+V":PURPLE}

    # KPIs
    kpis = dbc.Row([
        kpi("Barrier Price",  f"${p_analytical:.4f}", BLUE),
        kpi("Vanilla Price",  f"${p_vanilla:.4f}",    GREY),
        kpi("Barrier Discount",f"{discount:.1%}",     AMBER),
        kpi("MC Price",       f"${mc_price:.4f}",     GREEN),
        kpi("Knock-out Rate", f"{ko_rate:.1%}",       RED if ko_rate>0.1 else AMBER),
        kpi("Delta",          f"{delta:.4f}",         WHITE),
        kpi("Vega",           f"{vega:.4f}",          WHITE),
    ])

    # ── Paths ──────────────────────────────────────────────────────────────
    fig_paths = go.Figure(layout={**BASE,"title":{"text":f"Monte Carlo Paths (200/{n_paths:,})"}})
    show = min(200, n_paths)
    for i in range(show):
        c = RED if knocked[i] else f"{BLUE}55"
        fig_paths.add_trace(go.Scatter(
            x=list(range(T_days+1)), y=S[i], mode="lines",
            line=dict(width=0.4, color=c), showlegend=False))
    fig_paths.add_hline(y=B, line_color=AMBER, line_dash="dash", line_width=2,
                         annotation_text=f"Barrier B={B}", annotation_font_color=AMBER)
    fig_paths.add_hline(y=K, line_color=GREY, line_dash="dot", line_width=1,
                         annotation_text=f"Strike K={K}")
    fig_paths.update_xaxes(title="Day"); fig_paths.update_yaxes(title="Price ($)")
    # Legend entries
    fig_paths.add_trace(go.Scatter(x=[None],y=[None],mode="lines",
        line=dict(color=RED,width=1.5),name=f"Knocked out ({ko_rate:.1%})"))
    fig_paths.add_trace(go.Scatter(x=[None],y=[None],mode="lines",
        line=dict(color=BLUE,width=1.5),name=f"Survived ({1-ko_rate:.1%})"))

    # ── MC convergence ─────────────────────────────────────────────────────
    n_range  = np.arange(50, n_paths+1, max(50, n_paths//100))
    running  = [payoffs[:n].mean() for n in n_range]
    fig_conv = go.Figure(layout={**BASE,"title":{"text":"MC Price Convergence"}})
    fig_conv.add_trace(go.Scatter(x=n_range, y=running, line=dict(color=BLUE,width=2), name="MC"))
    fig_conv.add_hline(y=p_analytical, line_color=AMBER, line_dash="dash",
                        annotation_text=f"Analytical ${p_analytical:.4f}")
    fig_conv.update_xaxes(title="Paths"); fig_conv.update_yaxes(title="Price ($)")

    # ── PnL distributions ──────────────────────────────────────────────────
    fig_pnl = go.Figure(layout={**BASE,"title":{"text":"Hedging PnL Distribution by Strategy"}})
    for name, pnl in pnls.items():
        fig_pnl.add_trace(go.Histogram(x=pnl, nbinsx=50, name=name, opacity=0.6,
                                        marker_color=colors_strat[name]))
    fig_pnl.add_vline(x=0, line_color=WHITE, line_dash="dot", line_width=1)
    fig_pnl.update_layout(barmode="overlay")
    fig_pnl.update_xaxes(title="Terminal PnL ($)")

    # ── Std bar chart ──────────────────────────────────────────────────────
    stds  = {k: v.std() for k,v in pnls.items()}
    fig_std = go.Figure(layout={**BASE,"title":{"text":"PnL Volatility — lower is better"}})
    fig_std.add_trace(go.Bar(
        x=list(stds.keys()), y=list(stds.values()),
        marker_color=[colors_strat[k] for k in stds],
        text=[f"{v:.4f}" for v in stds.values()],
        textposition="outside", textfont_color=WHITE,
    ))
    fig_std.update_yaxes(title="Std(PnL) ($)")

    # ── Payoff distribution ─────────────────────────────────────────────────
    itm = payoffs[payoffs>0]
    fig_payoff = go.Figure(layout={**BASE,"title":{"text":"Option Payoff Distribution"}})
    fig_payoff.add_trace(go.Histogram(x=payoffs[~knocked], nbinsx=40,
                                       name="Survived paths", marker_color=GREEN, opacity=0.7))
    zero_count = knocked.sum() + (payoffs == 0).sum()
    fig_payoff.add_annotation(x=0, y=0, text=f"Zero payoff: {zero_count}",
                               showarrow=False, yshift=20, font=dict(color=AMBER))
    fig_payoff.update_xaxes(title="Payoff ($)")

    # ── Sensitivity ─────────────────────────────────────────────────────────
    B_range = list(range(103, 145, 2))
    prices_b  = [barrier_price(S0,K,b,T_ann,r,sigma) for b in B_range]
    deltas_b  = [barrier_delta(S0,K,b,T_ann,r,sigma) for b in B_range]
    vegas_b   = [barrier_vega(S0,K,b,T_ann,r,sigma)  for b in B_range]

    fig_sens = go.Figure(layout={**BASE,"title":{"text":"Greeks vs Barrier Level"}})
    fig_sens.add_trace(go.Scatter(x=B_range, y=prices_b,  name="Price ($)",
                                   line=dict(color=BLUE,width=2)))
    fig_sens.add_trace(go.Scatter(x=B_range, y=deltas_b,  name="Delta",
                                   line=dict(color=GREEN,width=2), yaxis="y2"))
    fig_sens.add_vline(x=B, line_color=AMBER, line_dash="dash",
                        annotation_text=f"Current B={B}", annotation_font_color=AMBER)
    fig_sens.update_layout(
        yaxis2=dict(overlaying="y", side="right", title="Delta",
                    gridcolor=BORDER, color=GREEN),
        yaxis=dict(title="Price ($)"),
    )
    fig_sens.update_xaxes(title="Barrier Level B")

    return kpis, fig_paths, fig_conv, fig_pnl, fig_std, fig_payoff, fig_sens, ""


if __name__ == "__main__":
    print("Barrier Option Dashboard → http://127.0.0.1:8053")
    app.run(host="0.0.0.0", port=int(os.environ.get("DASH_PORT", 8053)), debug=False)
