"""
Barrier Option Interactive Dashboard
=====================================
Interactive browser dashboard — adjust B, σ, paths and TC in real time.

Run:
    python3 dashboard.py
    → http://127.0.0.1:8053
"""

from __future__ import annotations
import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
from scipy.stats import norm

# ── Palette ───────────────────────────────────────────────────────────────────
BG    = "#0d1117"; CARD  = "#161b22"; BORDER = "#30363d"
GREEN = "#3fb950"; RED   = "#f85149"; BLUE   = "#58a6ff"
AMBER = "#d29922"; GREY  = "#8b949e"; WHITE  = "#e6edf3"
PURPLE= "#bc8cff"; FONT  = "Inter, sans-serif"

BASE = dict(
    paper_bgcolor=BG, plot_bgcolor=CARD,
    font=dict(family=FONT, color=WHITE, size=11),
    margin=dict(l=55, r=20, t=40, b=40),
    xaxis=dict(gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER),
    yaxis=dict(gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER),
)

STRAT_COLORS = {"Daily Δ": BLUE, "Weekly Δ": GREEN, "Daily Δ+V": AMBER, "Weekly Δ+V": PURPLE}


def hex_rgba(h: str, a: float = 0.15) -> str:
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


# ── Pricing (scalar) ──────────────────────────────────────────────────────────

def bs_call(S, K, T, r, sigma):
    if T <= 0: return float(max(S - K, 0))
    d1 = (np.log(S / K) + (r + .5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))

def bs_vega(S, K, T, r, sigma):
    if T <= 0: return 0.0
    d1 = (np.log(S/K) + (r + .5*sigma**2)*T) / (sigma*np.sqrt(T))
    return float(S * norm.pdf(d1) * np.sqrt(T))

def barrier_price(S, K, B, T, r, sigma):
    if S >= B or T <= 0: return 0.0
    sq  = sigma * np.sqrt(T)
    lam = r / sigma**2 + 0.5
    d1  = (np.log(S/K) + (r + .5*sigma**2)*T) / sq
    d2  = d1 - sq
    vanilla = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    y1  = np.log(B**2 / (S*K)) / sq + lam * sq
    y2  = np.log(B / S)        / sq + lam * sq
    ref = (S * (B/S)**(2*lam) * (norm.cdf(y1) - norm.cdf(y2))
           - K * np.exp(-r*T) * (B/S)**(2*lam-2)
           * (norm.cdf(y1-sq) - norm.cdf(y2-sq)))
    return float(max(vanilla - ref, 0))

def b_delta(S, K, B, T, r, sigma, h=0.01):
    return (barrier_price(S+h,K,B,T,r,sigma) - barrier_price(S-h,K,B,T,r,sigma)) / (2*h)

def b_vega_scalar(S, K, B, T, r, sigma, h=0.001):
    return (barrier_price(S,K,B,T,r,sigma+h) - barrier_price(S,K,B,T,r,sigma-h)) / (2*h)


# ── Vectorised pricing (operates on 1-D numpy arrays) ─────────────────────────

def _barrier_price_vec(S, K, B, T, r, sigma):
    """Vectorised up-and-out call price.  S: 1-D array.  Returns array."""
    out = np.zeros(len(S))
    if T <= 0:
        return out
    sq  = sigma * np.sqrt(T)
    lam = r / sigma**2 + 0.5
    active = S < B
    Sa = S[active]
    if Sa.size == 0:
        return out
    d1 = (np.log(Sa/K) + (r + .5*sigma**2)*T) / sq
    d2 = d1 - sq
    vanilla = Sa * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    ratio   = B / Sa
    y1  = np.log(B**2 / (Sa*K)) / sq + lam * sq
    y2  = np.log(ratio)          / sq + lam * sq
    ref = (Sa * ratio**(2*lam)   * (norm.cdf(y1) - norm.cdf(y2))
           - K * np.exp(-r*T) * ratio**(2*lam-2)
           * (norm.cdf(y1-sq)  - norm.cdf(y2-sq)))
    out[active] = np.maximum(vanilla - ref, 0)
    return out

def _b_delta_vec(S, K, B, T, r, sigma, h=0.01):
    return (_barrier_price_vec(S+h, K, B, T, r, sigma) -
            _barrier_price_vec(S-h, K, B, T, r, sigma)) / (2*h)

def _b_vega_vec(S, K, B, T, r, sigma, h=0.001):
    return (_barrier_price_vec(S, K, B, T, r, sigma+h) -
            _barrier_price_vec(S, K, B, T, r, sigma-h)) / (2*h)

def _bs_call_vec(S, K, T, r, sigma):
    if T <= 0:
        return np.maximum(S - K, 0)
    d1 = (np.log(S/K) + (r + .5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def _bs_vega_vec(S, K, T, r, sigma):
    if T <= 0:
        return np.zeros_like(S)
    d1 = (np.log(S/K) + (r + .5*sigma**2)*T) / (sigma*np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)


# ── Vectorised simulation ─────────────────────────────────────────────────────

def simulate(S0, K, B, T_days, r, sigma, n_paths, tc, seed=42):
    """Fully vectorised: runs ~100× faster than path-by-path loop."""
    np.random.seed(seed)
    dt = 1 / 252

    # Generate all paths at once
    Z = np.random.randn(n_paths, T_days)
    S = np.empty((n_paths, T_days + 1))
    S[:, 0] = S0
    for t in range(T_days):
        S[:, t+1] = S[:, t] * np.exp((r - .5*sigma**2)*dt + sigma*np.sqrt(dt)*Z[:, t])

    # Knock-out mask (cumulative)
    ko_step = S[:, 1:] >= B                        # (n_paths, T_days)
    knocked = ko_step.any(axis=1)                  # (n_paths,)
    ko_time = np.where(knocked,
                       ko_step.argmax(axis=1) + 1,  # first step >= B
                       T_days)                      # never knocked

    payoffs = np.where(knocked, 0.0,
                       np.maximum(S[:, -1] - K, 0.0)) * np.exp(-r * T_days/252)

    pnls = {}
    for rebal, use_v, label in [(1, False, "Daily Δ"),  (5, False, "Weekly Δ"),
                                 (1, True,  "Daily Δ+V"), (5, True,  "Weekly Δ+V")]:
        port  = np.full(n_paths, barrier_price(S0, K, B, T_days/252, r, sigma))
        delta = np.zeros(n_paths)
        vn    = np.zeros(n_paths)          # vanilla units for vega hedge
        alive = np.ones(n_paths, dtype=bool)

        for t in range(T_days):
            Tr    = max((T_days - t) / 252, 1e-9)
            St    = S[:, t]
            St1   = S[:, t+1]

            # Mark newly knocked-out paths
            alive &= (St < B)
            if not alive.any():
                break

            if t % rebal == 0:
                new_d = np.zeros(n_paths)
                new_d[alive] = _b_delta_vec(St[alive], K, B, Tr, r, sigma)
                dc = new_d - delta
                port[alive] -= dc[alive] * St[alive] + tc * np.abs(dc[alive]) * St[alive]
                delta = new_d

                if use_v:
                    vb_arr = _b_vega_vec(St[alive], K, B, Tr, r, sigma)
                    vv_arr = _bs_vega_vec(St[alive], K, Tr, r, sigma)
                    safe   = np.abs(vv_arr) > 1e-6
                    new_vn = np.zeros(n_paths)
                    idx    = np.where(alive)[0][safe]
                    raw_vn = np.clip(-vb_arr[safe] / vv_arr[safe], -3.0, 3.0)
                    new_vn[idx] = raw_vn
                    dv = new_vn - vn
                    pv = _bs_call_vec(St, K, Tr, r, sigma)
                    port[alive] -= dv[alive] * pv[alive] + tc * np.abs(dv[alive]) * pv[alive]
                    vn = new_vn

            # Portfolio P&L from delta hedge
            port[alive] += delta[alive] * (St1[alive] - St[alive])

            # Portfolio P&L from vega hedge
            if use_v and alive.any():
                Tr2  = max((T_days - t - 1) / 252, 1e-9)
                pv1  = _bs_call_vec(St1, K, Tr2, r, sigma)
                pv0  = _bs_call_vec(St,  K, Tr,  r, sigma)
                port[alive] += vn[alive] * (pv1[alive] - pv0[alive])

        # Final settlement
        alive_final = ~knocked
        port[alive_final] += np.maximum(S[alive_final, -1] - K, 0.0)
        port[alive_final] -= (delta[alive_final] * S[alive_final, -1]
                               + tc * np.abs(delta[alive_final]) * S[alive_final, -1])
        if use_v:
            port[alive_final] -= vn[alive_final] * _bs_call_vec(
                S[alive_final, -1], K, 1e-9, r, sigma)

        pnls[label] = port * np.exp(-r * T_days / 252)

    return S, payoffs, knocked, pnls


# ── App ───────────────────────────────────────────────────────────────────────

app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY],
           title="Barrier Option Dashboard", suppress_callback_exceptions=True)

def kpi_card(label, value, color=WHITE):
    return dbc.Col(
        dbc.Card(dbc.CardBody([
            html.P(label, style={"fontSize": "0.68rem", "color": GREY,
                                  "textTransform": "uppercase", "letterSpacing": "0.06em",
                                  "marginBottom": 2}),
            html.H4(value, style={"color": color, "fontWeight": 700, "margin": 0}),
        ]), style={"background": CARD, "border": f"1px solid {BORDER}", "borderRadius": 8}),
        xs=6, sm=4, md=2, className="mb-2",
    )

app.layout = dbc.Container(fluid=True,
    style={"backgroundColor": BG, "minHeight": "100vh", "padding": "24px"},
    children=[
        dbc.Row([
            dbc.Col(html.H3("⚡ Up-and-Out Call Barrier Option",
                            style={"color": WHITE, "fontWeight": 700})),
            dbc.Col(html.Small("S₀=100  K=100  T=21d  r=2%",
                               style={"color": GREY}),
                    width="auto", className="d-flex align-items-center"),
        ], className="mb-3"),

        # Controls
        dbc.Card(dbc.CardBody(dbc.Row([
            dbc.Col([
                html.Label("Barrier B", style={"color": GREY, "fontSize": "0.8rem"}),
                dcc.Slider(id="sl-B", min=104, max=145, step=1, value=110,
                           marks={v: str(v) for v in [104, 110, 115, 120, 130, 145]},
                           tooltip={"placement": "bottom", "always_visible": False}),
            ], md=3),
            dbc.Col([
                html.Label("Volatility σ", style={"color": GREY, "fontSize": "0.8rem"}),
                dcc.Slider(id="sl-sig", min=0.10, max=0.45, step=0.01, value=0.20,
                           marks={0.10: "10%", 0.20: "20%", 0.30: "30%", 0.45: "45%"},
                           tooltip={"placement": "bottom", "always_visible": False}),
            ], md=3),
            dbc.Col([
                html.Label("Paths", style={"color": GREY, "fontSize": "0.8rem"}),
                dcc.Slider(id="sl-n", min=500, max=5000, step=500, value=1000,
                           marks={500: "500", 1000: "1k", 2000: "2k", 5000: "5k"},
                           tooltip={"placement": "bottom", "always_visible": False}),
            ], md=3),
            dbc.Col([
                html.Label("Transaction Cost", style={"color": GREY, "fontSize": "0.8rem"}),
                dcc.Slider(id="sl-tc", min=0, max=0.005, step=0.0005, value=0.001,
                           marks={0: "0%", 0.001: "0.1%", 0.003: "0.3%", 0.005: "0.5%"},
                           tooltip={"placement": "bottom", "always_visible": False}),
            ], md=3),
        ])), style={"background": CARD, "border": f"1px solid {BORDER}", "marginBottom": 16}),

        dbc.Row(id="kpi-row", className="mb-3"),

        dcc.Loading(type="circle", color=BLUE, children=[
            dbc.Row([
                dbc.Col(dcc.Graph(id="fig-paths", config={"displayModeBar": False}), md=8),
                dbc.Col(dcc.Graph(id="fig-conv",  config={"displayModeBar": False}), md=4),
            ], className="mb-3"),

            dbc.Row([
                dbc.Col(dcc.Graph(id="fig-pnl",   config={"displayModeBar": False}), md=7),
                dbc.Col(dcc.Graph(id="fig-vol",   config={"displayModeBar": False}), md=5),
            ], className="mb-3"),

            dbc.Row([
                dbc.Col(dcc.Graph(id="fig-pay",   config={"displayModeBar": False}), md=5),
                dbc.Col(dcc.Graph(id="fig-sens",  config={"displayModeBar": False}), md=7),
            ], className="mb-3"),
        ]),
    ])


@app.callback(
    Output("kpi-row",  "children"),
    Output("fig-paths","figure"),
    Output("fig-conv", "figure"),
    Output("fig-pnl",  "figure"),
    Output("fig-vol",  "figure"),
    Output("fig-pay",  "figure"),
    Output("fig-sens", "figure"),
    Input("sl-B",   "value"),
    Input("sl-sig", "value"),
    Input("sl-n",   "value"),
    Input("sl-tc",  "value"),
)
def update(B, sigma, n_paths, tc):
    S0 = 100; K = 100; T_days = 21; r = 0.02
    T_ann = T_days / 252

    S, payoffs, knocked, pnls = simulate(S0, K, B, T_days, r, sigma, n_paths, tc)

    p_bar = barrier_price(S0, K, B, T_ann, r, sigma)
    p_van = bs_call(S0, K, T_ann, r, sigma)
    mc_px = float(payoffs.mean())
    ko_rt = float(knocked.mean())
    disc  = 1 - p_bar / p_van if p_van > 0 else 0
    dlt   = b_delta(S0, K, B, T_ann, r, sigma)
    vga   = b_vega_scalar(S0, K, B, T_ann, r, sigma)

    # ── KPIs ──────────────────────────────────────────────────────────────────
    kpis = dbc.Row([
        kpi_card("Barrier Price",  f"${p_bar:.4f}", BLUE),
        kpi_card("Vanilla Price",  f"${p_van:.4f}", GREY),
        kpi_card("Discount",       f"{disc:.1%}",   AMBER),
        kpi_card("MC Price",       f"${mc_px:.4f}", GREEN),
        kpi_card("Knock-out Rate", f"{ko_rt:.1%}",  RED if ko_rt > 0.15 else AMBER),
        kpi_card("Delta",          f"{dlt:.4f}",    WHITE),
        kpi_card("Vega",           f"{vga:.4f}",    WHITE),
    ])

    # ── Paths ──────────────────────────────────────────────────────────────────
    show = min(200, n_paths)
    fig_paths = go.Figure(layout={**BASE,
        "title": {"text": f"Monte Carlo Paths — {ko_rt:.1%} knocked out (red)"}})
    for i in range(show):
        col = RED if knocked[i] else hex_rgba(BLUE, 0.20)
        fig_paths.add_trace(go.Scatter(
            x=list(range(T_days + 1)), y=list(S[i]),
            mode="lines", line=dict(width=0.6, color=col), showlegend=False,
        ))
    fig_paths.add_hline(y=B, line_color=AMBER, line_width=2, line_dash="dash",
                         annotation_text=f"Barrier B={B}", annotation_font_color=AMBER)
    fig_paths.add_hline(y=K, line_color=GREY,  line_width=1, line_dash="dot",
                         annotation_text=f"Strike K={K}")
    fig_paths.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
        line=dict(color=RED,  width=1.5), name=f"Knocked ({ko_rt:.1%})"))
    fig_paths.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
        line=dict(color=BLUE, width=1.5), name=f"Survived ({1-ko_rt:.1%})"))
    fig_paths.update_xaxes(title="Day")
    fig_paths.update_yaxes(title="Price ($)")

    # ── MC convergence ─────────────────────────────────────────────────────────
    step    = max(1, n_paths // 80)
    n_range = np.arange(step, n_paths + 1, step)
    running = [payoffs[:n].mean() for n in n_range]
    ci_up   = [payoffs[:n].mean() + 1.96*payoffs[:n].std()/np.sqrt(n) for n in n_range]
    ci_lo   = [payoffs[:n].mean() - 1.96*payoffs[:n].std()/np.sqrt(n) for n in n_range]

    fig_conv = go.Figure(layout={**BASE, "title": {"text": "MC Price Convergence"}})
    fig_conv.add_trace(go.Scatter(
        x=list(n_range) + list(n_range[::-1]),
        y=ci_up + ci_lo[::-1],
        fill="toself", fillcolor=hex_rgba(BLUE, 0.10),
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
    ))
    fig_conv.add_trace(go.Scatter(x=n_range, y=running,
        line=dict(color=BLUE, width=2), name="MC price"))
    fig_conv.add_hline(y=p_bar, line_color=AMBER, line_dash="dash", line_width=1.5,
                        annotation_text=f"Analytical ${p_bar:.4f}",
                        annotation_font_color=AMBER)
    fig_conv.update_xaxes(title="# Paths")
    fig_conv.update_yaxes(title="Price ($)")

    # ── PnL distributions ──────────────────────────────────────────────────────
    fig_pnl = go.Figure(layout={**BASE, "title": {"text": "Hedging PnL — 4 Strategies"}})
    for name, pnl in pnls.items():
        fig_pnl.add_trace(go.Histogram(
            x=pnl, nbinsx=50, name=f"{name}  σ={pnl.std():.4f}",
            opacity=0.60, marker_color=STRAT_COLORS[name],
        ))
    fig_pnl.add_vline(x=0, line_color=WHITE, line_dash="dot", line_width=1)
    fig_pnl.update_layout(barmode="overlay")
    fig_pnl.update_xaxes(title="Terminal PnL ($)")
    fig_pnl.update_yaxes(title="Count")

    # ── Volatility comparison ───────────────────────────────────────────────────
    names = list(pnls.keys())
    stds  = [pnls[n].std() for n in names]
    cols  = [STRAT_COLORS[n] for n in names]
    fig_vol = go.Figure(layout={**BASE, "title": {"text": "PnL Volatility by Strategy"}})
    fig_vol.add_trace(go.Bar(
        x=names, y=stds, marker_color=cols, opacity=0.85,
        text=[f"{v:.5f}" for v in stds],
        textposition="outside", textfont=dict(color=WHITE, size=11),
    ))
    fig_vol.update_yaxes(title="Std(PnL) — lower is better")

    # ── Payoff histogram ────────────────────────────────────────────────────────
    survived_payoffs = payoffs[~knocked]
    fig_pay = go.Figure(layout={**BASE,
        "title": {"text": f"Option Payoff — {(~knocked).sum()} survived paths"}})
    if len(survived_payoffs) > 0:
        fig_pay.add_trace(go.Histogram(
            x=survived_payoffs, nbinsx=40,
            marker_color=GREEN, opacity=0.75, name="Payoffs",
        ))
    fig_pay.add_vline(x=0, line_color=GREY, line_dash="dot", line_width=1)
    fig_pay.update_xaxes(title="Payoff ($)")
    fig_pay.update_yaxes(title="Count")

    # ── Greeks vs barrier ───────────────────────────────────────────────────────
    B_range  = np.arange(103, 148, 2)
    prices_s = [barrier_price(S0, K, int(b), T_ann, r, sigma) for b in B_range]
    deltas_s = [b_delta(S0, K, int(b), T_ann, r, sigma) for b in B_range]

    fig_sens = go.Figure(layout={**BASE,
        "title": {"text": "Price & Delta vs Barrier Level"},
        "yaxis2": dict(overlaying="y", side="right",
                       title="Delta", gridcolor=BORDER,
                       color=GREEN, showgrid=False),
    })
    fig_sens.add_trace(go.Scatter(
        x=B_range, y=prices_s, name="Price ($)",
        line=dict(color=BLUE, width=2.5),
    ))
    fig_sens.add_trace(go.Scatter(
        x=B_range, y=deltas_s, name="Delta",
        line=dict(color=GREEN, width=2.5), yaxis="y2",
    ))
    fig_sens.add_vline(x=B, line_color=AMBER, line_dash="dash", line_width=1.5,
                        annotation_text=f"B={B}", annotation_font_color=AMBER)
    fig_sens.update_xaxes(title="Barrier B")
    fig_sens.update_yaxes(title="Price ($)")

    return kpis, fig_paths, fig_conv, fig_pnl, fig_vol, fig_pay, fig_sens


if __name__ == "__main__":
    print("=" * 55)
    print("  Barrier Option Dashboard → http://127.0.0.1:8053")
    print("=" * 55)
    app.run(host="0.0.0.0", port=int(os.environ.get("DASH_PORT", 8053)), debug=False)
