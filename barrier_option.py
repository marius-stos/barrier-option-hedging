"""
Pricing and Hedging an Up-and-Out Call Barrier Option
======================================================
Monte Carlo simulation under Black-Scholes dynamics.
Compares daily vs weekly delta-vega hedging with transaction costs.

Parameters (matching IESEG project):
  S0=100, K=100, B=130 (barrier), T=21 days, r=0.02, sigma=0.20
  Transaction cost: 0.1% per rebalance
  Paths: 10,000

Run:
    python3 barrier_option.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm
from scipy.optimize import brentq
import warnings
warnings.filterwarnings("ignore")

# ── Parameters ─────────────────────────────────────────────────────────────────

S0         = 100.0    # initial stock price
K          = 100.0    # strike
B          = 110.0    # up-and-out barrier (10% OTM — gives ~10% knock-out probability)
T_DAYS     = 21       # total horizon in trading days
r          = 0.02     # annual risk-free rate
SIGMA      = 0.20     # annual volatility (Black-Scholes)
N_PATHS    = 10_000   # Monte Carlo paths
TC_RATE    = 0.001    # 0.1% transaction cost per delta/vega rebalance
SEED       = 42

dt_daily  = 1 / 252
dt_weekly = 5 / 252   # rebalance every 5 trading days

np.random.seed(SEED)


# ── Black-Scholes closed-form (vanilla call, for reference) ────────────────────

def bs_call(S, K, T, r, sigma):
    if T <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_delta(S, K, T, r, sigma):
    if T <= 0:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

def bs_vega(S, K, T, r, sigma):
    if T <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

def implied_vol(price, S, K, T, r, tol=1e-6):
    """Back out implied vol via Brent's method."""
    try:
        f = lambda s: bs_call(S, K, T, r, s) - price
        return brentq(f, 1e-6, 5.0, xtol=tol)
    except Exception:
        return SIGMA


# ── Analytical barrier option price (reflection principle) ────────────────────

def barrier_call_price(S, K, B, T, r, sigma):
    """
    Closed-form price of an up-and-out call under BS.
    Uses the reflection principle (Rubinstein & Reiner 1991).
    """
    if S >= B:
        return 0.0  # already knocked out
    if T <= 0:
        return max(S - K, 0) if S < B else 0.0

    mu    = (r - 0.5 * sigma**2)
    lam   = (r + 0.5 * sigma**2 - mu) / (sigma**2)   # simplifies to r/sigma^2 + 0.5
    x1    = np.log(S / K)  / (sigma * np.sqrt(T)) + (1 + lam) * sigma * np.sqrt(T)
    x2    = np.log(S / B)  / (sigma * np.sqrt(T)) + (1 + lam) * sigma * np.sqrt(T)
    y1    = np.log(B**2 / (S * K)) / (sigma * np.sqrt(T)) + (1 + lam) * sigma * np.sqrt(T)
    y2    = np.log(B / S)          / (sigma * np.sqrt(T)) + (1 + lam) * sigma * np.sqrt(T)

    vanilla = bs_call(S, K, T, r, sigma)
    # reflected term
    ref = (S * (B / S) ** (2 * lam) *
           (norm.cdf(y1) - norm.cdf(y2)) -
           K * np.exp(-r * T) * ((B / S) ** (2 * lam - 2)) *
           (norm.cdf(y1 - sigma * np.sqrt(T)) -
            norm.cdf(y2 - sigma * np.sqrt(T))))

    price = vanilla - ref
    return max(price, 0.0)


def barrier_delta(S, K, B, T, r, sigma, dS=0.01):
    """Numerical delta of barrier option."""
    return (barrier_call_price(S + dS, K, B, T, r, sigma) -
            barrier_call_price(S - dS, K, B, T, r, sigma)) / (2 * dS)

def barrier_vega(S, K, B, T, r, sigma, ds=0.001):
    """Numerical vega of barrier option."""
    return (barrier_call_price(S, K, B, T, r, sigma + ds) -
            barrier_call_price(S, K, B, T, r, sigma - ds)) / (2 * ds)


# ── Monte Carlo simulation ─────────────────────────────────────────────────────

def simulate_paths(n_paths: int = N_PATHS, dt: float = dt_daily) -> np.ndarray:
    """
    Simulate GBM paths with daily steps.
    Returns array of shape (n_paths, T_DAYS+1).
    We always simulate at daily granularity for payoff computation.
    """
    n_steps = T_DAYS
    Z       = np.random.standard_normal((n_paths, n_steps))
    S       = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0

    for t in range(n_steps):
        S[:, t + 1] = S[:, t] * np.exp(
            (r - 0.5 * SIGMA**2) * dt_daily + SIGMA * np.sqrt(dt_daily) * Z[:, t]
        )
    return S


def compute_payoffs(paths: np.ndarray) -> np.ndarray:
    """
    Up-and-out call payoff.
    Returns 0 if max(path) >= B, else max(S_T - K, 0).
    """
    max_prices = paths.max(axis=1)
    final      = paths[:, -1]
    knocked    = max_prices >= B
    payoffs    = np.where(knocked, 0.0, np.maximum(final - K, 0.0))
    return np.exp(-r * T_DAYS / 252) * payoffs


# ── Hedging simulation ─────────────────────────────────────────────────────────

def hedge_pnl(paths: np.ndarray, rebal_freq: int = 1,
              use_vega_hedge: bool = False) -> np.ndarray:
    """
    Simulate delta (or delta-vega) hedging PnL for each path.

    rebal_freq = 1  → daily rebalance
    rebal_freq = 5  → weekly rebalance
    use_vega_hedge  → add a vega hedge via a vanilla option

    Returns array of terminal PnL per path (after transaction costs).
    """
    n_paths, n_steps_plus1 = paths.shape
    n_steps = n_steps_plus1 - 1
    T_annual = T_DAYS / 252

    # Option price at t=0
    V0      = barrier_call_price(S0, K, B, T_annual, r, SIGMA)
    pnl     = np.zeros(n_paths)

    for i in range(n_paths):
        path      = paths[i]
        knocked   = False
        portfolio = V0          # start with option premium received
        delta_held = 0.0
        vega_notional = 0.0     # notional in vanilla call for vega hedge

        for t in range(n_steps):
            # Check knockout
            if path[t] >= B:
                knocked = True
                break

            T_rem = (n_steps - t) / 252

            if T_rem <= 0:
                break

            # Rebalance on schedule
            if t % rebal_freq == 0:
                S_t = path[t]

                new_delta = barrier_delta(S_t, K, B, T_rem, r, SIGMA)
                delta_chg = new_delta - delta_held
                tc        = TC_RATE * abs(delta_chg) * S_t

                portfolio  -= delta_chg * S_t + tc
                delta_held  = new_delta

                if use_vega_hedge:
                    vega_bar  = barrier_vega(S_t, K, B, T_rem, r, SIGMA)
                    vega_van  = bs_vega(S_t, K, T_rem, r, SIGMA)
                    if abs(vega_van) > 1e-6:
                        target_vega_n = -vega_bar / vega_van
                        vega_chg      = target_vega_n - vega_notional
                        van_price     = bs_call(S_t, K, T_rem, r, SIGMA)
                        tc_vega       = TC_RATE * abs(vega_chg) * van_price
                        portfolio    -= vega_chg * van_price + tc_vega
                        vega_notional = target_vega_n

            # P&L from delta hedge over dt
            dS = path[t + 1] - path[t]
            portfolio += delta_held * dS

            if use_vega_hedge and vega_notional != 0:
                S_next = path[t + 1]
                T_rem_next = max((n_steps - t - 1) / 252, 1e-6)
                van_now  = bs_call(path[t],  K, max(T_rem, 1e-6), r, SIGMA)
                van_next = bs_call(S_next, K, T_rem_next,          r, SIGMA)
                portfolio += vega_notional * (van_next - van_now)

        # Terminal settlement
        if not knocked:
            payoff    = max(path[-1] - K, 0.0)
            portfolio += payoff

        # Unwind any remaining delta position
        if not knocked:
            tc_unwind  = TC_RATE * abs(delta_held) * path[-1]
            portfolio -= delta_held * path[-1] + tc_unwind

        # Discount
        pnl[i] = portfolio * np.exp(-r * T_annual)

    return pnl


# ── Run everything ─────────────────────────────────────────────────────────────

def run():
    print("=" * 65)
    print("  Up-and-Out Call Barrier Option — Pricing & Hedging")
    print(f"  S0={S0}, K={K}, B={B}, T={T_DAYS}d, r={r:.0%}, σ={SIGMA:.0%}")
    print(f"  TC={TC_RATE:.1%} per trade  |  {N_PATHS:,} Monte Carlo paths")
    print("=" * 65)

    # ── 1. Analytical price ───────────────────────────────────────────────
    T_annual   = T_DAYS / 252
    price_analytical = barrier_call_price(S0, K, B, T_annual, r, SIGMA)
    price_vanilla    = bs_call(S0, K, T_annual, r, SIGMA)
    print(f"\n  Analytical barrier call price : ${price_analytical:.4f}")
    print(f"  Vanilla call price (reference): ${price_vanilla:.4f}")
    print(f"  Barrier discount              : {(1 - price_analytical/price_vanilla):.1%}")

    # ── 2. Monte Carlo pricing ────────────────────────────────────────────
    print(f"\n  Simulating {N_PATHS:,} paths…")
    paths    = simulate_paths(N_PATHS)
    payoffs  = compute_payoffs(paths)
    mc_price = payoffs.mean()
    mc_se    = payoffs.std() / np.sqrt(N_PATHS)

    ko_rate  = (paths.max(axis=1) >= B).mean()
    print(f"  MC price  : ${mc_price:.4f}  (SE ±{mc_se:.4f})")
    print(f"  95% CI    : [${mc_price - 1.96*mc_se:.4f}, ${mc_price + 1.96*mc_se:.4f}]")
    print(f"  Knock-out rate: {ko_rate:.1%} of paths")

    # ── 3. Hedging comparison ─────────────────────────────────────────────
    print("\n  Running hedging simulations…")

    pnl_daily_delta  = hedge_pnl(paths, rebal_freq=1,  use_vega_hedge=False)
    pnl_weekly_delta = hedge_pnl(paths, rebal_freq=5,  use_vega_hedge=False)
    pnl_daily_dv     = hedge_pnl(paths, rebal_freq=1,  use_vega_hedge=True)
    pnl_weekly_dv    = hedge_pnl(paths, rebal_freq=5,  use_vega_hedge=True)

    strategies = {
        "Daily Δ only":    pnl_daily_delta,
        "Weekly Δ only":   pnl_weekly_delta,
        "Daily Δ+Vega":    pnl_daily_dv,
        "Weekly Δ+Vega":   pnl_weekly_dv,
    }

    print(f"\n  {'Strategy':<20} {'Mean PnL':>10} {'Std PnL':>10} "
          f"{'Sharpe':>8} {'VaR 95%':>10}")
    print(f"  {'-'*65}")
    results = {}
    for name, pnl in strategies.items():
        mean  = pnl.mean()
        std   = pnl.std()
        sharpe = mean / std if std > 0 else 0
        var95  = np.percentile(pnl, 5)
        results[name] = {"mean": mean, "std": std, "sharpe": sharpe, "var95": var95, "pnl": pnl}
        print(f"  {name:<20} {mean:>+10.4f} {std:>10.4f} {sharpe:>8.3f} {var95:>+10.4f}")

    # Reduction in vol: weekly vs daily
    vol_reduction = (results["Weekly Δ only"]["std"] - results["Daily Δ only"]["std"]) \
                     / results["Weekly Δ only"]["std"]
    vega_reduction = (results["Daily Δ only"]["std"] - results["Daily Δ+Vega"]["std"]) \
                      / results["Daily Δ only"]["std"]
    print(f"\n  Daily vs weekly delta: PnL vol reduced by {-vol_reduction:.1%} with daily rebalancing")
    print(f"  Adding vega hedge   : PnL vol further reduced by {vega_reduction:.1%}")

    # ── 4. Sensitivity analysis ───────────────────────────────────────────
    print("\n  Barrier sensitivity (price vs barrier level):")
    barriers   = [110, 115, 120, 125, 130, 135, 140]
    bar_prices = [barrier_call_price(S0, K, b, T_annual, r, SIGMA) for b in barriers]
    for b, p in zip(barriers, bar_prices):
        print(f"    B={b}: ${p:.4f}")

    # ── 5. Plots ──────────────────────────────────────────────────────────
    _plot(paths, payoffs, results, barriers, bar_prices, price_analytical, mc_price)

    return results, paths, price_analytical, mc_price


def _plot(paths, payoffs, results, barriers, bar_prices, price_analytical, mc_price):
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("#0d1117")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.40, wspace=0.35)

    colors = {"Daily Δ only": "#58a6ff", "Weekly Δ only": "#3fb950",
              "Daily Δ+Vega": "#d29922",  "Weekly Δ+Vega": "#f85149"}
    CARD = "#161b22"

    # 1. Sample paths
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor(CARD)
    n_show = min(200, len(paths))
    t_axis = np.arange(paths.shape[1])
    for i in range(n_show):
        ko = paths[i].max() >= B
        ax1.plot(t_axis, paths[i], lw=0.4, alpha=0.25,
                 color="#f85149" if ko else "#58a6ff")
    ax1.axhline(B, color="#d29922", lw=1.5, ls="--", label=f"Barrier B={B}")
    ax1.axhline(K, color="#8b949e", lw=1.0, ls=":", label=f"Strike K={K}")
    ax1.set_title("Monte Carlo Paths (200 shown)", color="white", fontsize=11)
    ax1.set_xlabel("Day", color="#8b949e"); ax1.set_ylabel("S(t)", color="#8b949e")
    ax1.legend(fontsize=9); ax1.tick_params(colors="#8b949e")
    ax1.spines[:].set_color("#30363d")

    # 2. MC convergence
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(CARD)
    n_range = np.arange(100, len(payoffs) + 1, 100)
    running = [payoffs[:n].mean() for n in n_range]
    ax2.plot(n_range, running, color="#58a6ff", lw=1.5)
    ax2.axhline(price_analytical, color="#d29922", ls="--", lw=1.5, label="Analytical")
    ax2.axhline(mc_price, color="#3fb950", ls=":", lw=1, label=f"MC final ${mc_price:.3f}")
    ax2.set_title("MC Price Convergence", color="white", fontsize=11)
    ax2.set_xlabel("Paths", color="#8b949e"); ax2.set_ylabel("Price ($)", color="#8b949e")
    ax2.legend(fontsize=9); ax2.tick_params(colors="#8b949e")
    ax2.spines[:].set_color("#30363d")

    # 3. PnL distributions
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.set_facecolor(CARD)
    for name, res in results.items():
        ax3.hist(res["pnl"], bins=60, alpha=0.55, label=f"{name} (σ={res['std']:.3f})",
                 color=colors[name], density=True)
    ax3.axvline(0, color="white", lw=1, ls="--")
    ax3.set_title("Hedging PnL Distribution", color="white", fontsize=11)
    ax3.set_xlabel("Terminal PnL ($)", color="#8b949e")
    ax3.set_ylabel("Density", color="#8b949e")
    ax3.legend(fontsize=8); ax3.tick_params(colors="#8b949e")
    ax3.spines[:].set_color("#30363d")

    # 4. Std comparison
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_facecolor(CARD)
    stds  = [r["std"]   for r in results.values()]
    names = list(results.keys())
    bars  = ax4.barh(names, stds, color=[colors[n] for n in names], alpha=0.8)
    ax4.bar_label(bars, fmt="%.4f", padding=3, color="white", fontsize=8)
    ax4.set_title("PnL Volatility by Strategy", color="white", fontsize=11)
    ax4.set_xlabel("Std(PnL) — lower is better", color="#8b949e")
    ax4.tick_params(colors="#8b949e"); ax4.spines[:].set_color("#30363d")

    # 5. VaR comparison
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.set_facecolor(CARD)
    vars95 = [r["var95"] for r in results.values()]
    bars5  = ax5.barh(names, vars95, color=[colors[n] for n in names], alpha=0.8)
    ax5.bar_label(bars5, fmt="%.4f", padding=3, color="white", fontsize=8)
    ax5.axvline(0, color="white", lw=0.8, ls="--")
    ax5.set_title("VaR 95% by Strategy", color="white", fontsize=11)
    ax5.set_xlabel("VaR (higher = less loss)", color="#8b949e")
    ax5.tick_params(colors="#8b949e"); ax5.spines[:].set_color("#30363d")

    # 6. Payoff distribution
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.set_facecolor(CARD)
    ax6.hist(payoffs[payoffs > 0], bins=40, color="#3fb950", alpha=0.7, label="In-the-money")
    ax6.axvline(0, color="#8b949e", lw=0.8, ls="--")
    ax6.set_title("Option Payoff Distribution\n(non-zero paths only)", color="white", fontsize=11)
    ax6.set_xlabel("Payoff ($)", color="#8b949e")
    ax6.tick_params(colors="#8b949e"); ax6.spines[:].set_color("#30363d")
    ax6.legend(fontsize=9)

    # 7. Barrier sensitivity
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.set_facecolor(CARD)
    ax7.plot(barriers, bar_prices, "o-", color="#bc8cff", lw=2, ms=6)
    ax7.axhline(bs_call(S0, K, T_DAYS / 252, r, SIGMA),
                color="#58a6ff", ls="--", lw=1, label="Vanilla")
    ax7.set_title("Price vs Barrier Level", color="white", fontsize=11)
    ax7.set_xlabel("Barrier B ($)", color="#8b949e")
    ax7.set_ylabel("Option Price ($)", color="#8b949e")
    ax7.legend(fontsize=9); ax7.tick_params(colors="#8b949e")
    ax7.spines[:].set_color("#30363d")

    fig.suptitle(
        f"Up-and-Out Call  |  S₀={S0}  K={K}  B={B}  T={T_DAYS}d  σ={SIGMA:.0%}  "
        f"TC={TC_RATE:.1%}  N={N_PATHS:,}",
        color="white", fontsize=12, fontweight="bold", y=1.01,
    )
    plt.savefig("barrier_option_analysis.png", dpi=150, bbox_inches="tight",
                facecolor="#0d1117")
    print("\n  Saved → barrier_option_analysis.png")
    plt.show()


if __name__ == "__main__":
    run()
