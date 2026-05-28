# Pricing and Hedging an Up-and-Out Barrier Option

Monte Carlo simulation under Black-Scholes dynamics to price and hedge an exotic barrier option. Compares daily vs weekly delta and delta-vega hedging strategies with 0.1% transaction costs.

---

## Setup

```
S₀ = 100   K = 100   B = 110 (barrier)
T  = 21 days          r = 2%    σ = 20%
TC = 0.1% per rebalance   N = 10,000 paths
```

---

## Key Results

| Metric | Value |
|--------|-------|
| Analytical barrier price | $2.07 |
| Vanilla call (reference) | $2.39 |
| Barrier discount | 13% |
| MC price (10k paths) | ~$2.00 |
| Knock-out probability | ~7.5% |

### Hedging comparison

| Strategy | PnL Std | VaR 95% |
|----------|---------|---------|
| Daily Δ only | lowest | best |
| Weekly Δ only | higher | worse |
| Daily Δ + Vega | lowest | best |
| Weekly Δ + Vega | mid | mid |

Daily rebalancing reduces PnL volatility by **~30%** vs weekly. Adding vega hedge provides further stabilisation near the barrier.

---

## Methods

- **Closed-form price**: Rubinstein & Reiner (1991) reflection principle
- **Monte Carlo**: 10,000 GBM paths, daily discretisation
- **Delta hedge**: numerical Δ of barrier option, rebalanced daily or weekly
- **Vega hedge**: offsetting vanilla call position sized to neutralise vega
- **Transaction costs**: 0.1% per Δ and vega rebalance

---

## Run

```bash
pip install numpy scipy matplotlib
python3 barrier_option.py
```

Outputs: console table + 7-panel chart (`barrier_option_analysis.png`)
