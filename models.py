"""
src/models.py
─────────────────────────────────────────────────────────────────────────────
Core financial models:
  - Binomial (CRR) option pricing — European & American
  - Black-Scholes closed-form pricing + all five Greeks
  - GARCH(1,1) volatility fitting and forecasting
  - Implied volatility solver (bisection)

Author: Muhideen Ogunlowo
"""

import numpy as np
from scipy.stats import norm
from arch import arch_model


# ── Binomial (CRR) ────────────────────────────────────────────────────────────

def binomial_price(S, K, T, r, sigma, N=200,
                   option_type="call", exercise="european"):
    """
    Cox-Ross-Rubinstein binomial tree option pricer.

    Parameters
    ----------
    S : float  — Current stock price
    K : float  — Strike price
    T : float  — Time to maturity (years)
    r : float  — Risk-free rate (annualised, e.g. 0.0525)
    sigma : float  — Volatility (annualised, e.g. 0.20)
    N : int    — Number of time steps
    option_type : str — 'call' or 'put'
    exercise : str    — 'european' or 'american'

    Returns
    -------
    float — Option price
    """
    if T < 1e-9 or sigma < 1e-9:
        return max(S - K, 0) if option_type == "call" else max(K - S, 0)

    dt   = T / N
    u    = np.exp(sigma * np.sqrt(dt))
    d    = 1.0 / u
    p    = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    j  = np.arange(N + 1)
    ST = S * u ** (N - j) * d ** j
    V  = np.maximum(ST - K, 0) if option_type == "call" else np.maximum(K - ST, 0)

    for i in range(N - 1, -1, -1):
        V = disc * (p * V[:-1] + (1 - p) * V[1:])
        if exercise == "american":
            ji = np.arange(i + 1)
            Si = S * u ** (i - ji) * d ** ji
            pf = np.maximum(Si - K, 0) if option_type == "call" else np.maximum(K - Si, 0)
            V  = np.maximum(V, pf)

    return float(V[0])


# ── Black-Scholes ─────────────────────────────────────────────────────────────

def bs_price(S, K, T, r, sigma, option_type="call"):
    """
    Analytical Black-Scholes European option price.

    Parameters
    ----------
    S, K, T, r, sigma : float — Standard B-S inputs
    option_type : str — 'call' or 'put'

    Returns
    -------
    float — Option price
    """
    if T < 1e-9 or sigma < 1e-9:
        return max(S - K, 0) if option_type == "call" else max(K - S, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_greeks(S, K, T, r, sigma, option_type="call"):
    """
    Compute all five Black-Scholes Greeks.

    Returns
    -------
    dict with keys: delta, gamma, theta, vega, rho
      - theta  : per calendar day
      - vega   : per 1% change in vol
      - rho    : per 1% change in rate
    """
    if T < 1e-9 or sigma < 1e-9:
        return dict(delta=0., gamma=0., theta=0., vega=0., rho=0.)

    d1   = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2   = d1 - sigma * np.sqrt(T)
    pdf1 = norm.pdf(d1)

    delta = norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1
    gamma = pdf1 / (S * sigma * np.sqrt(T))

    base_theta = -S * pdf1 * sigma / (2 * np.sqrt(T))
    if option_type == "call":
        theta = (base_theta - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        theta = (base_theta + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

    vega = S * pdf1 * np.sqrt(T) / 100   # per 1% vol
    rho  = (K * T * np.exp(-r * T) * norm.cdf(d2)
            if option_type == "call"
            else -K * T * np.exp(-r * T) * norm.cdf(-d2)) / 100

    return dict(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)


# ── GARCH(1,1) ────────────────────────────────────────────────────────────────

def fit_garch(log_returns, p=1, q=1):
    """
    Fit GARCH(p,q) model to daily log returns.

    Parameters
    ----------
    log_returns : pd.Series — Daily log returns (not scaled)
    p, q : int — GARCH lag orders

    Returns
    -------
    (result, params_dict)
      result      : arch ARCHModelResult object
      params_dict : dict with keys omega, alpha, beta
    """
    model  = arch_model(log_returns * 100, vol="Garch", p=p, q=q, dist="normal")
    result = model.fit(disp="off")

    pm   = result.params
    keys = list(pm.index)
    omega = float(pm[next(k for k in keys if "omega" in k.lower())])
    alpha = float(pm[next(k for k in keys if "alpha" in k.lower())])
    beta  = float(pm[next(k for k in keys if "beta"  in k.lower())])

    return result, dict(omega=omega, alpha=alpha, beta=beta)


def garch_forecast(result, horizon=60):
    """
    Generate h-step-ahead annualised volatility forecast.

    Parameters
    ----------
    result  : ARCHModelResult
    horizon : int — Number of trading days to forecast

    Returns
    -------
    np.ndarray — Annualised vol forecast (length = horizon)
    """
    fc  = result.forecast(horizon=horizon)
    var = fc.variance.values[-1, :]          # daily variance in %^2
    return np.sqrt(var) / 100 * np.sqrt(252) # annualised


# ── Implied Volatility ────────────────────────────────────────────────────────

def implied_vol(mkt_price, S, K, T, r, option_type="call", tol=1e-7):
    """
    Bisection solver for Black-Scholes implied volatility.

    Parameters
    ----------
    mkt_price : float — Observed market option price (mid)
    S, K, T, r : float — Standard B-S inputs
    option_type : str
    tol : float — Convergence tolerance

    Returns
    -------
    float — Implied volatility (annualised)
    """
    lo, hi = 1e-6, 6.0
    for _ in range(150):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if bs_price(S, K, T, r, mid, option_type) < mkt_price else (lo, mid)
        if hi - lo < tol:
            break
    return float(mid)
