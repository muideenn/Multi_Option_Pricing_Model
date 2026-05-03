"""
src/data.py
─────────────────────────────────────────────────────────────────────────────
Market data fetching and preprocessing.

Provides:
  - fetch_data()      : Pull OHLCV from yfinance with synthetic fallback
  - _mock_data()      : Realistic synthetic data when offline
  - COMPANIES         : Registry of all 10 supported tickers

Author: Muhideen Ogunlowo
"""

import numpy as np
import pandas as pd
import yfinance as yf


# ── Company Registry ──────────────────────────────────────────────────────────

COMPANIES = {
    "DIS":  dict(name="Walt Disney Company",    sector="Communications",    mu=0.0001, sig=0.014, start=100),
    "AAPL": dict(name="Apple Inc.",              sector="Technology",        mu=0.0005, sig=0.016, start=180),
    "MSFT": dict(name="Microsoft Corporation",   sector="Technology",        mu=0.0006, sig=0.015, start=380),
    "GOOGL":dict(name="Alphabet Inc.",           sector="Technology",        mu=0.0004, sig=0.017, start=170),
    "AMZN": dict(name="Amazon.com Inc.",         sector="Consumer Disc.",    mu=0.0003, sig=0.019, start=190),
    "TSLA": dict(name="Tesla Inc.",              sector="Consumer Disc.",    mu=0.0002, sig=0.032, start=200),
    "NVDA": dict(name="NVIDIA Corporation",      sector="Technology",        mu=0.0010, sig=0.028, start=800),
    "META": dict(name="Meta Platforms Inc.",     sector="Technology",        mu=0.0005, sig=0.023, start=500),
    "NFLX": dict(name="Netflix Inc.",            sector="Communications",    mu=0.0003, sig=0.021, start=600),
    "JPM":  dict(name="JPMorgan Chase & Co.",    sector="Financials",        mu=0.0003, sig=0.013, start=200),
}


# ── Synthetic Fallback ────────────────────────────────────────────────────────

def _mock_data(ticker: str) -> pd.DataFrame:
    """
    Generate realistic synthetic OHLCV-style data for a given ticker.
    Uses company-specific drift and volatility parameters from COMPANIES.
    Injects a brief stress period ~2 years ago to simulate vol clustering.

    Parameters
    ----------
    ticker : str — One of the tickers in COMPANIES

    Returns
    -------
    pd.DataFrame with columns: Close, Log_Return, HV_21, HV_63, HV_126
    """
    c = COMPANIES.get(ticker, COMPANIES["DIS"])
    np.random.seed(abs(hash(ticker)) % (2 ** 31))

    rng = pd.bdate_range(end=pd.Timestamp.now(), periods=5 * 252)
    lr  = np.random.normal(c["mu"], c["sig"], len(rng))

    # Inject a stress spike 2 years ago
    mid = len(rng) - 2 * 252
    lr[mid:mid + 20] *= 3.0

    px = np.exp(np.cumsum(lr)) * c["start"]
    df = pd.DataFrame({"Close": px, "Log_Return": lr}, index=rng)

    for w in (21, 63, 126):
        df[f"HV_{w}"] = df["Log_Return"].rolling(w).std() * np.sqrt(252)

    return df.dropna()


# ── Live Data Fetch ───────────────────────────────────────────────────────────

def fetch_data(ticker: str = "DIS", period: str = "5y") -> pd.DataFrame:
    """
    Fetch historical OHLCV data from Yahoo Finance.
    Falls back to synthetic data if yfinance is unavailable or returns empty data.

    Parameters
    ----------
    ticker : str — Stock ticker symbol (e.g. 'DIS', 'AAPL')
    period : str — yfinance period string (e.g. '5y', '2y', '1y')

    Returns
    -------
    pd.DataFrame with columns: Close, Log_Return, HV_21, HV_63, HV_126
    """
    try:
        df = yf.Ticker(ticker).history(period=period)
        if df.empty:
            raise ValueError("yfinance returned empty DataFrame")

        df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
        for w in (21, 63, 126):
            df[f"HV_{w}"] = df["Log_Return"].rolling(w).std() * np.sqrt(252)

        return df.dropna()

    except Exception as e:
        print(f"[data] yfinance unavailable for {ticker} ({e}) — using synthetic data")
        return _mock_data(ticker)


def fetch_options_chain(ticker: str, option_type: str = "call"):
    """
    Fetch the nearest available options chain for a ticker.

    Parameters
    ----------
    ticker : str
    option_type : str — 'call' or 'put'

    Returns
    -------
    (exp_date, chain_df) or raises Exception if unavailable
    """
    tkr = yf.Ticker(ticker)
    exp_dates = tkr.options
    if not exp_dates:
        raise ValueError(f"No options data for {ticker}")

    exp_date = exp_dates[min(2, len(exp_dates) - 1)]
    chain = getattr(tkr.option_chain(exp_date), "calls" if option_type == "call" else "puts")
    chain = chain[(chain["bid"] > 0) & (chain["ask"] > 0)].copy()
    chain["mid"] = (chain["bid"] + chain["ask"]) / 2

    return exp_date, chain
