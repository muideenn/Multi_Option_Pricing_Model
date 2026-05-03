"""
Multi-Company Options Pricing Model
Author: Muhideen Ogunlowo
"""
from .models import binomial_price, bs_price, bs_greeks, fit_garch, garch_forecast, implied_vol
from .data import fetch_data, fetch_options_chain, COMPANIES
from .style import C, TICKER_COLORS, BASE_LAYOUT, apply_layout, metric, slide_intro, chart_note
