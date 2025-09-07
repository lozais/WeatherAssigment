#!/usr/bin/env python3
"""
windgust.py
------------------
Compute wind gust from both turbulence-based and mix-down methods.
"""

import numpy as np

def wind_speed(u, v):
    """Return wind speed from components (m s^-1)."""
    return np.hypot(u, v)

def gust_from_tke(u10, v10, tke, C=2.4):
    """
    Turbulence-based gust: G = U10 + C*sqrt(TKE).

    """
    U10 = wind_speed(u10, v10)
    return U10 + C * np.sqrt(np.maximum(tke, 0.0))


def gust_from_925(u10, v10, u925, v925, frac=0.6):
    """
    Mix-down gust using 925 hPa wind: G = max(U10, frac*U925).

    """
    U10 = wind_speed(u10, v10)
    U925 = wind_speed(u925, v925)
    return np.maximum(U10, frac * U925)


def combined_gust(u10, v10, tke=None, u925=None, v925=None, C=2.4, frac=0.6):
    """
    TheBestThe best approach is to use the maximum of both methods.

    """
    estimates = []
    U10 = wind_speed(u10, v10)
    estimates.append(U10)

    if tke is not None:
        estimates.append(gust_from_tke(u10, v10, tke, C=C))
    if (u925 is not None) and (v925 is not None):
        estimates.append(gust_from_925(u10, v10, u925, v925, frac=frac))

    return np.maximum.reduce(estimates)
