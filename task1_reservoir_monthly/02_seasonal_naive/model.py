"""Seasonal naive: next value = same month last year."""


def predict(series, horizon=1):
    """series: inflow list ending at the last full month; returns the value 12 months back."""
    i = len(series) - 12 + horizon - 1
    return series[i] if 0 <= i < len(series) else None
