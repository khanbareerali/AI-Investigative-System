import pandas as pd
import re


def coerce_str(v) -> str:
    if v is None:
        return ''
    try:
        if isinstance(v, float) and pd.isna(v):
            return ''
    except Exception:
        pass
    return str(v).strip()


def detect_column(df: pd.DataFrame, candidates: list) -> str:
    """Return the first column name from candidates that exists in df (case-insensitive)."""
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        match = cols_lower.get(c.lower())
        if match:
            return match
    return ''


def format_confidence_badge(conf: str) -> str:
    colors = {
        'Very High': '#1a7a1a',
        'High': '#0a5c8f',
        'Medium': '#8a6d00',
        'Low': '#a0000a',
        'Ignore': '#666666',
    }
    bg = {
        'Very High': '#c6efce',
        'High': '#9dc3e6',
        'Medium': '#ffeb9c',
        'Low': '#fce4d6',
        'Ignore': '#e0e0e0',
    }
    color = colors.get(conf, '#333')
    background = bg.get(conf, '#f0f0f0')
    return f'<span style="background:{background};color:{color};padding:2px 8px;border-radius:4px;font-weight:bold;">{conf}</span>'
