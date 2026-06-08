"""
CSV-side address clustering.

Groups MCMIS CSV records by normalized carrier address and computes
aggregate statistics. Flags clusters that have more than one unique
carrier name or USDOT number as requiring manual review.

These clusters are investigative leads, not proof of fraud.
"""

import pandas as pd
from .normalizers import normalize_address, normalize_zip


def compute_address_clusters(csv_prep: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame summarising each unique normalized carrier address
    found in the prepared CSV.

    Parameters
    ----------
    csv_prep : DataFrame
        Must contain ``csv_carrier_address_normalized`` produced by
        ``address_matcher.prepare_csv_records``.
    """
    if csv_prep is None or csv_prep.empty:
        return pd.DataFrame()
    if 'csv_carrier_address_normalized' not in csv_prep.columns:
        return pd.DataFrame()

    def _col(row, name):
        return str(row.get(name, '') or '').strip()

    rows = []
    grouped = csv_prep.groupby('csv_carrier_address_normalized', sort=False)

    for addr_key, grp in grouped:
        if not addr_key:
            continue

        carrier_names = sorted(set(
            v for v in grp.get('Carrier Name', pd.Series(dtype=str)).dropna()
            if str(v).strip()
        ))
        usdot_numbers = sorted(set(
            v for v in grp.get('USDOT Number', pd.Series(dtype=str)).dropna()
            if str(v).strip()
        ))
        crash_ids = list(grp.get('Crash ID', pd.Series(dtype=str)).dropna().unique())
        counties = sorted(set(
            v for v in grp.get('County Name', pd.Series(dtype=str)).dropna()
            if str(v).strip()
        ))
        plates = sorted(set(
            v for v in grp.get('Vehicle License Number', pd.Series(dtype=str)).dropna()
            if str(v).strip()
        ))

        crash_dates = pd.to_datetime(
            grp.get('Crash Date', pd.Series(dtype=str)), errors='coerce'
        )
        first_crash = crash_dates.min()
        last_crash = crash_dates.max()

        n_carriers = len(carrier_names)
        n_dots = len(usdot_numbers)
        review = n_carriers > 1 or n_dots > 1

        review_reasons = []
        if n_carriers > 1:
            review_reasons.append(f"{n_carriers} unique carrier names")
        if n_dots > 1:
            review_reasons.append(f"{n_dots} unique USDOT numbers")

        rows.append({
            'normalized_address': addr_key,
            'crash_count': len(crash_ids) if crash_ids else len(grp),
            'unique_carrier_names': n_carriers,
            'unique_usdot_numbers': n_dots,
            'first_crash_date': first_crash.date() if pd.notna(first_crash) else None,
            'last_crash_date': last_crash.date() if pd.notna(last_crash) else None,
            'carrier_names': '; '.join(carrier_names[:15]),
            'usdot_numbers': '; '.join(usdot_numbers[:15]),
            'counties': '; '.join(counties[:10]),
            'vehicle_plates': '; '.join(plates[:10]),
            'review_flag': review,
            'review_reason': '; '.join(review_reasons) if review_reasons else '',
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values(
        ['review_flag', 'crash_count'],
        ascending=[False, False],
    ).reset_index(drop=True)
    return df
