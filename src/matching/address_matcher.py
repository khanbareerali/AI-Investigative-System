"""
Core address-matching engine.

Prepares PDF and CSV records with normalized fields, then scores every
PDF record against every CSV record using address similarity. Only pairs
that meet the threshold are returned.
"""

import pandas as pd
from .normalizers import (
    normalize_address,
    normalize_zip,
    normalize_carrier_name,
    normalize_vin,
    normalize_plate,
)
from .scoring import score_address_pair, get_confidence

_S = lambda v: '' if (v is None or (isinstance(v, float) and v != v)) else str(v).strip()


# ---------------------------------------------------------------------------
# Preparation helpers
# ---------------------------------------------------------------------------

def prepare_pdf_records(pdf_df: pd.DataFrame) -> pd.DataFrame:
    df = pdf_df.copy()

    def _addr(row, addr_col, zip_col):
        return normalize_address(_S(row.get(addr_col, '')), _S(row.get(zip_col, '')))

    df['pdf_carrier_address_normalized'] = df.apply(
        lambda r: _addr(r, 'carrier_address_raw', 'carrier_zip'), axis=1
    )
    df['pdf_owner_address_normalized'] = df.apply(
        lambda r: _addr(r, 'owner_address_raw', 'owner_zip'), axis=1
    )
    df['pdf_carrier_name_normalized'] = df['carrier_name'].apply(normalize_carrier_name) \
        if 'carrier_name' in df.columns else ''
    df['pdf_owner_name_normalized'] = df['owner_name'].apply(normalize_carrier_name) \
        if 'owner_name' in df.columns else ''
    df['pdf_usdot_clean'] = df['usdot_number'].apply(_S) if 'usdot_number' in df.columns else ''
    df['pdf_vin_clean'] = df['vin'].apply(normalize_vin) if 'vin' in df.columns else ''
    df['pdf_plate_clean'] = df['vehicle_plate_number'].apply(normalize_plate) \
        if 'vehicle_plate_number' in df.columns else ''

    return df


def prepare_csv_records(csv_df: pd.DataFrame) -> pd.DataFrame:
    df = csv_df.copy()

    def _build_raw(row):
        parts = [
            _S(row.get('Carrier Street', '')),
            _S(row.get('Carrier City', '')),
            _S(row.get('Carrier State', '')),
            _S(row.get('Carrier Zip Code', '')),
        ]
        return ' '.join(p for p in parts if p)

    def _build_norm(row):
        street_city_state = ' '.join(filter(None, [
            _S(row.get('Carrier Street', '')),
            _S(row.get('Carrier City', '')),
            _S(row.get('Carrier State', '')),
        ]))
        return normalize_address(street_city_state, _S(row.get('Carrier Zip Code', '')))

    df['csv_carrier_address_raw'] = df.apply(_build_raw, axis=1)
    df['csv_carrier_address_normalized'] = df.apply(_build_norm, axis=1)

    df['csv_carrier_name_normalized'] = (
        df['Carrier Name'].apply(normalize_carrier_name)
        if 'Carrier Name' in df.columns else ''
    )
    df['csv_usdot_clean'] = (
        df['USDOT Number'].apply(_S) if 'USDOT Number' in df.columns else ''
    )
    df['csv_plate_clean'] = (
        df['Vehicle License Number'].apply(normalize_plate)
        if 'Vehicle License Number' in df.columns else ''
    )

    if 'csv_row_id' not in df.columns:
        df.insert(0, 'csv_row_id', range(1, len(df) + 1))

    return df


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def run_matching(
    pdf_df: pd.DataFrame,
    csv_df: pd.DataFrame,
    threshold: float = 75.0,
) -> pd.DataFrame:
    """
    Match every PDF record against every CSV record using address scoring.

    Primary key: pdf.carrier_address_raw  →  csv.Carrier Street/City/State/Zip
    Fallback   : pdf.owner_address_raw    →  csv.Carrier Street/City/State/Zip
      (used only when carrier address is blank)

    Returns a DataFrame sorted by match_score descending, filtered to >= threshold.
    """
    pdf_prep = prepare_pdf_records(pdf_df)
    csv_prep = prepare_csv_records(csv_df)

    match_rows: list[dict] = []
    match_id = 0

    for _, pdf in pdf_prep.iterrows():
        carrier_addr = _S(pdf.get('pdf_carrier_address_normalized', ''))
        carrier_zip = normalize_zip(_S(pdf.get('carrier_zip', '')))
        owner_addr = _S(pdf.get('pdf_owner_address_normalized', ''))
        owner_zip = normalize_zip(_S(pdf.get('owner_zip', '')))

        if carrier_addr.strip():
            pdf_addr_to_use = carrier_addr
            pdf_zip_to_use = carrier_zip
            addr_source = 'carrier'
        elif owner_addr.strip():
            pdf_addr_to_use = owner_addr
            pdf_zip_to_use = owner_zip
            addr_source = 'owner (fallback)'
        else:
            continue

        for _, csv in csv_prep.iterrows():
            csv_street_norm = normalize_address(_S(csv.get('Carrier Street', '')))
            csv_city = _S(csv.get('Carrier City', ''))
            csv_state = _S(csv.get('Carrier State', ''))
            csv_zip = _S(csv.get('Carrier Zip Code', ''))

            if not csv_street_norm.strip() and not csv_city.strip():
                continue

            result = score_address_pair(
                pdf_addr_to_use,
                pdf_zip_to_use,
                csv_street_norm,
                csv_city,
                csv_state,
                csv_zip,
            )

            if result['score'] < threshold:
                continue

            match_id += 1

            pdf_carrier_name = _S(pdf.get('pdf_carrier_name_normalized', ''))
            csv_carrier_name = _S(csv.get('csv_carrier_name_normalized', ''))
            pdf_usdot = _S(pdf.get('pdf_usdot_clean', ''))
            csv_usdot = _S(csv.get('csv_usdot_clean', ''))

            diff_carrier = bool(pdf_carrier_name and csv_carrier_name
                                and pdf_carrier_name != csv_carrier_name)
            diff_usdot = bool(pdf_usdot and csv_usdot and pdf_usdot != csv_usdot)

            match_reason = result['match_reason']
            if diff_carrier:
                match_reason += "; Same address but different carrier name"
            if diff_usdot:
                match_reason += "; Same address but different USDOT number"

            review_needed = result['review_needed'] or diff_carrier

            match_rows.append({
                # Core match info
                'match_id': match_id,
                'match_type': f'fuzzy_address ({addr_source})',
                'confidence': result['confidence'],
                'match_score': result['score'],
                'match_reason': match_reason,
                # PDF side
                'pdf_record_id': _S(pdf.get('pdf_record_id', '')),
                'pdf_page_number': _S(pdf.get('pdf_page_number', '')),
                'pdf_crash_report_number': _S(pdf.get('crash_report_number', '')),
                'pdf_driver_name': _S(pdf.get('driver_name', '')),
                'pdf_carrier_name': _S(pdf.get('carrier_name', '')),
                'pdf_carrier_address': _S(pdf.get('carrier_address_raw', '')),
                'pdf_owner_name': _S(pdf.get('owner_name', '')),
                'pdf_owner_address': _S(pdf.get('owner_address_raw', '')),
                'pdf_usdot': _S(pdf.get('usdot_number', '')),
                'pdf_vin': _S(pdf.get('vin', '')),
                'pdf_plate': _S(pdf.get('vehicle_plate_number', '')),
                # CSV side
                'csv_row_id': _S(csv.get('csv_row_id', '')),
                'csv_crash_id': _S(csv.get('Crash ID', '')),
                'csv_crash_date': _S(csv.get('Crash Date', '')),
                'csv_carrier_name': _S(csv.get('Carrier Name', '')),
                'csv_usdot': _S(csv.get('USDOT Number', '')),
                'csv_carrier_address': _S(csv.get('csv_carrier_address_raw', '')),
                'csv_county': _S(csv.get('County Name', '')),
                'csv_vehicle_plate': _S(csv.get('Vehicle License Number', '')),
                # Characteristics (multi-address flags filled post-process)
                'same_or_similar_address': result['score'] >= 75,
                'different_carrier_name': diff_carrier,
                'different_usdot': diff_usdot,
                'multiple_csv_carriers_at_address': False,
                'multiple_csv_dots_at_address': False,
                'pdf_vin_available': bool(
                    normalize_vin(_S(pdf.get('vin', '')))
                ),
                'pdf_person_name_available': bool(_S(pdf.get('driver_name', ''))),
                'repeated_csv_crash_history_at_address': False,
                'review_needed': review_needed,
                'review_notes': '',
            })

    if not match_rows:
        return pd.DataFrame()

    df = pd.DataFrame(match_rows)

    # Post-process: fill multi-carrier / multi-dot / repeated-crash flags
    if 'csv_carrier_address' in df.columns:
        addr_carrier = df.groupby('csv_carrier_address')['csv_carrier_name'].nunique()
        addr_dot = df.groupby('csv_carrier_address')['csv_usdot'].nunique()
        addr_crash = df.groupby('csv_carrier_address')['csv_crash_id'].count()

        df['multiple_csv_carriers_at_address'] = df['csv_carrier_address'].map(
            lambda a: addr_carrier.get(a, 0) > 1
        )
        df['multiple_csv_dots_at_address'] = df['csv_carrier_address'].map(
            lambda a: addr_dot.get(a, 0) > 1
        )
        df['repeated_csv_crash_history_at_address'] = df['csv_carrier_address'].map(
            lambda a: addr_crash.get(a, 0) > 2
        )
        df['review_needed'] = (
            df['review_needed']
            | df['multiple_csv_carriers_at_address']
            | df['multiple_csv_dots_at_address']
        )

        def _notes(row):
            n = []
            if row['multiple_csv_carriers_at_address']:
                n.append("Multiple CSV carriers at this address")
            if row['multiple_csv_dots_at_address']:
                n.append("Multiple USDOT numbers at this address")
            if row['different_carrier_name']:
                n.append("PDF and CSV carrier names differ")
            if row['different_usdot']:
                n.append("PDF and CSV USDOT numbers differ")
            return '; '.join(n)

        df['review_notes'] = df.apply(_notes, axis=1)

    return df.sort_values('match_score', ascending=False).reset_index(drop=True)
