"""
chunk_builder.py — Convert DataFrames into text chunks for RAG indexing.

Each chunk has:
    chunk_id   : unique string identifier
    chunk_type : "pdf_page" | "pdf_record" | "csv_row" | "match_result" | "address_cluster"
    text       : human-readable text block for embedding
    metadata   : dict of structured fields for filtering and citation
"""

import hashlib
import pandas as pd


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def safe_get(row, field: str, default: str = "") -> str:
    """Return row[field] as a stripped string, or default if missing/null."""
    try:
        val = row[field]
        if val is None or (isinstance(val, float) and val != val):
            return default
        return str(val).strip()
    except (KeyError, TypeError):
        return default


def dataframe_has_rows(df) -> bool:
    """True if df is a non-None, non-empty DataFrame."""
    return df is not None and isinstance(df, pd.DataFrame) and not df.empty


def _make_id(prefix: str, raw: str) -> str:
    """Stable short ID: prefix + first 8 hex chars of the SHA-1 of raw."""
    digest = hashlib.sha1(raw.encode("utf-8", errors="replace")).hexdigest()[:8]
    return f"{prefix}_{digest}"


# ---------------------------------------------------------------------------
# Per-type chunk builders
# ---------------------------------------------------------------------------

def build_pdf_page_chunks(pdf_pages_df, project_id: str) -> list[dict]:
    """
    Build one chunk per row of a 'PDF pages' DataFrame.

    Expected columns (any subset): source_file, page_number, raw_text
    """
    if not dataframe_has_rows(pdf_pages_df):
        return []

    chunks = []
    for i, row in pdf_pages_df.iterrows():
        source = safe_get(row, "source_file")
        page   = safe_get(row, "page_number")
        text   = safe_get(row, "raw_text")

        if not text:
            continue

        body = (
            f"PDF PAGE\n"
            f"Source file: {source}\n"
            f"Page number: {page}\n"
            f"Text:\n{text}"
        )
        chunk_id = _make_id("pp", f"{project_id}|{source}|{page}|{i}")

        chunks.append({
            "chunk_id":   chunk_id,
            "chunk_type": "pdf_page",
            "text":       body,
            "metadata": {
                "project_id":  project_id,
                "chunk_type":  "pdf_page",
                "source_file": source,
                "page_number": page,
            },
        })

    return chunks


def build_pdf_record_chunks(pdf_records_df, project_id: str) -> list[dict]:
    """
    Build one chunk per extracted PDF vehicle/carrier record.

    Expected columns (any subset): pdf_record_id, source_file, pdf_page_number,
    crash_report_number, case_number, crash_date, driver_name, carrier_name,
    carrier_address_raw, owner_name, owner_address_raw, usdot_number, vin,
    vehicle_plate_number, insurance_company, raw_text
    """
    if not dataframe_has_rows(pdf_records_df):
        return []

    chunks = []
    for i, row in pdf_records_df.iterrows():
        record_id  = safe_get(row, "pdf_record_id")
        source     = safe_get(row, "source_file")
        page       = safe_get(row, "pdf_page_number")
        crash_num  = safe_get(row, "crash_report_number")
        case_num   = safe_get(row, "case_number")
        crash_date = safe_get(row, "crash_date")
        driver     = safe_get(row, "driver_name")
        carrier    = safe_get(row, "carrier_name")
        c_addr     = safe_get(row, "carrier_address_raw")
        owner      = safe_get(row, "owner_name")
        o_addr     = safe_get(row, "owner_address_raw")
        usdot      = safe_get(row, "usdot_number")
        vin        = safe_get(row, "vin")
        plate      = safe_get(row, "vehicle_plate_number")
        insurance  = safe_get(row, "insurance_company")
        raw        = safe_get(row, "raw_text")

        body = (
            f"PDF EXTRACTED RECORD\n"
            f"Source file: {source}\n"
            f"Page: {page}\n"
            f"Crash report number: {crash_num}\n"
            f"Case number: {case_num}\n"
            f"Crash date: {crash_date}\n"
            f"Driver/person name: {driver}\n"
            f"Carrier name: {carrier}\n"
            f"Carrier address: {c_addr}\n"
            f"Owner name: {owner}\n"
            f"Owner address: {o_addr}\n"
            f"USDOT: {usdot}\n"
            f"VIN: {vin}\n"
            f"Plate: {plate}\n"
            f"Insurance: {insurance}\n"
            f"Raw text:\n{raw}"
        )
        chunk_id = _make_id("pr", f"{project_id}|{record_id}|{i}")

        chunks.append({
            "chunk_id":   chunk_id,
            "chunk_type": "pdf_record",
            "text":       body,
            "metadata": {
                "project_id":   project_id,
                "chunk_type":   "pdf_record",
                "source_file":  source,
                "page_number":  page,
                "pdf_record_id": record_id,
                "carrier_name": carrier,
                "address":      c_addr or o_addr,
                "usdot":        usdot,
                "vin":          vin,
            },
        })

    return chunks


def build_csv_row_chunks(csv_df, project_id: str) -> list[dict]:
    """
    Build one chunk per MCMIS CSV row.

    Expected columns (any subset): csv_row_id, Crash ID, Crash Date,
    Carrier Name, USDOT Number, Carrier Street, Carrier City, Carrier State,
    Carrier Zip Code, Vehicle License Number, Vehicle License State, County Name,
    City, Location
    """
    if not dataframe_has_rows(csv_df):
        return []

    chunks = []
    for i, row in csv_df.iterrows():
        row_id   = safe_get(row, "csv_row_id") or str(i + 1)
        crash_id = safe_get(row, "Crash ID")
        c_date   = safe_get(row, "Crash Date")
        carrier  = safe_get(row, "Carrier Name")
        usdot    = safe_get(row, "USDOT Number")
        street   = safe_get(row, "Carrier Street")
        city     = safe_get(row, "Carrier City")
        state    = safe_get(row, "Carrier State")
        zipcode  = safe_get(row, "Carrier Zip Code")
        plate    = safe_get(row, "Vehicle License Number")
        county   = safe_get(row, "County Name")
        loc_city = safe_get(row, "City")
        location = safe_get(row, "Location")

        address = " ".join(p for p in [street, city, state, zipcode] if p)

        body = (
            f"CSV MCMIS ROW\n"
            f"CSV row ID: {row_id}\n"
            f"Crash ID: {crash_id}\n"
            f"Crash date: {c_date}\n"
            f"Carrier name: {carrier}\n"
            f"USDOT: {usdot}\n"
            f"Carrier address: {address}\n"
            f"Vehicle plate: {plate}\n"
            f"County: {county}\n"
            f"City: {loc_city}\n"
            f"Location: {location}"
        )
        chunk_id = _make_id("cr", f"{project_id}|{row_id}|{i}")

        chunks.append({
            "chunk_id":   chunk_id,
            "chunk_type": "csv_row",
            "text":       body,
            "metadata": {
                "project_id":  project_id,
                "chunk_type":  "csv_row",
                "row_id":      row_id,
                "csv_row_id":  row_id,
                "carrier_name": carrier,
                "address":     address,
                "usdot":       usdot,
            },
        })

    return chunks


def build_match_result_chunks(matches_df, project_id: str) -> list[dict]:
    """
    Build one chunk per address match result.

    Expected columns (any subset): match_id, confidence, match_score,
    match_reason, pdf_carrier_name, pdf_carrier_address, pdf_owner_name,
    pdf_owner_address, pdf_usdot, pdf_vin, pdf_plate, csv_carrier_name,
    csv_usdot, csv_carrier_address, different_carrier_name, different_usdot,
    multiple_csv_carriers_at_address, multiple_csv_dots_at_address,
    review_needed, review_notes
    """
    if not dataframe_has_rows(matches_df):
        return []

    chunks = []
    for i, row in matches_df.iterrows():
        mid          = safe_get(row, "match_id")
        confidence   = safe_get(row, "confidence")
        score        = safe_get(row, "match_score")
        reason       = safe_get(row, "match_reason")
        pdf_carrier  = safe_get(row, "pdf_carrier_name")
        pdf_addr     = safe_get(row, "pdf_carrier_address")
        pdf_owner    = safe_get(row, "pdf_owner_name")
        pdf_o_addr   = safe_get(row, "pdf_owner_address")
        pdf_usdot    = safe_get(row, "pdf_usdot")
        pdf_vin      = safe_get(row, "pdf_vin")
        pdf_plate    = safe_get(row, "pdf_plate")
        csv_carrier  = safe_get(row, "csv_carrier_name")
        csv_addr     = safe_get(row, "csv_carrier_address")
        csv_usdot    = safe_get(row, "csv_usdot")
        diff_carrier = safe_get(row, "different_carrier_name")
        diff_usdot   = safe_get(row, "different_usdot")
        multi_carr   = safe_get(row, "multiple_csv_carriers_at_address")
        multi_dot    = safe_get(row, "multiple_csv_dots_at_address")
        review       = safe_get(row, "review_needed")
        notes        = safe_get(row, "review_notes")

        body = (
            f"ADDRESS MATCH RESULT\n"
            f"Match ID: {mid}\n"
            f"Confidence: {confidence}\n"
            f"Score: {score}\n"
            f"Reason: {reason}\n"
            f"PDF carrier: {pdf_carrier}\n"
            f"PDF address: {pdf_addr}\n"
            f"PDF owner: {pdf_owner}\n"
            f"PDF owner address: {pdf_o_addr}\n"
            f"PDF USDOT: {pdf_usdot}\n"
            f"PDF VIN: {pdf_vin}\n"
            f"PDF plate: {pdf_plate}\n"
            f"CSV carrier: {csv_carrier}\n"
            f"CSV address: {csv_addr}\n"
            f"CSV USDOT: {csv_usdot}\n"
            f"Characteristics:\n"
            f"  Different carrier name: {diff_carrier}\n"
            f"  Different USDOT: {diff_usdot}\n"
            f"  Multiple CSV carriers at address: {multi_carr}\n"
            f"  Multiple CSV DOTs at address: {multi_dot}\n"
            f"Review needed: {review}\n"
            f"Review notes: {notes}"
        )
        chunk_id = _make_id("mr", f"{project_id}|{mid}|{i}")

        chunks.append({
            "chunk_id":   chunk_id,
            "chunk_type": "match_result",
            "text":       body,
            "metadata": {
                "project_id":  project_id,
                "chunk_type":  "match_result",
                "match_id":    str(mid),
                "confidence":  confidence,
                "match_score": str(score),
                "carrier_name": pdf_carrier or csv_carrier,
                "address":     pdf_addr or csv_addr,
                "usdot":       pdf_usdot or csv_usdot,
                "vin":         pdf_vin,
                "match_reason": reason,
            },
        })

    return chunks


def build_address_cluster_chunks(clusters_df, project_id: str) -> list[dict]:
    """
    Build one chunk per address cluster row.

    Expected columns (any subset): normalized_address,
    carrier_full_address_normalized, crash_count, unique_carrier_names,
    unique_usdot_numbers, first_crash_date, last_crash_date, carrier_names,
    usdot_numbers, counties, vehicle_plates, review_flag, review_reason,
    address_cluster_flag, cluster_priority_score
    """
    if not dataframe_has_rows(clusters_df):
        return []

    chunks = []
    for i, row in clusters_df.iterrows():
        # Accept both possible address column names
        address = (
            safe_get(row, "normalized_address")
            or safe_get(row, "carrier_full_address_normalized")
        )
        crash_count    = safe_get(row, "crash_count")
        n_carriers     = safe_get(row, "unique_carrier_names")
        n_dots         = safe_get(row, "unique_usdot_numbers")
        first_date     = safe_get(row, "first_crash_date")
        last_date      = safe_get(row, "last_crash_date")
        carrier_names  = safe_get(row, "carrier_names")
        usdot_nums     = safe_get(row, "usdot_numbers")
        counties       = safe_get(row, "counties")
        plates         = safe_get(row, "vehicle_plates")
        cluster_flag   = (
            safe_get(row, "address_cluster_flag")
            or safe_get(row, "review_flag")
        )
        priority       = (
            safe_get(row, "cluster_priority_score")
            or safe_get(row, "review_reason")
        )

        if not address:
            continue

        body = (
            f"ADDRESS CLUSTER\n"
            f"Address: {address}\n"
            f"Crash count: {crash_count}\n"
            f"Unique carrier names: {n_carriers}\n"
            f"Unique USDOT numbers: {n_dots}\n"
            f"First crash date: {first_date}\n"
            f"Last crash date: {last_date}\n"
            f"Carrier names: {carrier_names}\n"
            f"USDOT numbers: {usdot_nums}\n"
            f"Counties: {counties}\n"
            f"Vehicle plates: {plates}\n"
            f"Cluster flag: {cluster_flag}\n"
            f"Priority score: {priority}"
        )
        chunk_id = _make_id("ac", f"{project_id}|{address}|{i}")

        chunks.append({
            "chunk_id":   chunk_id,
            "chunk_type": "address_cluster",
            "text":       body,
            "metadata": {
                "project_id":   project_id,
                "chunk_type":   "address_cluster",
                "address":      address,
                "carrier_name": carrier_names,
                "usdot":        usdot_nums,
                "crash_count":  str(crash_count),
            },
        })

    return chunks


# ---------------------------------------------------------------------------
# Combined builder
# ---------------------------------------------------------------------------

def build_all_chunks(
    project_id: str,
    pdf_pages_df=None,
    pdf_records_df=None,
    csv_df=None,
    matches_df=None,
    clusters_df=None,
) -> list[dict]:
    """
    Build and return all chunks from every available DataFrame.

    Any None or empty DataFrame is silently skipped.
    Duplicate chunk_ids are deduplicated (last writer wins).
    """
    all_chunks: list[dict] = []

    all_chunks.extend(build_pdf_page_chunks(pdf_pages_df, project_id))
    all_chunks.extend(build_pdf_record_chunks(pdf_records_df, project_id))
    all_chunks.extend(build_csv_row_chunks(csv_df, project_id))
    all_chunks.extend(build_match_result_chunks(matches_df, project_id))
    all_chunks.extend(build_address_cluster_chunks(clusters_df, project_id))

    # Remove chunks with blank text (defensive)
    all_chunks = [c for c in all_chunks if c.get("text", "").strip()]

    # Deduplicate by chunk_id
    seen: dict[str, dict] = {}
    for chunk in all_chunks:
        seen[chunk["chunk_id"]] = chunk

    return list(seen.values())
