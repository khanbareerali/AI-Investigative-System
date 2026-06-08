"""
Document Matching MVP — Streamlit application.

Default preset: PDF Crash Report → MCMIS CSV Address Matching

Run with:
    streamlit run app.py
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Must be silenced BEFORE streamlit is imported so the placeholder loggers
# are already at ERROR level when streamlit populates them.
for _logger_name in (
    "streamlit.runtime.scriptrunner_utils.script_run_context",
    "streamlit.runtime.state.session_state_proxy",
    "streamlit.runtime.scriptrunner_utils",
):
    logging.getLogger(_logger_name).setLevel(logging.ERROR)

import streamlit as st
import streamlit.runtime as _st_runtime
import pandas as pd

import io
import json

from src.ingestion.file_loader import load_csv, load_tabular
from src.ingestion.pdf_extractor import (
    extract_pdf_records,
    extract_docx_records,
    extract_plaintext_records,
)
from src.ingestion.drive_loader import (
    load_drive_folder,
    list_folder_files,
    KNOWN_FOLDER_ID,
)

_DOCX_MIMES = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
}
_PLAINTEXT_MIMES = {
    "text/plain", "application/rtf", "text/rtf",
    "application/vnd.google-apps.document",
    "application/vnd.google-apps.presentation",
}
from src.matching.address_matcher import run_matching, prepare_csv_records
from src.matching.clustering import compute_address_clusters
from src.export.export_results import to_csv_bytes, to_excel_bytes
from src.rag.rag_pipeline import build_project_index, ask_project_question, explain_match
from src.rag.vector_store import collection_count

# ---------------------------------------------------------------------------
# Constants / defaults (no Streamlit calls here — safe to import)
# ---------------------------------------------------------------------------

_CSS = """
<style>
    .disclaimer-box {
        background: #fff8e1;
        border-left: 5px solid #f59e0b;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 16px;
        font-size: 0.88rem;
    }
    .conf-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.82rem;
    }
    div[data-testid="stDataFrame"] { font-size: 0.82rem; }
</style>
"""

DISCLAIMER_HTML = """
<div class="disclaimer-box">
<strong>⚠ INVESTIGATIVE LEADS ONLY</strong> — These results identify potential address matches
for investigative purposes. They do <em>not</em> constitute proof of chameleon-carrier behavior,
insurance fraud, or any other misconduct. All matches require manual verification before any
investigative or enforcement action.
</div>
"""

_STATE_DEFAULTS: dict = {
    'pdf_df': None,
    'csv_df': None,
    'csv_prep': None,
    'matches_df': None,
    'clusters_df': None,
    'last_pdf_name': '',
    'last_csv_name': '',
    'match_config': {
        'preset': 'PDF Crash Report to MCMIS CSV Address Matching',
        'pdf_field': 'carrier_address_raw',
        'csv_fields': ['Carrier Street', 'Carrier City', 'Carrier State', 'Carrier Zip Code'],
        'match_method': 'Fuzzy Address',
        'threshold': 75,
    },
    'matching_ran': False,
}

PRESETS = {
    'PDF Crash Report to MCMIS CSV Address Matching': {
        'pdf_field': 'carrier_address_raw',
        'csv_fields': ['Carrier Street', 'Carrier City', 'Carrier State', 'Carrier Zip Code'],
        'match_method': 'Fuzzy Address',
        'threshold': 75,
        'description': (
            "Matches **PDF carrier address + ZIP** to **CSV Carrier Street/City/State/ZIP**. "
            "Falls back to owner address when carrier address is blank. "
            "USDOT is shown as supporting evidence only, never as the primary match key."
        ),
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init_state() -> None:
    for k, v in _STATE_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Tab builders
# ---------------------------------------------------------------------------

def _get_service_account_key():
    """
    Return a service account key source from secrets.toml, or None.

    Checks two secret keys in order:
      GOOGLE_SERVICE_ACCOUNT_JSON         — file path to the JSON key
      GOOGLE_SERVICE_ACCOUNT_JSON_CONTENT — full JSON content as a string
    """
    try:
        path = st.secrets.get("GOOGLE_SERVICE_ACCOUNT_JSON", "")
        if path and path.strip():
            return path.strip()
        content = st.secrets.get("GOOGLE_SERVICE_ACCOUNT_JSON_CONTENT", "")
        if content and content.strip().startswith("{"):
            return content.strip()
    except Exception:
        pass
    return None


def _show_loaded_status() -> None:
    """Show green status banners for whatever is already loaded in session state."""
    if st.session_state.pdf_df is not None:
        n = len(st.session_state.pdf_df)
        st.success(f"PDF ready — **{n} record(s)** from `{st.session_state.last_pdf_name}`")
        _show_pdf_coverage()
    if st.session_state.csv_df is not None:
        n = len(st.session_state.csv_df)
        st.success(f"CSV ready — **{n} rows** from `{st.session_state.last_csv_name}`")
    if st.session_state.pdf_df is not None and st.session_state.csv_df is not None:
        st.info(
            "Files loaded. Go to **3 · Configure Matching**, then **4 · Run Matching**."
        )


def _show_pdf_coverage() -> None:
    if st.session_state.pdf_df is None:
        return
    n = len(st.session_state.pdf_df)
    with st.expander("PDF field coverage"):
        key_fields = [
            'crash_report_number', 'carrier_name', 'carrier_address_raw',
            'carrier_zip', 'owner_name', 'owner_address_raw',
            'usdot_number', 'vin', 'vehicle_plate_number',
        ]
        coverage = {
            col: st.session_state.pdf_df[col].apply(lambda x: bool(str(x).strip())).sum()
            for col in key_fields
            if col in st.session_state.pdf_df.columns
        }
        st.dataframe(
            pd.DataFrame({
                'Field': list(coverage.keys()),
                'Records with data': list(coverage.values()),
                '% filled': [f"{v / n * 100:.0f}%" for v in coverage.values()],
            }),
            hide_index=True,
            use_container_width=True,
        )


def _tab_upload() -> None:
    st.header("Load Files")

    source = st.radio(
        "Source",
        options=["Google Drive", "Local Upload"],
        horizontal=True,
        key="source_selector",
    )

    # -----------------------------------------------------------------------
    # GOOGLE DRIVE MODE
    # -----------------------------------------------------------------------
    if source == "Google Drive":
        st.markdown("#### Google Drive Folder")

        folder_id = st.text_input(
            "Folder ID",
            value=KNOWN_FOLDER_ID,
            key="drive_folder_id",
            help="The long ID from your Google Drive folder URL.",
        )

        # --- Service account credential input ---
        secrets_key = _get_service_account_key()
        if secrets_key:
            st.success("Service account key loaded from `secrets.toml`.")
            key_source = secrets_key
        else:
            st.markdown("**Service account JSON key**")
            with st.expander("How to create a service account key", expanded=False):
                st.markdown("""
1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. **APIs & Services → Library** → search **Google Drive API** → Enable
3. **IAM & Admin → Service Accounts → Create Service Account**
   - Name it anything (e.g. `drive-reader`)
   - Skip role assignment — click Done
4. Click the service account → **Keys → Add Key → Create new key → JSON** → Download
5. **In Google Drive**: open the folder, click Share, paste the service account email
   (shown in the JSON as `client_email`) and give it **Viewer** access
6. Either:
   - Add to `.streamlit/secrets.toml`:
     `GOOGLE_SERVICE_ACCOUNT_JSON = "path/to/downloaded_key.json"`
   - **Or** upload the JSON file below
                """)

            uploaded_key = st.file_uploader(
                "Upload service account JSON key",
                type=["json"],
                key="sa_key_uploader",
                help="The JSON file downloaded from GCP when you created the service account key.",
            )
            if uploaded_key is not None:
                try:
                    key_source = json.loads(uploaded_key.read().decode("utf-8"))
                    if "client_email" not in key_source:
                        st.error("This doesn't look like a valid service account key JSON.")
                        return
                    st.success(f"Key loaded for: `{key_source['client_email']}`")
                except Exception as exc:
                    st.error(f"Could not parse JSON key: {exc}")
                    return
            else:
                st.info(
                    "Upload your service account JSON key above, or add it to "
                    "`.streamlit/secrets.toml` to avoid uploading it each session."
                )
                # Still show status if data is already loaded from a previous run
                if st.session_state.pdf_df is not None or st.session_state.csv_df is not None:
                    _show_loaded_status()
                return

        if not folder_id:
            st.warning("Enter a folder ID to continue.")
            return

        # Preview folder contents (non-destructive)
        if st.button("Preview folder contents", key="btn_preview_drive"):
            with st.spinner("Connecting to Google Drive…"):
                try:
                    files = list_folder_files(key_source, folder_id)
                    st.session_state["_drive_manifest"] = files
                except Exception as exc:
                    st.error(f"Could not list folder: {exc}")
                    st.session_state.pop("_drive_manifest", None)

        manifest = st.session_state.get("_drive_manifest")
        if manifest:
            pdf_count    = sum(1 for f in manifest if f.category == "pdf_like")
            sheet_count  = sum(1 for f in manifest if f.category == "sheet_like")
            skip_count   = sum(1 for f in manifest if f.category == "skipped")

            _CAT_LABEL = {"pdf_like": "Document (PDF/Word/Doc)", "sheet_like": "Tabular (CSV/Excel/Sheet)", "skipped": "Skipped"}
            st.dataframe(
                pd.DataFrame([{
                    "File name": f.name,
                    "Processing": _CAT_LABEL.get(f.category, f.category),
                    "MIME": f.mime_type,
                    "Size": f"{f.size_bytes / 1024:.1f} KB" if f.size_bytes else "—",
                } for f in manifest]),
                hide_index=True,
                use_container_width=True,
            )
            st.caption(
                f"Will process: {pdf_count} document(s) as crash report, "
                f"{sheet_count} tabular file(s) as MCMIS data, "
                f"{skip_count} file(s) skipped (unsupported format)"
            )

            if pdf_count == 0 and sheet_count == 0:
                st.error("No processable files found in this folder.")
                return

            if st.button("Load & process all files", type="primary", key="btn_load_drive"):
                progress = st.progress(0, text="Connecting to Drive…")
                try:
                    pdf_files, sheet_files, _ = load_drive_folder(key_source, folder_id)
                    total = max(len(pdf_files) + len(sheet_files), 1)
                    done = 0

                    all_pdf_records = []
                    for fname, raw, mime in pdf_files:
                        progress.progress(done / total, text=f"Extracting {fname}…")
                        try:
                            if mime in _DOCX_MIMES:
                                df = extract_docx_records(raw, fname)
                            elif mime in _PLAINTEXT_MIMES or mime == "application/pdf":
                                if mime in _PLAINTEXT_MIMES:
                                    df = extract_plaintext_records(raw, fname)
                                else:
                                    df = extract_pdf_records(io.BytesIO(raw), fname)
                            else:
                                df = extract_pdf_records(io.BytesIO(raw), fname)
                            if not df.empty:
                                all_pdf_records.append(df)
                        except Exception as file_exc:
                            st.warning(f"Could not extract {fname}: {file_exc}")
                        done += 1

                    all_csv_frames = []
                    for fname, raw, mime in sheet_files:
                        progress.progress(done / total, text=f"Loading {fname}…")
                        try:
                            df = load_tabular(raw, mime, fname)
                            if not df.empty:
                                all_csv_frames.append(df)
                        except Exception as file_exc:
                            st.warning(f"Could not load {fname}: {file_exc}")
                        done += 1

                    progress.progress(1.0, text="Done.")

                    if all_pdf_records:
                        st.session_state.pdf_df = pd.concat(all_pdf_records, ignore_index=True)
                        st.session_state.last_pdf_name = ", ".join(n for n, _, __ in pdf_files)

                    if all_csv_frames:
                        combined_csv = pd.concat(all_csv_frames, ignore_index=True)
                        st.session_state.csv_df = combined_csv
                        st.session_state.csv_prep = prepare_csv_records(combined_csv)
                        st.session_state.last_csv_name = ", ".join(n for n, _, __ in sheet_files)

                    st.session_state.matching_ran = False

                except Exception as exc:
                    st.error(f"Load failed: {exc}")
                    st.exception(exc)
                    return

        _show_loaded_status()

    # -----------------------------------------------------------------------
    # LOCAL UPLOAD MODE (fallback — unchanged)
    # -----------------------------------------------------------------------
    else:
        col_pdf, col_csv = st.columns(2)

        with col_pdf:
            st.subheader("Crash Report PDF")
            pdf_upload = st.file_uploader(
                "Upload PDF crash report",
                type=["pdf"],
                key="pdf_uploader",
                help="Text-based PDF. Scanned images require OCR (not yet supported).",
            )
            if pdf_upload is not None and pdf_upload.name != st.session_state.last_pdf_name:
                with st.spinner("Extracting data from PDF…"):
                    try:
                        df = extract_pdf_records(pdf_upload, pdf_upload.name)
                        st.session_state.pdf_df = df
                        st.session_state.last_pdf_name = pdf_upload.name
                        st.session_state.matching_ran = False
                    except Exception as exc:
                        st.error(f"PDF extraction failed: {exc}")

            if st.session_state.pdf_df is not None:
                n = len(st.session_state.pdf_df)
                st.success(f"**{st.session_state.last_pdf_name}** — {n} record(s) extracted")
                _show_pdf_coverage()
            else:
                st.info("No PDF uploaded yet.")

        with col_csv:
            st.subheader("MCMIS Crash CSV")
            csv_upload = st.file_uploader("Upload MCMIS CSV", type=["csv"], key="csv_uploader")
            if csv_upload is not None and csv_upload.name != st.session_state.last_csv_name:
                with st.spinner("Loading CSV…"):
                    try:
                        df = load_csv(csv_upload)
                        st.session_state.csv_df = df
                        st.session_state.csv_prep = prepare_csv_records(df)
                        st.session_state.last_csv_name = csv_upload.name
                        st.session_state.matching_ran = False
                    except Exception as exc:
                        st.error(f"CSV load failed: {exc}")

            if st.session_state.csv_df is not None:
                n = len(st.session_state.csv_df)
                st.success(f"**{st.session_state.last_csv_name}** — {n} rows loaded")
            else:
                st.info("No CSV uploaded yet.")


def _tab_preview() -> None:
    st.header("Preview Extracted Data")

    if st.session_state.pdf_df is None and st.session_state.csv_df is None:
        st.info("Upload files on the **Upload Files** tab first.")
        return

    if st.session_state.pdf_df is not None:
        st.subheader("PDF — Extracted Records")
        display_cols = [
            'pdf_record_id', 'pdf_page_number', 'crash_report_number', 'crash_date',
            'vehicle_number', 'driver_name', 'carrier_name', 'carrier_address_raw',
            'carrier_zip', 'owner_name', 'owner_address_raw', 'owner_zip',
            'usdot_number', 'vin', 'vehicle_plate_number', 'vehicle_plate_state',
            'vehicle_make', 'vehicle_year', 'insurance_company',
        ]
        visible = [c for c in display_cols if c in st.session_state.pdf_df.columns]
        st.dataframe(st.session_state.pdf_df[visible], use_container_width=True, hide_index=True)

        with st.expander("Raw text (first record)"):
            if 'raw_text' in st.session_state.pdf_df.columns:
                st.text(st.session_state.pdf_df.iloc[0]['raw_text'])

    if st.session_state.csv_df is not None:
        st.subheader("CSV — Preview (first 10 rows)")
        st.dataframe(st.session_state.csv_df.head(10), use_container_width=True, hide_index=True)

        st.subheader("Detected CSV Columns")
        expected = [
            'Crash ID', 'Crash Date', 'Carrier Name', 'USDOT Number',
            'Carrier Street', 'Carrier City', 'Carrier State', 'Carrier Zip Code',
            'Vehicle License Number', 'Vehicle License State', 'County Name',
        ]
        cols = list(st.session_state.csv_df.columns)
        st.dataframe(
            pd.DataFrame({'Column': expected, 'Present': ['✓' if c in cols else '✗' for c in expected]}),
            hide_index=True,
            use_container_width=False,
        )
        other = [c for c in cols if c not in expected]
        if other:
            st.caption(f"Other columns present: {', '.join(other)}")

    if st.session_state.csv_prep is not None:
        with st.expander("Normalized CSV carrier addresses (sample)"):
            norm_cols = [c for c in [
                'csv_carrier_address_raw', 'csv_carrier_address_normalized',
                'Carrier Name', 'csv_carrier_name_normalized', 'csv_usdot_clean',
            ] if c in st.session_state.csv_prep.columns]
            st.dataframe(
                st.session_state.csv_prep[norm_cols].head(10),
                use_container_width=True,
                hide_index=True,
            )


def _tab_configure() -> None:
    st.header("Matching Configuration")

    preset_name = st.selectbox("Preset", options=list(PRESETS.keys()), index=0, key="preset_select")
    preset = PRESETS[preset_name]
    st.info(preset['description'])

    st.markdown("#### Active Configuration")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**PDF matching field:** `{preset['pdf_field']}`")
        st.markdown("**Fallback:** `owner_address_raw` when carrier address is blank")
        st.markdown(f"**Method:** {preset['match_method']}")
    with c2:
        st.markdown(f"**CSV fields:** {', '.join(preset['csv_fields'])}")
        st.markdown(f"**Default threshold:** {preset['threshold']}")

    st.markdown("---")
    threshold_override = st.slider(
        "Minimum match score threshold",
        min_value=50, max_value=100, value=preset['threshold'], step=5,
        help="Pairs scoring below this are excluded from results.",
    )

    with st.expander("Manual field override"):
        st.caption("Override the preset if your files use different field structures.")
        pdf_fields = (
            list(st.session_state.pdf_df.columns)
            if st.session_state.pdf_df is not None
            else ['carrier_address_raw', 'owner_address_raw', 'carrier_name', 'usdot_number']
        )
        csv_fields = (
            list(st.session_state.csv_df.columns)
            if st.session_state.csv_df is not None
            else ['Carrier Street', 'Carrier City', 'Carrier State', 'Carrier Zip Code']
        )
        manual_pdf = st.selectbox(
            "PDF field",
            options=pdf_fields,
            index=pdf_fields.index('carrier_address_raw') if 'carrier_address_raw' in pdf_fields else 0,
        )
        manual_csv = st.multiselect(
            "CSV fields to combine",
            options=csv_fields,
            default=[f for f in preset['csv_fields'] if f in csv_fields],
        )
        manual_method = st.selectbox(
            "Match method",
            options=['Fuzzy Address', 'Exact', 'Fuzzy Text', 'Multi-Field'],
        )
        use_manual = st.checkbox("Use manual override instead of preset")

    st.session_state.match_config = {
        'preset': preset_name,
        'pdf_field': manual_pdf if use_manual else preset['pdf_field'],
        'csv_fields': manual_csv if use_manual else preset['csv_fields'],
        'match_method': manual_method if use_manual else preset['match_method'],
        'threshold': threshold_override,
        'use_manual': use_manual,
    }

    st.markdown("---")
    st.markdown("**Confidence level guide**")
    st.dataframe(
        pd.DataFrame([
            {'Score': '95 – 100', 'Confidence': 'Very High', 'Meaning': 'Exact or near-exact normalized address'},
            {'Score': '85 – 94',  'Confidence': 'High',      'Meaning': 'Very strong multi-component similarity'},
            {'Score': '75 – 84',  'Confidence': 'Medium',    'Meaning': 'Good similarity; worth investigating'},
            {'Score': '60 – 74',  'Confidence': 'Low',       'Meaning': 'Partial match; treat as weak lead'},
            {'Score': 'below 60', 'Confidence': 'Ignore',    'Meaning': 'Excluded from results at default threshold'},
        ]),
        hide_index=True,
        use_container_width=False,
    )


def _tab_run() -> None:
    st.header("Run Matching")

    ready = (
        st.session_state.pdf_df is not None
        and st.session_state.csv_df is not None
        and st.session_state.csv_prep is not None
    )

    if not ready:
        st.warning("Upload both a PDF and a CSV on the **Upload Files** tab before running.")
        return

    cfg = st.session_state.match_config
    st.markdown("**Configuration summary**")
    st.markdown(f"- Preset: **{cfg['preset']}**")
    st.markdown(f"- PDF field: `{cfg['pdf_field']}` (owner address fallback enabled)")
    st.markdown(f"- CSV fields: `{', '.join(cfg['csv_fields'])}`")
    st.markdown(f"- Method: **{cfg['match_method']}**")
    st.markdown(f"- Minimum score threshold: **{cfg['threshold']}**")
    st.markdown("---")

    n_pdf = len(st.session_state.pdf_df)
    n_csv = len(st.session_state.csv_df)
    st.markdown(
        f"Evaluating **{n_pdf}** PDF record(s) × **{n_csv}** CSV rows "
        f"= up to **{n_pdf * n_csv:,}** address pairs."
    )

    if st.button("Run Address Matching", type="primary"):
        with st.spinner("Scoring address pairs…"):
            try:
                matches = run_matching(
                    st.session_state.pdf_df,
                    st.session_state.csv_df,
                    threshold=float(cfg['threshold']),
                )
                clusters = compute_address_clusters(st.session_state.csv_prep)
                st.session_state.matches_df = matches
                st.session_state.clusters_df = clusters
                st.session_state.matching_ran = True
            except Exception as exc:
                st.error(f"Matching error: {exc}")
                st.exception(exc)

    if st.session_state.matching_ran and st.session_state.matches_df is not None:
        m = st.session_state.matches_df
        st.success(f"Done — **{len(m)}** match(es) at threshold {cfg['threshold']}.")

        if not m.empty and 'confidence' in m.columns:
            counts = m['confidence'].value_counts().reindex(
                ['Very High', 'High', 'Medium', 'Low'], fill_value=0
            )
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Very High", int(counts['Very High']))
            c2.metric("High", int(counts['High']))
            c3.metric("Medium", int(counts['Medium']))
            c4.metric("Low", int(counts['Low']))

        if 'review_needed' in m.columns:
            n_review = int(m['review_needed'].sum())
            if n_review:
                st.warning(f"{n_review} match(es) flagged for manual review.")

        cl = st.session_state.clusters_df
        if cl is not None and not cl.empty:
            n_flag = int(cl['review_flag'].sum()) if 'review_flag' in cl.columns else 0
            st.info(
                f"Address clusters: {len(cl)} unique normalized addresses — "
                f"{n_flag} flagged (multiple carriers or DOTs at same address)."
            )


def _tab_results() -> None:
    st.header("Results")

    if not st.session_state.matching_ran or st.session_state.matches_df is None:
        st.info("Run matching on the **Run Matching** tab to see results.")
        return

    m = st.session_state.matches_df
    if m.empty:
        st.warning("No matches above threshold. Try lowering the threshold in **Configure Matching**.")
        return

    # Filters
    st.subheader("Filters")
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        conf_filter = st.multiselect(
            "Confidence",
            options=['Very High', 'High', 'Medium', 'Low'],
            default=['Very High', 'High', 'Medium', 'Low'],
            key="conf_filter",
        )
    with fc2:
        review_filter = st.selectbox(
            "Review flag",
            options=['All', 'Review needed only', 'No review flag'],
            key="review_filter",
        )
    with fc3:
        min_score = st.number_input(
            "Min score",
            min_value=0, max_value=100,
            value=int(st.session_state.match_config.get('threshold', 75)),
            step=5, key="min_score_filter",
        )

    view = m[m['confidence'].isin(conf_filter)] if conf_filter else m.copy()
    if review_filter == 'Review needed only' and 'review_needed' in view.columns:
        view = view[view['review_needed'] == True]  # noqa: E712
    elif review_filter == 'No review flag' and 'review_needed' in view.columns:
        view = view[view['review_needed'] == False]  # noqa: E712
    if 'match_score' in view.columns:
        view = view[view['match_score'] >= min_score]

    st.markdown(f"Showing **{len(view)}** of {len(m)} match(es)")

    col_order = [
        'match_id', 'confidence', 'match_score', 'match_reason',
        'pdf_carrier_name', 'pdf_carrier_address', 'pdf_owner_name', 'pdf_owner_address',
        'pdf_vin', 'pdf_usdot', 'pdf_plate', 'pdf_driver_name', 'pdf_crash_report_number',
        'csv_carrier_name', 'csv_carrier_address', 'csv_usdot', 'csv_vehicle_plate',
        'csv_crash_id', 'csv_crash_date', 'csv_county',
        'same_or_similar_address', 'different_carrier_name', 'different_usdot',
        'multiple_csv_carriers_at_address', 'multiple_csv_dots_at_address',
        'pdf_vin_available', 'pdf_person_name_available',
        'repeated_csv_crash_history_at_address', 'review_needed', 'review_notes',
        'match_type', 'pdf_record_id', 'pdf_page_number', 'csv_row_id',
    ]
    ordered = [c for c in col_order if c in view.columns]
    rest = [c for c in view.columns if c not in ordered]

    st.dataframe(
        view[ordered + rest],
        use_container_width=True,
        hide_index=True,
        column_config={
            'match_score': st.column_config.ProgressColumn('Score', min_value=0, max_value=100, format="%d"),
            'confidence': st.column_config.TextColumn('Confidence'),
            'review_needed': st.column_config.CheckboxColumn('Review?'),
            'same_or_similar_address': st.column_config.CheckboxColumn('Same addr'),
            'different_carrier_name': st.column_config.CheckboxColumn('Diff carrier'),
            'different_usdot': st.column_config.CheckboxColumn('Diff DOT'),
            'pdf_vin_available': st.column_config.CheckboxColumn('VIN avail'),
            'multiple_csv_carriers_at_address': st.column_config.CheckboxColumn('Multi carrier'),
            'multiple_csv_dots_at_address': st.column_config.CheckboxColumn('Multi DOT'),
            'repeated_csv_crash_history_at_address': st.column_config.CheckboxColumn('Repeat crash'),
        },
    )

    cl = st.session_state.clusters_df
    if cl is not None and not cl.empty:
        st.markdown("---")
        st.subheader("CSV Address Clusters")
        st.caption(
            "Groups of CSV crash records at the same normalized carrier address. "
            "Flagged clusters (🚩) have >1 unique carrier name or USDOT number. "
            "These are investigative leads, not evidence of wrongdoing."
        )
        show_flagged = st.checkbox("Show flagged clusters only", value=False)
        cl_view = cl[cl['review_flag'] == True] if show_flagged else cl  # noqa: E712
        st.dataframe(
            cl_view,
            use_container_width=True,
            hide_index=True,
            column_config={
                'review_flag': st.column_config.CheckboxColumn('🚩 Flag'),
                'crash_count': st.column_config.NumberColumn('# Crashes'),
                'unique_carrier_names': st.column_config.NumberColumn('Unique carriers'),
                'unique_usdot_numbers': st.column_config.NumberColumn('Unique DOTs'),
            },
        )


def _tab_export() -> None:
    st.header("Export Results")

    if not st.session_state.matching_ran or st.session_state.matches_df is None:
        st.info("Run matching on the **Run Matching** tab to enable export.")
        return

    m = st.session_state.matches_df
    cl = st.session_state.clusters_df
    cfg = st.session_state.match_config

    st.markdown(f"Ready to export **{len(m)}** match record(s).")
    st.markdown(DISCLAIMER_HTML, unsafe_allow_html=True)

    ec1, ec2 = st.columns(2)

    with ec1:
        st.subheader("CSV Export")
        st.caption("Flat file — all matches.")
        st.download_button(
            "⬇ Download matches.csv",
            data=to_csv_bytes(m),
            file_name="matches.csv",
            mime="text/csv",
        )
        if cl is not None and not cl.empty:
            st.download_button(
                "⬇ Download address_clusters.csv",
                data=to_csv_bytes(cl),
                file_name="address_clusters.csv",
                mime="text/csv",
            )

    with ec2:
        st.subheader("Excel Export")
        st.caption(
            "8-tab workbook: All Matches · Very High / High / Medium / Low Confidence · "
            "Address Clusters · Match Settings · Extraction Notes."
        )
        xlsx = to_excel_bytes(
            matches_df=m,
            clusters_df=cl,
            match_settings=cfg,
            extraction_notes=(
                f"PDF: {st.session_state.last_pdf_name}\n"
                f"CSV: {st.session_state.last_csv_name}\n"
                f"PDF records: {len(st.session_state.pdf_df)}\n"
                f"CSV rows: {len(st.session_state.csv_df)}\n"
                f"Matches (threshold {cfg['threshold']}): {len(m)}"
            ),
        )
        st.download_button(
            "⬇ Download matches.xlsx",
            data=xlsx,
            file_name="matches.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    with st.expander("Output field legend"):
        st.dataframe(
            pd.DataFrame([
                ('match_id',                         'Sequential ID for this pair'),
                ('match_type',                       'fuzzy_address (carrier) or fuzzy_address (owner fallback)'),
                ('confidence',                       'Very High / High / Medium / Low'),
                ('match_score',                      '0–100 composite address similarity score'),
                ('match_reason',                     'Human-readable explanation'),
                ('same_or_similar_address',          'True when score ≥ 75'),
                ('different_carrier_name',           'PDF and CSV carrier names differ after normalization'),
                ('different_usdot',                  'PDF and CSV USDOT numbers differ'),
                ('multiple_csv_carriers_at_address', '>1 CSV carrier name at this address'),
                ('multiple_csv_dots_at_address',     '>1 USDOT at this address'),
                ('pdf_vin_available',                'PDF record contains a VIN'),
                ('pdf_person_name_available',        'PDF record contains a driver name'),
                ('repeated_csv_crash_history_at_address', '3+ CSV crashes at this address'),
                ('review_needed',                    'Manual review recommended'),
                ('review_notes',                     'Specific flags triggering review'),
            ], columns=['Field', 'Description']),
            hide_index=True,
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# RAG / Ask tab
# ---------------------------------------------------------------------------

_RAG_DISCLAIMER = (
    "RAG answers are generated from indexed project evidence. "
    "They are investigative summaries, not legal or regulatory conclusions."
)

_CHUNK_TYPE_OPTIONS = {
    "All":              None,
    "PDF pages":        "pdf_page",
    "PDF records":      "pdf_record",
    "CSV rows":         "csv_row",
    "Match results":    "match_result",
    "Address clusters": "address_cluster",
}

_SUGGESTED_QUESTIONS = [
    "Why was the top match flagged?",
    "Which matches have different carrier names and different DOT numbers?",
    "Which addresses have multiple carriers?",
    "Which addresses have multiple DOT numbers?",
    "Summarize the strongest high-confidence leads.",
    "Which matches need manual review?",
    "Show evidence for the top address cluster.",
]


def _tab_rag() -> None:
    st.header("Ask / RAG Evidence")
    st.caption(_RAG_DISCLAIMER)

    # -----------------------------------------------------------------------
    # Project ID
    # -----------------------------------------------------------------------
    project_id = st.text_input(
        "Project ID",
        value="default_project",
        key="rag_project_id",
        help="Namespace for the ChromaDB vector index. Change this to isolate separate projects.",
    )

    # -----------------------------------------------------------------------
    # Current index status
    # -----------------------------------------------------------------------
    n_chunks = collection_count(project_id)
    if n_chunks > 0:
        st.success(f"Index ready — **{n_chunks} chunks** in project `{project_id}`.")
    else:
        st.warning("Index is empty. Click **Build / Refresh RAG Index** below.")

    # -----------------------------------------------------------------------
    # Build index button
    # -----------------------------------------------------------------------
    if st.button("Build / Refresh RAG Index", key="btn_build_rag"):
        # Map current session_state DataFrames to the pipeline's expected args.
        # pdf_df serves as both pdf_records_df and a proxy for pdf_pages_df
        # (each record row contains raw_text).
        pdf_df    = st.session_state.get("pdf_df")
        csv_df    = st.session_state.get("csv_df")
        matches   = st.session_state.get("matches_df")
        clusters  = st.session_state.get("clusters_df")

        has_data = any(
            df is not None and not (hasattr(df, "empty") and df.empty)
            for df in [pdf_df, csv_df, matches, clusters]
        )
        if not has_data:
            st.error("No data loaded yet. Upload files and run matching first.")
        else:
            with st.spinner("Building RAG index — this may take a moment…"):
                try:
                    summary = build_project_index(
                        project_id=project_id,
                        pdf_pages_df=pdf_df,      # raw_text column works as page text
                        pdf_records_df=pdf_df,
                        csv_df=csv_df,
                        matches_df=matches,
                        clusters_df=clusters,
                        reset_existing=True,
                    )
                    st.success(
                        f"Index built — **{summary['chunks_indexed']} chunks** indexed."
                    )
                    type_counts = summary.get("chunk_type_counts", {})
                    if type_counts:
                        st.dataframe(
                            pd.DataFrame(
                                [{"Chunk type": k, "Count": v}
                                 for k, v in type_counts.items()]
                            ),
                            hide_index=True,
                            use_container_width=False,
                        )
                except Exception as exc:
                    st.error(f"Index build failed: {exc}")
                    st.exception(exc)

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Question input + suggested questions
    # -----------------------------------------------------------------------
    st.subheader("Ask a question")

    # Suggested question buttons MUST appear before the text_area so that
    # when one is clicked we can write to the widget key before it renders.
    # We stage the choice in "_rag_staged_q" on the first rerun, then move
    # it into the actual widget key before the text_area is instantiated.
    if "_rag_staged_q" in st.session_state:
        st.session_state["rag_question_input"] = st.session_state.pop("_rag_staged_q")

    st.markdown("**Suggested questions**")
    cols = st.columns(3)
    for i, q in enumerate(_SUGGESTED_QUESTIONS):
        if cols[i % 3].button(q, key=f"sq_{i}"):
            st.session_state["_rag_staged_q"] = q
            st.rerun()

    question = st.text_area(
        "Ask a question about the uploaded documents, matches, or address clusters",
        height=80,
        key="rag_question_input",
    )

    # -----------------------------------------------------------------------
    # Search options
    # -----------------------------------------------------------------------
    opt_col1, opt_col2 = st.columns([2, 1])
    with opt_col1:
        chunk_type_label = st.selectbox(
            "Filter by chunk type",
            options=list(_CHUNK_TYPE_OPTIONS.keys()),
            index=0,
            key="rag_chunk_type",
        )
    with opt_col2:
        top_k = st.slider("Top K results", min_value=3, max_value=20, value=8, key="rag_top_k")

    chunk_type_filter = _CHUNK_TYPE_OPTIONS[chunk_type_label]

    # -----------------------------------------------------------------------
    # Ask button
    # -----------------------------------------------------------------------
    if st.button("Ask", type="primary", key="btn_ask_rag"):
        q = st.session_state.get("rag_question_input", "").strip()
        if not q:
            st.warning("Enter a question above.")
        elif n_chunks == 0:
            st.error("The index is empty. Build the index first.")
        else:
            with st.spinner("Retrieving evidence and generating answer…"):
                try:
                    result = ask_project_question(
                        project_id=project_id,
                        question=q,
                        top_k=top_k,
                        chunk_type_filter=chunk_type_filter,
                    )
                    st.session_state["_rag_last_result"] = result
                except Exception as exc:
                    st.error(f"RAG query failed: {exc}")
                    st.exception(exc)

    # -----------------------------------------------------------------------
    # Display answer + evidence table
    # -----------------------------------------------------------------------
    last = st.session_state.get("_rag_last_result")
    if last:
        st.markdown("### Answer")
        st.markdown(last["answer"])

        context_chunks = last.get("context_chunks", [])
        if context_chunks:
            st.markdown("### Retrieved evidence")
            rows = []
            for c in context_chunks:
                meta = c.get("metadata", {})
                rows.append({
                    "chunk_type":   meta.get("chunk_type", ""),
                    "source_file":  meta.get("source_file", ""),
                    "page_number":  meta.get("page_number", ""),
                    "row_id":       meta.get("row_id") or meta.get("csv_row_id", ""),
                    "match_id":     meta.get("match_id", ""),
                    "carrier_name": meta.get("carrier_name", ""),
                    "address":      meta.get("address", ""),
                    "score":        c.get("score", ""),
                    "text_preview": c.get("text", "")[:200].replace("\n", " "),
                })
            st.dataframe(
                pd.DataFrame(rows),
                hide_index=True,
                use_container_width=True,
            )

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Explain a specific match
    # -----------------------------------------------------------------------
    st.subheader("Explain a specific match")

    matches_df = st.session_state.get("matches_df")
    if matches_df is None or (hasattr(matches_df, "empty") and matches_df.empty):
        st.info("Run matching to enable per-match explanation.")
    else:
        match_ids = matches_df["match_id"].tolist() if "match_id" in matches_df.columns else []
        if match_ids:
            selected_mid = st.selectbox(
                "Select match ID",
                options=match_ids,
                key="rag_explain_match_id",
            )
            if st.button("Explain Selected Match", key="btn_explain_match"):
                row = matches_df[matches_df["match_id"] == selected_mid].iloc[0].to_dict()
                with st.spinner("Generating explanation…"):
                    try:
                        result = explain_match(
                            project_id=project_id,
                            match_id=selected_mid,
                            match_row=row,
                        )
                        st.session_state["_rag_explain_result"] = result
                    except Exception as exc:
                        st.error(f"Explain failed: {exc}")
                        st.exception(exc)

            explain_result = st.session_state.get("_rag_explain_result")
            if explain_result:
                st.markdown("### Explanation")
                st.markdown(explain_result["answer"])

                ev = explain_result.get("context_chunks", [])
                if ev:
                    with st.expander("Source evidence chunks"):
                        for i, c in enumerate(ev, 1):
                            meta = c.get("metadata", {})
                            st.markdown(
                                f"**Chunk {i}** — type: `{meta.get('chunk_type','')}` | "
                                f"score: {c.get('score','')} | "
                                f"match_id: {meta.get('match_id','')} | "
                                f"address: {meta.get('address','')}"
                            )
                            st.text(c.get("text", "")[:500])
                            st.markdown("---")


# ---------------------------------------------------------------------------
# Main entry point — only runs inside a live Streamlit server
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Document Matching",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(_CSS, unsafe_allow_html=True)
    _init_state()

    st.title("Document Matching")
    st.caption("PDF Crash Report  →  MCMIS CSV  |  Address-Based Matching")
    st.markdown(DISCLAIMER_HTML, unsafe_allow_html=True)

    tabs = st.tabs([
        "1 · Upload Files",
        "2 · Preview Data",
        "3 · Configure Matching",
        "4 · Run Matching",
        "5 · Results",
        "6 · Export",
        "7 · Ask / RAG Evidence",
    ])

    with tabs[0]: _tab_upload()
    with tabs[1]: _tab_preview()
    with tabs[2]: _tab_configure()
    with tabs[3]: _tab_run()
    with tabs[4]: _tab_results()
    with tabs[5]: _tab_export()
    with tabs[6]: _tab_rag()


if _st_runtime.exists():
    main()
