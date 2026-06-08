import io
import pandas as pd

# MIME types that should be loaded as Excel workbooks
_EXCEL_MIMES = {
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",   # .xlsx
    "application/vnd.ms-excel",                                              # .xls
    "application/vnd.oasis.opendocument.spreadsheet",                        # .ods
}


def load_csv(file) -> pd.DataFrame:
    """Load a CSV from an uploaded Streamlit file object or a file path."""
    if hasattr(file, 'read'):
        content = file.read()
        try:
            return pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False)
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(content), encoding='latin-1', dtype=str, keep_default_na=False)
    return pd.read_csv(file, dtype=str, keep_default_na=False)


def load_excel(file_bytes: bytes, filename: str = '') -> pd.DataFrame:
    """
    Load an Excel (.xlsx / .xls / .ods) workbook from raw bytes.

    If the workbook has multiple sheets, all sheets are concatenated with a
    'source_sheet' column added so rows stay traceable.
    """
    buf = io.BytesIO(file_bytes)
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''

    engine = 'xlrd' if ext == 'xls' else ('odf' if ext == 'ods' else 'openpyxl')

    try:
        sheets: dict = pd.read_excel(buf, sheet_name=None, dtype=str,
                                     keep_default_na=False, engine=engine)
    except Exception:
        # Fallback: let pandas auto-detect engine
        buf.seek(0)
        sheets = pd.read_excel(buf, sheet_name=None, dtype=str, keep_default_na=False)

    if not sheets:
        return pd.DataFrame()

    frames = []
    for sheet_name, df in sheets.items():
        if df.empty:
            continue
        df = df.copy()
        df['source_sheet'] = sheet_name
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def load_tabular(file_bytes: bytes, mime_type: str, filename: str = '') -> pd.DataFrame:
    """
    Route-based loader: pick CSV or Excel based on MIME type / filename.
    Always returns a DataFrame with string dtype columns.
    """
    if mime_type in _EXCEL_MIMES or filename.lower().endswith(('.xlsx', '.xls', '.ods')):
        return load_excel(file_bytes, filename)
    # Default: treat as CSV (covers text/csv, text/plain, tab-separated, Google Sheet export)
    try:
        return pd.read_csv(io.BytesIO(file_bytes), dtype=str, keep_default_na=False)
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(file_bytes), encoding='latin-1', dtype=str, keep_default_na=False)
    except Exception:
        # Try tab-separated as final fallback
        return pd.read_csv(io.BytesIO(file_bytes), sep='\t', dtype=str, keep_default_na=False)


def get_csv_preview(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    return df.head(n)


def detect_csv_columns(df: pd.DataFrame) -> list:
    return list(df.columns)
