"""
Google Drive folder loader — Service Account authentication.

Downloads every supported file from a Drive folder and classifies each
into one of three processing categories:

  "pdf_like"   — crash-report documents (PDF, Google Doc, Word .docx)
  "sheet_like" — tabular MCMIS data (CSV, plain text, Excel, Google Sheets)
  "skipped"    — everything else (images, audio, binaries, etc.)

Service account setup (one-time):
  1. GCP Console → IAM & Admin → Service Accounts → Create
  2. Download the JSON key (no project roles needed)
  3. In Google Drive, share the target folder with the service account's
     client_email (Viewer permission is sufficient)
  4. Set GOOGLE_SERVICE_ACCOUNT_JSON in .streamlit/secrets.toml to the
     relative path of the JSON key file
"""

import io
import json
import pathlib
from typing import NamedTuple

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

KNOWN_FOLDER_ID = "14Ae9SXKeUYBjkoQUeoH5RUAhy3K-FiMx"

# ---------------------------------------------------------------------------
# MIME → processing category
# ---------------------------------------------------------------------------

# Native Google formats that can be exported
_GDOC_TYPE    = "application/vnd.google-apps.document"
_GSHEET_TYPE  = "application/vnd.google-apps.spreadsheet"
_GSLIDE_TYPE  = "application/vnd.google-apps.presentation"

_PDF_LIKE_TYPES = {
    "application/pdf",
    # Word documents
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
    "application/msword",                                                         # .doc (legacy)
    # Rich-text / plain text that may contain crash report narrative
    "text/plain",
    "application/rtf",
    "text/rtf",
}

_SHEET_LIKE_TYPES = {
    "text/csv",
    "text/tab-separated-values",
    # Excel
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",   # .xlsx
    "application/vnd.ms-excel",                                              # .xls
    # OpenDocument
    "application/vnd.oasis.opendocument.spreadsheet",                        # .ods
}


def _classify(mime: str) -> str:
    if mime in _PDF_LIKE_TYPES or mime == "application/pdf":
        return "pdf_like"
    if mime in _SHEET_LIKE_TYPES:
        return "sheet_like"
    if mime == _GDOC_TYPE:
        return "pdf_like"   # export as plain text → crash-report extraction
    if mime == _GSHEET_TYPE:
        return "sheet_like"  # export as CSV
    if mime == _GSLIDE_TYPE:
        return "pdf_like"   # export text where possible
    return "skipped"


# ---------------------------------------------------------------------------
# Named tuple for file metadata
# ---------------------------------------------------------------------------

class DriveFile(NamedTuple):
    file_id:   str
    name:      str
    mime_type: str
    size_bytes: int
    category:  str   # "pdf_like" | "sheet_like" | "skipped"


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def _build_service(key_source):
    """
    Build a Drive API v3 service from:
    - a file-path string (ends in .json or exists as a path)
    - a raw JSON string (starts with '{')
    - a dict (already parsed)
    """
    if isinstance(key_source, dict):
        info = key_source
    elif isinstance(key_source, str):
        s = key_source.strip()
        if s.startswith("{"):
            info = json.loads(s)
        else:
            info = json.loads(pathlib.Path(s).read_text(encoding="utf-8"))
    else:
        raise TypeError(f"Unsupported key_source type: {type(key_source)}")

    creds = service_account.Credentials.from_service_account_info(info, scopes=_SCOPES)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


# ---------------------------------------------------------------------------
# Core API operations
# ---------------------------------------------------------------------------

def list_folder_files(key_source, folder_id: str) -> list[DriveFile]:
    """
    Return metadata for every non-trashed file directly inside *folder_id*.
    Sub-folders are not traversed.
    """
    service = _build_service(key_source)
    query = f"'{folder_id}' in parents and trashed = false"

    results: list[DriveFile] = []
    page_token = None

    while True:
        kwargs = dict(
            q=query,
            fields="nextPageToken, files(id, name, mimeType, size)",
            pageSize=200,
        )
        if page_token:
            kwargs["pageToken"] = page_token

        resp = service.files().list(**kwargs).execute()

        for item in resp.get("files", []):
            mime = item.get("mimeType", "")
            results.append(DriveFile(
                file_id=item["id"],
                name=item.get("name", ""),
                mime_type=mime,
                size_bytes=int(item.get("size", 0)),
                category=_classify(mime),
            ))

        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    return results


def _download_bytes(service, file_id: str, mime_type: str) -> bytes:
    """Download a Drive file's bytes, exporting Google Workspace types as needed."""
    buf = io.BytesIO()

    if mime_type == _GDOC_TYPE or mime_type == _GSLIDE_TYPE:
        req = service.files().export_media(fileId=file_id, mimeType="text/plain")
    elif mime_type == _GSHEET_TYPE:
        req = service.files().export_media(fileId=file_id, mimeType="text/csv")
    else:
        req = service.files().get_media(fileId=file_id)

    dl = MediaIoBaseDownload(buf, req, chunksize=8 * 1024 * 1024)
    done = False
    while not done:
        _, done = dl.next_chunk()

    return buf.getvalue()


# ---------------------------------------------------------------------------
# High-level loader
# ---------------------------------------------------------------------------

def load_drive_folder(
    key_source,
    folder_id: str = KNOWN_FOLDER_ID,
) -> tuple[list[tuple[str, bytes, str]], list[tuple[str, bytes, str]], list[DriveFile]]:
    """
    Download all supported files in *folder_id*.

    Returns
    -------
    pdf_like_files  : list of (filename, raw_bytes, mime_type)
    sheet_like_files: list of (filename, raw_bytes, mime_type)
    all_files       : full DriveFile list (for display / manifest)
    """
    all_files = list_folder_files(key_source, folder_id)
    service = _build_service(key_source)

    pdf_like: list[tuple[str, bytes, str]] = []
    sheet_like: list[tuple[str, bytes, str]] = []

    for f in all_files:
        if f.category == "skipped":
            continue
        raw = _download_bytes(service, f.file_id, f.mime_type)
        if f.category == "pdf_like":
            pdf_like.append((f.name, raw, f.mime_type))
        else:
            sheet_like.append((f.name, raw, f.mime_type))

    return pdf_like, sheet_like, all_files
