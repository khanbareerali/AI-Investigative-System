import re

STREET_ABBREVS = {
    'STREET': 'ST',
    'ROAD': 'RD',
    'AVENUE': 'AVE',
    'BOULEVARD': 'BLVD',
    'DRIVE': 'DR',
    'LANE': 'LN',
    'COURT': 'CT',
    'PLACE': 'PL',
    'PARKWAY': 'PKWY',
    'HIGHWAY': 'HWY',
    'NORTH': 'N',
    'SOUTH': 'S',
    'EAST': 'E',
    'WEST': 'W',
    'NORTHEAST': 'NE',
    'NORTHWEST': 'NW',
    'SOUTHEAST': 'SE',
    'SOUTHWEST': 'SW',
    'SUITE': 'STE',
    'APARTMENT': 'APT',
}

LEGAL_SUFFIXES = frozenset({
    'LLC', 'INC', 'INCORPORATED', 'CORP', 'CORPORATION',
    'CO', 'COMPANY', 'LTD', 'LIMITED', 'LP', 'LLP', 'PLC', 'DBA',
})

# Build sorted keys longest-first so longer forms replace before shorter substrings do
_ABBREV_SORTED = sorted(STREET_ABBREVS.keys(), key=len, reverse=True)


def _safe(v) -> str:
    if v is None:
        return ''
    try:
        import math
        if isinstance(v, float) and math.isnan(v):
            return ''
    except Exception:
        pass
    return str(v).strip()


def normalize_zip(z) -> str:
    digits = re.sub(r'\D', '', _safe(z))
    return digits[:5] if len(digits) >= 5 else digits


def normalize_address(address, zip_code='') -> str:
    addr = _safe(address).upper()
    addr = re.sub(r'[^\w\s]', ' ', addr)

    for full in _ABBREV_SORTED:
        addr = re.sub(r'\b' + full + r'\b', STREET_ABBREVS[full], addr)

    addr = ' '.join(addr.split())

    z = normalize_zip(zip_code)
    if z and z not in addr:
        addr = f"{addr} {z}".strip()

    return addr


def normalize_carrier_name(name) -> str:
    n = _safe(name).upper()
    n = re.sub(r'[^\w\s]', ' ', n)
    words = n.split()
    while words and words[-1] in LEGAL_SUFFIXES:
        words.pop()
    return ' '.join(words)


def normalize_vin(vin) -> str:
    return re.sub(r'[^A-Z0-9]', '', _safe(vin).upper())


def normalize_plate(plate) -> str:
    return re.sub(r'[^A-Z0-9]', '', _safe(plate).upper())


def is_po_box(address) -> bool:
    return bool(re.search(r'\bP\.?O\.?\s*BOX\b', _safe(address).upper()))


def extract_house_number(norm_addr: str) -> str:
    m = re.match(r'^(\d+[A-Z]?)\s', norm_addr.strip() + ' ')
    return m.group(1) if m else ''


def extract_street_name_only(norm_addr: str) -> str:
    s = re.sub(r'^\d+[A-Z]?\s+', '', norm_addr.strip())
    s = re.sub(r'\b(STE|APT|UNIT|#)\s*\S*', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\b\d{5}(-\d{4})?\b', '', s)
    return ' '.join(s.split())
