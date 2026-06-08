from difflib import SequenceMatcher
from .normalizers import (
    extract_house_number,
    extract_street_name_only,
    normalize_zip,
    is_po_box,
)

CONFIDENCE_THRESHOLDS = [
    (95, 'Very High'),
    (85, 'High'),
    (75, 'Medium'),
    (60, 'Low'),
]


def get_confidence(score: float) -> str:
    for threshold, label in CONFIDENCE_THRESHOLDS:
        if score >= threshold:
            return label
    return 'Ignore'


def _sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def score_address_pair(
    pdf_addr_norm: str,
    pdf_zip: str,
    csv_street_norm: str,
    csv_city: str,
    csv_state: str,
    csv_zip: str,
) -> dict:
    """
    Score a normalized PDF address against CSV address components.

    Weights:
      house_number   20 pts
      street_name    40 pts
      city           15 pts
      state          10 pts
      zip            15 pts
    """
    score = 0

    pdf_addr = pdf_addr_norm.upper().strip()
    csv_street = csv_street_norm.upper().strip()
    csv_city_u = csv_city.upper().strip()
    csv_state_u = csv_state.upper().strip()
    pdf_zip_n = normalize_zip(pdf_zip)
    csv_zip_n = normalize_zip(csv_zip)

    # House number (20 pts)
    pdf_house = extract_house_number(pdf_addr)
    csv_house = extract_house_number(csv_street)
    house_match = bool(pdf_house and csv_house and pdf_house == csv_house)
    if house_match:
        score += 20

    # Street name similarity (40 pts)
    pdf_street_name = extract_street_name_only(pdf_addr)
    csv_street_name = extract_street_name_only(csv_street)
    street_sim = _sim(pdf_street_name, csv_street_name)
    score += int(street_sim * 40)

    # City match (15 pts)
    city_match = False
    if csv_city_u:
        if csv_city_u in pdf_addr:
            city_match = True
            score += 15
        else:
            city_words = [w for w in csv_city_u.split() if len(w) > 3]
            if city_words and all(w in pdf_addr for w in city_words):
                city_match = True
                score += 8

    # State match (10 pts)
    state_match = False
    if csv_state_u:
        pdf_words = set(pdf_addr.split())
        if csv_state_u in pdf_words:
            state_match = True
            score += 10
        else:
            score -= 5

    # ZIP match (15 pts)
    zip_match = False
    if pdf_zip_n and csv_zip_n:
        if pdf_zip_n == csv_zip_n:
            zip_match = True
            score += 15
        elif pdf_zip_n[:3] == csv_zip_n[:3]:
            score += 5

    # Boost when house + zip both confirmed
    if house_match and zip_match:
        score = min(100, score + 5)

    # Hard rule: city/state only without street or zip evidence is not a real match
    if not house_match and street_sim < 0.4 and not zip_match:
        score = min(30, score)

    score = max(0, min(100, score))

    # PO Box
    po_box = is_po_box(pdf_addr_norm)
    review_needed = po_box

    # Build match reason string
    if score >= 95:
        reason = "Exact normalized address match"
    elif house_match and zip_match and street_sim >= 0.65:
        reason = "Same house number and ZIP with similar street name"
    elif street_sim >= 0.75 and city_match and state_match:
        reason = "Similar street address with same city/state/ZIP"
    elif house_match and street_sim >= 0.65:
        reason = "Same house number and similar street name"
    elif zip_match and street_sim >= 0.55:
        reason = "Same ZIP with similar street address"
    elif score >= 60:
        reason = "Partial address similarity"
    else:
        reason = "Low address similarity"

    if po_box:
        reason += "; PO Box address — manual review required"

    return {
        'score': score,
        'confidence': get_confidence(score),
        'house_match': house_match,
        'street_similarity': round(street_sim, 3),
        'city_match': city_match,
        'state_match': state_match,
        'zip_match': zip_match,
        'po_box': po_box,
        'review_needed': review_needed,
        'match_reason': reason,
    }
