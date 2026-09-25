"""Country-agnostic text normalization for business names and addresses.

Design rules (so the pipeline generalizes to unseen countries like France):
- no hand-written abbreviation / legal-suffix / state lists; frequent tokens are
  down-weighted later by IDF, which is learned from the data itself
- Latin accents are stripped (e -> e for é), but Indic scripts are preserved intact
"""
import unicodedata
import pandas as pd

def _build_latin_table():
    """Map accented Latin characters (U+00C0..U+024F) to their plain ASCII base."""
    table = {}
    for cp in range(0x00C0, 0x0250):
        ch = chr(cp)
        base = "".join(c for c in unicodedata.normalize("NFKD", ch) if not unicodedata.combining(c))
        if base and base != ch and base.isascii():
            table[cp] = base
    # characters that NFKD does not decompose
    table.update({ord(k): v for k, v in {"ß": "ss", "æ": "ae", "Æ": "AE", "ø": "o", "Ø": "O",
                                          "œ": "oe", "Œ": "OE", "đ": "d", "Đ": "D", "ł": "l", "Ł": "L"}.items()})
    return table

_LATIN = _build_latin_table()
_INDIC = "\u0900-\u0DFF"  # Devanagari .. Sinhala blocks; vowel signs there must count as word chars

def normalize_series(s: pd.Series) -> pd.Series:
    """Vectorized normalization. Examples:
    'AF-0684, NANDGRAM'   -> 'af 684 nandgram'
    '1056c Belden Ave'    -> '1056 c belden ave'
    'Payne Énterprises'   -> 'payne enterprises'
    'KANSAS CITY, null'   -> 'kansas city'
    """
    s = s.fillna("").astype(str).str.translate(_LATIN).str.lower()
    s = s.str.replace("&", " and ", regex=False)
    # split digit/letter boundaries: 1056c -> 1056 c, 45nd -> 45 nd
    s = s.str.replace(r"(?<=\d)(?=[^\W\d_])|(?<=[^\W\d_])(?=\d)", " ", regex=True)
    # punctuation -> space, keeping Indic letters and their combining vowel signs
    s = s.str.replace(rf"[^\w{_INDIC}]+|_", " ", regex=True)
    s = s.str.replace(r"\bnull\b", " ", regex=True)          # literal 'null' placeholders
    s = s.str.replace(r"\b0+(?=\d)", "", regex=True)          # leading zeros: 0684 -> 684
    return s.str.replace(r"\s+", " ", regex=True).str.strip()
