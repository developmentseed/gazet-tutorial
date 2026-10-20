"""India-specific vocabulary: held-out states, abbreviations, common name aliases.

These lists are hand-written and worth a second look from anyone who knows the
geography. They are *not* used to build the SFT training data; they drive the
`noisy` benchmark split (and, later, preference / RL data), so the fine-tuned
model is measured on name forms it never saw in training.
"""

from __future__ import annotations

import re

# ISO 3166-2 codes of the states held out entirely: neither the state nor any of
# its districts is ever a training anchor. Chosen by hand to mix south, east and
# a small union-territory-sized state.
HOLDOUT_STATES = {"IN-KL": "Kerala", "IN-OD": "Odisha", "IN-GA": "Goa"}

# How people actually write state names in a query. Keys are Overture primary names.
STATE_ABBREVIATIONS: dict[str, list[str]] = {
    "Uttar Pradesh": ["UP"],
    "Madhya Pradesh": ["MP"],
    "Himachal Pradesh": ["HP"],
    "Tamil Nadu": ["TN"],
    "Andhra Pradesh": ["AP"],
    "West Bengal": ["WB"],
    "Jammu and Kashmir": ["J&K", "J and K"],
    "Uttarakhand": ["UK"],
    "Chhattisgarh": ["CG"],
    "Telangana": ["TS"],
    "Arunachal Pradesh": ["Arunachal"],
    "Andaman and Nicobar Islands": ["A&N Islands", "Andaman"],
}

# Older or colloquial names still in wide use. Keys are Overture primary names
# (after the " district" suffix strip in geodata.download_divisions); a key that
# is not present in the data is simply skipped by `resolve_aliases`.
NAME_ALIASES: dict[str, list[str]] = {
    "Bengaluru Urban": ["Bangalore Urban", "Bangalore"],
    "Bengaluru Rural": ["Bangalore Rural"],
    "Mysuru": ["Mysore"],
    "Belagavi": ["Belgaum"],
    "Shivamogga": ["Shimoga"],
    "Tumakuru": ["Tumkur"],
    "Ballari": ["Bellary"],
    "Vijayapura": ["Bijapur"],
    "Kalaburagi": ["Gulbarga"],
    "Chikkamagaluru": ["Chikmagalur"],
    "Thiruvananthapuram": ["Trivandrum"],
    "Kozhikode": ["Calicut"],
    "Alappuzha": ["Alleppey"],
    "Thrissur": ["Trichur"],
    "Kannur": ["Cannanore"],
    "Kollam": ["Quilon"],
    "Palakkad": ["Palghat"],
    "Puducherry": ["Pondicherry"],
    "Chennai": ["Madras"],
    "Thoothukudi": ["Tuticorin"],
    "Kanniyakumari": ["Kanyakumari"],
    "Tiruchirappalli": ["Trichy"],
    "Prayagraj": ["Allahabad"],
    "Gurugram": ["Gurgaon"],
    "Vadodara": ["Baroda"],
    "Kolkata": ["Calcutta"],
    "Hooghly": ["Hugli"],
    "Ahilyanagar": ["Ahmednagar"],
    "Chhatrapati Sambhajinagar": ["Sambhajinagar"],
    "Dharashiv": ["Osmanabad"],
    "Khordha": ["Khurda"],
    "Odisha": ["Orissa"],
    "Uttarakhand": ["Uttaranchal"],
}


def resolve_aliases(present_names: set[str]) -> dict[str, list[str]]:
    """Keep only alias entries whose primary name exists in the loaded data."""
    return {k: v for k, v in NAME_ALIASES.items() if k in present_names}


def resolve_abbreviations(present_names: set[str]) -> dict[str, list[str]]:
    return {k: v for k, v in STATE_ABBREVIATIONS.items() if k in present_names}


def replace_name(text: str, old: str, new: str) -> str | None:
    """Replace one whole-word occurrence of a place name; None if it does not occur."""
    pattern = r"(?<![\w])" + re.escape(old) + r"(?![\w])"
    if not re.search(pattern, text):
        return None
    return re.sub(pattern, new, text, count=1)
