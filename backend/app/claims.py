"""Regex and rule-based sentence detector for factual and policy claims."""
import re

# Patterns that signal verifiable claims: numbers, %, superlatives, rankings, guarantees
CLAIM_PATTERNS = [
    r"\b\d+(\.\d+)?%?\b",
    r"\b#\d+\b",
    r"\b(number\s+one|no\.\s*1)\b",
    r"\b(best|cheapest|fastest|highest|top-rated|premier|leading)\b",
    r"\b(guarantee|guaranteed|guarantees|risk-free|bulletproof)\b",
    r"\b(launched|founded|headquartered|users|customers|languages|save|savings)\b"
]


def extract_claims(text: str) -> list:
    """Split draft into sentences and return those containing factual or promotional claims."""
    # Split text into candidate sentences
    raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    detected_claims = []

    for sentence in raw_sentences:
        clean = sentence.strip()
        if not clean or len(clean) < 10:
            continue

        for pattern in CLAIM_PATTERNS:
            if re.search(pattern, clean, flags=re.IGNORECASE):
                detected_claims.append(clean)
                break

    return detected_claims