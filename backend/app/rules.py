"""Deterministic rule engine for brand terminology, length limits, and disclaimers."""
import os
import json
import re

BRAND_PACK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "brand_pack"))


def load_terminology() -> dict:
    """Load terminology configuration from JSON file."""
    path = os.path.join(BRAND_PACK_DIR, "terminology.json")
    if not os.path.exists(path):
        return {
            "preferred_terms": {},
            "banned_words": [],
            "channel_length_limits": {},
            "required_disclaimers": []
        }
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_rules(text: str, channel: str = "twitter") -> list:
    """Run deterministic checks for channel limits, banned words, preferred terms, and disclaimers."""
    findings = []
    config = load_terminology()

    # 1. Channel character length check
    limits = config.get("channel_length_limits", {})
    if channel in limits:
        max_len = limits[channel]
        actual_len = len(text)
        if actual_len > max_len:
            findings.append({
                "category": "channel_limit",
                "severity": "high",
                "original_text": text[:60] + "...",
                "verdict": "CONTRADICTED",
                "reason": f"Text exceeds {channel} limit of {max_len} characters (currently {actual_len}).",
                "source_name": "terminology.json",
                "source_quote": f'"{channel}": {max_len}',
                "suggestion": text[:max_len],
                "confidence": 1.0,
                "status": "open"
            })

    # 2. Banned words check (case-insensitive boundary match)
    for banned in config.get("banned_words", []):
        pattern = r"\b" + re.escape(banned) + r"\b"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            matched_str = match.group(0)
            findings.append({
                "category": "banned_word",
                "severity": "high",
                "original_text": matched_str,
                "verdict": "CONTRADICTED",
                "reason": f"The term '{matched_str}' is strictly prohibited under Finwise claims policy.",
                "source_name": "claims_policy.md - Section 2",
                "source_quote": "Guarantees of financial return, zero risk, or instant wealth are strictly prohibited.",
                "suggestion": "",
                "confidence": 1.0,
                "status": "open"
            })

    # 3. Preferred terminology check
    for disfavored, preferred in config.get("preferred_terms", {}).items():
        pattern = r"\b" + re.escape(disfavored) + r"\b"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            matched_str = match.group(0)
            findings.append({
                "category": "preferred_term",
                "severity": "medium",
                "original_text": matched_str,
                "verdict": "CONTRADICTED",
                "reason": f"Use preferred brand term '{preferred}' instead of '{matched_str}'.",
                "source_name": "terminology.json",
                "source_quote": f'"{disfavored}": "{preferred}"',
                "suggestion": preferred,
                "confidence": 1.0,
                "status": "open"
            })

    # 4. Required disclaimers check
    for req in config.get("required_disclaimers", []):
        triggers = req.get("trigger_terms", [])
        disclaimer = req.get("disclaimer_text", "")
        has_trigger = any(re.search(r"\b" + re.escape(t) + r"\b", text, re.IGNORECASE) for t in triggers)
        if has_trigger and disclaimer.lower() not in text.lower():
            findings.append({
                "category": "missing_disclaimer",
                "severity": "high",
                "original_text": text[:60] + "...",
                "verdict": "CONTRADICTED",
                "reason": "Text mentions investments or partner bank features without the required regulatory disclaimer.",
                "source_name": "claims_policy.md - Section 5",
                "source_quote": disclaimer,
                "suggestion": f"{text.rstrip()} {disclaimer}",
                "confidence": 1.0,
                "status": "open"
            })

    return findings