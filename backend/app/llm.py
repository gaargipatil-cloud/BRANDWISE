"""LLM claim verification client connecting to Google Gemini REST endpoint."""
import os
import json
import re
import asyncio
import httpx
from dotenv import load_dotenv

# Load local environment variables from .env
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# HTTP codes that usually mean "busy right now, try again"
RETRY_CODES = (429, 500, 503)


def normalize_string(s: str) -> str:
    """Normalize string by lowercasing and collapsing whitespace."""
    return re.sub(r"\s+", " ", s.lower()).strip()


def verify_quote_presence(quote: str, passages: list) -> bool:
    """Confirm that the LLM quoted string literally appears within retrieved passages."""
    if not quote or not quote.strip():
        return False
    norm_quote = normalize_string(quote)
    for p in passages:
        norm_passage = normalize_string(p.get("text", ""))
        if norm_quote in norm_passage:
            return True
    return False


def make_fallback(claim: str, reason: str) -> dict:
    """Safe UNVERIFIABLE result used whenever the automated check cannot complete."""
    return {
        "verdict": "UNVERIFIABLE",
        "source_name": "None",
        "source_quote": "",
        "reason": reason,
        "confidence": 0.0,
        "suggestion": claim
    }


async def verify_claim_with_gemini(claim: str, retrieved_passages: list) -> dict:
    """Send claim and top passages to Gemini, returning a strict 3-state verdict JSON."""
    if not GEMINI_API_KEY:
        return make_fallback(claim, "Automated check unavailable: no API key configured.")

    # Format the prompt with retrieved passages
    passages_text = "\n\n".join(
        [f"Source [{p['source_name']}]:\n{p['text']}" for p in retrieved_passages]
    )

    system_instruction = (
        "You are Brandwise, a strict factual verification assistant for 'Finwise'.\n"
        "Review the given CLAIM against ONLY the provided APPROVED PASSAGES.\n"
        "Return strict JSON matching these requirements:\n"
        "{\n"
        '  "verdict": "SUPPORTED" | "CONTRADICTED" | "UNVERIFIABLE",\n'
        '  "source_name": "<exact source file name or None>",\n'
        '  "source_quote": "<verbatim exact quote from passages or empty string>",\n'
        '  "reason": "<short plain explanation>",\n'
        '  "confidence": <float between 0.0 and 1.0>,\n'
        '  "suggestion": "<minimal edit to make it compliant keeping original meaning>"\n'
        "}\n"
        "RULES:\n"
        "1. If no approved passage covers the claim, verdict MUST be 'UNVERIFIABLE'. Never guess.\n"
        "2. If verdict is UNVERIFIABLE, source_quote must be empty string.\n"
        "3. If verdict is SUPPORTED or CONTRADICTED, source_quote MUST BE A VERBATIM SUBSTRING of the passages.\n"
        "4. Respond with pure JSON only."
    )

    user_prompt = f"APPROVED PASSAGES:\n{passages_text}\n\nCLAIM TO VERIFY:\n{claim}"

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    headers = {"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": f"{system_instruction}\n\n{user_prompt}"}]
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json"
        }
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = None
            # Try up to 3 times when Google reports it is busy
            for attempt in range(3):
                resp = await client.post(url, json=payload, headers=headers)
                if resp.status_code not in RETRY_CODES:
                    break
                await asyncio.sleep(2 * (attempt + 1))

            if resp.status_code != 200:
                if resp.status_code in RETRY_CODES:
                    note = "Google's service is busy right now. Try again in a minute."
                else:
                    note = "Check the API key and model name."
                return make_fallback(
                    claim,
                    f"Automated check unavailable (Gemini returned HTTP {resp.status_code}). {note}"
                )

            data = resp.json()
            raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()

            # Clean possible markdown wrapping
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]

            parsed = json.loads(raw_text.strip())

            verdict = str(parsed.get("verdict", "UNVERIFIABLE")).upper()
            if verdict not in ["SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"]:
                verdict = "UNVERIFIABLE"

            source_quote = str(parsed.get("source_quote", "")).strip()
            reason = parsed.get("reason", "Automated check completed.")
            confidence = float(parsed.get("confidence", 0.9))

            # Plain Python quote verification: a quote must really exist in the sources
            if verdict in ["SUPPORTED", "CONTRADICTED"]:
                if not verify_quote_presence(source_quote, retrieved_passages):
                    verdict = "UNVERIFIABLE"
                    confidence = 0.0
                    source_quote = ""
                    reason = "Quoted source could not be verified in brand documents."

            return {
                "verdict": verdict,
                "source_name": parsed.get("source_name", "None"),
                "source_quote": source_quote,
                "reason": reason,
                "confidence": confidence,
                "suggestion": parsed.get("suggestion", claim)
            }

    except Exception as exc:
        return make_fallback(
            claim,
            f"Automated check unavailable ({type(exc).__name__})."
        )