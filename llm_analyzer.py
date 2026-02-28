"""
llm_analyzer.py
Uses Claude API to generate human-readable manipulation explanations.
Falls back gracefully if API key is not set.
"""

import os
import anthropic

_client = None

def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return None
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def generate_explanation(text: str, analysis: dict) -> dict:
    """Return structured LLM explanation for the detected manipulation."""
    client = _get_client()
    if client is None:
        return {
            "success": False,
            "full_explanation": "Set ANTHROPIC_API_KEY environment variable to enable LLM analysis.",
            "sections": {}
        }

    score         = analysis.get("overall_score", 0)
    tactic_summary = ", ".join(analysis.get("top_tactics", [])) or "none"

    prompt = f"""You are a media-literacy expert specialising in psychological manipulation.

The automated transformer scored this text {score}/100 for manipulation.
Top tactics detected: {tactic_summary}

TEXT:
\"\"\"{text[:1500]}\"\"\"

Write a structured analysis with exactly these 5 headings (use them verbatim):
SUMMARY
KEY TACTICS
PSYCHOLOGICAL MECHANISM
WHAT TO WATCH OUT FOR
VERDICT

Rules:
- Cite actual phrases from the text as evidence.
- Be direct and specific.
- Total response under 420 words.
"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=650,
            messages=[{"role": "user", "content": prompt}]
        )
        explanation = response.content[0].text
        return {
            "success":          True,
            "full_explanation": explanation,
            "sections":         _parse_sections(explanation)
        }
    except Exception as exc:
        return {
            "success":          False,
            "full_explanation": f"LLM error: {exc}",
            "sections":         {}
        }


def _parse_sections(text: str) -> dict:
    headings = ["SUMMARY", "KEY TACTICS", "PSYCHOLOGICAL MECHANISM",
                "WHAT TO WATCH OUT FOR", "VERDICT"]
    sections: dict = {}
    current = None
    buf: list = []

    for line in text.splitlines():
        stripped = line.strip()
        matched = next((h for h in headings if stripped.upper().startswith(h)), None)
        if matched:
            if current:
                sections[current] = " ".join(buf).strip()
            current = matched
            buf = []
        elif current and stripped:
            buf.append(stripped)

    if current:
        sections[current] = " ".join(buf).strip()

    return sections