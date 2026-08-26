"""
Turns SHAP contributions into a one-paragraph, analyst-readable explanation. Two paths:
  - LLM: fluent prose, same facts, nothing invented — the prompt hands it the SHAP
    contributions and asks it to DESCRIBE them, never to reason about fraud itself.
  - Template: deterministic, zero-dependency, always available.

THE DESIGN INVARIANT (CLAUDE.md sec 6): the LLM never decides — policy.py already chose
the action before this module is ever called. Delete this module entirely and every
decision is byte-identical; only the prose degrades. narrate() below is built so it can
never raise: every LLM failure mode falls back to the template rather than blocking.
"""
import os
import re

MAX_NARRATIVE_LEN = 800
MIN_NARRATIVE_LEN = 20
# Required break-test: "LLM returns garbage -> validate the response; reject and
# template." A response that succeeds as an API call but carries an obvious injection
# marker, or is empty, or absurdly long, is rejected — not trusted just because the call
# didn't error.
_SUSPICIOUS_PATTERNS = re.compile(r"(ignore previous|system prompt|you are now|<\|)", re.I)


class NarrativeError(Exception):
    pass


def render_template(action: str, calibrated_prob: float, amount_inr: float,
                     contributions: list[dict]) -> str:
    """Always available, always correct given its inputs — the floor every narrative
    degrades to. Never raises, never returns a blank field."""
    parts = []
    for c in contributions:
        sign = "+" if c["contribution"] >= 0 else ""
        val = c["value"]
        if val is None:
            val_str = " missing"
        elif isinstance(val, float):
            val_str = f"={val:.2f}"
        else:
            val_str = f"={val}"
        parts.append(f"{c['feature']}{val_str} ({sign}{c['contribution']:.3f})")
    reasons = "; ".join(parts) if parts else "no individual feature dominated"
    return (
        f"Flagged for {action} at {calibrated_prob:.1%} estimated fraud probability on a "
        f"Rs{amount_inr:,.0f} transaction. Top contributing factors: {reasons}."
    )


def validate(text: str) -> bool:
    """Exposed separately from render_llm so it can be unit-tested directly with crafted
    bad inputs, without needing a live API call to provoke a real LLM into misbehaving."""
    if not text or not text.strip():
        return False
    if len(text) < MIN_NARRATIVE_LEN or len(text) > MAX_NARRATIVE_LEN:
        return False
    if _SUSPICIOUS_PATTERNS.search(text):
        return False
    return True


def render_llm(action: str, calibrated_prob: float, amount_inr: float,
                contributions: list[dict], timeout_s: float = 6.0,
                _client=None) -> str:
    """Raises NarrativeError on ANY failure — missing key, package missing, network
    error, timeout, or a response that fails validate(). Callers always catch this and
    fall back; this function never returns something untrusted.

    _client: internal override for testing (inject a fake client / point at an
    unreachable base_url) without needing a real API key.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if _client is None and not api_key:
        raise NarrativeError("ANTHROPIC_API_KEY not set")

    try:
        import anthropic
    except ImportError as e:
        raise NarrativeError(f"anthropic package not installed: {e}") from e

    fact_lines = "\n".join(
        f"- {c['feature']} = {c['value']} (contributed {c['contribution']:+.3f} to the fraud score)"
        for c in contributions
    )
    prompt = (
        "You are writing a one-paragraph note for a fraud analyst reviewing a flagged "
        "payment. Describe ONLY the facts below in plain English -- do not add reasons, "
        "assumptions, or context that isn't listed. Keep it to 2-3 sentences.\n\n"
        f"Action taken: {action}\n"
        f"Estimated fraud probability: {calibrated_prob:.1%}\n"
        f"Transaction amount: Rs{amount_inr:,.0f}\n"
        f"Top contributing factors (SHAP, sign = direction toward/away from fraud):\n"
        f"{fact_lines}"
    )

    try:
        client = _client or anthropic.Anthropic(api_key=api_key, timeout=timeout_s)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        raise NarrativeError(f"LLM call failed: {type(e).__name__}: {e}") from e

    text = response.content[0].text.strip() if response.content else ""
    if not validate(text):
        raise NarrativeError(f"LLM response failed validation (len={len(text)})")
    return text


def narrate(action: str, calibrated_prob: float, amount_inr: float,
            contributions: list[dict]) -> tuple[str, bool, list[str]]:
    """The orchestrator every caller should use. Returns (narrative, used_llm, fallbacks).
    Never raises — render_template has no failure mode, so this always returns something
    usable, exactly the "the LLM never decides, only renders" invariant made concrete."""
    fallbacks = []
    try:
        text = render_llm(action, calibrated_prob, amount_inr, contributions)
        return text, True, fallbacks
    except NarrativeError as e:
        fallbacks.append(f"llm_narrative_failed:{e}")
    template = render_template(action, calibrated_prob, amount_inr, contributions)
    return template, False, fallbacks
