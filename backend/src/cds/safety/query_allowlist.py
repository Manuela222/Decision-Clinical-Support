"""Phase 10, part 2: query input allowlist for the chat/query panel.

MATCHING APPROACH (documented per spec): exact, case-insensitive token
matching — not fuzzy. A token is allowed through if it's a bare number, an
ICD-9-code-shaped token, or an exact (lowercased) match against a fixed
vocabulary built from (a) a small hardcoded list of common English words
needed for an ordinary clinical question to read naturally, and (b) every
individual word appearing in D_ICD_DIAGNOSES' ICD9_CODE/SHORT_TITLE/LONG_TITLE.

Fuzzy matching was deliberately rejected: it would let injected text ride
along on near-matches, which weakens exactly the security boundary this
allowlist exists to provide. Exact matching produces some false rejects on
legitimate medical vocabulary not covered by (a)/(b) — see the measured
false-reject rate over sample clinical queries in
tests/test_query_allowlist.py. That's an accepted trade-off here: a
clinician whose word gets stripped can rephrase; a prompt injection that
slips through cannot be undone after the fact.
"""
import re
from dataclasses import dataclass, field
from typing import List, Set

import pandas as pd

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\.\-]*")
_ICD9_CODE_RE = re.compile(r"^[Vv]?\d{2,3}(\.\d{1,2})?$")
_NUMBER_RE = re.compile(r"^\d+(\.\d+)?$")

# Common English words needed for an ordinary clinical question to read
# naturally, kept deliberately small and boring (function words, plus a
# handful of generic clinical-workflow nouns/verbs this project's queries
# will legitimately use).
SAFE_COMMON_WORDS: Set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "and", "or", "but", "not", "no", "yes", "with", "without", "for", "of",
    "to", "in", "on", "at", "by", "from", "given", "about", "as",
    "what", "which", "who", "when", "where", "why", "how", "does", "do",
    "did", "can", "could", "should", "would", "will", "may", "might",
    "patient", "patients", "medication", "medications", "medicine", "drug",
    "drugs", "class", "classes", "recommend", "recommended", "recommendation",
    "recommendations", "suggest", "suggested", "check", "safe", "safety",
    "unsafe", "compatible", "compatibility", "interaction", "interactions",
    "history", "condition", "conditions", "diagnosis", "diagnoses", "admitted",
    "admission", "discharge", "discharged", "hypertension", "hypertensive",
    "blood", "pressure", "current", "prior", "chronic", "acute", "level",
    "levels", "lab", "labs", "value", "values", "high", "low", "normal",
    "abnormal", "risk", "renal", "kidney", "function", "reason",
    "this", "that", "these", "those", "it", "its", "their", "they", "he",
    "she", "his", "her", "him", "years", "year", "old", "male", "female",
    "antihypertensive", "antihypertensives", "inhibitor", "inhibitors",
    "continue", "continued", "continuing",
}


@dataclass(frozen=True)
class QueryAllowlist:
    allowed_words: Set[str] = field(default_factory=set)
    allowed_codes: Set[str] = field(default_factory=set)


def build_query_allowlist(d_icd_diagnoses: pd.DataFrame) -> QueryAllowlist:
    """Build the allowlist from D_ICD_DIAGNOSES: every ICD9 code, plus every
    individual word in each code's SHORT_TITLE/LONG_TITLE."""
    allowed_words = set(SAFE_COMMON_WORDS)
    allowed_codes: Set[str] = set()
    for _, row in d_icd_diagnoses.iterrows():
        code = str(row.get("ICD9_CODE", "")).strip().lower()
        if code:
            allowed_codes.add(code)
        for title_col in ("SHORT_TITLE", "LONG_TITLE"):
            title = row.get(title_col)
            if isinstance(title, str):
                for word in re.findall(r"[a-zA-Z]+", title.lower()):
                    if len(word) > 2:
                        allowed_words.add(word)
    return QueryAllowlist(allowed_words=allowed_words, allowed_codes=allowed_codes)


@dataclass
class SanitizedQuery:
    original_query: str
    sanitized_text: str
    rejected_tokens: List[str]
    is_rejected: bool  # whole query rejected: too much of it was unrecognized to safely pass through


def sanitize_query(raw_query: str, allowlist: QueryAllowlist, max_rejected_fraction: float = 0.5) -> SanitizedQuery:
    """Strip any token not in the allowlist. If more than
    `max_rejected_fraction` of tokens are unrecognized, reject the whole
    query (returns empty sanitized_text) rather than pass through a
    fragment that could still carry an injected instruction."""
    tokens = _TOKEN_RE.findall(raw_query or "")
    kept: List[str] = []
    rejected: List[str] = []
    for token in tokens:
        # Match against the token with trailing punctuation stripped (e.g.
        # "list." -> "list"), so ordinary sentence punctuation doesn't cause
        # a false reject; the original token (with punctuation) is what
        # gets kept, so output stays readable.
        lowered = token.strip(".-").lower()
        if (
            _NUMBER_RE.match(lowered)
            or _ICD9_CODE_RE.match(lowered)
            or lowered in allowlist.allowed_codes
            or lowered in allowlist.allowed_words
        ):
            kept.append(token)
        else:
            rejected.append(token)

    is_rejected = bool(tokens) and (len(rejected) / len(tokens)) > max_rejected_fraction
    sanitized_text = "" if is_rejected else " ".join(kept)

    return SanitizedQuery(
        original_query=raw_query,
        sanitized_text=sanitized_text,
        rejected_tokens=rejected,
        is_rejected=is_rejected,
    )
