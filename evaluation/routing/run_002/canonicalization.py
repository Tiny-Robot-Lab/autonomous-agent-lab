"""Normative prompt canonicalization for Run 002 offline shadow evaluation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CanonicalPrompt:
    """Typed canonicalized prompt and its strict UTF-8 SHA-256 fingerprint.

    `canonical_prompt` is excluded from repr to reduce accidental prompt leakage
    into logs; callers must never serialize `canonical_prompt` into telemetry.
    """

    canonical_prompt: str = field(repr=False)
    canonical_input_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.canonical_prompt, str):
            raise TypeError(
                f"canonical_prompt must be str, got {type(self.canonical_prompt).__name__}"
            )
        if (
            not isinstance(self.canonical_input_sha256, str)
            or len(self.canonical_input_sha256) != 64
        ):
            raise ValueError("canonical_input_sha256 must be a 64-character hex digest")

    @property
    def is_empty(self) -> bool:
        return self.canonical_prompt == ""


def canonicalize_prompt(prompt: str) -> CanonicalPrompt:
    """Canonicalize a prompt per the Run 002 offline shadow contract.

    Steps:
    1. Require Python `str`.
    2. Remove exactly one leading U+FEFF if present.
    3. Replace CRLF with LF, then remaining CR with LF.
    4. Split only on U+000A (`str.splitlines()` is prohibited).
    5. Remove trailing ASCII space (' ') and U+0009 ('\\t') from each line.
    6. Remove leading and trailing lines empty after removing ' ' and '\\t'.
    7. Rejoin using U+000A.
    8. Preserve internal blank lines, leading indentation, case, punctuation,
       word order, and Unicode code points without Unicode normalization.
    9. Encode with strict UTF-8 (rejecting lone surrogates) and compute SHA-256.
    """
    if type(prompt) is not str and not isinstance(prompt, str):
        raise TypeError(f"prompt must be str, got {type(prompt).__name__}")

    text = prompt.removeprefix("\ufeff")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = [line.rstrip(" \t") for line in text.split("\n")]

    start = 0
    end = len(lines)
    while start < end and lines[start] == "":
        start += 1
    while end > start and lines[end - 1] == "":
        end -= 1

    canonical = "\n".join(lines[start:end])
    encoded = canonical.encode("utf-8", errors="strict")
    digest = hashlib.sha256(encoded).hexdigest()
    return CanonicalPrompt(canonical_prompt=canonical, canonical_input_sha256=digest)


@dataclass(frozen=True)
class CanonicalizationVector:
    """Normative reusable test vector for prompt canonicalization."""

    vector_id: str
    raw_prompt: str
    expected_canonical_prompt: str | None
    expected_sha256: str | None
    expect_unicode_error: bool = False
    expect_empty_manifest_rejection: bool = False


def _sha256_utf8(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="strict")).hexdigest()


NORMATIVE_CANONICALIZATION_VECTORS: tuple[CanonicalizationVector, ...] = (
    CanonicalizationVector(
        vector_id="ordinary_input",
        raw_prompt="What is 15% of 850?",
        expected_canonical_prompt="What is 15% of 850?",
        expected_sha256=_sha256_utf8("What is 15% of 850?"),
    ),
    CanonicalizationVector(
        vector_id="single_leading_bom",
        raw_prompt="\ufeffResearch consensus algorithms",
        expected_canonical_prompt="Research consensus algorithms",
        expected_sha256=_sha256_utf8("Research consensus algorithms"),
    ),
    CanonicalizationVector(
        vector_id="double_leading_bom_removes_only_one",
        raw_prompt="\ufeff\ufeffResearch consensus algorithms",
        expected_canonical_prompt="\ufeffResearch consensus algorithms",
        expected_sha256=_sha256_utf8("\ufeffResearch consensus algorithms"),
    ),
    CanonicalizationVector(
        vector_id="crlf_cr_lf_combinations",
        raw_prompt="line1\r\nline2\rline3\nline4",
        expected_canonical_prompt="line1\nline2\nline3\nline4",
        expected_sha256=_sha256_utf8("line1\nline2\nline3\nline4"),
    ),
    CanonicalizationVector(
        vector_id="trailing_ascii_spaces",
        raw_prompt="alpha   \nbeta  ",
        expected_canonical_prompt="alpha\nbeta",
        expected_sha256=_sha256_utf8("alpha\nbeta"),
    ),
    CanonicalizationVector(
        vector_id="trailing_horizontal_tabs",
        raw_prompt="alpha\t\t\nbeta\t \t",
        expected_canonical_prompt="alpha\nbeta",
        expected_sha256=_sha256_utf8("alpha\nbeta"),
    ),
    CanonicalizationVector(
        vector_id="leading_and_trailing_blank_lines",
        raw_prompt=" \t \n\n  keep leading\n\t \n\n",
        expected_canonical_prompt="  keep leading",
        expected_sha256=_sha256_utf8("  keep leading"),
    ),
    CanonicalizationVector(
        vector_id="internal_blank_lines_preserved",
        raw_prompt="first\n   \n\t\nsecond",
        expected_canonical_prompt="first\n\n\nsecond",
        expected_sha256=_sha256_utf8("first\n\n\nsecond"),
    ),
    CanonicalizationVector(
        vector_id="leading_indentation_preserved",
        raw_prompt="    def f():\n        return 1  ",
        expected_canonical_prompt="    def f():\n        return 1",
        expected_sha256=_sha256_utf8("    def f():\n        return 1"),
    ),
    CanonicalizationVector(
        vector_id="non_lf_unicode_line_separators_not_split",
        raw_prompt="a\u000b \nb\u000c \nc\u0085 \nd\u2028 \ne\u2029 ",
        expected_canonical_prompt="a\u000b\nb\u000c\nc\u0085\nd\u2028\ne\u2029",
        expected_sha256=_sha256_utf8("a\u000b\nb\u000c\nc\u0085\nd\u2028\ne\u2029"),
    ),
    CanonicalizationVector(
        vector_id="nfc_and_nfd_preserved_unchanged",
        raw_prompt="caf\u00e9 vs cafe\u0301",
        expected_canonical_prompt="caf\u00e9 vs cafe\u0301",
        expected_sha256=_sha256_utf8("caf\u00e9 vs cafe\u0301"),
    ),
    CanonicalizationVector(
        vector_id="lone_utf16_surrogate_rejected",
        raw_prompt="bad\ud800prompt",
        expected_canonical_prompt=None,
        expected_sha256=None,
        expect_unicode_error=True,
    ),
    CanonicalizationVector(
        vector_id="empty_prompt_rejected_by_manifest",
        raw_prompt="",
        expected_canonical_prompt="",
        expected_sha256=_sha256_utf8(""),
        expect_empty_manifest_rejection=True,
    ),
    CanonicalizationVector(
        vector_id="whitespace_only_prompt_rejected_by_manifest",
        raw_prompt="\ufeff \t\r\n  \t \n",
        expected_canonical_prompt="",
        expected_sha256=_sha256_utf8(""),
        expect_empty_manifest_rejection=True,
    ),
)
