from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

_WORD_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])[\"'’”)\]]*\s+")
_PARAGRAPH_BOUNDARY = re.compile(r"\n\s*\n")
_IGNORED_TERMS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "can",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "his",
        "how",
        "i",
        "in",
        "is",
        "it",
        "its",
        "may",
        "of",
        "on",
        "or",
        "our",
        "she",
        "that",
        "the",
        "their",
        "them",
        "there",
        "they",
        "this",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
    }
)
_MAX_REPEATED_TERMS = 5


@dataclass(frozen=True)
class AnalysisReport:
    paragraph_count: int
    sentence_count: int
    word_count: int
    repeated_terms: tuple[tuple[str, int], ...]
    question_sentences: tuple[str, ...]

    def render(self) -> str:
        repeated_terms = (
            ", ".join(f"{term} ({count})" for term, count in self.repeated_terms)
            or "None"
        )
        questions = "\n".join(f"- {sentence}" for sentence in self.question_sentences)
        if not questions:
            questions = "None"

        return (
            "Analysis report\n"
            f"Paragraphs: {self.paragraph_count}\n"
            f"Sentences: {self.sentence_count}\n"
            f"Words: {self.word_count}\n"
            f"Repeated terms: {repeated_terms}\n"
            f"Question sentences:\n{questions}\n"
            "Scope: Only supplied material was analyzed; no retrieval was performed."
        )


def analyze_material(material: str) -> AnalysisReport:
    """Measure text structure and repeated terms without making factual claims."""
    if not isinstance(material, str):
        raise TypeError(f"material must be a str, got {type(material).__name__}")

    clean_material = material.strip()
    paragraphs = [
        paragraph
        for paragraph in _PARAGRAPH_BOUNDARY.split(clean_material)
        if paragraph.strip()
    ]
    sentences = [
        sentence.strip()
        for sentence in _SENTENCE_BOUNDARY.split(clean_material)
        if sentence.strip()
    ]
    words = _WORD_PATTERN.findall(clean_material)

    term_counts = Counter(
        word.casefold()
        for word in words
        if word.casefold() not in _IGNORED_TERMS
    )
    repeated_terms = tuple(
        sorted(
            (
                (term, count)
                for term, count in term_counts.items()
                if count > 1
            ),
            key=lambda item: (-item[1], item[0]),
        )[:_MAX_REPEATED_TERMS]
    )
    question_sentences = tuple(
        sentence
        for sentence in sentences
        if sentence.rstrip("\"'’”)]}").endswith("?")
    )

    return AnalysisReport(
        paragraph_count=len(paragraphs),
        sentence_count=len(sentences),
        word_count=len(words),
        repeated_terms=repeated_terms,
        question_sentences=question_sentences,
    )
