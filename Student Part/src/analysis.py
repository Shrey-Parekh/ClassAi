from __future__ import annotations

from dataclasses import dataclass
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .vector_store import Chunk


@dataclass
class UnitMatch:
    unit: str
    title: str
    count: int
    examples: list[str]


def is_unit_mapping_query(question: str) -> bool:
    lowered = question.lower()
    return (
        "unit" in lowered
        and any(word in lowered for word in ["map", "count", "how many", "which"])
        and any(word in lowered for word in ["question", "questions", "paper", "papers"])
    )


def build_unit_mapping_answer(chunks: list[Chunk]) -> str:
    units = extract_units(chunks)
    questions = extract_questions(chunks)

    if not units:
        return (
            "I could not find clear Unit I / Unit II style syllabus sections in the indexed syllabus. "
            "Re-index the correct semester syllabus, then ask again."
        )
    if not questions:
        return (
            "I could not extract actual questions from the indexed question papers. "
            "The OCR may have captured only instructions, or the selected subject has no 2020+ papers."
        )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    unit_texts = [f"{unit.title} {unit.text}" for unit in units]
    question_texts = [question["text"] for question in questions]
    matrix = vectorizer.fit_transform([*unit_texts, *question_texts])
    unit_matrix = matrix[: len(units)]
    question_matrix = matrix[len(units) :]
    scores = cosine_similarity(question_matrix, unit_matrix)

    matches = [
        UnitMatch(unit=unit.number, title=unit.title, count=0, examples=[])
        for unit in units
    ]
    uncategorized = 0
    for question_index, row in enumerate(scores):
        best_index = int(row.argmax())
        best_score = float(row[best_index])
        if best_score < 0.035:
            uncategorized += 1
            continue
        match = matches[best_index]
        match.count += 1
        if len(match.examples) < 2:
            match.examples.append(question_texts[question_index])

    lines = ["Question distribution by syllabus unit:"]
    lines.append("")
    lines.append("| Unit | Matched questions | Evidence |")
    lines.append("| --- | ---: | --- |")
    for match in matches:
        examples = " / ".join(shorten(example, 120) for example in match.examples)
        lines.append(
            f"| {match.unit}: {shorten(match.title, 48)} | {match.count} | {examples or '-'} |"
        )

    lines.append("")
    lines.append(f"Total extracted questions considered: {len(questions)}")
    if uncategorized:
        lines.append(f"Uncategorized low-confidence questions: {uncategorized}")
    lines.append("")
    lines.append(
        "Note: this is a local OCR + keyword-similarity estimate, so use it as a study-priority map rather than an official marking distribution."
    )
    return "\n".join(lines)


@dataclass
class UnitSection:
    number: str
    title: str
    text: str


def extract_units(chunks: list[Chunk]) -> list[UnitSection]:
    syllabus_text = "\n".join(
        chunk.text for chunk in chunks if chunk.metadata.get("document_type") == "syllabus"
    )
    if not syllabus_text.strip():
        return []

    pattern = re.compile(
        r"\b(?:Unit|UNIT)\s*[-:]?\s*([IVX]+|\d{1,2})\b[:.\-\s]*(.*?)(?=\b(?:Unit|UNIT)\s*[-:]?\s*(?:[IVX]+|\d{1,2})\b|$)",
        re.DOTALL,
    )
    units: list[UnitSection] = []
    for match in pattern.finditer(syllabus_text):
        number = normalize_unit_number(match.group(1))
        body = clean_text(match.group(2))
        if len(body) < 30:
            continue
        title = infer_unit_title(body)
        units.append(UnitSection(number=number, title=title, text=body))

    return dedupe_units(units)


def extract_questions(chunks: list[Chunk]) -> list[dict[str, str]]:
    questions: list[dict[str, str]] = []
    for chunk in chunks:
        if chunk.metadata.get("document_type") != "question_paper":
            continue
        text = clean_text(chunk.text)
        if is_instruction_text(text):
            continue
        for question in split_questions(text):
            if is_real_question(question):
                questions.append(
                    {
                        "source": str(chunk.metadata.get("source_name", "")),
                        "page": str(chunk.metadata.get("page_number", "")),
                        "text": question,
                    }
                )
    return dedupe_questions(questions)


def split_questions(text: str) -> list[str]:
    marker = re.compile(
        r"(?=(?:^|\n|\s)(?:Q\.?\s*\d+[a-z]?|Question\s+\d+|\d+\s*[.)])\s+)",
        re.IGNORECASE,
    )
    parts = [clean_text(part) for part in marker.split(text) if clean_text(part)]
    if len(parts) <= 1:
        sentences = re.split(r"(?<=[?])\s+|\n+", text)
        return [clean_text(sentence) for sentence in sentences if clean_text(sentence)]
    return parts


def is_instruction_text(text: str) -> bool:
    lowered = text.lower()
    instruction_hits = [
        "instructions",
        "candidates should read",
        "cover page",
        "question paper and on the cover",
        "marks are indicated",
        "answer all questions",
    ]
    return sum(1 for phrase in instruction_hits if phrase in lowered) >= 2


def is_real_question(text: str) -> bool:
    lowered = text.lower()
    if len(text) < 35:
        return False
    if is_instruction_text(text):
        return False
    verbs = [
        "explain",
        "describe",
        "discuss",
        "define",
        "differentiate",
        "compare",
        "write",
        "calculate",
        "derive",
        "what",
        "why",
        "how",
        "list",
        "draw",
        "classify",
        "analyze",
    ]
    return any(verb in lowered for verb in verbs)


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip(" -:\n\t")


def infer_unit_title(body: str) -> str:
    first = re.split(r"[.;\n]", body, maxsplit=1)[0]
    return clean_text(first)[:90] or "Untitled unit"


def normalize_unit_number(value: str) -> str:
    roman = {"I": "1", "II": "2", "III": "3", "IV": "4", "V": "5", "VI": "6"}
    return roman.get(value.upper(), value)


def dedupe_units(units: list[UnitSection]) -> list[UnitSection]:
    seen: set[str] = set()
    unique: list[UnitSection] = []
    for unit in units:
        if unit.number in seen:
            continue
        seen.add(unit.number)
        unique.append(unit)
    return unique


def dedupe_questions(questions: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    for question in questions:
        key = re.sub(r"[^a-z0-9]+", "", question["text"].lower())[:160]
        if key in seen:
            continue
        seen.add(key)
        unique.append(question)
    return unique


def shorten(text: str, max_chars: int) -> str:
    text = clean_text(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."
