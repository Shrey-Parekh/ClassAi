from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from .config import Settings
from .pdf_processing import list_pdfs


ROMAN_TO_INT = {
    "I": 1,
    "II": 2,
    "III": 3,
    "IV": 4,
    "V": 5,
    "VI": 6,
    "VII": 7,
    "VIII": 8,
    "IX": 9,
    "X": 10,
    "XI": 11,
    "XII": 12,
}


@dataclass(frozen=True)
class SemesterOption:
    id: str
    label: str
    number: int
    path: str


@dataclass(frozen=True)
class SubjectOption:
    id: str
    label: str
    pdf_count: int
    path: str


def get_semester_options(settings: Settings) -> list[SemesterOption]:
    options: list[SemesterOption] = []
    for directory in settings.question_papers_dir.rglob("*"):
        if not directory.is_dir():
            continue
        number = parse_semester_number(directory.name)
        if number is None:
            continue
        options.append(
            SemesterOption(
                id=str(number),
                label=f"Semester {number}",
                number=number,
                path=str(directory),
            )
        )
    return sorted(options, key=lambda option: option.number)


def get_subject_options(settings: Settings, semester_id: str) -> list[SubjectOption]:
    semester_dir = semester_path(settings, semester_id)
    if semester_dir is None:
        return []

    subjects: list[SubjectOption] = []
    for directory in sorted(item for item in semester_dir.rglob("*") if item.is_dir()):
        pdfs = list_pdfs(directory)
        direct_pdfs = sorted(item for item in directory.glob("*.pdf") if item.is_file())
        if not direct_pdfs:
            continue
        relative = directory.relative_to(semester_dir)
        subject_id = relative.as_posix()
        subjects.append(
            SubjectOption(
                id=subject_id,
                label=" / ".join(relative.parts),
                pdf_count=len(pdfs),
                path=str(directory),
            )
        )
    return subjects


def semester_path(settings: Settings, semester_id: str) -> Path | None:
    for option in get_semester_options(settings):
        if option.id == str(semester_id):
            return Path(option.path)
    return None


def selected_question_paper_pdfs(
    settings: Settings,
    semester_id: str | None,
    subject_ids: list[str] | None,
) -> list[Path]:
    if not semester_id:
        return list_pdfs(settings.question_papers_dir)

    semester_dir = semester_path(settings, semester_id)
    if semester_dir is None:
        return []

    if not subject_ids:
        return list_pdfs(semester_dir)

    pdfs: list[Path] = []
    for subject_id in subject_ids:
        subject_dir = (semester_dir / subject_id).resolve()
        try:
            subject_dir.relative_to(semester_dir.resolve())
        except ValueError:
            continue
        pdfs.extend(list_pdfs(subject_dir))
    return sorted(set(pdfs))


def selected_syllabus_pdfs(settings: Settings, semester_id: str | None) -> list[Path]:
    if not semester_id:
        return list_pdfs(settings.syllabus_dir)

    number = int(semester_id)
    matches = [
        path
        for path in list_pdfs(settings.syllabus_dir)
        if parse_semester_number(path.stem) == number
    ]
    return matches


def parse_semester_number(text: str) -> int | None:
    normalized = text.upper().replace("_", " ")
    match = re.search(r"\bSEM(?:ESTER)?\s*[- ]*\s*(\d{1,2})\b", normalized)
    if match:
        return int(match.group(1))

    match = re.search(r"\bSEM(?:ESTER)?\s+([IVX]+)\b", normalized)
    if match:
        return ROMAN_TO_INT.get(match.group(1))

    return None
