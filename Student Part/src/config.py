from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


load_dotenv()


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    tavily_api_key: str | None
    syllabus_dir: Path
    question_papers_dir: Path
    index_dir: Path
    index_file: str
    top_k: int
    chunk_size: int
    chunk_overlap: int
    render_dpi: int
    min_question_paper_year: int

    @property
    def index_path(self) -> Path:
        return self.index_dir / self.index_file


def get_settings() -> Settings:
    return Settings(
        tavily_api_key=os.getenv("TAVILY_API_KEY", "").strip() or None,
        syllabus_dir=Path(os.getenv("SYLLABUS_DIR", "Syllabus")),
        question_papers_dir=Path(
            os.getenv("QUESTION_PAPERS_DIR", "Question Papers")
        ),
        index_dir=Path(os.getenv("INDEX_DIR", "data/index")),
        index_file=os.getenv("INDEX_FILE", "academic_index.joblib"),
        top_k=int(os.getenv("TOP_K", "8")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1200")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "180")),
        render_dpi=int(os.getenv("RENDER_DPI", "180")),
        min_question_paper_year=int(os.getenv("MIN_QUESTION_PAPER_YEAR", "2020")),
    )


def ensure_project_dirs(settings: Settings) -> None:
    settings.syllabus_dir.mkdir(parents=True, exist_ok=True)
    settings.question_papers_dir.mkdir(parents=True, exist_ok=True)
    settings.index_dir.mkdir(parents=True, exist_ok=True)
