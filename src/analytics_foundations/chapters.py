"""A small, explicit registry of CLI-accessible chapter experiments."""

from dataclasses import dataclass
from collections.abc import Callable

from analytics_foundations.chapter_00 import run as run_chapter_00
from analytics_foundations.chapter_01 import run as run_chapter_01
from analytics_foundations.chapter_02 import run as run_chapter_02
from analytics_foundations.chapter_03 import run as run_chapter_03


@dataclass(frozen=True)
class Chapter:
    """Metadata needed to expose a chapter experiment from the CLI."""

    slug: str
    title: str
    available: bool = False
    run: Callable[[], int] | None = None


# Add a record here when a chapter gains an executable experiment. Keeping the
# registry explicit makes the book's execution model easy for readers to trace.
CHAPTERS: dict[str, Chapter] = {
    "chapter-00": Chapter(
        "chapter-00", "The Analytics Laboratory", available=True, run=run_chapter_00
    ),
    "chapter-01": Chapter(
        "chapter-01", "Functions Become Models", available=True, run=run_chapter_01
    ),
    "chapter-02": Chapter(
        "chapter-02", "Exponents, Logs & Growth", available=True, run=run_chapter_02
    ),
    "chapter-03": Chapter(
        "chapter-03", "Summation & Aggregation", available=True, run=run_chapter_03
    ),
}


def get_chapter(slug: str) -> Chapter | None:
    """Return registered chapter metadata, or ``None`` for an unknown slug."""

    return CHAPTERS.get(slug)
