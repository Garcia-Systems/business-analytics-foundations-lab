"""A small, explicit registry of CLI-accessible chapter experiments."""

from dataclasses import dataclass
from collections.abc import Callable

from analytics_foundations.chapter_00 import run as run_chapter_00
from analytics_foundations.chapter_01 import run as run_chapter_01
from analytics_foundations.chapter_02 import run as run_chapter_02
from analytics_foundations.chapter_03 import run as run_chapter_03
from analytics_foundations.chapter_04 import run as run_chapter_04
from analytics_foundations.chapter_05 import run as run_chapter_05
from analytics_foundations.chapter_06 import run as run_chapter_06
from analytics_foundations.chapter_07 import run as run_chapter_07
from analytics_foundations.chapter_08 import run as run_chapter_08
from analytics_foundations.chapter_09 import run as run_chapter_09
from analytics_foundations.chapter_10 import run as run_chapter_10
from analytics_foundations.chapter_11 import run as run_chapter_11
from analytics_foundations.chapter_12 import run as run_chapter_12
from analytics_foundations.chapter_13 import run as run_chapter_13


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
    "chapter-04": Chapter(
        "chapter-04", "Change & Derivatives", available=True, run=run_chapter_04
    ),
    "chapter-05": Chapter(
        "chapter-05", "Accumulation & Integrals", available=True, run=run_chapter_05
    ),
    "chapter-06": Chapter(
        "chapter-06", "Vectors: Data Becomes Geometry", available=True, run=run_chapter_06
    ),
    "chapter-07": Chapter(
        "chapter-07", "Matrices: Data Becomes Structure", available=True, run=run_chapter_07
    ),
    "chapter-08": Chapter(
        "chapter-08", "Linear Algebra for Models", available=True, run=run_chapter_08
    ),
    "chapter-09": Chapter(
        "chapter-09", "Arrays & Vectorized Thinking", available=True, run=run_chapter_09
    ),
    "chapter-10": Chapter(
        "chapter-10", "Tables & DataFrames", available=True, run=run_chapter_10
    ),
    "chapter-11": Chapter(
        "chapter-11", "Messy Data", available=True, run=run_chapter_11
    ),
    "chapter-12": Chapter(
        "chapter-12", "Transform, Group & Join", available=True, run=run_chapter_12
    ),
    "chapter-13": Chapter(
        "chapter-13", "Seeing Data", available=True, run=run_chapter_13
    ),
}


def get_chapter(slug: str) -> Chapter | None:
    """Return registered chapter metadata, or ``None`` for an unknown slug."""

    return CHAPTERS.get(slug)
