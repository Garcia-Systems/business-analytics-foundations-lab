"""A small, explicit registry of CLI-accessible chapter experiments."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Chapter:
    """Metadata needed to expose a chapter experiment from the CLI."""

    slug: str
    title: str
    available: bool = False


# Add a record here when a chapter gains an executable experiment. Keeping the
# registry explicit makes the book's execution model easy for readers to trace.
CHAPTERS: dict[str, Chapter] = {
    "chapter-00": Chapter("chapter-00", "The Analytics Laboratory"),
}


def get_chapter(slug: str) -> Chapter | None:
    """Return registered chapter metadata, or ``None`` for an unknown slug."""

    return CHAPTERS.get(slug)

