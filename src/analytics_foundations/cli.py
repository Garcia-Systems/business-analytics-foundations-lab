"""Command line interface for chapter experiments."""

import argparse
from collections.abc import Sequence

from analytics_foundations.chapters import CHAPTERS, get_chapter


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser separately so its behavior is easy to test."""

    parser = argparse.ArgumentParser(
        prog="analytics-foundations",
        description="Run experiments from the Business Analytics Foundations Lab.",
    )
    parser.add_argument(
        "chapter",
        nargs="?",
        choices=sorted(CHAPTERS),
        help="registered chapter experiment to run",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and run a registered chapter when it is available."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.chapter is None:
        parser.print_help()
        return 0

    chapter = get_chapter(args.chapter)
    # argparse choices ensure this is present; the guard keeps the boundary clear.
    if chapter is None:  # pragma: no cover
        parser.error(f"unknown chapter: {args.chapter}")
    if not chapter.available:
        parser.exit(
            2,
            f"{chapter.slug} ({chapter.title}) is registered as a placeholder; "
            "its experiment is not implemented yet.\n",
        )
    return 0

