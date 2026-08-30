"""Tests for the minimal chapter command line interface."""

import pytest

from analytics_foundations.chapters import get_chapter
from analytics_foundations.cli import main


def test_help_works(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["--help"])
    assert error.value.code == 0
    assert "chapter-00" in capsys.readouterr().out


def test_invalid_chapter_fails_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["chapter-99"])
    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_registry_recognizes_chapter_zero_placeholder() -> None:
    chapter = get_chapter("chapter-00")
    assert chapter is not None
    assert chapter.title == "The Analytics Laboratory"
    assert chapter.available is False


def test_placeholder_exits_with_explanation(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        main(["chapter-00"])
    assert error.value.code == 2
    assert "placeholder" in capsys.readouterr().err

