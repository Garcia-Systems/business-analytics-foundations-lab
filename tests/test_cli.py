"""Tests for the minimal chapter command line interface."""

import pytest

from analytics_foundations.chapters import CHAPTERS, Chapter, get_chapter
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


def test_registry_recognizes_available_chapter_zero() -> None:
    chapter = get_chapter("chapter-00")
    assert chapter is not None
    assert chapter.title == "The Analytics Laboratory"
    assert chapter.available is True
    assert chapter.run is not None


def test_registry_recognizes_available_chapter_one() -> None:
    chapter = get_chapter("chapter-01")
    assert chapter is not None
    assert chapter.title == "Functions Become Models"
    assert chapter.available is True
    assert chapter.run is not None


def test_chapter_zero_executes(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def fake_run() -> int:
        nonlocal called
        called = True
        return 0

    monkeypatch.setitem(
        CHAPTERS,
        "chapter-00",
        Chapter("chapter-00", "The Analytics Laboratory", True, fake_run),
    )
    assert main(["chapter-00"]) == 0
    assert called


def test_chapter_one_executes(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def fake_run() -> int:
        nonlocal called
        called = True
        return 0

    monkeypatch.setitem(
        CHAPTERS,
        "chapter-01",
        Chapter("chapter-01", "Functions Become Models", True, fake_run),
    )
    assert main(["chapter-01"]) == 0
    assert called


def test_registry_recognizes_available_chapter_two() -> None:
    chapter = get_chapter("chapter-02")
    assert chapter is not None
    assert chapter.title == "Exponents, Logs & Growth"
    assert chapter.available is True
    assert chapter.run is not None


def test_chapter_two_executes(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def fake_run() -> int:
        nonlocal called
        called = True
        return 0

    monkeypatch.setitem(
        CHAPTERS,
        "chapter-02",
        Chapter("chapter-02", "Exponents, Logs & Growth", True, fake_run),
    )
    assert main(["chapter-02"]) == 0
    assert called


def test_registry_recognizes_available_chapter_three() -> None:
    chapter = get_chapter("chapter-03")
    assert chapter is not None
    assert chapter.title == "Summation & Aggregation"
    assert chapter.available is True
    assert chapter.run is not None


def test_chapter_three_executes(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def fake_run() -> int:
        nonlocal called
        called = True
        return 0

    monkeypatch.setitem(
        CHAPTERS,
        "chapter-03",
        Chapter("chapter-03", "Summation & Aggregation", True, fake_run),
    )
    assert main(["chapter-03"]) == 0
    assert called
