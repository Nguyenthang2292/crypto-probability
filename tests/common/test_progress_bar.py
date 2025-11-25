import io
from contextlib import redirect_stdout

import pytest

from modules.ProgressBar import ProgressBar


def test_progress_bar_reaches_total_and_prints_label():
    buf = io.StringIO()
    bar = ProgressBar(total=5, label="Test", width=10)

    with redirect_stdout(buf):
        for _ in range(5):
            bar.update()
        bar.finish()

    output = buf.getvalue()
    assert "Test" in output
    assert "5/5" in output


def test_progress_bar_incomplete():
    """Test ProgressBar that doesn't reach total."""
    buf = io.StringIO()
    bar = ProgressBar(total=10, label="Test", width=10)

    with redirect_stdout(buf):
        for _ in range(3):
            bar.update()
        bar.finish()

    output = buf.getvalue()
    assert "Test" in output
    assert "3" in output  # Should show current progress


def test_progress_bar_zero_total():
    """Test ProgressBar with zero total."""
    buf = io.StringIO()
    bar = ProgressBar(total=0, label="Test", width=10)

    with redirect_stdout(buf):
        bar.update()
        bar.finish()

    output = buf.getvalue()
    assert "Test" in output or len(output) > 0


def test_progress_bar_single_update():
    """Test ProgressBar with single update."""
    buf = io.StringIO()
    bar = ProgressBar(total=1, label="Single", width=10)

    with redirect_stdout(buf):
        bar.update()
        bar.finish()

    output = buf.getvalue()
    assert "Single" in output
    assert "1" in output


def test_progress_bar_multiple_updates():
    """Test ProgressBar with more updates than total."""
    buf = io.StringIO()
    bar = ProgressBar(total=5, label="Test", width=10)

    with redirect_stdout(buf):
        for _ in range(7):  # More than total
            bar.update()
        bar.finish()

    output = buf.getvalue()
    assert "Test" in output
