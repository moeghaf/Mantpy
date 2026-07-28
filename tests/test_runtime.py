"""Tests for Mantpy's import-time runtime behaviour."""

from __future__ import annotations

import subprocess
import sys
import textwrap


def test_import_does_not_mutate_process_or_plotting_state() -> None:
    """A fresh import must not configure process-wide runtime state."""
    code = textwrap.dedent(
        """
        import os
        import sys
        import matplotlib as mpl

        original_environment = dict(os.environ)
        original_rcparams = dict(mpl.rcParams)
        original_hook = sys.unraisablehook

        import mantpy

        assert dict(os.environ) == original_environment
        assert dict(mpl.rcParams) == original_rcparams
        assert sys.unraisablehook is original_hook
        assert "mantpy.pl" not in sys.modules
        assert "mantpy.nn" not in sys.modules
        print("OK")
        """
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
