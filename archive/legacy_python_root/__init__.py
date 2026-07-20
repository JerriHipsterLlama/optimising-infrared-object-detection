"""Compatibility package for legacy imports from the ``src.python`` tree.

New code should prefer ``src.python`` imports. This shim keeps the existing
LPQ tests and scripts importable without requiring an editable installation.
"""

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent.parent / "src" / "python")]

