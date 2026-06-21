"""Pytest configuration shared by the whole test suite.

Its single job is to put the **repository root** on ``sys.path`` so that
``import xsretrieval`` resolves to the flat package at the repo root regardless
of the directory pytest is invoked from. The ``xsretrieval`` package lives at
``<repo_root>/xsretrieval`` (a flat layout, no ``src/``), and this file sits in
``<repo_root>/tests``, so the repo root is exactly this file's parent's parent.
"""

from __future__ import annotations

import sys
from pathlib import Path

# <repo_root>/tests/conftest.py -> parents[1] == <repo_root>
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
