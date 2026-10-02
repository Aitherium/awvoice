"""Import the package FROM THE TREE, the awrouter/awmine tests idiom.

Without this the suite needs `pip install -e` before it can even be collected, and
the hermetic CI gate installs nothing -- a runner whose python happened to carry an
editable install passed, a fresh one errored at collection (measured 2026-10-02).
PYTHONPATH carries the same root to the `python -m awvoice.cli` child processes.
"""
import os as _os
import sys as _sys
from pathlib import Path as _Path

_PKG_ROOT = _Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PKG_ROOT))
_os.environ["PYTHONPATH"] = _os.pathsep.join(
    p for p in (str(_PKG_ROOT), _os.environ.get("PYTHONPATH", "")) if p
)
