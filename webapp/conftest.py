"""webapp-level pytest configuration: ensure repo root is on sys.path.

This allows tests to use ``from webapp.core.xxx import yyy`` regardless of
whether pytest is invoked from the repo root or from within webapp/.
"""
import sys
from pathlib import Path

# Insert repo root (parent of this conftest) so `webapp` is importable.
_repo_root = str(Path(__file__).parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
