"""zmodel package shim.

This package exposes modules stored in the local python/ directory so users can
import with names like `zmodel.functions` while keeping CLI scripts separate.
"""

__version__ = "dev"

from pathlib import Path
import sys

_pkg_root = Path(__file__).resolve().parent
_python_dir = _pkg_root / "python"

# Support `import zmodel.<module>` where modules live under python/.
if _python_dir.is_dir():
    __path__.append(str(_python_dir))
    if str(_python_dir) not in sys.path:
        sys.path.insert(0, str(_python_dir))
