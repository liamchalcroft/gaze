"""Put the package ``src`` directory on ``sys.path`` for the test suite.

This environment is a standalone package (``src`` layout) that is normally
installed before use. The conftest lets the tests run in place from the repo
root without an editable install. Only test runs that descend into this
directory load it; the repository's own ``tests/`` and ``examples/`` runs are
unaffected.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
