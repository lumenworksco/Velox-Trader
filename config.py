"""Configuration — LEGACY SHIM (V12 item 1.2).

This file previously contained the full configuration. The canonical source
is now ``config/settings.py``, re-exported by ``config/__init__.py``.

When Python sees ``import config`` it resolves to the ``config/`` **package**
(not this file), so this file is effectively unreachable at runtime.  It is
kept only as a safety net: if any tooling or script somehow imports the bare
file, it will still get the correct values via the re-export below.

DO NOT add new settings here.  Edit ``config/settings.py`` instead.
"""

# Re-export everything from the canonical source so that if this file
# is somehow imported directly, callers still get the right values.
from config.settings import *  # noqa: F401, F403
