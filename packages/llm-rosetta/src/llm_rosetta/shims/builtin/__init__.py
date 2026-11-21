"""The built-in provider shims: vendor data only.

Each ``<vendor>/`` directory below holds ``provider.yaml`` files and their
sidecars (``transforms.py``, ``quota.py``, ``auth.py``).  The code that reads
them is :mod:`llm_rosetta.shims._loader`; this package exists so the sidecars
load under a real package name and so the loader can find this directory.
"""

from __future__ import annotations

from pathlib import Path

#: The built-in providers root.  :mod:`llm_rosetta.shims._loader` imports it;
#: rebind it there to redirect the scan.
_PROVIDERS_DIR = Path(__file__).parent
