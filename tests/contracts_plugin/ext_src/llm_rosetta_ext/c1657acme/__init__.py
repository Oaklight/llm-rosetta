"""An extension fixture whose sidecars import every contract module (#1657).

The loader runs sidecars while ``import llm_rosetta`` is still executing, so
those sidecars import the contracts at module scope on purpose.
"""
