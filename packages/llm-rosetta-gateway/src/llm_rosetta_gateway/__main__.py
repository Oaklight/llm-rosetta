"""Allow running the gateway as ``python -m llm_rosetta_gateway``."""

from .cli import main

raise SystemExit(main())
