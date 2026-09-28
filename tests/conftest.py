"""Use a writable plotting cache without changing global torch precision."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "build/matplotlib"))
