"""PyInstaller entry point; application behavior lives in the package."""

import sys
from pathlib import Path


source_tree = Path(__file__).resolve().parent / "src"
if source_tree.is_dir():
    sys.path.insert(0, str(source_tree))

from fireball_edge.__main__ import main


if __name__ == "__main__":
    raise SystemExit(main())
