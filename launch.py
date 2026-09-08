"""Run the source checkout without installing the package."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from mumax_sonic.__main__ import main

if __name__ == "__main__":
    main()
