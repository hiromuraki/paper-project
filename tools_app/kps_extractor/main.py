"""Cross-platform entry point for the extractor's independent uv project."""
import runpy
from pathlib import Path
import sys

if __name__ == "__main__":
    source = Path(__file__).resolve().parent / "src"
    sys.path.insert(0, str(source))
    runpy.run_path(str(source / "main.py"), run_name="__main__")
