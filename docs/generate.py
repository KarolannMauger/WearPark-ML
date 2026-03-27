"""Generate HTML documentation for the WearPark ML source modules.

Reads all ``src/*.py`` files and produces a self-contained HTML site under
``docs/html/`` using `pdoc <https://pdoc.dev>`_.

Usage::

    python docs/generate.py

The output directory is ``docs/html/``. Open ``docs/html/index.html`` in any
browser to browse the documentation locally.

Requirements:
    pdoc >= 14.0.0  (included in requirements.txt)

Note:
    Run from the repository root. The script resolves ``src/`` relative to
    its own location, so it works regardless of the current working directory.
"""

import os
import subprocess
import sys

ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC    = os.path.join(ROOT, "src")
OUT    = os.path.join(ROOT, "docs", "html")

MODULES = [
    "model",
    "dataset",
    "train",
    "evaluate",
    "predict",
    "preprocess_monipar",
    "api",
]


def generate() -> None:
    """Build the HTML documentation site and print the output path."""
    os.makedirs(OUT, exist_ok=True)

    cmd = [
        sys.executable, "-m", "pdoc",
        "--html",
        "--output-dir", OUT,
        "--force",
    ] + MODULES

    print(f"Generating docs from {SRC} → {OUT}")
    result = subprocess.run(cmd, cwd=SRC, capture_output=True, text=True)

    if result.returncode != 0:
        print("pdoc error:")
        print(result.stderr)
        sys.exit(1)

    print(f"\nDocs generated successfully.")
    print(f"Open: {os.path.join(OUT, 'index.html')}")


if __name__ == "__main__":
    generate()
