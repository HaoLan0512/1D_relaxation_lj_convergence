"""Render the supplied LJ convergence dataset from the repository root."""
from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent
            / "experiments" / "lj_step_convergence" / "main.py"),
        run_name="__main__",
    )
