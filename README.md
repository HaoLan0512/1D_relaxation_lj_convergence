# 1D Relaxation LJ step profile and convergence

A standalone package for reproducing a continuum profile and two
atomistic-to-continuum convergence figures for a fixed Lennard-Jones model.
The repository includes all required source code and the saved, validated
profiles. Ordinary execution reads those profiles and generates figures;
it does not run a new optimization.

## Download and run

Download this repository using **Code > Download ZIP** and extract the entire
archive, or clone it:

```text
git clone https://github.com/HaoLan0512/lj_step_convergence.git
cd lj_step_convergence
```

If you downloaded the ZIP, open a terminal in the extracted repository folder
containing `main.py` before running the commands below.

Use **Python 3.12**; the supplied results were prepared with Python 3.12.12.
The five numerical dependencies are pinned in the included requirements file.
Install them into an isolated environment.

Windows Command Prompt:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python main.py
```

macOS or Linux:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

If you already have an environment containing the pinned dependencies, activate
it and run `python main.py`. For example, users of the original CLR
`relax` environment can run `conda activate relax` first. Use
`python -c "import sys; print(sys.executable)"` to check which interpreter is
active if a dependency fails to import.

## Choose the first plotted size

```text
python main.py
python main.py --plot N=320
python main.py --plot N=1280
```

The default starts at `N=20`. The accepted starting values are
`20, 40, 80, 160, 320, 640, 1280`. Both convergence figures show all supplied
sizes greater than or equal to the selected value, up to `N=10240`.
Axes and rate guides adapt to the displayed range while figure dimensions stay
fixed. Invalid values exit with an error before any output is written.

This option changes only the convergence figures. The continuum figure,
profiles, full-data tables, and fits are unchanged. The predefined fit uses
`N=1280,2560,5120,10240`. Gray curves are rate guides, not fitted curves.

## Generated files

The default dataset and outputs are under `outputs/lj_step_convergence/`:

- `figures/v_phase_fixed.png` and `.pdf`: continuum profile with displacement inset.
- `figures/convergence_loglog.png` and `.pdf`: pair errors and first-order rate guides.
- `figures/relative_convergence_loglog.png` and `.pdf`: relative-displacement errors and second-order rate guides.
- `tables/`: all saved error values, fitted slopes, adjacent rates, accuracy budgets, and endpoint diagnostics.

Figures are saved without opening image windows. Successful rendering replaces
the entire `figures/` and `tables/` directories in the selected dataset,
including extra files or manual edits there. Saved profiles, the publication
manifest, and the verification report are not changed by rendering.

To render another complete, compatible dataset, select its folder:

```text
python main.py --plot N=320 --data-dir outputs/lj_step_convergence_fresh
```

That folder must already contain compatible profiles and their matching
`publication_manifest.json`. It is used for both inputs and generated outputs.

## Package contents

```text
main.py                          # Root launcher
requirements.txt                 # Pinned dependency list
experiments/lj_step_convergence/  # Self-contained source and detailed documentation
outputs/lj_step_convergence/
  profiles/                      # One continuum and twelve atomistic bundles
  publication_manifest.json      # Configuration and profile integrity hashes
  verification.json              # Saved numerical validation evidence
  figures/
  tables/
```

Keep both folders intact. The original command
`python experiments/lj_step_convergence/main.py` remains supported.
The code locates the default dataset relative to its source; no original CLR
checkout or machine-specific data path is required.

## Numerical details and advanced modes

See the [experiment README](experiments/lj_step_convergence/README.md) for the
model, coordinates, error definitions, validation criteria, source organization,
and full preparation workflow.

`python main.py --verify` performs fresh checks of all saved endpoints without
optimization. Ordinary rendering only checks the stored dataset and uses saved
values; it does not repeat that numerical validation.

`--prepare` explicitly constructs or resumes profiles in a new data folder.
Fresh preparation is expensive and has a cumulative eight-hour numerical
budget. Use the detailed instructions before starting it. With a complete,
compatible dataset, `--prepare` only renders the saved results.
`--plot`, `--verify`, and `--prepare` are mutually exclusive.

The included convergence results are numerical observations for this fixed
experiment. They do not certify global minimizers or constitute convergence
proofs.
