# 1D Relaxation LJ step profile and convergence

A standalone package for reproducing a continuum profile and two
atomistic-to-continuum convergence figures for a fixed Lennard-Jones model.
The repository includes all required source code and the saved, validated
profiles. Ordinary execution reads those profiles and generates figures;
it does not run a new optimization.

These numerical examples accompany
[Continuum Limit for One-Dimensional Moiré Relaxation](https://arxiv.org/abs/2609.37943)
by Hao Lan, Michael Hott, Jeff Calder, and Alexander B. Watson (2026).
See [Citation](#citation) for a copyable reference.

## Download and run

Download this repository using **Code > Download ZIP** and extract the entire
archive, or clone it:

```text
git clone https://github.com/HaoLan0512/1D_relaxation_lj_convergence.git
cd 1D_relaxation_lj_convergence
```

If you downloaded the ZIP, open a terminal in the extracted repository folder
containing `main.py` before running the commands below.

Use **Python 3.12**; the supplied results were prepared with Python 3.12.12.
The five numerical dependencies are pinned in the included requirements file.
Install them into an isolated environment.

These versions specify the tested environment. The program does not enforce
exact dependency-version matches, but missing or incompatible packages can
cause import or runtime errors. Downloading the ZIP does not install Python
or the dependencies.

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

The supplied convergence figures start at `N=320`. To reproduce that displayed
range in all supplied figures, run:

```text
python main.py --plot N=320
```

Plain `python main.py` instead shows the full range starting at `N=20`.
The continuum figure is the same in both cases. The default dataset and outputs
are under `outputs/lj_step_convergence/`:

The linked [arXiv v1](https://arxiv.org/abs/2609.37943v1) does not contain these
figures. The table identifies the accompanying numerical examples by purpose
and filename, without assigning manuscript figure numbers.

| Figure | PNG preview | PDF |
| --- | --- | --- |
| Normalized continuum profile with relative-displacement inset | [v_phase_fixed.png](outputs/lj_step_convergence/figures/v_phase_fixed.png) | [v_phase_fixed.pdf](outputs/lj_step_convergence/figures/v_phase_fixed.pdf) |
| Phase-matched two-layer displacement errors and first-order rate guides | [convergence_loglog.png](outputs/lj_step_convergence/figures/convergence_loglog.png) | [convergence_loglog.pdf](outputs/lj_step_convergence/figures/convergence_loglog.pdf) |
| Reconstructed relative-displacement errors and second-order rate guides | [relative_convergence_loglog.png](outputs/lj_step_convergence/figures/relative_convergence_loglog.png) | [relative_convergence_loglog.pdf](outputs/lj_step_convergence/figures/relative_convergence_loglog.pdf) |

The [tables folder](outputs/lj_step_convergence/tables/) contains all saved error
values, fitted slopes, adjacent rates, accuracy budgets, and endpoint diagnostics.

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
requirements.txt                 # Forwards the experiment's pinned dependency list
experiments/lj_step_convergence/  # Self-contained source and detailed documentation
outputs/lj_step_convergence/
  profiles/                      # One continuum and twelve atomistic bundles
  publication_manifest.json      # Configuration and profile integrity hashes
  verification.json              # Saved numerical validation evidence
  figures/
  tables/
```

Keep the complete repository intact. The root `main.py` is a convenience
launcher: it runs `experiments/lj_step_convergence/main.py` and forwards the
command-line arguments. Both entry points use the same implementation and
produce the same results for the same arguments, dataset, and environment.

For example, from the repository root these commands are equivalent:

```text
python main.py --plot N=320
python experiments/lj_step_convergence/main.py --plot N=320
```

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

## Citation

If you use this package in research, please cite the associated paper:

Hao Lan, Michael Hott, Jeff Calder, and Alexander B. Watson.
*Continuum Limit for One-Dimensional Moiré Relaxation*.
arXiv:2609.37943 (2026). [Paper and version history](https://arxiv.org/abs/2609.37943).

```bibtex
@misc{lan2026continuum,
  title = {Continuum Limit for One-Dimensional Moir{\'e} Relaxation},
  author = {Lan, Hao and Hott, Michael and Calder, Jeff and Watson, Alexander B.},
  year = {2026},
  eprint = {2609.37943},
  archivePrefix = {arXiv},
  primaryClass = {math-ph},
  url = {https://arxiv.org/abs/2609.37943}
}
```
