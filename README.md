# 1D Relaxation LJ step profile and convergence

A standalone package reproducing a continuum profile and two atomistic-to-continuum
convergence figures for a fixed Lennard-Jones model. Source code and validated
profiles are included; ordinary execution generates figures without new optimization.
The examples accompany [Continuum Limit for One-Dimensional Moiré Relaxation](https://arxiv.org/abs/2609.37943).

## Download and run

Download **Code > Download ZIP** and extract the entire repository, or clone it:

```text
git clone https://github.com/HaoLan0512/1D_relaxation_lj_convergence.git
cd 1D_relaxation_lj_convergence
```

Install Python 3.12, then open a terminal in the repository folder containing
`main.py`. Create an environment, install dependencies, and run:

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

## Commands

Run these commands from the repository root with your environment active.

| Command | Role |
| --- | --- |
| `python main.py` | Generate figures and tables from supplied profiles; convergence plots start at `N=20`. |
| `python main.py --plot N=320` | Start both convergence plots at `N=320`, matching the supplied figures' displayed range. |
| `python main.py --data-dir PATH` | Use another complete, compatible dataset for both input and output; replace `PATH` with its folder. |
| `python main.py --verify` | Recheck saved states without optimization and update `verification.json`. |
| `python main.py --prepare --data-dir outputs/new_run` | Construct or resume a dataset; expensive. Read the [preparation guide](experiments/lj_step_convergence/README.md#changing-lj-parameters-and-fresh-calculation). |
| `python main.py --help` | Display available options. |

Accepted plotting cutoffs: `20, 40, 80, 160, 320, 640, 1280`. Both convergence
figures include supplied sizes at or above the cutoff, through `10240`; profiles,
the continuum figure, full-data tables, and fits are unchanged.

`--plot`, `--verify`, and `--prepare` are mutually exclusive; `--data-dir` can
accompany any mode. With complete, compatible data, preparation renders
existing results without new optimization.

## Generated files

The default dataset and outputs are in `outputs/lj_step_convergence/`.
Figure numbers refer to the [associated paper](https://arxiv.org/abs/2609.37943).

| Manuscript figure | PNG | PDF |
| --- | --- | --- |
| **Figure 1.** Continuum profile and relative-displacement inset | [PNG](outputs/lj_step_convergence/figures/v_phase_fixed.png) | [PDF](outputs/lj_step_convergence/figures/v_phase_fixed.pdf) |
| **Figure 2.** Phase-matched pair errors | [PNG](outputs/lj_step_convergence/figures/convergence_loglog.png) | [PDF](outputs/lj_step_convergence/figures/convergence_loglog.pdf) |
| **Figure 3.** Reconstructed relative-displacement errors | [PNG](outputs/lj_step_convergence/figures/relative_convergence_loglog.png) | [PDF](outputs/lj_step_convergence/figures/relative_convergence_loglog.pdf) |

The [tables folder](outputs/lj_step_convergence/tables/) contains errors, fitted
slopes, rates, and diagnostics. Rendering saves PNG/PDF files without opening
windows and replaces the selected dataset's **entire `figures/` and `tables/`
directories**, including manual additions. Saved profiles remain unchanged.

## Further details

See the [experiment README](experiments/lj_step_convergence/README.md) for
mathematical definitions, validation, changing LJ parameters, preparation,
and source organization.

## Citation

Please cite Hao Lan, Michael Hott, Jeff Calder, and Alexander B. Watson,
*Continuum Limit for One-Dimensional Moiré Relaxation*,
[arXiv:2609.37943](https://arxiv.org/abs/2609.37943) (2026).

<details>
<summary>BibTeX</summary>

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

</details>
