# LJ step profile and convergence

This self-contained experiment reproduces a continuum profile and two
atomistic-to-continuum convergence figures from supplied profiles, using
the fixed Lennard--Jones model:

```text
(d, sigma, epsilon) = (2*pi, 1.8*pi, 16)
V(s) = 4*epsilon * (sigma**12/(d*d+s*s)**6 - sigma**6/(d*d+s*s)**3)
W(s) = 4*sum_m V(s-2*pi*m).
```

Lengths use units with registry period $2\pi$. Here `d` (`interlayer_distance`)
is the layer separation and `s` the horizontal separation. Parameters in
`config.py` are exact expressions evaluated at each arithmetic precision;
high-precision calculations do not inherit rounded binary64 parameters.

## Run and distribute

Use a Python environment containing the packages in `requirements.txt`; commands
never install dependencies. Run these examples from the package root containing
`experiments/` and `outputs/`. Double-clicking uses the operating system's Python
association; configure it or select the intended environment in an IDE.

### Commands and overwrite behavior

Without `--data-dir`, commands use `outputs/lj_step_convergence/` relative to the
source tree, regardless of working directory. Selecting another dataset never
changes this default.

| Mode | Work performed | Writes within the selected data folder |
| --- | --- | --- |
| Default | Renders three figures and tables from saved profiles; no optimization or fresh numerical validation. | Replaces only `figures/` and `tables/`. |
| `--plot N=...` | Same cached rendering, with both convergence figures starting at the selected `N`; all tables keep the complete data. | Replaces only `figures/` and `tables/`. |
| `--verify` | Recomputes equations, gradients, constraints, curvature/eigenvector residuals, observables, and accuracy checks for all 48 saved endpoints; no optimization. | Replaces only `verification.json` when checks finish. |
| `--prepare`: complete compatible data | Same cached rendering as default. | Replaces only `figures/` and `tables/`. |
| `--prepare`: new/empty folder or unfinished workspace | Constructs/resumes the continuum reference and 12 atomistic sizes; validates all before publication. | Checkpoints in `preparation/`; on success, profiles, manifest, verification report, figures, and tables. |

```text
python experiments/lj_step_convergence/main.py
python experiments/lj_step_convergence/main.py --verify
python experiments/lj_step_convergence/main.py --prepare
```

`--plot`, `--verify`, and `--prepare` are mutually exclusive. Successful rendering
replaces the entire `figures/` and `tables/` directories, including manual edits and extra
files. Temporary writing directories are removed after success; no permanent
dated folder is created. PNGs/PDFs are saved without opening windows; the terminal
reports progress and locations. No manuscript snippets or insertion instructions
are generated.

Ordinary execution and verification require complete compatible data. All modes,
including preparation, reject an existing publication with missing/corrupt
profiles or incompatible configuration instead of silently replacing it.
Profile hashes enforce integrity; recorded configuration settings must match.
For completed data, `source_fingerprints` record provenance only: Python edits
require no hash updates in any mode. Editing numerical formulas neither
recalculates profiles nor extends earlier validation to the edited formulas.

### Selecting the first plotted grid size

The default command starts both convergence figures at `N=20`. Use `--plot N=...`
to display only sizes greater than or equal to the selected value:

```text
python experiments/lj_step_convergence/main.py
python experiments/lj_step_convergence/main.py --plot N=320
python experiments/lj_step_convergence/main.py --plot N=1280
python experiments/lj_step_convergence/main.py --plot N=320 --data-dir outputs/lj_step_convergence_fresh
```

Accepted starting values are `20, 40, 80, 160, 320, 640, 1280`, including `1280`.
Use the literal `N=integer` format. A malformed value, a value outside the study's
size array, or a value above `1280` produces a command-line error with the allowed
choices and exits with status 2 before generating figures or writing files.
Combining `--plot` with `--prepare` or `--verify` also exits with status 2 without
writing files.

This option changes only the displayed points and rate guides in the two
convergence figures. Logarithmic axes automatically scale to the displayed
curves while figure dimensions stay fixed. The continuum profile figure,
profile calculations, saved profiles, and numerical configuration are unchanged.
All tables still contain the complete data, including smaller sizes; principal
fits, all-point fits, and adjacent rates are unchanged. In particular, the
principal fit uses the fixed window `N=1280,2560,5120,10240`, which every accepted
starting value retains.

Output filenames and replacement behavior are unchanged: the selected dataset's
`figures/` and `tables/` are replaced after successful rendering. PNGs and PDFs are
saved using the noninteractive Agg backend, without opening image windows. No
manifest update or numerical recalculation is needed. For other display edits,
change `plotting.py` and rerun the rendering command.

### Changing LJ parameters and fresh calculation

After changing `epsilon`, `sigma`, or `interlayer_distance` in `config.py`,
calculate compatible profiles; editing the old manifest cannot validate old
solutions for a new potential. Preserve supplied results by preparing a
new/empty folder:

```text
python experiments/lj_step_convergence/main.py --prepare --data-dir outputs/lj_step_convergence_fresh
```

Fresh preparation is expensive, writes its manifest automatically, and has a
cumulative eight-hour numerical budget. Repeat the same command to resume
checkpoints while budget remains; rerunning never resets it. Exhausted or
unfinished work stays in `preparation/`, without publishing failed/unfinished
sizes as complete. Resumption requires matching Python source fingerprints:
restore the original sources or use a new folder after edits. Successful
publication and rendering remove the preparation workspace; a successful retry
also removes a compatible completed workspace left by interrupted rendering.
Cleanup retains workspaces with mismatched source fingerprints.

Handled interruptions retain measured elapsed time. If worker termination
prevents reliable timing, its outstanding reservation is charged conservatively;
the record distinguishes measured execution from conservative accounting.

After completion, render or verify that dataset with:

```text
python experiments/lj_step_convergence/main.py --data-dir outputs/lj_step_convergence_fresh
python experiments/lj_step_convergence/main.py --verify --data-dir outputs/lj_step_convergence_fresh
```

Another full calculation needs another new folder; completed-data preparation
only renders cached profiles.

### Selecting and relocating data

In every mode, `--data-dir PATH` selects both inputs and generated outputs.
Supply it each time, pointing to the folder containing `publication_manifest.json`
and `profiles/`, not the profiles subfolder. Relative paths use the caller's
working directory; absolute paths work too. Quote paths containing spaces.

Selecting data never changes `config.py`; restore matching settings to reuse
original data after parameter edits. For independent replotting/verification,
copy the complete output folder and select the copy. Empty folders support only
fresh preparation.

### Replacing the default dataset

Replace the default folder's contents with a complete validated dataset in this
package's format: `profiles/`, its matching `publication_manifest.json`, and
corresponding `verification.json`. Retain exactly the profiles named in the
manifest, and match the current configuration, including LJ parameters and
lattice sizes. Then the default command renders the replacement dataset.
New profiles paired with an old manifest generally fail integrity checks.

### Files to distribute

Distribute both complete folders; no other repository code or outputs are needed.
Retain all Python files and every profile listed in the manifest.

```text
experiments/lj_step_convergence/
  [11 Python files, requirements.txt, README.md]
outputs/lj_step_convergence/
  publication_manifest.json
  profiles/
    continuum.npz
    N00020.npz ... N10240.npz
  figures/
    [three 300-dpi PNGs and three vector PDFs]
  tables/
    [plotted values, slopes, adjacent rates, budgets, diagnostics]
  verification.json
```

## Source organization

| Module | Responsibility |
| --- | --- |
| `__init__.py` | Package marker |
| `config.py` | Exact model parameters, grids, seeds, tolerances, resource limits |
| `model.py` | Potential, analytic derivatives, Fourier periodization, tails |
| `continuum.py` | Scalar continuum construction and independent Fourier check |
| `atomistic.py` | Native lattices, energies, gradients, initialization, Hessian actions |
| `solvers.py` | L-BFGS, matrix-free Newton, curvature diagnostics |
| `observables.py` | Phase matching, all-mode Fourier reconstruction, error norms |
| `validation.py` | Saved-state verification, sensitivity checks |
| `storage.py` | Compact pickle-free profiles, checkpoints, hashes, publication |
| `plotting.py` | Figures and tables |
| `main.py` | Reader, preparation, verification orchestration |

Imports perform no calculations or writes. Routines are self-contained;
comments identify adapted formulas.

## Continuum reference and atomistic states

Write $q=v_0-x$. The normalized continuum profile satisfies

\[
 v_0(0)=0,\qquad v_0(x+2\pi)=v_0(x)+2\pi,\qquad
 v_0'=\sqrt{2(W(v_0)+c)},\qquad
 \int_{-\pi}^{\pi}\frac{ds}{\sqrt{2(W(s)+c)}}=2\pi.
\]

Analytic Fourier periodization supplies the potential; scalar traversal time
determines the profile. Refinement allows Fourier degrees 64, 128, 256, and 512.
The supplied degree-256, 128-bit reference is compared with 80-bit arithmetic
and independent Fourier Newton construction. Accuracy targets are $10^{-13}$
in $H^2$ and $10^{-14}$ in maximum norm. Normalization, periodicity, oddness,
positive derivative, and Euler--Lagrange residual are checked separately.
The 32,769 display samples control rendering density, not solution accuracy.

The minimum derivative is approximately 0.016483903; transition from $-0.8\pi$
to $0.8\pi$ occupies approximately 25.2865% of the period, leaving broad plateaus.

The native layers contain $N$ and $N+1$ atoms, for

```text
N = 20, 40, 80, 160, 320, 640, 1280, 2560, 3840, 5120, 7680, 10240.
```

Each size starts from the deterministic seed-3 five-mode Fourier initialization
with maximum amplitude 0.1. L-BFGS is followed by matrix-free Newton corrections,
using preconditioned MINRES and three low curvature modes from LOBPCG. The full
mean-sum constraint space retains the relative-phase direction: neither layer
mean is separately fixed. Corrections use 128-bit force evaluation and saved
high-precision state strings. Native corrections are capped at 0.05 and accepted
through energy and full-residual backtracking; a conditional tighter MINRES retry
must still pass the same independently measured linear-residual criterion.

The four distinct saved states per size are the endpoints at interaction
cutoffs $M=160,320,640$ and a further tightened $M=320$ endpoint. The latter is
used in the plots; its comparison with the independently refined $M=640$ state
measures cutoff sensitivity. Cutoffs retain offsets $|m|\le M$ in the atomistic
sums. Each of the 12 compact bundles stores these four states, one shared
initialization, exact high-precision strings, the production observables, and
the necessary diagnostics. The continuum bundle makes 13 profile files in total.
Curvature eigenvectors and comparison observables are recomputed when verifying
rather than stored redundantly.

## Errors and accuracy checks

For each fixed atomistic state, define

\[
 \delta_N=\langle u_d^2\rangle-\langle u_d^1\rangle,\qquad
 q_{\delta_N}(x)=q(x-\delta_N)-\delta_N.
\]

Means are arithmetic means on the native grids. Pair errors compare the native
states with $(q_{\delta_N}/2,-q_{\delta_N}/2)$ on their respective grids. The
discrete $H_h^2$ norm combines layer-weighted displacement, forward-difference,
and centered second-difference terms. The maximum norm takes the largest
absolute nodal error over both layers. Here $h=4\pi/(2N+1)$.

Let $\mathcal T_M$ be periodic Fourier reconstruction from $M$ native samples.
The relative-displacement error is

\[
 e_N^{\mathrm{rel}}(x)=\mathcal T_Nu_d^1(x)
 -\mathcal T_{N+1}u_d^2(x)-q_{\delta_N}(x).
\]

It is measured in continuous $H^2$ and maximum norms. “Relative” means the
difference between layers, not division by a reference norm. Reconstructions
use the same undeformed coordinate, all native Fourier modes, and the real
even-grid Nyquist convention.

Evaluation avoids cancellation by forming accurate native residuals before
including exact continuum aliasing. Coarse-grid aliasing remains discretization
error, without smoothing or mode removal. Reconstruction sensitivity compares
independent evaluations of the same complete error, excluding genuine aliasing
from numerical uncertainty.

Every endpoint must have finite full and reduced gradients, satisfy the
mean-sum constraint, and pass curvature/eigenvector-residual checks. The
Euler--Lagrange criterion is $\|\mathrm{EL}\|_\infty\le10^{-5}h^2$, together
with the corresponding mesh-weighted gradient targets. The production endpoint
uses a further factor of 0.1 in its stationarity targets. Solver status alone
does not establish acceptance.

Each plotted relative norm $E_N$ has sensitivity budget
$B_N=\min(10^{-4}h^2,0.05E_N)$. Endpoint refinement, continuum refinement,
cutoff changes, reconstruction, arithmetic, and maximum evaluation are recorded
separately. These checks are numerical sensitivities, not rigorous bounds.
`verification.json` and diagnostics/budget tables report them; no missing size
is extrapolated or silently accepted.

## Figures and interpretation

Figures show the blue continuum profile with $q=v_0-x$ inset, pair errors,
and reconstructed relative-displacement errors. Each convergence panel has one
green data curve and an offset gray $CN^{-1}$ or $CN^{-2}$ guide, not a fit.
The predefined principal slope window is $N=1280,2560,5120,10240$; all-point
fits and adjacent rates are also reported. The window never changes to favor
a desired rate.

The supplied profiles give these principal fitted slopes:

| Observable | Slope against $N$ |
| --- | ---: |
| Pair, discrete $H_h^2$ | -0.9998721648 |
| Pair, maximum | -1.0042774977 |
| Relative displacement, continuous $H^2$ | -1.9996754678 |
| Relative displacement, maximum | -1.9996632269 |

Since $h=4\pi/(2N+1)$, these observations are consistent with first-order pair
errors and second-order relative-displacement errors for the fixed experiment.
Coarse grids can remain preasymptotic. The stationary states are not certified
global minimizers, and these plots do not prove a second-order convergence
theorem or verify every hypothesis of the manuscript's analytical results.
