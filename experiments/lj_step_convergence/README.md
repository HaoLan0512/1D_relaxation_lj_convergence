# LJ step profile and convergence: mathematical companion

This guide explains the model, numerical construction, error definitions, and
validation behind the supplied results. For
[installation](../../README.md#download-and-run),
[commands](../../README.md#commands),
[figure downloads](../../README.md#generated-files), and
[citation](../../README.md#citation), use the root README.

- [Model and continuum reference](#model-and-continuum-reference)
- [Atomistic states and error definitions](#atomistic-states-and-error-definitions)
- [Validation and convergence interpretation](#validation-and-convergence-interpretation)
- [Changing LJ parameters and fresh calculation](#changing-lj-parameters-and-fresh-calculation)
- [Source map](#source-map)

## Model and continuum reference

Lengths use a registry period of $2\pi$. With layer separation $d$
(`interlayer_distance`) and horizontal separation $s$, the fixed Lennard-Jones
model is

```math
\begin{gathered}
(d,\sigma,\epsilon)=(2\pi,1.8\pi,16),\\
V(s)=4\epsilon\left(\frac{\sigma^{12}}{(d^2+s^2)^6}
-\frac{\sigma^6}{(d^2+s^2)^3}\right),\qquad
W(s)=4\sum_{m\in\mathbb Z}V(s-2\pi m).
\end{gathered}
```

Parameters in [config.py](config.py) are exact expressions evaluated at each
arithmetic precision; high-precision calculations do not inherit rounded
binary64 parameters.

Write $q=v_0-x$. The normalized continuum profile satisfies

```math
\begin{gathered}
v_0(0)=0,\qquad v_0(x+2\pi)=v_0(x)+2\pi,\\
\\
v_0'=\sqrt{2(W(v_0)+c)}, \qquad
\int_{-\pi}^{\pi}\frac{ds}{\sqrt{2(W(s)+c)}}=2\pi.
\end{gathered}
```

We evaluate $W$ from its analytic Fourier series and determine $c$ from the
traversal-time equation. The Fourier degree of $q=v_0-x$ increases through 64, 128, 256, and 512 as needed. The supplied calculation stops at degree 256 after comparison with degree 128

Arithmetic precision is separate: calculations use 128 bits and are checked at
80 bits at the same degree. An independent Fourier Newton solution provides a
further check. The summed changes from degree, precision, and method comparisons
must not exceed $10^{-13}$ in $H^2$ and $10^{-14}$ in maximum norm. Normalization,
periodicity, oddness, positive derivative, and the Euler--Lagrange residual are
also checked.

Figure 1 shows $v_0$, with inset $u_1-u_2=q=v_0-x$. Its 32,769 display samples
control rendering density, not solution accuracy. The minimum derivative is
approximately 0.016483903; transition from $-0.8\pi$ to $0.8\pi$ occupies
approximately 25.2865% of the period, leaving broad plateaus.

## Atomistic states and error definitions

### Native states

The layers contain $N$ and $N+1$ atoms, respectively, for

```text
N = 20, 40, 80, 160, 320, 640, 1280, 2560, 3840, 5120, 7680, 10240.
```

Each size uses a deterministic seed-3, five-mode Fourier initialization with
maximum amplitude 0.1. L-BFGS is followed by matrix-free Newton corrections using
preconditioned MINRES and three low curvature modes from LOBPCG. The arithmetic layer means satisfy $\langle u_d^1\rangle+\langle u_d^2\rangle=0$; their individual values remain free. Corrections use 128-bit forces and saved
high-precision state strings. Native corrections are capped at 0.05 and accepted
through energy and full-residual backtracking. A conditional tighter MINRES
retry must pass the same independently measured linear-residual criterion.

The interaction cutoff $M$ limits atomistic-sum offsets to $|m|\le M$. For each
$N$, the solver starts at $M=160$, then refines at $M=320$ and $640$, saving all
three states. A fourth state uses $M=320$ with tenfold tighter solver targets
and supplies the plotted errors. Comparing it with the original $M=320$ state
measures solver sensitivity; comparing it with $M=640$ measures cutoff sensitivity.

### Phase matching and pair errors

For each fixed atomistic state, use arithmetic means on its native grids:

```math
\begin{gathered}
\delta_N=\langle u_d^2\rangle-\langle u_d^1\rangle,\\
q_{\delta_N}(x)=q(x-\delta_N)-\delta_N.
\end{gathered}
```

Pair errors compare the native states with
$(q_{\delta_N}/2,-q_{\delta_N}/2)$ on their respective grids.
Figure 2(a) uses the discrete $H_h^2$ norm, combining layer-weighted
displacement, forward-difference, and centered second-difference terms.
The differences and weights use each layer's own spacing, $2\pi/N$ or
$2\pi/(N+1)$. Figure 2(b) takes the largest absolute nodal error over both layers.

### Reconstructed relative-displacement errors

Let $\mathcal T_N$ and $\mathcal T_{N+1}$ denote periodic Fourier reconstruction
from the two native grids. The relative-displacement error is

```math
e_N^{\mathrm{rel}}(x)=\mathcal T_Nu_d^1(x)
-\mathcal T_{N+1}u_d^2(x)-q_{\delta_N}(x).
```

Figure 3(a) measures its continuous $H^2$ norm and Figure 3(b) its maximum norm.
“Relative” means the difference between layers, not division by a reference norm.
Reconstructions use the same undeformed coordinate, all native Fourier modes,
and the real even-grid Nyquist convention.

Aliasing from finite-grid sampling is included in the plotted error, not in
reconstruction sensitivity. Reconstruction sensitivity measures only the
disagreement between two numerical evaluations of that same error.

## Validation and convergence interpretation

### Acceptance and sensitivity

Every endpoint must have finite full and reduced gradients, satisfy the
mean-sum constraint, and pass curvature/eigenvector-residual checks. The
Euler--Lagrange criterion is $\|\mathrm{EL}\|_\infty\le10^{-5}h^2$, together
with the corresponding mesh-weighted gradient targets.

For each plotted relative norm $E_N$, the total sensitivity must fit within

```math
B_N=\min(10^{-4}h^2,0.05E_N).
```

Endpoint refinement, continuum refinement, cutoff changes, reconstruction,
arithmetic, and maximum evaluation are recorded separately. These are numerical
sensitivities, not rigorous bounds.
[verification.json](../../outputs/lj_step_convergence/verification.json) and the
[diagnostic and budget tables](../../outputs/lj_step_convergence/tables/) report
the evidence; no missing size is extrapolated or silently accepted.

### Rates and limitations

Each convergence panel shows one green data curve and an offset gray rate guide:
$CN^{-1}$ for pair errors and $CN^{-2}$ for relative-displacement errors.
These guides are not fitted curves. The principal slope window is fixed at
$N=1280,2560,5120,10240$; all-point fits and adjacent rates are also reported.

The supplied [fitted slopes](../../outputs/lj_step_convergence/tables/fitted_slopes.csv)
are:

| Observable | Principal slope against $N$ |
| --- | ---: |
| Pair, discrete $H_h^2$ | -0.9998721648 |
| Pair, maximum | -1.0042774977 |
| Relative displacement, continuous $H^2$ | -1.9996754678 |
| Relative displacement, maximum | -1.9996632269 |

Since $h=4\pi/(2N+1)$, this experiment presents numerical validation for
first-order convergence of the computed atomistic displacement and numerical
evidence for second-order convergence of the reconstructed relative displacement.
These results do not prove a second-order convergence theorem.

## Changing LJ parameters and fresh calculation

1. **Set the LJ parameters.** Open
   [`experiments/lj_step_convergence/config.py`](config.py), find
   `EffectiveLJParameters`, and replace the quoted values you want to change:

   ```python
   epsilon: str = "16"
   sigma: str = "1.8*pi"
   interlayer_distance: str = "2*pi"
   ```

   These are the supplied values. Keep each value as a quoted positive decimal,
   optionally followed by `*pi`; for example, `"20"` or `"1.9*pi"`.

2. **Compute the new profiles.** From the repository root, with the
   [installed environment active](../../README.md#download-and-run), run:

   ```text
   python main.py --prepare --data-dir outputs/lj_step_convergence_custom
   ```

   Choose a new or empty output folder; the program creates it if needed.
   Use a separate folder for each parameter set to preserve the supplied results.

3. **Results.** After the continuum reference and all twelve atomistic
   sizes pass validation, the program saves the profiles and automatically
   generates figures and tables under `outputs/lj_step_convergence_custom/`.
   Open the PNG or PDF files in its `figures/` subfolder. Metadata is written
   automatically. To regenerate the figures later from these saved profiles, run:

   ```text
   python main.py --data-dir outputs/lj_step_convergence_custom
   ```

**Remarks:**

- Fresh preparation is expensive and has a cumulative eight-hour numerical
  budget. Restarting the command does not reset that budget.
- To resume unfinished work, repeat the preparation command while budget remains,
  keeping the experiment's Python files and parameter settings unchanged.
  Checkpoints remain in the selected folder's `preparation/` subfolder until
  publication and rendering succeed.
- Keep the matching parameter values in `config.py` when using the new dataset.
  To use the supplied dataset again, restore the three original values shown
  above. Selecting a data folder does not change the configuration automatically.

## Source map

| Component | Source and responsibility |
| --- | --- |
| Configuration and potential | [config.py](config.py): exact parameters, grids, seeds, tolerances, limits; [model.py](model.py): derivatives, Fourier periodization, tails |
| Continuum reference | [continuum.py](continuum.py): scalar construction and independent Fourier check |
| Atomistic construction | [atomistic.py](atomistic.py): lattices, energy, gradients, initialization, Hessian actions; [solvers.py](solvers.py): L-BFGS, Newton, curvature |
| Error definitions | [observables.py](observables.py): phase matching, Fourier reconstruction, norms |
| Validation | [validation.py](validation.py): saved-state checks and sensitivity budgets |
| Data and orchestration | [storage.py](storage.py): pickle-free profiles, checkpoints, hashes, publication; [main.py](main.py): rendering, preparation, verification |
| Presentation | [plotting.py](plotting.py): figures and tables |