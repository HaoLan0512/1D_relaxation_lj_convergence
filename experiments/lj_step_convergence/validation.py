"""Acceptance gates and focused mathematical checks for this experiment.

Endpoint residuals, curvature, and empirical sensitivity budgets have distinct
meanings.  Passing these checks does not certify a global minimizer or turn an
observed convergence rate into a theorem.
"""
from __future__ import annotations

import math
import copy
import time

import gmpy2 as mp
import numpy as np

from .config import (DISPLAY_POINTS, EL_OVER_H2_TOL, ERROR_BUDGET_FRACTION,
                     ERROR_BUDGET_OVER_H2, ERROR_FIELDS, MEAN_TOL, PAIR_CUTOFFS,
                     PRECISION_BITS, REFERENCE_H2_TOL, REFERENCE_MAX_TOL, START_NAME)
from .observables import (FourierProfile, coefficient_norms, coefficient_subtract,
                          compute_observables, continuum_alias_coefficients,
                          direct_relative_error_coefficients,
                          fourier_coefficients_from_samples, loglog_slope,
                          sample_continuum_pair, two_layer_errors)


def reference_profile(reference):
    """Use the authoritative decimal coefficients when evaluating small errors."""
    if not reference["diagnostics"].get("accepted"):
        raise ValueError("The continuum reference has not passed construction checks.")
    return FourierProfile(reference["coefficients"], reference["mp_coefficients"])


def reference_sensitivity(reference):
    """Return the saved degree/precision/independent-method construction changes."""
    diagnostics = reference["diagnostics"]
    return {"h2": float(diagnostics["continuum_h2_sensitivity"]),
            "max": float(diagnostics["continuum_max_sensitivity"])}


def evaluate_state(state, profile):
    """Evaluate native and reconstructed errors without modifying the endpoint."""
    return compute_observables(state["u"], int(state["N"]), profile,
                               u_mp_strings=state["u_mp_strings"])


def endpoint_gate(state):
    """Include the state's requested accuracy factor and resolved eigenscreen."""
    from .atomistic import stationary

    factor = float(state.get("accuracy_factor", 1.0))
    gate = endpoint_acceptance(state["diagnostics"], int(state["N"]))
    gate["accuracy_factor"] = factor
    gate["tightened_stationarity"] = bool(
        np.isfinite(factor) and 0 < factor <= 1 and stationary(state["diagnostics"], factor))
    gate["accepted"] = bool(gate["accepted"] and gate["tightened_stationarity"]
                             and state["curvature"].get("resolved", False))
    return gate


def stationarity_targets(N):
    """EL scales as h^2; physical objective gradients carry a native mesh weight."""
    h = 4 * np.pi / (2 * N + 1)
    target = EL_OVER_H2_TOL * h * h
    full_target = min(2 * np.pi / N, 2 * np.pi / (N + 1)) * target
    return {"atomistic_el_target": float(target),
            "atomistic_full_gradient_target": float(full_target),
            "atomistic_reduced_gradient_target": float(0.5 * full_target)}


def endpoint_acceptance(diagnostics, N):
    """Evaluate independent residual, constraint, and curvature gates.

    Curvature tolerance and eigenvector residuals must be supplied by the
    eigensolver. Missing evidence fails rather than being silently accepted.
    """
    targets = stationarity_targets(N)
    fields = (("el_residual_inf_norm", "atomistic_el_target"),
              ("full_gradient_inf_norm", "atomistic_full_gradient_target"),
              ("reduced_gradient_inf_norm", "atomistic_reduced_gradient_target"))
    checks = {}
    ratios = []
    for field, target in fields:
        value = float(diagnostics.get(field, float("nan")))
        checks[field] = bool(np.isfinite(value) and 0 <= value <= targets[target])
        ratios.append(value / targets[target])
    mean = float(diagnostics.get("mean_sum", float("nan")))
    checks["mean_sum"] = bool(np.isfinite(mean) and abs(mean) <= MEAN_TOL)
    h = 4 * np.pi / (2 * N + 1)
    means = [float(diagnostics.get(name, float("nan"))) for name in ("mean1", "mean2")]
    checks["normalized_means"] = bool(all(np.isfinite(value) and abs(value) <= h / 4 + MEAN_TOL
                                          for value in means))
    smallest = float(diagnostics.get("curvature_min_eigenvalue", float("nan")))
    tolerance = float(diagnostics.get("curvature_tolerance", float("nan")))
    residual = float(diagnostics.get("curvature_residual_max", float("nan")))
    checks["curvature"] = bool(np.isfinite(smallest) and np.isfinite(tolerance)
                                and tolerance > 0 and smallest >= -tolerance
                                and diagnostics.get("curvature_status") in ("soft", "positive"))
    # The negative-curvature band can depend on eigenvector residuals. An
    # independent residual target avoids circular acceptance of that band.
    eigenvector_target = float(diagnostics.get("curvature_eigenvector_target", 1e-8))
    checks["curvature_eigenvectors"] = bool(np.isfinite(residual) and np.isfinite(eigenvector_target)
                                             and eigenvector_target > 0
                                             and 0 <= residual <= eigenvector_target)
    return {"accepted": bool(all(checks.values())), "checks": checks, **targets,
            "stationarity_ratio": float(max(ratios)) if np.all(np.isfinite(ratios)) else float("inf")}


def assess_accuracy_budget(errors, sensitivities, N):
    """Compare measured field changes with min(1e-4*h^2, 5% of the error).

    Each named sensitivity must provide nonnegative finite ``h2`` and ``max``
    values. Required sources are endpoint, continuum, cutoff, reconstruction,
    arithmetic, and maximum_evaluation (whose H2 contribution is identically
    zero because H2 is evaluated by Parseval, not maximum searching).
    """
    required = ("endpoint", "continuum", "cutoff", "reconstruction", "arithmetic",
                "maximum_evaluation")
    h = 4 * np.pi / (2 * N + 1)
    report = {"N": int(N), "sources": sensitivities}
    complete = all(name in sensitivities for name in required)
    for norm, field in (("h2", "relative_h2_error"), ("max", "relative_max_error")):
        error = float(errors.get(field, float("nan")))
        limit = min(ERROR_BUDGET_OVER_H2 * h * h, ERROR_BUDGET_FRACTION * error)
        contributions = [float(sensitivities.get(name, {}).get(norm, float("nan")))
                         for name in required]
        valid = (complete and np.isfinite(error) and error > 0
                 and all(np.isfinite(x) and x >= 0 for x in contributions))
        total = math.fsum(contributions) if valid else float("inf")
        report[norm] = {"error": error, "budget": float(limit), "sensitivity": total,
                        "ratio": total / limit if valid and limit > 0 else float("inf"),
                        "resolved": bool(valid and total <= limit)}
    report["numerically_resolved"] = bool(report["h2"]["resolved"] and report["max"]["resolved"])
    return report


def _check(name, difference, tolerance):
    difference, tolerance = float(difference), float(tolerance)
    return {"name": name, "difference": difference, "tolerance": tolerance,
            "passed": bool(np.isfinite(difference) and difference <= tolerance)}


def run_observable_checks():
    """Small analytical and independent-DFT checks; no optimization or file I/O."""
    checks = []
    # Parseval: ||A sin(kx)||_H2^2 = pi A^2(1+k^2+k^4).
    k, amplitude = 3, 0.2
    coefficients = np.zeros(2 * k + 1, dtype=complex)
    coefficients[0], coefficients[-1] = 0.5j * amplitude, -0.5j * amplitude
    profile = FourierProfile(coefficients)
    norms = coefficient_norms(coefficients)
    expected = math.sqrt(np.pi * amplitude ** 2 * (1 + k ** 2 + k ** 4))
    checks.append(_check("continuous_H2_Parseval", abs(norms["h2"] - expected), 2e-14))
    checks.append(_check("continuous_maximum", abs(norms["max"] - amplitude), 2e-14))
    x = np.linspace(-np.pi, np.pi, 47)
    checks.append(_check("profile_derivative", np.max(np.abs(profile.derivative()(x)
                                                               - k * amplitude * np.cos(k * x))), 2e-14))
    checks.append(_check("profile_phase_shift", np.max(np.abs(profile.shifted(0.137, -0.137)(x)
                                                               - (profile(x - 0.137) - 0.137))), 2e-14))
    # The cosine Nyquist mode survives; its sine vanishes at native samples.
    count = 8
    x = 2 * np.pi * np.arange(count) / count
    nyquist = fourier_coefficients_from_samples(np.cos(count * x / 2))
    target = np.zeros(count + 1, complex)
    target[[0, -1]] = 0.5
    checks.append(_check("even_Nyquist_split", np.max(np.abs(nyquist - target)), 2e-14))
    sine = fourier_coefficients_from_samples(np.sin(count * x / 2))
    checks.append(_check("even_Nyquist_sine", np.max(np.abs(sine)), 2e-14))
    # Fixed polynomial with modes beyond both coarse Nyquist ranges.
    coefficients = np.zeros(25, complex)
    for k, amplitude in ((1, 0.18), (7, 0.04), (12, 0.007)):
        coefficients[12 - k], coefficients[12 + k] = 0.5j * amplitude, -0.5j * amplitude
    profile = FourierProfile(coefficients)
    N = 8
    generator = np.random.default_rng(13)
    u = generator.normal(scale=0.03, size=2 * N + 1) + sample_continuum_pair(profile, N)
    u -= 0.5 * (np.mean(u[:N]) + np.mean(u[N:]))
    observed = compute_observables(u, N, profile)
    direct = direct_relative_error_coefficients(u, N, profile)
    disagreement = coefficient_norms(coefficient_subtract(observed["error_coefficients"], direct))
    checks.append(_check("MP_direct_DFT_vs_residual_with_aliasing", disagreement["h2"], 1e-12))
    checks.append(_check("coarse_aliasing_is_retained",
                         0.0 if observed["physical_alias_h2"] > 0.01 else float("inf"), 0.0))
    delta = np.mean(u[N:]) - np.mean(u[:N])
    binary = two_layer_errors(u, sample_continuum_pair(profile, N, delta), N)
    checks.append(_check("native_pair_norm_parity", abs(binary["h2_error"]
                                                         - observed["phase_matched_h2_error"]), 1e-12))
    # Once both native spaces resolve the saved polynomial, aliasing is zero
    # structurally, rather than an O(eps) cancellation of rounded samples.
    alias = continuum_alias_coefficients(profile, 32, delta=0.137)
    checks.append(_check("resolved_alias_structural_zero", np.max(np.abs(alias)), 0.0))
    for N in (20, 80):
        u = np.random.default_rng(3).normal(scale=0.02, size=2 * N + 1)
        observed = compute_observables(u, N, profile)
        direct = direct_relative_error_coefficients(u, N, profile)
        disagreement = coefficient_norms(coefficient_subtract(observed["error_coefficients"], direct))
        checks.append(_check(f"MP_direct_DFT_N{N}", disagreement["h2"], 3e-11))
    checks.append(_check("negative_slope_convention", abs(loglog_slope([20, 40, 80],
                                                                 [1 / 400, 1 / 1600, 1 / 6400]) + 2), 2e-14))
    # Missing source checks and missing curvature evidence must not pass.
    incomplete = assess_accuracy_budget({"relative_h2_error": 1e-3,
                                         "relative_max_error": 1e-4}, {}, 20)
    checks.append(_check("missing_budget_evidence_rejected", int(incomplete["numerically_resolved"]), 0))
    missing = endpoint_acceptance({}, 20)
    checks.append(_check("missing_endpoint_evidence_rejected", int(missing["accepted"]), 0))
    return {"passed": bool(all(check["passed"] for check in checks)), "checks": checks,
            "count": len(checks)}


def _validate_state(original, N, cutoff, factor, profile, deadline):
    """Refresh force and full-tangent curvature evidence for one fixed state."""
    from . import atomistic as atom
    from .solvers import EIGENPAIR_RESIDUAL_TOL, curvature_screen

    started = time.monotonic()
    atom.check_deadline(deadline)
    if (original.get("N") != N or original.get("pair_cutoff") != cutoff
            or original.get("precision_bits") != PRECISION_BITS
            or original.get("start") != START_NAME
            or original.get("accuracy_factor") != factor):
        raise ValueError(f"N={N}, M={cutoff}: incompatible endpoint identity.")
    native = np.asarray(original["u"])
    strings = original["u_mp_strings"]
    if (native.shape != (2*N+1,) or len(strings) != native.size
            or native.dtype != np.dtype(float) or not np.all(np.isfinite(native))):
        raise ValueError(f"N={N}: invalid native state.")
    values = atom.mp_state(strings)
    if (not all(mp.is_finite(x) for x in values)
            or not np.array_equal(native, np.array([float(x) for x in values]))):
        raise ValueError(f"N={N}: native and multiprecision representations disagree.")
    energy, gradient, diagnostics = atom.mp_value_gradient(values, N, cutoff, deadline=deadline)
    if not mp.is_finite(energy) or not all(mp.is_finite(x) for x in gradient):
        raise ValueError(f"N={N}: nonfinite objective or gradient.")
    curvature, operator, _, _, vectors, eigenvalues = curvature_screen(
        native, N, cutoff, deadline=deadline)
    residuals = np.array([np.linalg.norm(operator @ vector - eigenvalue*vector)
                          for eigenvalue, vector in zip(eigenvalues, vectors.T)])
    orthogonality = float(np.max(abs(vectors.T @ vectors - np.eye(vectors.shape[1]))))
    vector_passed = bool(curvature["resolved"] and np.all(np.isfinite(residuals))
                         and np.max(residuals) <= EIGENPAIR_RESIDUAL_TOL
                         and np.isfinite(orthogonality) and orthogonality <= 1e-8)
    diagnostics.update({name: curvature[name] for name in (
        "curvature_min_eigenvalue", "curvature_tolerance", "curvature_status",
        "curvature_residual_max")})
    diagnostics["curvature_eigenvector_target"] = EIGENPAIR_RESIDUAL_TOL
    state = copy.deepcopy(original)
    state.update({"diagnostics": diagnostics, "curvature": curvature})
    gate = endpoint_gate(state)
    observed = evaluate_state(state, profile)
    finite = bool(observed.get("state_has_mp_values")
                  and observed.get("reference_has_mp_coefficients")
                  and all(np.isfinite(observed[name]) and observed[name] > 0
                          for name in ERROR_FIELDS)
                  and all(np.all(np.isfinite(observed[name])) for name in (
                      "error_coefficients", "native_residual1", "native_residual2")))
    state["accepted"] = bool(gate["accepted"] and vector_passed and finite)
    # The full eigenvectors are ephemeral. The independent residuals and
    # orthogonality defect record the check without duplicating large arrays.
    report = {"N": N, "pair_cutoff": cutoff, "accuracy_factor": factor,
              "endpoint_gate": gate, "diagnostics": diagnostics,
              "curvature": curvature, "independent_eigenpair_residuals": residuals.tolist(),
              "eigenvector_orthogonality_error": orthogonality,
              "representations_agree": True, "finite_observables": finite,
              "state_unchanged": True, "passed": state["accepted"],
              "elapsed_seconds": time.monotonic()-started}
    atom.check_deadline(deadline)
    return state, observed, report


def validate_case(case, reference, deadline=None, progress=None):
    """Revalidate four fixed endpoints and their accuracy budgets for one N.

    Returns ``(refreshed_case, report)``. Numerical arrays, decimal state strings,
    and the authoritative production observables are preserved exactly; only
    scientific diagnostics and sensitivity evidence are refreshed. ``progress``
    may be a callable accepting a short status string. No file I/O or optimizer
    is invoked. Invalid structure raises ValueError; failed scientific gates
    return an unaccepted case and a report with ``passed=False``.
    """
    from . import atomistic as atom
    from .observables import observable_difference

    started = time.monotonic()
    atom.check_deadline(deadline)
    N = int(case["N"])
    keys = [str(cutoff) for cutoff in PAIR_CUTOFFS] + ["tight"]
    if set(case["states"]) != set(keys) or case.get("production_cutoff") != PAIR_CUTOFFS[1]:
        raise ValueError(f"N={N}: all three cutoffs and the tightened production state are required.")
    initial = atom.build_mean_gauge(atom.random_start(N), N)
    if not np.array_equal(np.asarray(case["initial_u"]), initial):
        raise ValueError(f"N={N}: saved initialization is not the deterministic seed-3 start.")
    profile = reference_profile(reference)
    states, observed, endpoint_reports = {}, {}, {}
    for key in keys:
        cutoff = PAIR_CUTOFFS[1] if key == "tight" else int(key)
        factor = 0.1 if key == "tight" else 1.0
        if progress is not None:
            progress(f"N={N}, {key}: fixed-state forces, curvature, and observables.")
        states[key], observed[key], endpoint_reports[key] = _validate_state(
            case["states"][key], N, cutoff, factor, profile, deadline)
    production_key = str(PAIR_CUTOFFS[1])
    endpoint = observable_difference(observed[production_key], observed["tight"])
    cutoff = observable_difference(observed["tight"], observed[str(PAIR_CUTOFFS[2])])
    coarse = observable_difference(observed[str(PAIR_CUTOFFS[0])], observed[production_key])
    sensitivities = {"endpoint": {norm: endpoint[norm] for norm in ("h2", "max")},
                     "continuum": reference_sensitivity(reference),
                     "cutoff": {norm: cutoff[norm] for norm in ("h2", "max")},
                     **observed["tight"]["sensitivities"]}
    budget = assess_accuracy_budget(observed["tight"], sensitivities, N)
    original = case["observables"]
    scalar_differences = {field: abs(observed["tight"][field]-original[field])
                          for field in ERROR_FIELDS}
    fields = observable_difference(observed["tight"], original)
    scalar_tolerances = {field: max(1e-14, 1e-9*abs(original[field])) for field in ERROR_FIELDS}
    # Use the defined budget, not a stored report's unchecked budget value.
    h = 4*np.pi/(2*N+1)
    field_tolerances = {
        norm: 0.01*min(ERROR_BUDGET_OVER_H2*h*h,
                       ERROR_BUDGET_FRACTION*original["relative_"+norm+"_error"])
        for norm in ("h2", "max")}
    same = bool(all(np.isfinite(scalar_differences[field])
                    and scalar_differences[field] <= scalar_tolerances[field]
                    for field in ERROR_FIELDS)
                and all(np.isfinite(fields[norm]) and fields[norm] <= field_tolerances[norm]
                        for norm in ("h2", "max")))
    gates = {(key if key == "tight" else "M"+key): endpoint_gate(state)
             for key, state in states.items()}
    accepted = bool(same and budget["numerically_resolved"]
                    and all(state["accepted"] for state in states.values())
                    and all(gate["accepted"] for gate in gates.values()))
    result = copy.deepcopy(case)
    result.update({"states": states, "accuracy_budget": budget, "endpoint_gates": gates,
                   "tightening_resolved": bool(states["tight"]["accepted"]),
                   "coarse_cutoff_change": coarse, "cutoff_change": cutoff,
                   "endpoint_change": endpoint, "accepted": accepted})
    report = {"N": N, "passed": accepted, "endpoint_count": len(states),
              "initialization_agrees": True, "states_unchanged": True,
              "endpoints": endpoint_reports, "accuracy_budget": budget,
              "observable_agreement": {"passed": same,
                  "scalar_differences": scalar_differences, "scalar_tolerances": scalar_tolerances,
                  "field_differences": fields, "field_tolerances": field_tolerances},
              "elapsed_seconds": time.monotonic()-started}
    atom.check_deadline(deadline)
    return result, report


def validate_reference(reference, deadline=None, reconstruct=False):
    """Check the saved Fourier reference, optionally rebuilding independently.

    Returns a JSON-compatible report. The saved construction sensitivities
    retain their original meaning; fresh equation and coefficient checks are
    reported separately. ``reconstruct=True`` calls ``build_reference`` once
    from scratch and compares coefficients without replacing the saved data.
    """
    import mpmath as mpm
    from . import continuum, model

    started = time.monotonic()
    continuum._deadline(deadline)
    coefficients = np.asarray(reference["coefficients"])
    degree = (coefficients.size-1)//2
    if (coefficients.ndim != 1 or coefficients.size != 2*degree+1 or degree < 1
            or coefficients.dtype != np.dtype(complex) or not np.all(np.isfinite(coefficients))
            or len(reference["mp_coefficients"]) != coefficients.size
            or not np.array_equal(reference["modes"], np.arange(-degree, degree+1))):
        raise ValueError("Invalid saved continuum Fourier representation.")
    checks = []
    with mpm.workprec(PRECISION_BITS):
        exact = [mpm.mpc(real, imaginary) for real, imaginary in reference["mp_coefficients"]]
        if (not all(mpm.isfinite(value) for value in exact)
                or not np.array_equal(coefficients, np.array([complex(value) for value in exact]))):
            raise ValueError("Continuum binary64 and decimal coefficients disagree.")
        constant, potential = model.mp_fourier_coefficients(max_mode=continuum.POTENTIAL_MODES)
        certificate = model.certify_global_minimum(constant, potential)
        fresh = continuum._mp_profile_checks(exact, potential, deadline)
        left = mpm.fsum(c*mpm.exp(-mpm.j*k*mpm.pi)
                        for k, c in zip(range(-degree, degree+1), exact))
        right = mpm.fsum(c*mpm.exp(mpm.j*k*mpm.pi)
                         for k, c in zip(range(-degree, degree+1), exact))
        checks.extend([
            _check("degree_one_endpoint", abs(right-left), REFERENCE_MAX_TOL),
            _check("phase", fresh["phase_defect"], REFERENCE_MAX_TOL),
            _check("oddness", fresh["oddness_coefficient_defect"], REFERENCE_MAX_TOL),
            _check("EL_l2", fresh["el_residual_l2"], REFERENCE_H2_TOL),
            _check("positive_derivative", int(not (
                np.isfinite(fresh["minimum_v_prime"]) and fresh["minimum_v_prime"] > 0)), 0),
        ])
        fresh["minimum_derivative_sign_margin"] = float(certificate["derivative_sign_margin"])
        fresh["potential_tail_bounds"] = {key: float(value)
                                          for key, value in certificate["potential_tail_bounds"].items()}
    diagnostics = reference["diagnostics"]
    if (diagnostics.get("precision_bits") != PRECISION_BITS
            or diagnostics.get("degree") != degree
            or diagnostics.get("display_points") != DISPLAY_POINTS):
        raise ValueError("Saved continuum precision, degree, or display-grid identity differs.")
    checks.extend([
        _check("construction_H2_sensitivity", diagnostics["continuum_h2_sensitivity"], REFERENCE_H2_TOL),
        _check("construction_max_sensitivity", diagnostics["continuum_max_sensitivity"], REFERENCE_MAX_TOL),
    ])
    x = np.linspace(-np.pi, np.pi, DISPLAY_POINTS)
    q, dq, d2q = continuum._uniform_float_profile(coefficients, DISPLAY_POINTS-1)
    q, dq, d2q = (np.r_[values, values[0]] for values in (q, dq, d2q))
    expected = {"x_display": x, "q_display": q, "v_display": x+q,
                "dv_display": 1+dq, "d2v_display": d2q}
    for key, values in expected.items():
        saved = np.asarray(reference[key])
        if saved.shape != values.shape or not np.all(np.isfinite(saved)):
            raise ValueError(f"Invalid continuum display array {key}.")
        # Display values are binary64 evaluations, not the accuracy reference.
        tolerance = 16*np.finfo(float).eps*max(1, float(np.max(abs(values))))
        checks.append(_check(key+"_evaluation", np.max(abs(saved-values)), tolerance))
    report = {"passed": bool(all(check["passed"] for check in checks)), "checks": checks,
              "fresh_profile_diagnostics": fresh, "reference_unchanged": True,
              "construction_sensitivities": reference_sensitivity(reference),
              "construction_evidence": "Saved degree, precision, and independent Fourier comparisons; not rerun unless independent_reconstruction is present.",
              "independent_reconstruction": None}
    if reconstruct:
        rebuilt = continuum.build_reference(deadline=deadline)
        with mpm.workprec(PRECISION_BITS):
            candidate = [mpm.mpc(real, imaginary) for real, imaginary in rebuilt["mp_coefficients"]]
            difference = continuum._coefficient_norms(exact, candidate)
            comparison = {key: float(value) for key, value in difference.items()}
        independent = {"passed": bool(rebuilt["diagnostics"]["accepted"]
                                       and comparison["h2"] <= REFERENCE_H2_TOL
                                       and comparison["max"] <= REFERENCE_MAX_TOL),
                       "coefficient_difference": comparison,
                       "diagnostics": rebuilt["diagnostics"]}
        report["independent_reconstruction"] = independent
        report["passed"] = bool(report["passed"] and independent["passed"])
    report["elapsed_seconds"] = time.monotonic()-started
    continuum._deadline(deadline)
    return report
