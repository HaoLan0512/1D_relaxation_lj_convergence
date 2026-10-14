"""High-precision scalar continuum reference with independent Fourier checks.

The autonomous equation v''=W'(v) has first integral
v'^2=2*(b+W(v)-min(W)). Its period determines b>0. We invert the scalar
travel integral at genuinely multiprecision coordinates and retain Fourier
coefficients, rather than promoting a binary64 profile to high precision.
"""

import math
import time

import mpmath as mp
import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.linalg import solve
from scipy.optimize import brentq

from .config import (
    CHECK_PRECISION_BITS, DISPLAY_POINTS, PARAMETERS, PRECISION_BITS,
    REFERENCE_DEGREES, REFERENCE_H2_TOL, REFERENCE_MAX_TOL,
)
from .model import (
    certify_global_minimum, fourier_Wprime, fourier_Wsecond,
    mp_fourier_coefficients,
)


POTENTIAL_MODES = 24


def _deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Continuum reference exceeded the shared computation deadline.")


def _emit(progress, message):
    if progress is not None:
        progress(message)


def _mp_fft(values, inverse=False):
    """Radix-two FFT at the current MP precision, with the NumPy convention."""
    n = len(values)
    if n < 1 or n & (n - 1):
        raise ValueError("The MP FFT length must be a power of two.")
    result = [mp.mpc(v) for v in values]
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            result[i], result[j] = result[j], result[i]
    width = 2
    while width <= n:
        root = mp.exp((2j if inverse else -2j) * mp.pi / width)
        for start in range(0, n, width):
            factor = mp.mpc(1)
            for offset in range(width // 2):
                first = result[start + offset]
                second = factor * result[start + offset + width // 2]
                result[start + offset] = first + second
                result[start + offset + width // 2] = first - second
                factor *= root
        width *= 2
    return [z / n for z in result] if inverse else result


def _mp_force(s, amplitudes):
    sine, cosine = mp.sin(s), mp.cos(s)
    previous, current = mp.mpf(0), sine
    result = mp.mpf(0)
    for k, amplitude in enumerate(amplitudes, 1):
        result -= k * amplitude * current
        previous, current = current, 2 * cosine * current - previous
    return result


def _mp_difference_from_pi(s, amplitudes):
    """Evaluate W(s)-W(pi) without subtracting its large constant mode."""
    angle = (mp.pi - s) / 2
    sine, cosine = mp.sin(angle), mp.cos(angle)
    previous, current = mp.mpf(0), sine
    result = mp.mpf(0)
    for k, amplitude in enumerate(amplitudes, 1):
        result += -2 * (-1)**k * amplitude * current**2
        previous, current = current, 2 * cosine * current - previous
    return result


def _coefficient_norms(first, second=None):
    """MP continuous H2 norm and a Fourier l1 maximum bound of a difference."""
    degree = max((len(first) - 1) // 2, 0 if second is None else (len(second) - 1) // 2)
    entries = [mp.mpc(0) for _ in range(2 * degree + 1)]
    for sign, coefficients in ((1, first), (-1, second)):
        if coefficients is None:
            continue
        local_degree = (len(coefficients) - 1) // 2
        for j, coefficient in enumerate(coefficients):
            entries[degree - local_degree + j] += sign * coefficient
    h2 = mp.sqrt(2 * mp.pi * mp.fsum(
        (1 + k*k + k**4) * abs(c)**2
        for k, c in zip(range(-degree, degree + 1), entries)))
    maximum_bound = mp.fsum(abs(c) for c in entries)
    return {"h2": h2, "max": maximum_bound}


def _float_seed(amplitudes):
    """A binary64 trajectory is used only to initialize MP inverse solves."""
    amplitudes = np.array([float(a) for a in amplitudes])
    modes = np.arange(1, len(amplitudes) + 1)
    parity = (-1.)**modes

    def difference(s):
        return float(np.sum(-2 * parity * amplitudes * np.sin(modes * (np.pi - s) / 2)**2))

    def period(log_b):
        b = np.exp(log_b)
        return 2 * quad(lambda s: 1 / np.sqrt(2 * (b + difference(s))),
                        0, np.pi, epsabs=2e-11, epsrel=2e-11)[0]

    lower = -2.
    while period(lower) < 2 * np.pi:
        lower *= 2
        if lower < -500:
            raise RuntimeError("Scalar period root is beyond this experiment's resolved range.")
    log_b = brentq(lambda z: period(z) - 2 * np.pi, lower, math.log(.5), xtol=1e-13)
    b = math.exp(log_b)

    def rhs(x, q):
        speed2 = 2 * (b + difference(x + q[0]))
        return [(speed2 - 1) / (np.sqrt(speed2) + 1)]

    trajectory = solve_ivp(rhs, (0, np.pi), [0.], method="DOP853",
                           rtol=3e-13, atol=1e-14, max_step=.025, dense_output=True)
    if not trajectory.success:
        raise RuntimeError("The scalar initial-guess trajectory did not finish.")
    return b, trajectory


class _TravelReference:
    """One precision's period solve and cached, MP inverse-travel samples."""

    def __init__(self, bits, deadline=None):
        self.bits = bits
        self.deadline = deadline
        self.calls = 0
        self.samples = {}
        self.constant, self.amplitudes = mp_fourier_coefficients(PARAMETERS, POTENTIAL_MODES)
        self.minimum = certify_global_minimum(self.constant, self.amplitudes, PARAMETERS)
        self.curvature = -mp.fsum(
            k*k * (-1)**k * a for k, a in enumerate(self.amplitudes, 1))
        if self.curvature <= 0:
            raise RuntimeError("The verified minimum is not nondegenerate.")
        float_b, self.seed = _float_seed(self.amplitudes)

        def equation(log_b):
            _deadline(deadline)
            b = mp.exp(log_b)
            beta = mp.sqrt(2 * b / self.curvature)
            ymax = mp.asinh(mp.pi / beta)
            return self._integral(ymax, b, beta) - mp.pi

        initial = mp.log(mp.mpf(str(float_b)))
        self.log_b = mp.findroot(equation, (initial - mp.mpf(".05"), initial + mp.mpf(".05")),
                                tol=mp.eps * 32, maxsteps=30, verify=True)
        self.b = mp.exp(self.log_b)
        self.beta = mp.sqrt(2 * self.b / self.curvature)
        self.ymax = mp.asinh(mp.pi / self.beta)
        self.period_residual = 2 * (self._integral(self.ymax, self.b, self.beta) - mp.pi)
        self.maximum_inverse_residual = mp.mpf(0)

    def _integrand(self, y, b, beta):
        self.calls += 1
        if self.calls % 256 == 0:
            _deadline(self.deadline)
        displacement = beta * mp.sinh(y)
        s = mp.pi - displacement
        return beta * mp.cosh(y) / mp.sqrt(
            2 * (b + _mp_difference_from_pi(s, self.amplitudes)))

    def _integral(self, y, b=None, beta=None):
        b = self.b if b is None else b
        beta = self.beta if beta is None else beta
        return mp.quad(lambda t: self._integrand(t, b, beta), [0, y],
                       method="gauss-legendre")

    def sample(self, numerator, denominator):
        """Return q at x=pi*numerator/denominator using MP pi and inversion."""
        common = math.gcd(numerator, denominator)
        key = (numerator // common, denominator // common)
        if key in self.samples:
            return self.samples[key]
        if numerator == 0 or numerator == denominator:
            self.samples[key] = mp.mpf(0)
            return self.samples[key]
        _deadline(self.deadline)
        x = mp.pi * numerator / denominator
        float_x = float(x)
        float_v = float_x + float(self.seed.sol(float_x)[0])
        y = mp.asinh((mp.pi - mp.mpf(str(float_v))) / self.beta)
        target = mp.pi - x
        tolerance = 256 * mp.eps
        for _ in range(10):
            residual = self._integral(y) - target
            if abs(residual) <= tolerance:
                break
            correction = residual / self._integrand(y, self.b, self.beta)
            candidate = y - correction
            if not 0 < candidate < self.ymax:
                raise RuntimeError("MP inverse-travel Newton step left its monotone interval.")
            y = candidate
        else:
            raise RuntimeError("MP inverse-travel quadrature did not resolve a requested sample.")
        self.maximum_inverse_residual = max(self.maximum_inverse_residual, abs(residual))
        q = mp.pi - self.beta * mp.sinh(y) - x
        self.samples[key] = q
        return q

    def coefficients(self, degree):
        # Four times the retained degree separates Fourier truncation from the
        # sampling Nyquist limit. Nested grids reuse previously inverted nodes.
        count = 4 * degree
        half = count // 2
        values = [mp.mpf(0)] * count
        for j in range(1, half):
            value = self.sample(j, half)
            values[j], values[count - j] = value, -value
        transformed = _mp_fft(values)
        positive = [mp.mpc(0, mp.im(transformed[k]) / count) for k in range(1, degree + 1)]
        return [mp.conj(c) for c in reversed(positive)] + [mp.mpc(0)] + positive


def _sine_coefficients_to_samples(amplitudes, count):
    buffer = [mp.mpc(0)] * count
    for k, amplitude in enumerate(amplitudes, 1):
        buffer[k] = -mp.j * amplitude * count / 2
        buffer[-k] = mp.j * amplitude * count / 2
    return [mp.re(z) for z in _mp_fft(buffer, inverse=True)]


def _independent_fourier_newton(degree, potential, deadline=None):
    """Independent zero-start continuation, then MP-residual Newton polishing.

    The binary64 Jacobian solves provide correction preconditioners; nonlinear
    residuals and all final coefficients are recomputed at the active MP
    precision. No scalar-reference coefficients initialize this computation.
    """
    count = 4 * degree
    modes = np.arange(1, degree + 1, dtype=float)
    x = 2 * np.pi * np.arange(count) / count
    sine = np.sin(x[:, None] * modes)
    float_potential = np.array([float(a) for a in potential])
    amplitudes = np.zeros(degree)
    continuation_iterations = []
    for scale in (1/32, 1/16, 1/8, 1/4, 1/2, 1):
        for iteration in range(24):
            _deadline(deadline)
            q = sine @ amplitudes
            force = scale * fourier_Wprime(x + q, float_potential)
            residual = modes**2 * amplitudes + (2 / count) * sine.T @ force
            curvature = scale * fourier_Wsecond(x + q, float_potential)
            jacobian = np.diag(modes**2) + (2 / count) * sine.T @ (curvature[:, None] * sine)
            correction = solve(jacobian, -residual, assume_a="sym")
            amplitudes += correction
            if np.max(np.abs(correction)) < 3e-15:
                break
        else:
            raise RuntimeError("Independent continuum Newton continuation did not converge.")
        continuation_iterations.append(iteration + 1)

    mp_amplitudes = [mp.mpf(float(a)) for a in amplitudes]
    mp_x = [2 * mp.pi * j / count for j in range(count)]
    corrections = []
    for _ in range(8):
        _deadline(deadline)
        q = _sine_coefficients_to_samples(mp_amplitudes, count)
        force = [_mp_force(t + value, potential) for t, value in zip(mp_x, q)]
        transformed = _mp_fft(force)
        residual = [k*k * a - 2 * mp.im(transformed[k]) / count
                    for k, a in enumerate(mp_amplitudes, 1)]
        float_q = np.array([float(value) for value in q])
        curvature = fourier_Wsecond(x + float_q, float_potential)
        jacobian = np.diag(modes**2) + (2 / count) * sine.T @ (curvature[:, None] * sine)
        correction = solve(jacobian, -np.array([float(r) for r in residual]), assume_a="sym")
        mp_amplitudes = [a + mp.mpf(float(c)) for a, c in zip(mp_amplitudes, correction)]
        correction_h2 = mp.sqrt(mp.pi * mp.fsum(
            (1 + k*k + k**4) * mp.mpf(float(c))**2
            for k, c in enumerate(correction, 1)))
        corrections.append(correction_h2)
        if correction_h2 < mp.power(2, -mp.mp.prec + 18):
            break
    else:
        raise RuntimeError("Independent MP Fourier Newton residual polishing failed.")
    positive = [-mp.j * a / 2 for a in mp_amplitudes]
    coefficients = [mp.conj(c) for c in reversed(positive)] + [mp.mpc(0)] + positive
    return coefficients, {
        "continuation_iterations": continuation_iterations,
        "mp_polishing_iterations": len(corrections),
        "last_mp_correction_h2": float(corrections[-1]),
        "method": "independent_zero_start_sine_fourier_newton_with_mp_residual_polishing",
    }


def _mp_profile_checks(coefficients, potential, deadline=None):
    """Check the persisted Fourier profile itself on an oversampled MP grid."""
    degree = (len(coefficients) - 1) // 2
    count = 8 * degree
    q_buffer = [mp.mpc(0)] * count
    derivative_buffer = [mp.mpc(0)] * count
    second_buffer = [mp.mpc(0)] * count
    for k, coefficient in zip(range(-degree, degree + 1), coefficients):
        q_buffer[k % count] = count * coefficient
        derivative_buffer[k % count] = count * mp.j * k * coefficient
        second_buffer[k % count] = -count * k*k * coefficient
    q = [mp.re(z) for z in _mp_fft(q_buffer, inverse=True)]
    derivative = [1 + mp.re(z) for z in _mp_fft(derivative_buffer, inverse=True)]
    second = [mp.re(z) for z in _mp_fft(second_buffer, inverse=True)]
    residual = []
    for j in range(count):
        if j % 128 == 0:
            _deadline(deadline)
        residual.append(second[j] - _mp_force(2 * mp.pi * j / count + q[j], potential))
    return {
        "el_residual_inf": float(max(abs(r) for r in residual)),
        "el_residual_l2": float(mp.sqrt(2 * mp.pi / count * mp.fsum(r*r for r in residual))),
        "minimum_v_prime": float(min(derivative)),
        "phase_defect": float(abs(mp.fsum(coefficients))),
        "oddness_coefficient_defect": float(max(abs(mp.re(c)) for c in coefficients)),
        "periodicity": "exact_by_integer_fourier_modes",
        "residual_grid_points": count,
    }


def _uniform_float_profile(coefficients, count):
    """Evaluate q and its first two derivatives by FFT on [-pi,pi)."""
    degree = (len(coefficients) - 1) // 2
    modes = np.arange(-degree, degree + 1)
    results = []
    for order in range(3):
        buffer = np.zeros(count, complex)
        buffer[modes % count] = count * coefficients * (-1.)**modes * (1j * modes)**order
        results.append(np.fft.ifft(buffer).real)
    return results


def build_reference(deadline=None, progress=None):
    """Build the cached-reference payload without writing files or old outputs.

    Returned MP coefficients are decimal (real, imaginary) string pairs in
    signed-mode order. Sensitivities are measured comparisons, not certified
    total-error bounds. An unresolved requested target raises an exception.
    """
    started = time.monotonic()
    degree_comparisons = []
    with mp.workprec(PRECISION_BITS):
        _emit(progress, f"Continuum: {PRECISION_BITS}-bit scalar period solve.")
        production = _TravelReference(PRECISION_BITS, deadline)
        previous = None
        for degree in REFERENCE_DEGREES:
            _emit(progress, f"Continuum: inverse quadrature and MP Fourier degree {degree}.")
            coefficients = production.coefficients(degree)
            if previous is not None:
                change = _coefficient_norms(coefficients, previous)
                degree_comparisons.append({
                    "degree": degree, "h2": float(change["h2"]), "max": float(change["max"])})
                if (change["h2"] <= REFERENCE_H2_TOL / 4
                        and change["max"] <= REFERENCE_MAX_TOL / 4):
                    break
            previous = coefficients
        else:
            raise RuntimeError("Continuum Fourier degree refinement did not resolve the targets.")
        production_coefficients = list(coefficients)
        production_degree = degree
        diagnostics = {
            "precision_bits": PRECISION_BITS, "check_precision_bits": CHECK_PRECISION_BITS,
            "degree": degree, "potential_modes": POTENTIAL_MODES,
            "degree_comparisons": degree_comparisons,
            "travel_b": mp.nstr(production.b, 45),
            "travel_constant_C": mp.nstr(production.b - production.minimum["value"], 45),
            "period_residual": float(abs(production.period_residual)),
            "maximum_inverse_travel_residual": float(production.maximum_inverse_residual),
            "minimum_certificate_method": production.minimum["method"],
            "minimum_location": "pi",
            "minimum_derivative_sign_margin": float(production.minimum["derivative_sign_margin"]),
            "potential_tail_bounds": {
                key: float(value) for key, value in production.minimum["potential_tail_bounds"].items()},
            "mp_quadrature_evaluations": production.calls,
        }

    with mp.workprec(CHECK_PRECISION_BITS):
        _emit(progress, f"Continuum: independent {CHECK_PRECISION_BITS}-bit travel calculation.")
        precision_check = _TravelReference(CHECK_PRECISION_BITS, deadline)
        check_coefficients = precision_check.coefficients(production_degree)

    with mp.workprec(PRECISION_BITS):
        precision_change = _coefficient_norms(production_coefficients, check_coefficients)
        _emit(progress, "Continuum: independent Fourier Newton validation.")
        independent, newton_diagnostics = _independent_fourier_newton(
            production_degree, production.amplitudes, deadline)
        independent_change = _coefficient_norms(production_coefficients, independent)
        profile_checks = _mp_profile_checks(production_coefficients, production.amplitudes, deadline)
        sensitivity_h2 = (degree_comparisons[-1]["h2"] + float(precision_change["h2"])
                          + float(independent_change["h2"]))
        sensitivity_max = (degree_comparisons[-1]["max"] + float(precision_change["max"])
                           + float(independent_change["max"]))
        diagnostics.update({
            "precision_comparison": {key: float(value) for key, value in precision_change.items()},
            "check_precision_period_residual": float(abs(precision_check.period_residual)),
            "check_precision_inverse_travel_residual": float(precision_check.maximum_inverse_residual),
            "independent_newton_comparison": {key: float(value) for key, value in independent_change.items()},
            "independent_newton": newton_diagnostics,
            "continuum_h2_sensitivity": sensitivity_h2,
            "continuum_max_sensitivity": sensitivity_max,
            "sensitivity_interpretation": "sum_of_measured_refinement_and_independent_method_changes",
            **profile_checks,
        })
        accepted = (sensitivity_h2 <= REFERENCE_H2_TOL
                    and sensitivity_max <= REFERENCE_MAX_TOL
                    and profile_checks["minimum_v_prime"] > 0
                    and profile_checks["el_residual_l2"] <= REFERENCE_H2_TOL
                    and profile_checks["phase_defect"] <= REFERENCE_MAX_TOL)
        diagnostics["accepted"] = bool(accepted)
        if not accepted:
            raise RuntimeError(f"Continuum reference validation did not resolve its targets: {diagnostics}")
        coefficient_strings = [
            (mp.nstr(mp.re(c), 45), mp.nstr(mp.im(c), 45)) for c in production_coefficients]
        float_coefficients = np.array([complex(c) for c in production_coefficients])

    count = DISPLAY_POINTS - 1
    x = np.linspace(-np.pi, np.pi, DISPLAY_POINTS)
    q, dq, d2q = _uniform_float_profile(float_coefficients, count)
    q, dq, d2q = (np.r_[values, values[0]] for values in (q, dq, d2q))
    diagnostics["elapsed_seconds"] = time.monotonic() - started
    diagnostics["display_points"] = DISPLAY_POINTS
    _emit(progress, f"Continuum reference accepted at degree {production_degree}.")
    return {
        "coefficients": float_coefficients,
        "mp_coefficients": coefficient_strings,
        "modes": np.arange(-production_degree, production_degree + 1),
        "x_display": x, "q_display": q, "v_display": x + q,
        "dv_display": 1 + dq, "d2v_display": d2q,
        "diagnostics": diagnostics,
    }
