"""Phase alignment and native/Fourier error norms for the two unequal grids.

The definitions follow ``two_layer_errors``, ``FourierProfile``, and
``relative_observables`` in the September 2026 H2-rate experiment.  The new
evaluation forms small native residuals in MPFR before using a binary64 FFT.
It keeps every native mode, including the split even-grid Nyquist mode, and
adds the continuum interpolation alias exactly as a coefficient operation.
No smoothing, mode filtering, or division by a reference norm is performed.
"""
from __future__ import annotations

import math

import gmpy2 as mp
import numpy as np
from scipy import fft as scipy_fft
from scipy.optimize import brentq


TWOPI = 2.0 * np.pi


def _mp_complex(real, imag=0):
    return mp.mpc(mp.mpfr(real), mp.mpfr(imag))


def _mp_strings(coefficients):
    return np.asarray([(format(z.real, ".50g"), format(z.imag, ".50g"))
                       for z in coefficients], dtype=str)


class FourierProfile:
    """Real periodic polynomial, with coefficients ordered from -degree to +degree.

    ``mp_coefficients`` optionally supplies decimal real/imaginary pairs from
    the high-precision solve.  Promoting binary64 coefficients to MPFR cannot
    restore their lost digits; the ``has_mp_coefficients`` flag records this.
    """

    def __init__(self, coefficients, mp_coefficients=None):
        self.coefficients = np.asarray(coefficients, dtype=complex).copy()
        self.degree = (self.coefficients.size - 1) // 2
        if (self.coefficients.ndim != 1 or self.coefficients.size == 0
                or self.coefficients.size != 2 * self.degree + 1
                or not np.all(np.isfinite(self.coefficients))):
            raise ValueError("Use a finite, nonempty, odd-length coefficient vector.")
        self.has_mp_coefficients = mp_coefficients is not None
        if mp_coefficients is None:
            # Decimal strings with 17 significant digits round-trip binary64.
            self.mp_coefficients = np.asarray(
                [(format(z.real, ".17g"), format(z.imag, ".17g"))
                 for z in self.coefficients], dtype=str)
        else:
            self.mp_coefficients = np.asarray(mp_coefficients, dtype=str).copy()
            if self.mp_coefficients.shape != (self.coefficients.size, 2):
                raise ValueError("mp_coefficients must contain a real/imag pair per mode.")

    def __call__(self, x):
        coordinates = np.asarray(x, dtype=float)
        d = self.degree
        # Taking the real part combines negative and positive modes without
        # allocating a point-by-mode exponential matrix.
        positive = self.coefficients[d:].copy()
        if d:
            positive[1:] += self.coefficients[d - 1::-1].conjugate()
        values = np.polynomial.polynomial.polyval(np.exp(1j * coordinates), positive).real
        return float(values) if coordinates.ndim == 0 else values

    def derivative(self, order=1):
        if int(order) != order or order < 0:
            raise ValueError("Derivative order must be a nonnegative integer.")
        modes = np.arange(-self.degree, self.degree + 1)
        coefficients = self.coefficients * (1j * modes) ** int(order)
        with mp.context(precision=128):
            high = [z * _mp_complex(0, int(k)) ** int(order)
                    for z, k in zip(self._mp_coefficients(), modes)]
            strings = _mp_strings(high) if self.has_mp_coefficients else None
        return FourierProfile(coefficients, strings)

    def shifted(self, delta, offset=0.0):
        """Return ``x -> self(x-delta) + offset``; preserve MP phase arithmetic."""
        with mp.context(precision=128):
            high = _shifted_mp_coefficients(self, mp.mpfr(delta), mp.mpfr(offset))
            coefficients = np.asarray([complex(z) for z in high])
            strings = _mp_strings(high) if self.has_mp_coefficients else None
        return FourierProfile(coefficients, strings)

    def _mp_coefficients(self):
        return [_mp_complex(real, imag) for real, imag in self.mp_coefficients]


def coefficient_pad(coefficients, degree):
    """Embed centered signed coefficients in a larger centered mode range."""
    coefficients = np.asarray(coefficients, dtype=complex)
    old = (len(coefficients) - 1) // 2
    if degree < old:
        raise ValueError("Padding must not discard modes.")
    return np.pad(coefficients, degree - old)


def coefficient_subtract(first, second):
    degree = max((len(first) - 1) // 2, (len(second) - 1) // 2)
    return coefficient_pad(first, degree) - coefficient_pad(second, degree)


def fourier_coefficients_from_samples(samples, origin=0.0, alternate=False):
    """Periodic Fourier interpolation; an even Nyquist coefficient is split equally.

    The alternate path uses a real FFT with orthonormal normalization, giving
    a separate evaluation of the same polynomial for sensitivity checks.
    """
    values = np.asarray(samples, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)) or values.size < 2:
        raise ValueError("Fourier reconstruction needs at least two finite samples.")
    count = values.size
    degree = count // 2
    result = np.zeros(2 * degree + 1, dtype=complex)
    if alternate:
        positive = scipy_fft.rfft(values, norm="ortho") / math.sqrt(count)
        result[degree] = positive[0]
        for k in range(1, degree + 1):
            factor = 0.5 if count % 2 == 0 and k == degree else 1.0
            value = factor * positive[k] * np.exp(-1j * k * origin)
            result[degree + k] = value
            result[degree - k] = value.conjugate()
    else:
        native = np.fft.fft(values) / count
        modes = np.rint(np.fft.fftfreq(count, 1.0 / count)).astype(int)
        for j, k in enumerate(modes):
            if count % 2 == 0 and k == -degree:
                result[0] = 0.5 * native[j] * np.exp(1j * degree * origin)
                result[-1] = 0.5 * native[j] * np.exp(-1j * degree * origin)
            else:
                result[degree + k] = native[j] * np.exp(-1j * k * origin)
    return result


def uniform_values(coefficients, count):
    """Evaluate every retained signed mode on a resolving periodic grid."""
    coefficients = np.asarray(coefficients, dtype=complex)
    degree = (len(coefficients) - 1) // 2
    if count <= 2 * degree:
        raise ValueError("The evaluation grid must resolve all signed modes.")
    modes = np.arange(-degree, degree + 1)
    buffer = np.zeros(count, dtype=complex)
    buffer[modes % count] = count * coefficients
    return np.fft.ifft(buffer).real


def _maximum(coefficients, point_count):
    """Resolve extrema by FFT sampling and roots of the polynomial derivative."""
    profile = FourierProfile(coefficients)
    derivative = profile.derivative()
    values = uniform_values(coefficients, point_count)
    derivative_values = uniform_values(derivative.coefficients, point_count)
    maximum = float(np.max(np.abs(values)))
    spacing = TWOPI / point_count
    crossings = np.flatnonzero(derivative_values * np.roll(derivative_values, -1) < 0.0)
    for index in crossings:
        left, right = spacing * index, spacing * (index + 1)
        if derivative(left) * derivative(right) < 0.0:
            root = brentq(derivative, left, right, xtol=1e-14,
                          rtol=8.0 * np.finfo(float).eps)
            maximum = max(maximum, abs(profile(root)))
    return maximum


def coefficient_norms(coefficients, maximum_tolerance=None):
    """Continuous L2/H1/H2 norms by Parseval and an adaptively checked maximum.

    ``max_sensitivity`` is a measured grid-refinement change plus a floating
    evaluation allowance, not a certified enclosure of the exact maximum.
    """
    coefficients = np.asarray(coefficients, dtype=complex)
    if (coefficients.ndim != 1 or coefficients.size % 2 != 1
            or not np.all(np.isfinite(coefficients))):
        raise ValueError("Norms require finite centered signed coefficients.")
    degree = (len(coefficients) - 1) // 2
    modes = np.arange(-degree, degree + 1, dtype=float)
    power = np.abs(coefficients) ** 2
    roundoff = 64.0 * np.finfo(float).eps * float(np.sum(np.abs(coefficients)))
    tolerance = (max(64.0 * np.finfo(float).eps, 2.0 * roundoff)
                 if maximum_tolerance is None else float(maximum_tolerance))
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Maximum norm tolerance must be finite and positive.")
    if roundoff > tolerance:
        raise RuntimeError("Requested maximum accuracy is below its evaluation allowance.")
    count = 1 << int(np.ceil(np.log2(max(64, 8 * coefficients.size))))
    # Tiny sensitivity polynomials can be dominated by harmless high modes;
    # resolving every derivative root would then be far more expensive than
    # the solve. Their coefficient l1 norm already encloses the maximum.
    tiny = float(np.sum(np.abs(coefficients))) <= tolerance
    if tiny:
        sampled = float(np.max(np.abs(uniform_values(coefficients, count))))
        return {
            "l2": float(np.sqrt(TWOPI * np.sum(power))),
            "h1": float(np.sqrt(TWOPI * np.sum((1 + modes ** 2) * power))),
            "h2": float(np.sqrt(TWOPI * np.sum((1 + modes ** 2 + modes ** 4) * power))),
            "max": sampled, "max_sensitivity": float(np.sum(np.abs(coefficients))) + roundoff,
            "maximum_grid_points": count,
        }
    coarse = _maximum(coefficients, count)
    for _ in range(8):
        count *= 2
        fine = _maximum(coefficients, count)
        sensitivity = abs(fine - coarse) + roundoff
        maximum = max(fine, coarse)
        if sensitivity <= tolerance:
            break
        coarse = fine
    else:
        raise RuntimeError("Maximum norm evaluation failed eight refinement checks.")
    return {
        "l2": float(np.sqrt(TWOPI * np.sum(power))),
        "h1": float(np.sqrt(TWOPI * np.sum((1 + modes ** 2) * power))),
        "h2": float(np.sqrt(TWOPI * np.sum((1 + modes ** 2 + modes ** 4) * power))),
        "max": maximum, "max_sensitivity": float(sensitivity),
        "maximum_grid_points": count,
    }


def _shifted_mp_coefficients(reference, delta, offset=0):
    coefficients = reference._mp_coefficients()
    for j, k in enumerate(range(-reference.degree, reference.degree + 1)):
        sine, cosine = mp.sin_cos(-k * delta)
        coefficients[j] *= mp.mpc(cosine, sine)
    coefficients[reference.degree] += offset
    return coefficients


def _real_positive_polynomial(coefficients):
    d = (len(coefficients) - 1) // 2
    return [coefficients[d]] + [coefficients[d + k] + coefficients[d - k].conjugate()
                               for k in range(1, d + 1)]


def _mp_uniform_values(coefficients, count):
    """MP Horner evaluation on exact native coordinates 2*pi*j/count."""
    positive = _real_positive_polynomial(coefficients)
    step = 2 * mp.const_pi() / count
    values = []
    for j in range(count):
        sine, cosine = mp.sin_cos(step * j)
        z = mp.mpc(cosine, sine)
        value = positive[-1]
        for coefficient in positive[-2::-1]:
            value = value * z + coefficient
        values.append(value.real)
    return values


def _folded_modes(k, count):
    """Locations/weights of mode k in the native real interpolation space."""
    index = k % count
    if count % 2 == 0 and index == count // 2:
        return ((-count // 2, 0.5), (count // 2, 0.5))
    mode = index if index <= count // 2 else index - count
    return ((mode, 1.0),)


def _alias_bracket_mp(coefficients, N):
    """Compute (T_N q + T_(N+1) q)/2 - q by exact linear mode folding.

    Resolved modes cancel structurally, avoiding subtraction of O(1) rounded
    coefficient vectors.  Coarse-grid alias terms remain part of the error.
    """
    degree = (len(coefficients) - 1) // 2
    result = [_mp_complex(0) for _ in coefficients]
    for j, k in enumerate(range(-degree, degree + 1)):
        weights = {k: -1.0}
        for count in (N, N + 1):
            for mode, factor in _folded_modes(k, count):
                weights[mode] = weights.get(mode, 0.0) + 0.5 * factor
        for mode, factor in weights.items():
            if factor:
                result[degree + mode] += factor * coefficients[j]
    return result


def continuum_alias_coefficients(reference, N, delta=0.0, bits=128):
    """Public diagnostic: alias contribution of the saved polynomial itself."""
    with mp.context(precision=bits):
        shifted = _shifted_mp_coefficients(reference, mp.mpfr(delta), -mp.mpfr(delta))
        return np.asarray([complex(z) for z in _alias_bracket_mp(shifted, N)])


def _native_residuals(u, N, reference, u_mp_strings, bits):
    with mp.context(precision=bits):
        states = ([mp.mpfr(value) for value in u_mp_strings] if u_mp_strings is not None
                  else [mp.mpfr(float(value)) for value in u])
        first, second = states[:N], states[N:]
        delta = mp.fsum(second) / (N + 1) - mp.fsum(first) / N
        shifted = _shifted_mp_coefficients(reference, delta, -delta)
        sampled_first = _mp_uniform_values(shifted, N)
        sampled_second = _mp_uniform_values(shifted, N + 1)
        residual_first = [value - target / 2 for value, target in zip(first, sampled_first)]
        residual_second = [value + target / 2 for value, target in zip(second, sampled_second)]
        bracket = _alias_bracket_mp(shifted, N)
        return {"first": residual_first, "second": residual_second,
                "delta": delta, "alias": bracket, "shifted": shifted}


def _pair_norms_mp(first, second, bits=128):
    with mp.context(precision=bits):
        squared = mp.mpfr(0)
        maximum = mp.mpfr(0)
        for values in (first, second):
            count = len(values)
            spacing = 2 * mp.const_pi() / count
            gradient = [(values[(j + 1) % count] - values[j]) / spacing for j in range(count)]
            laplacian = [(values[(j + 1) % count] - 2 * values[j] + values[j - 1]) / spacing ** 2
                         for j in range(count)]
            squared += spacing * mp.fsum(value * value for array in (values, gradient, laplacian)
                                         for value in array)
            maximum = max(maximum, max(abs(value) for value in values))
        return {"h2": float(mp.sqrt(squared)), "max": float(maximum)}


def _error_coefficients(native, alternate=False):
    first = fourier_coefficients_from_samples(np.asarray(native["first"], dtype=float),
                                             alternate=alternate)
    second = fourier_coefficients_from_samples(np.asarray(native["second"], dtype=float),
                                              alternate=alternate)
    difference = coefficient_subtract(first, second)
    alias = np.asarray([complex(z) for z in native["alias"]])
    return coefficient_subtract(difference, -alias)


def _native_difference(first, second, bits):
    with mp.context(precision=bits):
        return {key: [a - b for a, b in zip(first[key], second[key])]
                for key in ("first", "second", "alias")}


def compute_observables(u, N, reference, u_mp_strings=None, bits=128,
                        check_bits=80, check_precision=True):
    """Compute the four plotted errors and measured evaluation sensitivities.

    ``u`` has N entries in layer 1 and N+1 in layer 2.  MP state strings and
    MP reference coefficients are required for the production accuracy claim.
    Without them the same definitions remain available for baseline audits.
    Arithmetic sensitivity compares evaluation at ``check_bits`` and ``bits``
    with the saved state fixed; it does not estimate optimization error.
    """
    u = np.asarray(u, dtype=float)
    if u.shape != (2 * N + 1,) or not np.all(np.isfinite(u)):
        raise ValueError("The two-layer state must have 2*N+1 finite entries.")
    if u_mp_strings is not None and np.asarray(u_mp_strings).shape != u.shape:
        raise ValueError("MP state strings must match the native state length.")
    h = 4 * np.pi / (2 * N + 1)
    maximum_tolerance = 1e-4 * h * h
    native = _native_residuals(u, N, reference, u_mp_strings, bits)
    pair = _pair_norms_mp(native["first"], native["second"], bits)
    coefficients = _error_coefficients(native)
    norms = coefficient_norms(coefficients, maximum_tolerance)
    alternate = _error_coefficients(native, alternate=True)
    reconstruction = coefficient_norms(coefficient_subtract(coefficients, alternate))
    # This measured difference concerns two evaluations of the SAME complete
    # observable. The physical alias bracket is not charged to its budget.
    if check_precision:
        if check_bits >= bits:
            raise ValueError("Precision check must use fewer bits than production.")
        lower = _native_residuals(u, N, reference, u_mp_strings, check_bits)
        arithmetic_native = _native_difference(native, lower, bits)
        arithmetic = coefficient_norms(_error_coefficients(arithmetic_native))
        pair_arithmetic = _pair_norms_mp(arithmetic_native["first"],
                                         arithmetic_native["second"], bits)
    else:
        arithmetic = {"h2": float("nan"), "max": float("nan")}
        pair_arithmetic = dict(arithmetic)
    alias = np.asarray([complex(z) for z in native["alias"]])
    alias_norms = coefficient_norms(alias)
    return {
        "N": int(N), "h": float(h), "phase_shift": float(native["delta"]),
        "phase_shift_mp": format(native["delta"], ".50g"),
        "phase_matched_h2_error": pair["h2"],
        "phase_matched_max_error": pair["max"],
        "relative_h2_error": norms["h2"], "relative_max_error": norms["max"],
        "relative_l2_error": norms["l2"], "relative_h1_error": norms["h1"],
        "error_coefficients": coefficients,
        "alias_coefficients": alias,
        "native_residual1": np.asarray(native["first"], dtype=float),
        "native_residual2": np.asarray(native["second"], dtype=float),
        "native_residual1_mp": np.asarray([format(x, ".50g") for x in native["first"]]),
        "native_residual2_mp": np.asarray([format(x, ".50g") for x in native["second"]]),
        "alias_coefficients_mp": _mp_strings(native["alias"]),
        "reconstruction_h2_sensitivity": reconstruction["h2"],
        "reconstruction_max_sensitivity": reconstruction["max"],
        "arithmetic_h2_sensitivity": arithmetic["h2"],
        "arithmetic_max_sensitivity": arithmetic["max"],
        "pair_arithmetic_h2_sensitivity": pair_arithmetic["h2"],
        "pair_arithmetic_max_sensitivity": pair_arithmetic["max"],
        "maximum_evaluation_sensitivity": norms["max_sensitivity"],
        "physical_alias_h2": alias_norms["h2"], "physical_alias_max": alias_norms["max"],
        "maximum_grid_points": norms["maximum_grid_points"],
        "state_has_mp_values": u_mp_strings is not None,
        "reference_has_mp_coefficients": reference.has_mp_coefficients,
        "evaluation_bits": bits, "comparison_bits": check_bits if check_precision else None,
        "sensitivities": {
            "reconstruction": {"h2": reconstruction["h2"], "max": reconstruction["max"]},
            "arithmetic": {"h2": arithmetic["h2"], "max": arithmetic["max"]},
            "maximum_evaluation": {"h2": 0.0, "max": norms["max_sensitivity"]},
        },
    }


def observable_difference(first, second):
    """Norm the field difference of two evaluations, not their scalar norm gap."""
    with mp.context(precision=128):
        if all("native_residual1_mp" in item and "alias_coefficients_mp" in item
               for item in (first, second)):
            native = {
                "first": [mp.mpfr(a) - mp.mpfr(b)
                          for a, b in zip(first["native_residual1_mp"], second["native_residual1_mp"])],
                "second": [mp.mpfr(a) - mp.mpfr(b)
                           for a, b in zip(first["native_residual2_mp"], second["native_residual2_mp"])],
            }
            degree = max((len(item["alias_coefficients_mp"]) - 1) // 2 for item in (first, second))
            alias = [_mp_complex(0) for _ in range(2 * degree + 1)]
            for item, sign in ((first, 1), (second, -1)):
                offset = degree - (len(item["alias_coefficients_mp"]) - 1) // 2
                for j, (real, imag) in enumerate(item["alias_coefficients_mp"]):
                    alias[offset + j] += sign * _mp_complex(real, imag)
            native["alias"] = alias
            relative = coefficient_norms(_error_coefficients(native))
            pair = _pair_norms_mp(native["first"], native["second"])
        else:
            relative = coefficient_norms(coefficient_subtract(first["error_coefficients"],
                                                               second["error_coefficients"]))
            r1 = np.asarray(first["native_residual1"]) - np.asarray(second["native_residual1"])
            r2 = np.asarray(first["native_residual2"]) - np.asarray(second["native_residual2"])
            pair = _pair_norms_mp([mp.mpfr(float(x)) for x in r1],
                                   [mp.mpfr(float(x)) for x in r2])
    return {"h2": relative["h2"], "max": relative["max"],
            "pair_h2": pair["h2"], "pair_max": pair["max"],
            "relative_h2_error": relative["h2"], "relative_max_error": relative["max"],
            "phase_matched_h2_error": pair["h2"], "phase_matched_max_error": pair["max"]}


def direct_relative_error_coefficients(u, N, reference, u_mp_strings=None, bits=128):
    """Independent MP direct DFT audit; quadratic work, intended for N <= 80.

    This reconstructs the full native states, subtracts the shifted reference
    in MP arithmetic, and only then converts the final coefficients to floats.
    It does not use the residual/alias identity or a binary64 FFT.
    """
    if N > 80:
        raise ValueError("Use the direct DFT audit only at N <= 80.")
    with mp.context(precision=bits):
        states = ([mp.mpfr(value) for value in u_mp_strings] if u_mp_strings is not None
                  else [mp.mpfr(float(value)) for value in u])
        delta = mp.fsum(states[N:]) / (N + 1) - mp.fsum(states[:N]) / N
        degree = max(reference.degree, (N + 1) // 2)
        result = [_mp_complex(0) for _ in range(2 * degree + 1)]
        for values, sign in ((states[:N], 1), (states[N:], -1)):
            count = len(values)
            for k in range(count // 2 + 1):
                terms = []
                for j, value in enumerate(values):
                    sine, cosine = mp.sin_cos(-2 * mp.const_pi() * k * j / count)
                    terms.append(value * mp.mpc(cosine, sine))
                coefficient = sum(terms, _mp_complex(0)) / count
                if k == 0:
                    result[degree] += sign * coefficient
                else:
                    factor = mp.mpfr("0.5") if count % 2 == 0 and k == count // 2 else 1
                    result[degree + k] += sign * factor * coefficient
                    result[degree - k] += sign * factor * coefficient.conjugate()
        shifted = _shifted_mp_coefficients(reference, delta, -delta)
        for k, coefficient in zip(range(-reference.degree, reference.degree + 1), shifted):
            result[degree + k] -= coefficient
        return np.asarray([complex(z) for z in result])


def sample_continuum_pair(reference, N, delta=0.0):
    """Binary64 compatibility helper for the native N and N+1 grids."""
    matched = reference.shifted(delta, offset=-delta)
    return np.concatenate((0.5 * matched(TWOPI * np.arange(N) / N),
                           -0.5 * matched(TWOPI * np.arange(N + 1) / (N + 1))))


def two_layer_errors(u, reference, N):
    """Binary64 compatibility implementation of the original discrete norms."""
    difference = np.asarray(u, dtype=float) - np.asarray(reference, dtype=float)
    if difference.shape != (2 * N + 1,):
        raise ValueError("Native vectors must have length 2*N+1.")
    squared = 0.0
    for values in (difference[:N], difference[N:]):
        spacing = TWOPI / len(values)
        first = (np.roll(values, -1) - values) / spacing
        second = (np.roll(values, -1) - 2 * values + np.roll(values, 1)) / spacing ** 2
        squared += spacing * np.sum(values ** 2 + first ** 2 + second ** 2)
    return {"h2_error": float(np.sqrt(squared)),
            "max_error": float(np.max(np.abs(difference)))}


def loglog_slope(N, errors):
    """Least-squares slope of log(error) against log(N), without a rate prior."""
    sizes, values = np.asarray(N, dtype=float), np.asarray(errors, dtype=float)
    if (sizes.size < 2 or sizes.shape != values.shape or not np.all(np.isfinite(values))
            or not np.all(values > 0) or not np.all(sizes > 0)):
        raise ValueError("A slope fit requires matching positive finite data.")
    return float(np.polyfit(np.log(sizes), np.log(values), 1)[0])
