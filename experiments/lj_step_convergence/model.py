"""Local, paper-scaled Lennard--Jones model and analytic periodization.

The paper coordinate has period 2*pi. Its horizontal separation is s, and
interlayer_distance is the fixed positive vertical distance in the same units.
These formulas are adapted from clr.potentials.lj_periodic, without importing
the older experiment or reusable module, so this experiment is self-contained.
"""

import mpmath as mp
import numpy as np

from .config import PARAMETERS, evaluate_parameters


def pair_potential(s, parameters=PARAMETERS):
    """Evaluate V(s) with radius sqrt(d**2+s**2) in paper coordinate units."""
    s = np.asarray(s, dtype=float)
    epsilon, sigma, distance = evaluate_parameters(parameters)
    radius2 = distance**2 + s**2
    sr6 = sigma**6 / radius2**3
    return 4 * epsilon * (sr6**2 - sr6)


def pair_potential_prime(s, parameters=PARAMETERS):
    """Differentiate V with respect to the paper coordinate s."""
    s = np.asarray(s, dtype=float)
    epsilon, sigma, distance = evaluate_parameters(parameters)
    radius2 = distance**2 + s**2
    sr6 = sigma**6 / radius2**3
    return 24 * epsilon * s * (sr6 - 2 * sr6**2) / radius2


def pair_potential_second(s, parameters=PARAMETERS):
    """Evaluate the second phase derivative of the same effective pair."""
    s = np.asarray(s, dtype=float)
    epsilon, sigma, distance = evaluate_parameters(parameters)
    radius2 = distance**2 + s**2
    sr6 = sigma**6 / radius2**3
    first = (1 / radius2 - 2 * s**2 / radius2**2) * (sr6 - 2 * sr6**2)
    second = -6 * s**2 * sr6 * (1 - 4 * sr6) / radius2**2
    return 24 * epsilon * (first + second)


V = pair_potential
Vprime = pair_potential_prime
Vsecond = pair_potential_second


def _mp_parameters(parameters):
    return evaluate_parameters(parameters, number=mp.mpf, pi=mp.pi)


def mp_fourier_coefficients(parameters=PARAMETERS, max_mode=24):
    """Return MP C0, A1,...,AK for W(s)=4 sum_m V(s-2*pi*m).

    Computation takes place at the caller's current mpmath precision.
    The constant mode is separate and is not counted among the cosine modes.
    """
    epsilon, sigma, distance = _mp_parameters(parameters)

    def transforms(k):
        z = k * distance
        exponential = mp.exp(-z)
        i3 = mp.pi * exponential * (z*z + 3*z + 3) / (8 * distance**5)
        i6 = mp.pi * exponential * (
            z**5 + 15*z**4 + 105*z**3 + 420*z*z + 945*z + 945
        ) / (3840 * distance**11)
        return i3, i6

    i3, i6 = transforms(0)
    # Periodization over 2*pi gives the constant Fourier coefficient 1/(2*pi)
    # times the line transform, and twice that coefficient for cosine modes.
    constant = 8 * epsilon / mp.pi * (sigma**12 * i6 - sigma**6 * i3)
    coefficients = []
    for k in range(1, max_mode + 1):
        i3, i6 = transforms(k)
        coefficients.append(16 * epsilon / mp.pi * (sigma**12 * i6 - sigma**6 * i3))
    return constant, coefficients


def fourier_coefficients(parameters=PARAMETERS, max_mode=24):
    """Return binary64 coefficients, evaluated first at 80-bit precision."""
    with mp.workprec(80):
        constant, coefficients = mp_fourier_coefficients(parameters, max_mode)
        return float(constant), np.array([float(a) for a in coefficients])


def fourier_W(s, constant, cosine_coefficients):
    s = np.asarray(s, dtype=float)
    coefficients = np.asarray(cosine_coefficients, dtype=float)
    modes = np.arange(1, len(coefficients) + 1)
    return constant + np.sum(coefficients * np.cos(s[..., None] * modes), axis=-1)


def fourier_Wprime(s, cosine_coefficients):
    s = np.asarray(s, dtype=float)
    coefficients = np.asarray(cosine_coefficients, dtype=float)
    modes = np.arange(1, len(coefficients) + 1)
    return -np.sum(modes * coefficients * np.sin(s[..., None] * modes), axis=-1)


def fourier_Wsecond(s, cosine_coefficients):
    s = np.asarray(s, dtype=float)
    coefficients = np.asarray(cosine_coefficients, dtype=float)
    modes = np.arange(1, len(coefficients) + 1)
    return -np.sum(modes**2 * coefficients * np.cos(s[..., None] * modes), axis=-1)


def mp_potential_tail_bounds(max_mode, parameters=PARAMETERS):
    """Bound all omitted modes of W, W', W'' by positive geometric envelopes."""
    epsilon, sigma, distance = _mp_parameters(parameters)
    first = max_mode + 1
    beta = distance
    z = beta * first
    i3 = mp.pi * mp.exp(-z) * (z*z + 3*z + 3) / (8 * distance**5)
    i6 = mp.pi * mp.exp(-z) * (
        z**5 + 15*z**4 + 105*z**3 + 420*z*z + 945*z + 945
    ) / (3840 * distance**11)
    envelope = 16 * epsilon / mp.pi * (sigma**12 * i6 + sigma**6 * i3)
    bounds = {}
    for order, name in enumerate(("value", "force", "curvature")):
        ratio = mp.exp(-beta) * (1 + mp.mpf(1) / first)**(5 + order)
        if ratio >= 1:
            raise ValueError("Increase potential modes until the tail envelope contracts.")
        bounds[name] = first**order * envelope / (1 - ratio)
    return bounds


def certify_global_minimum(constant, coefficients, parameters=PARAMETERS):
    """Prove the minimum location for this model instead of assuming it.

    Since |sin(k*s)| <= k*sin(s) on [0,pi], A1 exceeding
    sum_{k>=2} k^2 |Ak| plus the infinite curvature tail proves W'<0 on
    (0,pi). Evenness and periodicity then prove that pi is a global minimum.
    A changed parameter set not covered by this certificate is rejected.
    """
    tails = mp_potential_tail_bounds(len(coefficients), parameters)
    margin = coefficients[0] - mp.fsum(
        k*k * abs(a) for k, a in enumerate(coefficients, 1) if k >= 2
    ) - tails["curvature"]
    if margin <= 0:
        raise RuntimeError(
            "Global potential minimum was not certified: first-harmonic "
            "dominance fails for these parameters. No minimum is assumed.")
    minimum = constant + mp.fsum((-1)**k * a for k, a in enumerate(coefficients, 1))
    return {
        "location": mp.pi,
        "value": minimum,
        "derivative_sign_margin": margin,
        "potential_tail_bounds": tails,
        "method": "first_harmonic_derivative_dominance_with_infinite_tail",
    }
