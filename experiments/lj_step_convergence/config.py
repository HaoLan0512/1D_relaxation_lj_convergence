"""Paper-coordinate parameters and named numerical targets for this experiment.

All interaction lengths use the paper's coordinate units, with period 2*pi
and radius sqrt(d**2+s**2). Exact parameter expressions are evaluated with
the caller's arithmetic, so multiprecision calculations do not inherit a
rounded binary64 value of pi. ``interlayer_distance`` is the separation d,
not the period L used in the accompanying research note.
"""
from dataclasses import asdict, dataclass
import math
import re


_PARAMETER_EXPRESSION = re.compile(
    r"([+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)(\*pi)?")


def _parameter_components(value):
    """Read only a positive decimal, optionally multiplied by pi; never eval."""
    expression = str(value).strip()
    match = _PARAMETER_EXPRESSION.fullmatch(expression)
    if match is None:
        raise ValueError("LJ parameters must be a positive decimal or decimal*pi.")
    decimal, pi_factor = match.groups()
    ordinary = float(decimal) * (math.pi if pi_factor else 1)
    if not math.isfinite(ordinary) or ordinary <= 0:
        raise ValueError("All LJ parameters must be positive and finite.")
    return decimal, bool(pi_factor)


@dataclass(frozen=True)
class EffectiveLJParameters:
    epsilon: str = "16"
    sigma: str = "1.8*pi"
    interlayer_distance: str = "2*pi"

    def __post_init__(self):
        for name, value in asdict(self).items():
            decimal, pi_factor = _parameter_components(value)
            object.__setattr__(self, name, decimal + ("*pi" if pi_factor else ""))


PARAMETERS = EffectiveLJParameters()


def evaluate_parameters(parameters=PARAMETERS, number=float, pi=math.pi):
    """Return epsilon, sigma, d using the supplied constructor and current pi.

    Multiprecision callers must supply their number constructor and pi inside
    the active precision context; no intermediate binary64 values are used.
    """
    result = []
    for name in ("epsilon", "sigma", "interlayer_distance"):
        decimal, pi_factor = _parameter_components(getattr(parameters, name))
        result.append(number(decimal) * pi if pi_factor else number(decimal))
    return tuple(result)


N_VALUES = (20, 40, 80, 160, 320, 640, 1280, 2560, 3840, 5120, 7680, 10240)
PREPARATION_ORDER = (20, 80, 320, 1280, 40, 160, 640, 2560, 3840, 5120, 7680, 10240)
FIT_N = (1280, 2560, 5120, 10240)
PAIR_CUTOFFS = (160, 320, 640)
SEED = 3
RANDOM_MODES = 5
RANDOM_MAX_AMPLITUDE = 0.1
PRECISION_BITS = 128
CHECK_PRECISION_BITS = 80
REFERENCE_DEGREES = (64, 128, 256, 512)
REFERENCE_H2_TOL = 1e-13
REFERENCE_MAX_TOL = 1e-14
DISPLAY_POINTS = 32769
EL_OVER_H2_TOL = 1e-5
ERROR_BUDGET_OVER_H2 = 1e-4
ERROR_BUDGET_FRACTION = 0.05
MEAN_TOL = 1e-12
COMPUTE_BUDGET_SECONDS = 8 * 60 * 60
WORKING_MEMORY_TARGET_BYTES = int(1.5 * 1024**3)
SCHEMA_VERSION = 2
START_NAME = "random_fourier_seed_3"
ERROR_FIELDS = ("phase_matched_h2_error", "phase_matched_max_error",
                "relative_h2_error", "relative_max_error")


def scientific_configuration():
    """Return the immutable model/observable identity attached to saved data."""
    return {"parameters": asdict(PARAMETERS),
            "unit_convention": "paper coordinates; period=2*pi; radius=sqrt(d^2+s^2)",
            "parameter_representation": "exact decimal or decimal*pi",
            "n_values": list(N_VALUES),
            "fit_n": list(FIT_N), "seed": SEED, "random_modes": RANDOM_MODES,
            "random_max_amplitude": RANDOM_MAX_AMPLITUDE,
            "phase": "delta=mean(u2)-mean(u1); q_delta=q(x-delta)-delta",
            "norms": "native weighted H_h^2; continuous reconstructed H^2; maximum",
            "aliasing": "retained in error; same-observable evaluation sensitivity",
            "el_over_h2": EL_OVER_H2_TOL,
            "budget_over_h2": ERROR_BUDGET_OVER_H2,
            "budget_fraction": ERROR_BUDGET_FRACTION,
            "precision_bits": PRECISION_BITS, "schema_version": SCHEMA_VERSION}
