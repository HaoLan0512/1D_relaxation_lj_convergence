"""Portable, pickle-free numerical artifacts and explicit cache identities."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np

from .config import N_VALUES, scientific_configuration


SOURCE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SOURCE_DIR.parents[1] / "outputs" / "lj_step_convergence"
SOURCE_FILES = ("__init__.py", "main.py", "config.py", "model.py", "continuum.py",
                "atomistic.py", "solvers.py", "observables.py", "validation.py",
                "storage.py", "plotting.py")


def timestamp():
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_fingerprints():
    return {name: sha256(SOURCE_DIR / name) for name in SOURCE_FILES}


def scientific_fingerprints():
    names = ("config.py", "model.py", "continuum.py", "atomistic.py", "solvers.py",
             "observables.py", "validation.py")
    return {name: sha256(SOURCE_DIR / name) for name in names}


def reference_fingerprints():
    return {name: sha256(SOURCE_DIR / name) for name in ("config.py", "model.py", "continuum.py")}


def configuration_id():
    content = json.dumps(scientific_configuration(), sort_keys=True).encode()
    return hashlib.sha256(content).hexdigest()


def runtime_versions():
    import gmpy2
    import matplotlib
    import mpmath
    import scipy
    return {"python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "numpy": np.__version__,
            "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
            "gmpy2": gmpy2.version(), "mpmath": mpmath.__version__}


def write_json(path, value):
    """Atomically update bookkeeping; numerical bundles remain immutable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_bundle(path, value):
    """Atomically save nested data and arrays without executable pickle data."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}

    def pack(item):
        if isinstance(item, np.ndarray):
            if item.dtype.hasobject:
                raise TypeError("Object arrays must be converted to lossless strings first.")
            name = f"array_{len(arrays):04d}"
            arrays[name] = item
            return {"__array__": name}
        if isinstance(item, dict):
            return {str(key): pack(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [pack(val) for val in item]
        if isinstance(item, np.generic):
            return pack(item.item())
        if isinstance(item, complex):
            return {"__complex__": [item.real, item.imag]}
        if isinstance(item, float) and not np.isfinite(item):
            return {"__nonfinite__": repr(item)}
        if item is None or isinstance(item, (str, float, int, bool)):
            return item
        raise TypeError(f"Unsupported saved value: {type(item).__name__}")

    metadata = pack(value)
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, allow_nan=False))
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)
    return path


def save_bundle(directory, stem, value):
    """Use unique filenames only for temporary preparation checkpoints."""
    return write_bundle(Path(directory) / f"{stem}_{timestamp()}.npz", value)


def load_bundle(path):
    with np.load(path, allow_pickle=False) as archive:
        def unpack(item):
            if isinstance(item, dict):
                if set(item) == {"__array__"}:
                    return archive[item["__array__"]].copy()
                if set(item) == {"__complex__"}:
                    return complex(*item["__complex__"])
                if set(item) == {"__nonfinite__"}:
                    return float(item["__nonfinite__"])
                return {key: unpack(val) for key, val in item.items()}
            if isinstance(item, list):
                return [unpack(val) for val in item]
            return item
        return unpack(json.loads(str(archive["metadata_json"])))


def require_configuration(record):
    if record.get("configuration_id") != configuration_id():
        raise ValueError("Saved data do not match this scientific configuration. "
                         "Use main.py --prepare to prepare matching data.")


def publication_manifest(data_dir):
    """Check dataset configuration, structure, and integrity of thirteen profiles."""
    data_dir = Path(data_dir).resolve()
    path = data_dir / "publication_manifest.json"
    if not path.is_file():
        raise FileNotFoundError("No complete publication dataset is available. "
                                "Run main.py --prepare explicitly; partial preparation "
                                "is retained in preparation/.")
    record = read_json(path)
    require_configuration(record)
    if (record.get("format_version") != 3 or record.get("status") != "complete"
            or record.get("configuration") != scientific_configuration()):
        raise ValueError("The publication manifest format or scientific configuration is incompatible.")
    expected_cases = {str(N): f"profiles/N{N:05d}.npz" for N in N_VALUES}
    expected_paths = {"profiles/continuum.npz", *expected_cases.values()}
    if (record.get("reference") != "profiles/continuum.npz"
            or record.get("cases") != expected_cases
            or set(record.get("artifact_hashes", {})) != expected_paths):
        raise ValueError("The publication must list exactly the continuum and twelve native profiles.")
    profiles = data_dir / "profiles"
    if (profiles.is_symlink() or profiles.is_junction()
            or not profiles.is_dir()
            or {p.name for p in profiles.iterdir()} != {Path(p).name for p in expected_paths}):
        raise ValueError("The profiles folder has missing, extra, or redirected artifacts.")
    for relative, expected in record["artifact_hashes"].items():
        artifact = data_dir / relative
        if artifact.is_symlink() or not artifact.is_file() or artifact.resolve().parent != profiles.resolve():
            raise ValueError(f"Unsafe saved artifact path: {relative}")
        if sha256(artifact) != expected:
            raise ValueError(f"Saved artifact hash mismatch: {relative}")
    return record
