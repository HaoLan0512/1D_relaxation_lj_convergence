"""Reader entry point: cached figures by default, explicit resumable preparation."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

sys.dont_write_bytecode = True

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "lj_step_convergence"

import numpy as np

from .config import (COMPUTE_BUDGET_SECONDS, FIT_N, N_VALUES,
                     PAIR_CUTOFFS, PREPARATION_ORDER, scientific_configuration)
from .storage import (DEFAULT_DATA_DIR, configuration_id, load_bundle,
                      publication_manifest, read_json, require_configuration,
                      runtime_versions, save_bundle,
                      scientific_fingerprints, sha256, source_fingerprints,
                      timestamp, write_bundle, write_json)

from .validation import (reference_profile as _reference_profile,
                         evaluate_state as _evaluate, endpoint_gate as _endpoint_gate,
                         reference_sensitivity as _reference_sensitivity)

def log(message):
    print(message, flush=True)


WORKER_ACCOUNT_INTERVAL_SECONDS = 30.0


def _peak_memory_bytes():
    """Read this worker's peak resident memory without a monitoring dependency."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
                "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        query = ctypes.windll.psapi.GetProcessMemoryInfo
        query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if query(wintypes.HANDLE(-1), ctypes.byref(counters), counters.cb):
            return int(counters.PeakWorkingSetSize)
        return None
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else 1024*value)


def relative_path(path, root):
    return str(Path(path).relative_to(root)).replace("\\", "/")



def _case_worker(run, N, deadline):
    """Prepare one size; every numerical checkpoint has an immutable filename."""
    from .observables import observable_difference
    from .solvers import refine_case, solve_case
    from .validation import assess_accuracy_budget, validate_case

    manifest = read_json(run / "run_manifest.json")
    require_configuration(manifest)
    reference = load_bundle(run / manifest["reference"])
    profile = _reference_profile(reference)
    directory = run / f"N{N:05d}"
    directory.mkdir(exist_ok=True)
    progress_path = directory / "progress.json"
    progress = read_json(progress_path) if progress_path.exists() else {"stages": {}}
    identity = scientific_fingerprints()
    if progress.get("scientific_fingerprints", identity) != identity:
        raise ValueError("Preparation checkpoint scientific sources do not match.")
    progress["scientific_fingerprints"] = identity
    write_json(progress_path, progress)
    states, observed = {}, {}

    def checkpoint(payload):
        saved = save_bundle(directory, "checkpoint", payload)
        progress["checkpoint"] = saved.name
        progress["checkpoint_stage"] = payload.get("stage", "polishing")
        write_json(progress_path, progress)

    for cutoff in PAIR_CUTOFFS:
        key = str(cutoff)
        if key in progress["stages"]:
            states[cutoff] = load_bundle(directory / progress["stages"][key])
            log(f"N={N}: reuse saved cutoff {cutoff} endpoint.")
        else:
            log(f"N={N}: {'seed-3 optimization' if cutoff == PAIR_CUTOFFS[0] else 'cutoff refinement'}, M={cutoff}.")
            if cutoff == PAIR_CUTOFFS[0]:
                resume = load_bundle(directory / progress["checkpoint"]) if progress.get("checkpoint") else None
                if resume is not None and resume.get("stage") != "lbfgs" and "u_mp_strings" in resume:
                    resume.setdefault("u", np.asarray(resume["u_mp_strings"], dtype=float))
                    states[cutoff] = refine_case(resume, N, cutoff, deadline=deadline,
                                                 accuracy_factor=1.0, checkpoint=checkpoint)
                else:
                    kwargs = {"resume_state": resume} if resume is not None else {}
                    states[cutoff] = solve_case(N, cutoff=cutoff, deadline=deadline,
                                                checkpoint=checkpoint, **kwargs)
            else:
                previous = PAIR_CUTOFFS[PAIR_CUTOFFS.index(cutoff)-1]
                states[cutoff] = refine_case(states[previous], N, cutoff, deadline=deadline,
                                             accuracy_factor=1.0, checkpoint=checkpoint)
            path = save_bundle(directory, f"state_M{cutoff}", states[cutoff])
            progress["stages"][key] = path.name
            progress.pop("checkpoint", None)
            write_json(progress_path, progress)
        if time.monotonic() >= deadline:
            raise TimeoutError("Preparation deadline reached after endpoint checkpoint.")
        observed[cutoff] = _evaluate(states[cutoff], profile)
        gate = _endpoint_gate(states[cutoff])
        log(f"N={N}, M={cutoff}: endpoint {'accepted' if gate['accepted'] else 'unresolved'}, "
            f"EL={states[cutoff]['diagnostics']['el_residual_inf_norm']:.3e}.")

    # M=320 is retained only with an independently refined M=640 comparison.
    # This avoids calling an unchecked highest-cutoff state the production result.
    production_cutoff = PAIR_CUTOFFS[1]
    state = states[production_cutoff]
    baseline_observed = observed[production_cutoff]
    if progress.get("tight_state"):
        tight = load_bundle(directory / progress["tight_state"])
    else:
        log(f"N={N}: tighten the M={production_cutoff} endpoint by tenfold.")
        tight = refine_case(state, N, production_cutoff, deadline=deadline,
                            accuracy_factor=0.1, checkpoint=checkpoint)
        path = save_bundle(directory, "state_tight", tight)
        progress["tight_state"] = path.name
        write_json(progress_path, progress)
    tight_observed = _evaluate(tight, profile)
    endpoint_change = observable_difference(baseline_observed, tight_observed)
    cutoff_change = observable_difference(tight_observed, observed[PAIR_CUTOFFS[-1]])
    coarse_cutoff_change = observable_difference(observed[PAIR_CUTOFFS[0]], baseline_observed)
    obs = tight_observed
    sensitivities = {"endpoint": {k: endpoint_change[k] for k in ("h2", "max")},
                     "continuum": _reference_sensitivity(reference),
                     "cutoff": {k: cutoff_change[k] for k in ("h2", "max")},
                     **obs["sensitivities"]}
    budget = assess_accuracy_budget(obs, sensitivities, N)
    gates = {f"M{cutoff}": _endpoint_gate(states[cutoff]) for cutoff in PAIR_CUTOFFS}
    gates["tight"] = _endpoint_gate(tight)
    # A no-op polishing trial at only the original tolerance is not evidence of
    # tenfold tightening. The solver's factor-0.1 acceptance is recorded separately.
    tightening_resolved = bool(tight.get("accepted", False))
    accepted = bool(all(g["accepted"] for g in gates.values()) and tightening_resolved
                    and budget["numerically_resolved"] and obs["state_has_mp_values"]
                    and obs["reference_has_mp_coefficients"])
    retained_keys = ("u", "u_mp_strings", "N", "start", "pair_cutoff", "precision_bits",
                     "accuracy_factor", "normalization", "diagnostics", "curvature", "accepted")
    compact_states = {str(k): {key: value[key] for key in retained_keys if key in value}
                      for k, value in {**states, "tight": tight}.items()}
    result = {"N": N, "initial_u": states[PAIR_CUTOFFS[0]]["initial_u"],
              "states": compact_states, "observables": obs, "accuracy_budget": budget,
              "endpoint_gates": gates, "tightening_resolved": tightening_resolved,
              "coarse_cutoff_change": coarse_cutoff_change,
              "cutoff_change": cutoff_change, "endpoint_change": endpoint_change,
              "production_cutoff": production_cutoff, "accepted": accepted}
    result, verification = validate_case(result, reference, deadline=deadline, progress=log)
    result["verification"] = verification
    accepted = bool(result["accepted"] and verification["passed"])
    result["accepted"] = accepted
    path = save_bundle(directory, "case_result", result)
    write_json(directory / "result.json", {"path": path.name, "accepted": accepted,
                                            "N": N, "configuration_id": configuration_id()})
    log(f"N={N}: {'VALIDATED' if accepted else 'UNRESOLVED'}; "
        f"H2 sensitivity/budget={budget['h2']['ratio']:.3g}, "
        f"maximum={budget['max']['ratio']:.3g}.")


def _worker(args):
    run = Path(args.run_dir).resolve()
    deadline = time.monotonic() + args.seconds
    try:
        if args.worker == "reference":
            from .continuum import build_reference
            from .validation import validate_reference
            reference = build_reference(deadline=deadline, progress=log)
            reference["verification"] = validate_reference(reference, deadline=deadline)
            if not reference["verification"]["passed"]:
                raise RuntimeError("The independently checked continuum reference is unresolved.")
            path = save_bundle(run, "continuum_reference", reference)
            # The parent owns run_manifest while it periodically accounts time.
            # Hand off this pointer separately to avoid competing JSON updates.
            write_json(run / "reference_worker_result.json", {
                "reference": path.name, "pid": os.getpid(),
                "configuration_id": configuration_id(),
            })
        else:
            _case_worker(run, args.case, deadline)
    finally:
        write_json(run / f"worker_resources_{args.worker}_{args.case}_{timestamp()}.json",
                   {"N": args.case, "pid": os.getpid(), "peak_memory_bytes": _peak_memory_bytes()})


def _new_run(data_dir):
    """Create the one resumable workspace; never start a timestamped archive."""
    data_dir.mkdir(parents=True, exist_ok=True)
    run = data_dir / "preparation"
    if run.exists():
        raise ValueError("The preparation workspace already exists; inspect its manifest before resuming.")
    run.mkdir()
    write_json(run / "run_manifest.json", {
        "configuration_id": configuration_id(), "configuration": scientific_configuration(),
        "runtime": runtime_versions(), "source_fingerprints": source_fingerprints(),
        "status": "preparing", "elapsed_seconds": 0.0,
        "budget_seconds": COMPUTE_BUDGET_SECONDS, "cases": {}, "reference": None})
    return run


def _worker_pid_running(pid):
    """Treat an inaccessible process as live; uncertainty never permits duplicates."""
    pid = int(pid or 0)
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            # Only ERROR_INVALID_PARAMETER proves this PID does not exist.
            return ctypes.get_last_error() != 87
        try:
            code = wintypes.DWORD()
            return not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _recover_interrupted_worker(run):
    """Charge an unobserved worker's entire outstanding reservation after a crash.

    Ordinary handled exits use elapsed monotonic time. A hard parent crash can
    leave an unobserved child running longer than one polling interval, so the
    unknown interval is conservatively charged in full, never reset on resume.
    """
    record = read_json(run / "run_manifest.json")
    active = record.get("active_worker")
    if not active:
        return record
    if (_worker_pid_running(active.get("pid"))
            or _worker_pid_running(active.get("parent_pid"))):
        raise RuntimeError("A recorded preparation process is live or inaccessible; "
                           "do not launch a second preparation process.")
    baseline = float(active["baseline_elapsed_seconds"])
    reserved = float(active["reserved_seconds"])
    charged = float(active.get("accounted_worker_seconds", 0.0))
    record["elapsed_seconds"] = max(float(record["elapsed_seconds"]), baseline + reserved)
    record.setdefault("worker_history", []).append({
        "worker": active["worker"], "N": active.get("N"),
        "status": "interruption_recovered", "accounted_seconds": max(charged, reserved),
        "accounting": "conservative unknown-runtime reservation",
        "measured_seconds_before_interruption": charged,
        "unmeasured_reservation_seconds": max(0.0, reserved-charged)})
    record.pop("active_worker", None)
    write_json(run / "run_manifest.json", record)
    log("Recovered interrupted preparation; charged its unknown remaining reservation conservatively.")
    return record


def _run_child(run, worker, seconds, N=None):
    """Supervise a worker with a reservation persisted before the process starts."""
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", worker,
               "--run-dir", str(run), "--seconds", str(seconds)]
    if N is not None:
        command += ["--case", str(N)]
    environment = os.environ.copy()
    environment.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1", "PYTHONUTF8": "1"})
    interval = min(30.0, float(WORKER_ACCOUNT_INTERVAL_SECONDS))
    if interval <= 0 or not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("Worker accounting interval and reservation must be positive.")
    initial = read_json(run / "run_manifest.json")
    require_configuration(initial)
    if initial.get("source_fingerprints") != source_fingerprints():
        raise ValueError("Preparation sources changed; the checkpoint cannot be resumed.")
    if initial.get("active_worker"):
        raise RuntimeError("Recover the recorded unfinished worker before launching another.")
    baseline = float(initial["elapsed_seconds"])
    seconds = min(float(seconds), float(initial["budget_seconds"])-baseline)
    if seconds <= 0:
        raise RuntimeError("The cumulative preparation budget is exhausted.")
    command[command.index("--seconds")+1] = str(seconds)
    reservation = {"worker": worker, "N": N, "pid": None, "parent_pid": os.getpid(),
                   "baseline_elapsed_seconds": baseline, "reserved_seconds": seconds,
                   "accounted_worker_seconds": 0.0, "started_at_unix": time.time()}
    initial["active_worker"] = reservation
    write_json(run / "run_manifest.json", initial)
    started = time.monotonic()
    process = None
    status = "launch_failed"

    def charge(final=False):
        elapsed = time.monotonic()-started
        record = read_json(run / "run_manifest.json")
        record["elapsed_seconds"] = max(float(record["elapsed_seconds"]), baseline+elapsed)
        if final:
            record.pop("active_worker", None)
            record.setdefault("worker_history", []).append({
                "worker": worker, "N": N, "seconds": elapsed, "status": status,
                "worker_pid": process.pid if process is not None else None,
                "accounting": "measured monotonic worker time"})
        else:
            record["active_worker"] = {**reservation,
                "pid": process.pid if process is not None else None,
                "accounted_worker_seconds": elapsed, "last_accounted_at_unix": time.time()}
        write_json(run / "run_manifest.json", record)

    def stop():
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    try:
        process = subprocess.Popen(command, env=environment)
        status = "running"
        charge()
        while True:
            remaining = seconds-(time.monotonic()-started)
            if remaining <= 0:
                stop()
                status = "time_limit"
                break
            try:
                code = process.wait(timeout=min(interval, remaining))
                status = "finished" if code == 0 else "failed"
                break
            except subprocess.TimeoutExpired:
                charge()
    except BaseException:
        if process is not None:
            status = "interrupted"
            stop()
        raise
    finally:
        # A failed terminate/kill/wait leaves an unobserved live worker. Retain
        # its reservation until exit is established, rather than freeing time
        # or permitting a second preparation process.
        charge(final=process is None or process.poll() is not None)
    if worker == "reference" and status == "finished":
        result = read_json(run / "reference_worker_result.json")
        require_configuration(result)
        if result.get("pid") != process.pid:
            raise ValueError("Reference worker result belongs to a different process.")
        reference = load_bundle(run / result["reference"])
        if not (reference["diagnostics"].get("accepted")
                and reference.get("verification", {}).get("passed")):
            raise ValueError("Reference worker produced an unresolved artifact.")
        record = read_json(run / "run_manifest.json")
        record["reference"] = result["reference"]
        write_json(run / "run_manifest.json", record)
    return status


def _remove_directory(path, parent):
    """Remove a known temporary/output directory only within its expected parent."""
    path, parent = Path(path), Path(parent).resolve()
    if path.is_symlink() or path.is_junction() or path.resolve().parent != parent:
        raise ValueError(f"Refusing to remove a redirected or out-of-scope directory: {path}")
    if path.exists():
        shutil.rmtree(path)


def _cleanup_completed_preparation(data_dir):
    """Finish cleanup after publication succeeded but rendering was interrupted."""
    run = data_dir / "preparation"
    manifest = run / "run_manifest.json"
    if not run.exists():
        return
    if run.is_symlink() or run.is_junction() or not manifest.is_file():
        log("Retained an unrecognized preparation workspace; inspect it before removal.")
        return
    record = read_json(manifest)
    compatible = (record.get("configuration_id") == configuration_id()
                  and record.get("source_fingerprints") == source_fingerprints()
                  and record.get("status") == "complete"
                  and not record.get("active_worker")
                  and all(record.get("cases", {}).get(str(N), {}).get("accepted")
                          for N in N_VALUES))
    if compatible:
        _remove_directory(run, data_dir)
    else:
        log("Retained an incomplete, incompatible, or active preparation workspace.")


def _load_publication(data_dir):
    record = publication_manifest(data_dir)
    reference = load_bundle(data_dir / record["reference"])
    _reference_profile(reference)
    cases = [load_bundle(data_dir / record["cases"][str(N)]) for N in N_VALUES]
    for N, case in zip(N_VALUES, cases):
        if int(case["N"]) != N or not case.get("accepted"):
            raise ValueError(f"Invalid or unaccepted saved case N={N}.")
        _require_case_evidence(case)
    if not reference.get("verification", {}).get("passed"):
        raise ValueError("Continuum validation evidence is missing or failed.")
    return record, reference, cases


def _require_case_evidence(case):
    states = case.get("states", {})
    if set(states) != {"160", "320", "640", "tight"}:
        raise ValueError(f"Case N={case.get('N')} must retain all four validation states.")
    if not case.get("accepted") or not case.get("verification", {}).get("passed"):
        raise ValueError(f"Case N={case.get('N')} has no passing fresh validation evidence.")
    endpoints = case["verification"].get("endpoints", {})
    if (set(endpoints) != set(states)
            or not all(evidence.get("passed") for evidence in endpoints.values())):
        raise ValueError(f"Case N={case.get('N')} has incomplete endpoint verification.")
    if not all(state.get("accepted") for state in states.values()):
        raise ValueError(f"Case N={case.get('N')} contains an unaccepted state.")


def publish_dataset(data_dir, reference, cases, verification=None):
    """Write thirteen authenticated bundles only after all twelve sizes pass."""
    data_dir = Path(data_dir).resolve()
    if tuple(case["N"] for case in cases) != N_VALUES:
        raise ValueError("Publication requires all twelve requested sizes exactly once and in order.")
    if not (reference["diagnostics"].get("accepted")
            and reference.get("verification", {}).get("passed")):
        raise ValueError("Cannot publish an unvalidated continuum reference.")
    for case in cases:
        _require_case_evidence(case)
    if verification is not None and not verification.get("passed"):
        raise ValueError("Cannot publish a dataset with failed verification.")
    if (data_dir / "publication_manifest.json").exists():
        raise ValueError("A publication already exists; preparation never overwrites it.")
    data_dir.mkdir(parents=True, exist_ok=True)
    candidate = Path(tempfile.mkdtemp(prefix=".publication_", dir=data_dir))
    try:
        write_bundle(candidate / "profiles/continuum.npz", reference)
        for case in cases:
            write_bundle(candidate / f"profiles/N{case['N']:05d}.npz", case)
        cases_index = {str(N): f"profiles/N{N:05d}.npz" for N in N_VALUES}
        paths = ["profiles/continuum.npz", *cases_index.values()]
        record = {"format_version": 3, "status": "complete",
                  "configuration_id": configuration_id(), "configuration": scientific_configuration(),
                  "source_fingerprints": source_fingerprints(), "runtime": runtime_versions(),
                  "reference": "profiles/continuum.npz", "cases": cases_index,
                  "artifact_hashes": {p: sha256(candidate/p) for p in paths}}
        write_json(candidate / "publication_manifest.json", record)
        publication_manifest(candidate)
        destination = data_dir / "profiles"
        # An unmanifested profiles directory can only be a failed publication.
        if destination.exists():
            _remove_directory(destination, data_dir)
        os.replace(candidate / "profiles", destination)
        os.replace(candidate / "publication_manifest.json", data_dir / "publication_manifest.json")
        if verification is not None:
            write_json(data_dir / "verification.json", verification)
    finally:
        _remove_directory(candidate, data_dir)
    return publication_manifest(data_dir)


def _render(data_dir, plot_min_n=20):
    """Regenerate only derived files, with no writes to authenticated profiles."""
    from .plotting import make_figures, write_csv
    record, reference, cases = _load_publication(data_dir)
    temporary = Path(tempfile.mkdtemp(prefix=".render_", dir=data_dir))
    try:
        figures, tables = [temporary / name for name in ("figures", "tables")]
        tables.mkdir()
        make_figures(reference, [c["observables"] for c in cases], figures,
                     table_directory=tables, plot_min_n=plot_min_n)
        accuracy_rows = []
        diagnostic_rows = []
        for case in cases:
            for source, values in case["accuracy_budget"]["sources"].items():
                accuracy_rows.append({"N": case["N"], "source": source, "h2": values["h2"],
                                      "max": values["max"], "accepted": True})
            for name, state in case["states"].items():
                diagnostic = state["diagnostics"]
                diagnostic_rows.append({"N": case["N"], "state": name, "cutoff": state["pair_cutoff"],
                    "el_residual_inf": diagnostic["el_residual_inf_norm"],
                    "el_over_h_squared": diagnostic["el_residual_over_h2"],
                    "full_gradient_inf": diagnostic["full_gradient_inf_norm"],
                    "reduced_gradient_inf": diagnostic["reduced_gradient_inf_norm"],
                    "curvature_min": diagnostic["curvature_min_eigenvalue"],
                    "curvature_eigenvector_residual": diagnostic["curvature_residual_max"],
                    "h2_sensitivity_budget_ratio": case["accuracy_budget"]["h2"]["ratio"],
                    "max_sensitivity_budget_ratio": case["accuracy_budget"]["max"]["ratio"]})
        write_csv(tables / "accuracy_budget.csv", accuracy_rows)
        write_csv(tables / "endpoint_diagnostics.csv", diagnostic_rows)
        # Each complete directory is swapped only after every derived file exists.
        # Roll back an interrupted swap before surfacing an error.
        installed, backups = [], []
        try:
            for name in ("figures", "tables"):
                destination, backup = data_dir / name, temporary / (name+"_previous")
                if destination.exists():
                    if destination.is_symlink() or destination.is_junction():
                        raise ValueError(f"Refusing redirected output directory: {destination}")
                    os.replace(destination, backup)
                    backups.append((destination, backup))
                os.replace(temporary/name, destination)
                installed.append(destination)
        except BaseException:
            for path in reversed(installed):
                _remove_directory(path, data_dir)
            for destination, backup in reversed(backups):
                os.replace(backup, destination)
            raise
    finally:
        _remove_directory(temporary, data_dir)
    log(f"Figures: {data_dir / 'figures'}")
    log(f"Tables: {data_dir / 'tables'}")
    return data_dir / "figures"


def prepare(data_dir):
    from .validation import run_observable_checks
    if (data_dir / "publication_manifest.json").exists():
        _load_publication(data_dir)
        log("Reuse the complete validated dataset; no numerical preparation is needed.")
        _render(data_dir)
        _cleanup_completed_preparation(data_dir)
        return 0
    run = data_dir / "preparation"
    if not run.exists():
        run = _new_run(data_dir)
    record = read_json(run / "run_manifest.json")
    require_configuration(record)
    if record.get("source_fingerprints") != source_fingerprints():
        raise ValueError("Preparation sources changed; use the matching sources or a new data folder.")
    record = _recover_interrupted_worker(run)
    checks = run_observable_checks()
    if not checks["passed"]:
        raise RuntimeError("Focused observable checks failed; preparation stopped.")
    if not record["reference"]:
        remaining = record["budget_seconds"]-record["elapsed_seconds"]
        if remaining < 30:
            raise RuntimeError("The cumulative preparation budget is exhausted.")
        _run_child(run, "reference", remaining)
        record = read_json(run / "run_manifest.json")
        if not record["reference"]:
            raise RuntimeError("Continuum preparation is unresolved; temporary checkpoints are retained.")
    for N in PREPARATION_ORDER:
        record = read_json(run / "run_manifest.json")
        entry = record["cases"].get(str(N), {})
        if entry.get("accepted"):
            cached = load_bundle(run / entry["path"])
            _require_case_evidence(cached)
            log(f"N={N}: reuse the completed validated case.")
            continue
        remaining = record["budget_seconds"]-record["elapsed_seconds"]
        successful = [(int(k), v["seconds"]) for k, v in record["cases"].items()
                      if v.get("accepted") and v.get("seconds")]
        estimate = max((seconds*N/n for n, seconds in successful if n <= N), default=0)
        if remaining < max(30, 1.1*estimate):
            log(f"Stop before N={N}: {remaining/60:.1f} minutes remain in the cumulative budget.")
            break
        started = time.monotonic()
        status = _run_child(run, "case", remaining, N)
        record = read_json(run / "run_manifest.json")
        entry = {"status": status, "accepted": False, "seconds": time.monotonic()-started}
        result_file = run / f"N{N:05d}/result.json"
        if result_file.exists() and status == "finished":
            result = read_json(result_file)
            require_configuration(result)
            entry.update({"path": f"N{N:05d}/{result['path']}", "accepted": result["accepted"],
                          "status": "validated" if result["accepted"] else "unresolved"})
        record["cases"][str(N)] = entry
        write_json(run / "run_manifest.json", record)
        if status == "time_limit":
            break
    record = read_json(run / "run_manifest.json")
    complete = all(record["cases"].get(str(N), {}).get("accepted") for N in N_VALUES)
    record["status"] = "complete" if complete else "incomplete"
    write_json(run / "run_manifest.json", record)
    if not complete:
        log("Preparation remains incomplete; no publication or figures have been created.")
        return 2
    reference = load_bundle(run / record["reference"])
    cases = [load_bundle(run / record["cases"][str(N)]["path"]) for N in N_VALUES]
    evidence = {"passed": True, "local_checks": checks, "reference": reference["verification"],
                "cases": [c["verification"] for c in cases], "endpoint_count": 48,
                "preparation_elapsed_seconds": record["elapsed_seconds"],
                "preparation_budget_seconds": record["budget_seconds"],
                "source_fingerprints": source_fingerprints(), "runtime": runtime_versions()}
    publish_dataset(data_dir, reference, cases, evidence)
    _render(data_dir)
    _remove_directory(run, data_dir)
    return 0


def verify(data_dir):
    """Recheck all forty-eight saved endpoints without modifying any profile."""
    from .validation import run_observable_checks, validate_case, validate_reference
    _, reference, cases = _load_publication(data_dir)
    started = time.monotonic()
    checks = run_observable_checks()
    reference_report = validate_reference(reference)
    reports = []
    for case in cases:
        _, report = validate_case(case, reference, progress=log)
        reports.append(report)
        log(f"Verify N={case['N']}: {'PASS' if report['passed'] else 'FAIL'}")
    passed = bool(checks["passed"] and reference_report["passed"]
                  and len(reports) == len(N_VALUES) and all(r["passed"] for r in reports))
    report = {"passed": passed, "local_checks": checks, "reference": reference_report,
              "cases": reports, "endpoint_count": 4*len(reports),
              "elapsed_seconds": time.monotonic()-started,
              "source_fingerprints": source_fingerprints(), "runtime": runtime_versions()}
    write_json(data_dir / "verification.json", report)
    log(f"Verification report: {data_dir / 'verification.json'}")
    return 0 if passed else 2


def _parse_plot_n(value):
    """Parse N=<integer> as a plotting cutoff retaining the full fit window."""
    allowed = tuple(N for N in N_VALUES if N <= FIT_N[0])
    message = (f"Use --plot N=<integer>; select a number less than or equal to {FIT_N[0]} "
               f"from {{{', '.join(map(str, allowed))}}}.")
    if not value.startswith("N=") or not value[2:].isascii() or not value[2:].isdigit():
        raise argparse.ArgumentTypeError(message)
    try:
        N = int(value[2:])
    except ValueError:
        raise argparse.ArgumentTypeError(message) from None
    if N not in allowed:
        raise argparse.ArgumentTypeError(message)
    return N


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--prepare", action="store_true", help="Explicitly prepare/resume expensive results.")
    actions.add_argument("--verify", action="store_true", help="Check all saved states without optimization.")
    actions.add_argument("--plot", type=_parse_plot_n, metavar="N=<integer>",
                         help="Render saved profiles with convergence plots starting at N (default: 20).")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--worker", choices=("reference", "case"), help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", help=argparse.SUPPRESS)
    parser.add_argument("--case", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--seconds", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.plot is not None and args.worker:
        parser.error("--plot cannot be combined with internal --worker execution.")
    try:
        if args.worker:
            if args.run_dir is None or args.seconds is None or args.seconds <= 0:
                raise ValueError("Internal workers require a workspace and positive time limit.")
            if args.worker == "case" and args.case not in N_VALUES:
                raise ValueError("The worker size must belong to the configured study.")
            _worker(args)
            return 0
        data_dir = args.data_dir.resolve()
        if args.prepare:
            return prepare(data_dir)
        if args.verify:
            return verify(data_dir)
        _render(data_dir, plot_min_n=20 if args.plot is None else args.plot)
        return 0
    except (FileNotFoundError, ValueError, RuntimeError, TimeoutError) as error:
        if args.worker:
            traceback.print_exc()
        else:
            log(f"Cannot complete this action: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
