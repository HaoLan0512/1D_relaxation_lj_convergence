"""Recreate the three publication figures from saved, validated numbers."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterSciNotation, LogLocator
import numpy as np

from .config import ERROR_FIELDS, FIT_N


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def slope_rows(rows):
    result = []
    for name, selected in (("predefined_fine", [r for r in rows if int(r["N"]) in FIT_N]),
                           ("all_points", rows)):
        complete = name != "predefined_fine" or tuple(int(r["N"]) for r in selected) == FIT_N
        for field in ERROR_FIELDS:
            slope = float(np.polyfit(np.log([r["N"] for r in selected]),
                                     np.log([r[field] for r in selected]), 1)[0]) if len(selected) >= 2 and complete else None
            result.append({"fit": name, "error": field,
                           "N": " ".join(str(r["N"]) for r in selected),
                           "slope": slope, "complete_window": complete})
    return result


def make_figures(reference, rows, directory, incomplete=False, table_directory=None,
                 plot_min_n=20):
    """Write figures from N >= plot_min_n and tables retaining every input row."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    table_directory = directory if table_directory is None else Path(table_directory)
    table_directory.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: int(row["N"]))
    if any(not np.isfinite(float(r[key])) or float(r[key]) <= 0
           for r in rows for key in ERROR_FIELDS):
        raise ValueError("Logarithmic plots require finite positive errors.")
    prefix = "incomplete_" if incomplete else ""
    style = {"font.family": "DejaVu Sans", "font.size": 9,
             "axes.labelsize": 10, "legend.fontsize": 8,
             "xtick.labelsize": 9, "ytick.labelsize": 9,
             "axes.linewidth": 0.7, "pdf.fonttype": 42, "savefig.dpi": 300}
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(4.8, 3.35), layout="constrained")
        x = np.asarray(reference["x_display"])
        ax.plot(x, reference["v_display"], color="#1f77b4", lw=1.6)
        ticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
        labels = [r"$-\pi$", r"$-\pi/2$", "$0$", r"$\pi/2$", r"$\pi$"]
        ax.set(xlim=(-np.pi, np.pi), ylim=(-3.3, 3.3), xlabel="$x$", ylabel="$v_0(x)$")
        ax.set_xticks(ticks, labels)
        ax.set_yticks([-np.pi, 0, np.pi], [r"$-\pi$", "$0$", r"$\pi$"])
        ax.grid(alpha=0.20, lw=0.5)
        inset = ax.inset_axes([0.58, 0.11, 0.36, 0.32])
        inset.plot(x, reference["q_display"], color="#1f77b4", lw=1.0)
        inset.set_xlim(-np.pi, np.pi)
        inset.set_xticks([-np.pi, 0, np.pi], [r"$-\pi$", "$0$", r"$\pi$"])
        inset.tick_params(labelsize=7, pad=1)
        inset.set_title(r"$u_1(x)-u_2(x)$", fontsize=8, pad=3)
        inset.grid(alpha=0.16, lw=0.4)
        if incomplete:
            ax.text(0.02, 0.98, "Incomplete preparation", transform=ax.transAxes,
                    va="top", fontsize=7, color="0.4")
        for extension in ("png", "pdf"):
            fig.savefig(directory / f"{prefix}v_phase_fixed.{extension}")
        plt.close(fig)

        if rows:
            n = np.asarray([r["N"] for r in rows], dtype=float)
            if len(n) != len(set(n)) or np.any(np.diff(n) <= 0):
                raise ValueError("Each N must occur once in increasing order.")
        display_rows = [r for r in rows if int(r["N"]) >= plot_min_n]
        if display_rows:
            n = np.asarray([r["N"] for r in display_rows], dtype=float)
            specifications = [
                ("convergence_loglog", ERROR_FIELDS[:2],
                 (r"$H_h^2$ error", "Maximum error"),
                 (r"$\|u_d-u_c\|_{H_h^2}$", r"$\|u_d-u_c\|_\infty$"), 1),
                ("relative_convergence_loglog", ERROR_FIELDS[2:],
                 (r"$H^2$ error", "Maximum error"),
                 (r"$\|e_N^{\mathrm{rel}}\|_{H^2}$", r"$\|e_N^{\mathrm{rel}}\|_{L^\infty}$"), 2)]
            for name, fields, titles, labels, power in specifications:
                fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), layout="constrained")
                for index, (ax, field, title, label) in enumerate(zip(axes, fields, titles, labels)):
                    values = np.asarray([r[field] for r in display_rows])
                    ax.loglog(n, values, "s-", color="#2ca02c", lw=1.3,
                              markersize=3.4, label=label)
                    # Fixed C, chosen above all data: this is a guide, not a fit.
                    constant = 2.0 * np.max(values * n**power)
                    ax.loglog(n, constant * n**(-power), "--", color="0.45",
                              lw=1.2, label=rf"$N^{{-{power}}}$ rate guide")
                    ax.set(xlabel="$N$", ylabel=title)
                    if n[-1] / n[0] < 10:
                        # Avoid crowded minor labels when less than a decade is shown.
                        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
                        ax.xaxis.set_minor_formatter(LogFormatterSciNotation(
                            base=10, labelOnlyBase=False, minor_thresholds=(np.inf, np.inf)))
                    ax.text(0.04, 0.06, f"({chr(97+index)})", transform=ax.transAxes)
                    ax.grid(which="major", alpha=0.22, lw=0.5)
                    ax.legend(frameon=False, loc="upper right")
                    if len(ax.lines) != 2:
                        raise AssertionError("A convergence panel must have exactly two curves.")
                if incomplete:
                    fig.suptitle("Incomplete preparation", fontsize=8, color="0.4")
                for extension in ("png", "pdf"):
                    fig.savefig(directory / f"{prefix}{name}.{extension}")
                plt.close(fig)
    values = [{"N": int(r["N"]), "h": 4*np.pi/(2*int(r["N"])+1),
               **{key: float(r[key]) for key in ERROR_FIELDS}} for r in rows]
    write_csv(table_directory / "plotted_values.csv", values)
    slopes = slope_rows(values)
    write_csv(table_directory / "fitted_slopes.csv", slopes)
    adjacent = []
    for first, second in zip(values[:-1], values[1:]):
        adjacent.append({"N_from": first["N"], "N_to": second["N"],
                         **{key: float(np.log(second[key]/first[key]) /
                                       np.log(second["N"]/first["N"])) for key in ERROR_FIELDS}})
    write_csv(table_directory / "adjacent_rates.csv", adjacent)
    return slopes
