#!/usr/bin/env python3
"""One-click MSD slope check from two-column lag/MSD data.

Accepts a LAMMPS-like or plain two-column file (lag_time, MSD), fits
M(τ) = A + Bτ over a chosen or auto-detected window, reports D = B/(2d),
runs the offset-corrected log-slope consistency check, and writes figures.

Examples
--------
  python3 msd_check.py examples/msd.dat --window 80 150
  python3 msd_check.py msd.dat --auto --dim 3 --out results/
  python3 msd_check.py msd.dat --col 0 4 --timestep 0.001 --window LO HI
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Style (self-contained; no external plot_style import)
# ---------------------------------------------------------------------------

PALETTE = {
    "trace": "#440154",
    "fit": "#C56718",
    "reference": "#8A94A6",
    "grid": "#DCE2E8",
    "text": "#25344A",
    "spine": "#687787",
}

RC_PARAMS = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.labelcolor": PALETTE["text"],
    "axes.edgecolor": PALETTE["spine"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
}

LOG_SLOPE_POINTS = 400
FIGURE_DPI = 300


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FitResult:
    """Affine fit M = A + B τ over a lag window, plus slope-check metrics."""

    lag_lo: float
    lag_hi: float
    intercept: float  # A
    slope: float  # B = dM/dτ
    D: float
    dim: int
    r_squared: float
    beta_raw_mean: float
    beta_corrected_mean: float
    window_stability: float  # max relative |ΔD| under ±10% window shifts


@dataclass(frozen=True)
class MSDData:
    lags: np.ndarray
    msd: np.ndarray


def load_msd(
    path: Path,
    *,
    lag_col: int = 0,
    msd_col: int = 1,
    timestep: float | None = None,
    skip_zero: bool = False,
) -> MSDData:
    """Load lag/MSD columns from whitespace-delimited text (LAMMPS-friendly).

    Lines starting with ``#`` are ignored. ``timestep`` converts a first
    column of timesteps into lag time via ``lag = step * timestep`` when
    set. Columns are 0-based.
    """
    rows: list[list[float]] = []
    with path.open() as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            # LAMMPS sometimes writes a leading integer row count; skip non-floats
            parts = stripped.replace(",", " ").split()
            try:
                values = [float(p) for p in parts]
            except ValueError:
                continue
            if len(values) <= max(lag_col, msd_col):
                continue
            rows.append(values)

    if len(rows) < 3:
        raise ValueError(f"{path}: need at least 3 numeric rows, found {len(rows)}.")

    array = np.asarray(rows, dtype=np.float64)
    lags = array[:, lag_col].copy()
    msd = array[:, msd_col].copy()
    if timestep is not None:
        if timestep <= 0:
            raise ValueError("--timestep must be positive.")
        # Treat column as absolute step; convert to lag relative to first row
        lags = (lags - lags[0]) * timestep

    order = np.argsort(lags)
    lags, msd = lags[order], msd[order]
    # Drop duplicate lags (keep last)
    _, unique = np.unique(lags, return_index=True)
    lags, msd = lags[unique], msd[unique]

    if skip_zero:
        keep = lags > 0
        lags, msd = lags[keep], msd[keep]

    if len(lags) < 3:
        raise ValueError("Fewer than 3 points remain after cleaning.")
    if not np.all(np.isfinite(lags)) or not np.all(np.isfinite(msd)):
        raise ValueError("Non-finite values in lag or MSD columns.")
    if np.any(np.diff(lags) <= 0):
        raise ValueError("Lag times must be strictly increasing after cleaning.")

    return MSDData(lags=lags, msd=msd)


def calculate_log_slope(
    lags: np.ndarray, values: np.ndarray, n_points: int = LOG_SLOPE_POINTS
) -> tuple[np.ndarray, np.ndarray]:
    """Return β = d ln M / d ln τ on an evenly log-spaced grid (unsmoothed)."""
    selected = np.flatnonzero((lags > 0) & (values > 0))
    if len(selected) < 3:
        raise ValueError("Logarithmic slopes need ≥3 positive (lag, value) pairs.")
    # Require a contiguous positive block for a meaningful slope curve
    grid = np.geomspace(lags[selected[0]], lags[selected[-1]], n_points)
    interpolated = np.interp(grid, lags[selected], values[selected])
    beta = np.gradient(np.log(interpolated), np.log(grid))
    return grid, beta


def fit_window(
    data: MSDData, lag_lo: float, lag_hi: float, dim: int
) -> FitResult:
    """Fit A + Bτ on [lag_lo, lag_hi] and compute slope-check diagnostics."""
    if dim not in (1, 2, 3):
        raise ValueError("--dim must be 1, 2, or 3.")
    if lag_hi <= lag_lo:
        raise ValueError("Window upper bound must exceed lower bound.")

    mask = (data.lags >= lag_lo) & (data.lags <= lag_hi)
    n = int(mask.sum())
    if n < 3:
        raise ValueError(
            f"Fitting window [{lag_lo}, {lag_hi}] contains {n} points; need ≥3."
        )

    tau = data.lags[mask]
    m = data.msd[mask]
    slope, intercept = np.polyfit(tau, m, 1)
    predicted = intercept + slope * tau
    ss_res = float(np.sum((m - predicted) ** 2))
    ss_tot = float(np.sum((m - m.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    D = float(slope) / (2 * dim)

    # Mean β in-window for raw and offset-corrected MSD
    try:
        grid, beta_raw = calculate_log_slope(data.lags, data.msd)
        in_win = (grid >= lag_lo) & (grid <= lag_hi)
        beta_raw_mean = float(np.nanmean(beta_raw[in_win])) if in_win.any() else float("nan")
    except ValueError:
        beta_raw_mean = float("nan")

    corrected = data.msd - intercept
    try:
        grid_c, beta_c = calculate_log_slope(data.lags, corrected)
        in_win_c = (grid_c >= lag_lo) & (grid_c <= lag_hi)
        beta_corrected_mean = (
            float(np.nanmean(beta_c[in_win_c])) if in_win_c.any() else float("nan")
        )
    except ValueError:
        beta_corrected_mean = float("nan")

    stability = _window_stability(data, lag_lo, lag_hi, dim, D)
    return FitResult(
        lag_lo=float(lag_lo),
        lag_hi=float(lag_hi),
        intercept=float(intercept),
        slope=float(slope),
        D=D,
        dim=dim,
        r_squared=r_squared,
        beta_raw_mean=beta_raw_mean,
        beta_corrected_mean=beta_corrected_mean,
        window_stability=stability,
    )


def _window_stability(
    data: MSDData, lag_lo: float, lag_hi: float, dim: int, D_ref: float
) -> float:
    """Max relative |ΔD| when shifting each bound by ±10% of window width."""
    if D_ref == 0.0:
        return float("nan")
    width = lag_hi - lag_lo
    shifts = (-0.1, 0.0, 0.1)
    relative: list[float] = []
    for d_lo in shifts:
        for d_hi in shifts:
            lo = lag_lo + d_lo * width
            hi = lag_hi + d_hi * width
            if hi <= lo:
                continue
            mask = (data.lags >= lo) & (data.lags <= hi)
            if mask.sum() < 3:
                continue
            slope, _ = np.polyfit(data.lags[mask], data.msd[mask], 1)
            D = slope / (2 * dim)
            relative.append(abs(D - D_ref) / abs(D_ref))
    return float(max(relative)) if relative else float("nan")


def suggest_window(
    data: MSDData,
    *,
    dim: int = 3,
    frac_lo: float = 0.10,
    frac_hi: float = 0.40,
    min_points: int = 20,
) -> tuple[float, float]:
    """Suggest a mid-range lag window by maximising local linear-slope flatness.

    Searches candidate windows in the middle of the lag range (default 10–40%
    of max lag for the lower edge, with variable width) and picks the interval
    whose local finite-difference slope has the smallest coefficient of
    variation while keeping a positive mean slope.
    """
    positive = data.lags > 0
    lags = data.lags[positive]
    msd = data.msd[positive]
    if len(lags) < min_points:
        raise ValueError("Too few points for automatic window suggestion.")

    t_max = float(lags[-1])
    # Local linear slope via central differences on the raw series
    s = np.gradient(msd, lags)

    best: tuple[float, float, float] | None = None  # (score, lo, hi)
    lower_candidates = np.linspace(frac_lo * t_max, 0.35 * t_max, 12)
    widths = np.linspace(0.08 * t_max, 0.30 * t_max, 10)

    for lo in lower_candidates:
        for width in widths:
            hi = lo + width
            if hi > 0.6 * t_max:
                continue
            mask = (lags >= lo) & (lags <= hi)
            if mask.sum() < min_points:
                continue
            segment = s[mask]
            mean_s = float(np.mean(segment))
            if mean_s <= 0:
                continue
            cv = float(np.std(segment) / mean_s)
            # Prefer flatter slopes; mild preference for longer windows
            score = cv / np.sqrt(mask.sum())
            if best is None or score < best[0]:
                best = (score, float(lo), float(hi))

    if best is None:
        # Fallback: middle third of the record
        lo, hi = 0.15 * t_max, 0.40 * t_max
        return lo, hi
    return best[1], best[2]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _apply_style() -> None:
    plt.rcParams.update(RC_PARAMS)


def _style_axes(*axes: plt.Axes) -> None:
    for axis in axes:
        axis.grid(True, which="major", color=PALETTE["grid"], linewidth=0.8)
        axis.grid(False, which="minor")
        axis.set_axisbelow(True)


def plot_linear(
    data: MSDData, fit: FitResult, out: Path, time_unit: str, msd_unit: str
) -> None:
    """Linear-axis MSD with fit and short-lag inset."""
    _apply_style()
    figure, axis = plt.subplots(figsize=(7.2, 4.0), layout="constrained")
    end = float(data.lags[-1])
    zoom = min(0.1 * end, fit.lag_lo) if fit.lag_lo > 0 else 0.1 * end
    zoom = max(zoom, data.lags[1] if len(data.lags) > 1 else end * 0.1)

    stride = max(1, len(data.lags) // 800)
    shown = np.unique(np.append(np.arange(0, len(data.lags), stride), len(data.lags) - 1))

    axis.plot(
        data.lags[shown], data.msd[shown],
        color=PALETTE["trace"], linewidth=1.8, label="MSD",
    )
    fit_lags_ext = np.array([0.0, end])
    axis.plot(
        fit_lags_ext, fit.intercept + fit.slope * fit_lags_ext,
        color=PALETTE["fit"], linewidth=1.3, linestyle="--",
        label=r"Extrapolated $A+B\tau$",
    )
    fit_ends = np.array([fit.lag_lo, fit.lag_hi])
    axis.plot(
        fit_ends, fit.intercept + fit.slope * fit_ends,
        color=PALETTE["fit"], linewidth=2.6, marker="o", markersize=3.5,
        label=f"Fit: {fit.lag_lo:.4g}–{fit.lag_hi:.4g}",
    )
    axis.set(
        xlim=(0, end),
        ylim=(0, max(data.msd.max(), fit.intercept + fit.slope * end) * 1.06),
        xlabel=rf"Lag time $\tau$ [{time_unit}]",
        ylabel=rf"MSD, $M$ [{msd_unit}]",
    )
    axis.set_title("(a) Linear MSD", loc="left", color=PALETTE["text"])
    axis.legend(loc="lower right")

    inset = axis.inset_axes((0.07, 0.55, 0.38, 0.37), facecolor="white", zorder=5)
    zmask = data.lags <= zoom
    inset.plot(data.lags[zmask], data.msd[zmask], color=PALETTE["trace"], linewidth=1.8)
    inset.plot(
        [0, zoom], fit.intercept + fit.slope * np.array([0.0, zoom]),
        color=PALETTE["fit"], linewidth=1.3, linestyle="--",
    )
    upper = max(data.msd[zmask].max(), fit.intercept + fit.slope * zoom) * 1.08
    inset.set(
        xlim=(0, zoom), ylim=(0, max(upper, 1e-30)),
        xlabel=rf"$\tau$ [{time_unit}]", ylabel=rf"$M$ [{msd_unit}]",
    )
    inset.set_title(f"(b) 0–{zoom:.4g}", loc="left", color=PALETTE["text"], fontsize=9.5)
    inset.tick_params(labelsize=8)
    _style_axes(axis, inset)

    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out.with_suffix(".png"), dpi=FIGURE_DPI, bbox_inches="tight")
    figure.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_log_slope_check(
    data: MSDData, fit: FitResult, out: Path, time_unit: str, msd_unit: str
) -> None:
    """Four-panel log MSD + β before/after offset correction (the slope check)."""
    _apply_style()
    figure, axes = plt.subplots(
        2, 2, figsize=(7.2, 5.0), layout="constrained",
        gridspec_kw={"height_ratios": [2.4, 1.0]},
    )

    panels = (
        (False, "(a) MSD", rf"MSD, $M$ [{msd_unit}]", r"$\widehat A+\widehat B\tau$",
         "(c)", r"$\beta$"),
        (True, "(b) Offset-corrected MSD", rf"$M-\widehat A$ [{msd_unit}]",
         r"$\widehat B\tau$", "(d)", r"$\beta_A$"),
    )
    lag_limits = (
        max(data.lags[data.lags > 0].min(), data.lags[-1] * 1e-4),
        float(data.lags[-1]),
    )

    for col, (corrected, title, ylabel, fit_label, slope_title, slope_ylabel) in enumerate(panels):
        values = data.msd - fit.intercept if corrected else data.msd
        ax = axes[0, col]
        positive = (data.lags > 0) & (values > 0)
        ref = np.geomspace(data.lags[positive][0], data.lags[-1], 300)
        ref_vals = (fit.slope * ref) if corrected else (fit.intercept + fit.slope * ref)
        ax.loglog(
            data.lags[positive], values[positive],
            color=PALETTE["trace"], linewidth=1.8, label="Measured",
        )
        ax.loglog(
            ref[ref_vals > 0], ref_vals[ref_vals > 0],
            color=PALETTE["fit"], linewidth=1.3, linestyle="--", label=fit_label,
        )
        fit_ends = np.array([fit.lag_lo, fit.lag_hi])
        fit_vals = fit.slope * fit_ends if corrected else fit.intercept + fit.slope * fit_ends
        ax.loglog(
            fit_ends, fit_vals,
            color=PALETTE["fit"], linewidth=2.6, marker="o", markersize=3.5,
            label=f"{fit.lag_lo:.4g}–{fit.lag_hi:.4g}",
        )
        ax.set_xlim(lag_limits)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", color=PALETTE["text"])
        ax.legend(loc="upper left")

        sax = axes[1, col]
        try:
            grid, beta = calculate_log_slope(data.lags, values)
            in_win = (grid >= fit.lag_lo) & (grid <= fit.lag_hi)
            sax.axhline(1.0, color=PALETTE["reference"], linewidth=1.0, linestyle="--")
            clipped = np.where((beta >= -2) & (beta <= 4), beta, np.nan)
            sax.semilogx(grid, clipped, color=PALETTE["trace"], linewidth=1.8)
            sax.semilogx(grid[in_win], beta[in_win], color=PALETTE["fit"], linewidth=2.6)
        except ValueError:
            sax.text(0.5, 0.5, "β undefined", transform=sax.transAxes, ha="center")
        sax.set(
            xlim=lag_limits, ylim=(0, 2), yticks=[0, 1, 2],
            xlabel=rf"Lag time $\tau$ [{time_unit}]", ylabel=slope_ylabel,
        )
        sax.set_title(slope_title, loc="left", color=PALETTE["text"])
        ax.sharex(sax)
        ax.tick_params(labelbottom=False)

    _style_axes(*axes.ravel())
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out.with_suffix(".png"), dpi=FIGURE_DPI, bbox_inches="tight")
    figure.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="msd_check",
        description=(
            "One-click MSD slope check: fit A+Bτ, report D=B/(2d), "
            "and plot the offset-corrected log-slope consistency check."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Input: whitespace-delimited text. Comment lines start with #.\n"
            "Default columns: 0 = lag time, 1 = MSD (LAMMPS total MSD often col 4).\n"
            "Either --window or --auto is required.\n"
            "\n"
            "Examples:\n"
            "  python3 msd_check.py examples/msd.dat --window 80 150\n"
            "  python3 msd_check.py msd.dat --auto --dim 3\n"
            "  python3 msd_check.py msd.dat --col 0 4 --timestep 0.001 --window LO HI\n"
        ),
    )
    parser.add_argument(
        "input", type=Path,
        help="Two-column (or multi-column) MSD file: lag_time and MSD",
    )
    window = parser.add_mutually_exclusive_group(required=True)
    window.add_argument(
        "--window", nargs=2, type=float, metavar=("LO", "HI"),
        help="Fitting lag window [lo, hi] in the same units as the lag column",
    )
    window.add_argument(
        "--auto", action="store_true",
        help=(
            "Heuristic mid-range window from local-slope flatness "
            "(candidate only; not a validated diffusion window)"
        ),
    )
    parser.add_argument(
        "--dim", type=int, default=3, choices=(1, 2, 3),
        help="Spatial dimensionality for D = B/(2*dim) (default: 3 → B/6)",
    )
    parser.add_argument(
        "--col", nargs=2, type=int, default=[0, 1], metavar=("LAG", "MSD"),
        help="0-based column indices for lag and MSD (default: 0 1)",
    )
    parser.add_argument(
        "--timestep", type=float, default=None,
        help="If set, multiply (step - step0) by this factor to get lag time",
    )
    parser.add_argument(
        "--time-unit", default="ns",
        help="Label for lag-time axis (default: ns)",
    )
    parser.add_argument(
        "--msd-unit", default="L^2",
        help="Label for MSD axis (default: L^2)",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("output"),
        help="Output directory for figures and summary JSON (default: output/)",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip figure generation; only print and write JSON summary",
    )
    return parser


def print_summary(
    data: MSDData, fit: FitResult, time_unit: str, msd_unit: str
) -> None:
    factor = 2 * fit.dim
    d_unit = f"{msd_unit}/{time_unit}"
    print()
    print("=" * 60)
    print("  MSD slope check")
    print("=" * 60)
    print(f"  Points           : {len(data.lags)}")
    print(f"  Lag range       : {data.lags[0]:.6g} – {data.lags[-1]:.6g} {time_unit}")
    print(f"  Fit window       : {fit.lag_lo:.6g} – {fit.lag_hi:.6g} {time_unit}")
    print(f"  Intercept A      : {fit.intercept:.6g} {msd_unit}")
    print(f"  Slope B = dM/dτ  : {fit.slope:.6g} {d_unit}")
    print(f"  D = B/({factor})       : {fit.D:.6g} {d_unit}")
    print(f"  R² (OLS)         : {fit.r_squared:.6f}")
    print(f"  ⟨β⟩ raw (window) : {fit.beta_raw_mean:.4f}")
    print(f"  ⟨β_A⟩ corrected  : {fit.beta_corrected_mean:.4f}   (expect ≈ 1)")
    print(f"  Window stability : {fit.window_stability:.2%} max |ΔD|/|D| (±10% bounds)")
    print("=" * 60)
    print("  Notes")
    print("  • OLS R² / stderr are not reliable uncertainties for correlated MSD.")
    print("  • ⟨β_A⟩≈1 is a consistency check after subtracting Â, not proof of D.")
    print("  • Prefer Bullerjahn et al. (2020) for covariance-aware errors.")
    print("=" * 60)
    print()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.input.is_file():
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        return 1

    try:
        data = load_msd(
            args.input,
            lag_col=args.col[0],
            msd_col=args.col[1],
            timestep=args.timestep,
        )
        if args.auto:
            lo, hi = suggest_window(data, dim=args.dim)
            print(
                "warning: --auto window is a heuristic candidate only; "
                "not a validated diffusion window.",
                file=sys.stderr,
            )
            print(f"Auto window suggestion: [{lo:.6g}, {hi:.6g}]")
        else:
            lo, hi = args.window
        fit = fit_window(data, lo, hi, args.dim)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print_summary(data, fit, args.time_unit, args.msd_unit)

    args.out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out / "summary.json"
    payload = {
        "input": str(args.input.resolve()),
        "n_points": len(data.lags),
        "lag_min": float(data.lags[0]),
        "lag_max": float(data.lags[-1]),
        "time_unit": args.time_unit,
        "msd_unit": args.msd_unit,
        "fit": asdict(fit),
    }
    summary_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {summary_path}")

    if not args.no_plot:
        try:
            plot_linear(
                data, fit, args.out / "msd-linear",
                args.time_unit, args.msd_unit,
            )
            plot_log_slope_check(
                data, fit, args.out / "msd-slope-check",
                args.time_unit, args.msd_unit,
            )
            print(f"Wrote {args.out / 'msd-linear.png'}")
            print(f"Wrote {args.out / 'msd-slope-check.png'}")
        except Exception as exc:  # noqa: BLE001 — surface plotting errors cleanly
            print(f"warning: plotting failed: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
