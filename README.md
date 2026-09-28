# MSD Slope Check

Fit \(M(\tau)=A+B\tau\) from a two-column MSD file (lag time, MSD), report \(D=B/(2d)\), and run the offset-corrected log-slope consistency check (slope check). Companion notes: [tutorial.md](tutorial.md). Docs site: `zensical serve`.

## Suggested path

1. Install dependencies (Python 3.10+):

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. Run the example with the tutorial window:

```bash
python3 msd_check.py examples/msd.dat --window 80 150 --out output
```

3. Inspect `output/msd-linear.png`, `output/msd-slope-check.png`, and the terminal summary, then read [tutorial.md](tutorial.md) §§1–6 (appendices as needed).

### Documentation site

```bash
pip install zensical   # also listed in requirements.txt
zensical serve         # http://127.0.0.1:8000
# zensical build       # writes site/
```

## Expected summary (example)

For `examples/msd.dat --window 80 150 --dim 3` you should see roughly:

```text
  Fit window       : 80 – 150 ns
  Intercept A      : ~26
  Slope B = dM/dτ  : ~1.25
  D = B/(6)        : ~0.208   [L^2/ns]
  ⟨β_A⟩ corrected  : ~1.00    (expect ≈ 1)
```

Exact digits depend on floating point; the order of magnitude and \(\langle\beta_A\rangle\approx 1\) are the checks that matter.

## Input format

Whitespace-delimited text; lines starting with `#` are ignored. Default columns: `0` = lag time, `1` = MSD.

```text
# lag_time_ns    MSD
0.0000000000e+00  0.0000000000e+00
1.0000000000e-01  1.5061700000e-01
...
```

**LAMMPS `fix ave/time` + `compute msd`:** after the timestep column, components are \(x,y,z,\) then the total MSD. Use the total column (0-based index `4`) and convert timesteps to lag time:

```bash
python3 msd_check.py msd.dat --col 0 4 --timestep 0.001 --time-unit ns --window LO HI
```

`compute msd` uses a single reference configuration (atom average, not multi-origin). Prefer a time-origin-averaged MSD from post-processing when that is the intended estimator; see tutorial Appendix B.

## Options

| Option | Meaning |
|--------|---------|
| `--window LO HI` | Fit over `[LO, HI]` (same units as lag). **Required** unless `--auto`. |
| `--auto` | Heuristic mid-range window from local-slope flatness. **Candidate only — not a validated diffusion window.** |
| `--dim {1,2,3}` | \(D=B/(2\cdot\mathrm{dim})\) (default `3` → \(B/6\)) |
| `--col LAG MSD` | 0-based column indices (default `0 1`) |
| `--timestep DT` | Lag = `(step − step0) · DT` |
| `--time-unit` / `--msd-unit` | Axis / unit labels (default `ns`, `L^2`) |
| `--out DIR` | Output directory (default `output/`) |
| `--no-plot` | JSON + print only |

You must pass either `--window` or `--auto`.

## Outputs

- `output/summary.json` — fit parameters and diagnostics
- `output/msd-linear.png` — linear MSD + fit + short-lag inset
- `output/msd-slope-check.png` — log MSD and \(\beta\) / \(\beta_A\)

## Caveats

- \(\langle\beta_A\rangle\approx 1\) and high OLS \(R^2\) are consistency checks of the same linear model, not independent proof of diffusion.
- OLS standard errors are not reliable for correlated MSD points; see Bullerjahn et al. (2020) in the tutorial references.
- Match `--dim` to whether the MSD column is the total or a single Cartesian component.

## Layout

| Path | Role |
|------|------|
| `msd_check.py` | CLI |
| `tutorial.md` | Method notes |
| `figures/` | Tutorial figures |
| `examples/msd.dat` | Example two-column MSD |
| `zensical.toml` / `docs/` | Documentation site |
| `requirements.txt` | `numpy`, `matplotlib`, `zensical` |
