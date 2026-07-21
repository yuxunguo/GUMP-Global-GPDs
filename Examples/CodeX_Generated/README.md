# GUMP observable tutorials

These examples form a progression from parton distributions to exclusive
cross sections:

1. [`01_parton_observables.py`](01_parton_observables.py) evaluates PDFs,
   t-dependent PDFs, GPDs, and generalized form factors (GFFs).
2. [`02_dvcs_observables.py`](02_dvcs_observables.py) evaluates Compton form
   factors (CFFs), DVCS cross sections, spin asymmetries, and the HERA
   photon-proton convention.
3. [`03_dvmp_observables.py`](03_dvmp_observables.py) evaluates transition
   form factors (TFFs) and rho/J/psi production cross sections.

Using a virtual environment is strongly recommended. It keeps GUMP's NumPy,
Numba, and llvmlite versions isolated from older system installations, which
is especially important on Apple Silicon.

Create the environment and install GUMP from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Then run any tutorial with the environment active:

```bash
python Examples/CodeX_Generated/01_parton_observables.py
python Examples/CodeX_Generated/02_dvcs_observables.py
python Examples/CodeX_Generated/03_dvmp_observables.py
```

The first evaluation initializes evolution kernels and numerical caches, so it
can be much slower than later evaluations. The examples intentionally contain
only a few points. Increase the DataFrame sizes only after the first run works.

## The common conventions

All high-level dataset interfaces accept a `pandas.DataFrame` and return that
table with a `pred f` column. If both measured `f` and uncertainty `delta f`
are supplied, GUMP also adds the pointwise contribution
`cost = ((pred f - f) / delta f)**2`.

Kinematic columns use the following conventions:

| column | meaning |
| --- | --- |
| `x` | average parton momentum fraction |
| `xi` | GPD skewness |
| `xB` | Bjorken x |
| `y` | lepton inelasticity |
| `t` | invariant momentum transfer squared in GeV^2; physical points normally have `t < 0` |
| `Q` | hard scale in GeV (that is, the positive square root of Q^2) |
| `phi` | azimuthal angle in radians |

`p_order=1` selects LO and `p_order=2` selects NLO. The DVCS wrappers spell
the same argument `P_order` with a capital `P`; the examples make this
difference explicit.

### GPD species (`spe`)

The fitted parameters are stacked in the order below. This integer is required
by the PDF, tPDF, GPD, and GFF DataFrame interfaces.

| `spe` | GPD | parity sector |
| ---: | --- | --- |
| 0 | H | vector |
| 1 | E | vector |
| 2 | Htilde | axial vector |
| 3 | Etilde | axial vector |

Use `Para_Comb_off_forward` with these interfaces. It contains all four
species in exactly this order.

### Flavor (`flv`)

The supported selectors are `u`, `d`, `g`, `NS`, and `S`. Here `NS = u - d`
is the non-singlet combination and `S = u + d` is the quark singlet
combination. The internal fit basis separately contains valence and sea
components; the public selectors return their physical combinations.

## Which interface should I use?

| desired quantity | function | required DataFrame columns |
| --- | --- | --- |
| PDF (the forward limit) | `PDF_theo` | `x, t, Q, spe, flv`; set `t=0` |
| t-dependent PDF | `tPDF_theo` | `x, t, Q, spe, flv` |
| GPD | `GPD_theo` | `x, xi, t, Q, spe, flv` |
| GFF/conformal moment | `GFF_theo` | `j, t, Q, spe, flv` |
| four DVCS CFFs | `CFF_theo` | scalar `xB, t, Q` arguments |
| BH, pure-DVCS, or interference term | `dsigma_BH`, `dsigma_DVCS`, `dsigma_INT` | scalar kinematics and (except BH) four CFFs |
| lepton-proton DVCS cross section | `DVCSxsec_theo` | `y, xB, t, Q, phi, pol` |
| DVCS spin asymmetry | `DVCSAsym_theo` | `y, xB, t, Q, phi, pol` |
| HERA photon-proton DVCS cross section | `DVCSxsecHERA_theo` | `y, xB, t, Q, pol` |
| two DVMP TFFs | `TFF_theo` | scalar `xB, t, Q` plus a meson code |
| DVMP longitudinal/transverse ratio | `R_fitted` | scalar `Q` and `meson=1` |
| DVMP differential cross section | `DVMPxsec_theo` | `y, xB, t, Q` |

For DVCS, `pol` is one of `UU`, `LU`, `UL`, `LL`, `UTin`, `LTin`, `UTout`,
or `LTout`. The first part describes the beam and the second the target;
`U`, `L`, and `T` mean unpolarized, longitudinal, and transverse. `in` and
`out` select the transverse target-spin direction. Not every analytic
contribution is nonzero in every channel.

For DVMP, the working meson codes are `1 = rho0` and `3 = J/psi`. Code `2`
is reserved for `phi`, but its charge factor and high-level cross-section
dispatch are not implemented yet. The rho0 wrapper returns longitudinal
`d sigma_L/dt`; the J/psi wrapper returns total `d sigma/dt`. These cross
sections are returned in nb/GeV^2.

## Parameters and performance

The examples use the packaged best-fit arrays:

- `Para_Comb_off_forward` for one selected GPD species;
- `Para_Unp_off_forward` for the H/E sector;
- `Para_Pol_off_forward` for the Htilde/Etilde sector.

The high-level functions parallelize over kinematic points. These tutorials
call `get_pool()` without an explicit process count, so the pool uses the
available CPUs. Use a clean virtual environment with current Numba/llvmlite
packages, particularly on Apple Silicon. Always place calls inside
`if __name__ == "__main__":`, especially on macOS and Windows, and call
`close_pool()` when a standalone program is done. Cross-section wrappers group
rows with identical `(xB, t, Q)` and compute their CFFs/TFFs only once, so put
different `phi`, `y`, or `pol` values for the same hard point in one DataFrame.

The wrappers add columns to the DataFrame passed to them. Use `.copy()` first
when the original table must remain unchanged.
