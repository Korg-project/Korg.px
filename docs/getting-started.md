# Getting started

This page walks through a solar spectrum from the ground up: abundances, a model atmosphere, a
linelist, a wavelength grid, and then the synthesis itself. It uses the one-shot
`synthesize(atm, linelist, wavelengths_angstrom, A_X)` function, which mirrors Korg.jl's
`synthesize(atm, linelist, A_X, wavelengths)`. If you are going to synthesize more than one
spectrum, read [The synthesizer closure](the-synthesizer-closure.md) instead — this path throws
away the expensive half of the work after every call.

## Importing

```python
import numpy as np
import korg
```

Importing `korg` enables JAX's 64-bit mode as a side effect. Do it before you configure JAX
yourself, or set `JAX_ENABLE_X64=true` in the environment.

A number of functions that Korg.jl exports are present in Korg.px but are not re-exported at the
top level. Import them from their modules:

```python
from korg.synthesis_plan import synthesize, prepare_synthesis
from korg.abundances import get_metals_H, get_alpha_H, A_X_to_absolute
from korg.utils import apply_LSF, compute_LSF_matrix, apply_rotation
from korg.synthesis import filter_linelist, blackbody
```

Note in particular that `korg.synthesize` is **not** the function used on this page. See
[Differences from Korg.jl](differences-from-korg-jl.md#two-functions-named-synthesize) for the
full story.

## Abundances

Abundances are specified as a 92-element vector of `A(X) = log10(n_X/n_H) + 12`, for hydrogen
through uranium, exactly as in Korg.jl. `format_A_X` builds one:

```python
A_X = korg.format_A_X()                                      # solar (Bergemann et al. 2025)
A_X = korg.format_A_X(-1.0)                                  # [metals/H] = -1
A_X = korg.format_A_X(-1.0, -0.6)                            # [metals/H] = -1, [alpha/H] = -0.6
A_X = korg.format_A_X(-0.5, abundances={"C": -0.25})         # [metals/H] = -0.5, [C/H] = -0.25
A_X = korg.format_A_X(abundances={"Ni": 1.0})                # solar except [Ni/H] = +1
```

The first positional argument sets the *default* abundance of everything heavier than helium; it
is not the resulting metallicity, because per-element overrides change the sum. `get_metals_H`
computes what the metallicity actually came out as:

```python
from korg.abundances import get_metals_H
get_metals_H(korg.format_A_X(-1.0))                             # -1.0
get_metals_H(korg.format_A_X(-1.0, abundances={"N": -0.5}))     # -0.849
```

This is the same caveat Korg.jl's [Abundances](https://ajwheeler.github.io/Korg.jl/stable/Abundances/)
page describes, with the same numbers. `get_metals_H`'s `ignore_alpha` keyword behaves as in
Korg.jl v1.2.1 — see
[Differences](differences-from-korg-jl.md#get_metals_h-follows-korgjls-code-not-its-docstring).

Korg.px's positional names are `default_metals_H` and `default_alpha_H` (Korg.jl's
`default_metals_H`/`default_alpha_H`), and the per-element dictionary is a keyword argument
called `abundances`, not a third positional.

## Model atmospheres

`korg.interpolate_marcs` interpolates the MARCS grid, which is downloaded (~380 MB) on first use
and cached in `~/.korg/` (override with `KORG_DATA_DIR`).

```python
atm = korg.interpolate_marcs(5777.0, 4.44, 0.0)          # Teff, log g, [M/H]
atm = korg.interpolate_marcs(5777.0, 4.44, A_X)          # or pass A_X and let Korg derive them
atm.n_layers                                              # 56
```

As in Korg.jl, passing an `A_X` vector lets Korg.px derive `[M/H]`, `[alpha/M]` and `[C/metals]`
for the grid. Passing them directly (`interpolate_marcs(Teff, logg, M_H, alpha_M, C_M)`) is also
supported but easier to get wrong.

The return type is a `PlanarAtmosphere` or a `ShellAtmosphere`; by default the choice is made
from `log g < 3.5`, and it can be forced with `spherical=True`/`False`. A `ShellAtmosphere` also
carries `R_photosphere` in cm.

To read a model from a file instead:

```python
atm = korg.read_model_atmosphere("path/to/model.mod")     # MARCS .mod or PHOENIX format
```

Both atmosphere types expose `layers`, `n_layers`, and the array properties `T`, `ne`, `n_total`,
`z`, `tau_ref` and `log_tau_ref`. **`log_tau_ref` is base-10** — `np.log10(tau_ref)` — which is
what the transfer kernels and `Synthesizer.from_atmosphere` both expect, so it passes straight
through. See [Differences](differences-from-korg-jl.md#log_tau_ref-is-base-10-everywhere).

## Linelists

Korg.px ships the same convenience linelists as Korg.jl:

```python
linelist = korg.get_VALD_solar_linelist()      # the Sun, 3000-9000 A (41861 lines)
linelist = korg.get_GALAH_DR3_linelist()
linelist = korg.get_APOGEE_DR17_linelist()
linelist = korg.get_GES_linelist()
```

and reads VALD, MOOG, Turbospectrum, Korg HDF5 and ExoMol files:

```python
linelist = korg.read_linelist("path/to/linelist.vald")              # defaults to "vald"
linelist = korg.read_linelist("path/to/lines.h5")                   # .h5 defaults to "korg"
linelist = korg.read_linelist("path/to/lines.moog", format="moog")
from korg.linelist import load_ExoMol_linelist
```

`format` is one of `"vald"`, `"moog"`, `"moog_air"`, `"turbospectrum"`,
`"turbospectrum_vac"`, `"korg"`. It is not sniffed from the file contents: the default is
`"korg"` for a `.h5` filename and `"vald"` otherwise. **Kurucz linelists are not supported**,
unlike Korg.jl.

Each entry is a `korg.Line`; you rarely need to look inside one.

You can hand a full linelist straight to the synthesis functions. Both trim it to the synthesis
range first, as Korg.jl does: `line_buffer` is 10 Å by default, and lines further than that outside
the grid are discarded before anything else happens. For 5000–5005 Å that is 166 lines out of the
VALD solar list's 41861.

Pass `line_buffer=None` (or `line_buffer_cm=None` to `prepare_synthesis`) to keep every line.
`korg.synthesis.filter_linelist` is still public if you want to trim explicitly.

`korg.prune_linelist` and `korg.merge_close_lines` are also available, as in Korg.jl.

## Wavelengths

Korg.px does not have a single wavelength convention, so be deliberate:

| Function | Wavelength argument |
|---|---|
| `synthesis_plan.synthesize` | 1-D array, **Å** |
| `synthesis_plan.prepare_synthesis` | 1-D array, **cm** |
| `korg.utils.apply_LSF`, `apply_rotation`, `compute_LSF_matrix` | anything `Wavelengths` accepts — `(start, stop)`, `(start, stop, step)`, a list of those, or an array |
| `korg.synthesis.filter_linelist` | array in **cm**, buffer in **cm** |

Korg.jl's tuple form is therefore available in the post-processing functions but not in the
synthesis functions, which want an explicit array. Build it with `np.arange`:

```python
wavelengths_angstrom = np.arange(5000.0, 5005.0, 0.01)
wavelengths_cm = wavelengths_angstrom * 1e-8
```

Wavelengths are *in vacuo*, as in Korg.jl. Convert with `korg.air_to_vacuum` and
`korg.vacuum_to_air`.

## Synthesizing a solar spectrum

```python
import numpy as np, korg
from korg.synthesis_plan import synthesize

wavelengths = np.arange(5000.0, 5005.0, 0.01)                 # Angstroms
A_X = korg.format_A_X()
atm = korg.interpolate_marcs(5777.0, 4.44, A_X)
lines = korg.get_VALD_solar_linelist()

flux, continuum = synthesize(atm, lines, wavelengths, A_X, vmic=1.0)
```

`vmic` here is in **km/s**, matching Korg.jl. (The closure's `vmic_cm_s` is in cm/s — this is a
real inconsistency, documented in
[Differences](differences-from-korg-jl.md#vmic-is-kms-in-synthesize-and-cms-in-the-closure).)

The return value is a plain `(flux, continuum)` tuple of JAX arrays, not a `SynthesisResult`.
Pass `return_cntm=False` to get the flux alone. The flux is *not* continuum-normalized; divide:

```python
rectified = flux / continuum
```

Plotting:

```python
import matplotlib.pyplot as plt
plt.plot(wavelengths, np.asarray(flux / continuum), "k-")
plt.xlabel(r"$\lambda$ [Å]")
plt.ylabel("rectified flux")
```

The geometry follows the atmosphere object you passed: a `ShellAtmosphere` gets spherical
transfer, a `PlanarAtmosphere` gets plane-parallel. Override with `geometry="spherical"` or
`geometry="plane-parallel"`.

`synthesize` accepts the `prepare_synthesis` keyword arguments (`cntm_step_cm`, `window_safety`,
`reference`, `n_mu`) and forwards them. It does **not** accept Korg.jl's `line_buffer`,
`hydrogen_lines`, `hydrogen_line_window_size`, `line_cutoff_threshold`, `mu_values`,
`use_MHD_for_hydrogen_lines` or `molecular_cross_sections`; see
[Differences](differences-from-korg-jl.md#keyword-arguments-that-do-not-exist).

### Warnings you can ignore

Building a plan emits

```
line_absorption.py:744: RuntimeWarning: divide by zero encountered in divide
  (np.ceil(2.0 * max_wins / wl_spacing) + 2).astype(int), 0, n_wl
```

three times. It comes from evaluating the line opacity on the single-point grid at the 5000 Å
reference wavelength, where the pixel spacing is zero by construction. The resulting
`inf`/`nan` window widths are clipped immediately afterwards. It is noise, not a problem with
your inputs.

### First call is slow

JAX compiles the synthesis kernel the first time it runs, and the compile is much more expensive
than the run. Expect the first synthesis in a process to take on the order of a minute (more on
CPU, less on GPU) and subsequent ones on the same wavelength grid and linelist to be far faster.
`synthesize` rebuilds its plan every time it is called, which wastes most of that; that is the
reason to use `prepare_synthesis` for anything beyond a single spectrum.

## Post-processing

The LSF and rotation functions are ports of Korg.jl's and take Korg.jl's wavelength formats:

```python
from korg.utils import apply_LSF, compute_LSF_matrix, apply_rotation

low_res = apply_LSF(flux, wavelengths, R=10_000)
broadened = apply_rotation(flux, wavelengths, vsini=7.0, epsilon=0.6)

# faster if you are applying the same LSF to many spectra
obs_wavelengths = np.arange(5000.0, 5005.0, 0.05)
LSF = compute_LSF_matrix(wavelengths, obs_wavelengths, R=10_000)
resampled = LSF @ np.asarray(flux)
```

`R` may be a callable of wavelength in Å for a varying resolving power. Both `apply_LSF` and
`apply_rotation` are differentiable with respect to the flux, and with respect to `R` and `vsini`
respectively.

## Fitting

`korg.fit_spectrum` is the analogue of `Korg.Fit.fit_spectrum`:

```python
result = korg.fit_spectrum(obs_wls, obs_flux, obs_err, linelist,
                           initial_guesses={"Teff": 5700.0, "logg": 4.4},
                           fixed_params={"M_H": 0.0},
                           R=20_000)
result["best_fit_params"]
```

The recognised parameter names are `Teff`, `logg`, `M_H`, `alpha_H`, `vmic`, `vsini`, `epsilon`,
`cntm_offset`, `cntm_slope` and any atomic symbol (as `[X/H]`). Note the capital `M_H` here, and
the lower-case `m_H` on the synthesizer closure — see
[Differences](differences-from-korg-jl.md#metallicity-is-spelled-m_h-in-one-place-and-m_h-in-another).

It returns a `dict` (keys `best_fit_params`, `best_fit_flux`, `obs_wl_mask`, `solver_result`,
`trace`, `covariance`), not a named struct. Equivalent-width workflows are
`korg.calculate_EWs`, `korg.ews_to_abundances`, `korg.ews_to_stellar_parameters` and
`korg.ews_to_stellar_parameters_direct`.

## Next

Everything above rebuilds the synthesis plan on every call. To keep it and reuse it — and to get
`jit`, `vmap` and `grad` — see [The synthesizer closure](the-synthesizer-closure.md).
