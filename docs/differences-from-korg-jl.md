# Differences from Korg.jl

Korg.px targets Korg.jl v1.2.1. The physics and data are ports; the interface is not a
translation. This page lists every place the Python interface departs from the Julia one, and
every place the numbers can be expected to differ.

Read the first section before you use the package.

---

## Known defects

### `log_tau_ref` is base-10 everywhere

Both radiative-transfer kernels — `radiative_transfer.core._compute_tau_anchored_planar` and
`radiative_transfer.spherical.ray_tau_anchored` — take `log_tau_ref` as a **base-10** logarithm.
They recover `tau_ref = 10 ** log_tau_ref` and convert the step to natural units with a factor of
`ln 10`. `Atmosphere.log_tau_ref` is `np.log10(tau_ref)`, so for an atmosphere object you pass
`atm.log_tau_ref` straight through.

This matters for `Synthesizer.from_atmosphere`, the one entry point that takes the quantity from
you rather than deriving it. Passing a natural logarithm integrates the transfer against
`tau_ref ** ln(10)`, which makes lines roughly 2.2x too shallow — a silent, plausible-looking
error rather than a crash.

!!! note "Fixed"
    The traced path itself built a natural logarithm until it was corrected in
    `traced_synthesis._interpolate_marcs_traced` and `synthesis_plan.synthesize`. On the Sun over
    5000–5002 Å the deepest rectified point was 0.352 against the validated path's 0.161; it is
    now 0.161, agreeing to 2.4e-4.

    Nothing in the suite caught it, which was the deeper problem: the closure tests compared the
    traced path against *itself*, and the Korg.jl-reference tests all went through
    `synthesize_jit` with `np.log10`. `TestAgreementWithTheValidatedPath` in
    `tests/test_synthesizer_closure.py` now compares absolute flux across the two paths, and
    `test_traced_log_tau_ref_is_base_10` pins the unit directly.

### `max_needed_px` is computed but not returned

`traced_lines.line_alpha_traced` returns, alongside the opacity, the number of pixels each line
bucket actually needed — the quantity you would check to detect that a line profile was truncated
by its bucket's fixed capacity. `traced_synthesis._synthesize_traced` binds it and then discards
it; `Synthesizer` returns only `(flux, continuum)`.

The `synthesis_plan` module docstring says "the synthesizer returns `max_needed_px` so that can be
asserted outside `jit` rather than being silent". It does not. Window truncation is currently
silent; see [Window truncation](the-synthesizer-closure.md#window-truncation) for how to avoid it
by construction.

---

## Interface

### There is no `synth`

Korg.jl's one-stop `synth(; Teff, logg, wavelengths, M_H, ...)` has no Python equivalent. The
closest thing is `prepare_synthesis(...)` followed by calling the result, which takes `Teff`,
`logg`, `m_H`, `alpha_m` and `C_m` as positional arguments. Post-processing that `synth` folds in
(`R`, `vsini`) has to be applied separately with `korg.utils.apply_LSF` and
`korg.utils.apply_rotation`.

### Two functions named `synthesize`

There are two, and the one exported at the top level is not the one you want:

| | `korg.synthesize` | `korg.synthesis_plan.synthesize` |
|---|---|---|
| defined in | `korg/synthesis.py` | `korg/synthesis_plan.py` |
| signature | `(atmosphere, linelist, wavelengths_angstrom, abundances, ...)` | `(atmosphere, linelist, wavelengths_angstrom, A_X, ...)` |
| fourth argument | **absolute number fractions** `n_X/n_total` | **`A(X)`**, as in Korg.jl |
| returns | `SynthesisResult` | `(flux, continuum)` tuple |
| traceable | no | yes |
| status | wraps the deprecated `synthesize_spectrum` and raises `DeprecationWarning` | current |

`korg.synthesize` delegates to `korg.synthesis.synthesize_spectrum`, which is deprecated: it
orchestrates jitted kernels from Python and drops to host NumPy in places, so it cannot be placed
on a GPU as one kernel, `vmap`ped, or differentiated. It is retained because it still covers
options the traced path does not, and because it is the implementation that the Korg.jl reference
fixtures validate.

Note the argument-order difference from Korg.jl as well: Korg.jl is
`synthesize(atm, linelist, A_X, wavelengths)`; both Python versions put wavelengths before
abundances.

### Most public functions are not exported at the top level

Korg.jl exports `synth`, `synthesize`, `interpolate_marcs` and `format_A_X`, and everything else
is reachable as `Korg.something`. In Korg.px, `import korg` gives you a curated subset, and a
number of functions that Korg.jl documents as public are only reachable from their modules:

| Function | Korg.px location |
|---|---|
| `prepare_synthesis`, `Synthesizer`, `synthesize` (traced) | `korg.synthesis_plan` |
| `apply_LSF`, `compute_LSF_matrix`, `apply_rotation` | `korg.utils` |
| `get_metals_H`, `get_alpha_H`, `A_X_to_absolute` | `korg.abundances` |
| `blackbody`, `filter_linelist`, `synthesize_jit`, `SynthesisResult` | `korg.synthesis` |
| `Wavelengths` | `korg.wavelengths` |
| `load_ExoMol_linelist` | `korg.linelist` |
| `default_solar_abundances` (as `DEFAULT_SOLAR_ABUNDANCES`) | `korg.abundances` |

`korg.__all__` is not a reliable guide to what is public.

### Wavelengths

Korg.jl v1.2.1 takes wavelengths as `(start, stop)`, `(start, stop, step)`, or a vector of such
tuples, everywhere, and turns them into a `Korg.Wavelengths`. Korg.px is inconsistent:

- `synthesis_plan.synthesize` takes a 1-D array in **Å**.
- `synthesis_plan.prepare_synthesis` takes a 1-D array in **cm**.
- `korg.utils.apply_LSF`, `apply_rotation` and `compute_LSF_matrix` accept anything
  `korg.wavelengths.Wavelengths` accepts, including Korg.jl's tuples.
- `korg.synthesis.filter_linelist` takes cm, with the buffer in cm.

**Multiple disjoint wavelength windows are not supported by the synthesis path.** The traced
kernel takes a single monotonic grid and uses its median spacing to size line windows.
`Wavelengths` still models multi-window spectra, and the post-processing functions honour them,
but `prepare_synthesis` does not.

`korg.wavelengths.Wavelengths` also retains the "values ≥ 1 are Å, values < 1 are cm" heuristic
that Korg.jl v1.2.1 explicitly warns against relying on, and it still supports
`air_wavelengths=True`, which Korg.jl removed from `synthesize`.

### Return values

Korg.jl's `synthesize` returns a `SynthesisResult` carrying `flux`, `cntm`, `intensity`, `alpha`,
`mu_grid`, `number_densities`, `electron_number_density`, `wavelengths` and `subspectra`. The
traced path returns a bare `(flux, continuum)` tuple. The absorption coefficient, the per-species
number densities, the electron number density and the μ grid are computed but not exposed.

`korg.synthesis.SynthesisResult` (from the deprecated path) does carry `alpha`, `alpha_cntm`,
`number_densities` and `electron_number_density`, and has a `cntm` property aliasing `continuum`.

`korg.fit_spectrum` returns a `dict`, not a struct.

### `vmic` is km/s in `synthesize` and cm/s in the closure

`synthesis_plan.synthesize(..., vmic=1.0)` is in km/s, Korg.jl's convention, and is multiplied by
1e5 internally. `Synthesizer.__call__` and `Synthesizer.from_atmosphere` take `vmic_cm_s`, in
cm/s, defaulting to `1e5` — the same 1 km/s, spelled differently. The name carries the unit, but
the discrepancy is easy to trip over when moving between the two.

Korg.jl accepts either a scalar or a per-layer vector for `vmic`. Korg.px's traced path takes a
scalar.

### Metallicity is spelled `m_H` in one place and `M_H` in another

Korg.jl v1.0 renamed its `m_H` keyword arguments to `M_H` throughout. Korg.px is inconsistent:

- `Synthesizer.__call__` takes `m_H`, `alpha_m`, `C_m` (lower case, and `alpha`/`C` relative to
  metals, matching the MARCS grid axes).
- `korg.interpolate_marcs` takes `M_H_or_A_X`, `alpha_M`, `C_M`.
- `korg.fit_spectrum` recognises `M_H` and `alpha_H` and rejects `m_H` with
  `ValueError: Unknown parameter`.
- `korg.format_A_X`'s positional arguments are `default_metals_H` and `default_alpha_H`.

### Keyword arguments that do not exist

Korg.jl's `synthesize` keyword arguments and their status in Korg.px's traced path:

| Korg.jl | Korg.px traced path |
|---|---|
| `vmic` | `vmic` (km/s) on `synthesize`, `vmic_cm_s` on the closure; scalar only |
| `line_buffer` | `line_buffer` (Å) on `synthesize`, `line_buffer_cm` (cm) on `prepare_synthesis`; same 10 Å default. `None` disables it |
| `cntm_step` | `cntm_step_cm` on `prepare_synthesis` (cm, default 1e-8 = 1 Å) |
| `hydrogen_lines` | **not implemented** — hydrogen lines are always on |
| `use_MHD_for_hydrogen_lines` | **not implemented** — MHD occupation probabilities are always used. Korg.jl defaults this to *off* above 13 000 Å, so infrared syntheses will differ |
| `hydrogen_line_window_size` | **not implemented** — fixed at 150 Å |
| `mu_values` | `n_mu` on `prepare_synthesis` (integer only; an explicit μ vector is not accepted) |
| `line_cutoff_threshold` | **not implemented** — fixed at 3e-4 |
| `electron_number_density_warn_threshold` / `_min_value` | **not implemented** |
| `return_cntm` | `return_cntm` on `synthesize`; the closure always returns both |
| `use_internal_reference_linelist` | applied unconditionally when the deprecated path builds its reference linelist; not part of the traced path |
| `I_scheme`, `tau_scheme` | **not implemented** — `linear_flux_only` and `anchored`, as in Korg.jl's defaults |
| `ionization_energies`, `partition_funcs`, `log_equilibrium_constants` | **not implemented** on the traced path; the deprecated `synthesize_spectrum` accepts them |
| `molecular_cross_sections` | **not implemented** on the traced path. `korg.MolecularCrossSection` exists and works with the deprecated path |
| `use_chemical_equilibrium_from` | **not implemented** |

The traced path also adds `geometry`, `window_safety`, `reference`, `n_layers` and `data`, which
have no Korg.jl equivalent.

### Brackett hydrogen lines are not in the traced path

Korg.jl models the hydrogen Brackett series. Korg.px's traced path does not: the implementation
walks atmospheric layers in Python and has not been rewritten to trace. `prepare_synthesis`
exposes `Synthesizer.brackett_in_range`, `True` when any Brackett transition falls within 150 Å
of the synthesis window, so callers can refuse rather than silently omit them. This matters in
the infrared.

### Linelist formats

`korg.read_linelist` supports `"vald"`, `"moog"`, `"moog_air"`, `"turbospectrum"`,
`"turbospectrum_vac"` and `"korg"`. Korg.jl additionally supports `"kurucz"` and `"kurucz_vac"`;
**Korg.px does not**. The isotopic-abundance keyword is `iso_abundances`, not Korg.jl's
`isotopic_abundances`.

### `Korg.species` string macro

Korg.jl's `Korg.species"Mg I"` non-standard string literal has no Python equivalent. Use
`korg.Species("Mg I")`.

---

## Numerics

### The traced path computes the 5000 Å reference opacity differently

The anchored optical-depth scheme needs the opacity at the MARCS reference wavelength, 5000 Å.
Korg.jl computes it from the continuum *plus* an internal reference linelist, and Korg.px's
NumPy path does the same (`korg.synthesis.get_reference_wavelength_linelist`, used by
`precompute_atmosphere`).

The traced path does not use the internal reference linelist at all. It evaluates the continuum
at 5000 Å, and then, only if the synthesis grid happens to cover 5000 Å, replaces that with
continuum-plus-*your*-lines at the nearest pixel (`Synthesizer.ref_pixel`). So the reference
opacity depends on whether 5000 Å is inside your window and on which lines you passed. This is
part of the residual 2e-3 difference between the two implementations noted above.

### Precision floors

The bucketed Voigt kernel evaluates its profile in float32 — the wavelength offsets are of order
1e-8 cm, which float32 holds safely, and it halves the memory traffic of the dominant kernel.
The consequence is that XLA is free to fuse and reassociate that arithmetic differently in
different contexts, so results that ought to be identical are not:

- `jax.jit(synth)` versus eager `synth`: the test suite asserts `rtol=1e-7` and measures 1.2e-8
  on its reference configuration; ~5e-8 was measured on a CPU backend here.
- `Synthesizer.from_atmosphere` versus `Synthesizer.__call__` on the same atmosphere: 1.3e-8.
- `geometry=None` versus the explicitly-named branch: 4.8e-8, because `lax.cond` changes the
  fusion.

Do not assert bitwise equality between these. Korg.jl's [FAQ](https://ajwheeler.github.io/Korg.jl/stable/FAQ/)
promises bitwise reproducibility for single-threaded runs on the same machine with identical
inputs; Korg.px does not, and cannot while the line kernel runs in float32.

### Gradients versus finite differences

`jax.grad` through the stellar parameters is exact for the model being evaluated, but the model
is only piecewise smooth in them: `interpolate_marcs` is **multilinear**, so a central difference
does not converge under step refinement. Measured relative differences between AD and central
differences wander between about 8e-4 and 6e-3 as the step in `Teff` goes from 4 K to 0.5 K,
rather than shrinking. The test suite asserts sign agreement and 5% magnitude agreement, which is
what a difference quotient over a piecewise-linear interpolant supports. Finite differences are
not a reliable reference for these derivatives.

### Line windows are not differentiable, and recomputing them does not make them so

The traced kernel recomputes each line's window half-width from the traced amplitudes and
broadening parameters rather than freezing it at plan time. That is a *correctness* property: the
window tracks the parameters under `vmap` and away from the plan's reference point.

It is not a gradient property. The window reaches the output only through a hard mask
(`|delta| <= win`) and an integer `jnp.searchsorted`. Both have zero derivative, so the window's
contribution to `d(flux)/d(anything)` is structurally zero — exactly as in any hard-cutoff
scheme, Korg.jl's included.

### `geometry=None` is not differentiable across log g = 3.5

`prepare_synthesis(geometry=None)` selects the transfer scheme inside the traced region with
`jax.lax.cond` on `log g < 3.5`, matching Korg.jl's dispatch. The derivative you get is the
selected branch's; the switch contributes nothing and cannot, because log g = 3.5 is a
discontinuous change of model rather than a smooth transition. Under `vmap`, `lax.cond` becomes a
`select` and **both** branches are evaluated, which roughly doubles the cost.

Name the geometry explicitly when you know it.

### `get_metals_H` follows Korg.jl's code, not its docstring

Korg.jl v1.2.1's `get_metals_H` docstring says that `ignore_alpha=true` uses "all elements heavier
than He" and that `ignore_alpha=false` ignores "both carbon and the alpha elements". The
implementation does neither: `ignore_alpha=true` excludes the alpha elements and keeps everything
else including carbon, and `ignore_alpha=false` uses every element with Z ≥ 3.

Korg.px ports the implementation, so it agrees with Korg.jl numerically and disagrees with
Korg.jl's prose in the same way. Carbon is included in `[metals/H]` under both settings.

(`interpolate_marcs` is a separate case: when handed an `A_X` vector it derives the grid's `[M/H]`
with carbon *and* the alpha elements excluded, matching Korg.jl's atmosphere-grid convention.)

### `voigt_hjerting` has up to ~15% error near α = 1.4

Inherited from Korg.jl, not a porting artefact: Korg.jl v1.2.1 produces the same numbers to every
printed digit. The Hunger (1965) piecewise fit departs from the exact Voigt function by up to
15.5% approaching the α = 1.4 branch boundary from below, and the error is discontinuous across
the regime seams, so the profile value and its derivative both jump there. Full analysis and
reproducers in [`voigt-hjerting-accuracy.md`](voigt-hjerting-accuracy.md).

---

## Environment

### 64-bit mode

`import korg` sets `JAX_ENABLE_X64` and calls `jax.config.update("jax_enable_x64", True)` as an
import side effect. If JAX has already been configured in the process, set `JAX_ENABLE_X64=true`
in the environment before Python starts.

### Backend

Korg.px makes no backend assumption. JAX will use a GPU if one is visible and a CUDA jaxlib is
installed; set `JAX_PLATFORMS=cpu` to force CPU. Compilation dominates the first synthesis in a
process on either.

### Data

Korg.jl ships its MARCS grid with the package. Korg.px downloads it (~380 MB) on first use of
`interpolate_marcs` and caches it in `~/.korg/`, or in `$KORG_DATA_DIR`.

### Solar abundances

The default is Bergemann et al. 2025, the same as Korg.jl v1.2.1. Asplund 2009, Asplund 2020 and
Grevesse 2007 are also available in `korg.abundances`.
