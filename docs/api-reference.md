# API reference

The public interface of Korg.px. Signatures are given as they appear in the source; keyword
arguments beyond the ones shown are documented in the docstrings.

Unlike Korg.jl, most of these are **not** re-exported at the top level. The "import from" column
is the module to import from; where it says `korg`, `import korg` is enough. See
[Differences](differences-from-korg-jl.md#most-public-functions-are-not-exported-at-the-top-level).

## Synthesis

| Import from | Function |
|---|---|
| `korg.synthesis_plan` | `prepare_synthesis`, `Synthesizer`, `synthesize` |
| `korg.synthesis` | `synthesize_jit`, `filter_linelist`, `blackbody`, `SynthesisResult`, `load_synthesis_data` |

### `prepare_synthesis`

```python
korg.synthesis_plan.prepare_synthesis(
    wavelengths_cm, linelist, data=None, *,
    geometry=None, cntm_step_cm=1e-8, window_safety=2.0,
    reference=(5777.0, 4.44, 0.0), n_layers=56, n_mu=20) -> Synthesizer
```

Fixes every array shape a synthesis needs — the hydrogen transitions in range, the coarse
continuum grid, the line-window buckets, the layer count — and returns a callable that takes
stellar parameters. `wavelengths_cm` is in centimetres. See
[The synthesizer closure](the-synthesizer-closure.md).

### `Synthesizer.__call__`

```python
synth(Teff, logg, m_H=0.0, alpha_m=0.0, C_m=0.0,
      abundances=None, vmic_cm_s=1e5) -> (flux, continuum)
```

Synthesize from stellar parameters, interpolating the MARCS grid inside the traced region.
Traced and differentiable in every argument. `abundances`, if given, is 92 number fractions
(`n_X/n_total`), not `A(X)`; if omitted it is derived from `(m_H, alpha_m, C_m)`.

### `Synthesizer.from_atmosphere`

```python
synth.from_atmosphere(T, n_total, ne, z, log_tau_ref, abundances,
                      vmic_cm_s=1e5, R_photosphere=None, logg=None) -> (flux, continuum)
```

Synthesize from atmosphere arrays, for models that do not come from the MARCS grid. `logg` is
required when the plan was built with `geometry=None`.

`log_tau_ref` is base-10, matching `atmosphere.log_tau_ref`; see
[Differences](differences-from-korg-jl.md#log_tau_ref-is-base-10-everywhere).

### `Synthesizer` attributes

| Attribute | Meaning |
|---|---|
| `n_wl`, `n_layers`, `n_lines` | fixed shapes |
| `bucket_line_idx`, `bucket_widths` | the line-window partition and each bucket's pixel capacity |
| `stark_keys` | hydrogen Stark transitions in range |
| `brackett_in_range` | `True` if Brackett lines fall in the window (they are **not** modelled) |
| `ref_pixel` | index of 5000 Å in the grid, or `None` |
| `geometry` | `'plane-parallel'`, `'spherical'`, or `None` |

### `synthesize` (traced, Korg.jl-compatible)

```python
korg.synthesis_plan.synthesize(atmosphere, linelist, wavelengths_angstrom, A_X, *,
                               vmic=1.0, geometry=None, return_cntm=True,
                               **plan_kwargs) -> (flux, continuum)
```

One-shot synthesis: builds a `Synthesizer` and calls it once. `wavelengths_angstrom` is in Å;
`A_X` is the 92-element `A(X)` vector; `vmic` is in km/s. `geometry=None` follows the atmosphere
object's type. Prefer `prepare_synthesis` for more than one spectrum.

### `synthesize` (deprecated, top-level)

```python
korg.synthesize(atmosphere, linelist, wavelengths_angstrom, abundances,
                vmic=1.0, line_buffer=10.0, hydrogen_lines=True,
                hydrogen_line_window_size=150.0, line_cutoff_threshold=3e-4,
                return_cntm=True, mu_values=20, verbose=True, profile=False)
    -> SynthesisResult
```

The NumPy path. Takes **absolute number fractions**, not `A(X)`. Wraps
`korg.synthesis.synthesize_spectrum`, which raises `DeprecationWarning` and cannot be traced,
`vmap`ped or differentiated. It supports Korg.jl options the traced path does not, and it is the
implementation the Korg.jl reference fixtures validate.

### `synthesize_jit`

```python
korg.synthesis.synthesize_jit(wavelengths_cm, T_layers, n_total_layers, ne_layers,
                              z_layers, log_tau_ref, abundances, vmic_cm_s,
                              data, linelist_data, precomputed_atm=None)
    -> (flux, continuum)
```

The jit-compatible kernel underneath the deprecated path. `log_tau_ref` is base-10 here.
Requires you to build `data` and `linelist_data` yourself; `prepare_synthesis` is the supported
way to do that.

### `filter_linelist`

```python
korg.synthesis.filter_linelist(linelist, wavelengths_cm, line_buffer_cm, warn_empty=True) -> list
```

Korg.jl's `line_buffer` as an explicit step. The synthesis path does not apply it for you.

### `blackbody`

```python
korg.synthesis.blackbody(T, wavelength_cm)
```

## Abundances

| Import from | Function |
|---|---|
| `korg` | `format_A_X`, `get_solar_abundances` |
| `korg.abundances` | `get_metals_H`, `get_alpha_H`, `A_X_to_absolute`, `DEFAULT_SOLAR_ABUNDANCES` |

```python
korg.format_A_X(default_metals_H=0.0, default_alpha_H=None, abundances=None,
                solar_relative=True, solar_abundances=None, alpha_elements=None) -> (92,) array
korg.abundances.get_metals_H(A_X, solar_abundances=None, ignore_alpha=True,
                             alpha_elements=None) -> float
korg.abundances.get_alpha_H(A_X, solar_abundances=None, alpha_elements=None) -> float
korg.abundances.A_X_to_absolute(A_X) -> (92,) array      # A(X) -> n_X/n_total
korg.get_solar_abundances(source="bergemann_2025") -> (92,) array
```

`get_solar_abundances` also accepts `"asplund_2009"`, `"asplund_2020"` and `"grevesse_2007"`.
`A_X_to_absolute` is the conversion the closure's `abundances` argument needs.

## Model atmospheres

| Import from | Function |
|---|---|
| `korg` | `interpolate_marcs`, `read_model_atmosphere` |
| `korg.atmosphere` | `PlanarAtmosphere`, `ShellAtmosphere`, `create_simple_atmosphere` |

```python
korg.interpolate_marcs(Teff, logg, M_H_or_A_X=0.0, alpha_M=0.0, C_M=0.0,
                       spherical=None, perturb_at_grid_values=True)
    -> PlanarAtmosphere | ShellAtmosphere
korg.read_model_atmosphere(fname, format=None) -> PlanarAtmosphere | ShellAtmosphere
```

`spherical=None` chooses from `log g < 3.5`. The grid covers Teff 2800–8000 K, log g −0.5 to 5.5,
[M/H] −2.5 to 1.0, [α/M] −1.0 to 1.0, [C/metals] −1.5 to 1.0; it is downloaded on first use.

Atmosphere objects expose `layers`, `n_layers`, and array properties `T`, `ne`, `n_total`, `z`,
`tau_ref` and `log_tau_ref` (base-10). `ShellAtmosphere` adds `R_photosphere`, `r` and
`photosphere_correction`, and the two convert with `PlanarAtmosphere.from_shell` /
`ShellAtmosphere.from_planar`.

## Linelists

All from `korg` unless noted.

```python
korg.read_linelist(filename, format=None, iso_abundances=None) -> list[Line]
    # format in {"vald", "moog", "moog_air", "turbospectrum", "turbospectrum_vac", "korg"}
    # default: "korg" if the filename ends in .h5, else "vald".  No Kurucz support.
korg.read_vald_linelist(filename) -> list[Line]
korg.read_korg_linelist(filename) -> list[Line]
korg.parse_moog_linelist(...), korg.parse_turbospectrum_linelist(...)
korg.linelist.load_ExoMol_linelist(spec, states_file, transitions_file, ...) -> list[Line]
korg.save_linelist(path, linelist) -> None

korg.get_VALD_solar_linelist()      # the Sun, 3000-9000 A, 41861 lines
korg.get_GALAH_DR3_linelist()
korg.get_APOGEE_DR17_linelist()
korg.get_GES_linelist()

korg.create_line(wl, log_gf, species, E_lower, gamma_rad=None, gamma_stark=None,
                 vdW=None, ionization_energies_dict=None) -> Line
korg.approximate_line_strength(line, T) -> float
korg.Line, korg.Species, korg.Formula
korg.isotopic_abundances
```

### Pruning and merging

```python
korg.prune_linelist(atmosphere, linelist, A_X, wavelengths, threshold=0.1,
                    sort_by_EW=True, max_distance=0.0, **synthesis_kwargs) -> list[Line]
korg.merge_close_lines(linelist, merge_distance=0.2) -> list[Line]
```

`prune_linelist` runs on the deprecated NumPy synthesis path.

### Molecular cross-sections

```python
korg.MolecularCrossSection(linelist, wavelengths, ...)
korg.interpolate_molecular_cross_sections(alpha, molecular_cross_sections,
                                          wavelengths_angstrom, temperatures, vmic,
                                          number_densities)
korg.save_molecular_cross_section(path, xs)
korg.read_molecular_cross_section(path)
```

Usable with `korg.synthesis.synthesize_spectrum` only; the traced path ignores them.

## Wavelengths and post-processing

| Import from | Function |
|---|---|
| `korg` | `air_to_vacuum`, `vacuum_to_air` |
| `korg.utils` | `apply_LSF`, `compute_LSF_matrix`, `apply_rotation` |
| `korg.wavelengths` | `Wavelengths` |

```python
korg.air_to_vacuum(wavelength, cgs=None)
korg.vacuum_to_air(wavelength, cgs=None)

korg.utils.apply_LSF(flux, wls, R, window_size=4) -> array
korg.utils.compute_LSF_matrix(synth_wls, obs_wls, R, window_size=4, verbose=True) -> array
korg.utils.apply_rotation(flux, wls, vsini, epsilon=0.6) -> array

korg.wavelengths.Wavelengths(wl_spec, air_wavelengths=False,
                             wavelength_conversion_warn_threshold=1e-4)
```

`R` may be a float or a callable of wavelength in Å. `wls` accepts Korg.jl's `(start, stop)` and
`(start, stop, step)` tuples, lists of them, or an explicit array. All three post-processing
functions are differentiable.

## Fitting

All from `korg`.

```python
korg.fit_spectrum(obs_wls, obs_flux, obs_err, linelist, initial_guesses,
                  fixed_params=None, *, windows=None, R=None, LSF_matrix=None,
                  synthesis_wls=None, wl_buffer=1.0, precision=1e-4,
                  postprocess=None, time_limit=10_000, adjust_continuum=False,
                  **synthesis_kwargs) -> dict
korg.validate_params(initial_guesses, fixed_params=None) -> (dict, dict)

korg.calculate_EWs(atm, linelist, A_X, ew_window_size=2.0, wl_step=0.01, ...)
korg.ews_to_abundances(atm, linelist, A_X, measured_EWs, ew_window_size=2.0, wl_step=0.01, ...)
korg.ews_to_abundances_approx(atm, linelist, A_X, measured_EWs, ...)
korg.ews_to_stellar_parameters(linelist, measured_EWs, abundance_adjustments=None,
                               Teff0=5000.0, logg0=3.5, vmic0=1.0, M_H0=0.0, ...)
korg.ews_to_stellar_parameters_direct(linelist, measured_EWs, measured_EW_err=None,
                                      Teff0=5000.0, logg0=3.5, vmic0=1.0, M_H0=0.0, ...)
```

`fit_spectrum` returns a `dict` with keys `best_fit_params`, `best_fit_flux`, `obs_wl_mask`,
`solver_result`, `trace` and `covariance`. Fitting runs on the deprecated NumPy synthesis path
except where every free parameter is a post-processing parameter, in which case it uses exact
`jax.grad` gradients.

## Radial-velocity precision

```python
korg.Qfactor(synth_flux, synth_wl, obs_wl, LSF_mat, obs_mask=None) -> float
korg.RV_prec_from_Q(Q, RMS_SNR, Npix) -> float
korg.RV_prec_from_noise(synth_flux, synth_wl, obs_wl, LSF_mat, obs_err, obs_mask=None) -> float
```

## Data management

```python
korg.get_korg_data_dir() -> str
korg.get_artifact_path(name) -> str
korg.download_artifact(name) -> str
korg.list_artifacts() -> list
korg.synthesis.load_synthesis_data() -> SynthesisData
korg.synthesis.save_synthesis_data(data, path)
```

`load_synthesis_data()` reads the partition functions, equilibrium constants and continuum tables
once; pass the result to `prepare_synthesis(data=...)` when building many plans.

## Constants

`korg.constants`, also re-exported at the top level: `c_cgs`, `hplanck_cgs`, `hplanck_eV`,
`kboltz_cgs`, `kboltz_eV`, `electron_mass_cgs`, `electron_charge_cgs`, `amu_cgs`, `Rydberg_eV`,
`MAX_ATOMIC_NUMBER`.

## Lower-level modules

Not part of the stable interface, but occasionally useful:

| Module | Contents |
|---|---|
| `korg.continuum` | total continuum absorption |
| `korg.continuum_absorption` | H⁻, He, metal bound-free, hydrogenic bf/ff, scattering |
| `korg.line_profiles` | `voigt_hjerting` and friends |
| `korg.line_absorption` | the NumPy line-opacity path |
| `korg.traced_lines`, `korg.traced_synthesis` | the traced kernels the closure composes |
| `korg.radiative_transfer` | planar and spherical transfer, μ grids, exponential integrals |
| `korg.statmech` | chemical equilibrium and partition functions |
| `korg.marcs_interpolation` | the MARCS grid and its interpolation |
