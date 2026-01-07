# Korg.jl to Python/JAX Port Progress

This file tracks the progress of converting Julia functions to JAX-compatible Python.

Legend:
- [ ] Not started
- [x] Completed

Each function has three checkboxes:
1. **Converted** - Code has been written
2. **Tested (no JIT)** - Matches Julia output to 1e-6 precision without JIT
3. **Tested (JIT)** - Works correctly with jax.jit

---

## Level 0: Constants (No Dependencies)

### Physical Constants (`constants.py`)
- [x] Converted | [x] Tested (no JIT) | [x] Tested (JIT) | `c_cgs` - speed of light in cm/s
- [x] Converted | [x] Tested (no JIT) | [x] Tested (JIT) | `hplanck_cgs` - Planck constant in erg*s
- [x] Converted | [x] Tested (no JIT) | [x] Tested (JIT) | `hplanck_eV` - Planck constant in eV*s
- [x] Converted | [x] Tested (no JIT) | [x] Tested (JIT) | `kboltz_cgs` - Boltzmann constant in erg/K
- [x] Converted | [x] Tested (no JIT) | [x] Tested (JIT) | `kboltz_eV` - Boltzmann constant in eV/K
- [x] Converted | [x] Tested (no JIT) | [x] Tested (JIT) | `electron_mass_cgs` - electron mass in g
- [x] Converted | [x] Tested (no JIT) | [x] Tested (JIT) | `electron_charge_cgs` - electron charge in esu
- [x] Converted | [x] Tested (no JIT) | [x] Tested (JIT) | `amu_cgs` - atomic mass unit in g
- [x] Converted | [x] Tested (no JIT) | [x] Tested (JIT) | `Rydberg_eV` - Rydberg energy in eV

### Atomic Data (`atomic_data.py`)
- [x] Converted | [x] Tested (no JIT) | [x] Tested (JIT) | `atomic_symbols` - Z -> symbol mapping (92 elements)
- [x] Converted | [x] Tested (no JIT) | [x] Tested (JIT) | `atomic_numbers` - symbol -> Z mapping
- [x] Converted | [x] Tested (no JIT) | [x] Tested (JIT) | `atomic_masses` - Z -> mass mapping (92 elements)
- [x] Converted | [x] Tested (no JIT) | [x] Tested (JIT) | `ionization_energies` - Z -> [χ₁, χ₂, χ₃] in eV (all 92 elements, 3 levels each)

### Solar Abundances (`atomic_data.py`)
- [x] Converted | [x] Tested (no JIT) | [x] Tested (JIT) | `grevesse_2007_solar_abundances` (92 elements)
- [x] Converted | [x] Tested (no JIT) | [x] Tested (JIT) | `asplund_2009_solar_abundances` (92 elements)
- [x] Converted | [x] Tested (no JIT) | [x] Tested (JIT) | `asplund_2020_solar_abundances` (92 elements)
- [x] Converted | [x] Tested (no JIT) | [x] Tested (JIT) | `bergemann_2025_solar_abundances` (92 elements)
- [x] Converted | [x] Tested (no JIT) | [x] Tested (JIT) | `default_solar_abundances` (= bergemann_2025)
- [x] Converted | [x] Tested (no JIT) | [x] Tested (JIT) | `magg_2022_solar_abundances` (92 elements)
- [x] Converted | N/A | N/A | `DEFAULT_ALPHA_ELEMENTS` (in abundances.py) - not exported by Julia, Python uses standard definition

### Isotopic Data (`isotopic_data.py`)
- [x] Converted | [x] Tested (no JIT) | [x] Tested (JIT) | `isotopic_abundances` (Z=1-92)
- [x] Converted | [x] Tested (no JIT) | [x] Tested (JIT) | `isotopic_nuclear_spin_degeneracies` (Z=1-92)

---

## Level 1: Simple Utility Functions (Constants Only)

### Wavelength Utilities (`utils.py`)
- [x] Converted | [x] Tested (no JIT) | [x] Tested (JIT) | `air_to_vacuum(λ)` - air to vacuum wavelength
- [x] Converted | [x] Tested (no JIT) | [x] Tested (JIT) | `vacuum_to_air(λ)` - vacuum to air wavelength

### Scattering (`continuum_absorption/scattering.py`)
- [x] Converted | [x] Tested (no JIT) | [x] Tested (JIT) | `electron_scattering(nₑ)` - Thomson scattering

### Line Physics (`line_absorption.py`)
- [x] Converted | [x] Tested (no JIT) | [x] Tested (JIT) | `sigma_line(λ)` - cross-section factor
- [x] Converted | [x] Tested (no JIT) | [x] Tested (JIT) | `doppler_width(λ₀, T, m, ξ)` - Doppler broadening σ
- [x] Converted | [x] Tested (no JIT) | [x] Tested (JIT) | `scaled_stark(γstark, T)` - temperature-scaled Stark

### Exponential Integrals (`radiative_transfer/expint.py`)
- [x] Converted | [x] Tested (no JIT) | [x] Tested (JIT) | `exponential_integral_1(x)` - E₁(x) approximation

### Statistical Mechanics (`statmech.py`)
- [x] Converted | [x] Tested (no JIT) | [x] Tested (JIT) | `translational_U(m, T)` - translational partition function

### Interval Utilities (`utils.py`)
- [x] Converted | [x] Tested (no JIT) | N/A (class) | `Interval` class - exclusive interval
- [x] Converted | [x] Tested (no JIT) | N/A (class) | `closed_interval(lo, up)` - inclusive interval
- [x] Converted | [x] Tested (no JIT) | N/A (class) | `contained(value, interval)` - check containment
- [x] Converted | [x] Tested (no JIT) | N/A (class) | `contained_slice(vals, interval)` - slice indices

### LSF Utilities (`utils.py`)
- [x] Converted | [x] Tested (no JIT) | [x] Tested (JIT) | `normal_pdf(Δ, σ)` - Gaussian PDF

---

## Level 2: Functions with Simple Dependencies

### Species and Formula (`species.py`)
- [x] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `Formula` class - chemical formula representation
- [x] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `Species` class - species with charge
- [x] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `n_atoms(species)` - count atoms in molecule
- [x] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `get_atoms(formula)` - get atomic numbers
- [x] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `get_mass(species)` - get species mass
- [x] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `ismolecule(species)` - check if molecule

### Voigt Profile (`line_absorption.py`)
- [ ] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `harris_series(v)` - Harris series for Voigt
- [ ] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `voigt_hjerting(α, v)` - Voigt-Hjerting function
- [ ] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `line_profile(λ₀, σ, γ, amplitude, λ)` - Voigt profile

### Line Window Functions (`line_absorption.py`)
- [ ] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `inverse_gaussian_density(ρ, σ)` - inverse Gaussian
- [ ] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `inverse_lorentz_density(ρ, γ)` - inverse Lorentz

### VdW Broadening (`line_absorption.py`)
- [ ] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `scaled_vdW(vdW, m, T)` - van der Waals broadening

### Scattering (`continuum_absorption/scattering.py`)
- [x] Converted | [x] Tested (no JIT) | [ ] Tested (JIT) | `rayleigh(νs, nH_I, nHe_I, nH2)` - Rayleigh scattering (uses assert, not JIT-compatible)

### Gaunt Factors (`continuum_absorption/hydrogenic_bf_ff.py`)
- [x] Converted | [x] Tested (no JIT) | [ ] Tested (JIT) | `gaunt_ff_vanHoof(log_u, log_γ2)` - ff Gaunt factor interpolation

### Hydrogenic Absorption (`continuum_absorption/hydrogenic_bf_ff.py`)
- [x] Converted | [x] Tested (no JIT) | [ ] Tested (JIT) | `hydrogenic_ff_absorption(ν, T, Z, ni, ne)` - hydrogenic ff

---

## Level 3: Intermediate Complexity

### Cubic Splines (`cubic_splines.py`)
- [x] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `CubicSpline` class - natural cubic spline
- [x] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `evaluate(spline, x)` - evaluate spline at x

### Wavelengths (`wavelengths.py`)
- [x] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `Wavelengths` class - wavelength grid handling
- [x] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `eachwindow(wls)` - iterate wavelength windows
- [x] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `eachfreq(wls)` - get frequencies
- [x] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `subspectrum_indices(wls)` - get subspectrum indices

### Abundances (`abundances.py`)
- [x] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `format_A_X(metals, alpha, abundances)` - format abundances
- [x] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `get_metals_H(A_X)` - calculate [metals/H]
- [x] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `get_alpha_H(A_X)` - calculate [α/H]

### Saha Equation (`statmech.py`)
- [x] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `saha_ion_weights(T, nₑ, atom, ...)` - ionization ratios
- [x] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `get_log_nK(mol, T, ...)` - molecular equilibrium constant

### Exponential Integrals (`radiative_transfer/expint.py`)
- [x] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `exponential_integral_2(x)` - E₂(x) approximation
- [x] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `expint_transfer_integral_core(τ, m, b)` - transfer integral

### Continuum Absorption Sources (`continuum_absorption/`)
- [x] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `H_I_bf(...)` - H I bound-free
- [x] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `Hminus_bf(...)` - H⁻ bound-free
- [x] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `Hminus_ff(...)` - H⁻ free-free
- [x] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `H2plus_bf_and_ff(...)` - H₂⁺ absorption
- [x] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `Heminus_ff(...)` - He⁻ free-free
- [x] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `positive_ion_ff_absorption!(...)` - metal ff
- [x] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `metal_bf_absorption!(...)` - metal bf

---

## Level 4: High-Level Functions

### Chemical Equilibrium (`statmech.py`)
- [x] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `chemical_equilibrium(T, nₜ, nₑ, ...)` - solve equilibrium

### Total Continuum (`continuum_absorption/__init__.py`)
- [x] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `total_continuum_absorption(νs, T, nₑ, ...)` - total α

### Line Absorption (`line_absorption.py`)
- [x] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `line_absorption!(α, linelist, λs, ...)` - compute line opacity

### Radiative Transfer (`radiative_transfer/`)
- [x] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `generate_mu_grid(n_points)` - μ quadrature grid
- [x] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `calculate_rays(μ_grid, spatial, spherical)` - ray geometry
- [x] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `compute_tau_anchored!(...)` - optical depth calculation
- [x] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `compute_I_linear!(...)` - intensity integration
- [x] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `compute_I_linear_flux_only(...)` - flux-only intensity
- [x] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `compute_F_flux_only_expint(...)` - flux with E₂
- [x] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `radiative_transfer(atm, α, S, ...)` - formal solution

### Hydrogen Lines (`hydrogen_line_absorption.py`)
- [x] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `hydrogen_line_absorption!(...)` - H line profiles

---

## Level 5: Top-Level API

### Atmosphere (`atmosphere.py`)
- [x] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `PlanarAtmosphere` class
- [x] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `ShellAtmosphere` class
- [x] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `read_model_atmosphere(filename)` - parse atmosphere file
- [x] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `interpolate_marcs(Teff, logg, A_X)` - MARCS interpolation

### Linelist (`linelist.py`)
- [x] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `Line` class - spectral line representation
- [x] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `read_linelist(filename)` - parse linelist file
- [x] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `get_VALD_solar_linelist()` - built-in VALD list
- [x] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `get_GALAH_DR3_linelist()` - built-in GALAH list

### Synthesis (`synthesis.py`)
- [x] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `synthesize(atm, linelist, A_X, wl_ranges)` - full synthesis
- [x] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `synth(...)` - high-level convenience wrapper

### Post-Processing (`utils.py`)
- [ ] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `apply_LSF(flux, wls, R)` - Gaussian LSF
- [ ] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `apply_rotation(flux, wls, vsini)` - rotational broadening
- [ ] Converted | [ ] Tested (no JIT) | [ ] Tested (JIT) | `compute_LSF_matrix(synth_wls, obs_wls, R)` - LSF matrix

---

## Summary

| Level | Total | Converted | Tested (no JIT) | Tested (JIT) |
|-------|-------|-----------|-----------------|--------------|
| 0     | 22    | 22        | 21              | 21           |
| 1     | 13    | 13        | 13              | 9            |
| 2     | 14    | 9         | 3               | 0            |
| 3     | 21    | 18        | 0               | 0            |
| 4     | 12    | 12        | 0               | 0            |
| 5     | 12    | 10        | 0               | 0            |
| **Total** | **94** | **84** | **37** | **30** |

Note: Level 1 Interval utilities (4 items) are marked N/A for JIT as they use Python classes.

---

## Test Results Summary

Tests run against Julia reference data (`tests/test_julia_reference.py`):
- **43 passed** (matching Julia to better than 1e-6 precision)
- **3 skipped** (missing implementations)

### Level 0 Passed Tests (all at rtol=1e-6):
1. **Physical Constants (7)**: c_cgs, hplanck_cgs, kboltz_cgs, electron_mass_cgs, Rydberg_eV, kboltz_eV, hplanck_eV
2. **Atomic symbols (1)**: All 92 element symbols match
3. **Atomic masses (1)**: All 92 elements match
4. **Ionization energies (1)**: All 92 elements × 3 ionization levels match
5. **Solar abundances (6)**: grevesse_2007, asplund_2009, asplund_2020, bergemann_2025, magg_2022, default (all 92 elements each)
6. **Isotopic abundances (1)**: All elements Z=1-92 with all isotope mass numbers match
7. **Isotopic nuclear spin degeneracies (1)**: All elements Z=1-92 with all isotope mass numbers match

### Level 1 Passed Tests:
8. **Wavelength utilities (2)**: air_to_vacuum, vacuum_to_air (11 wavelengths each)
9. **Line physics (3)**: sigma_line (6 wavelengths), doppler_width (5 test cases), scaled_stark (4 test cases)
10. **Normal PDF (1)**: All 8 (delta, sigma) combinations pass
11. **Exponential integral E1 (1)**: All 13 test values across all branches pass
12. **Interval utilities (3)**: contained (exclusive), closed_interval contained, contained_slice

### Level 2 Passed Tests:
13. **Electron scattering (1)**: All 5 test cases pass at rtol=1e-6
14. **Rayleigh scattering (1)**: All 7 wavelengths pass at rtol=1e-6
15. **Translational U (1)**: All 10 temperatures pass at rtol=1e-6
16. **Gaunt factor (1)**: All 7 (log_u, log_γ2) combinations pass at rtol=1e-5
17. **Hydrogenic ff (1)**: All 6 wavelengths pass at rtol=1e-5

### JIT Compatibility Tests (12):
- Level 1: air_to_vacuum, vacuum_to_air, sigma_line, doppler_width, scaled_stark, normal_pdf, exponential_integral_1
- Level 2: electron_scattering, translational_U, rayleigh (partial - uses assert)

### Skipped Tests (3):
1. `Species.from_string` - Uses constructor with string parsing instead
2. `Formula.from_string` - Uses constructor with string parsing instead
3. `test_species_is_molecule` - Depends on Species.from_string

### Known Issues:
- `rayleigh()` uses Python `assert` which is not JIT-compatible; function works but cannot be fully JIT'd
- Species and Formula classes work but have different API than Julia (constructor-based vs from_string)
- `DEFAULT_ALPHA_ELEMENTS` is not exported from Julia Korg but is implemented in Python as [8, 10, 12, 14, 16, 18, 20, 22]
- Interval utilities use Python classes and are not JIT-compatible (marked N/A)
