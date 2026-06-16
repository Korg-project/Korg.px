"""
Spectral synthesis orchestration - the main user-facing API.

This module combines all components (chemical equilibrium, continuum absorption,
line profiles, and radiative transfer) to compute synthetic stellar spectra.

Reference: Korg.jl synthesize.jl
"""

import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Callable, Union
from scipy.interpolate import interp1d

from .atmosphere import PlanarAtmosphere, ShellAtmosphere
from .statmech import (chemical_equilibrium, chemical_equilibrium_fast,
                       chemical_equilibrium_all_layers)
from .data_loader import (ionization_energies, default_partition_funcs,
                          default_log_equilibrium_constants,
                          default_chem_eq_data, default_mol_species)
from .continuum import (prepare_continuum_batch, prepare_continuum_batch_fast, batch_continuum_absorption,
                        Hminus_bf, Hminus_ff)
from .constants import (electron_mass_cgs, electron_charge_cgs, c_cgs,
                        kboltz_eV, hplanck_eV, hplanck_cgs, kboltz_cgs)
from .radiative_transfer import radiative_transfer, radiative_transfer_jit
from .linelist import Line
from .species import Species
from .line_absorption import line_absorption, _vdW_to_tuple, _voigt_profile_jax
from .hydrogen_line_absorption import (hydrogen_line_absorption, precompute_hummer_ws,
                                       hline_stark_profiles,
                                       hydrogen_line_absorption_stark_batched)
from .atomic_data import atomic_masses
from .abundances import A_X_to_absolute


@dataclass
class SynthesisResult:
    """
    Results from spectral synthesis.

    Attributes
    ----------
    wavelengths : array
        Wavelength grid [Å]
    flux : array
        Emergent flux [erg cm⁻² s⁻¹ Å⁻¹]
    continuum : array
        Continuum flux (no lines) [erg cm⁻² s⁻¹ Å⁻¹]
    intensities : array or None
        Intensity at each angle (if computed)
    """
    wavelengths: np.ndarray
    flux: np.ndarray
    continuum: np.ndarray
    intensities: Optional[np.ndarray] = None


def planck_function(nu, T):
    """
    Planck function B_ν(T).

    Parameters
    ----------
    nu : float or array
        Frequency [Hz]
    T : float or array
        Temperature [K]

    Returns
    -------
    B_nu : float or array
        Planck function [erg cm⁻² s⁻¹ sr⁻¹ Hz⁻¹]
    """
    from .constants import hplanck_cgs, kboltz_cgs

    h = hplanck_cgs
    k = kboltz_cgs
    c = c_cgs

    x = h * nu / (k * T)

    # Prevent overflow
    x = jnp.minimum(x, 100.0)

    return (2.0 * h * nu**3 / c**2) / (jnp.exp(x) - 1.0)


def blackbody(T, wavelength_cm):
    """
    Planck blackbody function B_λ(T) as a function of wavelength.

    This matches Julia's blackbody function in synthesize.jl.

    Parameters
    ----------
    T : float or array
        Temperature [K]
    wavelength_cm : float or array
        Wavelength [cm]

    Returns
    -------
    B_lambda : float or array
        Planck function [erg cm⁻² s⁻¹ sr⁻¹ cm⁻¹]

    Notes
    -----
    Uses the formula: B_λ = 2hc²/λ⁵ × 1/(exp(hc/λkT) - 1)
    """
    h = hplanck_cgs
    k = kboltz_cgs
    c = c_cgs

    x = h * c / (wavelength_cm * k * T)

    # Prevent overflow
    x = jnp.minimum(x, 100.0)

    return (2.0 * h * c**2 / wavelength_cm**5) / (jnp.exp(x) - 1.0)


def compute_continuum_absorption(
    wavelengths_cm: np.ndarray,
    T: float,
    ne: float,
    number_densities: Dict,
    partition_funcs: Dict,
) -> np.ndarray:
    """
    Compute continuum absorption coefficient at given wavelengths for one layer.

    Parameters
    ----------
    wavelengths_cm : array
        Wavelength grid [cm]
    T : float
        Temperature [K]
    ne : float
        Electron number density [cm⁻³]
    number_densities : dict
        Species -> number density mapping [cm⁻³]
    partition_funcs : dict
        Species -> partition function mapping

    Returns
    -------
    alpha_continuum : array
        Continuum absorption coefficient [cm⁻¹] at each wavelength
    """
    from .continuum import total_continuum_absorption

    # Convert wavelengths to frequencies
    frequencies = c_cgs / wavelengths_cm

    # Convert Species keys to strings for continuum module
    # Species.__str__() returns 'H I', 'Fe II', 'H2' etc (with spaces)
    # But continuum.py expects 'H_I', 'Fe_II', 'H2' (with underscores)
    def species_to_key(spec):
        s = str(spec)
        # Replace space with underscore: 'H I' -> 'H_I', 'Fe II' -> 'Fe_II'
        # But keep molecules like 'H2' unchanged
        return s.replace(' ', '_')

    number_densities_str = {species_to_key(k): v for k, v in number_densities.items()}
    partition_funcs_str = {species_to_key(k): v for k, v in partition_funcs.items()}

    # Call complete continuum absorption (includes all opacity sources)
    alpha_continuum = total_continuum_absorption(
        frequencies, T, ne, number_densities_str, partition_funcs_str
    )

    return np.array(alpha_continuum)


def filter_linelist(linelist: List[Line], wavelengths_cm: np.ndarray,
                    line_buffer_cm: float, warn_empty: bool = True) -> List[Line]:
    """
    Filter a sorted linelist to only lines that affect the given wavelength range.

    Parameters
    ----------
    linelist : list of Line
        Lines sorted by wavelength (cm)
    wavelengths_cm : array
        Synthesis wavelength grid (cm)
    line_buffer_cm : float
        Extra wavelength margin beyond the grid edges (cm)
    warn_empty : bool, optional
        Warn if the original linelist was non-empty but the result is empty (default: True)

    Returns
    -------
    filtered : list of Line
        Lines with wl in [wavelengths_cm[0] - line_buffer_cm, wavelengths_cm[-1] + line_buffer_cm]
    """
    import warnings
    import bisect

    n_before = len(linelist)
    if n_before == 0:
        return linelist

    lo = wavelengths_cm[0] - line_buffer_cm
    hi = wavelengths_cm[-1] + line_buffer_cm

    # Binary search for range limits in sorted list
    wls = [l.wl for l in linelist]
    i_start = bisect.bisect_left(wls, lo)
    i_stop = bisect.bisect_right(wls, hi)
    filtered = linelist[i_start:i_stop]

    if warn_empty and n_before > 0 and len(filtered) == 0:
        warnings.warn(
            "The provided linelist was not empty, but none of the lines were within "
            "the provided wavelength range."
        )
    return filtered


def get_reference_wavelength_linelist(linelist: List[Line],
                                      reference_wavelength_cm: float = 5e-5,
                                      use_internal_reference_linelist: bool = True
                                      ) -> List[Line]:
    """
    Return a linelist for computing absorption at the reference wavelength.

    Required for the anchored optical depth scheme (used by MARCS models at 5000 Å).
    If the user linelist has no lines near the reference wavelength, falls back to
    a built-in linelist (only available for 5000 Å / 5e-5 cm).

    Parameters
    ----------
    linelist : list of Line
        Sorted atomic/molecular lines
    reference_wavelength_cm : float
        Reference wavelength in cm (default: 5e-5 = 5000 Å, MARCS default)
    use_internal_reference_linelist : bool
        If True, fall back to the built-in 5000 Å linelist when the user linelist
        has insufficient coverage (default: True)

    Returns
    -------
    ref_linelist : list of Line
        Lines to use when computing opacity at the reference wavelength
    """
    # Filter to ±21 Å window around the reference wavelength (matches Julia)
    window_cm = np.array([reference_wavelength_cm, reference_wavelength_cm])
    buffer_cm = 21e-8  # 21 Å in cm
    filtered = filter_linelist(linelist, window_cm, buffer_cm, warn_empty=False)

    # If using a non-5000 Å reference and no lines found, error
    if reference_wavelength_cm != 5e-5 and len(filtered) == 0:
        raise ValueError(
            f"The provided linelist contains no lines near the reference wavelength "
            f"{reference_wavelength_cm * 1e8:.1f} Å. Korg has a built-in fallback "
            f"only for 5000 Å (the MARCS default)."
        )

    # If enough lines span the reference wavelength, use them
    if len(filtered) > 0 and filtered[0].wl <= reference_wavelength_cm <= filtered[-1].wl:
        return filtered

    # Fall back to or supplement with the built-in 5000 Å linelist
    if use_internal_reference_linelist and reference_wavelength_cm == 5e-5:
        from .data_loader import load_default_linelist
        try:
            builtin = load_default_linelist(reference_wavelength_cm)
        except Exception:
            builtin = []

        if len(filtered) == 0:
            return builtin

        # Supplement: prepend built-in lines below the user's coverage
        if len(filtered) > 0 and filtered[0].wl > reference_wavelength_cm:
            prefix = [l for l in builtin if l.wl < filtered[0].wl]
            return prefix + filtered

        # Supplement: append built-in lines above the user's coverage
        if len(filtered) > 0 and filtered[-1].wl < reference_wavelength_cm:
            suffix = [l for l in builtin if l.wl > filtered[-1].wl]
            return filtered + suffix

    return filtered


def synthesize_spectrum(
    atmosphere,
    linelist: List[Line],
    wavelengths_angstrom: np.ndarray,
    abundances: np.ndarray,
    vmic: float = 1.0,
    line_buffer: float = 10.0,
    cntm_step: float = 1.0,
    hydrogen_lines: bool = True,
    hydrogen_line_window_size: float = 150.0,
    line_cutoff_threshold: float = 3e-4,
    return_continuum: bool = True,
    partition_funcs: Optional[Dict] = None,
    ionization_energies_dict: Optional[Dict] = None,
    log_equilibrium_constants: Optional[Dict] = None,
    verbose: bool = True,
    profile: bool = False,
):
    """
    Compute synthetic spectrum with lines.

    This is the main synthesis function following Julia's synthesize().

    Parameters
    ----------
    atmosphere : PlanarAtmosphere or ShellAtmosphere
        Model atmosphere structure
    linelist : list of Line
        Atomic/molecular lines to include
    wavelengths_angstrom : array
        Wavelength grid [Å]
    abundances : array, shape (92,)
        Absolute abundances (N_X/N_total) for elements 1-92
    vmic : float, optional
        Microturbulent velocity [km/s] (default: 1.0)
    line_buffer : float, optional
        Distance [Å] from wavelength range to include lines (default: 10.0)
    cntm_step : float, optional
        Continuum sampling interval [Å] (default: 1.0)
    hydrogen_lines : bool, optional
        Include hydrogen lines (default: True)
    hydrogen_line_window_size : float, optional
        Window size for H lines [Å] (default: 150.0)
    line_cutoff_threshold : float, optional
        Line cutoff as fraction of continuum (default: 3e-4)
    return_continuum : bool, optional
        Whether to compute continuum spectrum (default: True)
    partition_funcs : dict, optional
        Partition functions (default: use built-in)
    ionization_energies_dict : dict, optional
        Ionization energies (default: use built-in)
    log_equilibrium_constants : dict, optional
        Equilibrium constants (default: use built-in)
    verbose : bool, optional
        Print progress messages (default: True)
    profile : bool, optional
        Print timing information for each synthesis step (default: False)

    Returns
    -------
    result : SynthesisResult
        Synthesis results with wavelengths, flux, and continuum
    """
    import time
    timings = {} if profile else None
    t_start = time.time() if profile else None

    using_defaults = (partition_funcs is None and ionization_energies_dict is None
                      and log_equilibrium_constants is None)
    if partition_funcs is None:
        partition_funcs = default_partition_funcs
    if ionization_energies_dict is None:
        ionization_energies_dict = ionization_energies
    if log_equilibrium_constants is None:
        log_equilibrium_constants = default_log_equilibrium_constants

    # Convert to working units (cm)
    wavelengths_cm = wavelengths_angstrom * 1e-8
    vmic_cm_s = vmic * 1e5  # km/s -> cm/s
    line_buffer_cm = line_buffer * 1e-8
    cntm_step_cm = cntm_step * 1e-8
    h_line_window_cm = hydrogen_line_window_size * 1e-8

    n_layers = atmosphere.n_layers
    n_wavelengths = len(wavelengths_angstrom)

    # Get atmosphere properties
    T = atmosphere.T
    ne_model = atmosphere.ne
    n_total = atmosphere.n_total
    log_tau_ref = atmosphere.log_tau_ref

    if isinstance(atmosphere, ShellAtmosphere):
        spatial_coord = atmosphere.r
        spherical = True
    else:
        spatial_coord = atmosphere.z
        spherical = False

    # Sort linelist by wavelength if needed
    if linelist and not all(linelist[i].wl <= linelist[i+1].wl
                            for i in range(len(linelist)-1)):
        linelist = sorted(linelist, key=lambda l: l.wl)

    # Filter linelist to wavelength range
    wl_min = wavelengths_cm[0] - line_buffer_cm
    wl_max = wavelengths_cm[-1] + line_buffer_cm
    linelist = [l for l in linelist if wl_min <= l.wl <= wl_max]

    if verbose:
        print(f"Synthesizing spectrum...")
        print(f"  Wavelengths: {n_wavelengths} points from "
              f"{wavelengths_angstrom[0]:.1f} to {wavelengths_angstrom[-1]:.1f} Å")
        print(f"  Lines: {len(linelist)} in wavelength range")
        print(f"  Layers: {n_layers}")

    # Set up continuum wavelength grid (coarser sampling)
    cntm_wl_min = wl_min - cntm_step_cm
    cntm_wl_max = wl_max + cntm_step_cm
    cntm_wavelengths_cm = np.arange(cntm_wl_min, cntm_wl_max + cntm_step_cm, cntm_step_cm)

    # Initialize arrays
    alpha = np.zeros((n_layers, n_wavelengths))  # Total absorption
    source_function = np.zeros((n_wavelengths, n_layers))

    # Reference wavelength for optical depth (5000 Å for MARCS models)
    lambda_ref_cm = 5e-5  # 5000 Å in cm

    # Convert A(X) format abundances to linear number fractions for chemical_equilibrium.
    # Julia's format_A_X() returns A(X) = log10(N_X/N_H) + 12; chemical_equilibrium
    # expects N(X)/N_total (values summing to ~1). Detect A(X) by H abundance > 1.
    if abundances[0] > 1.0:
        abs_abundances = A_X_to_absolute(np.asarray(abundances))
    else:
        abs_abundances = np.asarray(abundances)

    # Store chemical equilibrium results
    electron_densities = np.zeros(n_layers)
    alpha_ref = np.zeros(n_layers)  # Absorption at reference wavelength
    number_densities_list = []
    raw_arrays_list = []  # Raw numpy arrays from fast chemical equilibrium
    alpha_cntm_interps = []  # Continuum interpolators for each layer

    # Compute chemical equilibrium and continuum for each layer
    if verbose:
        print(f"Computing chemical equilibrium and continuum...")
    if profile:
        t_loop_start = time.time()
        t_chem_eq = 0.0
        t_cntm_abs = 0.0
        t_source_fn = 0.0

    # Pass 1: chemical equilibrium for all layers
    if profile:
        t0 = time.time()
    if using_defaults:
        # Batch all layers at once: single XLA dispatch, avoids 56× dict overhead
        electron_densities, number_densities_batch, raw_arrays_list = \
            chemical_equilibrium_all_layers(
                T, n_total, ne_model, abs_abundances,
                default_chem_eq_data, default_mol_species
            )
        number_densities_list = None  # not used in fast path
    else:
        for i in range(n_layers):
            T_i = T[i]
            ne_i = ne_model[i]
            n_i = n_total[i]
            ne_calc, n_dict = chemical_equilibrium(
                T_i, n_i, ne_i, abs_abundances,
                ionization_energies_dict,
                partition_funcs,
                log_equilibrium_constants,
                electron_density_warn_threshold=1.0
            )
            electron_densities[i] = ne_calc
            number_densities_list.append(n_dict)
    if profile:
        t_chem_eq = time.time() - t0

    # Pass 2: continuum absorption — batch all layers at once
    if profile:
        t0 = time.time()
    if using_defaults:
        # Fast path: vmapped JIT continuum over all layers
        cntm_frequencies = c_cgs / cntm_wavelengths_cm
        batch = prepare_continuum_batch_fast(raw_arrays_list, partition_funcs, T)
        alpha_cntm_all = np.array(batch_continuum_absorption(
            jnp.asarray(cntm_frequencies),
            jnp.asarray(T, dtype=np.float64),
            jnp.asarray(electron_densities, dtype=np.float64),
            batch
        ))
        for i in range(n_layers):
            alpha_cntm_interp = interp1d(cntm_wavelengths_cm, alpha_cntm_all[i],
                                          kind='linear', fill_value='extrapolate')
            alpha_cntm_interps.append(alpha_cntm_interp)
            alpha[i, :] = alpha_cntm_interp(wavelengths_cm)
            alpha_ref[i] = alpha_cntm_interp(lambda_ref_cm)
    else:
        for i in range(n_layers):
            alpha_cntm_coarse = compute_continuum_absorption(
                cntm_wavelengths_cm, T[i], electron_densities[i],
                number_densities_list[i], partition_funcs
            )
            alpha_cntm_interp = interp1d(cntm_wavelengths_cm, alpha_cntm_coarse,
                                          kind='linear', fill_value='extrapolate')
            alpha_cntm_interps.append(alpha_cntm_interp)
            alpha[i, :] = alpha_cntm_interp(wavelengths_cm)
            alpha_ref[i] = alpha_cntm_interp(lambda_ref_cm)
    if profile:
        t_cntm_abs = time.time() - t0

    # Source function (fast: vectorized blackbody over all layers and wavelengths)
    if profile:
        t0 = time.time()
    for i in range(n_layers):
        source_function[:, i] = blackbody(T[i], wavelengths_cm)
    if profile:
        t_source_fn = time.time() - t0

    # Build combined number_densities dict
    if using_defaults:
        # Already built as (n_layers,) arrays in chemical_equilibrium_all_layers
        number_densities = number_densities_batch
    else:
        all_species = set()
        for n_dict in number_densities_list:
            all_species.update(n_dict.keys())
        number_densities = {
            spec: np.array([n_dict.get(spec, 0.0) for n_dict in number_densities_list])
            for spec in all_species
        }

    if profile:
        timings['layer_loop'] = t_chem_eq + t_cntm_abs + t_source_fn
        timings['chemical_equilibrium'] = t_chem_eq
        timings['continuum_absorption'] = t_cntm_abs
        timings['source_function'] = t_source_fn
        t0 = time.time()

    # Compute continuum flux if requested
    continuum_flux = None
    if return_continuum:
        if verbose:
            print(f"Computing continuum spectrum...")
        if using_defaults and not spherical:
            flux_cntm, _ = radiative_transfer_jit(
                jnp.asarray(alpha.T), jnp.asarray(source_function),
                jnp.asarray(spatial_coord), jnp.asarray(log_tau_ref),
                jnp.asarray(alpha_ref)
            )
        else:
            flux_cntm, _ = radiative_transfer(
                alpha.T, source_function, spatial_coord, log_tau_ref,
                alpha_ref=alpha_ref, spherical=spherical,
                intensity_scheme="linear_flux_only", use_expint_flux=True
            )
        # Convert from erg/s/cm^5 to erg/s/cm^4/Å (same as flux below)
        continuum_flux = flux_cntm * 1e-8

    if profile:
        timings['continuum_rt'] = time.time() - t0
        t0 = time.time()

    # Add hydrogen line absorption
    if hydrogen_lines:
        if verbose:
            print(f"Adding hydrogen line absorption...")
        if profile:
            t_h_lines_start = time.time()

        # Pre-filter Stark profiles to only those near the synthesis wavelength range.
        # Approximate vacuum line centers via Rydberg formula; profiles whose center is
        # more than h_line_window_cm outside the range contribute exactly zero and are
        # skipped entirely (avoids O(84 × n_layers) JIT dispatches when the range
        # contains no H lines, e.g. 5000–5100 Å contains only the far wing of Hβ).
        _RYDBERG_CM = 1.0973731568539e5
        wl_min_cm = wavelengths_cm[0]
        wl_max_cm = wavelengths_cm[-1]
        nearby_stark = {
            k: v for k, v in hline_stark_profiles.items()
            if (wl_min_cm - h_line_window_cm
                <= 1.0 / (_RYDBERG_CM * (1.0/v.lower**2 - 1.0/v.upper**2))
                <= wl_max_cm + h_line_window_cm)
        }

        # Batch-precompute occupation probabilities for all layers (much faster than per-layer)
        if raw_arrays_list:
            nH_I_arr = np.array([ra['neutral_dens'][0] for ra in raw_arrays_list])
            nHe_I_arr = np.array([ra['neutral_dens'][1] for ra in raw_arrays_list])
        else:
            nH_I_arr = np.array([nd.get(Species("H_I"), 0.0) for nd in number_densities_list])
            nHe_I_arr = np.array([nd.get(Species("He_I"), 0.0) for nd in number_densities_list])
        ws_all = precompute_hummer_ws(T, nH_I_arr, nHe_I_arr, electron_densities)

        pf_H_I = partition_funcs[Species("H_I")]
        # Vectorized partition function evaluation (avoids 56 JAX scalar dispatches)
        log_T_arr = np.log(T)
        if hasattr(pf_H_I, 'numpy_eval'):
            U_H_I_arr = pf_H_I.numpy_eval(log_T_arr)
        else:
            U_H_I_arr = np.array([float(pf_H_I(lt)) for lt in log_T_arr])

        # Stehlé Stark profiles: process all layers at once (1 JIT dispatch per nearby line)
        if nearby_stark:
            alpha_stark = hydrogen_line_absorption_stark_batched(
                wavelengths_cm, T, electron_densities, nH_I_arr, U_H_I_arr,
                h_line_window_cm, vmic_cm_s, ws_all, nearby_stark
            )
            alpha += alpha_stark

        # Brackett series (lower=4): all series lines are in the IR (>1.4 μm), so skip
        # the per-layer loop entirely when the synthesis range is in the optical/UV.
        brackett_in_range = any(
            wl_min_cm - h_line_window_cm
            <= 1.0 / (_RYDBERG_CM * (1.0/16.0 - 1.0/m**2))
            <= wl_max_cm + h_line_window_cm
            for m in range(5, 31)
        )
        if brackett_in_range:
            for i in range(n_layers):
                alpha_H = hydrogen_line_absorption(
                    wavelengths_cm, T[i], electron_densities[i], nH_I_arr[i], nHe_I_arr[i],
                    float(U_H_I_arr[i]), vmic_cm_s,
                    h_line_window_cm, use_MHD=True, ws=ws_all[i],
                    stark_profiles={}   # Stark already handled above
                )
                alpha[i, :] += alpha_H
        if profile:
            timings['hydrogen_lines'] = time.time() - t_h_lines_start

    # Add atomic/molecular line absorption
    if linelist:
        if verbose:
            print(f"Adding line absorption for {len(linelist)} lines...")
        if profile:
            t_line_abs_start = time.time()

        # Create continuum opacity callable for line_absorption
        # Accepts scalar or 1D array of wavelengths; returns (n_layers,) or (n_wl, n_layers)
        def continuum_opacity(wl_cm):
            wl_arr = np.atleast_1d(wl_cm)
            result = np.stack([alpha_cntm_interps[i](wl_arr) for i in range(n_layers)], axis=-1)
            return result if np.ndim(wl_cm) > 0 else result[0]

        # Compute line absorption
        alpha_lines = line_absorption(
            linelist,
            wavelengths_cm,
            T,
            electron_densities,
            number_densities,
            partition_funcs,
            vmic_cm_s,
            continuum_opacity,
            cutoff_threshold=line_cutoff_threshold
        )

        alpha += alpha_lines
        if profile:
            timings['line_absorption'] = time.time() - t_line_abs_start

    # Solve radiative transfer with full opacity
    if verbose:
        print(f"Solving radiative transfer...")
    if profile:
        t_rt_start = time.time()

    if using_defaults and not spherical:
        flux_nu, _ = radiative_transfer_jit(
            jnp.asarray(alpha.T), jnp.asarray(source_function),
            jnp.asarray(spatial_coord), jnp.asarray(log_tau_ref),
            jnp.asarray(alpha_ref)
        )
    else:
        flux_nu, _ = radiative_transfer(
            alpha.T, source_function, spatial_coord, log_tau_ref,
            alpha_ref=alpha_ref, spherical=spherical,
            intensity_scheme="linear_flux_only", use_expint_flux=True
        )

    # Convert from erg/s/cm^5 (per cm wavelength) to erg/s/cm^4/Å (per Angstrom)
    # Since we use B_λ (wavelength-based Planck), we just multiply by 1e-8
    # (same as Julia's synthesize.jl line 304)
    flux_lambda = flux_nu * 1e-8

    if profile:
        timings['radiative_transfer'] = time.time() - t_rt_start
        timings['total'] = time.time() - t_start
        print("\n=== PROFILING RESULTS ===")
        print(f"  Layer loop total:       {timings.get('layer_loop', 0):.3f} s")
        print(f"    - Chemical equilibrium: {timings.get('chemical_equilibrium', 0):.3f} s")
        print(f"    - Continuum absorption: {timings.get('continuum_absorption', 0):.3f} s")
        print(f"    - Source function:      {timings.get('source_function', 0):.3f} s")
        if 'continuum_rt' in timings:
            print(f"  Continuum RT:           {timings['continuum_rt']:.3f} s")
        if 'hydrogen_lines' in timings:
            print(f"  Hydrogen lines:         {timings['hydrogen_lines']:.3f} s")
        if 'line_absorption' in timings:
            print(f"  Line absorption:        {timings['line_absorption']:.3f} s")
        print(f"  Radiative transfer:     {timings.get('radiative_transfer', 0):.3f} s")
        print(f"  TOTAL:                  {timings['total']:.3f} s")
        print("========================\n")

    if verbose:
        print(f"✓ Synthesis complete!")

    return SynthesisResult(
        wavelengths=wavelengths_angstrom,
        flux=np.array(flux_lambda),
        continuum=np.array(continuum_flux) if continuum_flux is not None else np.array(flux_lambda),
        intensities=None
    )


def synthesize_continuum(
    atmosphere,
    wavelengths_angstrom,
    abundances,
):
    """
    Compute continuum spectrum (without lines).

    This is a convenience wrapper that calls synthesize_spectrum with no linelist.

    Parameters
    ----------
    atmosphere : PlanarAtmosphere or ShellAtmosphere
        Model atmosphere structure
    wavelengths_angstrom : array
        Wavelength grid [Å]
    abundances : array, shape (92,)
        Absolute abundances (N_X/N_total) for elements 1-92

    Returns
    -------
    flux : array
        Continuum flux at each wavelength [erg cm⁻² s⁻¹ Å⁻¹]
    """
    result = synthesize_spectrum(
        atmosphere,
        linelist=[],
        wavelengths_angstrom=wavelengths_angstrom,
        abundances=abundances,
        hydrogen_lines=False,
        return_continuum=False,
        verbose=True
    )
    return result.flux


def synthesize(
    atmosphere,
    linelist: List[Line],
    wavelengths_angstrom: np.ndarray,
    abundances: np.ndarray,
    vmic: float = 1.0,
    line_buffer: float = 10.0,
    hydrogen_lines: bool = True,
    hydrogen_line_window_size: float = 150.0,
    line_cutoff_threshold: float = 3e-4,
    return_cntm: bool = True,
    verbose: bool = True,
    profile: bool = False,
    **kwargs
):
    """
    Main spectral synthesis function.

    Computes synthetic spectrum for a given atmosphere, linelist, and abundances.
    This is the primary user-facing API, matching Julia's `synthesize()`.

    Parameters
    ----------
    atmosphere : PlanarAtmosphere or ShellAtmosphere
        Model atmosphere structure with T, P, ρ vs depth
    linelist : list of Line
        Atomic and molecular lines to include. Use [] for continuum-only.
    wavelengths_angstrom : array
        Wavelength grid for output spectrum [Å]
    abundances : array, shape (92,)
        Absolute abundances (N_X/N_total) for elements 1-92
        Can be generated using format_A_X()
    vmic : float, optional
        Microturbulent velocity [km/s] (default: 1.0)
    line_buffer : float, optional
        Distance [Å] from wavelength range to include lines (default: 10.0)
    hydrogen_lines : bool, optional
        Include hydrogen lines (default: True)
    hydrogen_line_window_size : float, optional
        Window size for H lines [Å] (default: 150.0)
    line_cutoff_threshold : float, optional
        Line cutoff as fraction of continuum (default: 3e-4)
    return_cntm : bool, optional
        Whether to compute continuum spectrum (default: True)
    verbose : bool, optional
        Print progress messages (default: True)
    profile : bool, optional
        Print timing information for each synthesis step (default: False)

    Returns
    -------
    result : SynthesisResult
        Synthesis results with wavelengths, flux, and continuum

    Examples
    --------
    >>> from korg.atmosphere import create_solar_test_atmosphere
    >>> from korg.abundances import format_A_X
    >>> from korg.linelist import read_linelist
    >>>
    >>> # Create solar atmosphere
    >>> atm = create_solar_test_atmosphere()
    >>>
    >>> # Get solar abundances
    >>> A_X = format_A_X(M_H=0.0, alpha_M=0.0)
    >>> abundances = 10**(A_X - 12)
    >>> abundances /= abundances.sum()
    >>>
    >>> # Read linelist
    >>> linelist = read_linelist("path/to/linelist.vald")
    >>>
    >>> # Define wavelength grid
    >>> wavelengths = np.linspace(5000, 5100, 1000)  # Å
    >>>
    >>> # Synthesize
    >>> result = synthesize(atm, linelist, wavelengths, abundances)
    >>>
    >>> # Plot
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(result.wavelengths, result.flux)
    >>> plt.xlabel('Wavelength [Å]')
    >>> plt.ylabel('Flux')
    >>> plt.show()

    Notes
    -----
    This is the main user-facing function for spectral synthesis.

    The synthesis proceeds in stages:
    1. Chemical equilibrium: compute species densities at each layer
    2. Continuum absorption: compute opacity from all continuum sources
    3. Hydrogen line absorption: Stark-broadened H lines (if hydrogen_lines=True)
    4. Line absorption: Voigt profiles for atomic/molecular lines
    5. Radiative transfer: solve for emergent flux
    """
    wavelengths_angstrom = np.asarray(wavelengths_angstrom)

    return synthesize_spectrum(
        atmosphere=atmosphere,
        linelist=linelist,
        wavelengths_angstrom=wavelengths_angstrom,
        abundances=abundances,
        vmic=vmic,
        line_buffer=line_buffer,
        hydrogen_lines=hydrogen_lines,
        hydrogen_line_window_size=hydrogen_line_window_size,
        line_cutoff_threshold=line_cutoff_threshold,
        return_continuum=return_cntm,
        verbose=verbose,
        profile=profile,
        **kwargs
    )


def synth(
    atmosphere,
    linelist: List[Line],
    wavelengths_angstrom: np.ndarray,
    abundances: np.ndarray,
    vmic: float = 1.0,
    **kwargs
):
    """
    Convenience wrapper for synthesize() that returns simple tuple.

    Parameters
    ----------
    atmosphere : PlanarAtmosphere or ShellAtmosphere
        Model atmosphere
    linelist : list of Line
        Atomic and molecular lines. Use [] for continuum-only.
    wavelengths_angstrom : array
        Wavelength grid [Å]
    abundances : array
        Elemental abundances
    vmic : float, optional
        Microturbulent velocity [km/s] (default: 1.0)

    Returns
    -------
    wavelengths : array
        Wavelength grid [Å]
    flux : array
        Emergent flux
    continuum : array
        Continuum flux

    Examples
    --------
    >>> wl, flux, cont = synth(atm, linelist, wavelengths, abundances)
    """
    result = synthesize(atmosphere, linelist, wavelengths_angstrom, abundances, vmic=vmic, **kwargs)
    return result.wavelengths, result.flux, result.continuum


# =============================================================================
# JIT-COMPATIBLE SYNTHESIS
# =============================================================================
# The following functions provide a fully JAX JIT-compatible synthesis pipeline.
# Usage:
#   1. Call precompute_synthesis_data() once to create static data
#   2. Call preprocess_linelist() to convert Line objects to arrays
#   3. Call synthesize_jit() for JIT-compiled synthesis
# =============================================================================

import jax
from typing import NamedTuple
from .statmech import (ChemicalEquilibriumData, precompute_chemical_equilibrium_data,
                       chemical_equilibrium_jit, MAX_ATOMIC_NUMBER,
                       _compute_saha_weights_jit, translational_U,
                       _compute_mol_densities_jit,
                       _chemical_equilibrium_batch_jit,
                       _compute_mol_densities_batch_jit,
                       _compute_saha_weights_batch_jit)
from .continuum import (_batch_continuum_vmap, _get_metal_bf_idx,
                        get_metal_bf_cross_sections, _PEACH_IDX, _H2_MOL_IDX)


class LinelistData(NamedTuple):
    """
    Linelist data stored as JAX-compatible arrays.

    All arrays have shape (n_lines,) unless otherwise noted.
    """
    n_lines: int
    wl: jnp.ndarray           # Wavelengths [cm]
    log_gf: jnp.ndarray       # log(gf) values
    species_Z: jnp.ndarray    # Atomic number (Z) of species, shape (n_lines,)
    species_charge: jnp.ndarray  # Charge state (0=neutral, 1=ionized), shape (n_lines,)
    E_lower: jnp.ndarray      # Lower level energy [eV]
    gamma_rad: jnp.ndarray    # Radiative damping [rad/s]
    gamma_stark: jnp.ndarray  # Stark broadening parameter
    vdW_sigma: jnp.ndarray    # van der Waals sigma
    vdW_alpha: jnp.ndarray    # van der Waals alpha (-1 for simple scaling)
    mass: jnp.ndarray         # Species mass [g]
    mol_species_idx: jnp.ndarray  # Index into mol_densities array; -1 for atomic species


class SynthesisData(NamedTuple):
    """
    Pre-computed data for JIT-compatible synthesis.

    Combines chemical equilibrium data with additional synthesis parameters.
    """
    # Chemical equilibrium data
    chem_eq_data: ChemicalEquilibriumData

    # Gaunt factor interpolation grid (for free-free absorption)
    gaunt_log_u_grid: jnp.ndarray      # shape (n_u,)
    gaunt_log_gamma2_grid: jnp.ndarray  # shape (n_gamma2,)
    gaunt_table: jnp.ndarray           # shape (n_u, n_gamma2)

    # Metal bound-free cross-section tables (for _total_continuum_fast / _batch_continuum_vmap)
    metal_bf_tables: jnp.ndarray     # shape (n_metal, n_logT, n_nu)
    metal_bf_nu_grid: jnp.ndarray    # shape (n_nu,)
    metal_bf_logT_grid: jnp.ndarray  # shape (n_logT,)
    metal_bf_z_arr: jnp.ndarray      # shape (n_metal,) int32 — Z-1 index per species
    metal_bf_charge_arr: jnp.ndarray # shape (n_metal,) int32 — ionization charge per species


def precompute_synthesis_data(
    ionization_energies_dict,
    partition_funcs,
    log_equilibrium_constants,
    T_min: float = 1000.0,
    T_max: float = 50000.0,
    n_temps: int = 500
) -> SynthesisData:
    """
    Pre-compute all data needed for JIT-compatible synthesis.

    This should be called once before any JIT-compiled synthesis calls.

    Parameters
    ----------
    ionization_energies_dict : dict
        Ionization energies for each element
    partition_funcs : dict
        Partition functions for each species
    log_equilibrium_constants : dict
        Equilibrium constants for molecules
    T_min, T_max : float
        Temperature range for precomputation
    n_temps : int
        Number of temperature grid points

    Returns
    -------
    SynthesisData
        Pre-computed data structure
    """
    # Get chemical equilibrium data
    chem_eq_data = precompute_chemical_equilibrium_data(
        ionization_energies_dict, partition_funcs, log_equilibrium_constants,
        T_min=T_min, T_max=T_max, n_temps=n_temps
    )

    # Load Gaunt factor table for free-free absorption
    from .continuum_absorption.hydrogenic_bf_ff import _load_gauntff_table
    try:
        gaunt_table, log_gamma2_grid, log_u_grid = _load_gauntff_table()
        gaunt_table = jnp.array(gaunt_table)
        gaunt_log_u_grid = jnp.array(log_u_grid)
        gaunt_log_gamma2_grid = jnp.array(log_gamma2_grid)
    except Exception:
        # Fallback: use approximate Gaunt factor = 1
        gaunt_log_u_grid = jnp.array([-4.0, 4.0])
        gaunt_log_gamma2_grid = jnp.array([-4.0, 4.0])
        gaunt_table = jnp.ones((2, 2))

    # Pre-build metal BF tables and species index mapping for _batch_continuum_vmap
    _, metal_bf_idx = _get_metal_bf_idx()
    bf_data = get_metal_bf_cross_sections()
    metal_species_names = list(bf_data['species'].keys())
    metal_bf_tables = jnp.stack([jnp.array(bf_data['species'][s]) for s in metal_species_names])
    metal_bf_nu_grid = jnp.array(bf_data['nu_grid'])
    metal_bf_logT_grid = jnp.array(bf_data['logT_grid'])
    metal_bf_z_arr = jnp.array([z for z, _ in metal_bf_idx], dtype=jnp.int32)
    metal_bf_charge_arr = jnp.array([c for _, c in metal_bf_idx], dtype=jnp.int32)

    return SynthesisData(
        chem_eq_data=chem_eq_data,
        gaunt_log_u_grid=gaunt_log_u_grid,
        gaunt_log_gamma2_grid=gaunt_log_gamma2_grid,
        gaunt_table=gaunt_table,
        metal_bf_tables=metal_bf_tables,
        metal_bf_nu_grid=metal_bf_nu_grid,
        metal_bf_logT_grid=metal_bf_logT_grid,
        metal_bf_z_arr=metal_bf_z_arr,
        metal_bf_charge_arr=metal_bf_charge_arr,
    )


def save_synthesis_data(data: SynthesisData, path: str) -> None:
    """
    Save precomputed synthesis data to .npz file.

    Parameters
    ----------
    data : SynthesisData
        Pre-computed synthesis data from precompute_synthesis_data()
    path : str
        Path to save the .npz file
    """
    np.savez_compressed(
        path,
        # ChemicalEquilibriumData
        log_T_grid=np.asarray(data.chem_eq_data.log_T_grid),
        ionization_energies=np.asarray(data.chem_eq_data.ionization_energies),
        partition_func_values=np.asarray(data.chem_eq_data.partition_func_values),
        pf_orig_t=np.asarray(data.chem_eq_data.pf_orig_t),
        pf_orig_u=np.asarray(data.chem_eq_data.pf_orig_u),
        pf_orig_h=np.asarray(data.chem_eq_data.pf_orig_h),
        pf_orig_z=np.asarray(data.chem_eq_data.pf_orig_z),
        pf_orig_n=np.asarray(data.chem_eq_data.pf_orig_n),
        log_T_h=np.asarray(data.chem_eq_data.log_T_h),
        n_molecules=data.chem_eq_data.n_molecules,
        mol_atoms_array=np.asarray(data.chem_eq_data.mol_atoms_array),
        mol_charges=np.asarray(data.chem_eq_data.mol_charges),
        mol_n_atoms=np.asarray(data.chem_eq_data.mol_n_atoms),
        mol_log_K_values=np.asarray(data.chem_eq_data.mol_log_K_values),
        mol_partition_func_values=np.asarray(data.chem_eq_data.mol_partition_func_values),
        mol_partition_func_z=np.asarray(data.chem_eq_data.mol_partition_func_z),
        mol_atom_consume=np.asarray(data.chem_eq_data.mol_atom_consume),
        # Gaunt factor tables
        gaunt_log_u_grid=np.asarray(data.gaunt_log_u_grid),
        gaunt_log_gamma2_grid=np.asarray(data.gaunt_log_gamma2_grid),
        gaunt_table=np.asarray(data.gaunt_table),
        # Metal bound-free tables
        metal_bf_tables=np.asarray(data.metal_bf_tables),
        metal_bf_nu_grid=np.asarray(data.metal_bf_nu_grid),
        metal_bf_logT_grid=np.asarray(data.metal_bf_logT_grid),
        metal_bf_z_arr=np.asarray(data.metal_bf_z_arr),
        metal_bf_charge_arr=np.asarray(data.metal_bf_charge_arr),
    )


def load_synthesis_data(path: Optional[str] = None) -> SynthesisData:
    """
    Load precomputed synthesis data from .npz file.

    Parameters
    ----------
    path : str, optional
        Path to the .npz file. If None, loads from package data.

    Returns
    -------
    SynthesisData
        Pre-computed data structure ready for JIT-compiled synthesis.
    """
    if path is None:
        path = Path(__file__).parent / "data" / "synthesis_data.npz"

    _, metal_bf_idx = _get_metal_bf_idx()
    bf_data = get_metal_bf_cross_sections()
    metal_species_names = list(bf_data['species'].keys())

    with np.load(path) as f:
        chem_eq_data = ChemicalEquilibriumData(
            log_T_grid=jnp.array(f['log_T_grid']),
            ionization_energies=jnp.array(f['ionization_energies']),
            partition_func_values=jnp.array(f['partition_func_values']),
            pf_orig_t=jnp.array(f['pf_orig_t']),
            pf_orig_u=jnp.array(f['pf_orig_u']),
            pf_orig_h=jnp.array(f['pf_orig_h']),
            pf_orig_z=jnp.array(f['pf_orig_z']),
            pf_orig_n=jnp.array(f['pf_orig_n'], dtype=jnp.int32),
            log_T_h=jnp.array(f['log_T_h']),
            n_molecules=int(f['n_molecules']),
            mol_atoms_array=jnp.array(f['mol_atoms_array']),
            mol_charges=jnp.array(f['mol_charges']),
            mol_n_atoms=jnp.array(f['mol_n_atoms']),
            mol_log_K_values=jnp.array(f['mol_log_K_values']),
            mol_partition_func_values=jnp.array(f['mol_partition_func_values']),
            mol_partition_func_z=jnp.array(f['mol_partition_func_z']),
            mol_atom_consume=jnp.array(f['mol_atom_consume']),
        )
        return SynthesisData(
            chem_eq_data=chem_eq_data,
            gaunt_log_u_grid=jnp.array(f['gaunt_log_u_grid']),
            gaunt_log_gamma2_grid=jnp.array(f['gaunt_log_gamma2_grid']),
            gaunt_table=jnp.array(f['gaunt_table']),
            metal_bf_tables=jnp.stack([jnp.array(bf_data['species'][s]) for s in metal_species_names]),
            metal_bf_nu_grid=jnp.array(bf_data['nu_grid']),
            metal_bf_logT_grid=jnp.array(bf_data['logT_grid']),
            metal_bf_z_arr=jnp.array([z for z, _ in metal_bf_idx], dtype=jnp.int32),
            metal_bf_charge_arr=jnp.array([c for _, c in metal_bf_idx], dtype=jnp.int32),
        )


def preprocess_linelist(linelist: List[Line], chem_eq_data=None) -> LinelistData:
    """
    Convert a list of Line objects to JAX-compatible arrays.

    Parameters
    ----------
    linelist : list of Line
        Standard linelist with Line objects
    chem_eq_data : ChemicalEquilibriumData, optional
        Pre-computed chemical equilibrium data. When provided, molecular species
        are matched against mol_atoms_array and mol_species_idx is set to the
        molecule's index; otherwise mol_species_idx is -1 for all lines.

    Returns
    -------
    LinelistData
        Linelist data as arrays
    """
    if not linelist:
        return LinelistData(
            n_lines=0,
            wl=jnp.array([]),
            log_gf=jnp.array([]),
            species_Z=jnp.array([], dtype=jnp.int32),
            species_charge=jnp.array([], dtype=jnp.int32),
            E_lower=jnp.array([]),
            gamma_rad=jnp.array([]),
            gamma_stark=jnp.array([]),
            vdW_sigma=jnp.array([]),
            vdW_alpha=jnp.array([]),
            mass=jnp.array([]),
            mol_species_idx=jnp.array([], dtype=jnp.int32)
        )

    # Build molecule lookup table if chem_eq_data is provided
    mol_atoms_np = None
    mol_charges_np = None
    if chem_eq_data is not None and chem_eq_data.n_molecules > 0:
        import numpy as _np
        mol_atoms_np = _np.array(chem_eq_data.mol_atoms_array)   # (n_mols, 6)
        mol_charges_np = _np.array(chem_eq_data.mol_charges)     # (n_mols,)

    n_lines = len(linelist)
    wl = jnp.array([line.wl for line in linelist])
    log_gf = jnp.array([line.log_gf for line in linelist])

    # Extract species info
    species_Z = []
    species_charge = []
    masses = []
    mol_species_idx_list = []
    for line in linelist:
        atoms = line.species.get_atoms()
        Z = int(atoms[0]) if len(atoms) > 0 else 1
        species_Z.append(Z)
        species_charge.append(line.species.charge)
        masses.append(line.species.get_mass())

        # Determine mol_species_idx: match molecular lines against chem_eq_data
        mol_idx = -1
        if len(atoms) > 1 and mol_atoms_np is not None:
            atoms_z1 = sorted([int(a) - 1 for a in atoms])
            padded = atoms_z1 + [-1] * (6 - len(atoms_z1))
            line_charge = line.species.charge
            for i in range(mol_atoms_np.shape[0]):
                if (list(mol_atoms_np[i]) == padded and
                        int(mol_charges_np[i]) == line_charge):
                    mol_idx = i
                    break
        mol_species_idx_list.append(mol_idx)

    E_lower = jnp.array([line.E_lower for line in linelist])
    gamma_rad = jnp.array([line.gamma_rad for line in linelist])
    gamma_stark = jnp.array([line.gamma_stark for line in linelist])

    # van der Waals parameters (use _vdW_to_tuple for robust handling of
    # scalar log-gamma, tuple ABO, and None inputs)
    # For ABO lines (alpha >= 0), precompute the full Barklem-O'Mara pre-factor at
    # T_ref=10000 K so that per_layer only needs: 2 * vdW_sigma * 1e6 * (T/1e4)^(0.5*(1-alpha))
    from scipy.special import gamma as _gamma_fn
    import numpy as _np_vdw
    from .constants import amu_cgs as _amu_cgs, kboltz_cgs as _kboltz_cgs
    _v0 = 1e6   # reference velocity in cm/s
    _T_ref = 1e4
    _M_H = 1.008 * _amu_cgs

    vdW_params = [_vdW_to_tuple(line.vdW) for line in linelist]
    raw_sigma  = [p[0] for p in vdW_params]
    raw_alpha  = [p[1] for p in vdW_params]

    corrected_sigma = []
    for sigma, alpha, mass in zip(raw_sigma, raw_alpha, masses):
        if alpha >= 0.0 and sigma > 0.0:
            inv_mu = 1.0 / _M_H + 1.0 / mass
            vbar_ref = _np_vdw.sqrt(8.0 * _kboltz_cgs * _T_ref / _np_vdw.pi * inv_mu)
            C = ((4.0 / _np_vdw.pi) ** (alpha / 2.0)
                 * _gamma_fn((4.0 - alpha) / 2.0)
                 * (vbar_ref / _v0) ** (1.0 - alpha))
            corrected_sigma.append(sigma * C)
        else:
            corrected_sigma.append(sigma)

    vdW_sigma = jnp.array(corrected_sigma)
    vdW_alpha = jnp.array(raw_alpha)

    return LinelistData(
        n_lines=n_lines,
        wl=wl,
        log_gf=log_gf,
        species_Z=jnp.array(species_Z, dtype=jnp.int32),
        species_charge=jnp.array(species_charge, dtype=jnp.int32),
        E_lower=E_lower,
        gamma_rad=gamma_rad,
        gamma_stark=gamma_stark,
        vdW_sigma=vdW_sigma,
        vdW_alpha=vdW_alpha,
        mass=jnp.array(masses),
        mol_species_idx=jnp.array(mol_species_idx_list, dtype=jnp.int32)
    )


def _interp2d_jit(x, y, xgrid, ygrid, zgrid):
    """
    Simple 2D bilinear interpolation (JIT-compatible).

    Parameters
    ----------
    x, y : float
        Point to interpolate at
    xgrid, ygrid : array
        Grid coordinates
    zgrid : array
        Grid values, shape (len(xgrid), len(ygrid))

    Returns
    -------
    z : float
        Interpolated value
    """
    # Find indices
    ix = jnp.searchsorted(xgrid, x) - 1
    iy = jnp.searchsorted(ygrid, y) - 1

    # Clamp to valid range
    ix = jnp.clip(ix, 0, len(xgrid) - 2)
    iy = jnp.clip(iy, 0, len(ygrid) - 2)

    # Get surrounding values
    x0, x1 = xgrid[ix], xgrid[ix + 1]
    y0, y1 = ygrid[iy], ygrid[iy + 1]

    z00 = zgrid[ix, iy]
    z01 = zgrid[ix, iy + 1]
    z10 = zgrid[ix + 1, iy]
    z11 = zgrid[ix + 1, iy + 1]

    # Bilinear interpolation
    wx = (x - x0) / (x1 - x0 + 1e-30)
    wy = (y - y0) / (y1 - y0 + 1e-30)

    z = (z00 * (1 - wx) * (1 - wy) +
         z10 * wx * (1 - wy) +
         z01 * (1 - wx) * wy +
         z11 * wx * wy)

    return z


def _gaunt_ff_jit(nu, T, Z, data):
    """
    Thermally-averaged free-free Gaunt factor (JIT-compatible).

    Parameters
    ----------
    nu : float
        Frequency [Hz]
    T : float
        Temperature [K]
    Z : int
        Ion charge
    data : SynthesisData
        Pre-computed data

    Returns
    -------
    g_ff : float
        Gaunt factor
    """
    from .constants import Rydberg_eV, kboltz_eV, hplanck_eV

    # Dimensionless parameters
    gamma2 = Z**2 * Rydberg_eV / (kboltz_eV * T)
    u = hplanck_eV * nu / (kboltz_eV * T)

    log10_gamma2 = jnp.log10(jnp.clip(gamma2, 1e-10, 1e10))
    log10_u = jnp.log10(jnp.clip(u, 1e-10, 1e10))

    # Interpolate from table
    g_ff = _interp2d_jit(log10_u, log10_gamma2,
                         data.gaunt_log_u_grid, data.gaunt_log_gamma2_grid,
                         data.gaunt_table)

    return jnp.clip(g_ff, 0.1, 10.0)


def _hydrogenic_ff_jit(nu, T, Z, n_ion, ne, data):
    """
    Hydrogenic free-free absorption coefficient (JIT-compatible).

    Parameters
    ----------
    nu : float
        Frequency [Hz]
    T : float
        Temperature [K]
    Z : int
        Ion charge
    n_ion : float
        Ion number density [cm⁻³]
    ne : float
        Electron number density [cm⁻³]
    data : SynthesisData
        Pre-computed data

    Returns
    -------
    alpha_ff : float
        Absorption coefficient [cm⁻¹]
    """
    from .constants import electron_charge_cgs, electron_mass_cgs, hplanck_cgs, kboltz_cgs

    g_ff = _gaunt_ff_jit(nu, T, Z, data)

    # Free-free absorption coefficient
    # Formula from Rybicki & Lightman (1979) eq. 5.18b
    coeff = (4 * electron_charge_cgs**6 /
             (3 * electron_mass_cgs * hplanck_cgs * c_cgs) *
             jnp.sqrt(2 * jnp.pi / (3 * kboltz_cgs * electron_mass_cgs)))

    alpha_ff = (coeff * Z**2 * n_ion * ne * g_ff /
                (T**0.5 * nu**3) *
                (1 - jnp.exp(-hplanck_cgs * nu / (kboltz_cgs * T))))

    return alpha_ff


def _hminus_bf_jit(nu, T, nH_I_div_U, ne):
    """
    H⁻ bound-free absorption coefficient (JIT-compatible).

    Uses polynomial fit from John (1988).
    """
    from .constants import hplanck_eV, kboltz_eV, electron_mass_cgs

    # Photon energy in eV
    E_photon = hplanck_eV * nu

    # H⁻ binding energy
    E_bind = 0.7552  # eV

    # Check if photon can ionize H⁻
    valid = E_photon > E_bind

    # Polynomial fit coefficients (John 1988)
    wavelength_um = 1e4 * c_cgs / nu  # wavelength in microns

    # Cross section fit (valid for 0.125 < λ < 1.6419 μm)
    a = jnp.array([1.99654, -1.18267e-1, 2.64243e-2,
                   -4.40524e-3, 3.23992e-4, -1.39568e-5, 2.78701e-7])

    x = wavelength_um
    sigma = jnp.where(
        (wavelength_um > 0.125) & (wavelength_um < 1.6419),
        1e-18 * (a[0] + a[1]*x + a[2]*x**2 + a[3]*x**3 +
                 a[4]*x**4 + a[5]*x**5 + a[6]*x**6) * (x - 0.125)**1.5 / x**3,
        0.0
    )

    # Number density of H⁻ from Saha equation
    transU = translational_U(electron_mass_cgs, T)
    chi_Hminus = E_bind
    n_Hminus = nH_I_div_U * ne / (2 * transU) * jnp.exp(chi_Hminus / (kboltz_eV * T))

    alpha_bf = jnp.where(valid, sigma * n_Hminus, 0.0)

    return alpha_bf


def _hminus_ff_jit(nu, T, nH_I_div_U, ne):
    """
    H⁻ free-free absorption coefficient (JIT-compatible).

    Uses polynomial fit from John (1988).
    """
    from .constants import kboltz_cgs, hplanck_cgs

    wavelength_um = 1e4 * c_cgs / nu

    # Polynomial coefficients from John (1988) Table 3
    # For λ in range 0.182 - 10 μm
    f0 = -2.2763 - 1.6850 * jnp.log10(wavelength_um)
    f1 = 8.3618 + 5.9565 * jnp.log10(wavelength_um)
    f2 = -11.4770 - 7.8680 * jnp.log10(wavelength_um)

    theta = 5040.0 / T

    log_kappa = (f0 + f1 * jnp.log10(theta) + f2 * (jnp.log10(theta))**2 +
                 jnp.log10(nH_I_div_U * ne * 1e-26))

    alpha_ff = jnp.where(
        (wavelength_um > 0.182) & (wavelength_um < 10.0),
        10**log_kappa,
        0.0
    )

    return alpha_ff


def _rayleigh_jit(nu, nH_I, nHe_I, nH2):
    """
    Rayleigh scattering coefficient (JIT-compatible).
    """
    from .constants import hplanck_eV, Rydberg_eV

    sigma_th = 6.65246e-25  # Thomson cross section

    E_2Ryd_2 = (hplanck_eV * nu / (2 * Rydberg_eV))**2
    E_2Ryd_4 = E_2Ryd_2**2
    E_2Ryd_6 = E_2Ryd_2 * E_2Ryd_4
    E_2Ryd_8 = E_2Ryd_4**2

    # H (Colgan+ 2016)
    sigma_H = (20.24 * E_2Ryd_4 + 239.2 * E_2Ryd_6 + 2256 * E_2Ryd_8) * sigma_th

    # He (Colgan+ 2016)
    sigma_He = (1.913 * E_2Ryd_4 + 4.52 * E_2Ryd_6 + 7.90 * E_2Ryd_8) * sigma_th

    # H2 (Dalgarno & Williams 1962)
    inv_lambda_2 = (nu / (1e8 * c_cgs))**2
    inv_lambda_4 = inv_lambda_2**2
    inv_lambda_6 = inv_lambda_2 * inv_lambda_4
    inv_lambda_8 = inv_lambda_4**2
    alpha_H2 = (8.14e-13 * inv_lambda_4 + 1.28e-6 * inv_lambda_6 + 1.61 * inv_lambda_8) * nH2

    return nH_I * sigma_H + nHe_I * sigma_He + alpha_H2


def _electron_scattering_jit(ne):
    """Electron (Thomson) scattering coefficient."""
    sigma_th = 6.65246e-25
    return sigma_th * ne


def _continuum_absorption_jit(wavelength_cm, T, ne, nH_I, nH_II, nHe_I, nH2, U_H_I, data):
    """
    Compute continuum absorption at a single wavelength (JIT-compatible).

    Parameters
    ----------
    wavelength_cm : float
        Wavelength [cm]
    T : float
        Temperature [K]
    ne : float
        Electron density [cm⁻³]
    nH_I, nH_II, nHe_I, nH2 : float
        Species densities [cm⁻³]
    U_H_I : float
        H I partition function
    data : SynthesisData
        Pre-computed data

    Returns
    -------
    alpha : float
        Continuum absorption coefficient [cm⁻¹]
    """
    nu = c_cgs / wavelength_cm
    nH_I_div_U = nH_I / jnp.clip(U_H_I, 1e-10, jnp.inf)

    # Scattering
    alpha_rayleigh = _rayleigh_jit(nu, nH_I, nHe_I, nH2)
    alpha_electron = _electron_scattering_jit(ne)

    # H I free-free
    alpha_H_ff = _hydrogenic_ff_jit(nu, T, 1, nH_II, ne, data)

    # H⁻ bound-free and free-free — use accurate tabulated functions from continuum.py
    alpha_Hminus_bf = Hminus_bf(nu, T, nH_I_div_U, ne)
    alpha_Hminus_ff = Hminus_ff(nu, T, nH_I_div_U, ne)

    return alpha_rayleigh + alpha_electron + alpha_H_ff + alpha_Hminus_bf + alpha_Hminus_ff


def _voigt_jit(a, v):
    """
    Voigt-Hjerting function H(a, v) (JIT-compatible approximation).

    Uses a rational approximation valid for small a.
    """
    # For small a, use the Humlicek approximation
    z = v + 1j * a
    t = a - 1j * v

    # Region-based approximation (simplified)
    s = jnp.abs(v) + a

    # Approximation for different regions
    H = jnp.where(
        s >= 15,
        # Large |z|: asymptotic expansion
        a / (jnp.pi * (v**2 + a**2)),
        jnp.where(
            s >= 5.5,
            # Medium |z|
            a / jnp.pi * (1 / (v**2 + a**2) +
                         1.5 / (v**2 + a**2 + 1.5)),
            # Small |z|: more accurate approximation
            jnp.exp(-v**2) * (1 - a * 2 / jnp.sqrt(jnp.pi) *
                              jnp.where(jnp.abs(v) < 1e-6, 1.0, (1 - jnp.exp(-v**2)) / v))
        )
    )

    return jnp.clip(H, 0, 1e10)


def _line_profile_jit(wl_center, sigma_D, gamma_L, amplitude, wl):
    """
    Voigt line profile (JIT-compatible).

    Parameters
    ----------
    wl_center : float
        Line center wavelength [cm]
    sigma_D : float
        Doppler width [cm]
    gamma_L : float
        Lorentz HWHM [cm]
    amplitude : float
        Integrated absorption
    wl : float
        Wavelength at which to evaluate [cm]

    Returns
    -------
    alpha : float
        Absorption coefficient [cm⁻¹]
    """
    inv_sigma_sqrt2 = 1 / (sigma_D * jnp.sqrt(2) + 1e-30)
    a = gamma_L * inv_sigma_sqrt2
    v = jnp.abs(wl - wl_center) * inv_sigma_sqrt2

    H = _voigt_jit(a, v)
    return amplitude * inv_sigma_sqrt2 / jnp.sqrt(jnp.pi) * H


def _compute_number_densities_jit(T, n_total, ne, abundances, data):
    """
    Compute species number densities using Picard-iterated chemical equilibrium (JIT-compatible).

    Returns arrays indexed by (Z-1) for H I, H II, He I, H2, plus the self-consistent ne.
    """
    # Picard iteration for self-consistent electron density and neutral fractions
    ne_sol, neutral_fracs = chemical_equilibrium_jit(T, n_total, ne, abundances, data.chem_eq_data)

    # First-ionization weights at the self-consistent ne
    wII_ne1, _ = _compute_saha_weights_jit(T, 1.0, data.chem_eq_data)
    wII = wII_ne1 / jnp.clip(ne_sol, 1e-12, jnp.inf)

    # Atom number densities
    atom_n = abundances * (n_total - ne_sol)

    # Neutral densities
    n_neutral = atom_n * neutral_fracs

    # Ionized densities
    n_ion = atom_n * wII * neutral_fracs

    # H I, H II, He I
    nH_I = n_neutral[0]
    nH_II = n_ion[0]
    nHe_I = n_neutral[1]

    # H2 (very approximate - assume negligible for now)
    nH2 = 0.0

    # Partition function for H I (uses original CubicSpline knots — exact match to synthesize)
    log_T = jnp.log(T)
    U_H_I = _pf_orig_eval(
        log_T, data.chem_eq_data.pf_orig_t[0, 0], data.chem_eq_data.pf_orig_u[0, 0],
        data.chem_eq_data.pf_orig_h[0, 0], data.chem_eq_data.pf_orig_z[0, 0],
        data.chem_eq_data.pf_orig_n[0, 0],
    )

    return nH_I, nH_II, nHe_I, nH2, U_H_I, n_neutral, n_ion, neutral_fracs, ne_sol


def _pf_spline_eval(log_T, t_grid, h_grid, u_vals, z_vals):
    """Evaluate uniform-grid cubic spline at log_T. Used for molecular partition funcs."""
    log_T_c = jnp.clip(log_T, t_grid[0], t_grid[-1])
    i = jnp.clip(jnp.searchsorted(t_grid, log_T_c, side='right') - 1, 0, t_grid.shape[0] - 2)
    ti  = t_grid[i];   ti1 = t_grid[i + 1]
    ui  = u_vals[i];   ui1 = u_vals[i + 1]
    zi  = z_vals[i];   zi1 = z_vals[i + 1]
    hi1 = h_grid[i + 1]
    return (zi  * (ti1 - log_T_c)**3 / (6.0 * hi1)
            + zi1 * (log_T_c - ti )**3 / (6.0 * hi1)
            + (ui1 / hi1 - zi1 * hi1 / 6.0) * (log_T_c - ti )
            + (ui  / hi1 - zi  * hi1 / 6.0) * (ti1 - log_T_c))


def _pf_orig_eval(log_T, t_arr, u_arr, h_arr, z_arr, n_knots):
    """Evaluate original CubicSpline knots at log_T. Exactly matches CubicSpline.__call__.

    t_arr/u_arr/h_arr/z_arr are padded to 201 entries (inf in tail of t_arr).
    n_knots is the number of valid (unpadded) knots — may be a traced JAX integer.
    """
    t_max = t_arr[n_knots - 1]
    log_T_c = jnp.clip(log_T, t_arr[0], t_max)
    i = jnp.clip(jnp.searchsorted(t_arr, log_T_c, side='right') - 1, 0, n_knots - 2)
    ti  = t_arr[i];   ti1 = t_arr[i + 1]
    ui  = u_arr[i];   ui1 = u_arr[i + 1]
    zi  = z_arr[i];   zi1 = z_arr[i + 1]
    hi1 = h_arr[i + 1]
    return (zi  * (ti1 - log_T_c)**3 / (6.0 * hi1)
            + zi1 * (log_T_c - ti )**3 / (6.0 * hi1)
            + (ui1 / hi1 - zi1 * hi1 / 6.0) * (log_T_c - ti )
            + (ui  / hi1 - zi  * hi1 / 6.0) * (ti1 - log_T_c))


@jax.jit
def _compute_line_params_jit(
    T_layers, ne_all, nH_I_all, neutral_dens_final, ionized_dens_final,
    mol_densities_all, linelist_data, data, vmic_cm_s
):
    """Compute (amplitude, sigma_D, gamma_L) per (line, layer) without Voigt profiles.

    Used by synthesize_jit to compute max_win per line for exact bucketed line absorption.
    Returns three arrays each of shape (n_lines, n_layers).
    """
    pi_e2_mc = jnp.pi * electron_charge_cgs**2 / (electron_mass_cgs * c_cgs)

    def per_line_params(line_idx):
        wl_center     = linelist_data.wl[line_idx]
        log_gf_val    = linelist_data.log_gf[line_idx]
        Z             = linelist_data.species_Z[line_idx]
        charge        = linelist_data.species_charge[line_idx]
        E_lower       = linelist_data.E_lower[line_idx]
        gamma_rad_l   = linelist_data.gamma_rad[line_idx]
        gamma_stark_l = linelist_data.gamma_stark[line_idx]
        vdW_sigma_l   = linelist_data.vdW_sigma[line_idx]
        vdW_alpha_l   = linelist_data.vdW_alpha[line_idx]
        mass          = linelist_data.mass[line_idx]
        mol_idx       = linelist_data.mol_species_idx[line_idx]

        nu       = c_cgs / wl_center
        sigma_ln = pi_e2_mc * wl_center**2 / c_cgs

        n_mol_slots     = mol_densities_all.shape[1]
        safe_mol_idx    = jnp.clip(mol_idx, 0, n_mol_slots - 1)
        n_mols_real     = data.chem_eq_data.mol_partition_func_values.shape[0]
        safe_pf_mol_idx = jnp.clip(mol_idx, 0, jnp.maximum(n_mols_real - 1, 0))

        def per_layer_params(T_i, ne_i, nH_I_i, n_neutral_i, n_ion_i, mol_densities_i):
            n_atomic  = jnp.where(charge == 0, n_neutral_i[Z - 1], n_ion_i[Z - 1])
            n_mol     = mol_densities_i[safe_mol_idx]
            n_species = jnp.where(mol_idx >= 0, n_mol, n_atomic)
            log_T = jnp.log(T_i)
            U_atomic = _pf_orig_eval(
                log_T, data.chem_eq_data.pf_orig_t[Z - 1, charge],
                data.chem_eq_data.pf_orig_u[Z - 1, charge],
                data.chem_eq_data.pf_orig_h[Z - 1, charge],
                data.chem_eq_data.pf_orig_z[Z - 1, charge],
                data.chem_eq_data.pf_orig_n[Z - 1, charge],
            )
            U_mol = _pf_spline_eval(
                log_T, data.chem_eq_data.log_T_grid, data.chem_eq_data.log_T_h,
                data.chem_eq_data.mol_partition_func_values[safe_pf_mol_idx],
                data.chem_eq_data.mol_partition_func_z[safe_pf_mol_idx],
            )
            U = jnp.where(mol_idx >= 0, U_mol, U_atomic)
            sigma_D = wl_center * jnp.sqrt(kboltz_cgs * T_i / mass + vmic_cm_s**2 / 2.0) / c_cgs
            g_stark = gamma_stark_l * (T_i / 1e4)**(1.0 / 6.0) * ne_i
            g_vdW = jnp.where(
                vdW_alpha_l < 0.0,
                vdW_sigma_l * (T_i / 1e4)**0.3 * nH_I_i,
                2.0 * vdW_sigma_l * 1e6 * (T_i / 1e4)**(0.5 * (1.0 - vdW_alpha_l)) * nH_I_i
            )
            gamma_total = gamma_rad_l + g_stark + g_vdW
            gamma_L = gamma_total * wl_center**2 / (4.0 * jnp.pi * c_cgs)
            stim  = 1.0 - jnp.exp(-hplanck_eV * nu / (kboltz_eV * T_i))
            boltz = jnp.exp(-E_lower / (kboltz_eV * T_i))
            amplitude = (n_species / jnp.clip(U, 1e-10) *
                         10.0**log_gf_val * sigma_ln * boltz * stim)
            return amplitude, sigma_D, gamma_L

        return jax.vmap(per_layer_params)(
            T_layers, ne_all, nH_I_all, neutral_dens_final, ionized_dens_final,
            mol_densities_all
        )

    return jax.vmap(per_line_params)(jnp.arange(linelist_data.wl.shape[0]))


_voigt_profile_jax_jit = jax.jit(_voigt_profile_jax)


def synthesize_jit(
    wavelengths_cm: jnp.ndarray,
    T_layers: jnp.ndarray,
    n_total_layers: jnp.ndarray,
    ne_layers: jnp.ndarray,
    z_layers: jnp.ndarray,
    log_tau_ref: jnp.ndarray,
    abundances: jnp.ndarray,
    vmic_cm_s: float,
    data: SynthesisData,
    linelist_data: LinelistData,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Fully JIT-compatible spectral synthesis.

    Parameters
    ----------
    wavelengths_cm : array, shape (n_wl,)
        Wavelength grid [cm]
    T_layers : array, shape (n_layers,)
        Temperature at each layer [K]
    n_total_layers : array, shape (n_layers,)
        Total number density at each layer [cm⁻³]
    ne_layers : array, shape (n_layers,)
        Electron density at each layer [cm⁻³]
    z_layers : array, shape (n_layers,)
        Height coordinate at each layer [cm]
    log_tau_ref : array, shape (n_layers,)
        Log optical depth at reference wavelength
    abundances : array, shape (92,)
        Absolute abundances N(X)/N_total
    vmic_cm_s : float
        Microturbulent velocity [cm/s]
    data : SynthesisData
        Pre-computed synthesis data
    linelist_data : LinelistData
        Pre-processed linelist

    Returns
    -------
    flux : array, shape (n_wl,)
        Emergent flux [erg cm⁻² s⁻¹ cm⁻¹]
    continuum : array, shape (n_wl,)
        Continuum flux [erg cm⁻² s⁻¹ cm⁻¹]
    """
    n_layers = T_layers.shape[0]
    n_wl = wavelengths_cm.shape[0]
    lambda_ref_cm = 5e-5  # 5000 Å reference wavelength

    # ── Phase 1: Batch Picard iteration for ne and neutral fractions ─────────
    # Processes all layers at once; returns (n_layers,) and (n_layers, 92).
    ne_all, nf_picard = _chemical_equilibrium_batch_jit(
        T_layers, n_total_layers, ne_layers, abundances, data.chem_eq_data
    )

    # ── Phase 2: Molecular depletion correction (5 passes) ───────────────────
    # The Picard iteration doesn't account for atoms locked in molecules (e.g.
    # CO depletes 22% of C at 4500 K), causing n(C I) and hence n(C2) to be
    # overestimated.  We iterate: subtract molecular consumption from neutral
    # atom densities, recompute molecular densities, repeat until convergence.
    atom_dens = abundances[None, :] * (n_total_layers - ne_all)[:, None]   # (n_layers, 92)
    neutral_dens_picard = atom_dens * nf_picard                             # (n_layers, 92)
    M_consume = data.chem_eq_data.mol_atom_consume                          # (n_mols, 92)

    # Initial molecular densities (will be refined by correction loop)
    mol_dens = _compute_mol_densities_batch_jit(
        T_layers, n_total_layers, ne_all, abundances, nf_picard, data.chem_eq_data
    )  # (n_layers, n_mols)

    # 5 correction passes (unrolled at trace-time — XLA can fuse across passes).
    # Each pass: subtract molecular atom consumption from picard neutral densities →
    # recompute neutral fractions → recompute molecular densities.
    # neutral_dens_corr after the loop is picard - mol_{n-1} @ M (matching
    # chemical_equilibrium_all_layers which sets neutral_dens_all = neutral_dens_corrected
    # from the last loop body, while mol_dens is mol_n from the same body).
    neutral_dens_corr = neutral_dens_picard
    for _ in range(5):
        mol_atom_corr = mol_dens @ M_consume                                # (n_layers, 92)
        neutral_dens_corr = jnp.maximum(neutral_dens_picard - mol_atom_corr, 1e-99)
        nf_corr = neutral_dens_corr / jnp.maximum(atom_dens, 1e-99)
        mol_dens = _compute_mol_densities_batch_jit(
            T_layers, n_total_layers, ne_all, abundances, nf_corr, data.chem_eq_data
        )

    # neutral_dens_corr = picard - mol_{n-1} @ M  (matches chemical_equilibrium_all_layers)
    # mol_dens          = mol_n  (final molecular densities)
    neutral_dens_final = neutral_dens_corr
    wII_all, wIII_all = _compute_saha_weights_batch_jit(T_layers, ne_all, data.chem_eq_data)
    ionized_dens_final = wII_all * neutral_dens_final                       # (n_layers, 92)
    doubly_dens_final  = wIII_all * neutral_dens_final                      # (n_layers, 92)

    nH_I_all  = neutral_dens_final[:, 0]   # (n_layers,)
    nH_II_all = ionized_dens_final[:, 0]
    nHe_I_all = neutral_dens_final[:, 1]

    # Partition functions for H I and He I at each layer (original knots — exact match to synthesize)
    log_T_all = jnp.log(T_layers)
    U_H_I_all = jax.vmap(
        lambda lt: _pf_orig_eval(
            lt, data.chem_eq_data.pf_orig_t[0, 0], data.chem_eq_data.pf_orig_u[0, 0],
            data.chem_eq_data.pf_orig_h[0, 0], data.chem_eq_data.pf_orig_z[0, 0],
            data.chem_eq_data.pf_orig_n[0, 0],
        )
    )(log_T_all)  # (n_layers,)
    U_He_I_all = jax.vmap(
        lambda lt: _pf_orig_eval(
            lt, data.chem_eq_data.pf_orig_t[1, 0], data.chem_eq_data.pf_orig_u[1, 0],
            data.chem_eq_data.pf_orig_h[1, 0], data.chem_eq_data.pf_orig_z[1, 0],
            data.chem_eq_data.pf_orig_n[1, 0],
        )
    )(log_T_all)  # (n_layers,)

    # Pad mol_dens with a zero column so mol_species_idx == -1 can safely index it
    mol_densities_all = jnp.concatenate(
        [mol_dens, jnp.zeros((n_layers, 1))], axis=1
    )  # (n_layers, n_mols+1)

    # H2 density from molecular densities (index 50 in default mol list)
    nH2_all = mol_dens[:, _H2_MOL_IDX]  # (n_layers,)

    # Peach FF species: He_II (z=1), C_II (z=5), Si_II (z=13), Mg_II (z=11)
    _peach_z = jnp.array([z for z, _ in _PEACH_IDX])    # [1, 5, 13, 11]
    n_peach = ionized_dens_final[:, _peach_z]            # (n_layers, 4)

    # Z=1 FF: sum of all singly-ionized minus He II (index 1) — matches prepare_continuum_batch_fast
    n_Z1_ff = ionized_dens_final.sum(axis=1) - ionized_dens_final[:, 1]    # (n_layers,)

    # Z=2 FF: sum of all doubly-ionized
    n_Z2_ff = doubly_dens_final.sum(axis=1)                                 # (n_layers,)

    # Metal bound-free densities: select neutral/ionized/doubly column per species
    n_neutral_metal = neutral_dens_final[:, data.metal_bf_z_arr]   # (n_layers, n_metal)
    n_ionized_metal = ionized_dens_final[:, data.metal_bf_z_arr]   # (n_layers, n_metal)
    n_doubly_metal  = doubly_dens_final[:, data.metal_bf_z_arr]    # (n_layers, n_metal)
    metal_bf_dens = jnp.where(
        data.metal_bf_charge_arr[None, :] == 0, n_neutral_metal,
        jnp.where(data.metal_bf_charge_arr[None, :] == 1, n_ionized_metal, n_doubly_metal)
    )  # (n_layers, n_metal)

    # ── Phase 3: Continuum opacity + source function (batched over all layers) ─
    # Uses synthesize's coarse 1 Å grid + linear interpolation to match exactly.
    # synthesize computes continuum at cntm_step=1.0 Å grid, then interp1d(kind='linear').
    c_cgs_float = float(c_cgs)
    wl_min_cm = float(wavelengths_cm[0])
    wl_max_cm = float(wavelengths_cm[-1])
    cntm_step_cm = 1e-8  # 1.0 Å, matching synthesize default cntm_step
    cntm_wl_np = np.arange(wl_min_cm - cntm_step_cm, wl_max_cm + 2 * cntm_step_cm, cntm_step_cm)
    cntm_nu_np = c_cgs_float / cntm_wl_np  # (n_cntm,) decreasing frequencies

    alpha_cntm_coarse = _batch_continuum_vmap(
        jnp.array(cntm_nu_np), T_layers, ne_all, U_H_I_all, U_He_I_all,
        nH_I_all, nH_II_all, nHe_I_all, nH2_all,
        n_peach, n_Z1_ff, n_Z2_ff, metal_bf_dens,
        data.metal_bf_tables, data.metal_bf_nu_grid, data.metal_bf_logT_grid
    )  # (n_layers, n_cntm)

    # Interpolate to fine output grid (matches synthesize's interp1d linear)
    cntm_wl_jnp = jnp.array(cntm_wl_np)
    alpha_cntm_all = jax.vmap(
        lambda row: jnp.interp(wavelengths_cm, cntm_wl_jnp, row)
    )(alpha_cntm_coarse)  # (n_layers, n_wl)

    # Reference opacity at lambda_ref from coarse grid (matches synthesize's alpha_cntm_interp(lambda_ref))
    alpha_ref_all = jax.vmap(
        lambda row: jnp.interp(jnp.array([lambda_ref_cm]), cntm_wl_jnp, row)[0]
    )(alpha_cntm_coarse)  # (n_layers,)

    # Source function: Planck function per layer at all wavelengths
    S_all = jax.vmap(lambda T_i: blackbody(T_i, wavelengths_cm))(T_layers)  # (n_layers, n_wl)

    # ── Phase 4: Line absorption with exact bucketing (matches _line_absorption_fast) ──
    # Strategy: compute amplitude/sigma_D/gamma_L per (line, layer) via JIT, then
    # compute max_wins at Python level, bucket by window size, call the Voigt JIT
    # once per bucket.  This avoids allocating a single (n_lines, n_layers, W_MAX=512)
    # tensor for ALL lines and instead uses the actual required window per line.
    n_lines = linelist_data.wl.shape[0]

    if n_lines == 0:
        line_alpha = jnp.zeros((n_layers, n_wl))
    else:
        # Step 4a: params pass (no Voigt) — JIT-compiled, fast
        amp_jax, sigma_D_jax, gamma_L_jax = _compute_line_params_jit(
            T_layers, ne_all, nH_I_all, neutral_dens_final, ionized_dens_final,
            mol_densities_all, linelist_data, data, vmic_cm_s
        )  # each (n_lines, n_layers)

        # Step 4b: sync to numpy, compute max_wins per line
        amp_np     = np.asarray(amp_jax)      # (n_lines, n_layers) — sync point
        sigma_np   = np.asarray(sigma_D_jax)
        gamma_np   = np.asarray(gamma_L_jax)
        wl_np      = np.asarray(wavelengths_cm)
        wls_np     = np.asarray(linelist_data.wl)

        wl_spacing = float(np.median(np.diff(wl_np))) if n_wl > 1 else 5e-9

        # Interpolate continuum at line centers from the coarse grid (one interp step,
        # matching synthesize's alpha_cntm_interp(wl_centers) exactly)
        cntm_coarse_np = np.asarray(alpha_cntm_coarse)
        cntm_at_center = np.array(
            [np.interp(wls_np, cntm_wl_np, cntm_coarse_np[i]) for i in range(n_layers)]
        ).T  # (n_lines, n_layers)

        _CUTOFF = 3e-4
        rho_crit = _CUTOFF * cntm_at_center / np.maximum(np.abs(amp_np), 1e-300)
        sqrt2pi = np.sqrt(2.0 * np.pi)
        log_arg = sqrt2pi * sigma_np * rho_crit
        with np.errstate(invalid='ignore', divide='ignore'):
            win_G = np.where(log_arg >= 1.0, 0.0,
                             sigma_np * np.sqrt(-2.0 * np.log(np.maximum(log_arg, 1e-300))))
            win_L_arg = gamma_np / (np.pi * rho_crit)
            win_L = np.where(win_L_arg <= gamma_np**2, 0.0,
                             np.sqrt(np.maximum(win_L_arg - gamma_np**2, 0.0)))
        max_wins = np.max(np.sqrt(win_G**2 + win_L**2), axis=1)   # (n_lines,)
        max_wins_px = np.clip(
            np.ceil(2.0 * max_wins / wl_spacing + 2).astype(int), 0, n_wl
        )

        # Step 4c: bucketed Voigt — exactly as in _line_absorption_fast
        alpha_lines = np.zeros((n_layers, n_wl))
        BUCKET_WIDTHS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, n_wl]
        prev_W = 0
        for W_MAX in BUCKET_WIDTHS:
            W = min(W_MAX, n_wl)
            in_bucket = (max_wins_px > prev_W) & (max_wins_px <= W_MAX)
            n_b = int(in_bucket.sum())
            if n_b == 0:
                prev_W = W_MAX
                continue

            idx_b = np.where(in_bucket)[0]
            wls_b = wls_np[idx_b]

            # Window start: left edge placed at line_center - max_win (matches _line_absorption_fast)
            i_lo_b = np.searchsorted(wl_np, wls_b - max_wins[idx_b]).astype(int)
            i_lo_b = np.clip(i_lo_b, 0, n_wl - W)

            pix_idx = i_lo_b[:, None] + np.arange(W, dtype=int)[None, :]  # (n_b, W)
            wl_win  = wl_np[pix_idx]                                        # (n_b, W)
            delta_b = wl_win - wls_b[:, None]                               # (n_b, W)
            mask_b  = np.abs(delta_b) <= max_wins[idx_b, None]              # (n_b, W)

            # JAX Voigt — compiled once per unique (n_b, n_layers, W) shape
            delta_j = jnp.asarray(delta_b[:, None, :])             # (n_b, 1, W)
            sigma_j = jnp.asarray(sigma_np[idx_b, :, None])        # (n_b, n_layers, 1)
            gamma_j = jnp.asarray(gamma_np[idx_b, :, None])        # (n_b, n_layers, 1)
            profiles = np.asarray(
                _voigt_profile_jax_jit(delta_j, sigma_j, gamma_j)
            )  # (n_b, n_layers, W)

            contrib = mask_b[:, None, :] * amp_np[idx_b, :, None] * profiles  # (n_b, n_layers, W)

            for il, i_lo in enumerate(i_lo_b):
                alpha_lines[:, i_lo:i_lo + W] += contrib[il]

            prev_W = W_MAX

        line_alpha = jnp.asarray(alpha_lines)

    # ── Phase 4.5: Hydrogen line absorption (matches synthesize hydrogen_lines=True) ──
    _RYDBERG_CM = 1.0973731568539e5
    _H_LINE_WINDOW_CM = 150.0 * 1e-8   # synthesize's hydrogen_line_window_size default
    wl_min_cm = float(wavelengths_cm[0])
    wl_max_cm = float(wavelengths_cm[-1])
    nearby_stark = {
        k: v for k, v in hline_stark_profiles.items()
        if (wl_min_cm - _H_LINE_WINDOW_CM
            <= 1.0 / (_RYDBERG_CM * (1.0 / v.lower**2 - 1.0 / v.upper**2))
            <= wl_max_cm + _H_LINE_WINDOW_CM)
    }

    T_np     = np.asarray(T_layers)
    ne_np    = np.asarray(ne_all)
    nH_I_np  = np.asarray(nH_I_all)
    nHe_I_np = np.asarray(nHe_I_all)
    U_H_I_np = np.asarray(U_H_I_all)
    wl_cm_np = np.asarray(wavelengths_cm)

    ws_all_h = precompute_hummer_ws(T_np, nH_I_np, nHe_I_np, ne_np)

    h_alpha = np.zeros((n_layers, n_wl))
    if nearby_stark:
        h_alpha += hydrogen_line_absorption_stark_batched(
            wl_cm_np, T_np, ne_np, nH_I_np, U_H_I_np,
            _H_LINE_WINDOW_CM, float(vmic_cm_s), ws_all_h, nearby_stark
        )

    brackett_in_range = any(
        wl_min_cm - _H_LINE_WINDOW_CM
        <= 1.0 / (_RYDBERG_CM * (1.0 / 16.0 - 1.0 / m**2))
        <= wl_max_cm + _H_LINE_WINDOW_CM
        for m in range(5, 31)
    )
    if brackett_in_range:
        for i in range(n_layers):
            h_alpha[i] += hydrogen_line_absorption(
                wl_cm_np, T_np[i], ne_np[i], nH_I_np[i], nHe_I_np[i],
                float(U_H_I_np[i]), float(vmic_cm_s),
                _H_LINE_WINDOW_CM, use_MHD=True, ws=ws_all_h[i],
                stark_profiles={}
            )

    alpha_total = alpha_cntm_all + line_alpha + jnp.asarray(h_alpha)

    # ── Phase 5: Radiative transfer ───────────────────────────────────────────
    from .radiative_transfer import radiative_transfer_jit

    flux, _ = radiative_transfer_jit(
        alpha_total.T,
        S_all.T,
        z_layers,
        log_tau_ref,
        alpha_ref_all
    )

    flux_cntm, _ = radiative_transfer_jit(
        alpha_cntm_all.T,
        S_all.T,
        z_layers,
        log_tau_ref,
        alpha_ref_all
    )

    # Convert cm⁻¹ → Å⁻¹ (1 cm = 1e8 Å)
    return flux * 1e-8, flux_cntm * 1e-8
