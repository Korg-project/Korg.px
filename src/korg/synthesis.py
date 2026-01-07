"""
Spectral synthesis orchestration - the main user-facing API.

This module combines all components (chemical equilibrium, continuum absorption,
line profiles, and radiative transfer) to compute synthetic stellar spectra.

Reference: Korg.jl synthesize.jl
"""

import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Callable, Union
from scipy.interpolate import interp1d

from .atmosphere import PlanarAtmosphere, ShellAtmosphere
from .statmech import chemical_equilibrium
from .data_loader import (ionization_energies, default_partition_funcs,
                          default_log_equilibrium_constants)
from .constants import electron_mass_cgs, c_cgs, kboltz_eV, hplanck_eV, hplanck_cgs, kboltz_cgs
from .radiative_transfer import radiative_transfer
from .linelist import Line
from .species import Species
from .line_absorption import line_absorption
from .hydrogen_line_absorption import hydrogen_line_absorption
from .atomic_data import atomic_masses


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
    from .continuum_absorption.scattering import rayleigh, electron_scattering
    from .continuum_absorption.hydrogenic_bf_ff import hydrogenic_ff_absorption
    from .continuum_absorption.absorption_h_minus import Hminus_bf, Hminus_ff

    frequencies = c_cgs / wavelengths_cm
    n_wavelengths = len(wavelengths_cm)
    alpha_continuum = np.zeros(n_wavelengths)

    # Get number densities
    nH_I = number_densities.get(Species("H_I"), 0.0)
    nH_II = number_densities.get(Species("H_II"), 0.0)
    nHe_I = number_densities.get(Species("He_I"), 0.0)
    nH2 = number_densities.get(Species("H2_I"), 0.0)

    # Get H I partition function for H⁻ calculations
    U_H_I = partition_funcs[Species("H_I")](jnp.log(T))
    nH_I_div_partition = nH_I / U_H_I

    # Compute absorption for all wavelengths
    for j, (wl_cm, nu) in enumerate(zip(wavelengths_cm, frequencies)):
        # Scattering
        alpha_rayleigh = rayleigh(nu, nH_I, nHe_I, nH2)
        alpha_electron = electron_scattering(ne)

        # H I free-free
        alpha_H_I_ff = hydrogenic_ff_absorption(nu, T, 1, nH_II, ne)

        # H⁻ bound-free and free-free
        alpha_H_minus_bf = Hminus_bf(nu, T, nH_I_div_partition, ne)
        alpha_H_minus_ff = Hminus_ff(nu, T, nH_I_div_partition, ne)

        alpha_continuum[j] = (alpha_rayleigh + alpha_electron +
                              alpha_H_I_ff + alpha_H_minus_bf +
                              alpha_H_minus_ff)

    return alpha_continuum


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

    # Store chemical equilibrium results
    electron_densities = np.zeros(n_layers)
    alpha_ref = np.zeros(n_layers)  # Absorption at reference wavelength
    number_densities_list = []
    alpha_cntm_interps = []  # Continuum interpolators for each layer

    # Compute chemical equilibrium and continuum for each layer
    if verbose:
        print(f"Computing chemical equilibrium and continuum...")
    if profile:
        t_loop_start = time.time()
        t_chem_eq = 0.0
        t_cntm_abs = 0.0
        t_source_fn = 0.0

    for i in range(n_layers):
        T_i = T[i]
        ne_i = ne_model[i]
        n_i = n_total[i]

        # Chemical equilibrium
        if profile:
            t0 = time.time()
        ne_calc, n_dict = chemical_equilibrium(
            T_i, n_i, ne_i, abundances,
            ionization_energies_dict,
            partition_funcs,
            log_equilibrium_constants,
            electron_density_warn_threshold=1.0
        )
        if profile:
            t_chem_eq += time.time() - t0

        electron_densities[i] = ne_calc
        number_densities_list.append(n_dict)

        # Compute continuum at coarse grid
        if profile:
            t0 = time.time()
        alpha_cntm_coarse = compute_continuum_absorption(
            cntm_wavelengths_cm, T_i, ne_calc, n_dict, partition_funcs
        )
        if profile:
            t_cntm_abs += time.time() - t0

        # Create interpolator for this layer's continuum
        alpha_cntm_interp = interp1d(cntm_wavelengths_cm, alpha_cntm_coarse,
                                      kind='linear', fill_value='extrapolate')
        alpha_cntm_interps.append(alpha_cntm_interp)

        # Interpolate continuum to synthesis wavelengths
        alpha[i, :] = alpha_cntm_interp(wavelengths_cm)

        # Compute absorption at reference wavelength (5000 Å)
        alpha_ref[i] = alpha_cntm_interp(lambda_ref_cm)

        # Source function = Planck blackbody function B_λ(T)
        # Using wavelength-based Planck function to match Julia's convention
        if profile:
            t0 = time.time()
        source_function[:, i] = blackbody(T_i, wavelengths_cm)
        if profile:
            t_source_fn += time.time() - t0

    # Convert number densities from list of dicts to dict of arrays
    all_species = set()
    for n_dict in number_densities_list:
        all_species.update(n_dict.keys())
    number_densities = {
        spec: np.array([n_dict.get(spec, 0.0) for n_dict in number_densities_list])
        for spec in all_species
    }

    if profile:
        timings['layer_loop'] = time.time() - t_loop_start
        timings['chemical_equilibrium'] = t_chem_eq
        timings['continuum_absorption'] = t_cntm_abs
        timings['source_function'] = t_source_fn
        t0 = time.time()

    # Compute continuum flux if requested
    continuum_flux = None
    if return_continuum:
        if verbose:
            print(f"Computing continuum spectrum...")
        flux_cntm, _ = radiative_transfer(
            alpha.T,  # Transpose to (n_wavelengths, n_layers)
            source_function,
            spatial_coord,
            log_tau_ref,
            alpha_ref=alpha_ref,
            spherical=spherical,
            intensity_scheme="linear_flux_only",
            use_expint_flux=True
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
        for i in range(n_layers):
            T_i = T[i]
            ne_i = electron_densities[i]
            n_dict = number_densities_list[i]

            nH_I = n_dict.get(Species("H_I"), 0.0)
            nHe_I = n_dict.get(Species("He_I"), 0.0)
            U_H_I = partition_funcs[Species("H_I")](jnp.log(T_i))

            # Get vmic for this layer (if array) or use scalar
            xi = vmic_cm_s

            # Add hydrogen line absorption
            alpha_H = hydrogen_line_absorption(
                wavelengths_cm, T_i, ne_i, nH_I, nHe_I, U_H_I, xi,
                h_line_window_cm, use_MHD=True
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
        def continuum_opacity(wl_cm):
            """Return continuum opacity at all layers for a given wavelength."""
            return np.array([alpha_cntm_interps[i](wl_cm) for i in range(n_layers)])

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

    flux_nu, _ = radiative_transfer(
        alpha.T,  # Transpose to (n_wavelengths, n_layers)
        source_function,
        spatial_coord,
        log_tau_ref,
        alpha_ref=alpha_ref,
        spherical=spherical,
        intensity_scheme="linear_flux_only",
        use_expint_flux=True
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
                       _compute_saha_weights_jit, translational_U)


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

    return SynthesisData(
        chem_eq_data=chem_eq_data,
        gaunt_log_u_grid=gaunt_log_u_grid,
        gaunt_log_gamma2_grid=gaunt_log_gamma2_grid,
        gaunt_table=gaunt_table
    )


def preprocess_linelist(linelist: List[Line]) -> LinelistData:
    """
    Convert a list of Line objects to JAX-compatible arrays.

    Parameters
    ----------
    linelist : list of Line
        Standard linelist with Line objects

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
            mass=jnp.array([])
        )

    n_lines = len(linelist)
    wl = jnp.array([line.wl for line in linelist])
    log_gf = jnp.array([line.log_gf for line in linelist])

    # Extract species info
    species_Z = []
    species_charge = []
    masses = []
    for line in linelist:
        atoms = line.species.get_atoms()
        Z = int(atoms[0]) if len(atoms) > 0 else 1
        species_Z.append(Z)
        species_charge.append(line.species.charge)
        masses.append(line.species.get_mass())

    E_lower = jnp.array([line.E_lower for line in linelist])
    gamma_rad = jnp.array([line.gamma_rad for line in linelist])
    gamma_stark = jnp.array([line.gamma_stark for line in linelist])

    # van der Waals parameters
    vdW_sigma = jnp.array([line.vdW[0] for line in linelist])
    vdW_alpha = jnp.array([line.vdW[1] for line in linelist])

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
        mass=jnp.array(masses)
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

    # H⁻ bound-free and free-free
    alpha_Hminus_bf = _hminus_bf_jit(nu, T, nH_I_div_U, ne)
    alpha_Hminus_ff = _hminus_ff_jit(nu, T, nH_I_div_U, ne)

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
    Compute species number densities from Saha equation (JIT-compatible).

    Returns arrays indexed by (Z-1) for H I, H II, He I, H2.
    """
    # Get neutral fractions from Saha equation
    wII, wIII = _compute_saha_weights_jit(T, ne, data.chem_eq_data)

    neutral_fracs = 1.0 / (1.0 + wII + wIII)

    # Atom number densities
    atom_n = abundances * (n_total - ne)

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

    # Partition function for H I
    log_T = jnp.log(T)
    U_H_I = jnp.interp(log_T, data.chem_eq_data.log_T_grid,
                       data.chem_eq_data.partition_func_values[0, 0])

    return nH_I, nH_II, nHe_I, nH2, U_H_I, n_neutral, n_ion


@jax.jit
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

    # Compute number densities and continuum opacity for each layer
    def compute_layer(carry, layer_idx):
        i = layer_idx
        T_i = T_layers[i]
        n_i = n_total_layers[i]
        ne_i = ne_layers[i]

        # Get number densities
        nH_I, nH_II, nHe_I, nH2, U_H_I, n_neutral, n_ion = _compute_number_densities_jit(
            T_i, n_i, ne_i, abundances, data
        )

        # Compute continuum absorption at all wavelengths
        def compute_cntm_wl(wl):
            return _continuum_absorption_jit(wl, T_i, ne_i, nH_I, nH_II, nHe_I, nH2, U_H_I, data)

        alpha_cntm = jax.vmap(compute_cntm_wl)(wavelengths_cm)

        # Continuum at reference wavelength
        alpha_ref = _continuum_absorption_jit(lambda_ref_cm, T_i, ne_i, nH_I, nH_II, nHe_I, nH2, U_H_I, data)

        # Source function (Planck)
        S = blackbody(T_i, wavelengths_cm)

        return carry, (alpha_cntm, alpha_ref, S, ne_i, n_neutral, n_ion, nH_I, U_H_I)

    # Process all layers
    _, (alpha_cntm_all, alpha_ref_all, S_all, ne_calc, n_neutral_all, n_ion_all, nH_I_all, U_H_I_all) = jax.lax.scan(
        compute_layer, None, jnp.arange(n_layers)
    )

    # alpha_cntm_all: shape (n_layers, n_wl)
    # S_all: shape (n_layers, n_wl)

    # Add line absorption if there are lines
    n_lines = linelist_data.wl.shape[0]

    def add_line_absorption(alpha):
        """Add line absorption to continuum opacity."""
        # For each line, add its contribution to all layers and wavelengths
        def process_line(alpha, line_idx):
            wl_center = linelist_data.wl[line_idx]
            log_gf = linelist_data.log_gf[line_idx]
            Z = linelist_data.species_Z[line_idx]
            charge = linelist_data.species_charge[line_idx]
            E_lower = linelist_data.E_lower[line_idx]
            gamma_rad = linelist_data.gamma_rad[line_idx]
            gamma_stark = linelist_data.gamma_stark[line_idx]
            vdW_sigma = linelist_data.vdW_sigma[line_idx]
            vdW_alpha = linelist_data.vdW_alpha[line_idx]
            mass = linelist_data.mass[line_idx]

            def add_to_layer(alpha_layer, layer_idx):
                T_i = T_layers[layer_idx]
                ne_i = ne_calc[layer_idx]
                nH_I = nH_I_all[layer_idx]

                # Get species number density
                n_species = jnp.where(
                    charge == 0,
                    n_neutral_all[layer_idx, Z - 1],
                    n_ion_all[layer_idx, Z - 1]
                )

                # Get partition function
                log_T = jnp.log(T_i)
                U = jnp.interp(log_T, data.chem_eq_data.log_T_grid,
                              data.chem_eq_data.partition_func_values[Z - 1, charge])

                # Doppler width
                sigma_D = wl_center * jnp.sqrt(kboltz_cgs * T_i / mass + vmic_cm_s**2 / 2) / c_cgs

                # Lorentz width (damping)
                # Radiative + Stark + van der Waals
                gamma_stark_scaled = gamma_stark * (T_i / 10000)**0.166667
                gamma_vdW = jnp.where(
                    vdW_alpha == -1,
                    vdW_sigma * (T_i / 10000)**0.3,
                    vdW_sigma * 1e6 * (T_i / 10000)**0.4  # Simplified ABO
                )
                gamma_total = gamma_rad + gamma_stark_scaled * ne_i + gamma_vdW * nH_I

                # Convert to wavelength units
                gamma_L = gamma_total * wl_center**2 / (4 * jnp.pi * c_cgs)

                # Line strength
                # n * sigma = n/U * g * f * (πe²/mc) * exp(-E_lower/kT) * (1 - exp(-hν/kT))
                from .constants import electron_charge_cgs, electron_mass_cgs, hplanck_eV
                sigma_e = jnp.pi * electron_charge_cgs**2 / (electron_mass_cgs * c_cgs)
                nu = c_cgs / wl_center
                stim_correction = 1 - jnp.exp(-hplanck_eV * nu / (kboltz_eV * T_i))
                boltzmann = jnp.exp(-E_lower / (kboltz_eV * T_i))

                amplitude = (n_species / jnp.clip(U, 1e-10, jnp.inf) *
                            10**log_gf * sigma_e * boltzmann * stim_correction)

                # Compute profile at all wavelengths
                def profile_at_wl(wl):
                    return _line_profile_jit(wl_center, sigma_D, gamma_L, amplitude, wl)

                alpha_line = jax.vmap(profile_at_wl)(wavelengths_cm)

                return alpha_layer + alpha_line, None

            # Add line to all layers
            alpha_new, _ = jax.lax.scan(add_to_layer, alpha, jnp.arange(n_layers))
            return alpha_new, None

        # Process all lines
        alpha_final, _ = jax.lax.scan(process_line, alpha, jnp.arange(n_lines))
        return alpha_final

    # Add lines if present
    alpha_total = jnp.where(
        n_lines > 0,
        add_line_absorption(alpha_cntm_all),
        alpha_cntm_all
    )

    # Solve radiative transfer
    from .radiative_transfer import radiative_transfer_jit

    # Transpose to (n_wl, n_layers) for RT
    flux, _ = radiative_transfer_jit(
        alpha_total.T,
        S_all.T,
        z_layers,
        log_tau_ref,
        alpha_ref_all
    )

    # Also compute continuum flux
    flux_cntm, _ = radiative_transfer_jit(
        alpha_cntm_all.T,
        S_all.T,
        z_layers,
        log_tau_ref,
        alpha_ref_all
    )

    return flux, flux_cntm
