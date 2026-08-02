"""
Spectral synthesis orchestration - the main user-facing API.

This module combines all components (chemical equilibrium, continuum absorption,
line profiles, and radiative transfer) to compute synthetic stellar spectra.

Reference: Korg.jl synthesize.jl
"""

import functools
import warnings

import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Callable, Union
from scipy.interpolate import interp1d

from .atmosphere import PlanarAtmosphere, ShellAtmosphere
from .statmech import (chemical_equilibrium_fast, chemical_equilibrium_all_layers,
                       precompute_chemical_equilibrium_data)
from .data_loader import (ionization_energies, default_partition_funcs,
                          default_log_equilibrium_constants,
                          default_chem_eq_data, default_mol_species)
from .continuum import (prepare_continuum_batch_fast, batch_continuum_absorption,
                        Hminus_bf, Hminus_ff)
from .constants import (electron_mass_cgs, electron_charge_cgs, c_cgs,
                        kboltz_eV, hplanck_eV, hplanck_cgs, kboltz_cgs,
                        bohr_radius_cgs)
from .radiative_transfer import (radiative_transfer, radiative_transfer_jit,
                                 radiative_transfer_spherical)
from .linelist import Line
from .species import Species
from .line_absorption import line_absorption, _vdW_to_tuple, _voigt_profile_jax
from .hydrogen_line_absorption import (hydrogen_line_absorption, precompute_hummer_ws,
                                       hline_stark_profiles,
                                       hydrogen_line_absorption_stark_batched)
from .atomic_data import atomic_masses, atomic_symbols
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
    # Korg.jl's SynthesisResult also carries the absorption coefficient, the
    # per-species number densities and the electron number density.  They are
    # needed by prune_linelist (and are part of Korg.jl's public result), so
    # expose them here under Korg.jl's names.  They default to None so that
    # existing constructions of SynthesisResult keep working unchanged.
    alpha: Optional[np.ndarray] = None                     # (n_layers, n_wl), cm⁻¹
    alpha_cntm: Optional[np.ndarray] = None                # continuum-only alpha
    number_densities: Optional[dict] = None                # Species -> (n_layers,) cm⁻³
    electron_number_density: Optional[np.ndarray] = None   # (n_layers,) cm⁻³

    @property
    def cntm(self):
        """Alias for ``continuum``, matching Korg.jl's field name."""
        return self.continuum


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
    # Julia: when use_internal_reference_linelist=true and ref wl is 5000 Å,
    # always use the built-in list unconditionally (user linelist is ignored).
    if use_internal_reference_linelist and reference_wavelength_cm == 5e-5:
        from .data_loader import load_default_linelist
        try:
            return load_default_linelist(reference_wavelength_cm)
        except Exception:
            pass

    # For non-5000 Å references (or when built-in is disabled), use the user's linelist.
    window_cm = np.array([reference_wavelength_cm, reference_wavelength_cm])
    buffer_cm = 21e-8  # 21 Å in cm
    filtered = filter_linelist(linelist, window_cm, buffer_cm, warn_empty=False)

    if reference_wavelength_cm != 5e-5:
        if len(filtered) == 0:
            raise ValueError(
                f"The provided linelist contains no lines near the reference wavelength "
                f"{reference_wavelength_cm * 1e8:.1f} Å. Korg has a built-in fallback "
                f"only for 5000 Å (the MARCS default)."
            )
        # Korg.jl evaluates `filtered_linelist` here without returning it, so it
        # falls through into the 5000 Å merge below even for a non-5000 Å
        # reference.  That is an upstream slip (merging a 5000 Å fallback list
        # into a, say, 8000 Å reference is meaningless), so it is not
        # replicated: the user's lines are returned as-is.
        return filtered

    # 5000 Å reference with the internal list disabled.  Korg.jl still uses the
    # built-in list to fill in wherever the user's lines do not reach across
    # 5000 Å, so that alpha_5000 is never computed from a one-sided linelist.
    def _default_5000_linelist():
        from .data_loader import load_default_linelist
        try:
            return load_default_linelist(5e-5)
        except Exception:
            return []

    if len(filtered) == 0:
        return _default_5000_linelist()
    if filtered[0].wl > 5e-5:
        # user lines all sit redward of 5000 Å: prepend the built-in lines below them
        return [l for l in _default_5000_linelist()
                if l.wl < filtered[0].wl] + list(filtered)
    if filtered[-1].wl < 5e-5:
        # user lines all sit blueward of 5000 Å: append the built-in lines above them
        return list(filtered) + [l for l in _default_5000_linelist()
                                 if l.wl > filtered[-1].wl]
    # the user's lines span 5000 Å: they are sufficient on their own
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
    mu_values: int = 20,
    partition_funcs: Optional[Dict] = None,
    ionization_energies_dict: Optional[Dict] = None,
    log_equilibrium_constants: Optional[Dict] = None,
    verbose: bool = True,
    profile: bool = False,
):
    """
    Compute synthetic spectrum with lines.

    .. deprecated::
        Use :func:`synthesize_jit` instead. This implementation cannot be traced by
        ``jax.jit``: it orchestrates jitted kernels from Python and drops to host NumPy
        in places, so it cannot be placed on a GPU as one kernel, ``vmap``ped over
        stellar parameters, or differentiated end to end — the properties this package
        exists to provide. It is retained until ``synthesize_jit`` covers the same
        options, and will then be removed.

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
    mu_values : int, optional
        Number of Gauss-Legendre μ quadrature points used for the surface flux
        integral in *spherical* geometry (default: 20, matching Korg.jl's
        ``mu_values``).  Ignored for plane-parallel atmospheres, where the
        anchored/``linear_flux_only`` combination uses the exponential-integral
        shortcut and needs no μ grid — exactly as in Korg.jl.
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
    warnings.warn(
        "synthesize_spectrum() is deprecated and will be removed; it cannot be "
        "jit-compiled or differentiated. Use synthesize_jit() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    import time
    timings = {} if profile else None
    t_start = time.time() if profile else None

    # ── Input validation (ports the argument checks in Korg.jl's synthesize) ──
    wavelengths_angstrom = np.asarray(wavelengths_angstrom, dtype=np.float64)
    if wavelengths_angstrom.ndim != 1 or wavelengths_angstrom.size == 0:
        raise ValueError(
            "wavelengths_angstrom must be a non-empty 1-D array, got shape "
            f"{wavelengths_angstrom.shape}"
        )
    # Korg.jl: the Rayleigh scattering cross-sections are not valid below 1300 Å.
    if wavelengths_angstrom[0] < 1300.0:
        raise ValueError(
            f"Requested wavelength range starts at {wavelengths_angstrom[0]:.1f} Å, "
            "blueward of 1300 Å, the lowest allowed wavelength (a limitation of the "
            "Rayleigh scattering calculation)."
        )

    abundances = np.asarray(abundances, dtype=np.float64)
    if abundances.ndim != 1 or abundances.shape[0] != 92:
        raise ValueError(
            "abundances must be a 92-element 1-D array (one entry per element "
            f"H..U), got shape {abundances.shape}"
        )
    # Korg.jl: "A(H) must be a 92-element vector with A[1] == 12."  Absolute
    # number fractions (which sum to 1) are also accepted here and are
    # distinguished by A(H) <= 1.
    if abundances[0] > 1.0 and abundances[0] != 12.0:
        raise ValueError(
            "abundances look like A(X) = log10(N_X/N_H) + 12 but A(H) is "
            f"{abundances[0]}, not 12.0. Pass either a 92-element A(X) vector "
            "with A(H) == 12, or absolute number fractions summing to 1."
        )

    if len(atmosphere) == 0:
        raise ValueError("atmosphere has no layers")

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
        # Korg.jl: radii = [atm.R + l.z for l in atm.layers]
        spatial_coord = atmosphere.r
        spherical = True
        R_photosphere = atmosphere.R_photosphere
    else:
        spatial_coord = atmosphere.z
        spherical = False
        R_photosphere = None

    def _solve_rt(alpha_layers_wl):
        """
        Emergent flux for an (n_layers, n_wl) opacity grid.

        Mirrors Korg.jl's dispatch in ``RadiativeTransfer.radiative_transfer``:
        plane-parallel + anchored τ + ``linear_flux_only`` takes the
        exponential-integral shortcut; a shell atmosphere goes through the ray
        solver and has its flux rescaled from the outermost radius to the
        photospheric radius by ``(r[1]/R)²``.
        """
        if spherical:
            flux, _ = radiative_transfer_spherical(
                alpha_layers_wl.T, source_function, spatial_coord, log_tau_ref,
                alpha_ref, n_mu=mu_values,
                tau_scheme="anchored", intensity_scheme="linear_flux_only",
                R_photosphere=R_photosphere,
            )
            return flux
        if using_defaults:
            flux, _ = radiative_transfer_jit(
                jnp.asarray(alpha_layers_wl.T), jnp.asarray(source_function),
                jnp.asarray(spatial_coord), jnp.asarray(log_tau_ref),
                jnp.asarray(alpha_ref)
            )
            return flux
        flux, _ = radiative_transfer(
            alpha_layers_wl.T, source_function, spatial_coord, log_tau_ref,
            alpha_ref=alpha_ref, spherical=False,
            intensity_scheme="linear_flux_only", use_expint_flux=True
        )
        return flux

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

    # Convert A(X) format abundances to linear number fractions for chemical equilibrium.
    # Julia's format_A_X() returns A(X) = log10(N_X/N_H) + 12; the solver
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
    # Both paths use the same batched solver; they differ only in whether the
    # temperature-gridded coefficients are the cached default tables or built here from
    # caller-supplied partition functions and equilibrium constants.
    if using_defaults:
        chem_eq_data, mol_species = default_chem_eq_data, default_mol_species
    else:
        chem_eq_data = precompute_chemical_equilibrium_data(
            ionization_energies_dict, partition_funcs, log_equilibrium_constants
        )
        mol_species = list(log_equilibrium_constants.keys())

    # Batch all layers at once: single XLA dispatch, avoids 56× dict overhead
    electron_densities, number_densities_batch, raw_arrays_list = \
        chemical_equilibrium_all_layers(
            T, n_total, ne_model, abs_abundances, chem_eq_data, mol_species
        )

    if using_defaults:
        number_densities_list = None  # fast path consumes the batched arrays directly
    else:
        # The custom-data continuum path below still wants one dict per layer.
        number_densities_list = [
            {spec: float(arr[i]) for spec, arr in number_densities_batch.items()}
            for i in range(n_layers)
        ]
    if profile:
        t_chem_eq = time.time() - t0

    # Pass 2: continuum absorption — batch all layers at once
    if profile:
        t0 = time.time()
    # alpha_ref is the continuum opacity *at* the reference wavelength.  Korg.jl
    # evaluates it directly there:
    #   α_ref[i] = total_continuum_absorption([c/λ_ref], layer.temp, nₑ, ...)
    # It must NOT be obtained by extrapolating the synthesis window's coarse
    # continuum grid: for any window that does not contain 5000 Å, a linear
    # extrapolation over tens or hundreds of Å drives the opacity negative,
    # which makes the anchored optical depth negative and the flux NaN.
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
        alpha_ref[:] = np.array(batch_continuum_absorption(
            jnp.asarray(np.array([c_cgs / lambda_ref_cm])),
            jnp.asarray(T, dtype=np.float64),
            jnp.asarray(electron_densities, dtype=np.float64),
            batch
        ))[:, 0]
        for i in range(n_layers):
            alpha_cntm_interp = interp1d(cntm_wavelengths_cm, alpha_cntm_all[i],
                                          kind='linear', fill_value='extrapolate')
            alpha_cntm_interps.append(alpha_cntm_interp)
            alpha[i, :] = alpha_cntm_interp(wavelengths_cm)
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
            alpha_ref[i] = compute_continuum_absorption(
                np.array([lambda_ref_cm]), T[i], electron_densities[i],
                number_densities_list[i], partition_funcs
            )[0]
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

    # Add line absorption at reference wavelength to alpha_ref (matching Julia).
    # Julia's alpha_ref = continuum + line opacity at 5000 Å, which correctly accounts
    # for the total opacity that determines the MARCS depth scale.
    ref_ll_for_ref = get_reference_wavelength_linelist(linelist, lambda_ref_cm)
    if ref_ll_for_ref:
        # Korg.jl: `α_cntm_ref = [_ -> a for a in copy(α_ref)]`, i.e. a constant
        # function per layer returning the continuum opacity already computed at
        # the reference wavelength.  Re-interpolating the synthesis window's grid
        # here would reintroduce the extrapolation problem fixed above.
        _alpha_ref_cntm = alpha_ref.copy()

        def _cntm_at_ref_wl(wl_cm):
            n_out = np.size(wl_cm)
            result = np.repeat(_alpha_ref_cntm[None, :], n_out, axis=0)  # (n_wl, n_layers)
            return result if np.ndim(wl_cm) > 0 else result[0]
        line_at_ref = line_absorption(
            ref_ll_for_ref, np.array([lambda_ref_cm]),
            T, electron_densities, number_densities,
            partition_funcs, vmic_cm_s, _cntm_at_ref_wl,
            cutoff_threshold=line_cutoff_threshold,
        )  # (n_layers, 1)
        alpha_ref += np.asarray(line_at_ref[:, 0])

    # Save continuum-only alpha before H-lines/atomic lines are added.
    # The continuum RT is computed later (after the correct alpha_ref is known) so that
    # both continuum and total RT use the same tau scale.
    alpha_cntm_only = alpha.copy()  # (n_layers, n_wl), continuum opacity only
    continuum_flux = None

    if profile:
        timings['continuum_rt'] = 0.0  # measured later
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

        # Batch-precompute occupation probabilities for all layers (much faster
        # than per-layer).  chemical_equilibrium_all_layers always returns one
        # raw-array dict per layer, and an atmosphere with zero layers is
        # rejected above, so this list is never empty.  (The dead `else` branch
        # that used to stand here read `number_densities_list`, which is None on
        # the default-data path, so it would have raised had it ever run.)
        nH_I_arr = np.array([ra['neutral_dens'][0] for ra in raw_arrays_list])
        nHe_I_arr = np.array([ra['neutral_dens'][1] for ra in raw_arrays_list])
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
            # Korg.jl: use_MHD_for_hydrogen_lines defaults to wls[end] < 13,000 Å.
            # The MHD occupation-probability formalism is not applied to the
            # infrared series, which is exactly the regime this branch covers.
            use_MHD = wl_max_cm < 13_000e-8
            for i in range(n_layers):
                alpha_H = hydrogen_line_absorption(
                    wavelengths_cm, T[i], electron_densities[i], nH_I_arr[i], nHe_I_arr[i],
                    float(U_H_I_arr[i]), vmic_cm_s,
                    h_line_window_cm, use_MHD=use_MHD, ws=ws_all[i],
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

    # alpha_ref was set at line 527 (continuum + reference-linelist lines, NO H-lines),
    # matching Julia's alpha_5 = cntm + synthesis_linelist_lines.  Do not overwrite it here.

    # Compute continuum flux if requested (using correct alpha_ref so tau scale is consistent)
    if return_continuum:
        if verbose:
            print(f"Computing continuum spectrum...")
        if profile:
            t_cntm_rt = time.time()
        flux_cntm = _solve_rt(alpha_cntm_only)
        continuum_flux = flux_cntm * 1e-8
        if profile:
            timings['continuum_rt'] = time.time() - t_cntm_rt

    # Solve radiative transfer with full opacity
    if verbose:
        print(f"Solving radiative transfer...")
    if profile:
        t_rt_start = time.time()

    flux_nu = _solve_rt(alpha)

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
        intensities=None,
        alpha=alpha,
        alpha_cntm=alpha_cntm_only,
        number_densities=number_densities,
        electron_number_density=electron_densities,
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
    mu_values: int = 20,
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
    mu_values : int, optional
        Number of μ quadrature points for the spherical surface-flux integral
        (default: 20, matching Korg.jl).  Unused for planar atmospheres.
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
        mu_values=mu_values,
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
                       picard_chemical_equilibrium_guess, MAX_ATOMIC_NUMBER,
                       _compute_saha_weights_jit, translational_U,
                       _compute_mol_densities_jit,
                       _picard_chemical_equilibrium_guess_batch,
                       _compute_mol_densities_batch_jit,
                       _compute_saha_weights_batch_jit,
                       _chem_eq_newton_batch_jit,
                       _chem_eq_newton_scan_jit)
from .continuum import (_batch_continuum_vmap, _get_metal_bf_idx,
                        get_metal_bf_cross_sections, _PEACH_IDX, _H2_MOL_IDX)

# Effective vdW perturber density correction factors (matching Julia Korg.jl).
# Polarizabilities in atomic units from https://doi.org/10.1080/00268976.2018.1535143
_VDW_C_HE = (1.38375 / 4.50711)**0.4 * (4.002602 / 1.008)**-0.3  # He relative to H
_VDW_C_H2 = (5.503   / 4.50711)**0.4 * 2.0**-0.3                  # H2 relative to H


class BucketGeometry(NamedTuple):
    """Per-bucket precomputed window geometry for fast Voigt + scatter."""
    W: int                      # pixel window width (static for JIT)
    n_b: int                    # number of lines in this bucket
    amp_idx: np.ndarray         # (n_b,) indices into full linelist
    i_lo: jnp.ndarray           # (n_b,) starting pixels (centered windows)
    max_wins: jnp.ndarray       # (n_b,) half-widths [cm]
    wls: jnp.ndarray            # (n_b,) line center wavelengths [cm]


class HStarkPrecomputed(NamedTuple):
    """Precomputed per-layer data for fast H-line Stark + ABO absorption.

    Splitting the 3D (T, ne, log_delta_nu) Stark table interpolation into a
    precomputed bilinear (T, ne) step and a cheap per-call 1D step reduces the
    per-call XLA time from ~0.40 ms to ~0.10 ms for a single H transition.
    """
    profiles_1d: jnp.ndarray      # (n_layers, n_delta) bilinear-reduced Stark profile
    lambda0: jnp.ndarray          # (n_layers,) interpolated line centre [cm]
    lambda0_stehle: jnp.ndarray   # (n_layers,) Stark profile centre (const for Balmer)
    F0: jnp.ndarray               # (n_layers,) = 1.25e-9 * ne^(2/3)
    log_delta_nu_grid: jnp.ndarray  # (n_delta,)
    valid_mask: jnp.ndarray       # (n_layers,) bool — False if T/ne outside table grid
    window_pix: jnp.ndarray       # (n_win,) int32 pixel indices in synthesis grid
    window_cm: float              # half-window [cm]
    lower: int                    # lower quantum number (static for JIT)
    upper: int                    # upper quantum number (static for JIT)
    log_gf: float
    sigma_abo: float              # ABO sigma [cm^2]
    alpha_abo: float              # ABO alpha
    abo_active: float             # 1.0 if ABO active, 0.0 otherwise
    xi: float                     # microturbulence [cm/s]


class LinelistData(NamedTuple):
    """
    Linelist data stored as JAX-compatible arrays.

    All arrays have shape (n_lines,) unless otherwise noted.

    The optional ``wl_np_cached``, ``wls_np_cached``, ``wl_spacing_cached``, and ``cntm_wl_np_cached``
    fields cache wavelength-grid-derived numpy arrays used in ``synthesize_jit``
    to avoid recomputing them on every call.  They are populated when
    ``wavelengths_cm`` is passed to :func:`preprocess_linelist`, and are
    ``None`` otherwise (in which case ``synthesize_jit`` computes them lazily).
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
    # --- Cached wavelength-grid-derived arrays (optional) ---
    wl_np_cached: Optional[np.ndarray] = None      # wavelengths_cm as numpy float64, shape (n_wl,)
    wls_np_cached: Optional[np.ndarray] = None     # linelist wl as numpy float64, shape (n_lines,)
    wl_spacing_cached: Optional[float] = None      # median pixel spacing [cm]
    cntm_wl_np_cached: Optional[np.ndarray] = None  # coarse continuum wavelength grid [cm]


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


class PrecomputedAtmosphereData(NamedTuple):
    """
    Pre-computed atmosphere-dependent quantities for fast synthesis.

    Call precompute_atmosphere() once per (atmosphere, abundances, vmic, wavelengths)
    combination. Pass to synthesize_jit as precomputed_atm=... to skip the ~28 ms
    chemical-equilibrium and continuum phases.
    """
    # Atmosphere arrays (needed by line params and RT)
    T_layers: jnp.ndarray         # (n_layers,)
    ne_all: jnp.ndarray           # (n_layers,)
    z_layers: jnp.ndarray         # (n_layers,)
    log_tau_ref: jnp.ndarray      # (n_layers,)
    vmic_cm_s: float

    # Chemical equilibrium results
    nf_sol: jnp.ndarray           # (n_layers, 92) neutral fractions
    neutral_dens: jnp.ndarray     # (n_layers, 92) neutral number densities
    ionized_dens: jnp.ndarray     # (n_layers, 92) ionized number densities
    mol_dens_padded: jnp.ndarray  # (n_layers, n_mols+1) mol densities + zero column
    nH_I_all: jnp.ndarray         # (n_layers,)
    nH_II_all: jnp.ndarray        # (n_layers,)
    nHe_I_all: jnp.ndarray        # (n_layers,)
    U_H_I_all: jnp.ndarray        # (n_layers,)
    n_eff_vdW_all: jnp.ndarray    # (n_layers,) effective vdW perturber density = nH_I + c_He*nHe_I + c_H2*nH2

    # Continuum (already interpolated to fine wavelength grid)
    alpha_cntm_all: jnp.ndarray   # (n_layers, n_wl) fine-grid continuum opacity
    alpha_ref_all: jnp.ndarray    # (n_layers,) continuum+line opacity at lambda_ref

    # Coarse continuum grid (for line-center window-size computation)
    alpha_cntm_coarse: jnp.ndarray    # (n_layers, n_cntm) coarse continuum opacity
    cntm_wl_np: np.ndarray            # (n_cntm,) coarse wavelength grid [cm] as numpy array

    # Source function
    S_all: jnp.ndarray            # (n_layers, n_wl) Planck function

    # Pre-computed Hummer broadening widths for H-line absorption (per-layer, NOT per-line)
    ws_all_h: object = None       # list of per-layer dicts, or None

    # Fixed Voigt window width (pixels) for mega-JIT line path (0 = not computed yet)
    W_line: int = 0

    # Per-bucket geometry for fast bucket-JIT line absorption (None = not precomputed)
    bucket_geometry: tuple = None  # tuple of BucketGeometry

    # Cached JAX wavelength grid (avoids jnp.array(wl_np) on every synthesize_jit call)
    wl_jax: object = None         # jnp.ndarray (n_wl,) or None

    # Cached numpy conversions of atmospheric arrays (avoids repeated device→host copies)
    T_np: object = None           # np.ndarray (n_layers,)
    ne_np: object = None          # np.ndarray (n_layers,)
    nH_I_np: object = None        # np.ndarray (n_layers,)
    nHe_I_np: object = None       # np.ndarray (n_layers,)
    U_H_I_np: object = None       # np.ndarray (n_layers,)
    wl_cm_np: object = None       # np.ndarray (n_wl,)

    # Precomputed H-line Stark profiles (one HStarkPrecomputed per in-range transition)
    h_stark_precomp: tuple = None  # tuple of HStarkPrecomputed, or None

    # Precomputed partition function tables (avoids per-(line,layer) spline eval in line_params)
    U_atomic_table: object = None  # jnp.ndarray (92, 2, n_layers) — U[Z-1, charge, layer]
    U_mol_table: object = None     # jnp.ndarray (n_mol, n_layers) — U[mol_idx, layer]


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

    # Load Gaunt factor table for free-free absorption.  The import belongs
    # inside the try: the fallback below exists precisely for the case where
    # the tabulated data cannot be obtained, and an ImportError is one way for
    # that to happen.
    try:
        from .continuum_absorption.hydrogenic_bf_ff import _load_gauntff_table
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


@jax.jit
def _compute_U_atomic_table_jit(log_T, pf_t, pf_u, pf_h, pf_z, pf_n):
    """Compute partition functions for all (Z, charge, layer) combinations."""
    return jax.vmap(
        lambda z_idx: jax.vmap(
            lambda ch: jax.vmap(
                lambda lt: _pf_orig_eval(
                    lt, pf_t[z_idx, ch], pf_u[z_idx, ch],
                    pf_h[z_idx, ch], pf_z[z_idx, ch], pf_n[z_idx, ch]
                )
            )(log_T)
        )(jnp.arange(2))
    )(jnp.arange(92))


@jax.jit
def _compute_U_mol_table_jit(log_T, log_T_grid, log_T_h, mol_pf_values, mol_pf_z):
    """Compute molecular partition functions for all (mol_species, layer) combinations."""
    return jax.vmap(
        lambda mol_idx: jax.vmap(
            lambda lt: _pf_spline_eval(lt, log_T_grid, log_T_h, mol_pf_values[mol_idx], mol_pf_z[mol_idx])
        )(log_T)
    )(jnp.arange(mol_pf_values.shape[0]))


def precompute_atmosphere(
    wavelengths_cm,
    T_layers,
    n_total_layers,
    ne_layers,
    z_layers,
    log_tau_ref,
    abundances,
    vmic_cm_s,
    data: SynthesisData,
    linelist_data,
) -> PrecomputedAtmosphereData:
    """
    Pre-compute all atmosphere-dependent quantities for fast synthesis.

    Call this once whenever the atmosphere model or abundances change.
    Pass the result to synthesize_jit(precomputed_atm=...) to skip
    chemical equilibrium and continuum computation (~28 ms savings).

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
        Pre-processed linelist (used for cached wavelength grids)

    Returns
    -------
    PrecomputedAtmosphereData
        All atmosphere-dependent quantities ready for fast synthesis.
    """
    import jax
    n_layers = T_layers.shape[0]
    lambda_ref_cm = 5e-5  # 5000 Å reference wavelength
    c_cgs_float = float(c_cgs)

    # ── Phase 1: Picard initial guess ────────────────────────────────────────
    ne_init, nf_init = _picard_chemical_equilibrium_guess_batch(
        T_layers, n_total_layers, ne_layers, abundances, data.chem_eq_data
    )

    # ── Phase 2: Newton solver ────────────────────────────────────────────────
    ne_all, nf_sol = _chem_eq_newton_batch_jit(
        T_layers, n_total_layers, ne_init, nf_init, abundances, data.chem_eq_data
    )

    # Derive number densities from Newton solution
    atom_dens_all      = abundances[None, :] * (n_total_layers - ne_all)[:, None]
    neutral_dens_final = atom_dens_all * nf_sol
    wII_all, wIII_all  = _compute_saha_weights_batch_jit(T_layers, ne_all, data.chem_eq_data)
    ionized_dens_final = wII_all * neutral_dens_final
    doubly_dens_final  = wIII_all * neutral_dens_final

    mol_dens = _compute_mol_densities_batch_jit(
        T_layers, n_total_layers, ne_all, abundances, nf_sol, data.chem_eq_data
    )  # (n_layers, n_mols)

    nH_I_all  = neutral_dens_final[:, 0]
    nH_II_all = ionized_dens_final[:, 0]
    nHe_I_all = neutral_dens_final[:, 1]

    # Partition functions for H I and He I
    log_T_all = jnp.log(T_layers)
    U_H_I_all = jax.vmap(
        lambda lt: _pf_orig_eval(
            lt, data.chem_eq_data.pf_orig_t[0, 0], data.chem_eq_data.pf_orig_u[0, 0],
            data.chem_eq_data.pf_orig_h[0, 0], data.chem_eq_data.pf_orig_z[0, 0],
            data.chem_eq_data.pf_orig_n[0, 0],
        )
    )(log_T_all)
    U_He_I_all = jax.vmap(
        lambda lt: _pf_orig_eval(
            lt, data.chem_eq_data.pf_orig_t[1, 0], data.chem_eq_data.pf_orig_u[1, 0],
            data.chem_eq_data.pf_orig_h[1, 0], data.chem_eq_data.pf_orig_z[1, 0],
            data.chem_eq_data.pf_orig_n[1, 0],
        )
    )(log_T_all)

    # Pad mol_dens with a zero column
    mol_densities_all = jnp.concatenate(
        [mol_dens, jnp.zeros((n_layers, 1))], axis=1
    )  # (n_layers, n_mols+1)

    nH2_all = mol_dens[:, _H2_MOL_IDX]
    n_eff_vdW_all = nH_I_all  # currently matches HJXcT Korg.jl (H-only perturbers)

    _peach_z = jnp.array([z for z, _ in _PEACH_IDX])
    n_peach = ionized_dens_final[:, _peach_z]
    n_Z1_ff = ionized_dens_final.sum(axis=1) - ionized_dens_final[:, _peach_z].sum(axis=1)
    n_Z2_ff = doubly_dens_final.sum(axis=1)
    n_neutral_metal = neutral_dens_final[:, data.metal_bf_z_arr]
    n_ionized_metal = ionized_dens_final[:, data.metal_bf_z_arr]
    n_doubly_metal  = doubly_dens_final[:, data.metal_bf_z_arr]
    metal_bf_dens = jnp.where(
        data.metal_bf_charge_arr[None, :] == 0, n_neutral_metal,
        jnp.where(data.metal_bf_charge_arr[None, :] == 1, n_ionized_metal, n_doubly_metal)
    )

    # ── Phase 3: Continuum opacity ────────────────────────────────────────────
    if linelist_data is not None and linelist_data.cntm_wl_np_cached is not None:
        cntm_wl_np_cached = linelist_data.cntm_wl_np_cached
    else:
        wl_min_cm = float(wavelengths_cm[0])
        wl_max_cm = float(wavelengths_cm[-1])
        cntm_step_cm = 1e-8
        cntm_wl_np_cached = np.arange(wl_min_cm - cntm_step_cm, wl_max_cm + 2 * cntm_step_cm, cntm_step_cm)
    cntm_nu_np = c_cgs_float / cntm_wl_np_cached

    alpha_cntm_coarse = _batch_continuum_vmap(
        jnp.array(cntm_nu_np), T_layers, ne_all, U_H_I_all, U_He_I_all,
        nH_I_all, nH_II_all, nHe_I_all, nH2_all,
        n_peach, n_Z1_ff, n_Z2_ff, metal_bf_dens,
        data.metal_bf_tables, data.metal_bf_nu_grid, data.metal_bf_logT_grid
    )  # (n_layers, n_cntm)

    cntm_wl_jnp = jnp.array(cntm_wl_np_cached)
    alpha_cntm_all = jax.vmap(
        lambda row: jnp.interp(wavelengths_cm, cntm_wl_jnp, row)
    )(alpha_cntm_coarse)  # (n_layers, n_wl)

    # Continuum opacity *at* the reference wavelength.  Korg.jl evaluates it
    # directly there rather than reading it off the synthesis window's coarse
    # grid; `jnp.interp` clamps outside the grid, so for any window that does
    # not contain 5000 Å the old expression returned the continuum at the edge
    # of the window instead, silently rescaling the whole optical-depth scale.
    alpha_ref_all = _batch_continuum_vmap(
        jnp.array([c_cgs_float / lambda_ref_cm]), T_layers, ne_all,
        U_H_I_all, U_He_I_all, nH_I_all, nH_II_all, nHe_I_all, nH2_all,
        n_peach, n_Z1_ff, n_Z2_ff, metal_bf_dens,
        data.metal_bf_tables, data.metal_bf_nu_grid, data.metal_bf_logT_grid
    )[:, 0]  # (n_layers,)

    # Add line absorption at reference wavelength
    from .data_loader import load_default_linelist as _load_ref_ll
    _ref_ll = _load_ref_ll(lambda_ref_cm)
    if _ref_ll:
        _T_np = np.asarray(T_layers)
        _ne_np = np.asarray(ne_all)
        _neutral_np = np.asarray(neutral_dens_final)
        _ionized_np = np.asarray(ionized_dens_final)
        _nd_ref = {}
        for _Z in range(1, 93):
            _sym = atomic_symbols[_Z - 1]
            _nd_ref[Species(f'{_sym}_I')]  = _neutral_np[:, _Z - 1]
            _nd_ref[Species(f'{_sym}_II')] = _ionized_np[:, _Z - 1]
        _mol_np = np.asarray(mol_dens)
        for _i, _mol_sp in enumerate(default_mol_species):
            _nd_ref[_mol_sp] = _mol_np[:, _i]
        # Korg.jl passes a constant-per-layer continuum here: the alpha_ref just
        # computed.  See the note above about interpolating off-grid.
        _alpha_ref_cntm_np = np.asarray(alpha_ref_all)
        def _cntm_at_ref_fn(wl_cm):
            n_out = np.size(wl_cm)
            result = np.repeat(_alpha_ref_cntm_np[None, :], n_out, axis=0)
            return result if np.ndim(wl_cm) > 0 else result[0]
        _line_at_ref = line_absorption(
            _ref_ll, np.array([lambda_ref_cm]),
            _T_np, _ne_np, _nd_ref,
            default_partition_funcs, float(vmic_cm_s), _cntm_at_ref_fn,
            cutoff_threshold=3e-4,
        )  # (n_layers, 1)
        alpha_ref_all = alpha_ref_all + jnp.array(_line_at_ref[:, 0])

    # Source function
    S_all = jax.vmap(lambda T_i: blackbody(T_i, wavelengths_cm))(T_layers)

    # ── Pre-compute Hummer widths for H-line absorption (per-layer, not per-line) ──
    _T_np_h   = np.asarray(T_layers)
    _nH_np_h  = np.asarray(nH_I_all)
    _nHe_np_h = np.asarray(nHe_I_all)
    _ne_np_h  = np.asarray(ne_all)
    _ws_all_h_pre = precompute_hummer_ws(_T_np_h, _nH_np_h, _nHe_np_h, _ne_np_h)

    # ── Pre-compute per-bucket window geometry for fast Voigt + scatter ───────
    # One-time cost: compute line params, determine max_wins per line, assign to
    # buckets (same widths as _line_absorption_fast), precompute centered i_lo.
    # Stored as BucketGeometry NamedTuples so synthesize_jit can avoid Python scatter.
    _bucket_geometry = None
    if linelist_data is not None and linelist_data.wl.shape[0] > 0:
        _amp_w, _sig_w, _gam_w = _compute_line_params_jit(
            T_layers, ne_all, n_eff_vdW_all, neutral_dens_final, ionized_dens_final,
            mol_densities_all, linelist_data, data, float(vmic_cm_s)
        )
        _amp_np_w = np.asarray(_amp_w)
        _sig_np_w = np.asarray(_sig_w)
        _gam_np_w = np.asarray(_gam_w)
        _wls_np_w = (linelist_data.wls_np_cached if linelist_data.wls_np_cached is not None
                     else np.asarray(linelist_data.wl))
        _wl_np_w = (linelist_data.wl_np_cached if linelist_data.wl_np_cached is not None
                    else np.asarray(wavelengths_cm))
        _wl_sp_w = (linelist_data.wl_spacing_cached if linelist_data.wl_spacing_cached is not None
                    else float(np.median(np.diff(_wl_np_w))))
        _n_wl_w = _wl_np_w.shape[0]
        _cntm_np_w = np.asarray(alpha_cntm_coarse)
        _cntm_ctr_w = np.array([
            np.interp(_wls_np_w, cntm_wl_np_cached, _cntm_np_w[i]) for i in range(n_layers)
        ]).T  # (n_lines, n_layers)
        _rho_w = 3e-4 * _cntm_ctr_w / np.maximum(np.abs(_amp_np_w), 1e-300)
        _lg_w = np.sqrt(2.0 * np.pi) * _sig_np_w * _rho_w
        with np.errstate(invalid='ignore', divide='ignore'):
            _wG_w = np.where(_lg_w >= 1.0, 0.0,
                             _sig_np_w * np.sqrt(-2.0 * np.log(np.maximum(_lg_w, 1e-300))))
            _wLa_w = _gam_np_w / (np.pi * _rho_w)
            _wL_w = np.where(_wLa_w <= _gam_np_w**2, 0.0,
                             np.sqrt(np.maximum(_wLa_w - _gam_np_w**2, 0.0)))
        _mw_w = np.sqrt(np.max(_wG_w, axis=1)**2 + np.max(_wL_w, axis=1)**2) * (1.0 + 2e-5)
        _mwpx_w = np.clip(np.ceil(2.0 * _mw_w / _wl_sp_w + 2).astype(int), 0, _n_wl_w)
        # Build BucketGeometry for each non-empty bucket
        _BUCKET_WIDTHS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, _n_wl_w]
        _buckets = []
        _prev = 0
        for _W_MAX in _BUCKET_WIDTHS:
            _W = min(_W_MAX, _n_wl_w)
            _in_b = (_mwpx_w > _prev) & (_mwpx_w <= _W_MAX)
            _n_b = int(_in_b.sum())
            if _n_b == 0:
                _prev = _W_MAX
                continue
            _idx_b = np.where(_in_b)[0].astype(np.int32)
            _wls_b = _wls_np_w[_idx_b]
            _mw_b = _mw_w[_idx_b]
            # Centered window i_lo
            _ctr_px = np.searchsorted(_wl_np_w, _wls_b).astype(int)
            _i_lo_b = np.clip(_ctr_px - _W // 2, 0, _n_wl_w - _W).astype(np.int32)
            _buckets.append(BucketGeometry(
                W=_W,
                n_b=_n_b,
                amp_idx=_idx_b,
                i_lo=jnp.array(_i_lo_b),
                max_wins=jnp.array(_mw_b),
                wls=jnp.array(_wls_b),
            ))
            _prev = _W_MAX
        _bucket_geometry = tuple(_buckets)
        W_line = _buckets[-1].W if _buckets else 0
        _wl_jax_cached = jnp.array(_wl_np_w)
    else:
        W_line = 0
        _wl_jax_cached = None

    # Cache numpy conversions of atmospheric arrays (avoid repeated device→host copies)
    _T_np_cached     = np.asarray(T_layers)
    _ne_np_cached    = np.asarray(ne_all)
    _nH_I_np_cached  = np.asarray(nH_I_all)
    _nHe_I_np_cached = np.asarray(nHe_I_all)
    _U_H_I_np_cached = np.asarray(U_H_I_all)
    _wl_cm_np_cached = np.asarray(wavelengths_cm)

    # ── Precompute per-layer H-line Stark profiles ─────────────────────────────
    # Reduce the expensive 3D (T, ne, log_delta_nu) interpolation to a cheap 1D
    # by precomputing the bilinear (T, ne) part per layer at each log_delta_nu
    # grid point.  This cuts per-call H-line time from ~0.55 ms to ~0.13 ms.
    from .hydrogen_line_absorption import (
        hline_stark_profiles as _hline_stark,
        _interp_lambda0_all_layers_jit as _lam0_jit,
        _interp_linear_3d_jax as _i3d,
        _BALMER_ABO_PARAMS as _abo_params,
    )
    _RYDBERG_CM_PRE = 1.0973731568539e5
    _H_WIN_CM_PRE   = 150.0 * 1e-8
    _wl_min_pre = float(wavelengths_cm[0])
    _wl_max_pre = float(wavelengths_cm[-1])
    _wl_arr_np  = np.asarray(wavelengths_cm)
    _h_stark_list = []
    for _trans, _sline in _hline_stark.items():
        _lam0_vac = 1.0 / (_RYDBERG_CM_PRE * (1.0 / _sline.lower**2 - 1.0 / _sline.upper**2))
        if not (_wl_min_pre - _H_WIN_CM_PRE <= _lam0_vac <= _wl_max_pre + _H_WIN_CM_PRE):
            continue
        _valid_np = np.array([
            _sline.temp_min < _T_np_cached[_i] < _sline.temp_max and
            _sline.ne_min < _ne_np_cached[_i] < _sline.ne_max
            for _i in range(n_layers)
        ], dtype=bool)
        if not np.any(_valid_np):
            continue
        _temps_jax = jnp.asarray(_sline.temps, dtype=jnp.float64)
        _nes_jax   = jnp.asarray(_sline.electron_number_densities, dtype=jnp.float64)
        _ldnu_jax  = _sline.log_delta_nu_grid
        _prof3d    = _sline.profile_data
        # Interpolated line centre per layer
        _lam0_jax = _lam0_jit(T_layers, ne_all, _temps_jax, _nes_jax,
                               jnp.asarray(_sline.lambda0_data, dtype=jnp.float64))
        # Bilinear-reduced 1D profile per layer: (n_layers, n_delta)
        _profs_1d = jax.vmap(
            lambda Ti, nei: jax.vmap(
                lambda s: _i3d(Ti, nei, s, _temps_jax, _nes_jax, _ldnu_jax, _prof3d)
            )(_ldnu_jax)
        )(T_layers, ne_all)
        # F0 per layer
        _F0 = 1.25e-9 * ne_all**(2.0 / 3.0)
        # ABO / Balmer parameters
        if _sline.lower == 2 and _sline.upper in _abo_params:
            _lam0_abo, _sig_abo_a0, _alp_abo = _abo_params[_sline.upper]
            _lam0_stehle_jax = jnp.full(n_layers, _lam0_abo)
            _sigma_abo = float(_sig_abo_a0 * bohr_radius_cgs**2)
            _alpha_abo = float(_alp_abo)
            _abo_active = 1.0
            _lam0_ref = _lam0_abo
        else:
            _lam0_stehle_jax = _lam0_jax
            _sigma_abo = 0.0
            _alpha_abo = 0.0
            _abo_active = 0.0
            _lam0_ref = float(jnp.mean(_lam0_jax))
        # Window pixel indices (with small buffer)
        _win_mask = np.abs(_wl_arr_np - _lam0_ref) < _H_WIN_CM_PRE * 1.05
        _win_pix  = jnp.array(np.where(_win_mask)[0], dtype=jnp.int32)
        if _win_pix.shape[0] == 0:
            continue
        _h_stark_list.append(HStarkPrecomputed(
            profiles_1d=_profs_1d,
            lambda0=_lam0_jax,
            lambda0_stehle=_lam0_stehle_jax,
            F0=_F0,
            log_delta_nu_grid=_ldnu_jax,
            valid_mask=jnp.array(_valid_np),
            window_pix=_win_pix,
            window_cm=_H_WIN_CM_PRE,
            lower=int(_sline.lower),
            upper=int(_sline.upper),
            log_gf=float(_sline.log_gf),
            sigma_abo=_sigma_abo,
            alpha_abo=_alpha_abo,
            abo_active=_abo_active,
            xi=float(vmic_cm_s),
        ))
    _h_stark_precomp = tuple(_h_stark_list) if _h_stark_list else None

    # ── Precompute partition function tables ──────────────────────────────────
    # Replaces 42K per-(line,layer) spline evals with 10K precomputed + 42K lookups.
    # Atomic table: U[Z-1, charge, layer] for Z=1..92, charge=0..1
    _log_T_all = jnp.log(T_layers)
    _pf_t = data.chem_eq_data.pf_orig_t
    _pf_u = data.chem_eq_data.pf_orig_u
    _pf_h = data.chem_eq_data.pf_orig_h
    _pf_z = data.chem_eq_data.pf_orig_z
    _pf_n = data.chem_eq_data.pf_orig_n

    _U_atomic_table = _compute_U_atomic_table_jit(
        _log_T_all, _pf_t, _pf_u, _pf_h, _pf_z, _pf_n
    )
    _U_mol_table = _compute_U_mol_table_jit(
        _log_T_all, data.chem_eq_data.log_T_grid, data.chem_eq_data.log_T_h,
        data.chem_eq_data.mol_partition_func_values, data.chem_eq_data.mol_partition_func_z
    )
    jax.block_until_ready(_U_atomic_table)
    jax.block_until_ready(_U_mol_table)

    return PrecomputedAtmosphereData(
        T_layers=T_layers,
        ne_all=ne_all,
        z_layers=z_layers,
        log_tau_ref=log_tau_ref,
        vmic_cm_s=float(vmic_cm_s),
        nf_sol=nf_sol,
        neutral_dens=neutral_dens_final,
        ionized_dens=ionized_dens_final,
        mol_dens_padded=mol_densities_all,
        nH_I_all=nH_I_all,
        nH_II_all=nH_II_all,
        nHe_I_all=nHe_I_all,
        U_H_I_all=U_H_I_all,
        n_eff_vdW_all=n_eff_vdW_all,
        alpha_cntm_all=alpha_cntm_all,
        alpha_ref_all=alpha_ref_all,
        alpha_cntm_coarse=alpha_cntm_coarse,
        cntm_wl_np=cntm_wl_np_cached,
        S_all=S_all,
        ws_all_h=_ws_all_h_pre,
        W_line=W_line,
        bucket_geometry=_bucket_geometry,
        wl_jax=_wl_jax_cached,
        T_np=_T_np_cached,
        ne_np=_ne_np_cached,
        nH_I_np=_nH_I_np_cached,
        nHe_I_np=_nHe_I_np_cached,
        U_H_I_np=_U_H_I_np_cached,
        wl_cm_np=_wl_cm_np_cached,
        h_stark_precomp=_h_stark_precomp,
        U_atomic_table=_U_atomic_table,
        U_mol_table=_U_mol_table,
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
        mol_log_K_z=np.asarray(data.chem_eq_data.mol_log_K_z),
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
            mol_log_K_z=jnp.array(f['mol_log_K_z']),
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


def preprocess_linelist(linelist: List[Line], chem_eq_data=None,
                        wavelengths_cm=None) -> LinelistData:
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
    wavelengths_cm : array-like, optional
        Synthesis wavelength grid [cm].  When provided, several numpy arrays
        derived from the wavelength grid and linelist wavelengths are
        precomputed and stored on the returned :class:`LinelistData` so that
        :func:`synthesize_jit` does not need to recompute them on every call.

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

    # Precompute wavelength-grid-derived numpy arrays if wavelengths_cm was provided.
    # These are reused by synthesize_jit on every call, saving repeated recomputation.
    wl_np_cached = wls_np_cached = wl_spacing_cached = cntm_wl_np_cached = None
    if wavelengths_cm is not None:
        import numpy as _np_wl
        wl_np_cached = _np_wl.asarray(wavelengths_cm, dtype=_np_wl.float64)
        wls_np_cached = _np_wl.array([line.wl for line in linelist], dtype=_np_wl.float64)
        n_wl = len(wl_np_cached)
        diffs = _np_wl.diff(wl_np_cached)
        wl_spacing_cached = float(_np_wl.median(diffs)) if len(diffs) > 0 else 5e-9
        # Coarse continuum wavelength grid (1 Å step, matching synthesize_jit default)
        cntm_step_cm = 1e-8
        wl_min_cm = float(wl_np_cached[0])
        wl_max_cm = float(wl_np_cached[-1])
        cntm_wl_np_cached = _np_wl.arange(
            wl_min_cm - cntm_step_cm, wl_max_cm + 2 * cntm_step_cm, cntm_step_cm,
            dtype=_np_wl.float64,
        )

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
        mol_species_idx=jnp.array(mol_species_idx_list, dtype=jnp.int32),
        wl_np_cached=wl_np_cached,
        wls_np_cached=wls_np_cached,
        wl_spacing_cached=wl_spacing_cached,
        cntm_wl_np_cached=cntm_wl_np_cached,
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

    in_range = (wavelength_um > 0.125) & (wavelength_um < 1.6419)
    # (x - 0.125)**1.5 is complex for x < 0.125 and its derivative is infinite
    # at x == 0.125.  jnp.where selects the 0.0 branch, but it does not stop the
    # *unselected* branch from being evaluated: below 0.125 um the whole
    # expression came back complex (so the function returned a complex zero),
    # and jax.grad returned NaN.  Substituting a wavelength strictly inside the
    # fit's validity range makes the unselected branch real and smooth.  0.5 um
    # is used rather than clamping to 0.125, because (x - 0.125)**1.5 has an
    # infinite derivative exactly at the endpoint.
    x = jnp.where(in_range, wavelength_um, 0.5)
    sigma = jnp.where(
        in_range,
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


def _dawson_ratio(v):
    """
    ``(1 - exp(-v²)) / v``, finite in value *and* gradient at ``v == 0``.

    The limit is 0, and the derivative there is 0 as well.  Evaluating the
    ratio at a substituted non-zero ``v`` keeps the unselected branch of the
    surrounding ``jnp.where`` free of the 0/0 that otherwise poisons the
    reverse-mode cotangent.
    """
    tiny = jnp.abs(v) < 1e-6
    v_safe = jnp.where(tiny, 1.0, v)      # strictly non-zero substitute
    return jnp.where(tiny, 1.0, (1 - jnp.exp(-v_safe**2)) / v_safe)


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
            # Small |z|: more accurate approximation.
            # (1 - exp(-v^2))/v is 0/0 at v == 0.  jnp.where picks the 1.0
            # branch there, but the unselected branch is still differentiated
            # and its NaN cotangent survives the multiply-by-zero, so dH/dv was
            # NaN at exactly v == 0.  Feeding the ratio a strictly non-zero v
            # removes the singularity without touching any selected value.
            jnp.exp(-v**2) * (1 - a * 2 / jnp.sqrt(jnp.pi) *
                              _dawson_ratio(v))
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
    ne_sol, neutral_fracs = picard_chemical_equilibrium_guess(T, n_total, ne, abundances, data.chem_eq_data)

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
    # A species with a single tabulated knot -- H III, which does not exist --
    # makes `n_knots - 2` equal -1. jnp.clip(x, 0, -1) returns -1, so `t_arr[i]`
    # wraps to the last element, which is the *inf padding*, and `h_arr[i + 1]`
    # becomes a zero divisor. The value is masked downstream but the cotangent is
    # not, and inf/0 partials meeting a zero cotangent give NaN.
    #
    # This is the same degeneracy that was fixed in the table *builder*; the
    # shipped table predates that fix and still contains it (201 non-finite knot
    # entries against 199 in a freshly computed one). Guarding here means the
    # evaluator is correct for both, rather than depending on which table a
    # caller happens to load -- which is what made a verified-this-morning
    # gradient come back NaN in production.
    n_eff = jnp.maximum(n_knots, 2)
    t_safe = jnp.where(jnp.isfinite(t_arr), t_arr, jnp.finfo(jnp.float64).max)
    t_max = t_safe[n_eff - 1]
    log_T_c = jnp.clip(log_T, t_safe[0], t_max)
    i = jnp.clip(jnp.searchsorted(t_safe, log_T_c, side='right') - 1, 0, n_eff - 2)
    ti  = t_safe[i];  ti1 = t_safe[i + 1]
    ui  = u_arr[i];   ui1 = u_arr[i + 1]
    zi  = z_arr[i];   zi1 = z_arr[i + 1]
    hi1 = h_arr[i + 1]
    # Strictly positive substitute, not a clamp to zero: 1/h and h both appear,
    # so a zero divisor is infinite either way.
    hi1 = jnp.where(hi1 != 0.0, hi1, 1.0)
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


@jax.jit
def _compute_line_params_table_jit(
    T_layers, ne_all, nH_I_all, neutral_dens_final, ionized_dens_final,
    mol_densities_all, linelist_data, data, vmic_cm_s,
    U_atomic_table, U_mol_table,
):
    """Fast line_params using precomputed partition function tables.

    Replaces per-(line,layer) spline evaluations with O(1) table lookups,
    giving ~4× speedup over _compute_line_params_jit.

    U_atomic_table : (92, 2, n_layers) — U[Z-1, charge, layer]
    U_mol_table    : (n_mol, n_layers) — U[mol_idx, layer]
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
        n_mols_real     = U_mol_table.shape[0]
        safe_pf_mol_idx = jnp.clip(mol_idx, 0, jnp.maximum(n_mols_real - 1, 0))

        # Table lookup: (n_layers,) U values for this line's species
        U_atomic_this = U_atomic_table[Z - 1, charge]       # (n_layers,) dynamic gather
        U_mol_this    = U_mol_table[safe_pf_mol_idx]         # (n_layers,) dynamic gather
        U_this        = jnp.where(mol_idx >= 0, U_mol_this, U_atomic_this)

        def per_layer_params(T_i, ne_i, nH_I_i, n_neutral_i, n_ion_i, mol_densities_i, U_i):
            n_atomic  = jnp.where(charge == 0, n_neutral_i[Z - 1], n_ion_i[Z - 1])
            n_mol     = mol_densities_i[safe_mol_idx]
            n_species = jnp.where(mol_idx >= 0, n_mol, n_atomic)
            U         = U_i  # precomputed — no spline eval needed
            sigma_D   = wl_center * jnp.sqrt(kboltz_cgs * T_i / mass + vmic_cm_s**2 / 2.0) / c_cgs
            g_stark   = gamma_stark_l * (T_i / 1e4)**(1.0 / 6.0) * ne_i
            g_vdW     = jnp.where(
                vdW_alpha_l < 0.0,
                vdW_sigma_l * (T_i / 1e4)**0.3 * nH_I_i,
                2.0 * vdW_sigma_l * 1e6 * (T_i / 1e4)**(0.5 * (1.0 - vdW_alpha_l)) * nH_I_i
            )
            gamma_total = gamma_rad_l + g_stark + g_vdW
            gamma_L   = gamma_total * wl_center**2 / (4.0 * jnp.pi * c_cgs)
            stim      = 1.0 - jnp.exp(-hplanck_eV * nu / (kboltz_eV * T_i))
            boltz     = jnp.exp(-E_lower / (kboltz_eV * T_i))
            amplitude = (n_species / jnp.clip(U, 1e-10) *
                         10.0**log_gf_val * sigma_ln * boltz * stim)
            return amplitude, sigma_D, gamma_L

        return jax.vmap(per_layer_params)(
            T_layers, ne_all, nH_I_all, neutral_dens_final, ionized_dens_final,
            mol_densities_all, U_this
        )

    return jax.vmap(per_line_params)(jnp.arange(linelist_data.wl.shape[0]))


_voigt_profile_jax_jit = jax.jit(_voigt_profile_jax)


@functools.partial(jax.jit, static_argnames=('W', 'n_wl_s', 'n_layers_s'))
def _voigt_bucket_jit(
    amp, sigma_D, gamma_L,
    i_lo_jax, max_wins_jax, wls_jax, wl_jax,
    *,
    W: int, n_wl_s: int, n_layers_s: int,
):
    """JIT Voigt + scatter for one bucket of lines.

    Parameters already sliced to this bucket; window geometry (i_lo, max_wins)
    is precomputed at precompute_atmosphere time and passed as JAX arrays.

    amp, sigma_D, gamma_L : (n_b, n_layers)
    i_lo_jax              : (n_b,) int32  — centered window start pixels
    max_wins_jax          : (n_b,) float  — half-width in cm per line
    wls_jax               : (n_b,) float  — line centers [cm]
    wl_jax                : (n_wl,) float — synthesis wavelength grid [cm]
    """
    pix_idx  = i_lo_jax[:, None] + jnp.arange(W)[None, :]          # (n_b, W)
    delta    = wl_jax[pix_idx] - wls_jax[:, None]                   # (n_b, W)
    mask     = jnp.abs(delta) <= max_wins_jax[:, None]              # (n_b, W)
    profiles = _voigt_profile_jax(
        delta[:, None, :], sigma_D[:, :, None], gamma_L[:, :, None]
    )  # (n_b, n_layers, W)
    contrib      = mask[:, None, :] * amp[:, :, None] * profiles    # (n_b, n_layers, W)
    flat_pix     = pix_idx.ravel()                                    # (n_b * W,)
    flat_contrib = contrib.transpose(1, 0, 2).reshape(n_layers_s, -1)  # (n_layers, n_b*W)
    return jax.vmap(lambda c: jnp.zeros(n_wl_s).at[flat_pix].add(c))(flat_contrib)


@functools.partial(jax.jit, static_argnames=('bucket_Ws', 'bucket_n_bs', 'n_wl_s', 'n_layers_s'))
def _all_buckets_jit(
    amp_all, sigma_all, gamma_all,
    all_amp_idxs, all_i_los, all_max_wins, all_wls,
    wl_jax,
    *, bucket_Ws, bucket_n_bs, n_wl_s, n_layers_s,
):
    """Fused JIT: all buckets in one XLA program, one dispatch overhead total.

    Gather + Voigt + scatter for every bucket is unrolled at trace time.
    amp_all, sigma_all, gamma_all : (n_lines, n_layers)
    all_amp_idxs : tuple of (n_b,) int32 arrays, one per bucket
    all_i_los    : tuple of (n_b,) int32 arrays
    all_max_wins : tuple of (n_b,) float arrays
    all_wls      : tuple of (n_b,) float arrays
    bucket_Ws    : tuple of ints (static, controls unrolling)
    bucket_n_bs  : tuple of ints (static, shapes)
    """
    result = jnp.zeros((n_layers_s, n_wl_s))
    for W, amp_idx, i_lo, max_wins, wls in zip(
        bucket_Ws, all_amp_idxs, all_i_los, all_max_wins, all_wls
    ):
        amp    = amp_all[amp_idx]                                    # (n_b, n_layers)
        sigma  = sigma_all[amp_idx]
        gamma  = gamma_all[amp_idx]
        pix    = i_lo[:, None] + jnp.arange(W)[None, :]            # (n_b, W)
        # Compute delta precisely in float64, then cast to float32 for Voigt.
        # The wavelength differences are O(10^-8 cm) which float32 represents safely;
        # the cast introduces only ~3e-11 absolute error in line_alpha (within 1e-6 budget).
        delta_f64 = wl_jax[pix] - wls[:, None]                      # (n_b, W) float64
        mask   = jnp.abs(delta_f64) <= max_wins[:, None]             # (n_b, W) bool
        delta  = delta_f64.astype(jnp.float32)                       # (n_b, W) float32
        prof   = _voigt_profile_jax(
            delta[:, None, :],
            sigma[:, :, None].astype(jnp.float32),
            gamma[:, :, None].astype(jnp.float32),
        ).astype(jnp.float64)                                         # back to f64
        cont   = mask[:, None, :] * amp[:, :, None] * prof
        flat_p = pix.ravel()
        flat_c = cont.transpose(1, 0, 2).reshape(n_layers_s, -1)
        result = result + jax.vmap(lambda c: jnp.zeros(n_wl_s).at[flat_p].add(c))(flat_c)
    return result


@jax.jit
def _rt_both_jit(alpha_total_T, alpha_cntm_T, S_T, z, log_tau, alpha_ref):
    """Fused radiative transfer for both full and continuum-only in one XLA dispatch."""
    from .radiative_transfer import radiative_transfer_jit as _rt
    flux, _ = _rt(alpha_total_T, S_T, z, log_tau, alpha_ref)
    flux_cntm, _ = _rt(alpha_cntm_T, S_T, z, log_tau, alpha_ref)
    return flux, flux_cntm


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
    precomputed_atm: Optional['PrecomputedAtmosphereData'] = None,
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
    precomputed_atm : PrecomputedAtmosphereData, optional
        Pre-computed atmosphere-dependent quantities from precompute_atmosphere().
        When provided, skips chemical equilibrium and continuum phases (~28 ms).

    Returns
    -------
    flux : array, shape (n_wl,)
        Emergent flux [erg cm⁻² s⁻¹ cm⁻¹]
    continuum : array, shape (n_wl,)
        Continuum flux [erg cm⁻² s⁻¹ cm⁻¹]
    """
    n_wl = wavelengths_cm.shape[0]

    if precomputed_atm is not None:
        # Unpack precomputed quantities — skip phases 1-3
        T_layers          = precomputed_atm.T_layers
        ne_all            = precomputed_atm.ne_all
        z_layers          = precomputed_atm.z_layers
        log_tau_ref       = precomputed_atm.log_tau_ref
        vmic_cm_s         = precomputed_atm.vmic_cm_s
        nf_sol            = precomputed_atm.nf_sol
        neutral_dens_final = precomputed_atm.neutral_dens
        ionized_dens_final = precomputed_atm.ionized_dens
        mol_densities_all = precomputed_atm.mol_dens_padded
        nH_I_all          = precomputed_atm.nH_I_all
        nH_II_all         = precomputed_atm.nH_II_all
        nHe_I_all         = precomputed_atm.nHe_I_all
        U_H_I_all         = precomputed_atm.U_H_I_all
        n_eff_vdW_all     = precomputed_atm.n_eff_vdW_all
        alpha_cntm_all    = precomputed_atm.alpha_cntm_all
        alpha_ref_all     = precomputed_atm.alpha_ref_all
        alpha_cntm_coarse = precomputed_atm.alpha_cntm_coarse
        cntm_wl_np_cached = precomputed_atm.cntm_wl_np
        S_all             = precomputed_atm.S_all
        n_layers = T_layers.shape[0]
    else:
        n_layers = T_layers.shape[0]
        lambda_ref_cm = 5e-5  # 5000 Å reference wavelength

        # ── Phase 1: Picard initial guess for Newton solver ──────────────────────
        # A single Picard pass (300 iterations, all layers at once) gives ne and
        # neutral fractions close enough that Newton converges in ~5 iterations.
        ne_init, nf_init = _picard_chemical_equilibrium_guess_batch(
            T_layers, n_total_layers, ne_layers, abundances, data.chem_eq_data
        )

        # ── Phase 2: Newton solver — matches Julia's _solve_chemical_equilibrium ──
        # 93-dim Newton (∞-norm, ftol=1e-8, analytical Jacobian, LU solve).
        # Layers solved in parallel via vmap; Picard guess gives 3-4 Newton iterations.
        ne_all, nf_sol = _chem_eq_newton_batch_jit(
            T_layers, n_total_layers, ne_init, nf_init, abundances, data.chem_eq_data
        )

        # Derive number densities from Newton solution
        atom_dens_all      = abundances[None, :] * (n_total_layers - ne_all)[:, None]  # (n_layers, 92)
        neutral_dens_final = atom_dens_all * nf_sol                                     # (n_layers, 92)
        wII_all, wIII_all  = _compute_saha_weights_batch_jit(T_layers, ne_all, data.chem_eq_data)
        ionized_dens_final = wII_all * neutral_dens_final                               # (n_layers, 92)
        doubly_dens_final  = wIII_all * neutral_dens_final                              # (n_layers, 92)

        # Molecular densities from Newton-solved neutral fractions
        mol_dens = _compute_mol_densities_batch_jit(
            T_layers, n_total_layers, ne_all, abundances, nf_sol, data.chem_eq_data
        )  # (n_layers, n_mols)

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
        n_eff_vdW_all = nH_I_all  # currently matches HJXcT Korg.jl (H-only perturbers)

        # Peach FF species: He_II (z=1), C_II (z=5), Si_II (z=13), Mg_II (z=11)
        _peach_z = jnp.array([z for z, _ in _PEACH_IDX])    # [1, 5, 13, 11]
        n_peach = ionized_dens_final[:, _peach_z]            # (n_layers, 4)

        # Z=1 FF: sum of all singly-ionized minus all Peach species (He_II, C_II, Si_II, Mg_II)
        # Peach species are handled separately with Peach (1970) corrections; subtracting all 4
        # matches prepare_continuum_batch_fast which excludes all peach_z_idx from n_Z1_ff.
        n_Z1_ff = ionized_dens_final.sum(axis=1) - ionized_dens_final[:, _peach_z].sum(axis=1)    # (n_layers,)

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
        # Use cached coarse continuum wavelength grid from linelist_data if available,
        # otherwise compute it here (backward compatibility).
        if linelist_data.cntm_wl_np_cached is not None:
            cntm_wl_np_cached = linelist_data.cntm_wl_np_cached
        else:
            wl_min_cm = float(wavelengths_cm[0])
            wl_max_cm = float(wavelengths_cm[-1])
            cntm_step_cm = 1e-8  # 1.0 Å, matching synthesize default cntm_step
            cntm_wl_np_cached = np.arange(wl_min_cm - cntm_step_cm, wl_max_cm + 2 * cntm_step_cm, cntm_step_cm)
        cntm_nu_np = c_cgs_float / cntm_wl_np_cached  # (n_cntm,) decreasing frequencies

        alpha_cntm_coarse = _batch_continuum_vmap(
            jnp.array(cntm_nu_np), T_layers, ne_all, U_H_I_all, U_He_I_all,
            nH_I_all, nH_II_all, nHe_I_all, nH2_all,
            n_peach, n_Z1_ff, n_Z2_ff, metal_bf_dens,
            data.metal_bf_tables, data.metal_bf_nu_grid, data.metal_bf_logT_grid
        )  # (n_layers, n_cntm)

        # Interpolate to fine output grid (matches synthesize's interp1d linear)
        cntm_wl_jnp = jnp.array(cntm_wl_np_cached)
        alpha_cntm_all = jax.vmap(
            lambda row: jnp.interp(wavelengths_cm, cntm_wl_jnp, row)
        )(alpha_cntm_coarse)  # (n_layers, n_wl)

        # Reference opacity: evaluate the continuum *at* lambda_ref, as Korg.jl
        # does.  Reading it off the synthesis window's coarse grid returns the
        # clamped edge value whenever the window excludes 5000 Å, which
        # rescales the entire anchored optical-depth scale.
        alpha_ref_all = _batch_continuum_vmap(
            jnp.array([c_cgs_float / lambda_ref_cm]), T_layers, ne_all,
            U_H_I_all, U_He_I_all, nH_I_all, nH_II_all, nHe_I_all, nH2_all,
            n_peach, n_Z1_ff, n_Z2_ff, metal_bf_dens,
            data.metal_bf_tables, data.metal_bf_nu_grid, data.metal_bf_logT_grid
        )[:, 0]  # (n_layers,)

        # Add line absorption at reference wavelength to alpha_ref_all (matching Julia).
        # Julia's alpha_ref = continuum + lines at 5000 Å.
        from .data_loader import load_default_linelist as _load_ref_ll
        _ref_ll = _load_ref_ll(lambda_ref_cm)
        if _ref_ll:
            _T_np = np.asarray(T_layers)
            _ne_np = np.asarray(ne_all)
            _neutral_np = np.asarray(neutral_dens_final)   # (n_layers, 92)
            _ionized_np = np.asarray(ionized_dens_final)   # (n_layers, 92)
            _nd_ref = {}
            for _Z in range(1, 93):
                _sym = atomic_symbols[_Z - 1]
                _nd_ref[Species(f'{_sym}_I')]  = _neutral_np[:, _Z - 1]
                _nd_ref[Species(f'{_sym}_II')] = _ionized_np[:, _Z - 1]
            # Include molecular densities so reference linelist molecular lines (OZr, OV, HMg, etc.)
            # contribute to alpha_ref, matching Julia's behavior.
            _mol_np = np.asarray(mol_dens)
            for _i, _mol_sp in enumerate(default_mol_species):
                _nd_ref[_mol_sp] = _mol_np[:, _i]
            # Korg.jl: a constant per-layer continuum equal to alpha_ref.
            _alpha_ref_cntm_np = np.asarray(alpha_ref_all)
            def _cntm_at_ref_jit_fn(wl_cm):
                n_out = np.size(wl_cm)
                result = np.repeat(_alpha_ref_cntm_np[None, :], n_out, axis=0)
                return result if np.ndim(wl_cm) > 0 else result[0]
            _line_at_ref = line_absorption(
                _ref_ll, np.array([lambda_ref_cm]),
                _T_np, _ne_np, _nd_ref,
                default_partition_funcs, float(vmic_cm_s), _cntm_at_ref_jit_fn,
                cutoff_threshold=3e-4,
            )  # (n_layers, 1)
            alpha_ref_all = alpha_ref_all + jnp.array(_line_at_ref[:, 0])

        # Source function: Planck function per layer at all wavelengths
        S_all = jax.vmap(lambda T_i: blackbody(T_i, wavelengths_cm))(T_layers)  # (n_layers, n_wl)

    # ── Phase 4: Line absorption with exact bucketing (matches _line_absorption_fast) ──
    # Pre-compute H-line membership before Phase 4 so we can overlap H-line numpy work
    # with JAX device computation in the precomputed-atmosphere path.
    _RYDBERG_CM_P4 = 1.0973731568539e5
    _H_WIN_CM_P4   = 150.0 * 1e-8
    _wl_min_cm_p4  = float(wavelengths_cm[0])
    _wl_max_cm_p4  = float(wavelengths_cm[-1])
    nearby_stark_p4 = {
        k: v for k, v in hline_stark_profiles.items()
        if (_wl_min_cm_p4 - _H_WIN_CM_P4
            <= 1.0 / (_RYDBERG_CM_P4 * (1.0 / v.lower**2 - 1.0 / v.upper**2))
            <= _wl_max_cm_p4 + _H_WIN_CM_P4)
    }
    brackett_in_range_p4 = any(
        _wl_min_cm_p4 - _H_WIN_CM_P4
        <= 1.0 / (_RYDBERG_CM_P4 * (1.0 / 16.0 - 1.0 / m**2))
        <= _wl_max_cm_p4 + _H_WIN_CM_P4
        for m in range(5, 31)
    )

    n_lines = linelist_data.wl.shape[0]
    h_alpha = None  # set here in precomputed path; set in Phase 4.5 otherwise

    if n_lines == 0:
        line_alpha = jnp.zeros((n_layers, n_wl))
    elif precomputed_atm is not None and precomputed_atm.bucket_geometry:
        # ── Fused bucket-JIT path with precomputed H-lines ──
        from .hydrogen_line_absorption import _h_alpha_from_precomp_jit as _h_precomp_fn
        _ws_all_h = precomputed_atm.ws_all_h
        _wl_jax_p = precomputed_atm.wl_jax

        # Step 1: dispatch H-lines FIRST (precomputed 1D Stark profiles, ~0.10 ms)
        # — enters XLA queue before line_params so it runs earliest
        if precomputed_atm.h_stark_precomp:
            h_alpha = jnp.zeros((n_layers, n_wl))
            for _hsp in precomputed_atm.h_stark_precomp:
                _contrib_win = _h_precomp_fn(
                    _wl_jax_p[_hsp.window_pix],
                    T_layers, nH_I_all, U_H_I_all, _ws_all_h,
                    _hsp.valid_mask, _hsp.profiles_1d,
                    _hsp.lambda0, _hsp.lambda0_stehle, _hsp.F0,
                    _hsp.log_delta_nu_grid, _hsp.window_cm,
                    _hsp.log_gf, _hsp.sigma_abo, _hsp.alpha_abo,
                    _hsp.abo_active, _hsp.xi,
                    lower=_hsp.lower, upper=_hsp.upper,
                )  # (n_layers, n_win)
                h_alpha = h_alpha.at[:, _hsp.window_pix].add(_contrib_win)
            if brackett_in_range_p4:
                _nHe_I_np_b = precomputed_atm.nHe_I_np
                _wl_cm_np_b = precomputed_atm.wl_cm_np
                _T_np_b = precomputed_atm.T_np
                _ne_np_b = precomputed_atm.ne_np
                _nH_I_np_b = precomputed_atm.nH_I_np
                _U_H_I_np_b = precomputed_atm.U_H_I_np
                _h_brack = np.zeros((n_layers, n_wl))
                for _i in range(n_layers):
                    _h_brack[_i] += hydrogen_line_absorption(
                        _wl_cm_np_b, _T_np_b[_i], _ne_np_b[_i], _nH_I_np_b[_i],
                        _nHe_I_np_b[_i], float(_U_H_I_np_b[_i]), float(vmic_cm_s),
                        _H_WIN_CM_P4, use_MHD=True, ws=_ws_all_h[_i], stark_profiles={}
                    )
                h_alpha = h_alpha + jnp.array(_h_brack)
        else:
            h_alpha = None  # will use numpy H-lines path after Voigt

        # Step 2: dispatch line params to XLA — use fast table-lookup version when available
        if (precomputed_atm.U_atomic_table is not None
                and precomputed_atm.U_mol_table is not None):
            amp_jax, sigma_D_jax, gamma_L_jax = _compute_line_params_table_jit(
                T_layers, ne_all, n_eff_vdW_all, neutral_dens_final, ionized_dens_final,
                mol_densities_all, linelist_data, data, vmic_cm_s,
                precomputed_atm.U_atomic_table, precomputed_atm.U_mol_table,
            )
        else:
            amp_jax, sigma_D_jax, gamma_L_jax = _compute_line_params_jit(
                T_layers, ne_all, n_eff_vdW_all, neutral_dens_final, ionized_dens_final,
                mol_densities_all, linelist_data, data, vmic_cm_s
            )

        # Step 3: dispatch Voigt (chains after line params in XLA queue)
        _bgs = precomputed_atm.bucket_geometry
        line_alpha = _all_buckets_jit(
            amp_jax, sigma_D_jax, gamma_L_jax,
            tuple(_bg.amp_idx  for _bg in _bgs),
            tuple(_bg.i_lo     for _bg in _bgs),
            tuple(_bg.max_wins for _bg in _bgs),
            tuple(_bg.wls      for _bg in _bgs),
            _wl_jax_p,
            bucket_Ws=tuple(_bg.W   for _bg in _bgs),
            bucket_n_bs=tuple(_bg.n_b for _bg in _bgs),
            n_wl_s=n_wl, n_layers_s=n_layers,
        )

        # Numpy H-lines fallback (only if precomputed Stark not available)
        if h_alpha is None:
            _T_np     = precomputed_atm.T_np
            _ne_np    = precomputed_atm.ne_np
            _nH_I_np  = precomputed_atm.nH_I_np
            _nHe_I_np = precomputed_atm.nHe_I_np
            _U_H_I_np = precomputed_atm.U_H_I_np
            _wl_cm_np = precomputed_atm.wl_cm_np
            h_alpha = np.zeros((n_layers, n_wl))
            if nearby_stark_p4:
                h_alpha += hydrogen_line_absorption_stark_batched(
                    _wl_cm_np, _T_np, _ne_np, _nH_I_np, _U_H_I_np,
                    _H_WIN_CM_P4, float(vmic_cm_s), _ws_all_h, nearby_stark_p4
                )
            if brackett_in_range_p4:
                for _i in range(n_layers):
                    h_alpha[_i] += hydrogen_line_absorption(
                        _wl_cm_np, _T_np[_i], _ne_np[_i], _nH_I_np[_i], _nHe_I_np[_i],
                        float(_U_H_I_np[_i]), float(vmic_cm_s),
                        _H_WIN_CM_P4, use_MHD=True, ws=_ws_all_h[_i], stark_profiles={}
                    )
    else:
        # ── Standard bucketed path (used when no precomputed atmosphere) ──
        # Step 4a: params — stays on device as JAX arrays
        amp_jax, sigma_D_jax, gamma_L_jax = _compute_line_params_jit(
            T_layers, ne_all, n_eff_vdW_all, neutral_dens_final, ionized_dens_final,
            mol_densities_all, linelist_data, data, vmic_cm_s
        )  # each (n_lines, n_layers)

        # Cached numpy arrays for indexing (no JAX dependency)
        wl_np = linelist_data.wl_np_cached if linelist_data.wl_np_cached is not None else np.asarray(wavelengths_cm)
        wls_np = linelist_data.wls_np_cached if linelist_data.wls_np_cached is not None else np.asarray(linelist_data.wl)
        wl_spacing = linelist_data.wl_spacing_cached if linelist_data.wl_spacing_cached is not None else (float(np.median(np.diff(wl_np))) if n_wl > 1 else 5e-9)

        # Step 4b: sync to numpy, compute max_wins per line
        amp_np   = np.asarray(amp_jax)
        sigma_np = np.asarray(sigma_D_jax)
        gamma_np = np.asarray(gamma_L_jax)

        cntm_coarse_np = np.asarray(alpha_cntm_coarse)
        cntm_at_center = np.array(
            [np.interp(wls_np, cntm_wl_np_cached, cntm_coarse_np[i]) for i in range(n_layers)]
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
        max_wins = np.sqrt(np.max(win_G, axis=1)**2 + np.max(win_L, axis=1)**2) * (1.0 + 2e-5)
        max_wins_px = np.clip(np.ceil(2.0 * max_wins / wl_spacing + 2).astype(int), 0, n_wl)

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

            i_lo_b = np.searchsorted(wl_np, wls_b - max_wins[idx_b]).astype(int)
            i_lo_b = np.clip(i_lo_b, 0, n_wl - W)

            pix_idx = i_lo_b[:, None] + np.arange(W, dtype=int)[None, :]  # (n_b, W)
            wl_win  = wl_np[pix_idx]                                        # (n_b, W)
            delta_b = wl_win - wls_b[:, None]                               # (n_b, W)
            mask_b  = np.abs(delta_b) <= max_wins[idx_b, None]              # (n_b, W)

            delta_j = jnp.asarray(delta_b[:, None, :])
            sigma_j = jnp.asarray(sigma_np[idx_b, :, None])
            gamma_j = jnp.asarray(gamma_np[idx_b, :, None])
            profiles = np.asarray(
                _voigt_profile_jax_jit(delta_j, sigma_j, gamma_j)
            )  # (n_b, n_layers, W)

            contrib = mask_b[:, None, :] * amp_np[idx_b, :, None] * profiles

            for il, i_lo in enumerate(i_lo_b):
                alpha_lines[:, i_lo:i_lo + W] += contrib[il]

            prev_W = W_MAX

        line_alpha = jnp.asarray(alpha_lines)

    # ── Phase 4.5: Hydrogen line absorption ──────────────────────────────────────
    # Skipped when precomputed_atm is used (h_alpha computed in Phase 4 above,
    # overlapping with XLA line params computation).
    if h_alpha is None:
        _RYDBERG_CM = 1.0973731568539e5
        _H_LINE_WINDOW_CM = 150.0 * 1e-8
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

        if precomputed_atm is not None and precomputed_atm.ws_all_h is not None:
            ws_all_h = precomputed_atm.ws_all_h
        else:
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

    # Update alpha_ref_all to continuum + atomic-line opacity at the reference wavelength,
    # matching Julia's alpha_5 = cntm + synthesis_linelist_lines (NO H-lines).
    # Julia: get_alpha_5000_linelist uses the synthesis linelist filtered ±21 Å; H-lines
    # are added to alpha AFTER alpha_5 is computed, so alpha_5 never includes them.
    _lambda_ref = 5e-5  # 5000 Å MARCS reference wavelength [cm]
    _wl0 = float(wavelengths_cm[0]); _wl1 = float(wavelengths_cm[-1])
    if _wl0 <= _lambda_ref <= _wl1:
        _ref_idx = int(jnp.argmin(jnp.abs(wavelengths_cm - _lambda_ref)))
        alpha_ref_all = (alpha_cntm_all + line_alpha)[:, _ref_idx]  # no H-lines

    # ── Phase 5: Radiative transfer ───────────────────────────────────────────
    # Fused: both RT passes in one XLA dispatch (saves one dispatch + shared S/z/tau reads)
    if precomputed_atm is not None:
        flux, flux_cntm = _rt_both_jit(
            alpha_total.T, alpha_cntm_all.T, S_all.T, z_layers, log_tau_ref, alpha_ref_all
        )
    else:
        from .radiative_transfer import radiative_transfer_jit
        flux, _ = radiative_transfer_jit(
            alpha_total.T, S_all.T, z_layers, log_tau_ref, alpha_ref_all
        )
        flux_cntm, _ = radiative_transfer_jit(
            alpha_cntm_all.T, S_all.T, z_layers, log_tau_ref, alpha_ref_all
        )

    # Convert cm⁻¹ → Å⁻¹ (1 cm = 1e8 Å)
    return flux * 1e-8, flux_cntm * 1e-8
