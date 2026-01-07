"""
Line absorption calculation for atomic lines.

This module computes opacity from atomic spectral lines using Voigt profiles,
including proper treatment of Doppler and pressure broadening.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Dict, Callable, Tuple, Optional, Union

from .linelist import Line
from .species import Species
from .constants import (
    c_cgs, electron_charge_cgs, electron_mass_cgs, hplanck_eV,
    kboltz_eV, kboltz_cgs, amu_cgs, bohr_radius_cgs
)
from .line_profiles import voigt_hjerting
from .atomic_data import atomic_masses
from jax.scipy.special import gamma as gamma_function


def inverse_gaussian_density(rho: float, sigma: float) -> float:
    """
    Calculate the inverse of a (0-centered) Gaussian PDF with standard deviation sigma.

    Returns the value of x for which rho = exp(-0.5 x²/σ²) / √(2π),
    which is given by σ √[-2 log(√(2π)σρ)].

    Returns 0 when rho is larger than any value taken on by the PDF.

    Args:
        rho: Probability density value
        sigma: Standard deviation

    Returns:
        Distance from center where PDF equals rho, or 0 if rho is too large
    """
    max_density = 1.0 / (jnp.sqrt(2 * jnp.pi) * sigma)

    # JAX-compatible: use jnp.where instead of if/else
    result = sigma * jnp.sqrt(-2 * jnp.log(jnp.sqrt(2 * jnp.pi) * sigma * rho))
    return jnp.where(rho > max_density, 0.0, result)


def inverse_lorentz_density(rho: float, gamma: float) -> float:
    """
    Calculate the inverse of a (0-centered) Lorentz PDF with width gamma.

    Returns the value of x for which rho = 1 / (π γ (1 + x²/γ²)),
    which is given by √[γ/(πρ) - γ²].

    Returns 0 when rho is larger than any value taken on by the PDF.

    Args:
        rho: Probability density value
        gamma: Lorentz width (HWHM)

    Returns:
        Distance from center where PDF equals rho, or 0 if rho is too large
    """
    max_density = 1.0 / (jnp.pi * gamma)

    # JAX-compatible: use jnp.where instead of if/else
    result = jnp.sqrt(gamma / (jnp.pi * rho) - gamma**2)
    return jnp.where(rho > max_density, 0.0, result)


def sigma_line(wavelength: float) -> float:
    """
    Calculate the cross-section (divided by gf) at wavelength for a transition.

    This is the cross-section for which the product of the degeneracy and
    oscillator strength is 10^log_gf.

    Args:
        wavelength: Wavelength in cm

    Returns:
        Cross-section in cm²
    """
    # The factor of |dλ/dν| = λ²/c is because we are working in wavelength
    # rather than frequency
    return (jnp.pi * electron_charge_cgs**2 /
            (electron_mass_cgs * c_cgs)) * (wavelength**2 / c_cgs)


def doppler_width(wavelength: float, temperature: float, mass: float,
                  xi: float) -> float:
    """
    Calculate the standard deviation of the Doppler-broadening profile.

    Note: This is σ, not σ√2 as the "Doppler width" is often defined.

    Args:
        wavelength: Line center wavelength in cm
        temperature: Temperature in K
        mass: Atomic/molecular mass in g
        xi: Microturbulent velocity in cm/s

    Returns:
        Doppler width σ in cm
    """
    return wavelength * jnp.sqrt(kboltz_cgs * temperature / mass +
                                  (xi**2) / 2) / c_cgs


def scaled_stark(gamma_stark: float, temperature: float, T0: float = 10_000) -> float:
    """
    Scale the Stark broadening gamma according to its temperature dependence.

    Args:
        gamma_stark: Stark broadening parameter at T0 (rad/s)
        temperature: Temperature in K
        T0: Reference temperature (default: 10,000 K)

    Returns:
        Scaled Stark broadening parameter in rad/s
    """
    return gamma_stark * (temperature / T0)**(1/6)


def scaled_vdW(vdW: Tuple[float, float], mass: float,
               temperature: float) -> float:
    """
    Scale the van der Waals broadening gamma according to temperature dependence.

    Uses either simple scaling or ABO theory depending on the format of vdW.

    Args:
        vdW: Either (γ_vdW at 10,000 K, -1) or (σ, α) for ABO theory
        mass: Species mass in g
        temperature: Temperature in K

    Returns:
        Scaled van der Waals broadening parameter in rad/s
    """
    # JAX-compatible: use jnp.where instead of if/else
    # Simple scaling: γ_vdW ∝ T^0.3
    simple_result = vdW[0] * (temperature / 10_000)**0.3

    # ABO theory
    v0 = 1e6  # σ is given at 10,000 m/s = 10^6 cm/s
    sigma = vdW[0]
    alpha = vdW[1]

    # Inverse reduced mass: 1/μ = 1/m_H + 1/m_species
    inv_mu = 1 / (1.008 * amu_cgs) + 1 / mass

    # Mean relative velocity
    vbar = jnp.sqrt(8 * kboltz_cgs * temperature / jnp.pi * inv_mu)

    # ABO formula (n.b. gamma here is the gamma function, not broadening)
    abo_result = (2 * (4 / jnp.pi)**(alpha / 2) *
                  gamma_function((4 - alpha) / 2) *
                  v0 * sigma * (vbar / v0)**(1 - alpha))

    # Return simple if vdW[1] == -1, otherwise ABO
    return jnp.where(vdW[1] == -1, simple_result, abo_result)


def line_profile(wavelength_center: float, sigma: float, gamma: float,
                amplitude: float, wavelength: float) -> float:
    """
    Calculate a Voigt profile centered on wavelength_center.

    Args:
        wavelength_center: Line center in cm
        sigma: Doppler width (NOT √2 σ) in cm
        gamma: Lorentz HWHM in cm
        amplitude: Total integrated absorption coefficient
        wavelength: Wavelength at which to evaluate in cm

    Returns:
        Absorption coefficient in cm⁻¹
    """
    inv_sigma_sqrt2 = 1 / (sigma * jnp.sqrt(2))
    scaling = inv_sigma_sqrt2 / jnp.sqrt(jnp.pi) * amplitude

    # Call voigt_hjerting with proper arguments
    return (voigt_hjerting(gamma * inv_sigma_sqrt2,
                          jnp.abs(wavelength - wavelength_center) * inv_sigma_sqrt2) *
            scaling)


# ============================================================================
# JIT-compatible implementation
# ============================================================================

def prepare_linelist_arrays(
    linelist: List[Line],
    unique_species: List[Species]
) -> Dict[str, jnp.ndarray]:
    """
    Convert list of Line objects to JAX arrays for JIT compilation.

    This function runs in Python (not JIT-compiled). If you're JIT-compiling
    your entire synthesis pipeline, call this ONCE before entering the JIT boundary.

    Args:
        linelist: List of Line objects
        unique_species: List of unique Species in linelist (determines species IDs)

    Returns:
        Dictionary of JAX arrays containing line properties
    """
    if len(linelist) == 0:
        # Return empty arrays with correct structure
        return {
            'wls': jnp.array([]),
            'log_gfs': jnp.array([]),
            'species_ids': jnp.array([], dtype=jnp.int32),
            'E_lowers': jnp.array([]),
            'gamma_rads': jnp.array([]),
            'gamma_starks': jnp.array([]),
            'vdW_params': jnp.zeros((0, 2)),
            'masses': jnp.array([]),
            'is_molecule': jnp.array([], dtype=bool)
        }

    species_to_id = {sp: i for i, sp in enumerate(unique_species)}

    return {
        'wls': jnp.array([line.wl for line in linelist]),
        'log_gfs': jnp.array([line.log_gf for line in linelist]),
        'species_ids': jnp.array([species_to_id[line.species] for line in linelist], dtype=jnp.int32),
        'E_lowers': jnp.array([line.E_lower for line in linelist]),
        'gamma_rads': jnp.array([line.gamma_rad for line in linelist]),
        'gamma_starks': jnp.array([line.gamma_stark for line in linelist]),
        'vdW_params': jnp.array([line.vdW for line in linelist]),  # (n_lines, 2)
        'masses': jnp.array([line.species.get_mass() for line in linelist]),
        'is_molecule': jnp.array([line.species.formula.is_molecule() for line in linelist], dtype=bool)
    }


@jax.jit
def line_absorption_core(
    # Line properties as JAX arrays
    line_wls: jnp.ndarray,           # (n_lines,)
    line_log_gfs: jnp.ndarray,       # (n_lines,)
    line_species_ids: jnp.ndarray,   # (n_lines,) int32
    line_E_lowers: jnp.ndarray,      # (n_lines,)
    line_gamma_rads: jnp.ndarray,    # (n_lines,)
    line_gamma_starks: jnp.ndarray,  # (n_lines,)
    line_vdW_params: jnp.ndarray,    # (n_lines, 2)
    line_masses: jnp.ndarray,        # (n_lines,)
    line_is_molecule: jnp.ndarray,   # (n_lines,) bool
    # Wavelength grid
    wavelengths: jnp.ndarray,        # (n_wl,)
    # Atmospheric structure
    temperatures: jnp.ndarray,       # (n_layers,)
    electron_densities: jnp.ndarray, # (n_layers,)
    # Number densities as 2D array indexed by species ID
    number_densities_array: jnp.ndarray,  # (n_species, n_layers)
    # Partition functions pre-evaluated at these temperatures
    partition_funcs_array: jnp.ndarray,   # (n_species, n_layers)
    # H I densities (needed for vdW broadening)
    H_I_densities: jnp.ndarray,      # (n_layers,)
    # Continuum opacity pre-evaluated at all line centers
    continuum_opacities: jnp.ndarray,  # (n_lines, n_layers)
    # Parameters
    xi: float,
    cutoff_threshold: float = 3e-4
) -> jnp.ndarray:
    """
    Core JIT-compiled line absorption calculation.

    This function is fully JAX-traceable and uses:
    - jax.lax.fori_loop instead of Python for loops
    - Masking instead of dynamic slicing
    - jnp.where instead of if/else statements

    Returns:
        Absorption coefficient array of shape (n_layers, n_wavelengths) in cm⁻¹
    """
    n_layers = len(temperatures)
    n_wl = len(wavelengths)
    n_lines = len(line_wls)

    # Precompute beta for Boltzmann factors
    beta = 1 / (kboltz_eV * temperatures)  # (n_layers,)

    def process_one_line(i_line, alpha_accum):
        """Process line i_line and add its contribution to alpha_accum."""

        # Extract line properties (all scalars after indexing)
        wl = line_wls[i_line]
        log_gf = line_log_gfs[i_line]
        species_id = line_species_ids[i_line]
        E_lower = line_E_lowers[i_line]
        gamma_rad = line_gamma_rads[i_line]
        gamma_stark = line_gamma_starks[i_line]
        vdW = line_vdW_params[i_line]  # (2,)
        mass = line_masses[i_line]
        is_molecule = line_is_molecule[i_line]

        # Get species-specific data (n_layers,)
        n_species = number_densities_array[species_id]
        U_species = partition_funcs_array[species_id]
        alpha_cont = continuum_opacities[i_line]  # (n_layers,)

        # === Compute line parameters for all layers (vectorized) ===

        # Doppler width (n_layers,)
        sigma_vals = jax.vmap(doppler_width, in_axes=(None, 0, None, None))(
            wl, temperatures, mass, xi
        )

        # Damping parameters (n_layers,)
        Gamma = jnp.full_like(temperatures, gamma_rad)

        # Add Stark and vdW only for atoms (not molecules)
        # Use jnp.where instead of if statement
        Gamma_stark = electron_densities * jax.vmap(
            scaled_stark, in_axes=(None, 0)
        )(gamma_stark, temperatures)
        Gamma = Gamma + jnp.where(is_molecule, 0.0, Gamma_stark)

        Gamma_vdW = H_I_densities * jax.vmap(
            scaled_vdW, in_axes=(None, None, 0)
        )(vdW, mass, temperatures)
        Gamma = Gamma + jnp.where(is_molecule, 0.0, Gamma_vdW)

        # Convert to wavelength units (n_layers,)
        gamma_vals = Gamma * wl**2 / (c_cgs * 4 * jnp.pi)

        # Boltzmann factor (n_layers,)
        E_upper = E_lower + c_cgs * hplanck_eV / wl
        levels_factor = jnp.exp(-beta * E_lower) - jnp.exp(-beta * E_upper)

        # Amplitude (n_layers,)
        amplitude = 10.0**log_gf * sigma_line(wl) * levels_factor * n_species / U_species

        # === Compute line window for each layer ===

        # Critical density for cutoff (n_layers,)
        rho_crit = alpha_cont * cutoff_threshold / amplitude

        # Doppler and Lorentz windows (n_layers,)
        doppler_windows = jax.vmap(inverse_gaussian_density, in_axes=(0, 0))(
            rho_crit, sigma_vals
        )
        lorentz_windows = jax.vmap(inverse_lorentz_density, in_axes=(0, 0))(
            rho_crit, gamma_vals
        )

        # Total window size (n_layers,)
        window_sizes = jnp.sqrt(lorentz_windows**2 + doppler_windows**2)

        # Maximum window across all layers (scalar)
        max_window = jnp.max(window_sizes)

        # === KEY CHANGE: Use masking instead of dynamic slicing ===

        # Global window mask: wavelengths within max_window of line center
        # Shape: (n_wl,) - True if wavelength could contribute to ANY layer
        wl_in_global_window = jnp.abs(wavelengths - wl) < max_window

        def compute_layer_contribution(layer_idx):
            """Compute line contribution for one layer across all wavelengths."""
            sigma = sigma_vals[layer_idx]
            gamma = gamma_vals[layer_idx]
            amp = amplitude[layer_idx]
            window_size = window_sizes[layer_idx]

            # Mask: wavelengths within THIS layer's window
            # Shape: (n_wl,)
            in_layer_window = jnp.abs(wavelengths - wl) < window_size

            # Compute Voigt profile at all wavelengths (vectorized)
            # Shape: (n_wl,)
            profiles = jax.vmap(line_profile, in_axes=(None, None, None, None, 0))(
                wl, sigma, gamma, amp, wavelengths
            )

            # Mask out wavelengths outside window
            # Combine both masks to minimize computation
            mask = in_layer_window & wl_in_global_window
            profiles_masked = jnp.where(mask, profiles, 0.0)

            return profiles_masked

        # Compute for all layers (n_layers, n_wl)
        line_contribution = jax.vmap(compute_layer_contribution)(
            jnp.arange(n_layers)
        )

        # Add to accumulated alpha
        return alpha_accum + line_contribution

    # Initialize alpha
    alpha_init = jnp.zeros((n_layers, n_wl))

    # Loop over all lines using JAX's fori_loop
    # This is JIT-compatible unlike Python for loop
    alpha = jax.lax.fori_loop(0, n_lines, process_one_line, alpha_init)

    return alpha


def line_absorption(
    linelist: List[Line],
    wavelengths: np.ndarray,
    temperatures: np.ndarray,
    electron_densities: np.ndarray,
    number_densities: Dict[Species, np.ndarray],
    partition_functions: Dict[Species, Callable],
    xi: float,
    continuum_opacity: Callable[[float], np.ndarray],
    cutoff_threshold: float = 3e-4,
    use_jit: bool = True
) -> np.ndarray:
    """
    Calculate the opacity coefficient from all lines in linelist.

    This function automatically uses the JIT-compiled implementation by default.
    Set use_jit=False to use the original Python implementation (slower but easier to debug).

    IMPORTANT: This function evaluates partition_functions and continuum_opacity
    to convert them to arrays for JIT compilation. If you're JIT-compiling your
    entire synthesis pipeline, use line_absorption_core() directly with pre-prepared data.

    Args:
        linelist: List of Line objects to include
        wavelengths: Wavelength grid in cm (1D array)
        temperatures: Temperature at each layer in K (1D array)
        electron_densities: Electron number density in cm⁻³ at each layer
        number_densities: Dict mapping Species to number densities (cm⁻³) at each layer
        partition_functions: Dict mapping Species to partition function callables
        xi: Microturbulent velocity in cm/s
        continuum_opacity: Callable that takes wavelength (cm) and returns continuum
                          opacity at each layer (returns 1D array)
        cutoff_threshold: Lines contribute opacity only where significant
                         (default: 3e-4 of continuum)
        use_jit: If True, use JIT-compiled implementation (default: True)

    Returns:
        Absorption coefficient array of shape (n_layers, n_wavelengths) in cm⁻¹
    """
    if len(linelist) == 0:
        return jnp.zeros((len(temperatures), len(wavelengths)))

    # Get unique species
    unique_species = list(set([line.species for line in linelist]))

    # Check for H I in linelist
    if Species("H_I") in unique_species:
        raise ValueError("Atomic hydrogen should not be in the linelist. "
                        "Use hydrogen_line_absorption for H lines.")

    if not use_jit:
        # Use original Python implementation
        return _line_absorption_python(
            linelist, wavelengths, temperatures, electron_densities,
            number_densities, partition_functions, xi, continuum_opacity,
            cutoff_threshold
        )

    # === JIT path: prepare data ===

    # Convert linelist to arrays
    line_arrays = prepare_linelist_arrays(linelist, unique_species)

    # Convert number_densities dict to 2D array
    n_species = len(unique_species)
    n_layers = len(temperatures)
    species_to_id = {sp: i for i, sp in enumerate(unique_species)}

    number_densities_array = jnp.zeros((n_species, n_layers))
    for sp, idx in species_to_id.items():
        if sp in number_densities:
            number_densities_array = number_densities_array.at[idx].set(
                number_densities[sp]
            )

    # Pre-evaluate partition functions
    log_temps = jnp.log(temperatures)
    partition_funcs_array = jnp.zeros((n_species, n_layers))
    for sp, idx in species_to_id.items():
        if sp in partition_functions:
            U_vals = jnp.array([partition_functions[sp](lt) for lt in log_temps])
            partition_funcs_array = partition_funcs_array.at[idx].set(U_vals)

    # Pre-evaluate continuum opacity at all line centers
    continuum_opacities = jnp.array([
        continuum_opacity(line.wl) for line in linelist
    ])  # (n_lines, n_layers)

    # Get H I densities
    H_I_species = Species("H_I")
    H_I_densities = number_densities.get(H_I_species, jnp.zeros(n_layers))

    # Call JIT-compiled core
    return line_absorption_core(
        line_wls=line_arrays['wls'],
        line_log_gfs=line_arrays['log_gfs'],
        line_species_ids=line_arrays['species_ids'],
        line_E_lowers=line_arrays['E_lowers'],
        line_gamma_rads=line_arrays['gamma_rads'],
        line_gamma_starks=line_arrays['gamma_starks'],
        line_vdW_params=line_arrays['vdW_params'],
        line_masses=line_arrays['masses'],
        line_is_molecule=line_arrays['is_molecule'],
        wavelengths=wavelengths,
        temperatures=temperatures,
        electron_densities=electron_densities,
        number_densities_array=number_densities_array,
        partition_funcs_array=partition_funcs_array,
        H_I_densities=H_I_densities,
        continuum_opacities=continuum_opacities,
        xi=xi,
        cutoff_threshold=cutoff_threshold
    )


# ============================================================================
# Original Python implementation (kept for reference and testing)
# ============================================================================

def _line_absorption_python(
    linelist: List[Line],
    wavelengths: np.ndarray,
    temperatures: np.ndarray,
    electron_densities: np.ndarray,
    number_densities: Dict[Species, np.ndarray],
    partition_functions: Dict[Species, Callable],
    xi: float,
    continuum_opacity: Callable[[float], np.ndarray],
    cutoff_threshold: float = 3e-4
) -> np.ndarray:
    """
    Python implementation of line absorption (non-JIT compatible).

    This is the original implementation kept for reference and testing.
    Use line_absorption() with use_jit=True for the JIT-compatible version.

    Calculate the opacity coefficient from all lines in linelist.

    This is the main function for computing line absorption. It handles:
    - Voigt profile calculation for each line
    - Temperature-dependent broadening (Doppler, Stark, van der Waals)
    - Line cutoff based on continuum opacity
    - Boltzmann populations

    Args:
        linelist: List of Line objects to include
        wavelengths: Wavelength grid in cm (1D array)
        temperatures: Temperature at each layer in K (1D array)
        electron_densities: Electron number density in cm⁻³ at each layer
        number_densities: Dict mapping Species to number densities (cm⁻³) at each layer
        partition_functions: Dict mapping Species to partition function callables
        xi: Microturbulent velocity in cm/s
        continuum_opacity: Callable that takes wavelength (cm) and returns continuum
                          opacity at each layer (returns 1D array)
        cutoff_threshold: Lines contribute opacity only where significant
                         (default: 3e-4 of continuum)

    Returns:
        Absorption coefficient array of shape (n_layers, n_wavelengths) in cm⁻¹
    """
    if len(linelist) == 0:
        return jnp.zeros((len(temperatures), len(wavelengths)))

    # Precompute beta = 1/(kT) for Boltzmann factors
    beta = 1 / (kboltz_eV * temperatures)

    # Precompute number density / partition function for each species
    n_div_U = {}
    unique_species = list(set([line.species for line in linelist]))

    for spec in unique_species:
        if spec not in number_densities:
            raise ValueError(f"Species {spec} in linelist but not in number_densities")

        # Partition functions take log(T)
        log_temps = jnp.log(temperatures)
        U_values = jnp.array([partition_functions[spec](lt) for lt in log_temps])
        n_div_U[spec] = number_densities[spec] / U_values

    # Initialize absorption coefficient array
    alpha = jnp.zeros((len(temperatures), len(wavelengths)))

    # Process each line
    for line in linelist:
        mass = line.species.get_mass()

        # Doppler broadening width (σ, NOT √2 σ)
        sigma_vals = jnp.array([doppler_width(line.wl, T, mass, xi)
                                for T in temperatures])

        # Sum up damping parameters (these are FWHM in angular frequency)
        Gamma = jnp.full_like(temperatures, line.gamma_rad)

        # Add Stark and van der Waals for atoms (not molecules)
        if not line.species.formula.is_molecule():
            # Stark broadening
            Gamma = Gamma + electron_densities * jnp.array([
                scaled_stark(line.gamma_stark, T) for T in temperatures
            ])

            # van der Waals broadening (need H I density)
            if Species("H_I") in number_densities:
                Gamma = Gamma + number_densities[Species("H_I")] * jnp.array([
                    scaled_vdW(line.vdW, mass, T) for T in temperatures
                ])

        # Convert to Lorentz broadening parameter in wavelength units
        # Factor of λ²/c is |dλ/dν|, 1/(2π) for angular vs cyclical frequency,
        # and 1/2 for FWHM vs HWHM
        gamma_vals = Gamma * line.wl**2 / (c_cgs * 4 * jnp.pi)

        # Calculate energy levels and Boltzmann factor
        E_upper = line.E_lower + c_cgs * hplanck_eV / line.wl
        levels_factor = jnp.exp(-beta * line.E_lower) - jnp.exp(-beta * E_upper)

        # Total wavelength-integrated absorption coefficient
        amplitude = (10.0**line.log_gf * sigma_line(line.wl) *
                    levels_factor * n_div_U[line.species])

        # Determine line window based on cutoff threshold
        # Get continuum opacity at line center
        alpha_continuum_at_line = continuum_opacity(line.wl)

        # Critical density for cutoff
        rho_crit = alpha_continuum_at_line * cutoff_threshold / amplitude

        # Doppler wing extent
        inverse_densities_doppler = jnp.array([
            inverse_gaussian_density(rc, s)
            for rc, s in zip(rho_crit, sigma_vals)
        ])
        doppler_line_window = jnp.max(inverse_densities_doppler)

        # Lorentz wing extent
        inverse_densities_lorentz = jnp.array([
            inverse_lorentz_density(rc, g)
            for rc, g in zip(rho_crit, gamma_vals)
        ])
        lorentz_line_window = jnp.max(inverse_densities_lorentz)

        # Combined window (Pythagorean sum)
        window_size = jnp.sqrt(lorentz_line_window**2 + doppler_line_window**2)

        # Find wavelength indices in window
        lb = jnp.searchsorted(wavelengths, line.wl - window_size)
        ub = jnp.searchsorted(wavelengths, line.wl + window_size, side='right')

        # Skip if window is empty
        if lb >= ub:
            continue

        # Calculate line profile for all layers and wavelengths in window
        wl_window = wavelengths[lb:ub]

        # Vectorized calculation over layers and wavelengths
        for i_layer in range(len(temperatures)):
            profiles = jnp.array([
                line_profile(line.wl, sigma_vals[i_layer], gamma_vals[i_layer],
                           amplitude[i_layer], wl)
                for wl in wl_window
            ])
            alpha = alpha.at[i_layer, lb:ub].add(profiles)

    return alpha
