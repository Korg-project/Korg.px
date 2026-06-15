"""
Line absorption calculation for atomic lines.

This module computes opacity from atomic spectral lines using Voigt profiles,
including proper treatment of Doppler and pressure broadening.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
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

# Numpy scalar constants for the fast Python-loop implementation
kboltz_cgs_np = None  # filled lazily
c_cgs_np = None
amu_cgs_np = None
hplanck_eV_np = None

def _init_np_consts():
    global kboltz_cgs_np, c_cgs_np, amu_cgs_np, hplanck_eV_np
    if kboltz_cgs_np is None:
        from .constants import kboltz_cgs, c_cgs, amu_cgs, hplanck_eV
        kboltz_cgs_np = float(kboltz_cgs)
        c_cgs_np = float(c_cgs)
        amu_cgs_np = float(amu_cgs)
        hplanck_eV_np = float(hplanck_eV)
_init_np_consts()


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

def _vdW_to_tuple(vdW):
    """Convert vdW field (scalar, tuple, or None) to (gamma_or_sigma, alpha) pair."""
    if vdW is None:
        return (0.0, -1.0)
    if isinstance(vdW, (tuple, list)):
        return (float(vdW[0]), float(vdW[1]))
    v = float(vdW)
    if v < 0:
        return (10**v, -1.0)   # log10(gamma_vdW) → linear
    if v == 0:
        return (0.0, -1.0)
    if v < 20:
        return (v, -1.0)       # fudge factor; scaled_vdW treats [1] == -1 as simple scaling
    # ABO encoding: integer part = sigma/a0^2, fractional part = alpha
    import math
    sigma_over_a0sq = math.floor(v)
    alpha = v - sigma_over_a0sq
    from .constants import bohr_radius_cgs
    return (sigma_over_a0sq * bohr_radius_cgs**2, alpha)


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
        'gamma_starks': jnp.array([line.gamma_stark if line.gamma_stark is not None else 0.0 for line in linelist]),
        'vdW_params': jnp.array([_vdW_to_tuple(line.vdW) for line in linelist]),  # (n_lines, 2)
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
        return np.zeros((len(temperatures), len(wavelengths)))

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

    return _line_absorption_fast(
        linelist, unique_species, wavelengths, temperatures, electron_densities,
        number_densities, partition_functions, xi, continuum_opacity, cutoff_threshold
    )


# --------------------------------------------------------------------------
# JAX Voigt profile via Harris series (Hunger 1965 / Hjerting), matching Julia
# --------------------------------------------------------------------------

def _harris_H1_jax(v, v2):
    """Piecewise H1 component for Harris series (branchless via jnp.where)."""
    H1_lo = (-1.12470432 + (-0.15516677 + (3.288675912 + (-2.34357915 + 0.42139162 * v) * v) * v) * v)
    H1_mid = (-4.48480194 + (9.39456063 + (-6.61487486 + (1.98919585 - 0.22041650 * v) * v) * v) * v)
    # v >= 2.4 for this branch, so v2 - 1.5 >= 4.26 > 0 (no singularity)
    H1_hi = ((0.554153432 + (0.278711796 + (-0.1883256872 + (0.042991293 - 0.003278278 * v) * v) * v) * v) /
             (v2 - 1.5))
    return jnp.where(v < 1.3, H1_lo, jnp.where(v < 2.4, H1_mid, H1_hi))


def _voigt_hjerting_jax(alpha, v):
    """
    Voigt-Hjerting function H(alpha, v) = Re[w(v+i*alpha)].

    Matches Julia Korg.jl's voigt_hjerting exactly (Hunger 1965 Harris series).
    Branchless: all cases evaluated, jnp.where selects. alpha >= 0, v >= 0.
    """
    v2 = v * v
    sqrt_pi = jnp.sqrt(jnp.pi)

    # Harris series components (shared between cases 2 and 3)
    H0 = jnp.exp(-v2)
    H1 = _harris_H1_jax(v, v2)
    H2 = (1.0 - 2.0 * v2) * H0

    # Case 1: alpha <= 0.2, v >= 5  →  asymptotic correction
    safe_v2 = jnp.where(v2 > 0, v2, 1.0)
    invv2 = 1.0 / safe_v2
    r1 = (alpha / sqrt_pi * invv2) * (1.0 + 1.5 * invv2 + 3.75 * invv2 * invv2)

    # Case 2: alpha <= 0.2, v < 5  →  Harris series
    r2 = H0 + (H1 + H2 * alpha) * alpha

    # Case 3: alpha <= 1.4, alpha+v < 3.2  →  modified Harris series (Hunger 1965)
    inv_sqrt_pi = 1.0 / sqrt_pi
    two_inv_sqrt_pi = 2.0 * inv_sqrt_pi
    M0 = H0
    M1 = H1 + two_inv_sqrt_pi * M0
    M2 = H2 - M0 + two_inv_sqrt_pi * M1
    M3 = (2.0 / (3.0 * sqrt_pi)) * (1.0 - H2) - (2.0 / 3.0) * v2 * M1 + two_inv_sqrt_pi * M2
    M4 = (2.0 / 3.0) * v2 * v2 * M0 - (two_inv_sqrt_pi / 3.0) * M1 + two_inv_sqrt_pi * M3
    psi = 0.979895023 + (-0.962846325 + (0.532770573 - 0.122727278 * alpha) * alpha) * alpha
    r3 = psi * (M0 + (M1 + (M2 + (M3 + M4 * alpha) * alpha) * alpha) * alpha)

    # Case 4: else (large alpha or large alpha+v)  →  Lorentz-like
    safe_alpha = jnp.where(alpha > 0, alpha, 1.0)
    r2_lorentz = v2 / (safe_alpha * safe_alpha)
    alpha_invu = 1.0 / (jnp.sqrt(2.0) * (r2_lorentz + 1.0) * safe_alpha)
    a2_inv_u2 = alpha_invu * alpha_invu
    r4 = (jnp.sqrt(2.0 / jnp.pi) * alpha_invu *
          (1.0 + (3.0 * r2_lorentz - 1.0 + ((r2_lorentz - 2.0) * 15.0 * r2_lorentz + 2.0) * a2_inv_u2) * a2_inv_u2))

    cond1 = (alpha <= 0.2) & (v >= 5.0)
    cond2 = (alpha <= 0.2) & (v < 5.0)
    cond3 = (alpha <= 1.4) & (alpha + v < 3.2)
    return jnp.where(cond1, r1,
           jnp.where(cond2, r2,
           jnp.where(cond3, r3,
                            r4)))


def _voigt_profile_jax(delta, sigma, gamma):
    """Voigt profile (area-normalized). Matches scipy.special.voigt_profile."""
    s2 = sigma * jnp.sqrt(2.0)
    return _voigt_hjerting_jax(gamma / s2, jnp.abs(delta) / s2) / (sigma * jnp.sqrt(2.0 * jnp.pi))


# Module-level cache: (tuple of line ids) → (unique_species, per-line numpy arrays)
_LINELIST_PREP_CACHE = {}

def _line_absorption_fast(
    linelist, unique_species, wavelengths, temperatures, electron_densities,
    number_densities, partition_functions, xi, continuum_opacity, cutoff_threshold
):
    """
    Fast line absorption using bucketed JAX Voigt computation.

    Lines are grouped by window size. Each bucket evaluates Voigt profiles
    for all lines simultaneously (vectorized via XLA) then scatters into alpha.
    Uses _voigt_profile_jax (Harris series, matches Julia Korg.jl exactly).
    """
    n_layers = len(temperatures)
    n_wl = len(wavelengths)
    wl_np = np.asarray(wavelengths)
    T_np = np.asarray(temperatures)
    ne_np = np.asarray(electron_densities)

    # Cache line arrays keyed by tuple of Line object ids
    cache_key = tuple(id(l) for l in linelist)
    if cache_key not in _LINELIST_PREP_CACHE:
        _LINELIST_PREP_CACHE[cache_key] = _build_line_data(linelist, unique_species)
    ld = _LINELIST_PREP_CACHE[cache_key]

    # Number densities for each unique species: (n_species, n_layers)
    n_species = len(unique_species)
    nd_arr = np.zeros((n_species, n_layers))
    for sp, idx in ld['species_to_id'].items():
        if sp in number_densities:
            nd_arr[idx] = np.asarray(number_densities[sp])

    # Partition functions: (n_species, n_layers)
    log_T_np = np.log(T_np)
    pf_arr = np.zeros((n_species, n_layers))
    pf_available = np.zeros(n_species, dtype=bool)
    for sp, idx in ld['species_to_id'].items():
        if sp in partition_functions:
            pf = partition_functions[sp]
            if hasattr(pf, 'numpy_eval'):
                pf_arr[idx] = pf.numpy_eval(log_T_np)
            else:
                for j, lt in enumerate(log_T_np):
                    pf_arr[idx, j] = float(pf(lt))
            pf_available[idx] = True

    # H I densities for vdW broadening
    H_I_species = Species("H_I")
    nH_I = np.asarray(number_densities.get(H_I_species, np.zeros(n_layers)))

    # Continuum opacities at all line centers: batch if possible
    wl_centers = ld['wls']  # (n_lines,) numpy array
    try:
        cntm_opac = np.asarray(continuum_opacity(wl_centers))  # (n_lines, n_layers)
        if cntm_opac.shape != (len(linelist), n_layers):
            raise ValueError("unexpected shape")
    except Exception:
        cntm_opac = np.stack([np.asarray(continuum_opacity(w)) for w in wl_centers])

    # Precomputed constants
    pi_e2_mc = np.pi * float(electron_charge_cgs)**2 / (float(electron_mass_cgs) * float(c_cgs))
    beta = 1.0 / (float(kboltz_eV) * T_np)  # (n_layers,)
    sqrt2pi = np.sqrt(2 * np.pi)
    inv_10000 = 1.0 / 10_000.0

    # --- Vectorized pre-computation over all lines ---
    wls = ld['wls']         # (n_lines,)
    masses = ld['masses']   # (n_lines,)
    is_mol = ld['is_molecule']  # (n_lines,) bool
    vdW_arr = ld['vdW_params']  # (n_lines, 2)

    # Doppler width: (n_lines, n_layers)
    sigma_all = wls[:, None] * np.sqrt(
        kboltz_cgs_np * T_np[None, :] / masses[:, None] + xi**2 / 2
    ) / c_cgs_np

    # Damping Γ (n_lines, n_layers) — start with gamma_rad
    Gamma_all = ld['gamma_rads'][:, None] * np.ones((1, n_layers))  # broadcast copy

    # Stark: only for non-molecules
    stark_contrib = (ld['gamma_starks'][:, None] *
                     ne_np[None, :] * (T_np[None, :] * inv_10000)**(1/6))
    Gamma_all += np.where(is_mol[:, None], 0.0, stark_contrib)

    # vdW: separate simple (vdW[:,1] == -1) vs ABO
    simple_mask = (vdW_arr[:, 1] == -1.0) & ~is_mol
    abo_mask = (vdW_arr[:, 1] != -1.0) & ~is_mol

    if np.any(simple_mask):
        vdW_simple = vdW_arr[simple_mask, 0:1]  # (n_simple, 1)
        vdW_contrib = vdW_simple * nH_I[None, :] * (T_np[None, :] * inv_10000)**0.3
        Gamma_all[simple_mask] += vdW_contrib

    if np.any(abo_mask):
        from scipy.special import gamma as gamma_fn
        v0 = 1e6
        for idx in np.where(abo_mask)[0]:
            alpha_abo = vdW_arr[idx, 1]
            sigma_abo = vdW_arr[idx, 0]
            inv_mu = 1.0 / (1.008 * amu_cgs_np) + 1.0 / masses[idx]
            vbar = np.sqrt(8 * kboltz_cgs_np * T_np / np.pi * inv_mu)
            Gamma_all[idx] += nH_I * (2 * (4/np.pi)**(alpha_abo/2) *
                                       gamma_fn((4 - alpha_abo) / 2) *
                                       v0 * sigma_abo * (vbar / v0)**(1 - alpha_abo))

    # Convert Γ → wavelength HWHM in cm: (n_lines, n_layers)
    gamma_wl_all = Gamma_all * wls[:, None]**2 / (c_cgs_np * 4 * np.pi)

    # Amplitude: (n_lines, n_layers)
    sigma_ln_all = pi_e2_mc * wls**2 / c_cgs_np  # (n_lines,)
    E_upper_all = ld['E_lowers'] + c_cgs_np * hplanck_eV_np / wls  # (n_lines,)
    levels_all = (np.exp(-beta[None, :] * ld['E_lowers'][:, None]) -
                  np.exp(-beta[None, :] * E_upper_all[:, None]))  # (n_lines, n_layers)
    n_sp_all = nd_arr[ld['species_ids']]   # (n_lines, n_layers)
    U_sp_all = pf_arr[ld['species_ids']]   # (n_lines, n_layers)
    line_has_pf = pf_available[ld['species_ids']]  # (n_lines,) — False for species without pf
    amplitude_all = (10.0**ld['log_gfs'][:, None] * sigma_ln_all[:, None] *
                     levels_all * n_sp_all / np.maximum(U_sp_all, 1e-300))  # (n_lines, n_layers)
    # Zero out lines with no partition function (avoids division-by-~0 amplitudes)
    amplitude_all = np.where(line_has_pf[:, None], amplitude_all, 0.0)

    # Window sizes: (n_lines, n_layers)
    rho_crit_all = (cntm_opac * cutoff_threshold /
                    np.maximum(np.abs(amplitude_all), 1e-300))  # (n_lines, n_layers)
    with np.errstate(invalid='ignore', divide='ignore'):
        log_arg = np.sqrt(2 * np.pi) * sigma_all * rho_crit_all
        win_G_all = np.where(log_arg >= 1.0, 0.0,
                             sigma_all * np.sqrt(-2 * np.log(np.maximum(log_arg, 1e-300))))
        win_L_all = np.where(rho_crit_all >= 1.0 / (np.pi * gamma_wl_all), 0.0,
                             np.sqrt(np.maximum(gamma_wl_all / (np.pi * rho_crit_all) -
                                                gamma_wl_all**2, 0.0)))
    # Max window per line: (n_lines,)
    max_wins = np.max(np.sqrt(win_G_all**2 + win_L_all**2), axis=1)

    # --- Bucketed JAX Voigt accumulation ---
    # Lines are grouped by window size so each bucket fits a fixed W_MAX-pixel window.
    # Within each bucket, _voigt_profile_jax runs on a vectorized (n_b, n_layers, W_MAX)
    # array (no Python per-line overhead, XLA-fused). Scatter uses numpy slice-add.
    _voigt_jit = jax.jit(_voigt_profile_jax)

    # For non-contiguous grids (concatenated wavelength windows), the global
    # first-to-last spacing is far larger than the actual pixel spacing.
    # Use the median adjacent spacing so line windows are correctly sized
    # regardless of whether the grid has gaps.
    diffs = np.diff(wl_np)
    wl_spacing = float(np.median(diffs)) if len(diffs) > 0 else float(wl_np[-1] - wl_np[0])
    # Clip to n_wl so lines with huge windows (e.g., tiny continuum opacity) still
    # land in the last bucket rather than silently dropping.
    max_wins_px = np.clip(
        (np.ceil(2.0 * max_wins / wl_spacing) + 2).astype(int), 0, n_wl
    )

    alpha = np.zeros((n_layers, n_wl))

    BUCKET_WIDTHS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, n_wl]
    prev_W = 0
    for W_MAX_raw in BUCKET_WIDTHS:
        W_MAX = min(W_MAX_raw, n_wl)
        in_bucket = (max_wins_px > prev_W) & (max_wins_px <= W_MAX_raw)
        n_b = int(in_bucket.sum())
        if n_b == 0:
            prev_W = W_MAX_raw
            continue

        idx_b = np.where(in_bucket)[0]

        i_lo_b = np.searchsorted(wl_np, wls[idx_b] - max_wins[idx_b]).astype(int)
        i_lo_b = np.clip(i_lo_b, 0, n_wl - W_MAX)

        # Wavelength windows: (n_b, W_MAX)
        pix_idx = i_lo_b[:, None] + np.arange(W_MAX, dtype=int)[None, :]
        wl_win  = wl_np[pix_idx]

        # Window mask: (n_b, W_MAX)
        mask_b = np.abs(wl_win - wls[idx_b, None]) <= max_wins[idx_b, None]

        # JAX Voigt profiles: broadcast to (n_b, n_layers, W_MAX) then compute
        delta = jnp.asarray((wl_win - wls[idx_b, None])[:, None, :])  # (n_b, 1, W_MAX)
        sigma = jnp.asarray(sigma_all[idx_b, :, None])                  # (n_b, n_layers, 1)
        gamma = jnp.asarray(gamma_wl_all[idx_b, :, None])               # (n_b, n_layers, 1)

        profiles = np.asarray(_voigt_jit(delta, sigma, gamma))  # (n_b, n_layers, W_MAX)

        contrib = mask_b[:, None, :] * amplitude_all[idx_b, :, None] * profiles

        for il, i_lo in enumerate(i_lo_b):
            alpha[:, i_lo:i_lo + W_MAX] += contrib[il]

        prev_W = W_MAX_raw

    return alpha


def _build_line_data(linelist, unique_species):
    """Build numpy arrays from linelist (cached per unique linelist)."""
    species_to_id = {sp: i for i, sp in enumerate(unique_species)}
    n = len(linelist)
    wls = np.array([l.wl for l in linelist])
    log_gfs = np.array([l.log_gf for l in linelist])
    species_ids = np.array([species_to_id[l.species] for l in linelist], dtype=np.int32)
    E_lowers = np.array([l.E_lower for l in linelist])
    gamma_rads = np.array([l.gamma_rad for l in linelist])
    gamma_starks = np.array([l.gamma_stark if l.gamma_stark is not None else 0.0
                              for l in linelist])
    vdW_params = np.array([_vdW_to_tuple(l.vdW) for l in linelist])  # (n, 2)
    masses = np.array([l.species.get_mass() for l in linelist])
    is_molecule = np.array([l.species.formula.is_molecule() for l in linelist])
    return {
        'wls': wls, 'log_gfs': log_gfs, 'species_ids': species_ids,
        'E_lowers': E_lowers, 'gamma_rads': gamma_rads,
        'gamma_starks': gamma_starks, 'vdW_params': vdW_params,
        'masses': masses, 'is_molecule': is_molecule,
        'species_to_id': species_to_id,
    }


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
