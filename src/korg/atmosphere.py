"""
Model atmosphere data structures for stellar spectral synthesis.

Provides basic atmosphere representations with temperature, density, and
pressure stratification. Full MARCS interpolation to be added in future.

Reference: Korg.jl atmosphere.jl
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
import jax.numpy as jnp


@dataclass
class PlanarAtmosphereLayer:
    """
    A single layer of a plane-parallel atmosphere.

    Attributes
    ----------
    tau_ref : float
        Optical depth at reference wavelength (dimensionless)
    z : float
        Height above photosphere [cm]
    temperature : float
        Temperature [K]
    electron_number_density : float
        Electron number density [cm⁻³]
    number_density : float
        Total number density [cm⁻³]
    """
    tau_ref: float
    z: float
    temperature: float
    electron_number_density: float
    number_density: float


@dataclass
class PlanarAtmosphere:
    """
    A plane-parallel stellar atmosphere model.

    The atmosphere is specified as a vertical stratification of layers,
    with the photosphere at z = 0.

    Attributes
    ----------
    layers : list of PlanarAtmosphereLayer
        Atmospheric layers from surface to depth
    reference_wavelength : float
        Wavelength at which tau_ref is defined [cm]
        Default: 5000 Å = 5e-5 cm (MARCS standard)
    """
    layers: list
    reference_wavelength: float = 5e-5  # 5000 Å in cm

    def __len__(self):
        return len(self.layers)

    @property
    def n_layers(self):
        """Number of atmospheric layers."""
        return len(self.layers)

    @property
    def tau_ref(self):
        """Array of reference optical depths."""
        return np.array([layer.tau_ref for layer in self.layers])

    @property
    def log_tau_ref(self):
        """Array of log10(tau_ref)."""
        return np.log10(self.tau_ref)

    @property
    def z(self):
        """Array of heights [cm]."""
        return np.array([layer.z for layer in self.layers])

    @property
    def T(self):
        """Array of temperatures [K]."""
        return np.array([layer.temperature for layer in self.layers])

    @property
    def ne(self):
        """Array of electron densities [cm⁻³]."""
        return np.array([layer.electron_number_density for layer in self.layers])

    @property
    def n_total(self):
        """Array of total number densities [cm⁻³]."""
        return np.array([layer.number_density for layer in self.layers])

    def __repr__(self):
        return f"PlanarAtmosphere with {self.n_layers} layers"


@dataclass
class ShellAtmosphereLayer:
    """
    A single layer of a spherical shell atmosphere.

    Attributes
    ----------
    tau_ref : float
        Optical depth at reference wavelength (dimensionless)
    z : float
        Height above photosphere [cm] (use R_photosphere + z to get radius from center)
    temperature : float
        Temperature [K]
    electron_number_density : float
        Electron number density [cm⁻³]
    number_density : float
        Total number density [cm⁻³]
    """
    tau_ref: float
    z: float  # height above photosphere, NOT radius from center
    temperature: float
    electron_number_density: float
    number_density: float


@dataclass
class ShellAtmosphere:
    """
    A spherical shell stellar atmosphere model.

    Used for evolved stars (giants) where spherical geometry is important.

    Attributes
    ----------
    layers : list of ShellAtmosphereLayer
        Atmospheric layers from inner to outer radius
    R_photosphere : float
        Photospheric radius [cm] (where tau_ref = 1)
    reference_wavelength : float
        Wavelength at which tau_ref is defined [cm]
        Default: 5000 Å = 5e-5 cm
    """
    layers: list
    R_photosphere: float
    reference_wavelength: float = 5e-5

    def __len__(self):
        return len(self.layers)

    @property
    def n_layers(self):
        """Number of atmospheric layers."""
        return len(self.layers)

    @property
    def tau_ref(self):
        """Array of reference optical depths."""
        return np.array([layer.tau_ref for layer in self.layers])

    @property
    def log_tau_ref(self):
        """Array of log10(tau_ref)."""
        return np.log10(self.tau_ref)

    @property
    def z(self):
        """Array of heights above photosphere [cm]."""
        return np.array([layer.z for layer in self.layers])

    @property
    def T(self):
        """Array of temperatures [K]."""
        return np.array([layer.temperature for layer in self.layers])

    @property
    def ne(self):
        """Array of electron densities [cm⁻³]."""
        return np.array([layer.electron_number_density for layer in self.layers])

    @property
    def n_total(self):
        """Array of total number densities [cm⁻³]."""
        return np.array([layer.number_density for layer in self.layers])

    def __repr__(self):
        return f"ShellAtmosphere with {self.n_layers} layers, R={self.R_photosphere:.2e} cm"


def create_simple_atmosphere(
    temperatures,
    log_tau_ref_values,
    logg=4.44,
    mass_g=1.989e33,  # Solar mass in grams
    reference_wavelength=5e-5,
    spherical=False
):
    """
    Create a simple model atmosphere from temperature stratification.

    This is a utility function for testing and simple cases. For production
    synthesis, use interpolate_marcs() (to be implemented).

    Parameters
    ----------
    temperatures : array, shape (n_layers,)
        Temperature at each layer [K]
    log_tau_ref_values : array, shape (n_layers,)
        log10(optical depth) at reference wavelength
    logg : float, optional
        log10(surface gravity in cm/s²), default: 4.44 (solar)
    mass_g : float, optional
        Stellar mass [g], default: 1.989e33 (solar mass)
    reference_wavelength : float, optional
        Reference wavelength [cm], default: 5e-5 (5000 Å)
    spherical : bool, optional
        If True, create ShellAtmosphere; else PlanarAtmosphere

    Returns
    -------
    atmosphere : PlanarAtmosphere or ShellAtmosphere
        Model atmosphere structure

    Notes
    -----
    This function makes simplifying assumptions:
    - Hydrostatic equilibrium for pressure/density structure
    - Ideal gas law for number density
    - Electron density from approximate ionization (10% of H I)
    - Height scale from pressure scale height

    For realistic atmospheres, use MARCS interpolation.

    Examples
    --------
    >>> # Create a simple solar-type atmosphere
    >>> T = np.linspace(8000, 4000, 56)  # 56 layers
    >>> log_tau = np.linspace(-4, 2, 56)
    >>> atm = create_simple_atmosphere(T, log_tau)
    >>> atm.n_layers
    56
    """
    from .constants import kboltz_cgs, amu_cgs

    temperatures = np.asarray(temperatures)
    log_tau_ref_values = np.asarray(log_tau_ref_values)
    n_layers = len(temperatures)

    if len(log_tau_ref_values) != n_layers:
        raise ValueError("temperatures and log_tau_ref_values must have same length")

    # Convert log(tau) to tau
    tau_ref_values = 10.0 ** log_tau_ref_values

    # Surface gravity in cm/s²
    g = 10.0 ** logg

    # Estimate pressure structure using hydrostatic equilibrium
    # P = integral(rho * g * dz) ≈ integral(P / (kT) * mu * g * dz)
    # For simplicity, use exponential atmosphere with scale height H = kT / (mu * g)
    mean_molecular_weight = 1.3  # Approximate for solar composition
    mu = mean_molecular_weight * amu_cgs

    # Pressure scale heights [cm]
    H = kboltz_cgs * temperatures / (mu * g)

    # Estimate pressures using optical depth as a proxy for column mass
    # In hydrostatic equilibrium: dP/dz = -ρ*g
    # For an isothermal atmosphere: P ∝ exp(-z/H) where H = kT/(μg)
    # Approximate relation: log(P) ≈ a + b*log(tau) for photospheric layers

    # Reference pressure at tau=1 (typical photospheric pressure)
    P_ref = 1e5  # dyne/cm² (reasonable for solar photosphere)

    # Use simple power law: P ∝ tau^0.6 (empirical fit for stellar atmospheres)
    # This gives smooth pressure variation across all layers
    pressures = P_ref * tau_ref_values**0.6

    # Total number density from ideal gas: n = P / (kT)
    number_densities = pressures / (kboltz_cgs * temperatures)

    # Find index where tau ≈ 1 (photosphere definition)
    tau_ref_unity_idx = np.argmin(np.abs(tau_ref_values - 1.0))

    # Estimate electron density using Saha equation for H ionization
    # This provides a much better initial guess than a fixed fraction
    # Assume ~90% of particles are hydrogen by number
    from .constants import hplanck_cgs

    electron_densities = np.zeros(n_layers)
    for i in range(n_layers):
        T = temperatures[i]
        n_total = number_densities[i]

        # Hydrogen ionization energy
        chi_H = 13.598 * 1.60218e-12  # eV to erg

        # Saha equation for H I/H II equilibrium
        # Assume hydrogen is 90% of particles by number
        n_H_total = 0.9 * n_total

        # Partition functions (approximate)
        U_HI = 2.0  # ground state degeneracy
        U_HII = 1.0

        # Saha constant
        saha_const = (2.0 * np.pi * 9.10938e-28 * kboltz_cgs * T / hplanck_cgs**2)**(1.5)

        # Solve Saha equation: ne * n_HII / n_HI = saha_const * (2*U_HII/U_HI) * exp(-chi/kT)
        # With n_HI + n_HII = n_H_total and ne ≈ n_HII (assuming H dominates ionization)
        exp_factor = np.exp(-chi_H / (kboltz_cgs * T))
        saha_rhs = saha_const * (2.0 * U_HII / U_HI) * exp_factor

        # Solve quadratic: ne^2 + saha_rhs * ne - saha_rhs * n_H_total = 0
        # Using quadratic formula
        a = 1.0
        b = saha_rhs
        c = -saha_rhs * n_H_total
        discriminant = b**2 - 4*a*c

        if discriminant > 0:
            ne_estimate = (-b + np.sqrt(discriminant)) / (2*a)
            # Ensure reasonable bounds
            electron_densities[i] = np.clip(ne_estimate, 1e-10 * n_total, n_total)
        else:
            # Fallback to simple estimate
            electron_densities[i] = 0.1 * n_total

    # Height structure
    if spherical:
        # For spherical atmospheres, use radial coordinate
        # Photosphere at tau=1
        R_phot = np.sqrt(mass_g * g)  # Rough estimate

        # Heights relative to photosphere (increasing outward)
        z_values = np.cumsum(H * np.abs(np.diff(np.concatenate([[0], log_tau_ref_values]))))
        z_values = z_values - z_values[tau_ref_unity_idx]  # Center at photosphere

        layers = [
            ShellAtmosphereLayer(
                tau_ref=tau_ref_values[i],
                z=z_values[i],
                temperature=temperatures[i],
                electron_number_density=electron_densities[i],
                number_density=number_densities[i]
            )
            for i in range(n_layers)
        ]

        return ShellAtmosphere(
            layers=layers,
            R_photosphere=R_phot,
            reference_wavelength=reference_wavelength
        )

    else:
        # Planar atmosphere: use heights
        # Height increases outward from photosphere
        z_values = np.cumsum(H * np.abs(np.diff(np.concatenate([[0], log_tau_ref_values]))))
        z_values = z_values - z_values[tau_ref_unity_idx]  # Center at photosphere

        layers = [
            PlanarAtmosphereLayer(
                tau_ref=tau_ref_values[i],
                z=z_values[i],
                temperature=temperatures[i],
                electron_number_density=electron_densities[i],
                number_density=number_densities[i]
            )
            for i in range(n_layers)
        ]

        return PlanarAtmosphere(
            layers=layers,
            reference_wavelength=reference_wavelength
        )


def create_solar_test_atmosphere():
    """
    Create a simple solar-type test atmosphere for validation.

    Returns a plane-parallel atmosphere with solar parameters:
    - Teff ~ 5777 K
    - logg = 4.44
    - 56 layers from log(tau) = -4 to +2

    Returns
    -------
    atmosphere : PlanarAtmosphere
        Solar test atmosphere

    Examples
    --------
    >>> atm = create_solar_test_atmosphere()
    >>> atm.T[atm.tau_ref.argmin()]  # Surface temperature
    ~5777.0
    """
    # Temperature stratification roughly solar
    # Surface (low tau) is cooler, deep layers are hotter
    log_tau = np.linspace(-4, 2, 56)

    # Approximate T(tau) relation for solar atmosphere
    # T increases roughly as tau^0.25 in photosphere
    T_eff = 5777  # K
    T = T_eff * (0.5 + 0.75 * (10**log_tau / (1 + 10**log_tau))**0.25)

    return create_simple_atmosphere(
        temperatures=T,
        log_tau_ref_values=log_tau,
        logg=4.44,  # Solar
        spherical=False
    )
