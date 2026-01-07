"""
Stark profile data loading for hydrogen lines.

This module loads pre-computed Stark-broadened hydrogen line profiles
from Stehlé & Hutcheon (1999).
"""

import numpy as np
import jax.numpy as jnp
import h5py
from scipy.interpolate import RegularGridInterpolator
from typing import Dict, Tuple
import os


class StarkProfileLine:
    """
    Container for a single hydrogen line's Stark profile data.

    Attributes:
        temps: Temperature grid [K]
        electron_number_densities: Electron density grid [cm^-3]
        log_delta_nu_grid: Log of delta_nu/F0 grid
        profile_data: 3D array of log(profile) values with shape (temps, nes, delta_nu)
        lambda0_data: 2D array of line center [cm] with shape (temps, nes)
        lower: Lower level quantum number
        upper: Upper level quantum number
        Kalpha: Kalpha parameter
        log_gf: Log of gf value

        # Legacy scipy interpolators (for backward compatibility)
        profile: 3D interpolator for log(profile) over (T, ne, log(delta_nu/F0))
        lambda0: 2D interpolator for line center [cm] over (T, ne)
    """

    def __init__(self, temps, nes, profile_interp, lambda0_interp,
                 lower, upper, Kalpha, log_gf,
                 log_delta_nu_grid=None, profile_data=None, lambda0_data_array=None):
        self.temps = jnp.array(temps)
        self.electron_number_densities = jnp.array(nes)
        self.lower = lower
        self.upper = upper
        self.Kalpha = Kalpha
        self.log_gf = log_gf

        # Legacy scipy interpolators
        self.profile = profile_interp
        self.lambda0 = lambda0_interp

        # JAX-compatible data arrays
        self.log_delta_nu_grid = jnp.array(log_delta_nu_grid) if log_delta_nu_grid is not None else None
        self.profile_data = jnp.array(profile_data) if profile_data is not None else None
        self.lambda0_data = jnp.array(lambda0_data_array) if lambda0_data_array is not None else None

    def interpolate_lambda0_jax(self, T: float, ne: float) -> float:
        """
        JAX-compatible 2D interpolation for line center wavelength.

        Args:
            T: Temperature [K]
            ne: Electron number density [cm^-3]

        Returns:
            Line center wavelength [cm]
        """
        from .hydrogen_line_absorption import _interp_linear_2d_jax
        return _interp_linear_2d_jax(T, ne, self.temps, self.electron_number_densities,
                                      self.lambda0_data)

    def interpolate_profile_jax(self, T: float, ne: float, log_delta_nu: float) -> float:
        """
        JAX-compatible 3D interpolation for log(profile).

        Args:
            T: Temperature [K]
            ne: Electron number density [cm^-3]
            log_delta_nu: Log of scaled frequency detuning

        Returns:
            Log of profile value
        """
        from .hydrogen_line_absorption import _interp_linear_3d_jax
        return _interp_linear_3d_jax(T, ne, log_delta_nu,
                                      self.temps, self.electron_number_densities,
                                      self.log_delta_nu_grid, self.profile_data)


def _load_stark_profiles(fname: str) -> Dict[str, StarkProfileLine]:
    """
    Load Stark broadening profiles from HDF5 file.

    This follows the Julia implementation in src/hydrogen_line_absorption.jl.

    Args:
        fname: Path to HDF5 file containing Stark profiles

    Returns:
        Dictionary mapping transition names to StarkProfileLine objects
    """
    profiles = {}

    with h5py.File(fname, 'r') as fid:
        for transition in fid.keys():
            grp = fid[transition]

            # Read datasets
            temps = grp['temps'][:]
            nes = grp['electron_number_densities'][:]
            delta_nu_over_F0 = grp['delta_nu_over_F0'][:]
            P = grp['profile'][:]  # Shape: (delta_nu, ne, temps)
            lambda0_data = grp['lambda0'][:]  # Shape: (ne, temps)

            # Read attributes
            lower = int(grp.attrs['lower'])
            upper = int(grp.attrs['upper'])
            Kalpha = float(grp.attrs['Kalpha'])
            log_gf = float(grp.attrs['log_gf'])

            # Create log profile, handling -Inf values
            with np.errstate(divide='ignore', invalid='ignore'):
                logP = np.log(P)
            # Clipping to -700 (slightly larger than log(floatmin)) to avoid NaNs
            logP = np.where(np.isfinite(logP), logP, -700.0)

            # Prepare grid for interpolation
            # Julia uses: (temps, nes, [-floatmax; log.(delta_nu_over_F0[2:end])])
            # For the first delta_nu_over_F0 (which is 0), use -floatmax equivalent
            log_delta_nu = np.log(delta_nu_over_F0[1:])  # Skip first element (0)
            log_delta_nu_grid = np.concatenate([[-1e308], log_delta_nu])

            # Transpose logP to match interpolator convention: (temps, nes, delta_nu)
            # HDF5 has shape (delta_nu, ne, temps), we need (temps, nes, delta_nu)
            logP_transposed = np.transpose(logP, (2, 1, 0))

            # Create 3D interpolator for profile
            # Uses flat extrapolation (values outside bounds use nearest boundary value)
            profile_interp = RegularGridInterpolator(
                (temps, nes, log_delta_nu_grid),
                logP_transposed,
                method='linear',
                bounds_error=False,
                fill_value=None  # Uses nearest neighbor extrapolation
            )

            # Transpose lambda0 to match interpolator convention: (temps, nes)
            # HDF5 has shape (ne, temps), we need (temps, nes)
            lambda0_transposed = lambda0_data.T * 1e-8  # Convert Å to cm

            # Create 2D interpolator for lambda0
            lambda0_interp = RegularGridInterpolator(
                (temps, nes),
                lambda0_transposed,
                method='linear',
                bounds_error=False,
                fill_value=None
            )

            profiles[transition] = StarkProfileLine(
                temps, nes, profile_interp, lambda0_interp,
                lower, upper, Kalpha, log_gf,
                log_delta_nu_grid=log_delta_nu_grid,
                profile_data=logP_transposed,
                lambda0_data_array=lambda0_transposed
            )

    return profiles


# Load profiles at module import
_data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
_stark_profile_path = os.path.join(_data_dir, 'Stehle-Hutchson-hydrogen-profiles.h5')

if os.path.exists(_stark_profile_path):
    hline_stark_profiles = _load_stark_profiles(_stark_profile_path)
else:
    # If data file doesn't exist, create empty dict
    # (useful for testing or when data hasn't been downloaded)
    hline_stark_profiles = {}
