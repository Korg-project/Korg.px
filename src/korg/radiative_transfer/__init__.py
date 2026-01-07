"""
Radiative transfer solver for spectral synthesis.

This module implements the formal solution of the radiative transfer equation
for stellar atmospheres. It computes emergent flux from opacity and source
function profiles through the atmosphere.

Main functions:
- radiative_transfer: Main entry point for computing flux
- exponential_integral_2: E2(x) approximation for optimized flux calculation
"""

from .expint import exponential_integral_2, exponential_integral_3
from .optical_depth import compute_tau_anchored
from .intensity import compute_I_linear_flux_only, compute_F_flux_only_expint
from .core import (radiative_transfer, radiative_transfer_single_wavelength,
                   radiative_transfer_jit, radiative_transfer_single_wavelength_jit)

__all__ = [
    'radiative_transfer',
    'radiative_transfer_single_wavelength',
    'radiative_transfer_jit',
    'radiative_transfer_single_wavelength_jit',
    'exponential_integral_2',
    'exponential_integral_3',
    'compute_tau_anchored',
    'compute_I_linear_flux_only',
    'compute_F_flux_only_expint',
]
