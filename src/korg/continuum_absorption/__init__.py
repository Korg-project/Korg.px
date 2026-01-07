"""
Continuum absorption module for Korg.

Computes continuum opacity from various sources including:
- H I bound-free and free-free
- H⁻ bound-free and free-free
- H₂⁺ bound-free and free-free
- He I bound-free and free-free
- He⁻ free-free
- Positive ion free-free
- Metal bound-free
- Rayleigh scattering (H, He, H₂)
- Electron (Thomson) scattering
"""

from .scattering import rayleigh, electron_scattering
from .absorption_He import ndens_state_He_I, Heminus_ff
from .absorption_metals_bf import metal_bf_absorption, get_available_species

__all__ = [
    'rayleigh',
    'electron_scattering',
    'ndens_state_He_I',
    'Heminus_ff',
    'metal_bf_absorption',
    'get_available_species',
]
