"""
Korg: A Python/JAX implementation of 1D LTE spectral synthesis.

This is a port of the Julia package Korg.jl to Python using JAX for
automatic differentiation and GPU acceleration.
"""

# Enable 64-bit precision in JAX for accurate spectral synthesis.
# This MUST be done before any other JAX imports or operations.
# Note: If jax has already been imported elsewhere, you may need to set
# the environment variable JAX_ENABLE_X64=true before running Python.
import os
os.environ.setdefault("JAX_ENABLE_X64", "true")
import jax
jax.config.update("jax_enable_x64", True)

__version__ = "0.1.0"

# Import submodules
from . import constants
from . import atomic_data

# Import key functions for convenience
from .abundances import format_A_X, get_solar_abundances
from .species import Species, Formula

# These imports may fail if data files are not available
try:
    from .synthesis import synthesize, load_synthesis_data, save_synthesis_data
    from .marcs_interpolation import interpolate_marcs
except (ImportError, FileNotFoundError) as e:
    import warnings
    warnings.warn(f"Could not import synthesis functions: {e}")
    synthesize = None
    load_synthesis_data = None
    save_synthesis_data = None
    interpolate_marcs = None

# Re-export commonly used constants
from .constants import (
    c_cgs, hplanck_cgs, hplanck_eV, kboltz_cgs, kboltz_eV,
    electron_mass_cgs, electron_charge_cgs, amu_cgs, Rydberg_eV,
    MAX_ATOMIC_NUMBER
)

__all__ = [
    # Version
    "__version__",
    # Submodules
    "constants",
    "atomic_data",
    # Key functions
    "format_A_X",
    "get_solar_abundances",
    "synthesize",
    "load_synthesis_data",
    "save_synthesis_data",
    "interpolate_marcs",
    # Key classes
    "Species",
    "Formula",
    # Constants
    "c_cgs",
    "hplanck_cgs",
    "hplanck_eV",
    "kboltz_cgs",
    "kboltz_eV",
    "electron_mass_cgs",
    "electron_charge_cgs",
    "amu_cgs",
    "Rydberg_eV",
    "MAX_ATOMIC_NUMBER",
]
