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

# Import artifact system (always available)
from .artifacts import (
    download_artifact,
    get_artifact_path,
    list_artifacts,
    create_placeholder_artifact,
    get_korg_data_dir
)

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

# Linelist functions
from .linelist import (
    Line, approximate_line_strength, create_line,
    read_linelist, read_vald_linelist, read_korg_linelist,
    save_linelist, parse_moog_linelist, parse_turbospectrum_linelist,
    get_VALD_solar_linelist, get_GALAH_DR3_linelist,
    get_APOGEE_DR17_linelist, get_GES_linelist,
    air_to_vacuum, vacuum_to_air, isotopic_abundances,
)

# Prune/merge utilities
from .prune_linelist import merge_close_lines, prune_linelist

# Molecular cross-sections
from .molecular_cross_sections import (
    MolecularCrossSection,
    interpolate_molecular_cross_sections,
    save_molecular_cross_section,
    read_molecular_cross_section,
)

# Atmosphere I/O
from .atmosphere import read_model_atmosphere

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
    # Artifact system
    "download_artifact",
    "get_artifact_path",
    "list_artifacts",
    "create_placeholder_artifact",
    "get_korg_data_dir",
    # Key classes
    "Species",
    "Formula",
    "Line",
    "MolecularCrossSection",
    # Linelist functions
    "approximate_line_strength",
    "create_line",
    "read_linelist",
    "read_vald_linelist",
    "read_korg_linelist",
    "save_linelist",
    "parse_moog_linelist",
    "parse_turbospectrum_linelist",
    "get_VALD_solar_linelist",
    "get_GALAH_DR3_linelist",
    "get_APOGEE_DR17_linelist",
    "get_GES_linelist",
    "air_to_vacuum",
    "vacuum_to_air",
    "isotopic_abundances",
    # Prune/merge
    "merge_close_lines",
    "prune_linelist",
    # Molecular cross-sections
    "interpolate_molecular_cross_sections",
    "save_molecular_cross_section",
    "read_molecular_cross_section",
    # Atmosphere I/O
    "read_model_atmosphere",
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
