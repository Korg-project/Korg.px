"""
MARCS model atmosphere interpolation.

This module provides functions to interpolate MARCS model atmospheres
for spectral synthesis, validated against Korg.jl.

Reference: Korg.jl src/atmosphere.jl and src/lazy_multilinear_interpolation.jl
"""

import numpy as np
import jax.numpy as jnp
import h5py
import os
import warnings
from typing import Tuple, List, Optional
from pathlib import Path

from .atmosphere import PlanarAtmosphere, PlanarAtmosphereLayer
from .atmosphere import ShellAtmosphere, ShellAtmosphereLayer
from .constants import G_cgs
from .artifacts import get_artifact_path, is_placeholder_file, ARTIFACTS


class AtmosphereInterpolationError(Exception):
    """Raised when atmosphere interpolation fails."""
    pass


def lazy_multilinear_interpolation(
    params: jnp.ndarray,
    nodes: List[jnp.ndarray],
    grid: jnp.ndarray,
    param_names: Optional[List[str]] = None,
    perturb_at_grid_values: bool = False
) -> jnp.ndarray:
    """
    Multidimensional linear interpolation on a grid.

    This function performs multilinear interpolation where the first two dimensions
    of the grid represent the values being interpolated (layers x quantities).
    It is optimized to minimize memory-mapped file reads.

    Parameters
    ----------
    params : jnp.ndarray
        Parameters to interpolate at, shape (n_params,)
    nodes : List[jnp.ndarray]
        Grid node values for each parameter
    grid : jnp.ndarray
        Grid of atmosphere quantities, shape (n_layers, n_quant, *grid_dims)
    param_names : List[str], optional
        Names of parameters for error messages
    perturb_at_grid_values : bool, optional
        If True, slightly perturb params that exactly match grid values

    Returns
    -------
    jnp.ndarray
        Interpolated atmosphere quantities, shape (n_layers, n_quant)

    Reference
    ---------
    Korg.jl src/lazy_multilinear_interpolation.jl
    """
    params = jnp.array(params, dtype=jnp.float64)
    n_params = len(params)

    if param_names is None:
        param_names = [f"param {i}" for i in range(n_params)]

    # Perturb parameters that are exactly on grid points
    if perturb_at_grid_values:
        for i in range(n_params):
            if params[i] in nodes[i]:
                # Use nextafter to get next representable float
                params = params.at[i].set(jnp.nextafter(params[i], jnp.inf))

                # Handle case where param is at the last grid value
                if params[i] > nodes[i][-1]:
                    params = params.at[i].set(jnp.nextafter(params[i], -jnp.inf))
                    params = params.at[i].set(jnp.nextafter(params[i], -jnp.inf))

    # Find upper vertex indices for each parameter
    upper_vertex = []
    for i, (p, p_name, p_nodes) in enumerate(zip(params, param_names, nodes)):
        if not (p_nodes[0] <= p <= p_nodes[-1]):
            raise AtmosphereInterpolationError(
                f"Can't interpolate grid. {p_name} is out of bounds. "
                f"({p} ∉ [{p_nodes[0]}, {p_nodes[-1]}])"
            )
        # Find first index where p <= p_nodes[idx]
        upper_idx = jnp.searchsorted(p_nodes, p, side='right')
        if upper_idx == 0:
            upper_idx = 1  # Ensure we have a valid lower bound
        upper_vertex.append(int(upper_idx))

    # Check which params are exactly on grid points (check lower bound)
    isexact = jnp.array([params[i] == nodes[i][upper_vertex[i] - 1]
                         for i in range(n_params)])

    # Allocate 2^n hypercube for interpolation
    dims = tuple(2 for _ in range(n_params))
    structure = jnp.zeros((grid.shape[0], grid.shape[1]) + dims)

    # Fill hypercube with bounding atmospheres
    for idx in np.ndindex(dims):
        local_inds = list(idx)
        atm_inds = list(local_inds)

        # Use upper bound as lower bound if param is on grid point
        for i in range(n_params):
            if isexact[i]:
                atm_inds[i] = 1  # Use upper bound

        # Convert to grid indices
        # local_ind 0 (lower) → grid index upper_vertex - 1
        # local_ind 1 (upper) → grid index upper_vertex
        for i in range(n_params):
            atm_inds[i] += upper_vertex[i] - 1

        # Extract atmosphere from grid
        grid_slice = (slice(None), slice(None)) + tuple(atm_inds)
        structure_slice = (slice(None), slice(None)) + idx
        structure = structure.at[structure_slice].set(grid[grid_slice])

    # Perform multilinear interpolation
    for i in range(n_params):
        if isexact[i]:
            continue  # No interpolation needed

        # Bounding values of parameter i
        p1 = nodes[i][upper_vertex[i] - 1]
        p2 = nodes[i][upper_vertex[i]]

        # Linear interpolation weight
        x = (params[i] - p1) / (p2 - p1)

        # Indices for slices through uninterpolated dimensions
        # Julia uses 1-indexing, so index 1 = lower bound
        # Python uses 0-indexing, so index 0 = lower bound
        inds1 = tuple([slice(None), slice(None)] +
                     [0 if j < i else (0 if j == i else slice(None))
                      for j in range(n_params)])
        inds2 = tuple([slice(None), slice(None)] +
                     [0 if j < i else (1 if j == i else slice(None))
                      for j in range(n_params)])

        # Linear interpolation
        structure = structure.at[inds1].set(
            (1 - x) * structure[inds1] + x * structure[inds2]
        )

    # Extract final interpolated result
    # Julia uses all 1's (1-indexed), Python uses all 0's (0-indexed)
    final_idx = tuple([slice(None), slice(None)] + [0] * n_params)
    return structure[final_idx]


def get_marcs_grid_path(
    grid_type: str = 'standard',
    auto_download: bool = True
) -> Optional[Path]:
    """
    Get path to MARCS atmosphere grid, downloading if necessary.

    Parameters
    ----------
    grid_type : str, optional
        Type of grid: 'standard', 'low_Z', or 'cool_dwarfs' (default: 'standard')
    auto_download : bool, optional
        Automatically download if not present (default: True)

    Returns
    -------
    Path or None
        Path to HDF5 file, or None if not available

    Raises
    ------
    ValueError
        If grid_type is invalid
    FileNotFoundError
        If grid not found and auto_download is False
    """
    artifact_map = {
        'standard': ('SDSS_MARCS_atmospheres_v2', 'SDSS_MARCS_atmospheres.h5'),
        'low_Z': ('MARCS_metal_poor_atmospheres', 'MARCS_metal_poor_atmospheres.h5'),
        'cool_dwarfs': ('resampled_cool_dwarf_atmospheres', 'resampled_cool_dwarf_atmospheres.h5')
    }

    if grid_type not in artifact_map:
        raise ValueError(f"Invalid grid_type: {grid_type}. "
                        f"Must be one of {list(artifact_map.keys())}")

    artifact_name, h5_filename = artifact_map[grid_type]

    # Get artifact path (downloads if needed)
    artifact_dir = get_artifact_path(artifact_name, auto_download=auto_download)

    if artifact_dir is None:
        raise FileNotFoundError(
            f"MARCS {grid_type} grid not found. "
            "Set auto_download=True or run download_artifact manually."
        )

    # Construct full path to HDF5 file
    extract_dir = artifact_dir / ARTIFACTS[artifact_name]['extract_dir']
    h5_path = extract_dir / h5_filename

    # Check if it's a placeholder
    if is_placeholder_file(h5_path):
        if os.environ.get('CI') or os.environ.get('GITHUB_ACTIONS'):
            warnings.warn(
                f"Using placeholder for {artifact_name} in CI. "
                "MARCS interpolation will not work.",
                UserWarning
            )
            return None
        else:
            raise FileNotFoundError(
                f"MARCS grid file {h5_path} is a placeholder. "
                "Please download the full artifact."
            )

    if not h5_path.exists():
        raise FileNotFoundError(
            f"MARCS grid file not found at {h5_path}. "
            "The artifact may be corrupted."
        )

    return h5_path


def load_marcs_grid(
    path: Optional[str] = None,
    grid_type: str = 'standard'
) -> Tuple[List[jnp.ndarray], jnp.ndarray]:
    """
    Load MARCS atmosphere grid from HDF5 file.

    If path is not provided, automatically downloads the grid from AWS S3
    if it's not already cached locally.

    Parameters
    ----------
    path : str, optional
        Path to HDF5 file. If None, downloads/uses cached grid.
    grid_type : str, optional
        Type of grid: 'standard', 'low_Z', or 'cool_dwarfs' (default: 'standard')
        Only used if path is None.

    Returns
    -------
    nodes : List[jnp.ndarray]
        Grid node values for [Teff, logg, [M/H], [α/M], [C/metals]]
    grid : jnp.ndarray
        Atmosphere grid, shape (n_layers, 5, n_Teff, n_logg, n_MH, n_alpha, n_C)
        Quantities: [T, log_ne, log_n, tau_ref, asinh_z]

    Raises
    ------
    FileNotFoundError
        If grid cannot be found or downloaded
    RuntimeError
        If download or file reading fails
    """
    if path is None:
        h5_path = get_marcs_grid_path(grid_type=grid_type, auto_download=True)
        if h5_path is None:
            # Placeholder in CI - return dummy data
            warnings.warn(
                "Returning dummy MARCS grid data (CI placeholder mode)",
                UserWarning
            )
            # Return minimal valid structure for import testing
            nodes = [jnp.array([5000.0]), jnp.array([4.0]), jnp.array([0.0]),
                    jnp.array([0.0]), jnp.array([0.0])]
            grid = jnp.zeros((1, 5, 1, 1, 1, 1, 1))
            return nodes, grid
        path = str(h5_path)

    try:
        with h5py.File(path, 'r') as f:
            # Load grid node values
            # Parameter order: Teff, logg, M_H, alpha, C
            nodes = []
            for i in range(1, 6):  # 5 parameters
                nodes.append(jnp.array(f[f'grid_values/{i}'][:]))

            # Load atmosphere grid
            # HDF5 storage: (C, alpha, M_H, logg, Teff, quantities, layers)
            # We need: (layers, quantities, Teff, logg, M_H, alpha, C)
            # Permutation: (6, 5, 4, 3, 2, 1, 0)
            grid_raw = jnp.array(f['grid'][:])
            grid = jnp.transpose(grid_raw, (6, 5, 4, 3, 2, 1, 0))

        return nodes, grid

    except Exception as e:
        raise RuntimeError(f"Failed to load MARCS grid from {path}: {e}") from e


def interpolate_marcs(
    Teff: float,
    logg: float,
    M_H: float = 0.0,
    alpha_M: float = 0.0,
    C_M: float = 0.0,
    spherical: Optional[bool] = None,
    perturb_at_grid_values: bool = True
) -> PlanarAtmosphere:
    """
    Interpolate a MARCS model atmosphere.

    Returns a model atmosphere computed by interpolating models from MARCS
    (Gustafsson+ 2008) using multilinear interpolation. The MARCS atmosphere
    grid is automatically downloaded from AWS S3 on first use and cached in
    ~/.korg/ (or $KORG_DATA_DIR if set).

    Parameters
    ----------
    Teff : float
        Effective temperature [K]
    logg : float
        Surface gravity log10(g [cm/s²])
    M_H : float, optional
        Metallicity [M/H] (default: 0.0 = solar)
    alpha_M : float, optional
        Alpha enhancement [α/M] (default: 0.0)
    C_M : float, optional
        Carbon enhancement [C/metals] (default: 0.0)
    spherical : bool, optional
        If True, return ShellAtmosphere; else PlanarAtmosphere.
        Default: True if logg < 3.5, else False
    perturb_at_grid_values : bool, optional
        Slightly perturb parameters on grid points (default: True)

    Returns
    -------
    atmosphere : PlanarAtmosphere or ShellAtmosphere
        Interpolated model atmosphere

    Notes
    -----
    The MARCS grid covers:
    - Teff: 2800-8000 K
    - logg: -0.5 to 5.5
    - [M/H]: -2.5 to 1.0
    - [α/M]: -1.0 to 1.0
    - [C/metals]: -1.5 to 1.0

    Reference wavelength is 5000 Å (5e-5 cm) for MARCS models.

    On first call, this function will download the MARCS atmosphere grid
    (~380 MB) from AWS S3. The download is cached in ~/.korg/ so subsequent
    calls are fast. Set the KORG_DATA_DIR environment variable to use a
    different cache directory.

    Examples
    --------
    >>> # Solar-type star (downloads grid on first use)
    >>> atm = interpolate_marcs(5777, 4.44, 0.0, 0.0, 0.0)
    >>> atm.n_layers
    56

    Reference
    ---------
    Korg.jl src/atmosphere.jl interpolate_marcs()
    """
    if spherical is None:
        spherical = (logg < 3.5)

    # Reference wavelength for MARCS models
    reference_wavelength = 5e-5  # 5000 Å in cm

    # Load MARCS grid
    nodes, grid = load_marcs_grid()

    # Parameters for interpolation
    params = jnp.array([Teff, logg, M_H, alpha_M, C_M])
    param_names = ["Teff", "log(g)", "[M/H]", "[α/M]", "[C/metals]"]

    # Perform multilinear interpolation
    atm_quants = lazy_multilinear_interpolation(
        params, nodes, grid,
        param_names=param_names,
        perturb_at_grid_values=perturb_at_grid_values
    )

    # Filter out NaN layers (MARCS uses NaN to mark invalid layers)
    nanmask = ~jnp.isnan(atm_quants[:, 3])  # Check tau_ref column

    # Extract and transform quantities
    # Grid stores: [T, log_ne, log_n, tau_ref, asinh_z]
    T = atm_quants[nanmask, 0]
    log_ne = atm_quants[nanmask, 1]
    log_n = atm_quants[nanmask, 2]
    tau_ref = atm_quants[nanmask, 3]
    asinh_z = atm_quants[nanmask, 4]

    # Transform back to physical values
    ne = jnp.exp(log_ne)
    n = jnp.exp(log_n)
    z = jnp.sinh(asinh_z)

    # Create atmosphere structure
    n_layers = int(jnp.sum(nanmask))

    if spherical:
        # Calculate photospheric radius
        # R = sqrt(G * M_sun / g)
        solar_mass_cgs = 1.9885e33  # grams
        R_phot = jnp.sqrt(G_cgs * solar_mass_cgs / (10**logg))

        layers = [
            ShellAtmosphereLayer(
                tau_ref=float(tau_ref[i]),
                z=float(z[i]),
                temperature=float(T[i]),
                electron_number_density=float(ne[i]),
                number_density=float(n[i])
            )
            for i in range(n_layers)
        ]

        atm = ShellAtmosphere(
            layers=layers,
            R_photosphere=float(R_phot),
            reference_wavelength=reference_wavelength
        )
    else:
        layers = [
            PlanarAtmosphereLayer(
                tau_ref=float(tau_ref[i]),
                z=float(z[i]),
                temperature=float(T[i]),
                electron_number_density=float(ne[i]),
                number_density=float(n[i])
            )
            for i in range(n_layers)
        ]

        atm = PlanarAtmosphere(
            layers=layers,
            reference_wavelength=reference_wavelength
        )

    # Check for negative optical depths (indicates unreliable interpolation)
    if jnp.any(tau_ref < 0):
        raise AtmosphereInterpolationError(
            "Interpolated atmosphere has negative optical depths and is not reliable."
        )

    return atm
