"""
MARCS model atmosphere interpolation.

This module provides functions to interpolate MARCS model atmospheres
for spectral synthesis, validated against Korg.jl.

Reference: Korg.jl src/atmosphere.jl and src/lazy_multilinear_interpolation.jl
"""

import functools

import numpy as np
import jax
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

        # When param is exactly on a grid point, both corners use the same node
        for i in range(n_params):
            if isexact[i]:
                atm_inds[i] = 0  # Both corners map to the exact grid node (upper_vertex-1)

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


# ---------------------------------------------------------------------------
# JIT-compatible MARCS interpolation kernel
# ---------------------------------------------------------------------------

# Module-level cache for JIT data (nodes_padded, nodes_lengths, grid)
_marcs_jit_cache: dict = {}


def _get_marcs_jit_data() -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Return (nodes_padded, nodes_lengths, grid), computing and caching on first call.

    nodes_padded : shape (5, max_node_len) float64
        Grid node values for each parameter, padded with +inf so that
        searchsorted ignores the padding region.
    nodes_lengths : shape (5,) int32 — actual node count per dimension
    grid : shape (n_layers, 5, n_Teff, n_logg, n_MH, n_alpha, n_C) float64

    Notes
    -----
    This is a convenience function used by external callers (e.g. tests) that
    want direct access to the JIT-ready arrays.  ``interpolate_marcs`` manages
    the cache itself so it can re-detect when ``load_marcs_grid()`` returns a
    different grid object (e.g. after the environment changes in tests).
    """
    if "data" not in _marcs_jit_cache:
        nodes, grid = load_marcs_grid()
        lengths = [len(n) for n in nodes]
        max_len = max(lengths)
        # Pad with +inf so searchsorted naturally stops at the valid region
        nodes_padded = jnp.array(
            np.array([
                np.concatenate([np.asarray(n), np.full(max_len - len(n), np.inf)])
                for n in nodes
            ]),
            dtype=jnp.float64
        )
        nodes_lengths = jnp.array(lengths, dtype=jnp.int32)
        _marcs_jit_cache["data"] = (nodes_padded, nodes_lengths, grid)
        _marcs_jit_cache["grid_id"] = id(grid)
    return _marcs_jit_cache["data"]


@functools.partial(jax.jit, static_argnums=())
def _interpolate_marcs_jit(
    params: jnp.ndarray,
    nodes_padded: jnp.ndarray,
    nodes_lengths: jnp.ndarray,
    grid: jnp.ndarray,
) -> jnp.ndarray:
    """
    JIT-compiled multilinear interpolation kernel for the MARCS atmosphere grid.

    This is an inner function called by ``interpolate_marcs``.  It contains no
    Python-level conditionals on traced values and is fully differentiable.

    Parameters
    ----------
    params : jnp.ndarray, shape (5,)
        [Teff, logg, M_H, alpha_M, C_M] — values to interpolate at.
    nodes_padded : jnp.ndarray, shape (5, max_node_len)
        Grid node values for each parameter, padded with +inf to uniform length.
    nodes_lengths : jnp.ndarray, shape (5,) int32
        Actual number of nodes in each dimension.
    grid : jnp.ndarray, shape (n_layers, 5, n_Teff, n_logg, n_MH, n_alpha, n_C)
        Full MARCS atmosphere grid (5 quantities: T, log_ne, log_n, tau_ref, asinh_z).

    Returns
    -------
    jnp.ndarray, shape (n_layers, 5)
        Interpolated atmosphere quantities (NaN rows where the grid is masked).

    Notes
    -----
    The 5-dimensional multilinear interpolation is unrolled over all 2^5 = 32
    corners at Python trace time, so the compiled XLA graph is a fixed-size
    weighted sum — no dynamic loops or conditionals at runtime.
    """
    N_PARAMS = 5  # static: number of interpolation dimensions

    # --- 1. Perturb params that sit exactly on a grid node ---
    # We nudge by a tiny relative amount so that searchsorted(side='right')
    # places the value strictly inside a bracket rather than on a node.
    # Using a relative epsilon (1e-10 * scale) keeps this differentiable:
    # jnp.where is differentiable everywhere except where its condition changes,
    # but on_node being True only at exact grid nodes, and the nudge being a
    # linear function of params[i], makes the Jacobian well-defined almost
    # everywhere in practice.
    params = params.astype(jnp.float64)
    _NUDGE = jnp.finfo(jnp.float64).eps * 4.0  # ~8.9e-16, very small
    for i in range(N_PARAMS):
        node_row = nodes_padded[i]  # shape (max_node_len,)
        n_nodes = nodes_lengths[i]
        # Check if params[i] exactly matches any valid node
        # (nodes_padded uses +inf for padding so invalid entries are ignored)
        diffs = jnp.abs(node_row - params[i])
        on_node = jnp.min(diffs) < 1e-10
        # Nudge: use a small absolute offset based on the interval spacing
        last_node = node_row[n_nodes - 1]
        first_node = node_row[0]
        scale = last_node - first_node  # total grid span for this dimension
        nudge_abs = jnp.where(scale > 0.0, _NUDGE * scale, _NUDGE)
        nudged = params[i] + nudge_abs
        # If nudge pushed past the last node, nudge downward instead
        nudged = jnp.where(nudged > last_node, params[i] - nudge_abs, nudged)
        params = params.at[i].set(jnp.where(on_node, nudged, params[i]))

    # --- 2. Find bracket indices via searchsorted (all JIT-safe) ---
    # upper[i] is the index of the first node > params[i], clamped to [1, n_nodes-1].
    upper = jnp.stack([
        jnp.searchsorted(nodes_padded[i], params[i], side='right')
          .clip(1, nodes_lengths[i] - 1)
        for i in range(N_PARAMS)
    ])  # shape (5,)

    # --- 3. Compute interpolation weights ---
    # weight[i] = (params[i] - lower_node) / (upper_node - lower_node)
    lower_vals = jnp.stack([nodes_padded[i, upper[i] - 1] for i in range(N_PARAMS)])
    upper_vals = jnp.stack([nodes_padded[i, upper[i]]     for i in range(N_PARAMS)])
    weights = (params - lower_vals) / (upper_vals - lower_vals)  # shape (5,)

    # --- 4. Accumulate over all 2^5 = 32 corners (static Python loop) ---
    # grid shape: (n_layers, 5, d0, d1, d2, d3, d4)
    n_layers = grid.shape[0]
    n_quant = grid.shape[1]
    # Cut the 2x2x2x2x2 bracket out of the grid once, then read the corners from
    # that. The obvious form -- walking the full grid down to a scalar index
    # inside the corner loop -- reads the 619 MB grid constant 32 separate times,
    # and XLA sizes the compilation accordingly: 32 x 619 MB is ~20 GB, and a
    # forward synthesis through here measured a 25 GB compile against 740 MB for
    # the entire rest of the opacity stack (reverse mode doubled it to 46 GB).
    # The bracket is 56 x 5 x 32 float32 = 36 KB.
    #
    # ``upper[i]`` is clipped to [1, n_nodes-1], so the slice start is in range
    # and start + 2 never runs off the end. The corner values and the weights
    # they are combined with are exactly those the per-corner gathers produced.
    block = grid  # (n_layers, n_quant, d0, d1, d2, d3, d4)
    for i in range(N_PARAMS):
        # dynamic_slice_in_dim keeps the axis (size 2), so the next parameter's
        # axis is 2 + i rather than a fixed 2.
        block = jax.lax.dynamic_slice_in_dim(block, upper[i] - 1, 2, axis=2 + i)

    result = jnp.zeros((n_layers, n_quant), dtype=jnp.float64)

    for corner in range(1 << N_PARAMS):
        # bits[i] ∈ {0, 1}: whether to use the upper node in dimension i
        bit_list = [(corner >> i) & 1 for i in range(N_PARAMS)]
        bits = jnp.array(bit_list, dtype=jnp.int32)

        # Corner weight: product of (w if upper, (1-w) if lower)
        w_corner = jnp.prod(jnp.where(bits, weights, 1.0 - weights))

        # Static indices into the 2-wide bracket -- no traced index, no gather.
        val = block[(slice(None), slice(None)) + tuple(bit_list)]

        result = result + w_corner * val

    return result


def interpolate_marcs(
    Teff: float,
    logg: float,
    M_H_or_A_X=0.0,
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
    M_H_or_A_X : float or array, optional
        Metallicity [M/H] (default: 0.0 = solar), or a 92-element A(X) abundance
        vector from which M_H, alpha_M, and C_M are derived.
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
    # Python-level decision — not traced by JAX
    if spherical is None:
        spherical = float(logg) < 3.5

    # Accept either M_H (scalar) or A_X (92-element abundance vector) as third arg.
    # This matches Julia's two-method dispatch for interpolate_marcs.
    import numpy as _np
    M_H_or_A_X_arr = _np.asarray(M_H_or_A_X)
    if M_H_or_A_X_arr.ndim == 0:
        # Scalar: treat as M_H directly
        M_H = float(M_H_or_A_X)
    else:
        # Array: assume it's A_X; derive M_H, alpha_M, C_M from it
        from .abundances import (
            get_metals_H, get_alpha_H, GREVESSE_2007_SOLAR_ABUNDANCES, DEFAULT_ALPHA_ELEMENTS
        )
        A_X = _np.asarray(M_H_or_A_X, dtype=float)
        solar = GREVESSE_2007_SOLAR_ABUNDANCES
        # Julia excludes C (Z=6) from metals calculation, same as alpha elements
        alpha_and_C = list(DEFAULT_ALPHA_ELEMENTS) + [6]
        M_H = get_metals_H(A_X, solar_abundances=solar, ignore_alpha=True,
                           alpha_elements=alpha_and_C)
        alpha_H = get_alpha_H(A_X, solar_abundances=solar)
        alpha_M = alpha_H - M_H
        C_H = A_X[5] - solar[5]  # Z=6 carbon (0-indexed)
        C_M = C_H - M_H

    # Reference wavelength for MARCS models
    reference_wavelength = 5e-5  # 5000 Å in cm

    # Load MARCS grid — this handles placeholder / CI detection and emits warnings.
    # Always call load_marcs_grid() first so warnings are raised and the CI
    # compatibility path is exercised regardless of the JIT data cache.
    nodes, grid = load_marcs_grid()

    # Python-level bounds check (matches the original lazy_multilinear_interpolation
    # behaviour, which raised AtmosphereInterpolationError for out-of-bounds params).
    param_vals  = [float(Teff), float(logg), float(M_H), float(alpha_M), float(C_M)]
    param_names = ["Teff", "log(g)", "[M/H]", "[α/M]", "[C/metals]"]
    for i, (pv, pname, pnodes) in enumerate(zip(param_vals, param_names, nodes)):
        lo = float(pnodes[0])
        hi = float(pnodes[-1])
        if not (lo <= pv <= hi):
            raise AtmosphereInterpolationError(
                f"Can't interpolate grid. {pname} is out of bounds. "
                f"({pv} ∉ [{lo}, {hi}])"
            )

    # Build (or retrieve) JIT-compatible padded node arrays from the same grid.
    # Use the module-level cache keyed on grid identity to avoid re-computing.
    grid_id = id(grid)
    if "data" not in _marcs_jit_cache or _marcs_jit_cache.get("grid_id") != grid_id:
        lengths = [len(n) for n in nodes]
        max_len = max(lengths)
        nodes_padded = jnp.array(
            np.array([
                np.concatenate([np.asarray(n), np.full(max_len - len(n), np.inf)])
                for n in nodes
            ]),
            dtype=jnp.float64
        )
        nodes_lengths = jnp.array(lengths, dtype=jnp.int32)
        _marcs_jit_cache["data"] = (nodes_padded, nodes_lengths, grid)
        _marcs_jit_cache["grid_id"] = grid_id
    nodes_padded, nodes_lengths, grid = _marcs_jit_cache["data"]

    # Build parameter vector (float64)
    params = jnp.array([Teff, logg, M_H, alpha_M, C_M], dtype=jnp.float64)

    # --- JIT kernel: multilinear interpolation ---
    # perturb_at_grid_values is handled inside the JIT kernel when True.
    # When False, we skip the nudge by using params as-is (the kernel always
    # nudges, but calling with perturb_at_grid_values=False on non-exact points
    # is a no-op).  To faithfully honour the flag, we run the old path for the
    # False case (which is rare) and the JIT path for the common True case.
    if perturb_at_grid_values:
        atm_quants = _interpolate_marcs_jit(params, nodes_padded, nodes_lengths, grid)
    else:
        # Legacy path: use original non-JIT code (avoids perturbation)
        param_names = ["Teff", "log(g)", "[M/H]", "[α/M]", "[C/metals]"]
        atm_quants = lazy_multilinear_interpolation(
            params, nodes, grid,
            param_names=param_names,
            perturb_at_grid_values=False
        )

    # Back in Python: strip NaN layers and build atmosphere object
    atm_quants_np = np.asarray(atm_quants)
    valid = ~np.isnan(atm_quants_np[:, 3])  # tau_ref column

    T       = atm_quants_np[valid, 0]
    log_ne  = atm_quants_np[valid, 1]
    log_n   = atm_quants_np[valid, 2]
    tau_ref = atm_quants_np[valid, 3]
    asinh_z = atm_quants_np[valid, 4]

    # Transform back to physical values
    ne = np.exp(log_ne)
    n  = np.exp(log_n)
    z  = np.sinh(asinh_z)

    n_layers = int(valid.sum())

    # Check for negative optical depths (indicates unreliable interpolation)
    if np.any(tau_ref < 0):
        raise AtmosphereInterpolationError(
            "Interpolated atmosphere has negative optical depths and is not reliable."
        )

    if spherical:
        # Calculate photospheric radius: R = sqrt(G * M_sun / g)
        solar_mass_cgs = 1.9885e33  # grams
        R_phot = float(np.sqrt(G_cgs * solar_mass_cgs / (10.0 ** float(logg))))

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
            R_photosphere=R_phot,
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

    return atm
