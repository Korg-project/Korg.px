"""
Pre-processing utilities for JIT-compatible spectral synthesis.

These functions perform the Python-level work (linelist filtering, wavelength
grid creation, atmosphere and line-data packing) that cannot run inside a
``jax.jit`` boundary.  Their outputs are plain JAX/NumPy arrays that can be
passed directly to a JIT-compiled synthesis kernel.

Typical usage
-------------
::

    from korg.synthesis_preparation import (
        prepare_wavelength_grid,
        preprocess_linelist,
        prepare_atmosphere,
    )

    # One-time preparation (Python, slow path):
    wls_ang, wls_cm = prepare_wavelength_grid(5000.0, 5010.0, wl_step=0.01)
    pl = preprocess_linelist(linelist, wls_cm, line_buffer_cm=10e-8)
    atm_arrays = prepare_atmosphere(atmosphere)

    # Repeated JIT call (fast path, reuses compiled kernel):
    result = synthesize_jit(atm_arrays, pl, wls_cm, abs_abundances, ...)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import jax.numpy as jnp

from .linelist import Line
from .species import Species
from .atmosphere import PlanarAtmosphere, ShellAtmosphere


# ---------------------------------------------------------------------------
# PreparedLinelist
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PreparedLinelist:
    """
    Linelist pre-packed into contiguous NumPy arrays ready for a JIT kernel.

    All per-line arrays are 1-D with shape ``(n_lines,)`` and dtype float64
    unless noted.  The ordering matches the input linelist after filtering and
    sorting by wavelength.

    Attributes
    ----------
    wl : ndarray, shape (n_lines,)
        Line centre wavelengths [cm].
    log_gf : ndarray, shape (n_lines,)
        Oscillator strength log₁₀(gf).
    E_lower : ndarray, shape (n_lines,)
        Lower level excitation energy [eV].
    gamma_rad : ndarray, shape (n_lines,)
        Radiative damping constant [rad s⁻¹].
    gamma_stark : ndarray, shape (n_lines,)
        Stark damping constant (0.0 if absent).
    vdW_sigma : ndarray, shape (n_lines,)
        van der Waals σ parameter (ABO theory) or packed log|C₆| (< 0).
    vdW_alpha : ndarray, shape (n_lines,)
        van der Waals α parameter (ABO theory) or -1.0 for log|C₆| encoding.
    mass : ndarray, shape (n_lines,)
        Atomic/molecular mass [amu].
    is_molecule : ndarray of bool, shape (n_lines,)
        True for molecular lines.
    species_id : ndarray of int32, shape (n_lines,)
        Integer index into ``species_list``.
    species_list : tuple of Species
        Unique species referenced by ``species_id``, in index order.
    n_lines : int
        Number of lines after filtering.
    """
    wl: np.ndarray
    log_gf: np.ndarray
    E_lower: np.ndarray
    gamma_rad: np.ndarray
    gamma_stark: np.ndarray
    vdW_sigma: np.ndarray
    vdW_alpha: np.ndarray
    mass: np.ndarray
    is_molecule: np.ndarray
    species_id: np.ndarray
    species_list: tuple

    @property
    def n_lines(self) -> int:
        return len(self.wl)

    def as_jax(self) -> "PreparedLinelist":
        """Return a copy with all numeric arrays converted to JAX arrays."""
        return PreparedLinelist(
            wl=jnp.asarray(self.wl),
            log_gf=jnp.asarray(self.log_gf),
            E_lower=jnp.asarray(self.E_lower),
            gamma_rad=jnp.asarray(self.gamma_rad),
            gamma_stark=jnp.asarray(self.gamma_stark),
            vdW_sigma=jnp.asarray(self.vdW_sigma),
            vdW_alpha=jnp.asarray(self.vdW_alpha),
            mass=jnp.asarray(self.mass),
            is_molecule=jnp.asarray(self.is_molecule),
            species_id=jnp.asarray(self.species_id),
            species_list=self.species_list,
        )


# ---------------------------------------------------------------------------
# AtmosphereArrays
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AtmosphereArrays:
    """
    Atmosphere quantities packed into contiguous NumPy arrays.

    All arrays are 1-D with shape ``(n_layers,)`` and dtype float64.

    Attributes
    ----------
    T : ndarray, shape (n_layers,)
        Temperature at each layer [K].
    ne : ndarray, shape (n_layers,)
        Electron number density [cm⁻³].
    n_total : ndarray, shape (n_layers,)
        Total number density [cm⁻³].
    log_tau_ref : ndarray, shape (n_layers,)
        log₁₀(τ) at the reference wavelength (5000 Å for MARCS).
    z : ndarray, shape (n_layers,)
        Spatial coordinate: height [cm] for planar, radius [cm] for spherical.
    spherical : bool
        True for shell (spherical) atmospheres, False for planar.
    n_layers : int
        Number of atmospheric layers.
    """
    T: np.ndarray
    ne: np.ndarray
    n_total: np.ndarray
    log_tau_ref: np.ndarray
    z: np.ndarray
    spherical: bool

    @property
    def n_layers(self) -> int:
        return len(self.T)

    def as_jax(self) -> "AtmosphereArrays":
        """Return a copy with all numeric arrays converted to JAX arrays."""
        return AtmosphereArrays(
            T=jnp.asarray(self.T),
            ne=jnp.asarray(self.ne),
            n_total=jnp.asarray(self.n_total),
            log_tau_ref=jnp.asarray(self.log_tau_ref),
            z=jnp.asarray(self.z),
            spherical=self.spherical,
        )


# ---------------------------------------------------------------------------
# prepare_wavelength_grid
# ---------------------------------------------------------------------------

def prepare_wavelength_grid(
    wl_start: float,
    wl_end: float,
    *,
    n_points: Optional[int] = None,
    wl_step: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a uniform wavelength grid in Angstroms and centimetres.

    Exactly one of *n_points* or *wl_step* must specify the sampling; if both
    are given *n_points* takes priority.

    Parameters
    ----------
    wl_start : float
        Starting wavelength [Å] (inclusive).
    wl_end : float
        Ending wavelength [Å] (inclusive).
    n_points : int, optional
        Number of grid points.  If given, *wl_step* is ignored.
    wl_step : float, optional
        Grid spacing [Å] (default 0.01 Å).  Used when *n_points* is None.

    Returns
    -------
    wavelengths_ang : ndarray, shape (n,), float64
        Wavelength grid [Å].
    wavelengths_cm : ndarray, shape (n,), float64
        Wavelength grid [cm] (same grid, different units).

    Raises
    ------
    ValueError
        If *wl_start* >= *wl_end*, or the implied grid would be empty.

    Examples
    --------
    >>> wls_ang, wls_cm = prepare_wavelength_grid(5000.0, 5010.0, wl_step=0.01)
    >>> wls_ang.shape
    (1001,)
    >>> wls_cm[0]
    5e-05
    """
    if wl_start >= wl_end:
        raise ValueError(
            f"wl_start ({wl_start} Å) must be less than wl_end ({wl_end} Å)"
        )
    if n_points is not None:
        if n_points < 2:
            raise ValueError(f"n_points must be >= 2, got {n_points}")
        wavelengths_ang = np.linspace(wl_start, wl_end, int(n_points), dtype=np.float64)
    else:
        if wl_step <= 0:
            raise ValueError(f"wl_step must be positive, got {wl_step}")
        wavelengths_ang = np.arange(wl_start, wl_end + wl_step * 0.5, wl_step,
                                    dtype=np.float64)
    if len(wavelengths_ang) < 1:
        raise ValueError(
            f"Grid is empty: wl_start={wl_start}, wl_end={wl_end}, wl_step={wl_step}"
        )
    wavelengths_cm = wavelengths_ang * 1e-8
    return wavelengths_ang, wavelengths_cm


# ---------------------------------------------------------------------------
# preprocess_linelist
# ---------------------------------------------------------------------------

def _vdW_to_sigma_alpha(vdW) -> Tuple[float, float]:
    """
    Unpack a line's ``vdW`` field into (sigma, alpha) floats.

    The vdW field can be:
    - ``(sigma, alpha)`` tuple  — ABO theory parameters
    - scalar < 0               — log|C₆| packed as ``(log|C₆|, -1.0)``
    - ``(0.0, -1.0)``          — no broadening sentinel
    - ``None`` / ``float('nan')`` — no broadening
    """
    if vdW is None:
        return 0.0, -1.0
    if isinstance(vdW, (tuple, list)):
        return float(vdW[0]), float(vdW[1])
    v = float(vdW)
    if np.isnan(v):
        return 0.0, -1.0
    # Negative scalar: log|C₆| encoding
    return v, -1.0


def preprocess_linelist(
    linelist: List[Line],
    wavelengths_cm: np.ndarray,
    line_buffer_cm: float = 10e-8,
) -> PreparedLinelist:
    """
    Filter a linelist to a wavelength range and pack it into array form.

    This is the Python-level preparation step that must run *before* any
    ``jax.jit`` boundary.  The resulting :class:`PreparedLinelist` contains
    only plain NumPy arrays and can be passed directly to a JIT kernel.

    Parameters
    ----------
    linelist : list of Line
        Full (unfiltered) list of spectral lines, in any order.
    wavelengths_cm : array
        Synthesis wavelength grid [cm].  Lines whose centres fall outside
        ``[wavelengths_cm[0] - line_buffer_cm, wavelengths_cm[-1] + line_buffer_cm]``
        are excluded.
    line_buffer_cm : float, optional
        Extra wavelength margin beyond the synthesis range within which lines
        are still retained [cm].  Default: ``10e-8`` cm = 10 Å.

    Returns
    -------
    PreparedLinelist
        Packed representation of the filtered, wavelength-sorted linelist.
        :attr:`PreparedLinelist.n_lines` is 0 when no lines are in range.

    Notes
    -----
    - Lines with ``Species("H_I")`` are excluded (hydrogen opacity is handled
      separately via :func:`korg.synthesis.hydrogen_line_absorption`).
    - The original ``Line`` objects are not retained; only their scalar
      attributes are stored in the output arrays.

    Examples
    --------
    >>> wls_ang, wls_cm = prepare_wavelength_grid(5000.0, 5010.0)
    >>> pl = preprocess_linelist(my_linelist, wls_cm)
    >>> pl.n_lines   # number of lines in range
    42
    >>> pl.wl.shape  # (n_lines,) float64 array in cm
    (42,)
    """
    wls_cm = np.asarray(wavelengths_cm, dtype=np.float64)
    lo = wls_cm[0] - line_buffer_cm
    hi = wls_cm[-1] + line_buffer_cm

    h_i = Species("H_I")

    if not linelist:
        return _empty_prepared_linelist()

    # Filter: wavelength range + exclude H I
    filtered = [l for l in linelist if lo <= l.wl <= hi and l.species != h_i]

    if not filtered:
        return _empty_prepared_linelist()

    # Sort by wavelength (most functions expect sorted input)
    filtered.sort(key=lambda l: l.wl)

    # Unique species (preserve insertion order for stable indexing)
    seen: dict = {}
    for l in filtered:
        if l.species not in seen:
            seen[l.species] = len(seen)
    species_list = tuple(seen.keys())
    species_to_id = seen

    n = len(filtered)
    wl       = np.empty(n, dtype=np.float64)
    log_gf   = np.empty(n, dtype=np.float64)
    E_lower  = np.empty(n, dtype=np.float64)
    g_rad    = np.empty(n, dtype=np.float64)
    g_stark  = np.empty(n, dtype=np.float64)
    vdW_s    = np.empty(n, dtype=np.float64)
    vdW_a    = np.empty(n, dtype=np.float64)
    mass     = np.empty(n, dtype=np.float64)
    is_mol   = np.empty(n, dtype=bool)
    sp_id    = np.empty(n, dtype=np.int32)

    for i, line in enumerate(filtered):
        wl[i]      = line.wl
        log_gf[i]  = line.log_gf
        E_lower[i] = line.E_lower
        g_rad[i]   = line.gamma_rad if line.gamma_rad is not None else 0.0
        g_stark[i] = (line.gamma_stark if line.gamma_stark is not None else 0.0)
        s, a       = _vdW_to_sigma_alpha(line.vdW)
        vdW_s[i]   = s
        vdW_a[i]   = a
        mass[i]    = line.species.get_mass()
        is_mol[i]  = line.species.formula.is_molecule()
        sp_id[i]   = species_to_id[line.species]

    return PreparedLinelist(
        wl=wl, log_gf=log_gf, E_lower=E_lower,
        gamma_rad=g_rad, gamma_stark=g_stark,
        vdW_sigma=vdW_s, vdW_alpha=vdW_a,
        mass=mass, is_molecule=is_mol,
        species_id=sp_id, species_list=species_list,
    )


def _empty_prepared_linelist() -> PreparedLinelist:
    """Return a PreparedLinelist with zero lines."""
    empty_f = np.empty(0, dtype=np.float64)
    empty_b = np.empty(0, dtype=bool)
    empty_i = np.empty(0, dtype=np.int32)
    return PreparedLinelist(
        wl=empty_f, log_gf=empty_f, E_lower=empty_f,
        gamma_rad=empty_f, gamma_stark=empty_f,
        vdW_sigma=empty_f, vdW_alpha=empty_f,
        mass=empty_f, is_molecule=empty_b,
        species_id=empty_i, species_list=(),
    )


# ---------------------------------------------------------------------------
# prepare_atmosphere
# ---------------------------------------------------------------------------

def prepare_atmosphere(atmosphere) -> AtmosphereArrays:
    """
    Extract atmosphere properties into contiguous NumPy arrays.

    Accepts both :class:`~korg.atmosphere.PlanarAtmosphere` and
    :class:`~korg.atmosphere.ShellAtmosphere` instances.  The spatial
    coordinate is the height *z* [cm] for planar atmospheres and the radius
    *r* [cm] for spherical ones.

    Parameters
    ----------
    atmosphere : PlanarAtmosphere or ShellAtmosphere
        Model atmosphere, typically from :func:`~korg.atmosphere.read_model_atmosphere`
        or :func:`~korg.marcs_interpolation.interpolate_marcs`.

    Returns
    -------
    AtmosphereArrays
        Packed atmosphere with float64 NumPy arrays.

    Raises
    ------
    TypeError
        If *atmosphere* is not a recognised atmosphere type.

    Examples
    --------
    >>> atm_arrays = prepare_atmosphere(atm)
    >>> atm_arrays.T.shape
    (56,)
    >>> atm_arrays.spherical
    False
    """
    if isinstance(atmosphere, ShellAtmosphere):
        spherical = True
        z = np.asarray(atmosphere.r, dtype=np.float64)
    elif isinstance(atmosphere, PlanarAtmosphere):
        spherical = False
        z = np.asarray(atmosphere.z, dtype=np.float64)
    else:
        raise TypeError(
            f"Expected PlanarAtmosphere or ShellAtmosphere, got {type(atmosphere).__name__}"
        )

    return AtmosphereArrays(
        T=np.asarray(atmosphere.T, dtype=np.float64),
        ne=np.asarray(atmosphere.ne, dtype=np.float64),
        n_total=np.asarray(atmosphere.n_total, dtype=np.float64),
        log_tau_ref=np.asarray(atmosphere.log_tau_ref, dtype=np.float64),
        z=z,
        spherical=spherical,
    )
