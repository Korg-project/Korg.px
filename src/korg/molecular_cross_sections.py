"""
Precomputed molecular cross-sections for fast spectral synthesis.

MolecularCrossSection precomputes line absorption over a grid of
(vmic, log10_T, wavelength) and interpolates at runtime.
"""

import numpy as np
from typing import List, Optional


class MolecularCrossSection:
    """
    Precomputed molecular absorption cross-section on a (vmic, logT, wavelength) grid.

    Create with MolecularCrossSection(linelist, wavelengths). Nothing consumes one yet:
    the Python-orchestrated ``synthesize_spectrum`` that took a molecular_cross_sections
    argument has been deleted, and the traced synthesis does not implement it.

    See save_molecular_cross_section / read_molecular_cross_section for persistence.
    """

    def __init__(
        self,
        linelist: list,
        wavelengths_angstrom,
        cutoff_alpha: float = 1e-32,
        vmic_vals=None,
        log_temp_vals=None,
    ):
        """
        Precompute molecular cross-sections.

        Args:
            linelist: List of Line objects, all of the same species.
            wavelengths_angstrom: 1D array of wavelengths in Å.
            cutoff_alpha: Absorption threshold below which lines are truncated.
            vmic_vals: Microturbulence velocities in km/s to precompute at.
                Default: [0, 1/3, 2/3, 1, 1.5, 2, 8/3, 10/3, 4, 14/3, 16/3].
            log_temp_vals: log10(T) values to precompute at. Default: 3.0:0.04:5.0.
        """
        from .species import Species

        if not linelist:
            raise ValueError("linelist cannot be empty")

        species_set = set(l.species for l in linelist)
        if len(species_set) != 1:
            raise ValueError(
                f"All lines must be of the same species, got: {species_set}"
            )
        self.species = linelist[0].species

        wls = np.asarray(wavelengths_angstrom, dtype=float)
        self.wavelengths_angstrom = wls

        if vmic_vals is None:
            vmic_vals = [0.0, 1/3, 2/3, 1.0, 1.5, 2.0, 8/3, 10/3, 4.0, 14/3, 16/3]
        if log_temp_vals is None:
            log_temp_vals = np.arange(3.0, 5.0 + 0.02, 0.04)

        self.vmic_vals = np.asarray(vmic_vals, dtype=float)
        self.log_temp_vals = np.asarray(log_temp_vals, dtype=float)
        temperatures = 10.0 ** self.log_temp_vals

        # Pre-allocate cross-section grid
        alpha_grid = np.zeros(
            (len(self.vmic_vals), len(self.log_temp_vals), len(wls)),
            dtype=float
        )

        from .line_absorption import line_absorption
        from .data_loader import default_partition_funcs

        wls_cm = wls * 1e-8
        ne_zeros = np.zeros(len(temperatures))
        # Number density: 1/cutoff_alpha so absorption *= cutoff_alpha gives cross-section
        n_dict = {self.species: np.ones(len(temperatures)) / cutoff_alpha}
        pf = {self.species: default_partition_funcs.get(self.species, lambda x: 1.0)}
        continuum = lambda wl: np.ones(len(temperatures))

        for i_vmic, vmic_kms in enumerate(self.vmic_vals):
            xi_cms = vmic_kms * 1e5  # km/s -> cm/s
            alpha_raw = line_absorption(
                linelist, wls_cm, temperatures, ne_zeros,
                n_dict, pf, xi_cms, continuum,
                cutoff_threshold=1.0,  # no truncation (n_dict already scaled)
                use_jit=False
            )
            # Scale by cutoff_alpha to get actual cross-sections
            alpha_grid[i_vmic] = alpha_raw * cutoff_alpha

        self._grid = alpha_grid
        self._build_interpolator()

    def _build_interpolator(self):
        from scipy.interpolate import RegularGridInterpolator
        self._interp = RegularGridInterpolator(
            (self.vmic_vals, self.log_temp_vals, self.wavelengths_angstrom),
            self._grid,
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )

    def __call__(self, vmic_kms: float, log_T: float, wl_angstrom: float) -> float:
        """
        Interpolate the cross-section at a given vmic, logT, and wavelength.

        Args:
            vmic_kms: Microturbulence in km/s
            log_T: log10 of temperature
            wl_angstrom: Wavelength in Å

        Returns:
            Cross-section in cm⁻¹ per (cm⁻³ number density)
        """
        return float(self._interp([[vmic_kms, log_T, wl_angstrom]])[0])

    def interpolate_layer(self, vmic_kms: float, log_T: float) -> np.ndarray:
        """
        Interpolate cross-sections at all wavelengths for a single atmosphere layer.

        Args:
            vmic_kms: Microturbulence in km/s
            log_T: log10 of temperature

        Returns:
            Cross-section array (len(wavelengths),) in cm⁻¹ per (cm⁻³)
        """
        pts = np.column_stack([
            np.full(len(self.wavelengths_angstrom), vmic_kms),
            np.full(len(self.wavelengths_angstrom), log_T),
            self.wavelengths_angstrom
        ])
        return self._interp(pts)

    def __repr__(self):
        T_min = round(10.0 ** self.log_temp_vals[0])
        T_max = round(10.0 ** self.log_temp_vals[-1])
        wl_min = round(self.wavelengths_angstrom[0])
        wl_max = round(self.wavelengths_angstrom[-1])
        v_min = round(self.vmic_vals[0])
        v_max = round(self.vmic_vals[-1])
        return (
            f"MolecularCrossSection for {self.species} "
            f"({wl_min} Å ≤ λ ≤ {wl_max} Å, "
            f"{T_min} K ≤ T ≤ {T_max} K, "
            f"{v_min} km/s ≤ vmic ≤ {v_max} km/s)"
        )


def interpolate_molecular_cross_sections(
    alpha: np.ndarray,
    molecular_cross_sections: list,
    wavelengths_angstrom: np.ndarray,
    temperatures: np.ndarray,
    vmic,
    number_densities: dict,
) -> np.ndarray:
    """
    Add molecular cross-section contributions to absorption coefficient array.

    Args:
        alpha: Absorption array of shape (n_layers, n_wavelengths) in cm⁻¹.
            Modified in place.
        molecular_cross_sections: List of MolecularCrossSection objects.
        wavelengths_angstrom: Wavelength grid in Å.
        temperatures: Temperature at each layer in K.
        vmic: Microturbulence in km/s — scalar or per-layer array.
        number_densities: Dict mapping Species to number density arrays (cm⁻³).

    Returns:
        alpha with molecular contributions added (same array, modified in place).
    """
    if not molecular_cross_sections:
        return alpha

    n_layers = len(temperatures)
    log_temps = np.log10(temperatures)

    for sigma in molecular_cross_sections:
        spec = sigma.species
        n_dens = number_densities.get(spec, number_densities.get(str(spec)))
        if n_dens is None:
            continue
        n_arr = np.asarray(n_dens)

        for i in range(n_layers):
            vm = vmic if np.isscalar(vmic) else vmic[i]
            cross_section = sigma.interpolate_layer(float(vm), float(log_temps[i]))
            alpha[i] += cross_section * float(n_arr[i])

    return alpha


def save_molecular_cross_section(filename: str, cross_section: MolecularCrossSection) -> None:
    """
    Save a MolecularCrossSection to an HDF5 file.

    Args:
        filename: Output file path (should end in .h5)
        cross_section: MolecularCrossSection object to save
    """
    import h5py

    with h5py.File(filename, 'w') as f:
        f.create_dataset('wls_angstrom', data=cross_section.wavelengths_angstrom)
        f.create_dataset('vmic_vals', data=cross_section.vmic_vals)
        f.create_dataset('log_temp_vals', data=cross_section.log_temp_vals)
        f.create_dataset('alpha_grid', data=cross_section._grid)
        f.create_dataset('species', data=str(cross_section.species))


def read_molecular_cross_section(filename: str) -> MolecularCrossSection:
    """
    Read a MolecularCrossSection from an HDF5 file.

    Args:
        filename: Path to HDF5 file created by save_molecular_cross_section

    Returns:
        MolecularCrossSection object
    """
    import h5py
    from .species import Species

    with h5py.File(filename, 'r') as f:
        wls = f['wls_angstrom'][:]
        vmic_vals = f['vmic_vals'][:]
        log_temp_vals = f['log_temp_vals'][:]
        alpha_grid = f['alpha_grid'][:]
        species_str = f['species'][()].decode() if isinstance(f['species'][()], bytes) \
            else f['species'][()]

    obj = object.__new__(MolecularCrossSection)
    obj.wavelengths_angstrom = wls
    obj.vmic_vals = vmic_vals
    obj.log_temp_vals = log_temp_vals
    obj.species = Species(species_str)
    obj._grid = alpha_grid
    obj._build_interpolator()
    return obj
