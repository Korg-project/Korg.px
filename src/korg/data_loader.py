"""
Data loading utilities for Korg.

Functions for loading ionization energies, partition functions,
and other tabulated data needed for spectral synthesis.
"""

import os
import numpy as np
import h5py
from .species import Species
from .cubic_splines import cubic_spline


# Get the data directory path (relative to this file, inside the package)
_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class LazyPartitionFunction:
    """
    Lazy wrapper for partition function interpolation.

    Only creates the cubic spline interpolator when first called,
    avoiding expensive upfront computation.
    """

    def __init__(self, log_temps, partition_values):
        """
        Parameters
        ----------
        log_temps : array
            log(Temperature) grid points
        partition_values : array
            Partition function values at grid points
        """
        self._log_temps = log_temps
        self._values = partition_values
        self._interpolator = None

    def __call__(self, log_T):
        """
        Evaluate partition function at log(T).

        Creates interpolator on first call.
        """
        if self._interpolator is None:
            # Only create the spline when first needed
            self._interpolator = cubic_spline(self._log_temps, self._values,
                                             extrapolate=True)
        return self._interpolator(log_T)



def load_ionization_energies(filename=None):
    """
    Load ionization energies from Barklem & Collet 2016 data file.

    Parameters
    ----------
    filename : str, optional
        Path to ionization energies file. If None, uses default path.

    Returns
    -------
    dict
        Dictionary mapping atomic numbers (int) to arrays of ionization
        energies [χ₁, χ₂, χ₃] in eV.

    Notes
    -----
    Values of -1.0 indicate that ionization energy is not applicable
    (e.g., χ₃ for hydrogen).
    """
    if filename is None:
        filename = os.path.join(_DATA_DIR, "barklem_collet_2016",
                               "BarklemCollet2016-ionization_energies.dat")

    ionization_energies = {}

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue

            tokens = line.split()
            Z = int(tokens[0])
            # tokens[1] is the element symbol (ignored)
            chi1, chi2, chi3 = float(tokens[2]), float(tokens[3]), float(tokens[4])
            ionization_energies[Z] = np.array([chi1, chi2, chi3])

    return ionization_energies


def load_atomic_partition_functions(filename=None):
    """
    Load atomic partition functions from HDF5 file.

    Parameters
    ----------
    filename : str, optional
        Path to partition function HDF5 file. If None, uses default path.

    Returns
    -------
    dict
        Dictionary mapping Species to CubicSpline interpolators over log(T).
        Each interpolator maps log(T) to the partition function value.

    Notes
    -----
    The partition functions are custom calculated from NIST energy levels.
    They do NOT include plasma effects (occupation probabilities).
    """
    if filename is None:
        filename = os.path.join(_DATA_DIR, "atomic_partition_funcs",
                               "partition_funcs.h5")

    partition_funcs = {}

    with h5py.File(filename, 'r') as f:
        # Read temperature grid parameters
        logT_min = f['logT_min'][()]
        logT_step = f['logT_step'][()]
        logT_max = f['logT_max'][()]
        logTs = np.arange(logT_min, logT_max + logT_step/2, logT_step)

        # Load partition functions for each species
        from .atomic_data import atomic_symbols

        for symbol in atomic_symbols:
            for ionization in ["I", "II", "III"]:
                # Skip invalid combinations
                if (symbol == "H" and ionization != "I") or \
                   (symbol == "He" and ionization == "III"):
                    continue

                spec_str = f"{symbol} {ionization}"
                if spec_str in f:
                    species = Species(spec_str)
                    U_values = f[spec_str][:]
                    # Create cubic spline interpolator
                    partition_funcs[species] = cubic_spline(logTs, U_values,
                                                           extrapolate=True)

        # Handle bare nuclei (partition function = 1 always)
        all_ones = np.ones(len(logTs))
        partition_funcs[Species("H II")] = cubic_spline(logTs, all_ones,
                                                       extrapolate=True)
        partition_funcs[Species("He III")] = cubic_spline(logTs, all_ones,
                                                         extrapolate=True)

    return partition_funcs


def load_gauntff_table(filename=None):
    """
    Load thermally-averaged free-free Gaunt factors from van Hoof et al. 2014.

    Parameters
    ----------
    filename : str, optional
        Path to gaunt factor data file. If None, uses default path.

    Returns
    -------
    tuple
        (table_values, log10_gamma2, log10_u) where:
        - table_values: 2D array of gaunt factor values
        - log10_gamma2: 1D array of log₁₀(γ²) values
        - log10_u: 1D array of log₁₀(u) values

    Notes
    -----
    This loads the non-relativistic free-free data from van Hoof et al. (2014):
    https://ui.adsabs.harvard.edu/abs/2014MNRAS.444..420V

    The table is valid for temperatures up to ~100 MK.

    log₁₀(γ²) = log₁₀(Rydberg*Z²/(k*Tₑ))
    log₁₀(u) = log₁₀(h*ν/(k*Tₑ))
    """
    if filename is None:
        filename = os.path.join(_DATA_DIR, "vanHoof2014-nr-gauntff.dat")

    def parse_header_line(dtype, s):
        """Parse a header line, removing comments."""
        hash_pos = s.find('#')
        if hash_pos != -1:
            s = s[:hash_pos]
        return np.array(s.split(), dtype=dtype)

    with open(filename, 'r') as f:
        # Skip initial comment block
        line = f.readline()
        while line.startswith('#'):
            line = f.readline()

        # Parse header
        magic_number = parse_header_line(int, line)[0]
        assert magic_number == 20140210, f"Unexpected magic number: {magic_number}"

        num_gamma2, num_u = parse_header_line(int, f.readline())
        log10_gamma2_start = parse_header_line(float, f.readline())[0]
        log10_u_start = parse_header_line(float, f.readline())[0]
        step_size = parse_header_line(float, f.readline())[0]

        # Create coordinate arrays
        log10_gamma2 = np.arange(log10_gamma2_start,
                                 log10_gamma2_start + num_gamma2 * step_size,
                                 step_size)[:num_gamma2]
        log10_u = np.arange(log10_u_start,
                           log10_u_start + num_u * step_size,
                           step_size)[:num_u]

        # Skip second comment block
        line = f.readline()
        while line.startswith('#'):
            line = f.readline()

        # Read table values
        table_values = np.zeros((len(log10_u), len(log10_gamma2)))
        table_values[0, :] = np.array(line.split(), dtype=float)

        for i in range(1, len(log10_u)):
            line = f.readline()
            table_values[i, :] = np.array(line.split(), dtype=float)

    return table_values, log10_gamma2, log10_u


def load_Hminus_bf_data(filename=None):
    """
    Load H⁻ bound-free cross-sections from McLaughlin 2017.

    Parameters
    ----------
    filename : str, optional
        Path to H⁻ bf data file. If None, uses default path.

    Returns
    -------
    tuple
        (frequencies, cross_sections) where:
        - frequencies: 1D array of frequencies in Hz
        - cross_sections: 1D array of cross-sections in cm²

    Notes
    -----
    Data from McLaughlin+ 2017:
    https://ui.adsabs.harvard.edu/abs/2017JPhB...50k4001M
    """
    if filename is None:
        filename = os.path.join(_DATA_DIR, "McLaughlin2017Hminusbf.h5")

    with h5py.File(filename, 'r') as f:
        frequencies = f['nu'][:]
        cross_sections = f['sigma'][:]

    return frequencies, cross_sections


def load_HI_bf_cross_sections(filename=None, n_max=6):
    """
    Load H I bound-free cross-sections from Nahar 2021.

    Parameters
    ----------
    filename : str, optional
        Path to H I bf data file. If None, uses default path.
    n_max : int, optional
        Maximum principal quantum number to load (default: 6)

    Returns
    -------
    list of tuples
        List of (n, energies, cross_sections) tuples where:
        - n: principal quantum number (int)
        - energies: 1D array of photon energies in eV
        - cross_sections: 1D array of cross-sections in Megabarns

    Notes
    -----
    Cross-sections from Nahar 2021:
    https://ui.adsabs.harvard.edu/abs/2021Atoms...9...73N

    The cross-sections are tabulated as a function of photon energy above
    the ionization threshold for each level. For Korg, we typically use
    only n=1 through n=6, as higher levels are well-approximated by analytic
    formulas and/or dissolved by pressure effects.
    """
    if filename is None:
        filename = os.path.join(_DATA_DIR, "bf_cross-sections",
                               "individual_H_cross-sections.h5")

    cross_sections = []

    with h5py.File(filename, 'r') as f:
        n_values = f['n'][:]
        E_all = f['E'][:]  # Shape: (n_levels, n_energies)
        sigma_all = f['sigma'][:]  # Shape: (n_levels, n_energies)

        for i, n in enumerate(n_values):
            if n > n_max:
                break

            # Get energies and cross-sections for this level
            energies = E_all[i, :]
            sigmas = sigma_all[i, :]

            cross_sections.append((int(n), energies, sigmas))

    return cross_sections


# Load data when module is imported (like Julia does)
ionization_energies = load_ionization_energies()
atomic_partition_functions = load_atomic_partition_functions()
gauntff_table, gauntff_log10_gamma2, gauntff_log10_u = load_gauntff_table()
Hminus_bf_frequencies, Hminus_bf_cross_sections = load_Hminus_bf_data()
HI_bf_cross_sections = load_HI_bf_cross_sections(n_max=6)


# Lazy-loaded metal bf cross-sections (loaded on first access)
_metal_bf_cross_sections_cache = None


def load_metal_bf_cross_sections(filename=None):
    """
    Load metal bound-free photoionization cross-sections from HDF5 file.

    This function loads precomputed cross-section tables from TOPBase and NORAD
    for various metal species. The data is cached after first load.

    Parameters
    ----------
    filename : str, optional
        Path to cross-section HDF5 file. If None, uses default path.

    Returns
    -------
    dict
        Dictionary with the following structure:
        {
            'nu_grid': array of frequencies in Hz,
            'logT_grid': array of log10(T) values,
            'species': {
                'Fe I': 2D array of ln(cross-section in Mb),
                'Ca I': 2D array of ln(cross-section in Mb),
                ...
            }
        }

    Notes
    -----
    Cross-sections are tabulated for:
    - Temperature: 100 K < T < 100,000 K (log T = 2.0 to 5.0)
    - Wavelength: 500 Å < λ < 30,000 Å (frequencies 1e14 to 6e15 Hz)

    The tables are 2D: shape (n_logT, n_nu) where each element is ln(σ)
    in Megabarns.

    Species included: Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, S, Ar, Ca
    (from TOPBase) and Fe (from NORAD). For each element, neutral and singly
    ionized species are available assuming LTE distribution of energy levels.

    References
    ----------
    TOPBase: http://cdsweb.u-strasbg.fr/topbase/topbase.html
    NORAD: https://www.astronomy.ohio-state.edu/nahar.1/nahar_radiativeatomicdata/
    """
    global _metal_bf_cross_sections_cache

    # Return cached data if available
    if _metal_bf_cross_sections_cache is not None:
        return _metal_bf_cross_sections_cache

    if filename is None:
        filename = os.path.join(_DATA_DIR, "bf_cross-sections", "bf_cross-sections.h5")

    cross_sections = {'species': {}}

    with h5py.File(filename, 'r') as f:
        # Read grid parameters
        logT_min = f['logT_min'][()]
        logT_max = f['logT_max'][()]
        logT_step = f['logT_step'][()]
        nu_min = f['nu_min'][()]
        nu_max = f['nu_max'][()]
        nu_step = f['nu_step'][()]

        # Create grids
        logT_grid = np.arange(logT_min, logT_max + logT_step/2, logT_step)
        nu_grid = np.arange(nu_min, nu_max + nu_step/2, nu_step)

        cross_sections['logT_grid'] = logT_grid
        cross_sections['nu_grid'] = nu_grid

        # Load cross-sections for each species
        cs_group = f['cross-sections']
        for species_name in cs_group.keys():
            # Skip H I, He II, H II as they're handled separately
            if species_name in ['H I', 'He II', 'H II']:
                continue

            # Read cross-section data (stored as ln(σ) in Megabarns)
            # Shape: (n_logT, n_nu) = (31, 60185)
            cross_sections['species'][species_name] = cs_group[species_name][:]

    # Cache for future use
    _metal_bf_cross_sections_cache = cross_sections

    return cross_sections


def get_metal_bf_cross_sections():
    """
    Get metal bf cross-sections, loading them if necessary.

    This is a convenience function that handles lazy loading.

    Returns
    -------
    dict
        Dictionary containing metal bf cross-section data
    """
    return load_metal_bf_cross_sections()


def load_barklem_collet_molecular_partition_functions(filename=None):
    """
    Load molecular partition functions from Barklem & Collet 2016.

    Parameters
    ----------
    filename : str, optional
        Path to molecular partition function file. If None, uses default path.

    Returns
    -------
    dict
        Dictionary mapping Species to CubicSpline interpolators over log(T).

    Notes
    -----
    Data from Barklem & Collet 2016 for diatomic molecules.
    Temperature range is typically 1000-10000 K.

    Reference
    ---------
    Barklem & Collet 2016: https://ui.adsabs.harvard.edu/abs/2016A%26A...588A..96B
    """
    if filename is None:
        filename = os.path.join(_DATA_DIR, "barklem_collet_2016",
                               "BarklemCollet2016-molecular_partition.dat")

    temperatures = []
    partition_funcs = {}

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            # Temperature header line
            if len(line) >= 9 and "T [K]" in line:
                temp_str = line[9:].strip()
                temperatures.extend([float(t) for t in temp_str.split()])
            # Skip comments
            elif line.startswith('#'):
                continue
            # Data line
            elif line:
                tokens = line.split()
                species_code = tokens[0]

                # Skip deuterium molecules
                if species_code.startswith('D_'):
                    continue

                # Parse partition function values
                try:
                    species = Species(species_code)
                    U_values = np.array([float(t) for t in tokens[1:]])

                    # Create cubic spline over log(T)
                    log_temps = np.log(temperatures[:len(U_values)])
                    partition_funcs[species] = cubic_spline(log_temps, U_values,
                                                           extrapolate=True)
                except Exception:
                    # Skip species that can't be parsed
                    continue

    return partition_funcs


def load_barklem_collet_equilibrium_constants(filename=None):
    """
    Load equilibrium constants from Barklem & Collet 2016 HDF5 file.

    Parameters
    ----------
    filename : str, optional
        Path to equilibrium constants HDF5 file. If None, uses default path.

    Returns
    -------
    dict
        Dictionary mapping Species to CubicSpline interpolators from ln(T)
        to log₁₀(K) in partial pressure form.

    Notes
    -----
    Equilibrium constants are in partial pressure form:
        K_p = p(A) × p(B) / p(AB)

    For C2, we apply the correction from Visser+ 2019 as recommended by
    Aquilina+ 2024.

    References
    ----------
    Barklem & Collet 2016: https://ui.adsabs.harvard.edu/abs/2016A%26A...588A..96B
    Visser+ 2019: https://doi.org/10.1080/00268976.2018.1564849
    Aquilina+ 2024: https://ui.adsabs.harvard.edu/abs/2024MNRAS.531.4538A
    """
    if filename is None:
        filename = os.path.join(_DATA_DIR, "barklem_collet_2016",
                               "barklem_collet_ks.h5")

    from .constants import kboltz_eV

    equilibrium_constants = {}

    with h5py.File(filename, 'r') as f:
        mols = [Species(s.decode('utf-8') if isinstance(s, bytes) else s)
                for s in f['mols'][:]]
        # HDF5 file has shape (n_temps, n_molecules), so transpose to (n_molecules, n_temps)
        lnTs = f['lnTs'][:].T
        logKs = f['logKs'][:].T

        # Apply C2 correction (Visser+ 2019, recommended by Aquilina+ 2024)
        C2_species = Species("C2")
        if C2_species in mols:
            C2_ind = mols.index(C2_species)
            BC_C2_E0 = 6.371  # Barklem & Collet value, eV
            Visser_C2_E0 = 6.24  # Visser+ 2019 value, eV

            # Correction to log₁₀(K)
            temps = np.exp(lnTs[C2_ind, :])
            correction = (np.log10(np.e) / (kboltz_eV * temps) *
                         (Visser_C2_E0 - BC_C2_E0))
            logKs[C2_ind, :] += correction

        # Create interpolators for each molecule
        for mol, lnT_row, logK_row in zip(mols, lnTs, logKs):
            # Filter out non-finite values
            mask = np.isfinite(lnT_row) & np.isfinite(logK_row)
            if np.sum(mask) > 1:  # Need at least 2 points
                equilibrium_constants[mol] = cubic_spline(lnT_row[mask],
                                                         logK_row[mask],
                                                         extrapolate=True)

    return equilibrium_constants


def load_exomol_partition_functions(filename=None):
    """
    Load ExoMol partition functions for polyatomic molecules.

    Parameters
    ----------
    filename : str, optional
        Path to ExoMol HDF5 file. If None, uses default path.

    Returns
    -------
    dict
        Dictionary mapping Species to CubicSpline interpolators over log(T).

    Notes
    -----
    Partition functions are converted from the "physics" convention (which
    includes nuclear spin degeneracy) to the "astrophysics" convention (which
    does not) by dividing by the total nuclear spin degeneracy.

    Only the most abundant isotopologue is included for each molecule.

    Reference
    ---------
    ExoMol database: https://www.exomol.com/
    """
    if filename is None:
        filename = os.path.join(_DATA_DIR, "polyatomic_partition_funcs",
                               "polyatomic_partition_funcs.h5")

    from .isotopic_data import isotopic_abundances, isotopic_nuclear_spin_degeneracies

    partition_funcs = {}

    with h5py.File(filename, 'r') as f:
        for group_name in f.keys():
            try:
                species = Species(group_name)
                group = f[group_name]

                # Calculate total nuclear spin degeneracy
                total_g_ns = 1
                for Z in species.get_atoms():
                    # Get most abundant isotope
                    most_abundant_A = max(isotopic_abundances[Z].keys(),
                                         key=lambda A: isotopic_abundances[Z][A])
                    g_ns = isotopic_nuclear_spin_degeneracies[Z][most_abundant_A]
                    total_g_ns *= g_ns

                # Load data
                Ts = group['temp'][:]
                Us = group['partition_function'][:]

                # Convert from physics to astrophysics convention
                Us_astro = Us / total_g_ns

                # Create lazy interpolator over log(T)
                # The spline is only built when first called
                partition_funcs[species] = LazyPartitionFunction(np.log(Ts), Us_astro)
            except Exception:
                # Skip species that can't be loaded
                continue

    return partition_funcs


def calculate_polyatomic_equilibrium_constants(partition_funcs):
    """
    Calculate equilibrium constants for polyatomic molecules.

    Uses atomization energies from NIST CCCDB to compute equilibrium constants
    for polyatomic molecules not included in Barklem & Collet 2016.

    Parameters
    ----------
    partition_funcs : dict
        Dictionary of all partition functions (atoms and molecules)

    Returns
    -------
    dict
        Dictionary mapping Species to equilibrium constant functions

    Notes
    -----
    The equilibrium constant in number density form is:

        log₁₀(K_n) = log₁₀(Π U_atoms / U_molecule)
                     + 1.5 × log₁₀(Π m_atoms / m_molecule)
                     + (n_atoms - 1) × log₁₀((2πkT/h²)^1.5)
                     - D₀₀ / (kT × ln(10))

    where D₀₀ is the atomization energy at 0K.

    To convert to partial pressure form:
        log₁₀(K_p) = log₁₀(K_n) + (n_atoms - 1) × log₁₀(kT)

    Reference
    ---------
    NIST CCCBDB: https://cccbdb.nist.gov/
    Tsuji 1973, A&A 23, 411
    """
    import csv
    from .constants import kboltz_eV, kboltz_cgs, hplanck_cgs
    from .atomic_data import atomic_masses
    from .species import Formula

    atomization_file = os.path.join(_DATA_DIR, "polyatomic_partition_funcs",
                                    "atomization_energies.csv")

    equilibrium_constants = {}

    # Read atomization energies
    with open(atomization_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                species = Species(row['spec'])
                D00_kJ_mol = float(row['energy'])
                D00_eV = D00_kJ_mol * 0.01036  # Convert kJ/mol to eV

                # Get constituent atoms
                Zs = species.get_atoms()
                n_atoms = len(Zs)

                # Create equilibrium constant function
                def make_logK_func(spec, D00, Zs, n_atoms, pfuncs):
                    def logK(logT):
                        T = np.exp(logT)

                        # Partition function ratio: Π U_atoms / U_molecule
                        log_Us_ratio = np.sum([np.log10(pfuncs[Species(Formula(int(Z)), 0)](logT))
                                               for Z in Zs])
                        log_Us_ratio -= np.log10(pfuncs[spec](logT))

                        # Mass ratio: Π m_atoms / m_molecule
                        log_masses_ratio = (np.sum([np.log10(atomic_masses[int(Z)])
                                                    for Z in Zs])
                                           - np.log10(spec.get_mass()))

                        # Translational factor
                        log_translational_U_factor = 1.5 * np.log10(
                            2 * np.pi * kboltz_cgs * T / (hplanck_cgs**2))

                        # log₁₀(K_n) in number density form
                        log_nK = ((n_atoms - 1) * log_translational_U_factor
                                 + 1.5 * log_masses_ratio
                                 + log_Us_ratio
                                 - D00 / (kboltz_eV * T * np.log(10)))

                        # Convert to partial pressure form
                        log_pK = log_nK + (n_atoms - 1) * np.log10(kboltz_cgs * T)

                        return log_pK

                    return logK

                equilibrium_constants[species] = make_logK_func(species, D00_eV,
                                                                Zs, n_atoms,
                                                                partition_funcs)
            except Exception:
                # Skip molecules that can't be processed
                continue

    return equilibrium_constants


def setup_partition_funcs_and_equilibrium_constants():
    """
    Load and combine all partition functions and equilibrium constants.

    Returns
    -------
    tuple
        (partition_funcs, equilibrium_constants) where:
        - partition_funcs: Dict mapping Species to interpolator functions
        - equilibrium_constants: Dict mapping Species to log₁₀(K) functions

    Notes
    -----
    Partition functions come from:
    - Atoms: Custom calculations from NIST energy levels
    - Diatomic molecules: Barklem & Collet 2016
    - Polyatomic molecules: ExoMol

    Equilibrium constants come from:
    - Diatomic molecules: Barklem & Collet 2016 (with C2 correction)
    - Polyatomic molecules: Calculated from NIST atomization energies

    All equilibrium constants are in partial pressure form.
    """
    # Load partition functions from all sources
    atomic_pfuncs = load_atomic_partition_functions()
    molecular_pfuncs = load_barklem_collet_molecular_partition_functions()
    polyatomic_pfuncs = load_exomol_partition_functions()

    # Merge all partition functions
    partition_funcs = {}
    partition_funcs.update(atomic_pfuncs)
    partition_funcs.update(molecular_pfuncs)
    partition_funcs.update(polyatomic_pfuncs)

    # Load equilibrium constants
    BC_Ks = load_barklem_collet_equilibrium_constants()
    polyatomic_Ks = calculate_polyatomic_equilibrium_constants(partition_funcs)

    # Merge equilibrium constants
    equilibrium_constants = {}
    equilibrium_constants.update(BC_Ks)
    equilibrium_constants.update(polyatomic_Ks)

    return partition_funcs, equilibrium_constants


# Load default partition functions and equilibrium constants at module import
default_partition_funcs, default_log_equilibrium_constants = setup_partition_funcs_and_equilibrium_constants()
