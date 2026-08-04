"""
Linelist pruning utilities.

Provides merge_close_lines and prune_linelist to reduce linelists for plotting
or to speed up synthesis by removing weak lines.
"""

import numpy as np
from typing import List, Tuple


def _photosphere_opacities(atmosphere, linelist, wls_angstrom, A_X, *,
                           vmic: float = 1.0,
                           hydrogen_lines: bool = True,
                           hydrogen_line_window_size: float = 150.0,
                           line_buffer: float = 10.0,
                           cntm_step: float = 1.0,
                           line_cutoff_threshold: float = 3e-4):
    """Per-layer opacities and number densities. No radiative transfer.

    ``prune_linelist`` used to obtain these by calling ``synthesize_spectrum``
    twice and discarding both spectra: it wants the opacity that sets the
    optical-depth scale and the species densities at the photosphere, and never
    looked at a flux. ``synthesize_spectrum`` is gone, and the traced synthesis
    that replaced it returns ``(flux, continuum)`` and nothing else — a per-layer
    opacity array is not something a jitted, vmapped pass can hand back.

    So the opacity stack is assembled here from the same public pieces the
    deleted function used, in the same order and with the same defaults, which
    is why the pruned linelists still match Korg.jl's. One chemical-equilibrium
    solve now serves both arrays instead of two.

    The Brackett series is omitted. ``korg.hydrogen_line_absorption`` returns
    exactly zero for every Brackett wavelength (pinned by
    ``test_the_infrared_brackett_branch_runs_but_contributes_nothing``), so the
    per-layer loop that used to run for it added nothing but time.

    Returns
    -------
    alpha_total : (n_layers, n_wl) continuum + hydrogen + atomic/molecular lines
    alpha_cntm : (n_layers, n_wl) continuum + hydrogen lines, no linelist —
        what a synthesis with an empty linelist would have reported as ``alpha``
    number_densities : dict, Species -> (n_layers,) cm^-3
    """
    from scipy.interpolate import interp1d
    import jax.numpy as jnp

    from .abundances import A_X_to_absolute
    from .constants import c_cgs
    from .continuum import prepare_continuum_batch_fast, batch_continuum_absorption
    from .data_loader import (default_chem_eq_data, default_mol_species,
                              default_partition_funcs)
    from .hydrogen_line_absorption import (hline_stark_profiles,
                                           hydrogen_line_absorption_stark_batched,
                                           precompute_hummer_ws)
    from .line_absorption import line_absorption
    from .species import Species
    from .statmech import chemical_equilibrium_all_layers

    wls_cm = np.asarray(wls_angstrom, dtype=np.float64) * 1e-8
    vmic_cm_s = vmic * 1e5
    line_buffer_cm = line_buffer * 1e-8
    cntm_step_cm = cntm_step * 1e-8
    h_window_cm = hydrogen_line_window_size * 1e-8

    T = np.asarray(atmosphere.T, dtype=np.float64)
    n_total = np.asarray(atmosphere.n_total, dtype=np.float64)
    ne_model = np.asarray(atmosphere.ne, dtype=np.float64)
    n_layers = len(T)

    A_X = np.asarray(A_X, dtype=np.float64)
    abs_abundances = A_X_to_absolute(A_X) if A_X[0] > 1.0 else A_X

    ne, number_densities, raw_arrays = chemical_equilibrium_all_layers(
        T, n_total, ne_model, abs_abundances, default_chem_eq_data,
        default_mol_species)
    ne = np.asarray(ne)

    # Continuum on the coarse grid, then linearly interpolated onto the
    # synthesis grid -- the sampling the deleted path used.
    wl_lo = wls_cm[0] - line_buffer_cm - cntm_step_cm
    wl_hi = wls_cm[-1] + line_buffer_cm + cntm_step_cm
    cntm_wl = np.arange(wl_lo, wl_hi + cntm_step_cm, cntm_step_cm)
    batch = prepare_continuum_batch_fast(raw_arrays, default_partition_funcs, T)
    alpha_cntm_coarse = np.asarray(batch_continuum_absorption(
        jnp.asarray(c_cgs / cntm_wl), jnp.asarray(T), jnp.asarray(ne), batch))
    interps = [interp1d(cntm_wl, alpha_cntm_coarse[i], kind="linear",
                        fill_value="extrapolate") for i in range(n_layers)]
    alpha_cntm = np.stack([interps[i](wls_cm) for i in range(n_layers)])

    if hydrogen_lines:
        _RYDBERG_CM = 1.0973731568539e5
        nearby_stark = {
            k: v for k, v in hline_stark_profiles.items()
            if (wls_cm[0] - h_window_cm
                <= 1.0 / (_RYDBERG_CM * (1.0 / v.lower**2 - 1.0 / v.upper**2))
                <= wls_cm[-1] + h_window_cm)
        }
        if nearby_stark:
            nH_I = np.array([ra["neutral_dens"][0] for ra in raw_arrays])
            nHe_I = np.array([ra["neutral_dens"][1] for ra in raw_arrays])
            ws_all = precompute_hummer_ws(T, nH_I, nHe_I, ne)
            pf_H_I = default_partition_funcs[Species("H_I")]
            log_T = np.log(T)
            U_H_I = (pf_H_I.numpy_eval(log_T) if hasattr(pf_H_I, "numpy_eval")
                     else np.array([float(pf_H_I(lt)) for lt in log_T]))
            alpha_cntm = alpha_cntm + np.asarray(
                hydrogen_line_absorption_stark_batched(
                    wls_cm, T, ne, nH_I, U_H_I, h_window_cm, vmic_cm_s,
                    ws_all, nearby_stark))

    if linelist:
        def continuum_opacity(wl_cm):
            wl_arr = np.atleast_1d(wl_cm)
            out = np.stack([interps[i](wl_arr) for i in range(n_layers)], axis=-1)
            return out if np.ndim(wl_cm) > 0 else out[0]

        alpha_total = alpha_cntm + np.asarray(line_absorption(
            linelist, wls_cm, T, ne, number_densities, default_partition_funcs,
            vmic_cm_s, continuum_opacity,
            cutoff_threshold=line_cutoff_threshold))
    else:
        alpha_total = alpha_cntm

    return alpha_total, alpha_cntm, number_densities


def merge_close_lines(linelist: list, merge_distance: float = 0.2) -> list:
    """
    Merge lines of the same species that are within merge_distance Å of each other.

    Useful for labeling lines in a plot after running prune_linelist.

    Args:
        linelist: List of Line objects
        merge_distance: Maximum Å between lines of the same species to merge.
            Default 0.2 Å.

    Returns:
        List of tuples (wl, wl_low, wl_high, species_str) where:
        - wl: gf-weighted mean wavelength in Å
        - wl_low: minimum wavelength of merged group in Å
        - wl_high: maximum wavelength of merged group in Å
        - species_str: string identifying the species

        Sorted by wavelength.
    """
    lines_sorted = sorted(linelist, key=lambda l: l.wl)

    # Group lines by species, merging those within merge_distance
    # Dict: species -> list of current-group lines
    current_groups = {}

    # Completed groups
    completed = []

    for line in lines_sorted:
        spec = line.species
        if spec in current_groups:
            group = current_groups[spec]
            last_wl_angstrom = group[-1].wl * 1e8
            this_wl_angstrom = line.wl * 1e8
            if this_wl_angstrom - last_wl_angstrom < merge_distance:
                group.append(line)
            else:
                completed.append(group)
                current_groups[spec] = [line]
        else:
            current_groups[spec] = [line]

    # Flush remaining groups
    for group in current_groups.values():
        completed.append(group)

    # Build result tuples
    result = []
    for group in completed:
        wls_ang = [l.wl * 1e8 for l in group]
        gf_vals = [10.0 ** l.log_gf for l in group]
        total_gf = sum(gf_vals)
        mean_wl = sum(w * g for w, g in zip(wls_ang, gf_vals)) / total_gf
        result.append((mean_wl, min(wls_ang), max(wls_ang), str(group[0].species)))

    return sorted(result, key=lambda t: t[0])


def prune_linelist(
    atmosphere,
    linelist: list,
    A_X,
    wavelengths,
    threshold: float = 0.1,
    sort_by_EW: bool = True,
    max_distance: float = 0.0,
    **synthesis_kwargs
) -> list:
    """
    Return lines strong enough to be detected, optionally sorted by strength.

    Synthesizes the spectrum and continuum, then keeps lines whose estimated
    line-center absorption exceeds threshold * continuum absorption at the
    photosphere.

    Args:
        atmosphere: Atmosphere model (PlanarAtmosphere or SphericalAtmosphere)
        linelist: List of Line objects
        A_X: 92-element abundance array (from format_A_X())
        wavelengths: Wavelength range in Å — tuple (start, end) or 1D array
        threshold: Minimum ratio of line-center to continuum absorption.
            Default 0.1. Use 1e-4 for synthesis pruning.
        sort_by_EW: If True, sort by approximate equivalent width (slower).
            If False, return in wavelength order.
        max_distance: How far from wavelengths lines can be (Å) before exclusion.
        **synthesis_kwargs: Additional kwargs passed to the opacity calculation
            and, when sort_by_EW is True, to synthesize(): vmic,
            hydrogen_lines, line_buffer, cntm_step, line_cutoff_threshold.

    Returns:
        List of Line objects
    """
    from .constants import kboltz_eV, c_cgs, hplanck_eV
    from .line_absorption import doppler_width, sigma_line

    # Determine wavelength grid
    if isinstance(wavelengths, (tuple, list)) and len(wavelengths) == 2:
        wl_start, wl_stop = float(wavelengths[0]), float(wavelengths[1])
        wls_angstrom = np.linspace(wl_start, wl_stop, max(1000, int((wl_stop - wl_start) / 0.01)))
    else:
        wls_angstrom = np.asarray(wavelengths, dtype=float)
        wl_start, wl_stop = wls_angstrom[0], wls_angstrom[-1]

    # Filter linelist to wavelength window (+/- max_distance)
    window_lines = [
        l for l in linelist
        if (wl_start - max_distance) * 1e-8 <= l.wl <= (wl_stop + max_distance) * 1e-8
    ]

    # ``verbose`` was a synthesize_spectrum option; the traced path has no
    # progress printing to switch off. Accepted and dropped so existing callers
    # keep working rather than hitting an unexpected-keyword TypeError.
    synthesis_kwargs.pop("verbose", None)

    # Opacities and number densities. Pruning compares a line-centre opacity
    # against the continuum at the photosphere; no flux is needed here.
    alpha_arr, cntm_alpha, number_densities = _photosphere_opacities(
        atmosphere, window_lines, wls_angstrom, A_X, **synthesis_kwargs)

    # Get atmosphere properties
    zs = np.array(atmosphere.z)       # depths (cm), top to bottom
    temps = np.array(atmosphere.T)     # temperatures (K)

    # Compute cumulative optical depth to find photosphere (tau~1)
    # alpha_arr shape: (n_layers, n_wavelengths)
    dz = np.abs(np.diff(zs))         # layer thicknesses
    # Cumulative tau from top downward
    cum_tau = np.cumsum(alpha_arr[:-1] * dz[:, np.newaxis], axis=0)  # (n_layers-1, n_wl)

    # Photosphere index (first layer where tau > 1) for each wavelength
    n_layers_minus1, n_wl = cum_tau.shape
    photosphere_indices = np.argmax(cum_tau > 1.0, axis=0)
    # Where tau never exceeds 1, use deepest layer
    never_tau1 = np.all(cum_tau <= 1.0, axis=0)
    photosphere_indices[never_tau1] = n_layers_minus1 - 1

    # Precompute per-species Doppler widths and number densities / partition functions
    from .data_loader import default_partition_funcs
    from .species import Species

    unique_species = list(set(l.species for l in window_lines))
    n_div_Z = {}
    doppler_widths_by_spec = {}

    for spec in unique_species:
        spec_key = str(spec)
        n_densities = number_densities.get(spec, number_densities.get(spec_key))
        if n_densities is None:
            n_div_Z[spec] = np.zeros(len(temps))
        else:
            pf_func = (default_partition_funcs.get(spec) or
                       default_partition_funcs.get(spec_key) or
                       (lambda x: 1.0))
            U = np.array([float(pf_func(float(np.log(T)))) for T in temps])
            n_div_Z[spec] = np.array(n_densities) / np.where(U > 0, U, 1.0)

        # Doppler width at line reference wavelength
        mass = spec.formula.get_mass() if hasattr(spec.formula, 'get_mass') else 1.67e-24
        doppler_widths_by_spec[spec] = np.array([
            doppler_width(wl_start * 1e-8, T, mass, 0.0) * 1e8 for T in temps
        ])

    # Note: sigma_line is the single implementation in korg.line_absorption
    # (imported above), matching Korg.jl's sigma_line.

    strong_lines = []
    for line in window_lines:
        # Find closest wavelength in grid
        wl_center_ang = line.wl * 1e8
        idx_wl = np.argmin(np.abs(wls_angstrom - wl_center_ang))
        phot_idx = photosphere_indices[idx_wl]

        T = temps[phot_idx]
        beta = 1.0 / (kboltz_eV * T)

        E_upper = line.E_lower + hplanck_eV * c_cgs / line.wl
        levels_factor = np.exp(-beta * line.E_lower) - np.exp(-beta * E_upper)

        sigma = doppler_widths_by_spec[line.species][phot_idx]
        n_Z = n_div_Z[line.species][phot_idx]

        # Line center opacity (Å units: 1e8 converts cm->Å for Doppler width)
        alpha_line_center = (1e8 * 10.0 ** line.log_gf *
                             sigma_line(line.wl) * levels_factor * n_Z / max(sigma, 1e-30))

        alpha_cntm_phot = cntm_alpha[phot_idx, idx_wl]

        if alpha_cntm_phot > 0 and alpha_line_center > threshold * alpha_cntm_phot:
            strong_lines.append(line)

    if not sort_by_EW:
        return sorted(strong_lines, key=lambda l: l.wl)

    # Sort by approximate reduced EW
    from .synthesis_plan import synthesize

    # ``synthesize`` takes the plan kwargs, not the opacity ones. Only the
    # arguments it shares with _photosphere_opacities are forwarded.
    _synth_kw = {k: v for k, v in synthesis_kwargs.items()
                 if k in ("vmic", "hydrogen_lines", "line_buffer")}
    approx_EWs = []
    for line in strong_lines:
        wl_center = line.wl * 1e8
        # Tiny window: ±0.03% (~90 km/s)
        mini_wls = np.linspace(wl_center * 0.9997, wl_center * 1.0003, 100)
        flux, cntm = synthesize(atmosphere, [line], mini_wls, A_X, **_synth_kw)
        flux = np.asarray(flux)
        cntm = np.asarray(cntm)
        safe_cntm = np.where(cntm > 0, cntm, 1.0)
        EW = np.sum(1.0 - flux / safe_cntm) / wl_center
        approx_EWs.append(EW)

    order = np.argsort(approx_EWs)[::-1]
    return [strong_lines[i] for i in order]
