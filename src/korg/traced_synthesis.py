"""
The traced half: atmosphere interpolation and synthesis, with no Python control
flow over traced data.

Everything here is reachable from :meth:`korg.synthesis_plan.Synthesizer.__call__`
and is a pure function of its arguments plus the closure's host constants.  It
composes the kernels that were already traceable — the implicit-differentiation
chemical equilibrium solve, the batched continuum, the Voigt bucket scatter, the
Stehle Stark hydrogen path, radiative transfer — and supplies the shapes those
kernels need from the plan rather than from the values.
"""

import jax
import jax.numpy as jnp
import numpy as np

from .constants import c_cgs
from .traced_lines import line_alpha_traced, hydrogen_alpha_traced

LAMBDA_REF_CM = 5e-5
G_CGS = 6.67430e-8
M_SUN_CGS = 1.9885e33
SPHERICAL_LOGG_THRESHOLD = 3.5   # Korg.jl marcs_interpolation.py:524


def photosphere_radius(logg):
    """R = sqrt(G M_sun / g), Korg.jl's convention.

    A smooth function of a traced scalar, so it belongs inside the traced region.
    It used to be frozen into the plan at the *reference* log g, which meant a
    plan built for the Sun computed giant ray geometry with the solar radius --
    7e10 cm instead of 3e12 cm, a factor of 40 -- and d/dlogg missed the radius
    dependence entirely. Nothing about R determines a shape, so nothing required
    it to be static.
    """
    return jnp.sqrt(G_CGS * M_SUN_CGS / 10.0 ** jnp.asarray(logg, dtype=jnp.float64))


def _interpolate_marcs_traced(synth, Teff, logg, M_H, alpha_M, C_M):
    """MARCS interpolation with a fixed layer count, differentiable throughout.

    ``interpolate_marcs`` breaks the chain three times after the kernel: it calls
    ``np.asarray`` on the result, strips NaN layers with a boolean mask (making
    the layer count depend on values), and rebuilds the rows as dataclasses of
    Python floats.  Only the first and third are incidental; the second is a real
    data-dependent shape.

    It is resolved the same way as every other shape here: the count is fixed at
    ``synth.n_layers`` and taken from the plan.  That is safe because the MARCS
    grid yields 56 valid layers at every point checked, including its corners —
    but it is an assumption of the plan, not a property of the kernel, which is
    why it lives here and not inside the interpolation.

    Returns
    -------
    T, n_total, ne, z, log_tau_ref : (n_layers,) arrays
    """
    from .marcs_interpolation import _interpolate_marcs_jit

    nodes_padded, nodes_lengths, grid = synth._marcs
    params = jnp.stack([jnp.asarray(Teff, dtype=jnp.float64),
                        jnp.asarray(logg, dtype=jnp.float64),
                        jnp.asarray(M_H, dtype=jnp.float64),
                        jnp.asarray(alpha_M, dtype=jnp.float64),
                        jnp.asarray(C_M, dtype=jnp.float64)])
    q = _interpolate_marcs_jit(params, nodes_padded, nodes_lengths, grid)

    # Columns, in grid order: T, log(ne), log(n_total), tau_ref, asinh(z).
    T = q[:, 0]
    ne = jnp.exp(q[:, 1])
    n_total = jnp.exp(q[:, 2])
    tau_ref = q[:, 3]
    z = jnp.sinh(q[:, 4])
    # log10 of the anchored optical depth. The base matters: the transfer
    # kernels take log_tau_ref as base-10 (`tau_ref = 10 ** log_tau_ref`, and
    # step sizes scaled by ln 10), matching `Atmosphere.log_tau_ref`. Passing a
    # natural log here integrates against tau_ref ** ln(10) and makes lines
    # about twice too shallow.
    # tau_ref is positive throughout a valid model; the `where` keeps the
    # derivative finite if an interpolation strays, rather than merely masking
    # the value.
    positive = tau_ref > 0.0
    log_tau_ref = jnp.where(positive,
                            jnp.log10(jnp.where(positive, tau_ref, 1.0)),
                            -jnp.inf)
    return T, n_total, ne, z, log_tau_ref, photosphere_radius(logg)


def _synthesize_traced(synth, T_layers, n_total_layers, ne_layers, z_layers,
                       log_tau_ref, abundances, vmic_cm_s,
                       R_photosphere=None, logg=None):
    """One traced pass: chemistry, continuum, lines, hydrogen, transfer.

    Returns ``(flux, continuum)`` in erg cm^-2 s^-1 A^-1.
    """
    from .synthesis import (
        _picard_chemical_equilibrium_guess_batch, _chem_eq_newton_batch_jit,
        _compute_saha_weights_batch_jit, _compute_mol_densities_batch_jit,
        _compute_line_params_jit, _pf_orig_eval, _batch_continuum_vmap,
        _PEACH_IDX, _H2_MOL_IDX, blackbody,
    )
    from .continuum import _get_metal_bf_idx
    from .hydrogen_line_absorption import precompute_hummer_ws

    data = synth.data
    linelist_data = synth.linelist_data
    n_layers, n_wl = synth.n_layers, synth.n_wl
    wl = synth.wavelengths_cm

    # -- chemical equilibrium -------------------------------------------------
    ne_init, nf_init = _picard_chemical_equilibrium_guess_batch(
        T_layers, n_total_layers, ne_layers, abundances, data.chem_eq_data)
    ne_all, nf_sol = _chem_eq_newton_batch_jit(
        T_layers, n_total_layers, ne_init, nf_init, abundances, data.chem_eq_data)

    atom_dens = abundances[None, :] * (n_total_layers - ne_all)[:, None]
    neutral_dens = atom_dens * nf_sol
    wII, wIII = _compute_saha_weights_batch_jit(T_layers, ne_all, data.chem_eq_data)
    ionized_dens = wII * neutral_dens
    doubly_dens = wIII * neutral_dens

    mol_dens = _compute_mol_densities_batch_jit(
        T_layers, n_total_layers, ne_all, abundances, nf_sol, data.chem_eq_data)
    mol_densities_all = jnp.concatenate([mol_dens, jnp.zeros((n_layers, 1))], axis=1)

    nH_I_all = neutral_dens[:, 0]
    nH_II_all = ionized_dens[:, 0]
    nHe_I_all = neutral_dens[:, 1]
    nH2_all = mol_dens[:, _H2_MOL_IDX]
    n_eff_vdW_all = nH_I_all

    log_T_all = jnp.log(T_layers)
    ced = data.chem_eq_data
    U_H_I_all = jax.vmap(lambda lt: _pf_orig_eval(
        lt, ced.pf_orig_t[0, 0], ced.pf_orig_u[0, 0], ced.pf_orig_h[0, 0],
        ced.pf_orig_z[0, 0], ced.pf_orig_n[0, 0]))(log_T_all)
    U_He_I_all = jax.vmap(lambda lt: _pf_orig_eval(
        lt, ced.pf_orig_t[1, 0], ced.pf_orig_u[1, 0], ced.pf_orig_h[1, 0],
        ced.pf_orig_z[1, 0], ced.pf_orig_n[1, 0]))(log_T_all)

    # -- continuum ------------------------------------------------------------
    peach_z = jnp.array([z for z, _ in _PEACH_IDX])
    n_peach = ionized_dens[:, peach_z]
    n_Z1_ff = jnp.sum(ionized_dens, axis=1) - jnp.sum(n_peach, axis=1)
    n_Z2_ff = jnp.sum(doubly_dens, axis=1)

    _, metal_bf_idx = _get_metal_bf_idx()
    mb_z = jnp.array([zi for zi, _ in metal_bf_idx])
    mb_c = jnp.array([ci for _, ci in metal_bf_idx])
    metal_bf_dens = jnp.where(
        mb_c[None, :] == 0, neutral_dens[:, mb_z],
        jnp.where(mb_c[None, :] == 1, ionized_dens[:, mb_z], doubly_dens[:, mb_z]))

    cntm_nu = c_cgs / synth.cntm_wl_cm
    alpha_cntm_coarse = _batch_continuum_vmap(
        cntm_nu, T_layers, ne_all, U_H_I_all, U_He_I_all,
        nH_I_all, nH_II_all, nHe_I_all, nH2_all,
        n_peach, n_Z1_ff, n_Z2_ff, metal_bf_dens,
        data.metal_bf_tables, data.metal_bf_nu_grid, data.metal_bf_logT_grid)

    alpha_cntm_all = jax.vmap(
        lambda row: jnp.interp(wl, synth.cntm_wl_cm, row))(alpha_cntm_coarse)

    # Evaluated *at* lambda_ref, never extrapolated from the window's coarse grid.
    alpha_ref_all = _batch_continuum_vmap(
        jnp.array([c_cgs / LAMBDA_REF_CM]), T_layers, ne_all,
        U_H_I_all, U_He_I_all, nH_I_all, nH_II_all, nHe_I_all, nH2_all,
        n_peach, n_Z1_ff, n_Z2_ff, metal_bf_dens,
        data.metal_bf_tables, data.metal_bf_nu_grid, data.metal_bf_logT_grid)[:, 0]

    S_all = jax.vmap(lambda Ti: blackbody(Ti, wl))(T_layers)

    # -- lines ----------------------------------------------------------------
    if synth.n_lines and synth.bucket_line_idx:
        amp, sigma_D, gamma_L = _compute_line_params_jit(
            T_layers, ne_all, n_eff_vdW_all, neutral_dens, ionized_dens,
            mol_densities_all, linelist_data, data, vmic_cm_s)
        cntm_at_center = jax.vmap(
            lambda row: jnp.interp(linelist_data.wl, synth.cntm_wl_cm, row),
            out_axes=1)(alpha_cntm_coarse)          # (n_lines, n_layers)
        line_alpha, max_needed_px = line_alpha_traced(
            amp, sigma_D, gamma_L, linelist_data.wl, wl, cntm_at_center, synth)
    else:
        line_alpha = jnp.zeros((n_layers, n_wl))
        max_needed_px = jnp.zeros((0,), dtype=jnp.int32)

    # -- hydrogen lines -------------------------------------------------------
    ws_all = precompute_hummer_ws(T_layers, nH_I_all, nHe_I_all, ne_all)
    h_alpha = hydrogen_alpha_traced(wl, T_layers, ne_all, nH_I_all, nHe_I_all,
                                    U_H_I_all, ws_all, vmic_cm_s, synth)

    alpha_total = alpha_cntm_all + line_alpha + h_alpha

    # Korg anchors on continuum + atomic lines at 5000 A, never H lines.
    if synth.ref_pixel is not None:
        alpha_ref_all = (alpha_cntm_all + line_alpha)[:, synth.ref_pixel]

    # -- radiative transfer ---------------------------------------------------
    def _planar():
        from .radiative_transfer import radiative_transfer_jit as rt
        f, _ = rt(alpha_total.T, S_all.T, z_layers, log_tau_ref, alpha_ref_all)
        fc, _ = rt(alpha_cntm_all.T, S_all.T, z_layers, log_tau_ref, alpha_ref_all)
        return f, fc

    def _spherical():
        from .radiative_transfer.spherical import spherical_ray_flux
        from .radiative_transfer import generate_mu_grid
        mu, mu_w = generate_mu_grid(synth.n_mu)
        R = R_photosphere if R_photosphere is not None else photosphere_radius(logg)
        radii = R + z_layers
        out = spherical_ray_flux(alpha_total.T, S_all.T, radii, log_tau_ref,
                                 alpha_ref_all, mu, mu_w)
        out_c = spherical_ray_flux(alpha_cntm_all.T, S_all.T, radii, log_tau_ref,
                                   alpha_ref_all, mu, mu_w)
        f = out[0] if isinstance(out, tuple) else out
        fc = out_c[0] if isinstance(out_c, tuple) else out_c
        # Korg quotes the flux at the photospheric radius, not the outermost one.
        corr = (radii[0] / R) ** 2
        return f * corr, fc * corr

    if synth.geometry == "plane-parallel":
        flux, flux_cntm = _planar()
    elif synth.geometry == "spherical":
        flux, flux_cntm = _spherical()
    else:
        # geometry=None: choose the way Korg.jl does, from log g, inside the
        # traced region. lax.cond keeps this one program: under plain jit only
        # the selected branch runs; under vmap JAX converts it to a select and
        # evaluates both, which is a cost, not a correctness problem.
        #
        # The derivative is the *selected* branch's. The switch contributes
        # nothing, and cannot: log g = 3.5 is a discontinuous change of model,
        # not a smooth transition, so d(flux)/d(logg) genuinely does not exist
        # there. Differentiating or vmapping across it is the caller's to avoid.
        if logg is None:
            raise ValueError(
                "geometry=None resolves the geometry from log g, so from_atmosphere() "
                "needs logg=... . Pass it, or build the plan with "
                "geometry='spherical' or 'plane-parallel'.")
        flux, flux_cntm = jax.lax.cond(
            jnp.asarray(logg) < SPHERICAL_LOGG_THRESHOLD,
            lambda: _spherical(), lambda: _planar())

    return flux * 1e-8, flux_cntm * 1e-8
