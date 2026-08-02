"""
Line and hydrogen-line opacity with every shape fixed and every value traced.

The NumPy line-absorption path decided three things from traced values: how wide
each line's window is, where that window starts, and which lines share a bucket.
Only the last of those has to be fixed at trace time — it is a partition, and a
partition is a shape.  The other two are recomputed here from the traced
amplitudes and broadening parameters, which is what keeps
``d flux / d abundance`` exact through the window and not merely through the
profile inside a frozen one.

That works because ``jnp.searchsorted`` traces, and because ``.at[idx].add()``
accepts a traced ``idx`` — so a gather-evaluate-scatter over data-dependent
positions is expressible as long as the *count* of positions is not itself
data-dependent.
"""

import jax
import jax.numpy as jnp

from .line_absorption import _voigt_profile_jax

CUTOFF = 3e-4          # Korg's line_cutoff_threshold: fraction of continuum
SQRT_2PI = 2.5066282746310002


def line_window_cm(amp, sigma_D, gamma_L, cntm_at_center):
    """Half-width at which a line drops below ``CUTOFF`` of the continuum.

    Port of Korg's window calculation, kept in JAX so it is differentiable.
    ``amp``, ``sigma_D``, ``gamma_L`` are (n_lines, n_layers); the window is the
    worst case over layers, as in Korg.

    The two ``where`` guards are doubled: ``log`` of a non-positive argument and
    ``sqrt`` of a negative one are both infinite in the derivative, and masking
    the value alone would leave a NaN cotangent flowing back into ``amp``.
    """
    rho_crit = CUTOFF * cntm_at_center / jnp.maximum(jnp.abs(amp), 1e-300)

    log_arg = SQRT_2PI * sigma_D * rho_crit
    gauss_live = log_arg < 1.0
    safe_log_arg = jnp.where(gauss_live, jnp.maximum(log_arg, 1e-300), 1.0)
    radicand_G = -2.0 * jnp.log(safe_log_arg)
    g_live = gauss_live & (radicand_G > 0.0)
    win_G = jnp.where(g_live,
                      sigma_D * jnp.sqrt(jnp.where(g_live, radicand_G, 1.0)),
                      0.0)

    win_L_arg = gamma_L / (jnp.pi * rho_crit)
    lorentz_live = win_L_arg > gamma_L ** 2
    radicand_L = jnp.where(lorentz_live, win_L_arg - gamma_L ** 2, 1.0)
    l_live = lorentz_live & (radicand_L > 0.0)
    win_L = jnp.where(l_live, jnp.sqrt(jnp.where(l_live, radicand_L, 1.0)), 0.0)

    # Korg takes the worst layer for each broadening mechanism separately, then
    # combines in quadrature, with a 2e-5 relative buffer so window edges land on
    # the same pixel as the Julia implementation.
    #
    # The outer sqrt needs the same double-`where` as the two inner ones. A line
    # whose window collapses in *both* mechanisms -- zero amplitude, or damping
    # so weak the Lorentz branch never activates -- reaches sqrt(0), whose
    # derivative is infinite, and the zero cotangent that meets it makes
    # 0 x inf = NaN. It propagates from there to every parameter, which is how a
    # single unremarkable line took out d(flux)/d(Teff) for the whole spectrum.
    tot = jnp.max(win_G, axis=1) ** 2 + jnp.max(win_L, axis=1) ** 2
    live = tot > 0.0
    return jnp.where(live, jnp.sqrt(jnp.where(live, tot, 1.0)), 0.0) * (1.0 + 2e-5)


def line_alpha_traced(amp, sigma_D, gamma_L, line_wl, wl_grid,
                      cntm_at_center, plan):
    """Atomic and molecular line opacity, (n_layers, n_wl).

    Parameters
    ----------
    amp, sigma_D, gamma_L : (n_lines, n_layers) arrays
        Traced line parameters.
    line_wl : (n_lines,) array
        Line centres [cm].
    wl_grid : (n_wl,) array
        Synthesis grid [cm].
    cntm_at_center : (n_lines, n_layers) array
        Continuum opacity at each line centre, for the cutoff.
    plan : SynthesisPlan
        Supplies bucket membership and pixel capacities — the static half.

    Returns
    -------
    alpha : (n_layers, n_wl)
    max_needed_px : (n_buckets,) int
        Widest window actually required in each bucket, in pixels. Compare
        against ``plan.bucket_widths`` outside ``jit``: if it exceeds, the plan
        under-allocated and profiles were truncated early.
    """
    n_wl = plan.n_wl
    n_layers = plan.n_layers
    alpha = jnp.zeros((n_layers, n_wl))
    needed = []

    if not plan.bucket_line_idx:
        return alpha, jnp.zeros((0,), dtype=jnp.int32)

    max_wins_all = line_window_cm(amp, sigma_D, gamma_L, cntm_at_center)

    for idx, W in zip(plan.bucket_line_idx, plan.bucket_widths):
        sel = jnp.asarray(idx)
        a_b = amp[sel]                      # (n_b, n_layers)
        s_b = sigma_D[sel]
        g_b = gamma_L[sel]
        wl_b = line_wl[sel]                 # (n_b,)
        win_b = max_wins_all[sel]           # (n_b,) traced

        # Window start: traced index, static count. searchsorted gives the first
        # pixel at or past the blue edge; clipping keeps the fixed-width slice
        # inside the grid without changing which pixels are unmasked.
        i_lo = jnp.searchsorted(wl_grid, wl_b - win_b)
        i_lo = jnp.clip(i_lo, 0, max(n_wl - W, 0))

        pix = i_lo[:, None] + jnp.arange(W)[None, :]          # (n_b, W)
        delta = wl_grid[pix] - wl_b[:, None]                   # (n_b, W)
        mask = jnp.abs(delta) <= win_b[:, None]

        prof = _voigt_profile_jax(delta[:, None, :],
                                  s_b[:, :, None],
                                  g_b[:, :, None])             # (n_b, n_layers, W)
        contrib = mask[:, None, :] * a_b[:, :, None] * prof

        flat_pix = pix.ravel()
        flat_contrib = contrib.transpose(1, 0, 2).reshape(n_layers, -1)
        alpha = alpha + jax.vmap(
            lambda c: jnp.zeros(n_wl).at[flat_pix].add(c)
        )(flat_contrib)

        # How many pixels this bucket really wanted, for the caller to check.
        needed.append(jnp.ceil(2.0 * jnp.max(win_b) / plan.wl_spacing + 2.0))

    return alpha, jnp.asarray(needed, dtype=jnp.int32)


def hydrogen_alpha_traced(wl_grid, T, ne, nH_I, nHe_I, U_H_I, ws_all, vmic, plan):
    """Stehle Stark + ABO hydrogen line opacity, (n_layers, n_wl), fully traced.

    Which transitions contribute is fixed by ``plan.stark_keys`` — a function of
    the wavelength grid alone. Everything downstream of that (line centres, the
    3-D Stark table interpolation, the validity mask, amplitudes) is computed
    here from traced ``T`` and ``ne``, so Balmer-line gradients with respect to
    effective temperature survive.

    Brackett lines are *not* included: their implementation walks layers in
    Python and would have to be rewritten to trace. ``plan.brackett_in_range``
    records whether any fall in the window so the caller can refuse rather than
    quietly drop them.
    """
    from .hydrogen_line_absorption import (
        hline_stark_profiles, _interp_lambda0_all_layers_jit,
        _process_stehle_line_all_layers_jit, _BALMER_ABO_PARAMS,
    )
    from .constants import bohr_radius_cgs

    n_layers, n_wl = plan.n_layers, plan.n_wl
    alpha = jnp.zeros((n_layers, n_wl))
    if not plan.stark_keys:
        return alpha

    for key in plan.stark_keys:
        line = hline_stark_profiles[key]
        temps = jnp.asarray(line.temps, dtype=jnp.float64)
        nes = jnp.asarray(line.electron_number_densities, dtype=jnp.float64)

        lambda0 = _interp_lambda0_all_layers_jit(
            T, ne, temps, nes, jnp.asarray(line.lambda0_data, dtype=jnp.float64))

        # A layer is usable when both T and ne sit inside the tabulated grid.
        # Traced comparison, static shape — the NumPy path built the same mask
        # with a Python loop and `np.any` early-out.
        valid = ((T >= temps[0]) & (T <= temps[-1])
                 & (ne >= nes[0]) & (ne <= nes[-1]))

        if line.lower == 2 and line.upper in _BALMER_ABO_PARAMS:
            lam0_abo, sigma_abo_a0, alpha_abo = _BALMER_ABO_PARAMS[line.upper]
            lambda0_stehle = jnp.full((n_layers,), lam0_abo)
            sigma_abo = float(sigma_abo_a0 * bohr_radius_cgs ** 2)
            abo_active = 1.0
        else:
            lambda0_stehle = lambda0
            sigma_abo, alpha_abo, abo_active = 0.0, 0.0, 0.0

        alpha = alpha + _process_stehle_line_all_layers_jit(
            wl_grid, T, ne, nH_I, U_H_I,
            lambda0, lambda0_stehle, valid,
            150.0e-8, vmic, ws_all,
            line.lower, line.upper, float(line.log_gf),
            sigma_abo, alpha_abo, abo_active,
            temps, nes,
            jnp.asarray(line.log_delta_nu_grid, dtype=jnp.float64),
            jnp.asarray(line.profile_data, dtype=jnp.float64),
        )
    return alpha
