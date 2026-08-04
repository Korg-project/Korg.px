"""
Stellar parameter and abundance fitting via spectrum synthesis.

Port of Korg.jl Fit/fit_via_synthesis.jl.

Performance note
----------------
A synthesis is no longer the ~120 s this note used to quote.  Measured on a
5 A window with 174 lines: 3.3 s per objective evaluation through the numerical
path on GPU and 2.5 s on CPU (the first call is far slower -- it loads the
648 MB MARCS grid), against 0.35 s through a compiled traced synthesis on GPU.
A BFGS fit over three stellar parameters is therefore a few minutes rather than
hours for a window of that size, though the cost still scales with the linelist
and the wavelength range.  Julia's native-compiled code does the same synthesis
in ~0.1 s, so the Julia package remains the better tool for survey-scale
fitting.

Gradients
---------
Korg.jl differentiates its fitting objective with ForwardDiff.  Here there are
three paths, in decreasing order of preference:

* **Traced synthesis** (:func:`_make_traced_chi2`), opt-in via
  ``fit_spectrum(..., exact_gradients=True)``.
  :func:`korg.synthesis_plan.prepare_synthesis` returns a closure that runs the
  MARCS interpolation, the chemical equilibrium, the opacities and the transfer
  as one traced JAX program, so ``Teff``, ``logg``, ``M_H``, ``alpha_H``,
  ``vmic`` and the per-element abundances are differentiable too.  When every
  free parameter is one JAX can see, :func:`fit_spectrum` builds one plan for
  the whole fit and hands BFGS exact ``jax.value_and_grad`` derivatives.  A
  reverse-mode gradient costs about what one synthesis costs, whatever the
  number of parameters, where a forward difference costs one synthesis *per
  parameter* on top of the value.  It is not yet the default; see
  ``fit_spectrum``'s ``exact_gradients`` documentation for the measurement.
* **Post-processing only** (:func:`_make_autodiff_chi2`).  ``vsini``,
  ``epsilon``, ``cntm_offset`` and ``cntm_slope``
  (:data:`_POSTPROCESSING_PARAMS`) act only through
  :func:`_postprocess_flux`.  If those are the only free parameters the raw
  synthesis is a *constant* of the fit, so it is computed once — cheaper still
  than the traced path, which resynthesises at every step.
* **Numerical** (:func:`_chi2`).  The fallback: BFGS's own forward differences
  over :func:`_synthetic_spectrum`.  It is what runs when a ``postprocess``
  callback is supplied (it mutates a NumPy buffer in place), when
  ``synthesis_kwargs`` carry options the plan does not accept, or when building
  the plan fails.  :func:`ews_to_stellar_parameters` is still wholly numerical:
  its residuals come from ``ews_to_abundances``, a per-line bisection, not from
  a single traced spectrum.

One behavioural difference between the traced and numerical paths is worth
knowing.  ``interpolate_marcs`` raises ``AtmosphereInterpolationError`` outside
the MARCS grid, which the numerical objective turns into a large chi-squared
and the optimiser walks away from.  The traced kernel has no such check: it
clamps its bracket indices and *linearly extrapolates* instead.  A fit that
strays outside the grid therefore gets a smooth, plausible-looking, wrong
answer rather than a rejection, so :func:`fit_spectrum` checks the best-fit
point against the grid and warns.
"""

import warnings
from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from .abundances import format_A_X
from .atomic_data import atomic_symbols
# From ``synthesis_plan``, not ``synthesis``.  ``synthesis`` re-exports the name,
# so the old import resolved to whichever ``synthesize`` happened to be there --
# at one point a legacy host-orchestrated one whose spectra differed from the
# plan's by ~1e-3, which made the numerical and traced fitting paths disagree
# for a reason that had nothing to do with either.
from .synthesis_plan import synthesize
from .marcs_interpolation import interpolate_marcs
from .utils import apply_rotation, compute_LSF_matrix
from .wavelengths import Wavelengths


# ---------------------------------------------------------------------------
# Parameter scaling
# ---------------------------------------------------------------------------

# Bounds for tan-scaled parameters (same as Julia).
_TAN_SCALE_BOUNDS = {
    "epsilon":      (0.0, 1.0),
    "cntm_offset":  (-0.5, 0.5),
    "cntm_slope":   (-0.1, 0.1),
    "Teff":         (2800.0, 8000.0),
    "logg":         (-0.5, 5.5),
    "M_H":          (-5.0, 1.0),
    "alpha_H":      (-3.5, 2.0),
    **{el: (-10.0, 4.0) for el in atomic_symbols},
}


def _line_atoms(line):
    """
    Atomic numbers of the atoms in a line's species, padding removed.

    ``Species.formula.atoms`` is a fixed-width tuple, left-padded with zeros, so
    its ``len`` says nothing about the species.  This mirrors Julia's
    ``Korg.get_atoms``.  Objects that do not expose a species at all are treated
    as hydrogen, matching the pre-existing fallbacks in this module.
    """
    species = getattr(line, "species", None)
    formula = getattr(species, "formula", None)
    if formula is None:
        return [1]
    return [int(a) for a in formula.atoms if a != 0] or [1]


def _line_atomic_number(line):
    """Atomic number of the principal atom of ``line`` (Julia: ``get_atoms(...)[1]``)."""
    return _line_atoms(line)[0]


def _line_is_molecule(line):
    """True if ``line``'s species contains more than one atom (Julia: ``ismolecule``)."""
    species = getattr(line, "species", None)
    formula = getattr(species, "formula", None)
    if formula is None:
        return False
    return sum(1 for a in formula.atoms if a != 0) > 1


def _tan_scale(p, lower, upper):
    """Map p ∈ [lower, upper] to ℝ via atan-based transform."""
    if not (lower <= p <= upper):
        raise ValueError(f"p={p} is not in [{lower}, {upper}]")
    return float(np.tan(np.pi * ((p - lower) / (upper - lower) - 0.5)))


def _tan_unscale(p, lower, upper):
    """Inverse of _tan_scale."""
    return float((np.arctan(p) / np.pi + 0.5) * (upper - lower) + lower)


def _tan_unscale_jax(p, lower, upper):
    """``_tan_unscale`` written in JAX, for use inside a differentiated objective.

    Identical arithmetic to :func:`_tan_unscale`, minus the ``float()`` cast that
    would abort tracing.  ``jnp.arctan`` is smooth everywhere, so no guarding is
    needed: unlike ``sqrt``, the derivative ``1/(1+p²)`` is finite for every
    finite ``p``, and bounded by 1.
    """
    return (jnp.arctan(p) / jnp.pi + 0.5) * (upper - lower) + lower


def _unscale_param_jax(name, p):
    """Unscale one parameter from ℝ back to its physical range, differentiably."""
    if name in _TAN_SCALE_BOUNDS:
        lo, hi = _TAN_SCALE_BOUNDS[name]
        return _tan_unscale_jax(p, lo, hi)
    if name in ("vmic", "vsini"):
        # The forward map is tan_scale(sqrt(p)); the inverse squares, and x**2
        # has a finite derivative everywhere (the *forward* sqrt is the singular
        # direction, and it is never differentiated).
        return _tan_unscale_jax(p, 0.0, np.sqrt(250.0)) ** 2
    raise ValueError(f"Unknown parameter '{name}'")


def _scale_params(params):
    """Scale each parameter to ℝ for unconstrained optimisation."""
    scaled = {}
    for name, p in params.items():
        if name in _TAN_SCALE_BOUNDS:
            lo, hi = _TAN_SCALE_BOUNDS[name]
            scaled[name] = _tan_scale(p, lo, hi)
        elif name in ("vmic", "vsini"):
            scaled[name] = _tan_scale(np.sqrt(p), 0.0, np.sqrt(250.0))
        else:
            raise ValueError(f"Unknown parameter '{name}'")
    return scaled


def _unscale_params(params):
    """Unscale each parameter from ℝ back to its physical range."""
    unscaled = {}
    for name, p in params.items():
        if name in _TAN_SCALE_BOUNDS:
            lo, hi = _TAN_SCALE_BOUNDS[name]
            unscaled[name] = _tan_unscale(p, lo, hi)
        elif name in ("vmic", "vsini"):
            unscaled[name] = _tan_unscale(p, 0.0, np.sqrt(250.0)) ** 2
        else:
            raise ValueError(f"Unknown parameter '{name}'")
    return unscaled


# ---------------------------------------------------------------------------
# Spectrum synthesis helper
# ---------------------------------------------------------------------------

#: Parameters that affect the observed spectrum *only* through the
#: post-synthesis pipeline (:func:`_postprocess_flux`).  For a fit in which
#: every free parameter is one of these, the raw synthesis is a constant and
#: the whole objective is a JAX expression, so ``jax.grad`` supplies exact
#: gradients and only one synthesis is needed for the entire fit.  Everything
#: else (Teff, logg, M_H, vmic, per-element abundances) enters through
#: ``synthesize``/``interpolate_marcs``, which drop to host NumPy and cannot be
#: traced -- see the module docstring.
_POSTPROCESSING_PARAMS = frozenset({"vsini", "epsilon", "cntm_offset", "cntm_slope"})


def _concrete(value):
    """Concrete float for a value that may be a JAX tracer's primal.

    Window *shapes* in ``apply_rotation`` cannot depend on a traced quantity, so
    the ``vsini > 0`` short-circuit has to be decided from the primal value.
    Under eager ``jax.grad`` the primal is concrete and this succeeds.
    """
    try:
        return float(np.asarray(value))
    except Exception:
        return float(np.asarray(jax.lax.stop_gradient(value)))


def _postprocess_flux(raw_flux, raw_cntm, wavelengths, LSF_matrix,
                      cntm_offset, cntm_slope, vsini, epsilon):
    """
    Rectify, rotationally broaden, and LSF-convolve a raw synthesis.

    This is the entire dependence of the model spectrum on ``cntm_offset``,
    ``cntm_slope``, ``vsini`` and ``epsilon``.  Every operation here is a JAX
    primitive (``apply_rotation`` was rewritten in JAX earlier in this
    project), so ``jax.grad`` flows through it with respect to all four.

    Parameters
    ----------
    raw_flux, raw_cntm : ndarray, shape (n_synth,)
        Flux and continuum straight out of ``synthesize``.
    wavelengths : ndarray, shape (n_synth,)
        Synthesis wavelengths in Å.
    LSF_matrix : ndarray, shape (n_obs, n_synth)
    cntm_offset, cntm_slope, vsini, epsilon : float or JAX scalar

    Returns
    -------
    array, shape (n_obs,)
        Continuum-normalised, broadened, LSF-convolved flux.
    """
    central_wl = (wavelengths[0] + wavelengths[-1]) / 2.0
    cntm_adj = 1.0 - cntm_offset - cntm_slope * (wavelengths - central_wl)
    F = raw_flux / (raw_cntm * cntm_adj)

    if _concrete(vsini) > 0:
        F = apply_rotation(F, wavelengths, vsini, epsilon)

    return LSF_matrix @ F


def _raw_synthesis(synthesis_wls, linelist, params, synthesis_kwargs):
    """
    Run ``synthesize`` for a parameter dict and return ``(flux, cntm, wls_Å)``.

    This is the half of :func:`_synthetic_spectrum` that depends on the
    atmospheric parameters.  It is factored out so that a fit over
    post-processing-only parameters can call it exactly once.

    Returns
    -------
    tuple of ndarray
        ``(raw_flux, raw_continuum, wavelengths_angstrom)``.
    """
    # Build abundance vector from element-specific params
    element_abunds = {el: params[el] for el in atomic_symbols if el in params}
    alpha_H = params.get("alpha_H", params["M_H"])
    A_X = format_A_X(params["M_H"], alpha_H, element_abunds, solar_relative=True)

    atm = interpolate_marcs(params["Teff"], params["logg"], A_X,
                            perturb_at_grid_values=True)

    # Extract wavelength array in Å from Wavelengths object (or pass directly)
    if hasattr(synthesis_wls, "all_wls"):
        wl_angstrom = np.asarray(synthesis_wls.all_wls) * 1e8  # cm → Å
    else:
        wl_angstrom = np.asarray(synthesis_wls)

    # ``synthesize`` is ``synthesis_plan.synthesize``: it takes
    # (atmosphere, linelist, wavelengths_angstrom, A_X, ...) and returns a plain
    # ``(flux, continuum)`` pair.  It has no ``verbose`` keyword -- anything it
    # does not recognise is forwarded to ``prepare_synthesis``, so passing one
    # raised ``TypeError`` and made every real (unmocked) call to this function
    # fail.  ``line_buffer=0`` because the synthesis grid already carries
    # ``wl_buffer`` around each window, as in Korg.jl's ``fit_spectrum``.
    _synth_kw = {k: v for k, v in synthesis_kwargs.items()
                 if k not in ("verbose", "line_buffer")}
    result = synthesize(atm, linelist, wl_angstrom, A_X,
                        vmic=params.get("vmic", 1.0),
                        line_buffer=0,
                        **_synth_kw)

    # A tuple from ``synthesis_plan.synthesize``; an object with ``.flux`` from
    # the fakes the tests install and from any older result type.
    if isinstance(result, tuple):
        flux, cntm = result
        wls = wl_angstrom
    else:
        flux, cntm, wls = result.flux, result.continuum, result.wavelengths

    return np.asarray(flux), np.asarray(cntm), np.asarray(wls)


def _synthetic_spectrum(synthesis_wls, linelist, LSF_matrix, params, synthesis_kwargs):
    """
    Synthesise a spectrum, apply LSF, and rectify to continuum.

    Parameters
    ----------
    synthesis_wls : Wavelengths
        Wavelength grid for synthesis.
    linelist : list of Line
        Spectral lines.
    LSF_matrix : ndarray, shape (n_obs, n_synth)
        LSF convolution matrix.
    params : dict
        All parameters (merged initial_guesses + fixed_params, unscaled).
    synthesis_kwargs : dict
        Extra keyword arguments forwarded to synthesize().

    Returns
    -------
    ndarray, shape (n_obs,)
        Continuum-normalised, LSF-convolved flux at observed wavelengths.
    """
    raw_flux, raw_cntm, wls = _raw_synthesis(synthesis_wls, linelist, params,
                                             synthesis_kwargs)
    return _postprocess_flux(
        raw_flux, raw_cntm, wls, LSF_matrix,
        params.get("cntm_offset", 0.0), params.get("cntm_slope", 0.0),
        params.get("vsini", 0.0), params.get("epsilon", 0.6),
    )


# ---------------------------------------------------------------------------
# Continuum adjustment
# ---------------------------------------------------------------------------

def _linear_continuum_adjustment(obs_wls, windows, model_flux, obs_flux, obs_err):
    """
    Adjust model_flux in-place with the best-fit linear continuum correction.

    Within each window, fits a linear function f(λ) = a + b·λ such that
    model_flux *= a + b·λ minimises chi-squared against obs_flux.
    """
    if windows is None:
        windows = [(float(obs_wls[0]), float(obs_wls[-1]))]

    for lam_start, lam_stop in windows:
        lb = np.searchsorted(obs_wls, lam_start)
        ub = np.searchsorted(obs_wls, lam_stop, side="right")
        if ub <= lb:
            continue
        sl = slice(lb, ub)
        ivar = 1.0 / obs_err[sl] ** 2
        mf = model_flux[sl]
        wl = obs_wls[sl]
        X = np.column_stack([mf, mf * wl])
        XtW = X.T * ivar
        beta = np.linalg.solve(XtW @ X, XtW @ obs_flux[sl])
        model_flux[sl] *= beta[0] + beta[1] * wl


def _linear_continuum_adjustment_jax(obs_wls, windows, model_flux, obs_flux, obs_err):
    """
    Functional, JAX-traceable twin of :func:`_linear_continuum_adjustment`.

    Same arithmetic, but returns a new array instead of mutating in place (JAX
    arrays are immutable) so that it can sit inside a differentiated objective.
    The window index bounds come from ``obs_wls``, which is never a traced
    quantity, so the slicing stays static.
    """
    obs_wls = np.asarray(obs_wls, dtype=float)
    obs_err = np.asarray(obs_err, dtype=float)
    obs_flux = jnp.asarray(obs_flux)
    model_flux = jnp.asarray(model_flux)

    if windows is None:
        windows = [(float(obs_wls[0]), float(obs_wls[-1]))]

    for lam_start, lam_stop in windows:
        lb = int(np.searchsorted(obs_wls, lam_start))
        ub = int(np.searchsorted(obs_wls, lam_stop, side="right"))
        if ub <= lb:
            continue
        ivar = 1.0 / obs_err[lb:ub] ** 2
        mf = model_flux[lb:ub]
        wl = obs_wls[lb:ub]
        X = jnp.stack([mf, mf * wl], axis=1)
        XtW = X.T * ivar
        beta = jnp.linalg.solve(XtW @ X, XtW @ obs_flux[lb:ub])
        model_flux = model_flux.at[lb:ub].multiply(beta[0] + beta[1] * wl)

    return model_flux


# ---------------------------------------------------------------------------
# Exact-gradient (jax.grad) objective
# ---------------------------------------------------------------------------

def _make_autodiff_chi2(raw, LSF_matrix, params_to_fit, fixed_params, obs_wls,
                        obs_flux, obs_err, windows, adjust_continuum):
    """
    Build ``chi2(scaled_p)`` as a pure JAX function of the scaled parameters.

    ``raw`` is ``(raw_flux, raw_cntm, wavelengths)`` from a single call to
    :func:`_raw_synthesis`.  Because every name in ``params_to_fit`` is in
    :data:`_POSTPROCESSING_PARAMS`, the raw synthesis does not depend on them,
    so it is a constant and the objective is differentiable end to end.

    Returns a callable suitable for ``jax.value_and_grad``.
    """
    raw_flux, raw_cntm, wls = raw
    raw_flux = jnp.asarray(raw_flux)
    raw_cntm = jnp.asarray(raw_cntm)
    wls = np.asarray(wls, dtype=float)
    LSF_matrix = jnp.asarray(LSF_matrix)
    obs_flux = jnp.asarray(obs_flux)
    obs_err = jnp.asarray(obs_err)

    def chi2(scaled_p):
        # Weak Gaussian prior in scaled space to regularise (matches Julia)
        neg_log_prior = jnp.sum(scaled_p ** 2 / 100.0 ** 2)

        params = dict(fixed_params)
        for name, value in zip(params_to_fit, scaled_p):
            params[name] = _unscale_param_jax(name, value)

        flux = _postprocess_flux(
            raw_flux, raw_cntm, wls, LSF_matrix,
            params.get("cntm_offset", 0.0), params.get("cntm_slope", 0.0),
            params.get("vsini", 0.0), params.get("epsilon", 0.6),
        )
        if adjust_continuum:
            flux = _linear_continuum_adjustment_jax(obs_wls, windows, flux,
                                                    obs_flux, obs_err)
        return jnp.sum(((flux - obs_flux) / obs_err) ** 2) + neg_log_prior

    return chi2


def _objective_from_chi2(chi2, params_to_fit, trace, start_time, time_limit):
    """Wrap a pure-JAX ``chi2(scaled_p)`` for scipy's ``minimize(..., jac=True)``.

    Shared by both exact-gradient paths.  ``jax.value_and_grad`` is reverse
    mode, so the gradient costs roughly one extra evaluation *in total* rather
    than one per parameter.
    """
    from datetime import datetime

    value_and_grad = jax.value_and_grad(chi2)

    def objective(scaled_p):
        scaled_p = jnp.asarray(scaled_p, dtype=float)
        total, grad = value_and_grad(scaled_p)
        total = float(total)
        neg_log_prior = float(np.sum(np.asarray(scaled_p) ** 2 / 100.0 ** 2))
        guess = _unscale_params(dict(zip(params_to_fit, np.asarray(scaled_p))))
        trace.append({**guess, "chi2": total - neg_log_prior})
        if (datetime.now() - start_time).total_seconds() > time_limit:
            raise StopIteration("Time limit reached")
        return total, np.asarray(grad, dtype=float)

    return objective


def _make_autodiff_objective(raw, LSF_matrix, params_to_fit, fixed_params, obs_wls,
                             obs_flux, obs_err, windows, adjust_continuum, trace,
                             start_time, time_limit):
    """Wrap :func:`_make_autodiff_chi2` for scipy's ``minimize(..., jac=True)``."""
    chi2 = _make_autodiff_chi2(raw, LSF_matrix, params_to_fit, fixed_params, obs_wls,
                               obs_flux, obs_err, windows, adjust_continuum)
    return _objective_from_chi2(chi2, params_to_fit, trace, start_time, time_limit)


# ---------------------------------------------------------------------------
# Exact gradients through the synthesis itself (the traced closure)
# ---------------------------------------------------------------------------

#: Every parameter this module fits, all of which the traced closure can see.
#: The set is written out rather than derived from ``_ALLOWED_PARAMS`` so that a
#: parameter added later has to be classified deliberately.
_TRACEABLE_PARAMS = frozenset(
    {"Teff", "logg", "M_H", "alpha_H", "vmic"} | _POSTPROCESSING_PARAMS
    | set(atomic_symbols)
)

def _plan_kwarg_names():
    """``synthesis_kwargs`` the traced path can honour, read from the signature.

    They are forwarded to ``prepare_synthesis``, not to ``synthesize``; anything
    else means the caller wants something the plan does not model, and the
    numerical path runs instead.  Read from the signature rather than listed,
    because the plan keeps growing options (``hydrogen_lines`` arrived after
    this path was written) and a stale list would silently push a fit onto the
    numerical path the first time someone passed a new one.
    """
    import inspect
    from .synthesis_plan import prepare_synthesis

    return frozenset(
        name for name, p in inspect.signature(prepare_synthesis).parameters.items()
        if p.kind is inspect.Parameter.KEYWORD_ONLY)


def _traced_A_X(params, element_names):
    """A(X) as a JAX vector, mirroring :func:`korg.abundances.format_A_X`.

    ``format_A_X`` builds the vector with a Python loop over ``Z`` and indexes a
    dict, so it cannot consume a tracer.  This is the same three rules —
    hydrogen is 12 by definition, alpha elements get ``[alpha/H]``, other metals
    get ``[M/H]``, explicit elements override both — written as masked
    arithmetic on constants that are fixed before the fit starts.

    ``element_names`` is the (static) list of element symbols that appear in the
    parameter dict; its values may be traced.
    """
    from .abundances import DEFAULT_SOLAR_ABUNDANCES, DEFAULT_ALPHA_ELEMENTS
    from .atomic_data import MAX_ATOMIC_NUMBER, atomic_numbers

    Z = np.arange(1, MAX_ATOMIC_NUMBER + 1)
    solar = jnp.asarray(DEFAULT_SOLAR_ABUNDANCES, dtype=float)
    metal_mask = jnp.asarray((Z >= 3).astype(float))
    alpha_mask = jnp.asarray(np.isin(Z, DEFAULT_ALPHA_ELEMENTS).astype(float))

    M_H = params["M_H"]
    alpha_H = params.get("alpha_H", M_H)

    A_X = solar + metal_mask * M_H + alpha_mask * (alpha_H - M_H)
    A_X = A_X.at[0].set(12.0)      # A(H) = 12 by definition, as in format_A_X
    for el in element_names:
        i = atomic_numbers[el] - 1
        A_X = A_X.at[i].set(solar[i] + params[el])
    return A_X


def _marcs_grid_params_traced(A_X):
    """``(M_H, alpha_M, C_M)`` for the MARCS grid, from a traced ``A(X)``.

    ``interpolate_marcs`` given an ``A_X`` vector derives the three grid axes
    from it with :func:`korg.abundances.get_metals_H` / ``get_alpha_H`` against
    the **Grevesse 2007** solar scale, which is not the scale ``format_A_X``
    used to build the vector.  The difference is a fixed offset of order 0.05
    dex, so reproducing the convention here rather than passing ``[M/H]``
    straight through is what keeps the traced path on the same atmosphere as the
    numerical one; getting it wrong would show up as a systematic shift in
    fitted parameters between the two.
    """
    from .abundances import GREVESSE_2007_SOLAR_ABUNDANCES, DEFAULT_ALPHA_ELEMENTS
    from .atomic_data import MAX_ATOMIC_NUMBER

    Z = np.arange(1, MAX_ATOMIC_NUMBER + 1)
    solar = np.asarray(GREVESSE_2007_SOLAR_ABUNDANCES, dtype=float)
    alpha = np.asarray(DEFAULT_ALPHA_ELEMENTS)
    alpha_and_C = np.append(alpha, 6)

    metals = ((Z >= 3) & ~np.isin(Z, alpha_and_C)).astype(float)
    alphas = np.isin(Z, alpha).astype(float)

    def multi_X_H(mask):
        # Korg's _get_multi_X_H: log10 sum of 10**A(X) over a set of elements,
        # minus the same sum over the solar scale.  The 12s cancel.
        #
        # ``mask`` stays NumPy: under an enclosing ``jit`` a ``jnp`` constant is
        # staged out as a tracer, and the solar half of this expression is a
        # host constant that must not become one.
        num = jnp.log10(jnp.sum(jnp.asarray(mask) * 10.0 ** A_X))
        den = float(np.log10(np.sum(mask * 10.0 ** solar)))
        return num - den

    M_H = multi_X_H(metals)
    alpha_H = multi_X_H(alphas)
    C_H = A_X[5] - float(solar[5])
    return M_H, alpha_H - M_H, C_H - M_H


def _plan_for_fit(synthesis_wls, linelist, synthesis_kwargs):
    """Build the :class:`~korg.synthesis_plan.Synthesizer` a traced fit needs.

    ``line_buffer_cm=0`` matches :func:`_raw_synthesis` and Korg.jl's
    ``fit_spectrum``: the synthesis grid already carries ``wl_buffer`` around
    every window, so a second buffer would only widen the linelist.
    """
    from .synthesis_plan import prepare_synthesis

    # prepare_synthesis takes Angstroms, as synthesize and synth do. A
    # `Wavelengths` object stores `all_wls` in cm, so that branch converts up.
    if hasattr(synthesis_wls, "all_wls"):
        wl_a = np.asarray(synthesis_wls.all_wls, dtype=float) * 1e8
    else:
        wl_a = np.asarray(synthesis_wls, dtype=float)

    # A preprocessed ``LinelistData`` passes through untouched; anything else is
    # materialised, since ``prepare_synthesis`` indexes it.  ``line_buffer_cm``
    # is a default rather than a fixed argument so a caller can still widen it.
    lines = linelist if hasattr(linelist, "wl") else list(linelist)
    return prepare_synthesis(wl_a, lines,
                             **{"line_buffer_cm": 0.0, **synthesis_kwargs})


def _make_traced_model(synth, LSF_matrix, params_to_fit, fixed_params):
    """Build ``model(scaled_p) -> flux`` with the synthesis inside the JAX graph.

    Unlike :func:`_make_autodiff_chi2`'s model, the raw spectrum is recomputed at
    every step — it depends on the parameters — but it is recomputed by one
    compiled XLA program, and one reverse-mode sweep through that program yields
    the derivative with respect to every free parameter at once.

    Only the synthesis is wrapped in :func:`jax.jit`.  The post-processing stage
    is deliberately left eager because ``apply_rotation`` sizes its convolution
    window from a *concrete* ``vsini``: under ``jax.grad`` alone the primal is
    concrete and that works, but a ``jit`` spanning it would make ``vsini`` a
    dynamic tracer and abort.  ``jit`` composes with autodiff either way, so the
    synthesis is still compiled when its gradient is taken.
    """
    from .synthesis_plan import _A_X_to_absolute_traced

    wls_ang = np.asarray(synth.wavelengths_cm, dtype=float) * 1e8
    LSF_matrix = jnp.asarray(LSF_matrix)

    all_names = set(params_to_fit) | set(fixed_params)
    element_names = [el for el in atomic_symbols if el in all_names]

    @jax.jit
    def _synthesize(stellar, abundances):
        return synth(stellar[0], stellar[1], stellar[2], stellar[3], stellar[4],
                     abundances=abundances, vmic=stellar[5])

    def model(scaled_p):
        params = dict(fixed_params)
        for name, value in zip(params_to_fit, scaled_p):
            params[name] = _unscale_param_jax(name, value)

        A_X = _traced_A_X(params, element_names)
        M_H, alpha_M, C_M = _marcs_grid_params_traced(A_X)
        stellar = jnp.stack([
            jnp.asarray(params["Teff"], dtype=float),
            jnp.asarray(params["logg"], dtype=float),
            M_H, alpha_M, C_M,
            # km/s: the closure takes the same unit as synthesize now, so the
            # 1e5 that used to be applied here has moved inside it.
            jnp.asarray(params.get("vmic", 1.0), dtype=float),
        ])
        raw_flux, raw_cntm = _synthesize(stellar, _A_X_to_absolute_traced(A_X))

        return _postprocess_flux(
            raw_flux, raw_cntm, wls_ang, LSF_matrix,
            params.get("cntm_offset", 0.0), params.get("cntm_slope", 0.0),
            params.get("vsini", 0.0), params.get("epsilon", 0.6),
        )

    return model


def _make_traced_chi2(model, obs_wls, obs_flux, obs_err, windows, adjust_continuum):
    """``chi2(scaled_p)`` around a :func:`_make_traced_model` model.

    The model is passed in rather than built here so that a caller which also
    needs the best-fit spectrum reuses the *same* jitted synthesis: a second
    ``jax.jit`` over the same closure is a second compilation, and the synthesis
    program takes a minute or two to compile.
    """
    obs_flux = jnp.asarray(obs_flux)
    obs_err = jnp.asarray(obs_err)

    def chi2(scaled_p):
        # Weak Gaussian prior in scaled space to regularise (matches Julia)
        neg_log_prior = jnp.sum(scaled_p ** 2 / 100.0 ** 2)
        flux = model(scaled_p)
        if adjust_continuum:
            flux = _linear_continuum_adjustment_jax(obs_wls, windows, flux,
                                                    obs_flux, obs_err)
        return jnp.sum(((flux - obs_flux) / obs_err) ** 2) + neg_log_prior

    return chi2


def _warn_outside_marcs_grid(params):
    """Warn if ``params`` sits outside the MARCS grid the traced kernel extrapolates.

    ``interpolate_marcs`` raises there; ``_interpolate_marcs_jit`` clamps its
    bracket indices and extrapolates linearly, silently.  The traced objective
    therefore has no barrier at the grid edge, and the tan-scaled bounds do not
    supply one either -- ``[M/H]`` is scaled into [-5, 1] while the grid stops
    at -2.5.
    """
    try:
        from .marcs_interpolation import load_marcs_grid
        nodes, _ = load_marcs_grid()
        A_X = np.asarray(_traced_A_X({k: float(v) for k, v in params.items()},
                                     [el for el in atomic_symbols if el in params]))
        M_H, alpha_M, C_M = (float(x) for x in _marcs_grid_params_traced(jnp.asarray(A_X)))
        values = [float(params["Teff"]), float(params["logg"]), M_H, alpha_M, C_M]
    except Exception:
        return
    names = ["Teff", "log(g)", "[M/H]", "[alpha/M]", "[C/metals]"]
    for value, name, node in zip(values, names, nodes):
        lo, hi = float(node[0]), float(node[-1])
        if not (lo <= value <= hi):
            warnings.warn(
                f"{name} = {value:g} is outside the MARCS grid [{lo:g}, {hi:g}]. "
                "The traced synthesis extrapolates rather than raising, so this "
                "fit is not constrained by a model atmosphere.")


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------

_REQUIRED_PARAMS = ["Teff", "logg"]
_DEFAULT_PARAMS = {
    "M_H": 0.0,
    "vsini": 0.0,
    "vmic": 1.0,
    "epsilon": 0.6,
    "cntm_offset": 0.0,
    "cntm_slope": 0.0,
}
_ALLOWED_PARAMS = set(
    _REQUIRED_PARAMS
    + list(_DEFAULT_PARAMS)
    + ["alpha_H"]
    + list(atomic_symbols)
)


def validate_params(initial_guesses, fixed_params=None):
    """
    Validate fitting parameters and insert defaults.

    Parameters
    ----------
    initial_guesses : dict
        Parameters to optimise (name → initial value).
    fixed_params : dict, optional
        Parameters held fixed during fitting.

    Returns
    -------
    initial_guesses : dict
        Cleaned dict with float values.
    fixed_params : dict
        Cleaned dict with defaults filled in.
    """
    if fixed_params is None:
        fixed_params = {}

    initial_guesses = {str(k): float(v) for k, v in initial_guesses.items()}
    fixed_params = {str(k): float(v) for k, v in fixed_params.items()}

    all_params = set(initial_guesses) | set(fixed_params)

    for param in _REQUIRED_PARAMS:
        if param not in all_params:
            raise ValueError(
                f"Must specify '{param}' in initial_guesses or fixed_params."
            )

    unknown = all_params - _ALLOWED_PARAMS
    if unknown:
        raise ValueError(f"Unrecognised parameters: {unknown}")

    both = set(initial_guesses) & set(fixed_params)
    if both:
        raise ValueError(f"Parameters in both initial_guesses and fixed_params: {both}")

    if "cntm_offset" in initial_guesses or "cntm_slope" in initial_guesses:
        warnings.warn(
            "Instead of 'cntm_offset'/'cntm_slope', prefer adjust_continuum=True.",
            DeprecationWarning,
            stacklevel=3,
        )

    # Fill in defaults for parameters not specified anywhere
    defaults = {k: v for k, v in _DEFAULT_PARAMS.items()
                if k not in initial_guesses and k not in fixed_params}
    fixed_params = {**defaults, **fixed_params}

    return initial_guesses, fixed_params


# ---------------------------------------------------------------------------
# Wavelength / LSF setup
# ---------------------------------------------------------------------------

def _merge_windows(windows, buffer):
    """Merge overlapping windows (each extended by buffer on each side)."""
    expanded = [(lo - buffer, hi + buffer) for lo, hi in windows]
    expanded.sort()
    merged = [expanded[0]]
    for lo, hi in expanded[1:]:
        if lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def _setup_wavelengths_and_LSF(obs_wls, synthesis_wls_arg, LSF_matrix_arg, R, windows,
                                wl_buffer):
    """Set up synthesis wavelengths and LSF matrix."""
    obs_wls = np.asarray(obs_wls, dtype=float)
    if not np.all(np.diff(obs_wls) > 0):
        raise ValueError("obs_wls must be sorted in increasing order.")

    if LSF_matrix_arg is not None or synthesis_wls_arg is not None:
        if R is not None:
            raise ValueError("Cannot specify both R and LSF_matrix/synthesis_wls.")
        if windows is not None:
            raise ValueError("Cannot specify windows together with LSF_matrix/synthesis_wls.")
        if LSF_matrix_arg is None or synthesis_wls_arg is None:
            raise ValueError("Must specify both LSF_matrix and synthesis_wls together.")

        synthesis_wls = Wavelengths(synthesis_wls_arg)
        LSF_matrix = np.asarray(LSF_matrix_arg, dtype=float)
        if LSF_matrix.shape[0] != len(obs_wls):
            raise ValueError("LSF_matrix first dim must equal len(obs_wls).")
        if LSF_matrix.shape[1] != len(synthesis_wls):
            raise ValueError("LSF_matrix second dim must equal len(synthesis_wls).")
        obs_wl_mask = np.ones(len(obs_wls), dtype=bool)
        return synthesis_wls, obs_wl_mask, LSF_matrix

    if R is None:
        raise ValueError("Must specify R (or LSF_matrix + synthesis_wls).")

    if windows is None:
        windows = [(float(obs_wls[0]), float(obs_wls[-1]))]

    merged = _merge_windows(windows, wl_buffer)
    ranges = [(lo, hi) for lo, hi in merged]
    synthesis_wls = Wavelengths(ranges)

    obs_wl_mask = np.zeros(len(obs_wls), dtype=bool)
    for lo, hi in merged:
        lb = np.searchsorted(obs_wls, lo)
        ub = np.searchsorted(obs_wls, hi, side="right")
        obs_wl_mask[lb:ub] = True

    LSF_matrix = compute_LSF_matrix(synthesis_wls, obs_wls[obs_wl_mask], R)
    return synthesis_wls, obs_wl_mask, LSF_matrix


# ---------------------------------------------------------------------------
# Main fitting function
# ---------------------------------------------------------------------------

def fit_spectrum(obs_wls, obs_flux, obs_err, linelist, initial_guesses, fixed_params=None,
                 *, windows=None, R=None, LSF_matrix=None, synthesis_wls=None,
                 wl_buffer=1.0, precision=1e-4, postprocess=None, time_limit=10_000,
                 adjust_continuum=False, exact_gradients=None, **synthesis_kwargs):
    """
    Find the stellar parameters and abundances that best fit an observed spectrum.

    Uses BFGS optimisation with a tan-based parameter scaling so that bounded
    parameters can be optimised without constraints.

    BFGS is driven with exact gradients wherever they are available, which is
    now the case for ``Teff``, ``logg``, ``M_H``, ``alpha_H``, ``vmic`` and the
    per-element abundances as well as for the post-processing parameters -- see
    the module docstring, and ``exact_gradients`` below.

    Parameters
    ----------
    obs_wls : array, shape (n_obs,)
        Observed wavelengths in Å (must be sorted).
    obs_flux : array, shape (n_obs,)
        Observed continuum-normalised flux.
    obs_err : array, shape (n_obs,)
        1-σ uncertainty in obs_flux (must not contain zeros).
    linelist : list of Line
        Spectral lines used for synthesis.
    initial_guesses : dict
        Parameters to optimise, e.g. ``{"Teff": 5777, "logg": 4.44}``.
    fixed_params : dict, optional
        Parameters held fixed during fitting.
    windows : list of (float, float), optional
        Wavelength windows (in Å) to include in chi-squared.
        If None, the full obs_wls range is used.
    R : float or callable, optional
        Spectral resolution R = λ/δλ (may be a function of wavelength).
        Required unless LSF_matrix and synthesis_wls are provided.
    LSF_matrix : ndarray, optional
        Pre-computed LSF matrix, shape (n_obs, n_synth).
    synthesis_wls : array-like, optional
        Synthesis wavelength grid; required when LSF_matrix is given.
    wl_buffer : float, optional
        Extra wavelength range (Å) added around each window for synthesis.
        Default 1.0.
    precision : float, optional
        Convergence tolerance for the BFGS optimiser (in scaled-parameter
        space).  Default 1e-4.
    postprocess : callable, optional
        ``postprocess(flux, obs_flux, obs_err)`` — called after LSF
        convolution to modify flux in place.
    time_limit : float, optional
        Maximum wall time in seconds for the optimiser.  Default 10 000.
    adjust_continuum : bool, optional
        If True, apply a linear continuum correction within each window at
        every optimiser step.  Default False.
    exact_gradients : bool or None, optional
        Whether to differentiate the *synthesis* as well as the post-processing.

        ``None`` (the default) leaves the historical behaviour alone: a fit over
        post-processing parameters alone gets exact gradients, and a fit that
        touches the stellar parameters gets BFGS's forward differences.

        ``True`` puts the synthesis inside the differentiated graph, so BFGS
        gets exact derivatives with respect to ``Teff``, ``logg``, ``M_H``,
        ``alpha_H``, ``vmic`` and the abundances.  It raises rather than falling
        back if the fit rules that out (a ``postprocess`` callback, or
        ``synthesis_kwargs`` the synthesis plan cannot take).

        ``False`` forces the numerical path throughout, which is what to use to
        compare the two.

        It is opt-in rather than the default because the answer depends on the
        backend.  Measured on one machine, fitting ``Teff``, ``logg`` and
        ``M_H`` from a noiseless spectrum over a 5 A window (501 pixels, 174
        lines), starting 200 K, 0.25 dex and 0.3 dex away:

        ===========================  =====  ======  =========  =========
        path                         evals  iters   GPU total  CPU total
        ===========================  =====  ======  =========  =========
        numerical + forward diffs       88      16      297 s     1438 s
        traced + forward diffs          88      14       27 s      671 s
        traced + exact gradients        21      13      148 s      630 s
        ===========================  =====  ======  =========  =========

        All three land on the same answer -- within 0.001 K in ``Teff`` and 1e-6
        dex in ``M_H`` -- so this is purely about cost.  On the GPU the exact
        path halves the wall clock against the status quo *including* the ~130 s
        it spends compiling the backward pass; on CPU that compile is 356 s and
        one traced evaluation costs 7.3 s against 2.5 s for a numerical one, so
        it loses.  (The CPU column was measured under load and is not comparable
        row-to-row in absolute terms; the evaluation counts are exact.)

        The evaluation count is the durable part: 21 against 88, because a
        forward difference costs one synthesis *per parameter* on top of the
        value while a reverse-mode gradient costs about 1.1 in total.  That
        ratio grows with every parameter added, so the more you fit -- and
        abundances are parameters too -- the better this gets.
    **synthesis_kwargs
        Additional keyword arguments forwarded to synthesize().  On the exact
        path they are forwarded to
        :func:`~korg.synthesis_plan.prepare_synthesis` instead, so only plan
        options (``geometry``, ``window_safety``, ``n_mu``, ...) are accepted
        there; anything else falls back to the numerical path.

    Returns
    -------
    result : dict with keys:
        ``best_fit_params``   — dict of best-fit parameter values
        ``best_fit_flux``     — best-fit flux at observed wavelengths
        ``obs_wl_mask``       — bool array selecting wavelengths used
        ``solver_result``     — scipy OptimizeResult object
        ``trace``             — list of dicts (one per optimizer step)
        ``covariance``        — (param_names, Σ) approximate covariance
    """
    obs_wls = np.asarray(obs_wls, dtype=float)
    obs_flux = np.asarray(obs_flux, dtype=float)
    obs_err = np.asarray(obs_err, dtype=float)

    if len(obs_wls) != len(obs_flux) or len(obs_wls) != len(obs_err):
        raise ValueError("obs_wls, obs_flux, and obs_err must have the same length.")
    if not np.all(np.isfinite(obs_wls)) or not np.all(np.isfinite(obs_flux)):
        raise ValueError("obs_wls and obs_flux must not contain NaN or Inf.")
    if np.any(obs_err == 0):
        raise ValueError("obs_err must not contain zeros.")

    synthesis_wls, obs_wl_mask, LSF_mat = _setup_wavelengths_and_LSF(
        obs_wls, synthesis_wls, LSF_matrix, R, windows, wl_buffer
    )

    initial_guesses, fixed_params = validate_params(initial_guesses, fixed_params or {})

    if len(initial_guesses) == 0:
        raise ValueError("Must specify at least one parameter to fit.")

    scaled_guesses = _scale_params(initial_guesses)
    params_to_fit = list(scaled_guesses.keys())
    p0 = np.array([scaled_guesses[k] for k in params_to_fit])

    _obs_flux = obs_flux[obs_wl_mask]
    _obs_err = obs_err[obs_wl_mask]
    _obs_wls = obs_wls[obs_wl_mask]

    from datetime import datetime
    start_time = datetime.now()
    trace = []

    def _chi2(scaled_p):
        # Weak Gaussian prior in scaled space to regularise (matches Julia)
        neg_log_prior = float(np.sum(scaled_p ** 2 / 100.0 ** 2))

        p_dict = dict(zip(params_to_fit, scaled_p))
        guess = _unscale_params(p_dict)
        params = {**fixed_params, **guess}

        try:
            flux = _synthetic_spectrum(synthesis_wls, linelist, LSF_mat, params,
                                       synthesis_kwargs)
        except Exception:
            # Unphysical atmosphere → large chi-squared
            return float(np.sum(1.0 / _obs_err ** 2))

        if postprocess is not None:
            try:
                postprocess(flux, _obs_flux, _obs_err)
            except Exception:
                pass

        if adjust_continuum:
            try:
                _linear_continuum_adjustment(_obs_wls, windows, flux, _obs_flux, _obs_err)
            except Exception:
                pass

        chi2_val = float(np.sum(((flux - _obs_flux) / _obs_err) ** 2))
        total = chi2_val + neg_log_prior

        trace.append({**guess, "chi2": chi2_val})
        elapsed = (datetime.now() - start_time).total_seconds()
        if elapsed > time_limit:
            raise StopIteration("Time limit reached")

        return total

    # ------------------------------------------------------------------
    # Exact gradients via jax.grad, when every free parameter is one that
    # acts only after synthesis.  In that case the raw synthesis is a
    # constant of the fit: it is computed once, and BFGS gets analytic
    # derivatives instead of a numerical gradient costing one extra
    # synthesis per free parameter per iteration.
    # ------------------------------------------------------------------
    # postprocess mutates a NumPy buffer in place, which no JAX path can do.
    exact_possible = postprocess is None and exact_gradients is not False
    use_autodiff = exact_possible and set(params_to_fit) <= _POSTPROCESSING_PARAMS
    # ...and, failing that, the traced synthesis: exact in the stellar
    # parameters too, at the cost of resynthesising (once, not once per
    # parameter) at every step.  Opt-in for now -- see the ``exact_gradients``
    # documentation for the measurement behind that choice.
    plan_kwargs = _plan_kwarg_names()
    params_ok = set(params_to_fit) <= _TRACEABLE_PARAMS
    kwargs_ok = set(synthesis_kwargs) <= plan_kwargs
    use_traced = (exact_gradients is True and postprocess is None
                  and not use_autodiff and params_ok and kwargs_ok)
    if exact_gradients and not (use_autodiff or use_traced):
        raise ValueError(
            "exact_gradients=True, but this fit cannot use them: "
            + ("a postprocess callback mutates a NumPy buffer in place. "
               if postprocess is not None else "")
            + (f"parameters {sorted(set(params_to_fit) - _TRACEABLE_PARAMS)} are "
               "not traceable. " if not params_ok else "")
            + (f"synthesis_kwargs {sorted(set(synthesis_kwargs) - plan_kwargs)} are "
               "not synthesis-plan options. " if not kwargs_ok else ""))

    objective, jac, traced_model = _chi2, None, None
    if use_autodiff:
        try:
            raw = _raw_synthesis(synthesis_wls, linelist, fixed_params, synthesis_kwargs)
        except Exception as e:  # fall back to the numerical path
            if exact_gradients:
                raise
            warnings.warn(f"Falling back to numerical gradients: {e}")
            use_autodiff = False
        else:
            objective = _make_autodiff_objective(
                raw, LSF_mat, params_to_fit, fixed_params, _obs_wls, _obs_flux,
                _obs_err, windows, adjust_continuum, trace, start_time, time_limit,
            )
            jac = True
    elif use_traced:
        try:
            synth = _plan_for_fit(synthesis_wls, linelist, synthesis_kwargs)
            traced_model = _make_traced_model(synth, LSF_mat, params_to_fit,
                                              fixed_params)
            chi2 = _make_traced_chi2(traced_model, _obs_wls, _obs_flux, _obs_err,
                                     windows, adjust_continuum)
        except Exception as e:
            if exact_gradients:
                raise
            warnings.warn(f"Falling back to numerical gradients: {e}")
            use_traced = False
        else:
            objective = _objective_from_chi2(chi2, params_to_fit, trace, start_time,
                                             time_limit)
            jac = True

    try:
        res = minimize(
            objective, p0,
            method="BFGS", jac=jac,
            options={"gtol": precision, "maxiter": 10_000},
        )
    except StopIteration:
        # Return the best-seen point from the trace rather than the initial guess.
        from scipy.optimize import OptimizeResult
        best_chi2 = min((t["chi2"] for t in trace), default=np.inf)
        if trace and np.isfinite(best_chi2):
            best_t = min(trace, key=lambda t: t.get("chi2", np.inf))
            best_x = _scale_params({k: best_t[k] for k in params_to_fit})
            best_x = np.array([best_x[k] for k in params_to_fit])
        else:
            best_x, best_chi2 = p0, np.inf
        res = OptimizeResult(
            x=best_x, fun=best_chi2, success=False, message="Time limit reached",
            nit=len(trace), hess_inv=np.eye(len(p0)),
        )

    best_fit_params = _unscale_params(dict(zip(params_to_fit, res.x)))

    full_params = {**fixed_params, **best_fit_params}
    if use_traced:
        _warn_outside_marcs_grid(full_params)
    try:
        if traced_model is not None:
            # Reuse the compiled synthesis rather than building a second plan.
            # np.array, not np.asarray: _linear_continuum_adjustment writes in place.
            best_fit_flux = np.array(traced_model(jnp.asarray(res.x, dtype=float)))
        else:
            best_fit_flux = _synthetic_spectrum(synthesis_wls, linelist, LSF_mat,
                                                full_params, synthesis_kwargs)
        if adjust_continuum:
            _linear_continuum_adjustment(_obs_wls, windows, best_fit_flux, _obs_flux, _obs_err)
    except Exception as e:
        warnings.warn(f"Error synthesising best-fit spectrum: {e}")
        best_fit_flux = np.full(int(np.sum(obs_wl_mask)), np.nan)

    # Approximate covariance from BFGS inverse Hessian
    try:
        hess_inv = np.asarray(res.hess_inv)
        # Convert from scaled to unscaled parameter space
        dp_dscaled = np.array([
            _numerical_dp_dscaled(k, res.x[i]) for i, k in enumerate(params_to_fit)
        ])
        cov = hess_inv * np.outer(dp_dscaled, dp_dscaled)
    except Exception:
        cov = np.full((len(params_to_fit), len(params_to_fit)), np.nan)

    return {
        "best_fit_params": best_fit_params,
        "best_fit_flux": best_fit_flux,
        "obs_wl_mask": obs_wl_mask,
        "solver_result": res,
        "trace": trace,
        "covariance": (params_to_fit, cov),
    }


def _numerical_dp_dscaled(name, scaled_val, eps=1e-6):
    """Numerical derivative d(physical)/d(scaled) at scaled_val."""
    p_plus = _unscale_params({name: scaled_val + eps})[name]
    p_minus = _unscale_params({name: scaled_val - eps})[name]
    return (p_plus - p_minus) / (2 * eps)


# ---------------------------------------------------------------------------
# Equivalent width fitting (port of Korg.jl Fit/fit_via_EWs.jl)
# ---------------------------------------------------------------------------

def calculate_EWs(atm, linelist, A_X, ew_window_size=2.0, wl_step=0.01,
                  blend_warn_threshold=0.01, **synthesize_kwargs):
    """
    Compute the equivalent widths of spectral lines via synthesis.

    Port of Korg.jl calculate_EWs.

    Parameters
    ----------
    atm : PlanarAtmosphere or ShellAtmosphere
        Model atmosphere.
    linelist : list of Line
        Spectral lines (must be sorted by wavelength).
    A_X : array, shape (92,)
        Abundances in A(X) format (from format_A_X).
    ew_window_size : float, optional
        Half-width of each synthesis window in Å. Default 2.0.
    wl_step : float, optional
        Wavelength step within each window in Å. Default 0.01.
    blend_warn_threshold : float, optional
        Minimum absorption between adjacent lines before a blend warning.
        Default 0.01.
    **synthesize_kwargs
        Additional keyword arguments passed to synthesize().

    Returns
    -------
    EWs : ndarray, shape (n_lines,)
        Equivalent widths in mÅ.
    """
    lines = list(linelist)
    if not lines:
        return np.array([])

    # Sort check
    wls_cm = np.array([l.wl for l in lines])
    if not np.all(np.diff(wls_cm) >= 0):
        raise ValueError("linelist must be sorted by wavelength")

    wls_ang = wls_cm * 1e8  # cm → Å

    # Build one (lo, hi) window per line in Å, then merge overlapping ones.
    # Track which lines fall in each merged window.
    raw_windows = [(wl - ew_window_size, wl + ew_window_size) for wl in wls_ang]
    merged = _merge_windows([(lo, hi) for lo, hi in raw_windows], buffer=0.0)

    # Assign each line to a merged window
    lines_per_merged = [[] for _ in merged]
    for i, wl in enumerate(wls_ang):
        for j, (lo, hi) in enumerate(merged):
            if lo <= wl <= hi:
                lines_per_merged[j].append(i)
                break

    # Build synthesis wavelength ranges from merged windows
    wl_ranges = []
    for lo, hi in merged:
        n_pts = max(2, int(round((hi - lo) / wl_step)) + 1)
        wl_ranges.append(np.linspace(lo, hi, n_pts))

    # Synthesize all windows in one call (shares chemical equilibrium)
    all_wls = np.concatenate(wl_ranges)
    # `synthesize` is the traced closure now: it takes no `verbose`, and it
    # returns a (flux, continuum) tuple rather than a SynthesisResult. Passing
    # `verbose` reaches `prepare_synthesis` through **plan_kwargs and raises.
    flux, continuum = synthesize(
        atm, lines, all_wls, A_X,
        line_buffer=0.0, hydrogen_lines=False,
        **{k: v for k, v in synthesize_kwargs.items() if k != "verbose"})

    flux = np.asarray(flux)
    continuum = np.asarray(continuum)
    depth = 1.0 - flux / continuum

    EWs = np.zeros(len(lines))

    # Compute cumulative start index for each merged window
    cumulative = np.concatenate([[0], np.cumsum([len(r) for r in wl_ranges])])

    for win_idx, line_indices in enumerate(lines_per_merged):
        if not line_indices:
            continue

        i0 = cumulative[win_idx]
        i1 = cumulative[win_idx + 1]
        wl_range = np.asarray(all_wls[i0:i1])
        absorption = depth[i0:i1]
        n_local = len(line_indices)

        # Find boundary index (minimum absorption) between each pair of adjacent lines
        boundaries = [0]
        for k in range(n_local - 1):
            wl_left = wls_ang[line_indices[k]]
            wl_right = wls_ang[line_indices[k + 1]]
            l1_idx = int(round((wl_left - wl_range[0]) / wl_step))
            l2_idx = int(round((wl_right - wl_range[0]) / wl_step))
            l1_idx = max(0, min(l1_idx, len(wl_range) - 1))
            l2_idx = max(0, min(l2_idx, len(wl_range) - 1))
            seg = absorption[l1_idx:l2_idx + 1]
            if len(seg) > 0:
                bound = int(np.argmin(seg)) + l1_idx
            else:
                bound = l1_idx
            if len(seg) > 0 and absorption[bound] > blend_warn_threshold:
                warnings.warn(
                    f"Lines {line_indices[k]} and {line_indices[k+1]} "
                    f"({wls_ang[line_indices[k]]:.2f} Å and {wls_ang[line_indices[k+1]]:.2f} Å) "
                    f"appear blended (minimum absorption between them: "
                    f"{absorption[bound]:.4f} > {blend_warn_threshold}). "
                    f"Adjust blend_warn_threshold to suppress this warning.",
                    stacklevel=2,
                )
            boundaries.append(bound)
        boundaries.append(len(wl_range) - 1)

        for k, li in enumerate(line_indices):
            b0 = boundaries[k]
            b1 = boundaries[k + 1] + 1
            EWs[li] = np.trapezoid(absorption[b0:b1], wl_range[b0:b1]) * 1e3  # Å → mÅ

    return EWs


def ews_to_abundances(atm, linelist, A_X, measured_EWs, ew_window_size=2.0, wl_step=0.01,
                      blend_warn_threshold=0.01, abundance_tol=1e-4,
                      finite_difference_delta_A=0.01, **synthesize_kwargs):
    """
    Derive per-line chemical abundances from observed equivalent widths.

    For each line, adjusts the element's abundance in A_X until the synthetic
    EW matches the measured EW, using a Newton-like iteration.

    Port of Korg.jl ews_to_abundances.

    Parameters
    ----------
    atm : PlanarAtmosphere or ShellAtmosphere
        Model atmosphere.
    linelist : list of Line
        Spectral lines (must be sorted by wavelength).
    A_X : array, shape (92,)
        Starting abundances in A(X) = log10(n_X/n_H)+12 format.
    measured_EWs : array, shape (n_lines,)
        Observed equivalent widths in mÅ.
    ew_window_size : float, optional
        Half-width of each synthesis window in Å. Default 2.0.
    wl_step : float, optional
        Wavelength resolution in Å. Default 0.01.
    blend_warn_threshold : float, optional
        Blend warning threshold. Default 0.01.
    abundance_tol : float, optional
        Convergence tolerance in A(X) dex. Default 1e-4.
    finite_difference_delta_A : float, optional
        Step size for curve-of-growth slope estimation. Default 0.01 dex.
    **synthesize_kwargs
        Extra keyword arguments passed to synthesize().

    Returns
    -------
    abundances : ndarray, shape (n_lines,)
        Best-fit A(X) for each line.
    dA_d_log_EW : ndarray, shape (n_lines,)
        Curve-of-growth slope ∂A/∂log(EW) for each line.
    """
    from .atomic_data import atomic_numbers
    from .linelist import Line

    lines = list(linelist)
    measured_EWs = np.asarray(measured_EWs, dtype=float)
    A_X = np.asarray(A_X, dtype=float).copy()

    if len(lines) != len(measured_EWs):
        raise ValueError("linelist and measured_EWs must have the same length")

    abundances = np.zeros(len(lines))
    dA_d_log_EW = np.zeros(len(lines))

    for i, (line, ew_obs) in enumerate(zip(lines, measured_EWs)):
        # Identify the element (Z) from the line species.
        Z = _line_atomic_number(line)

        A_X_mod = A_X.copy()

        def _get_ew(A_val):
            A_X_mod[Z - 1] = A_val
            ews = calculate_EWs(atm, [line], A_X_mod,
                                 ew_window_size=ew_window_size, wl_step=wl_step,
                                 blend_warn_threshold=1.0,  # suppress blend warnings
                                 **synthesize_kwargs)
            return float(ews[0])

        # Initial EW at starting abundance
        A0 = float(A_X[Z - 1])
        ew0 = _get_ew(A0)

        # Finite difference for curve-of-growth slope
        ew_plus = _get_ew(A0 + finite_difference_delta_A)
        if ew_plus > 1e-10 and ew0 > 1e-10:
            d_log_ew = np.log10(ew_plus) - np.log10(ew0)
            dA_d_log_EW[i] = finite_difference_delta_A / d_log_ew if abs(d_log_ew) > 1e-10 else np.inf
        else:
            dA_d_log_EW[i] = np.inf

        # Newton iteration: adjust A until synthetic EW matches observed EW
        A_cur = A0
        for _ in range(50):
            ew_cur = _get_ew(A_cur)
            if ew_cur < 1e-10 or ew_obs < 1e-10:
                break
            delta_log_ew = np.log10(ew_obs) - np.log10(ew_cur)
            if abs(delta_log_ew) < 1e-6:
                break
            dA = dA_d_log_EW[i] * delta_log_ew if np.isfinite(dA_d_log_EW[i]) else delta_log_ew
            dA = np.clip(dA, -1.0, 1.0)  # limit step size to 1 dex
            A_new = A_cur + dA
            if abs(A_new - A_cur) < abundance_tol:
                A_cur = A_new
                break
            A_cur = A_new

        abundances[i] = A_cur

    return abundances, dA_d_log_EW


def ews_to_abundances_approx(atm, linelist, A_X, measured_EWs, ew_window_size=2.0,
                             wl_step=0.01, blend_warn_threshold=0.01, **synthesize_kwargs):
    """
    Fast approximate per-line abundances from equivalent widths.

    Assumes all lines are on the linear part of the curve of growth.
    Port of Korg.jl ews_to_abundances_approx.

    Parameters
    ----------
    atm : PlanarAtmosphere or ShellAtmosphere
    linelist : list of Line, sorted by wavelength
    A_X : array, shape (92,)
    measured_EWs : array, shape (n_lines,), observed EWs in mÅ
    ew_window_size : float, optional, default 2.0
    wl_step : float, optional, default 0.01
    blend_warn_threshold : float, optional, default 0.01
    **synthesize_kwargs : extra keyword args for synthesize()

    Returns
    -------
    abundances : ndarray, shape (n_lines,)
        A(X) = log10(n_X/n_H) + 12 for each line.
    """
    from .atomic_data import atomic_numbers

    lines = list(linelist)
    measured_EWs = np.asarray(measured_EWs, dtype=float)
    A_X = np.asarray(A_X, dtype=float)

    if len(lines) != len(measured_EWs):
        raise ValueError("linelist and measured_EWs must have the same length")

    EWs_synth = calculate_EWs(atm, lines, A_X,
                              ew_window_size=ew_window_size, wl_step=wl_step,
                              blend_warn_threshold=blend_warn_threshold,
                              **synthesize_kwargs)

    atoms = np.array([_line_atomic_number(line) for line in lines])

    A0 = A_X[atoms - 1]
    # Korg.jl computes `A0 + (log10(measured_EWs) - log10(EWs))` unguarded, so a
    # non-positive EW yields a non-finite result (+Inf for a vanishing synthetic
    # EW, -Inf for a vanishing measured one, NaN for a negative).  That is
    # load-bearing: the callers
    # (`_ews_stellar_param_residuals`) count non-finite entries to decide whether
    # enough lines converged.  Substituting a finite value for a line that did
    # not produce a measurable feature would silently defeat that check, so the
    # non-finite result is propagated, with the warnings suppressed rather than
    # the values.
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.log10(measured_EWs) - np.log10(EWs_synth)
    return A0 + log_ratio


# ---------------------------------------------------------------------------
# Stellar parameter fitting from equivalent widths
# ---------------------------------------------------------------------------

def _get_slope(xs, ys):
    """Slope of best-fit line through (xs, ys) with no intercept term (demeaned)."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    dx = xs - np.mean(xs)
    dy = ys - np.mean(ys)
    denom = np.sum(dx ** 2)
    return float(np.sum(dx * dy) / denom) if denom > 1e-30 else 0.0


def _get_slope_uncertainty(xs):
    """Propagated uncertainty in slope given xs scatter."""
    xs = np.asarray(xs, dtype=float)
    n = len(xs)
    denom = np.sum(xs ** 2) - np.sum(xs) ** 2 / n
    return float(np.sqrt(1.0 / denom)) if denom > 1e-30 else np.inf


def _ews_stellar_param_residuals(params, linelist, measured_EWs, abundance_adjustments,
                                 solar_abundances, fix_params, callback, approx, synthesize_kwargs):
    """Compute the four excitation/ionization balance residuals."""
    Teff, logg, vmic, M_H = params

    A_X = format_A_X(M_H, solar_abundances=solar_abundances)
    try:
        atm = interpolate_marcs(Teff, logg, A_X)
    except Exception as e:
        raise ValueError(f"interpolate_marcs failed: {e}")

    # Julia reaches the EW solvers through their `params` methods, which forward
    # `vmic=vmic` to synthesize.  Dropping it makes the microturbulence residual
    # independent of vmic, so the third column of the Newton Jacobian is
    # identically zero and the 4x4 system is singular: the solver then cannot
    # move any parameter, whatever the input.
    if approx:
        A = ews_to_abundances_approx(atm, linelist, A_X, measured_EWs,
                                     blend_warn_threshold=np.inf,
                                     vmic=vmic, **synthesize_kwargs)
    else:
        A, _ = ews_to_abundances(atm, linelist, A_X, measured_EWs,
                                 blend_warn_threshold=1.0,  # suppress warnings
                                 vmic=vmic, **synthesize_kwargs)

    A = A + np.asarray(abundance_adjustments)

    neutrals = np.array([l.species.charge == 0 for l in linelist])
    REWs = np.log10(np.asarray(measured_EWs) / np.array([l.wl * 1e8 for l in linelist]))

    finite = np.isfinite(A)
    if np.mean(finite) < 0.7:
        raise ValueError("Less than 70% of lines converged.")
    if neutrals[~neutrals].size > 0 and np.mean(finite[~neutrals]) < 0.5:
        raise ValueError("Less than 50% of ion lines converged.")

    neutral_finite = neutrals & finite
    ion_finite = ~neutrals & finite

    E_lower = np.array([l.E_lower for l in linelist])
    teff_res = _get_slope(E_lower[neutral_finite], A[neutral_finite])
    logg_res = float(np.mean(A[neutral_finite]) - np.mean(A[ion_finite]))

    vmic_res = _get_slope(REWs[neutral_finite], A[neutral_finite])

    # Julia: `Z = Korg.get_atoms(linelist[1].species)[1]`.  `formula.atoms` is a
    # fixed-width tuple zero-padded on the left, so `atoms[0]` is the padding,
    # not the element: taking it would evaluate the [m/H] residual against
    # solar_abundances[-1] (uranium) for every atomic linelist.
    Z = _line_atomic_number(linelist[0])
    feh_res = float(np.mean(A[finite]) - (M_H + solar_abundances[Z - 1]))

    residuals = np.array([teff_res, logg_res, vmic_res, feh_res])
    residuals[np.array(fix_params, dtype=bool)] = 0.0

    callback(params, residuals, A)
    return residuals


def ews_to_stellar_parameters_direct(linelist, measured_EWs,
                                     measured_EW_err=None,
                                     Teff0=5000.0, logg0=3.5, vmic0=1.0, M_H0=0.0,
                                     precision=1e-5, time_limit=500.0,
                                     solar_abundances=None,
                                     verbose=False, **synthesize_kwargs):
    """
    Find stellar parameters from EWs by forward modelling (chi-squared minimization).

    Port of Korg.jl ews_to_stellar_parameters_direct.

    Parameters
    ----------
    linelist : list of Line
    measured_EWs : array, shape (n_lines,), in mÅ
    measured_EW_err : array, shape (n_lines,), optional; default ones
    Teff0, logg0, vmic0, M_H0 : float, initial guesses
    precision : float, BFGS gtol; default 1e-5
    time_limit : float, wall time limit in seconds; default 500
    solar_abundances : array (92,), optional
    verbose : bool
    **synthesize_kwargs : extra kwargs for synthesize()

    Returns
    -------
    params : ndarray, shape (4,), [Teff, logg, vmic, M_H]
    uncertainties : ndarray, shape (4,4), approximate covariance from BFGS Hessian
    """
    from .abundances import get_solar_abundances

    if solar_abundances is None:
        solar_abundances = get_solar_abundances()
    lines = list(linelist)
    measured_EWs = np.asarray(measured_EWs, dtype=float)
    if measured_EW_err is None:
        measured_EW_err = np.ones(len(measured_EWs))
    measured_EW_err = np.asarray(measured_EW_err, dtype=float)

    from datetime import datetime
    start = datetime.now()

    def cost(p):
        Teff, logg, vmic, M_H = p[0] * 1e3, p[1], p[2], p[3]
        A_X = format_A_X(M_H, solar_abundances=solar_abundances)
        try:
            atm = interpolate_marcs(Teff, logg, A_X)
            EWs = calculate_EWs(atm, lines, A_X, verbose=False, **synthesize_kwargs)
        except Exception:
            return 1e10
        chi2 = float(np.sum(((EWs - measured_EWs) / measured_EW_err) ** 2))
        if verbose:
            print(f"  Teff={Teff:.0f}, logg={logg:.2f}, vmic={vmic:.2f}, M_H={M_H:.2f}"
                  f"  chi2={chi2:.3f}")
        if (datetime.now() - start).total_seconds() > time_limit:
            raise StopIteration("time limit")
        return chi2

    p0 = np.array([Teff0 / 1e3, logg0, vmic0, M_H0])
    try:
        res = minimize(cost, p0, method="BFGS",
                       options={"gtol": precision, "maxiter": 5000})
    except StopIteration:
        from scipy.optimize import OptimizeResult
        res = OptimizeResult(x=p0, success=False, message="Time limit",
                             hess_inv=np.eye(4))

    params = res.x.copy()
    params[0] *= 1e3  # back to Kelvin

    # scale uncertainty from (Teff/1e3, logg, vmic, M_H) to (Teff, logg, vmic, M_H)
    scales = np.array([1e3, 1.0, 1.0, 1.0])
    try:
        H_inv = np.asarray(res.hess_inv)
        uncertainties = H_inv * np.outer(scales, scales)
    except Exception:
        uncertainties = np.full((4, 4), np.nan)

    return params, uncertainties


def ews_to_stellar_parameters(linelist, measured_EWs,
                              abundance_adjustments=None,
                              Teff0=5000.0, logg0=3.5, vmic0=1.0, M_H0=0.0,
                              tolerances=None,
                              max_step_sizes=None,
                              parameter_ranges=None,
                              fix_params=None,
                              solar_abundances=None,
                              verbose=False,
                              callback=None,
                              max_iterations=30,
                              **synthesize_kwargs):
    """
    Find stellar parameters from EWs by excitation/ionization balance.

    The solver finds Teff, logg, vmic, [m/H] that simultaneously satisfy:

    - slope of A vs E_lower = 0  (excitation balance → Teff)
    - mean(A neutral) - mean(A ionized) = 0  (ionization balance → logg)
    - slope of A vs log10(EW/λ) = 0  (microturbulence balance → vmic)
    - mean(A) - (M_H + solar_A[Z]) = 0  (self-consistency → M_H)

    Port of Korg.jl ews_to_stellar_parameters.

    Parameters
    ----------
    linelist : list of Line, all from the same element, sorted by wavelength
    measured_EWs : array, shape (n_lines,), in mÅ
    abundance_adjustments : array, shape (n_lines,), optional abundance offsets
    Teff0, logg0, vmic0, M_H0 : float, initial guesses
    tolerances : list of 4 floats, default [1e-3, 1e-3, 1e-4, 1e-3]
    max_step_sizes : list of 4 floats, default [1000.0, 1.0, 0.3, 0.5]
    parameter_ranges : list of 4 (lo, hi) tuples
    fix_params : list of 4 bools, default [False, False, False, False]
    solar_abundances : array (92,), optional
    verbose : bool
    callback : callable(params, residuals, abundances), optional
    max_iterations : int, default 30
    **synthesize_kwargs

    Returns
    -------
    params : ndarray, shape (4,), [Teff, logg, vmic, M_H]
    uncertainties : ndarray, shape (4,), parameter uncertainties
    """
    from .abundances import get_solar_abundances

    # Julia raises the same ArgumentError: vmic is a fitted parameter here, and
    # it is forwarded to synthesize internally, so a caller-supplied vmic would
    # both be ignored and collide with that keyword.
    if "vmic" in synthesize_kwargs:
        raise ValueError(
            "vmic must not be specified, because it is a parameter fit by "
            "ews_to_stellar_parameters. Did you mean vmic0, the starting value?")

    if solar_abundances is None:
        solar_abundances = get_solar_abundances()
    if tolerances is None:
        tolerances = [1e-3, 1e-3, 1e-4, 1e-3]
    if max_step_sizes is None:
        max_step_sizes = [1000.0, 1.0, 0.3, 0.5]
    if parameter_ranges is None:
        parameter_ranges = [(2800.0, 8000.0), (-0.5, 5.5), (1e-3, 10.0), (-2.5, 1.0)]
    if fix_params is None:
        fix_params = [False, False, False, False]
    fix_params = np.array(fix_params, dtype=bool)

    lines = list(linelist)
    measured_EWs = np.asarray(measured_EWs, dtype=float)
    if abundance_adjustments is None:
        abundance_adjustments = np.zeros(len(lines))
    abundance_adjustments = np.asarray(abundance_adjustments, dtype=float)

    # Validate inputs
    if len(lines) != len(measured_EWs):
        raise ValueError("linelist and measured_EWs must have the same length")
    # Julia: `if Korg.ismolecule(linelist[1].species) throw(...)`.  `formula.atoms`
    # is a fixed-width tuple zero-padded on the left, so `len(atoms) > 1` is true
    # for *every* species, atomic or not; counting the non-zero entries is what
    # distinguishes a molecule.
    if any(_line_is_molecule(l) for l in lines):
        raise ValueError("All lines must be atomic (no molecules).")
    neutrals = np.array([l.species.charge == 0 for l in lines])
    if neutrals.sum() < 3 or (~neutrals).sum() < 1:
        raise ValueError("Need at least 3 neutral lines and 1 ion line.")

    if verbose and callback is None:
        def callback(p, res, A):
            print(f"Teff={p[0]:.0f} logg={p[1]:.2f} vmic={p[2]:.3f} M_H={p[3]:.2f}"
                  f" | res={np.array2string(np.array(res), precision=4)}")
    elif callback is None:
        callback = lambda p, r, A: None

    # Clamp initial guess
    params = np.array([Teff0, logg0, vmic0, M_H0], dtype=float)
    for i, (lo, hi) in enumerate(parameter_ranges):
        params[i] = np.clip(params[i], lo, hi)

    tolerances = np.asarray(tolerances, dtype=float)
    max_step_sizes = np.asarray(max_step_sizes, dtype=float)

    def _phase(approx, tol_scale):
        nonlocal params
        for _iter in range(max_iterations):
            # Compute Jacobian numerically
            eps = np.array([1.0, 0.01, 0.01, 0.01])  # finite difference step per param
            try:
                r0 = _ews_stellar_param_residuals(
                    params, lines, measured_EWs, abundance_adjustments,
                    solar_abundances, fix_params, callback, approx, synthesize_kwargs)
            except Exception as e:
                warnings.warn(f"Residual evaluation failed: {e}")
                return False

            if np.all(np.abs(r0[~fix_params]) < tolerances[~fix_params] * tol_scale):
                return True  # converged

            # Build Jacobian column by column
            J = np.zeros((4, 4))
            for j in range(4):
                if fix_params[j]:
                    continue
                p_plus = params.copy()
                p_plus[j] += eps[j]
                try:
                    r_plus = _ews_stellar_param_residuals(
                        p_plus, lines, measured_EWs, abundance_adjustments,
                        solar_abundances, fix_params, callback, approx, synthesize_kwargs)
                    J[:, j] = (r_plus - r0) / eps[j]
                except Exception:
                    J[:, j] = 0.0

            # Newton step
            free = ~fix_params
            try:
                step = np.zeros(4)
                step[free] = -np.linalg.solve(J[np.ix_(free, free)], r0[free])
            except np.linalg.LinAlgError:
                step = np.zeros(4)

            step = np.clip(step, -max_step_sizes, max_step_sizes)
            params = params + step
            for i, (lo, hi) in enumerate(parameter_ranges):
                params[i] = np.clip(params[i], lo, hi)

        warnings.warn(f"ews_to_stellar_parameters did not converge after {max_iterations} iterations")
        return False

    # Phase 1: approximate (fast)
    _phase(approx=True, tol_scale=10.0)

    if verbose:
        print("Approximate solve done. Starting exact solve.")

    # Phase 2: exact
    converged = _phase(approx=False, tol_scale=1.0)

    # Estimate uncertainties from line-to-line scatter
    A_X = format_A_X(float(params[3]), solar_abundances=solar_abundances)
    try:
        atm_final = interpolate_marcs(float(params[0]), float(params[1]), A_X)
        A_final, _ = ews_to_abundances(atm_final, lines, A_X, measured_EWs,
                                       blend_warn_threshold=1.0,
                                       vmic=float(params[2]), **synthesize_kwargs)
        A_final = A_final + abundance_adjustments
        finite = np.isfinite(A_final)
        neutral_finite = neutrals & finite
        E_lower = np.array([l.E_lower for l in lines])
        REWs = np.log10(measured_EWs / np.array([l.wl * 1e8 for l in lines]))
        estimated_err = float(np.std(A_final[finite])) if finite.sum() > 1 else np.inf
        n_finite = int(finite.sum())
        sigma_mean = estimated_err / np.sqrt(n_finite) if n_finite > 0 else np.inf
        teff_unc = estimated_err * _get_slope_uncertainty(E_lower[neutral_finite])
        vmic_unc = estimated_err * _get_slope_uncertainty(REWs[neutral_finite])
        uncertainties = np.array([teff_unc, sigma_mean, vmic_unc, sigma_mean])
    except Exception:
        uncertainties = np.full(4, np.nan)

    return params, uncertainties
