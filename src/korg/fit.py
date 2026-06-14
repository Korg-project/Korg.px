"""
Stellar parameter and abundance fitting via spectrum synthesis.

Port of Korg.jl Fit/fit_via_synthesis.jl.

Performance note
----------------
Each synthesis call currently takes ~120 s in Python (chemical equilibrium
dominates).  A BFGS fit with ~100 function evaluations will take hours.
Julia's native-compiled code does the same synthesis in ~0.1 s, so consider
using the Julia package directly for large-scale fitting.  This module is
provided for correctness testing, single-star analyses, and as a foundation
for future optimisation.
"""

import warnings
from copy import deepcopy

import numpy as np
from scipy.optimize import minimize

from .abundances import format_A_X
from .atomic_data import atomic_symbols
from .synthesis import synthesize
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


def _tan_scale(p, lower, upper):
    """Map p ∈ [lower, upper] to ℝ via atan-based transform."""
    if not (lower <= p <= upper):
        raise ValueError(f"p={p} is not in [{lower}, {upper}]")
    return float(np.tan(np.pi * ((p - lower) / (upper - lower) - 0.5)))


def _tan_unscale(p, lower, upper):
    """Inverse of _tan_scale."""
    return float((np.arctan(p) / np.pi + 0.5) * (upper - lower) + lower)


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

    # synthesize signature: (atmosphere, linelist, wavelengths_angstrom, abundances, ...)
    sol = synthesize(atm, linelist, wl_angstrom, A_X,
                     vmic=params.get("vmic", 1.0),
                     line_buffer=0,
                     verbose=False,
                     **synthesis_kwargs)

    # Continuum rectification with optional linear correction
    central_wl = (sol.wavelengths[0] + sol.wavelengths[-1]) / 2.0
    cntm_adj = (1.0
                - params.get("cntm_offset", 0.0)
                - params.get("cntm_slope", 0.0) * (sol.wavelengths - central_wl))
    F = np.asarray(sol.flux / (sol.continuum * cntm_adj))

    # Rotational broadening
    vsini = params.get("vsini", 0.0)
    epsilon = params.get("epsilon", 0.6)
    if vsini > 0:
        F = apply_rotation(F, np.asarray(sol.wavelengths), vsini, epsilon)

    return LSF_matrix @ F


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
                 adjust_continuum=False, **synthesis_kwargs):
    """
    Find the stellar parameters and abundances that best fit an observed spectrum.

    Uses BFGS optimisation with a tan-based parameter scaling so that bounded
    parameters can be optimised without constraints.

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
    **synthesis_kwargs
        Additional keyword arguments forwarded to synthesize().

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

    try:
        res = minimize(
            _chi2, p0,
            method="BFGS",
            options={"gtol": precision, "maxiter": 10_000},
        )
    except StopIteration:
        # Build a minimal result so we can still return something
        from scipy.optimize import OptimizeResult
        res = OptimizeResult(
            x=p0, fun=np.inf, success=False, message="Time limit reached",
            nit=len(trace), hess_inv=np.eye(len(p0)),
        )

    best_fit_params = _unscale_params(dict(zip(params_to_fit, res.x)))

    full_params = {**fixed_params, **best_fit_params}
    try:
        best_fit_flux = _synthetic_spectrum(synthesis_wls, linelist, LSF_mat, full_params,
                                             synthesis_kwargs)
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
