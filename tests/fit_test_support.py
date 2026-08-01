"""
Fast stand-ins for the expensive parts of the synthesis pipeline.

`src/korg/fit.py` is almost entirely control flow wrapped around two very
expensive calls, ``synthesize`` and ``interpolate_marcs`` (~7 s and ~2 s
respectively for even a trivial problem).  Testing the control flow against the
real ones would take hours, so these helpers substitute analytic replacements
that have the *shape* the fitting code depends on:

  * ``fake_synthesize`` produces Gaussian absorption lines whose depth grows
    with ``A_X[Z-1]`` and ``log_gf`` and whose width grows with ``vmic``, so a
    curve of growth exists and ``ews_to_abundances`` has something to converge
    to;
  * ``fake_interpolate_marcs`` returns a namespace carrying the requested
    stellar parameters, so a fake synthesis can depend on them.

Both count their calls, which is how the tests prove that the ``jax.grad`` path
in ``fit_spectrum`` needs exactly one synthesis.

This module is deliberately *not* named ``test_*``: pytest will not collect it.
"""

import types

import numpy as np


class FakeSynthesisResult:
    """Duck-type of ``korg.synthesis.SynthesisResult``."""

    def __init__(self, wavelengths, flux, continuum):
        self.wavelengths = np.asarray(wavelengths, dtype=float)
        self.flux = np.asarray(flux, dtype=float)
        self.continuum = np.asarray(continuum, dtype=float)
        self.intensities = None


def line_Z(line):
    """Atomic number of a (possibly mock) line's species; 1 if it has none."""
    formula = getattr(getattr(line, "species", None), "formula", None)
    if formula is None:
        return 1
    return next((int(a) for a in formula.atoms if a != 0), 1)


class FakeSynthesizer:
    """
    Callable with ``synthesize``'s signature that builds Gaussian lines.

    The depth of line *i* is

        d_i = depth_scale * s_i / (1 + saturation * s_i),
        s_i  = 10 ** (A_X[Z_i - 1] - 7.5 + log_gf_i)

    so that with ``saturation=0`` the equivalent width is exactly proportional
    to ``10**A(X)``: the curve of growth is linear with slope
    ``dA/dlog10(EW) = 1``, which makes the expected behaviour of
    ``ews_to_abundances`` analytically known.  With ``saturation>0`` the line
    saturates, exercising the non-linear branch.
    """

    def __init__(self, depth_scale=0.05, width=0.06, saturation=0.0,
                 continuum_level=1.0, continuum_slope=0.0):
        self.depth_scale = depth_scale
        self.width = width
        self.saturation = saturation
        self.continuum_level = continuum_level
        self.continuum_slope = continuum_slope
        self.n_calls = 0
        self.last_kwargs = None

    def __call__(self, atm, linelist, wavelengths_angstrom, A_X, vmic=1.0, **kwargs):
        self.n_calls += 1
        self.last_kwargs = dict(kwargs, vmic=vmic)
        wls = np.asarray(wavelengths_angstrom, dtype=float)
        A_X = np.asarray(A_X, dtype=float)

        cntm = self.continuum_level + self.continuum_slope * (wls - wls.mean())
        depth = np.zeros_like(wls)
        width = self.width * (0.5 + 0.5 * float(vmic))
        for line in linelist:
            wl0 = float(line.wl) * 1e8
            strength = 10.0 ** (A_X[line_Z(line) - 1] - 7.5 + float(line.log_gf))
            d = self.depth_scale * strength
            if self.saturation:
                d = d / (1.0 + self.saturation * strength)
            depth = depth + d * np.exp(-0.5 * ((wls - wl0) / width) ** 2)
        return FakeSynthesisResult(wls, cntm * (1.0 - depth), cntm)


class FakeMarcs:
    """Callable with ``interpolate_marcs``'s signature returning a namespace."""

    def __init__(self, fail_for=None):
        #: optional predicate ``(Teff, logg) -> bool`` selecting failures
        self.fail_for = fail_for
        self.n_calls = 0

    def __call__(self, Teff, logg, M_H_or_A_X=0.0, **kwargs):
        self.n_calls += 1
        if self.fail_for is not None and self.fail_for(Teff, logg):
            raise RuntimeError(f"fake MARCS failure at Teff={Teff}, logg={logg}")
        return types.SimpleNamespace(Teff=float(Teff), logg=float(logg),
                                     n_layers=1, layers=[])


def patch_synthesis(monkeypatch, synthesizer=None, marcs=None):
    """Install fakes into ``korg.fit`` and return ``(synthesizer, marcs)``."""
    import korg.fit as fit

    synthesizer = synthesizer if synthesizer is not None else FakeSynthesizer()
    marcs = marcs if marcs is not None else FakeMarcs()
    monkeypatch.setattr(fit, "synthesize", synthesizer)
    monkeypatch.setattr(fit, "interpolate_marcs", marcs)
    return synthesizer, marcs


def gaussian_spectrum(wls, centres=(5002.0, 5009.0), depths=(0.55, 0.35),
                      widths=(0.22, 0.17), continuum_slope=0.001):
    """A deterministic (flux, continuum) pair standing in for a raw synthesis."""
    wls = np.asarray(wls, dtype=float)
    cntm = 1.0 + continuum_slope * (wls - wls.mean())
    absorption = np.zeros_like(wls)
    for c, d, w in zip(centres, depths, widths):
        absorption = absorption + d * np.exp(-0.5 * ((wls - c) / w) ** 2)
    return cntm * (1.0 - absorption), cntm


def central_difference(f, x, i, h=None):
    """Central finite difference of scalar ``f`` w.r.t. component ``i`` of ``x``."""
    x = np.asarray(x, dtype=float)
    if h is None:
        h = 1e-6 * max(1.0, abs(float(x[i])))
    xp, xm = x.copy(), x.copy()
    xp[i] += h
    xm[i] -= h
    return (float(f(xp)) - float(f(xm))) / (2.0 * h)
