"""
Cubic spline interpolation.

This module contains functions for cubic spline interpolation, adapted from
DataInterpolations.jl (MIT license). See source for full license details.
"""

import jax.numpy as jnp
from jax.scipy.linalg import solve
from dataclasses import dataclass
from typing import Optional


@dataclass
class CubicSpline:
    """
    Cubic spline interpolant.

    Attributes
    ----------
    t : array
        Knot x-coordinates (abscissae), must be sorted.
    u : array
        Knot y-coordinates (ordinates).
    h : array
        Differences between consecutive t values (with 0 padding).
    z : array
        Second derivatives at knots.
    extrapolate : bool
        If False, out-of-bounds values raise errors.
        If True, uses flat extrapolation (returns extreme values).
    """
    t: jnp.ndarray
    u: jnp.ndarray
    h: jnp.ndarray
    z: jnp.ndarray
    extrapolate: bool

    def __call__(self, t_eval):
        """
        Evaluate the spline at t_eval.

        Parameters
        ----------
        t_eval : float or array
            Point(s) at which to evaluate the spline.

        Returns
        -------
        float or array
            Interpolated value(s).
        """
        # Check bounds
        if not self.extrapolate:
            if jnp.any((t_eval < self.t[0]) | (t_eval > self.t[-1])):
                raise ValueError(
                    f"Out-of-bounds value passed to interpolant. "
                    f"Must be between {float(self.t[0])} and {float(self.t[-1])}"
                )

        # Handle flat extrapolation
        if self.extrapolate:
            t_eval = jnp.clip(t_eval, self.t[0], self.t[-1])

        # Find the interval containing t_eval
        # searchsortedlast equivalent: largest i such that t[i] <= t_eval
        i = jnp.searchsorted(self.t, t_eval, side='right') - 1
        i = jnp.clip(i, 0, len(self.t) - 2)

        # Evaluate the cubic spline in interval i
        I = (self.z[i] * (self.t[i+1] - t_eval)**3 / (6 * self.h[i+1]) +
             self.z[i+1] * (t_eval - self.t[i])**3 / (6 * self.h[i+1]))
        C = ((self.u[i+1] / self.h[i+1] - self.z[i+1] * self.h[i+1] / 6) *
             (t_eval - self.t[i]))
        D = ((self.u[i] / self.h[i+1] - self.z[i] * self.h[i+1] / 6) *
             (self.t[i+1] - t_eval))

        return I + C + D

    def cumulative_integral(self, t1, t2, num_points=None):
        """
        Compute cumulative integral from t1 to t2.

        Given a curve described by the spline, calculates the integral from t1
        to t for all t = t1, t2, and all spline knots in between.

        Parameters
        ----------
        t1 : float
            Start of integration range.
        t2 : float
            End of integration range.
        num_points : int, optional
            If provided, number of output points. If None, uses knots between t1 and t2.

        Returns
        -------
        array
            Cumulative integral values.
        """
        # Find indices
        idx1 = jnp.searchsorted(self.t, t1, side='right') - 1
        idx1 = jnp.clip(idx1, 0, len(self.t) - 2)

        idx2 = jnp.searchsorted(self.t, t2, side='right') - 1
        idx2 = jnp.clip(idx2, 1, len(self.t) - 2)

        # If t2 exactly equals a knot, move back one interval
        if jnp.isclose(self.t[idx2], t2):
            idx2 = max(1, idx2 - 1)

        # Compute cumulative integral
        out = jnp.zeros(idx2 - idx1 + 2)

        for idx in range(int(idx1), int(idx2) + 1):
            lt1 = t1 if idx == idx1 else self.t[idx]
            lt2 = t2 if idx == idx2 else self.t[idx + 1]

            integral_idx = idx - idx1 + 1
            out = out.at[integral_idx].set(
                out[integral_idx - 1] +
                self._integral(idx, lt2) - self._integral(idx, lt1)
            )

        return out

    def _integral(self, idx, t):
        """
        Compute the definite integral from t[idx] to t in interval idx.

        Parameters
        ----------
        idx : int
            Spline interval index.
        t : float
            Upper limit of integration.

        Returns
        -------
        float
            Integral value.
        """
        t1 = self.t[idx]
        t2 = self.t[idx + 1]
        u1 = self.u[idx]
        u2 = self.u[idx + 1]
        z1 = self.z[idx]
        z2 = self.z[idx + 1]
        h2 = self.h[idx + 1]

        return (
            t**4 * (-z1 + z2) / (24 * h2) +
            t**3 * (-t1 * z2 + t2 * z1) / (6 * h2) +
            t**2 * (h2**2 * z1 - h2**2 * z2 + 3 * t1**2 * z2 - 3 * t2**2 * z1 - 6 * u1 + 6 * u2) / (12 * h2) +
            t * (h2**2 * t1 * z2 - h2**2 * t2 * z1 - t1**3 * z2 - 6 * t1 * u2 + t2**3 * z1 + 6 * t2 * u1) / (6 * h2)
        )


def cubic_spline(t, u, extrapolate=False):
    """
    Construct a cubic spline interpolant.

    Parameters
    ----------
    t : array_like
        Knot x-coordinates (abscissae). Must be sorted.
    u : array_like
        Knot y-coordinates (ordinates).
    extrapolate : bool, optional
        If False (default), out-of-bounds evaluation raises errors.
        If True, uses flat extrapolation.

    Returns
    -------
    CubicSpline
        The interpolant object, callable for evaluation.

    Examples
    --------
    >>> t = jnp.array([0., 1., 2., 3.])
    >>> u = jnp.array([0., 1., 4., 9.])
    >>> spline = cubic_spline(t, u)
    >>> spline(1.5)  # Evaluate at x=1.5
    """
    t = jnp.asarray(t)
    u = jnp.asarray(u)

    n = len(t) - 1

    # Compute differences h
    h_inner = jnp.diff(t)
    h = jnp.concatenate([jnp.array([0.0]), h_inner, jnp.array([0.0])])

    # Build tridiagonal system
    # Lower diagonal
    dl = h[1:n+1]
    # Main diagonal
    d_main = 2.0 * (h[0:n+1] + h[1:n+2])
    # Upper diagonal
    du = h[1:n+1]

    # Right-hand side
    d = jnp.zeros(n + 1, dtype=u.dtype)
    for i in range(1, n):
        d = d.at[i].set(
            6 * (u[i+1] - u[i]) / h[i+1] - 6 * (u[i] - u[i-1]) / h[i]
        )
    # Natural spline boundary conditions (d[0] = d[n] = 0)

    # Solve tridiagonal system: tA @ z = d
    # Construct the full tridiagonal matrix manually
    size = n + 1
    tA = jnp.zeros((size, size))

    # Main diagonal
    tA = tA.at[jnp.arange(size), jnp.arange(size)].set(d_main)
    # Upper diagonal (du has length n, goes in positions [0,1], [1,2], ..., [n-1,n])
    tA = tA.at[jnp.arange(size-1), jnp.arange(1, size)].set(du)
    # Lower diagonal (dl has length n, goes in positions [1,0], [2,1], ..., [n,n-1])
    tA = tA.at[jnp.arange(1, size), jnp.arange(size-1)].set(dl)

    z = solve(tA, d)

    return CubicSpline(t, u, h[0:n+1], z, extrapolate)
