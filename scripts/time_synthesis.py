#!/usr/bin/env python
"""Time 10 A of solar spectrum through the ``prepare_synthesis`` closure.

Synthesizes 4995-5005 A for the Sun three ways and prints the wall time of each:

    1. the plan itself (``prepare_synthesis``)
    2. the closure called directly
    3. the closure wrapped in ``jax.jit`` -- first call, which includes the XLA
       compile, then a second call, which does not

Usage:  python scripts/time_synthesis.py
"""

import os

# CPU only, before anything imports JAX.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"

import time

import jax
import numpy as np

from korg.data_loader import load_default_linelist
from korg.synthesis_plan import prepare_synthesis

TEFF, LOGG, M_H = 5777.0, 4.44, 0.0
WAVELENGTHS = np.linspace(4995.0, 5005.0, 1001)   # 10 A at 0.01 A


def timed(label, fn):
    start = time.perf_counter()
    flux, cntm = fn()
    flux.block_until_ready()
    elapsed = time.perf_counter() - start
    print(f"{label:<34s} {elapsed:8.2f} s")
    return elapsed, np.asarray(flux), np.asarray(cntm)


def main():
    linelist = load_default_linelist(5e-5)
    print(f"{len(WAVELENGTHS)} wavelengths, {len(linelist)} lines in the default list\n")

    start = time.perf_counter()
    plan = prepare_synthesis(WAVELENGTHS, linelist, geometry="planar")
    print(f"{'prepare_synthesis':<34s} {time.perf_counter() - start:8.2f} s")
    print(f"{'':<34s}          {plan!r}\n")

    _, eager_flux, _ = timed("closure, called directly", lambda: plan(TEFF, LOGG, M_H))

    jitted = jax.jit(plan)
    timed("jax.jit(closure), first call", lambda: jitted(TEFF, LOGG, M_H))
    _, jit_flux, _ = timed("jax.jit(closure), second call",
                           lambda: jitted(TEFF, LOGG + 1e-9, M_H))

    rel = np.max(np.abs(jit_flux - eager_flux) / eager_flux)
    print(f"\nmax relative difference, jit vs eager: {rel:.2e}")


if __name__ == "__main__":
    main()
