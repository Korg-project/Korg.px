"""
Pytest configuration for korg tests.

This file ensures JAX x64 mode is enabled before any tests run, and turns on
JAX's persistent compilation cache so the suite stops recompiling the same
kernels on every invocation.
"""

import os
from pathlib import Path

# Set x64 mode via environment variable BEFORE any imports
os.environ["JAX_ENABLE_X64"] = "true"

# CPU only. JAX picks up a visible CUDA device by default on this hardware, and
# that breaks the suite two ways: the reference comparisons are validated
# against the CPU backend and XLA fuses the float32 Voigt kernel differently on
# GPU, and running several xdist workers exhausts device memory loading the
# MARCS grid (RESOURCE_EXHAUSTED from the BFC allocator). Set before jax is
# imported, and overridable for anyone deliberately testing another backend.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

# ---------------------------------------------------------------------------
# Persistent compilation cache
# ---------------------------------------------------------------------------
# The chemical-equilibrium Newton kernel is the expensive thing to compile in
# this package, and it is recompiled for every distinct layer count, vmap and
# differentiation mode the suite exercises. Caching the compiled executables on
# disk makes a warm run skip all of it: measured 122 s -> 46 s on
# ``test_synthesis_preparation.py`` alone.
#
# This cannot change a result. The cache key is a hash of the HLO module
# together with the compile options and the backend version, so a hit hands
# back the executable that would have been produced anyway.
#
# ``KORG_NO_JAX_CACHE=1`` disables it, for when you are timing cold compiles.
if not os.environ.get("KORG_NO_JAX_CACHE"):
    _cache_dir = os.environ.get(
        "KORG_JAX_CACHE_DIR", str(Path(__file__).resolve().parent.parent / ".jax_cache")
    )
    Path(_cache_dir).mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", _cache_dir)
    # Cache anything that took real time to build. The default floor is high
    # enough that the kernels this suite cares about would be skipped.
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

# Now import korg which will also set x64 mode
import korg  # noqa: E402, F401

import functools  # noqa: E402
import numpy as np  # noqa: E402


# ---------------------------------------------------------------------------
# Plan memoization
# ---------------------------------------------------------------------------
# ``prepare_synthesis`` is a pure function of its arguments that costs ~9 s: it
# interpolates the reference MARCS atmosphere and runs a full chemical
# equilibrium on it so the line windows are measured rather than guessed. The
# suite builds the same plan dozens of times -- ``test_synthesizer_closure``
# alone builds 31, many with identical arguments, and every ``synthesize()``
# call builds and then discards one.
#
# ``Synthesizer`` assigns only in ``__init__`` and is never mutated afterwards,
# so handing back a shared instance is indistinguishable from building a new
# one. The patch is applied at conftest *import* time, before the test modules
# are collected, so that a module-level ``from korg.synthesis_plan import
# prepare_synthesis`` binds the memoized version too.
#
# ``KORG_NO_PLAN_CACHE=1`` disables it.

def _plan_cache_key(value):
    """A hashable stand-in for one ``prepare_synthesis`` argument."""
    if isinstance(value, np.ndarray):
        return ("array", value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, (str, bytes, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        # A linelist is usually written out at the call site -- ``[]`` or
        # ``[fe_line]`` -- so the container is a new object every time even when
        # its contents are the same fixture. Key on the elements, not the list.
        return ("seq", len(value), tuple(_plan_cache_key(v) for v in value))
    # Line objects, LinelistData and SynthesisData are large and are not mutated
    # after construction, so identity is the right granularity. The cache entry
    # holds a reference to the argument, so an id cannot be recycled while the
    # key that mentions it is live.
    return ("id", id(value))


def _install_plan_memo():
    from korg import synthesis_plan

    real = synthesis_plan.prepare_synthesis
    cache = {}

    @functools.wraps(real)
    def prepare_synthesis(wavelengths_angstrom, linelist, data=None, **kwargs):
        try:
            key = (
                _plan_cache_key(np.asarray(wavelengths_angstrom, dtype=np.float64)),
                _plan_cache_key(linelist),
                _plan_cache_key(data),
                tuple(sorted((k, _plan_cache_key(v)) for k, v in kwargs.items())),
            )
        except (TypeError, ValueError):
            # Anything we cannot key on is simply not cached.
            return real(wavelengths_angstrom, linelist, data=data, **kwargs)

        if key not in cache:
            # Built outside the assignment so a rejected argument raises on every
            # call rather than being cached as a success.
            plan = real(wavelengths_angstrom, linelist, data=data, **kwargs)
            cache[key] = (plan, linelist, data)
        return cache[key][0]

    synthesis_plan.prepare_synthesis = prepare_synthesis
    korg.prepare_synthesis = prepare_synthesis


if not os.environ.get("KORG_NO_PLAN_CACHE"):
    _install_plan_memo()
