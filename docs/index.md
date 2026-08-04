# Korg.px

Korg.px is a Python/JAX implementation of [Korg.jl](https://github.com/ajwheeler/Korg.jl), which
computes stellar spectra from 1D model atmospheres and linelists assuming local thermodynamic
equilibrium.

It targets **Korg.jl v1.2.1**. Every reference fixture in the test suite is generated from that
release, pinned in `Project.toml`.

The physics, the data tables and the numerical schemes are ports of Korg.jl. What is different is
the interface, and the reason it is different is JAX: the synthesis is written so that it can be
`jax.jit`-compiled, `jax.vmap`-ed over stellar parameters, and differentiated end to end with
`jax.grad`. That requirement — array shapes must be known before any parameter is varied — is
what produced [the synthesizer closure](the-synthesizer-closure.md), the one part of the API with
no Korg.jl equivalent.

!!! note
    This is a research project in development that has used large language models. No guarantee is
    given (yet) about the accuracy or completeness of the calculations. See
    [Differences from Korg.jl](differences-from-korg-jl.md) for what is and is not ported.

## Install

Korg.px is not on PyPI. Install from a checkout:

```bash
pip install -e /path/to/Korg.px
```

Requirements are `numpy>=1.24`, `jax>=0.4.20`, `jaxlib>=0.4.20`, `scipy>=1.10` and `h5py>=3.8`
(Python 3.10 or newer). JAX is installed CPU-only by default; follow the
[JAX install instructions](https://docs.jax.dev/en/latest/installation.html) if you want a
CUDA build. Korg.px does not care which backend it gets, and does not assume one.

Importing `korg` sets `jax_enable_x64`, so the synthesis runs in double precision. If something
else in your process has already configured JAX, set `JAX_ENABLE_X64=true` in the environment
before Python starts.

The MARCS model atmosphere grid (~380 MB) is downloaded on first use of
`korg.interpolate_marcs` and cached in `~/.korg/`. Set `KORG_DATA_DIR` to move the cache.

## Quickstart

```python
import numpy as np, korg
from korg.synthesis_plan import prepare_synthesis

wavelengths = np.arange(5000.0, 5005.0, 0.01)                    # Angstroms
linelist = korg.get_VALD_solar_linelist()
synth = prepare_synthesis(wavelengths, linelist, geometry="plane-parallel")
flux, continuum = synth(5777.0, 4.44, 0.0)                       # Teff, log g, [M/H]
```

`synth` is a callable that can be called again with different stellar parameters without
rebuilding anything, and that `jax.jit`, `jax.vmap` and `jax.grad` all accept. Building the plan
is the expensive step; calling it is not.

`prepare_synthesis` trims the linelist to the synthesis range for you, as Korg.jl does — the
`line_buffer_cm` keyword is 10 Å by default. The full 41861-line VALD list is 166 lines against a
5 Å window. See [Getting started](getting-started.md#linelists).

## Where to go next

| Page | Contents |
|---|---|
| [Getting started](getting-started.md) | Abundances, atmospheres, linelists, a solar spectrum, post-processing |
| [The synthesizer closure](the-synthesizer-closure.md) | `prepare_synthesis`, and using `jit`/`vmap`/`grad` |
| [Differences from Korg.jl](differences-from-korg-jl.md) | Every place the Python interface departs from the Julia one |
| [API reference](api-reference.md) | Public functions, signatures, and where they live |
| [`voigt_hjerting` accuracy](voigt-hjerting-accuracy.md) | A known ~15% error inherited from Korg.jl |

Korg.jl's own [documentation](https://ajwheeler.github.io/Korg.jl/stable/) remains the reference
for the physics; this documentation covers the Python interface and where it diverges.

## Citation

If you use this package, please cite the Korg papers:

- [Korg: A Modern 1D LTE Spectral Synthesis Package](https://ui.adsabs.harvard.edu/abs/2023AJ....165...68W/abstract)
- [Korg: fitting, model atmosphere interpolation, and Brackett lines](https://ui.adsabs.harvard.edu/abs/2023arXiv231019823W/abstract)
