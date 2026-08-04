# korg

[![Tests](https://github.com/Korg-project/Korg.px/actions/workflows/PythonTests.yml/badge.svg)](https://github.com/Korg-project/Korg.px/actions/workflows/PythonTests.yml)

A Python (JAX) implementation of [Korg.jl](https://github.com/ajwheeler/Korg.jl), a package for computing stellar spectra from 1D model atmospheres and linelists assuming local thermodynamic equilibrium.

**Target version: Korg.jl v1.2.1.** Every reference fixture the test suite compares against is generated
from that release, pinned exactly in `Project.toml`. To regenerate them all:

```bash
./run_tests.sh --regen
```

## Note
This is a research project in development that has used large language models. No guarantee is given (yet) about the accuracy or completeness of the calculations.

## Quick Start

```python
import numpy as np
import korg
from korg.synthesis_plan import synthesize

# Wavelengths are an explicit array in Angstroms, not a (start, stop) tuple
wavelengths = np.arange(5000.0, 5100.0, 0.01)

# Get solar abundances
A_X = korg.format_A_X()

# Interpolate a solar-like atmosphere
atm = korg.interpolate_marcs(5777.0, 4.44, A_X)

# Get a linelist. It is trimmed to the synthesis range for you, as in
# Korg.jl -- pass line_buffer=None to keep every line.
linelist = korg.get_VALD_solar_linelist()

# Synthesize spectrum
flux, continuum = synthesize(atm, linelist, wavelengths, A_X)

# Plot
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
plt.plot(wavelengths, flux / continuum, 'k-')
plt.xlabel(r'$\lambda$ [Å]')
plt.ylabel('continuum-normalized flux')
plt.show()
```

Synthesizing more than once? Build a plan and reuse it — see
[docs/the-synthesizer-closure.md](docs/the-synthesizer-closure.md).

```python
from korg.synthesis_plan import prepare_synthesis
synth = prepare_synthesis(wavelengths, linelist, geometry="plane-parallel")
flux, continuum = synth(5777.0, 4.44, 0.0)     # jit-able, differentiable
```

## Abundances

Abundances use the A(X) format: `A(X) = log10(N_X/N_H) + 12`

```python
# Solar abundances
A_X = korg.format_A_X()

# Metal-poor: [metals/H] = -1
A_X = korg.format_A_X(-1.0)

# Alpha-enhanced. The second argument is [alpha/H], not [alpha/M], so
# [metals/H] = -0.5 enhanced by 0.3 dex is -0.2.
A_X = korg.format_A_X(-0.5, -0.2)

# Custom element abundances. These are [X/H] by default -- pass
# solar_relative=False to give A(X) values instead.
A_X = korg.format_A_X(abundances={"Fe": -0.3, "C": 0.2})
A_X = korg.format_A_X(abundances={"Fe": 7.2, "C": 8.2}, solar_relative=False)
```


## Documentation

- [Korg.jl documentation](https://ajwheeler.github.io/Korg.jl/stable/)
- [Korg.jl API reference](https://ajwheeler.github.io/Korg.jl/stable/API/)

## Citation

If you use this package, please cite:
- [Korg: A Modern 1D LTE Spectral Synthesis Package](https://ui.adsabs.harvard.edu/abs/2023AJ....165...68W/abstract)
- [Korg: fitting, model atmosphere interpolation, and Brackett lines](https://ui.adsabs.harvard.edu/abs/2023arXiv231019823W/abstract)

## Getting Help

If you have trouble using or installing korg, please [open a GitHub issue](https://github.com/Korg-project/Korg.px/issues).
