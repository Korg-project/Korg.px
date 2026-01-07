# korg

[![Tests](https://github.com/ajwheeler/Korg.jl/actions/workflows/PythonTests.yml/badge.svg)](https://github.com/ajwheeler/Korg.jl/actions/workflows/PythonTests.yml)

A Python (JAX) implementation of [Korg.jl](https://github.com/ajwheeler/Korg.jl), a package for computing stellar spectra from 1D model atmospheres and linelists assuming local thermodynamic equilibrium.

## Note
This is a research project in development that has used large language models. No guarantee is given (yet) about the accuracy or completeness of the calculations.

## Quick Start

```python
import korg

# Get solar abundances
A_X = korg.format_A_X()

# Interpolate a solar-like atmosphere
atm = korg.interpolate_marcs(5777.0, 4.44, A_X)

# Get a linelist
linelist = korg.get_VALD_solar_linelist()

# Synthesize spectrum
wavelengths, flux, continuum = korg.synthesize(
    atm, linelist, A_X, (5000.0, 5100.0)
)

# Plot
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
plt.plot(wavelengths, flux, 'k-')
plt.xlabel(r'$\lambda$ [Å]')
plt.ylabel('continuum-normalized flux')
plt.show()
```

## Abundances

Abundances use the A(X) format: `A(X) = log10(N_X/N_H) + 12`

```python
# Solar abundances
A_X = korg.format_A_X()

# Metal-poor (-1 dex)
A_X = korg.format_A_X(metals=-1.0)

# Alpha-enhanced
A_X = korg.format_A_X(metals=-0.5, alpha=0.3)

# Custom element abundances
A_X = korg.format_A_X(abundances={"Fe": 7.0, "C": 8.5})
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
