# korg

[![Tests](https://github.com/ajwheeler/Korg.jl/actions/workflows/PythonTests.yml/badge.svg)](https://github.com/ajwheeler/Korg.jl/actions/workflows/PythonTests.yml)

Python wrapper for [Korg.jl](https://github.com/ajwheeler/Korg.jl), a package for computing stellar spectra from 1D model atmospheres and linelists assuming local thermodynamic equilibrium.

## Features

- Compute spectra from Teff, logg, abundances, etc.
- Model atmosphere interpolation (MARCS)
- Linelist parsing (VALD, Kurucz, MOOG, ExoMol, Turbospectrum formats)
- Synthesis with arbitrary abundances/solar abundance scales

## Installation

```bash
pip install korg
```

This will automatically install Julia and Korg.jl via `juliacall` on first import.

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

## Multithreading

Korg.jl uses multithreading to speed up line opacity calculation. Set the number of threads via the `JULIA_NUM_THREADS` environment variable before importing:

```bash
export JULIA_NUM_THREADS=4
python your_script.py
```

## Documentation

- [Korg.jl documentation](https://ajwheeler.github.io/Korg.jl/stable/)
- [API reference](https://ajwheeler.github.io/Korg.jl/stable/API/)

## Citation

If you use this package, please cite:
- [Korg: A Modern 1D LTE Spectral Synthesis Package](https://ui.adsabs.harvard.edu/abs/2023AJ....165...68W/abstract)
- [Korg: fitting, model atmosphere interpolation, and Brackett lines](https://ui.adsabs.harvard.edu/abs/2023arXiv231019823W/abstract)

## Getting Help

If you have trouble using or installing korg, please [open a GitHub issue](https://github.com/ajwheeler/Korg.jl/issues).
