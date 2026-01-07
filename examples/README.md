# Korg Examples

This directory contains example scripts demonstrating how to use Korg for spectral synthesis.

## MARCS Atmosphere Interpolation

### `interpolate_marcs_demo.py`

Demonstrates automatic downloading and interpolation of MARCS model atmospheres.

**Usage:**
```bash
python examples/interpolate_marcs_demo.py
```

**What it does:**
- Checks artifact availability
- Downloads MARCS grid on first run (~380 MB from AWS S3)
- Interpolates atmospheres for:
  - Solar-type star (Teff=5777, logg=4.44)
  - Metal-poor star ([M/H]=-2.0)
  - Giant star (spherical atmosphere)
- Plots temperature structure
- Saves plot to `marcs_solar_temperature.png`

**First run:**
```
Downloading SDSS_MARCS_atmospheres_v2 from AWS S3...
Progress: 100.0%
Download complete. Verifying...
Hash verified. Extracting...
Artifact SDSS_MARCS_atmospheres_v2 installed to ~/.korg/...
```

**Subsequent runs:**
Uses cached data from `~/.korg/` (fast!)

## Requirements

All examples require the package to be installed:
```bash
pip install -e ".[dev]"
```

Some examples may require additional packages:
```bash
pip install matplotlib  # For plotting examples
```

## Data Storage

Large data files are stored in:
- **Default:** `~/.korg/`
- **Custom:** Set `KORG_DATA_DIR` environment variable

Example:
```bash
export KORG_DATA_DIR=/path/to/custom/location
python examples/interpolate_marcs_demo.py
```

## CI/Testing

In CI environments, use placeholders to avoid downloading large files:
```bash
python .github/scripts/setup_ci_artifacts.py
```

This creates 0-byte placeholder files that allow imports to work without actual data.
