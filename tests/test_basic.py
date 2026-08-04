"""Basic tests for the korg Python wrapper."""

import numpy as np
import pytest

import korg


def test_import():
    """Test that the package imports correctly."""
    assert hasattr(korg, "synthesize")
    assert hasattr(korg, "interpolate_marcs")
    assert hasattr(korg, "format_A_X")


def test_format_A_X_solar():
    """Test solar abundance formatting."""
    A_X = korg.format_A_X()
    assert len(A_X) == 92
    # Hydrogen should be 12.0
    assert np.isclose(A_X[0], 12.0)


def test_format_A_X_metal_poor():
    """Test metal-poor abundance formatting."""
    A_X_solar = korg.format_A_X()
    A_X_poor = korg.format_A_X(default_metals_H=-1.0)

    # Hydrogen should still be 12.0
    assert np.isclose(A_X_poor[0], 12.0)
    # Iron (Z=26, index 25) should be 1 dex lower
    assert np.isclose(A_X_poor[25], A_X_solar[25] - 1.0)


def test_get_solar_abundances():
    """Test convenience function for solar abundances."""
    A_X = korg.get_solar_abundances()
    assert len(A_X) == 92


@pytest.mark.slow
def test_synthesis_basic():
    """Test basic synthesis workflow."""
    # Skip if synthesis functions are not available (e.g., missing data files)
    if korg.synthesize is None or korg.interpolate_marcs is None:
        pytest.skip("Synthesis functions not available (missing data files)")

    # Check if MARCS atmosphere grid is available. Under CI a 0-byte placeholder
    # makes get_marcs_grid_path return None rather than raise, and
    # load_marcs_grid then hands back a single-point dummy grid at Teff=5000 --
    # so testing for None matters as much as catching the exception.
    from korg.marcs_interpolation import get_marcs_grid_path
    try:
        marcs_file = get_marcs_grid_path(auto_download=False)
    except Exception:
        marcs_file = None
    if marcs_file is None or not marcs_file.exists():
        pytest.skip("MARCS atmosphere grid not available (run: python -c \"from korg.marcs_interpolation import get_marcs_grid_path; get_marcs_grid_path(auto_download=True)\")")

    # Get solar abundances
    A_X = korg.format_A_X()

    # Interpolate a solar-like atmosphere
    atm = korg.interpolate_marcs(5777.0, 4.44, A_X)

    # Get a linelist
    linelist = korg.get_VALD_solar_linelist()

    # Synthesize a small wavelength range
    wavelengths_ang = np.linspace(5000.0, 5010.0, 50)
    # ``synthesize`` returns ``(flux, continuum)``; the wavelength grid is the
    # one that was passed in. ``korg.synth`` returns all three.
    wavelengths, flux_abs, continuum = korg.synth(
        atm, linelist, wavelengths_ang, A_X
    )
    flux = np.asarray(flux_abs) / np.asarray(continuum)  # continuum-normalized
    continuum = np.asarray(continuum)

    # Check outputs
    assert len(wavelengths) > 0
    assert len(flux) == len(wavelengths)
    assert len(continuum) == len(wavelengths)

    # Flux should be between 0 and 1 (normalized)
    assert np.all(flux >= 0)
    assert np.all(flux <= 1.1)  # Allow some tolerance


@pytest.mark.slow
def test_species():
    """Test Species creation."""
    fe_neutral = korg.Species("Fe", 0)
    fe_ionized = korg.Species("Fe", 1)

    assert fe_neutral is not None
    assert fe_ionized is not None
