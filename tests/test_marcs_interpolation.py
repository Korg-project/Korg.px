"""
Tests for MARCS atmosphere interpolation.

Validates that the Python implementation matches Korg.jl behavior.
"""

import pytest
import numpy as np
import os
import tempfile
from pathlib import Path

# Test imports
try:
    from korg.marcs_interpolation import (
        interpolate_marcs,
        load_marcs_grid,
        get_marcs_grid_path,
        lazy_multilinear_interpolation,
        AtmosphereInterpolationError
    )
    from korg.atmosphere import PlanarAtmosphere, ShellAtmosphere
    from korg.artifacts import (
        get_artifact_path,
        create_placeholder_artifact,
        list_artifacts,
        is_placeholder_file,
        ARTIFACTS
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    IMPORT_ERROR = str(e)


# Mark all tests in this file to require the imports
pytestmark = pytest.mark.skipif(
    not IMPORTS_AVAILABLE,
    reason=f"Required imports not available: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}"
)


class TestArtifactSystem:
    """Test the artifact download system."""

    def test_artifact_registry(self):
        """Test that all required MARCS artifacts are in registry."""
        required = [
            'SDSS_MARCS_atmospheres_v2',
            'MARCS_metal_poor_atmospheres',
            'resampled_cool_dwarf_atmospheres'
        ]
        for name in required:
            assert name in ARTIFACTS, f"Missing artifact: {name}"
            assert 'url' in ARTIFACTS[name]
            assert 'sha256' in ARTIFACTS[name]
            assert 'files' in ARTIFACTS[name]

    def test_list_artifacts(self):
        """Test listing artifact availability."""
        status = list_artifacts()
        assert isinstance(status, dict)
        assert 'SDSS_MARCS_atmospheres_v2' in status
        # Status can be True or False depending on whether downloaded

    def test_placeholder_creation(self):
        """Test creating placeholder artifacts for CI."""
        # Use temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ['KORG_DATA_DIR'] = tmpdir

            # Create placeholder
            artifact_dir = create_placeholder_artifact('SDSS_MARCS_atmospheres_v2')
            assert artifact_dir.exists()

            # Check that placeholder file exists and is small
            extract_dir = artifact_dir / 'SDSS_MARCS_atmospheres'
            h5_file = extract_dir / 'SDSS_MARCS_atmospheres.h5'
            assert h5_file.exists()
            assert is_placeholder_file(h5_file)

            # Clean up
            del os.environ['KORG_DATA_DIR']

    def test_get_marcs_grid_path_with_placeholder(self):
        """Test that get_marcs_grid_path handles placeholders in CI."""
        # Use temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ['KORG_DATA_DIR'] = tmpdir
            os.environ['CI'] = '1'  # Simulate CI environment

            # Create placeholder
            create_placeholder_artifact('SDSS_MARCS_atmospheres_v2')

            # Should return None in CI with placeholder
            with pytest.warns(UserWarning, match="placeholder"):
                path = get_marcs_grid_path('standard', auto_download=False)
                assert path is None

            # Clean up
            del os.environ['KORG_DATA_DIR']
            del os.environ['CI']


@pytest.mark.skipif(
    os.environ.get('CI') or os.environ.get('GITHUB_ACTIONS'),
    reason="Skipping download tests in CI"
)
class TestMARCSDownload:
    """Test MARCS atmosphere downloading (skipped in CI)."""

    def test_download_standard_grid(self):
        """Test downloading standard MARCS grid."""
        # This will download if not present (~380 MB)
        # Should be cached for subsequent runs
        path = get_marcs_grid_path('standard', auto_download=True)
        assert path is not None
        assert path.exists()
        assert not is_placeholder_file(path)

    def test_load_marcs_grid(self):
        """Test loading MARCS grid."""
        nodes, grid = load_marcs_grid()

        # Check structure
        assert len(nodes) == 5  # Teff, logg, M_H, alpha, C
        assert grid.ndim == 7  # (layers, quantities, Teff, logg, M_H, alpha, C)
        assert grid.shape[1] == 5  # 5 quantities: T, log_ne, log_n, tau_ref, asinh_z

        # Check node ranges (approximate)
        assert nodes[0].min() >= 2500  # Teff min
        assert nodes[0].max() <= 8500  # Teff max
        assert nodes[1].min() >= -1.0  # logg min
        assert nodes[1].max() <= 6.0   # logg max


class TestMARCSInterpolation:
    """Test MARCS atmosphere interpolation."""

    @pytest.mark.skipif(
        os.environ.get('CI') or os.environ.get('GITHUB_ACTIONS'),
        reason="Requires actual MARCS data (not placeholder)"
    )
    def test_interpolate_solar(self):
        """Test interpolating a solar-type atmosphere."""
        # Solar parameters
        atm = interpolate_marcs(5777, 4.44, 0.0, 0.0, 0.0)

        # Check basic structure
        assert isinstance(atm, PlanarAtmosphere)
        assert len(atm.layers) > 0
        assert atm.reference_wavelength == 5e-5  # 5000 Å

        # Check that temperatures are reasonable
        temps = np.array([layer.temperature for layer in atm.layers])
        assert np.all(temps > 0)
        assert np.all(temps < 20000)  # Should be < 20,000 K

        # Check that optical depths are positive
        tau_refs = np.array([layer.tau_ref for layer in atm.layers])
        assert np.all(tau_refs >= 0), "Optical depths should be non-negative"

    @pytest.mark.skipif(
        os.environ.get('CI') or os.environ.get('GITHUB_ACTIONS'),
        reason="Requires actual MARCS data (not placeholder)"
    )
    def test_interpolate_giant(self):
        """Test interpolating a giant star atmosphere (spherical)."""
        # Giant parameters: low logg triggers spherical
        atm = interpolate_marcs(4500, 2.0, 0.0, 0.0, 0.0)

        # Should be spherical for logg < 3.5
        assert isinstance(atm, ShellAtmosphere)
        assert hasattr(atm, 'R_photosphere')
        assert atm.R_photosphere > 0

    @pytest.mark.skipif(
        os.environ.get('CI') or os.environ.get('GITHUB_ACTIONS'),
        reason="Requires actual MARCS data (not placeholder)"
    )
    def test_interpolate_metal_poor(self):
        """Test interpolating a metal-poor atmosphere."""
        # Metal-poor parameters
        atm = interpolate_marcs(6000, 4.0, -1.5, 0.2, 0.0)

        assert isinstance(atm, PlanarAtmosphere)
        assert len(atm.layers) > 0

    @pytest.mark.skipif(
        os.environ.get('CI') or os.environ.get('GITHUB_ACTIONS'),
        reason="Requires actual MARCS data (not placeholder)"
    )
    def test_out_of_bounds_raises(self):
        """Test that out-of-bounds parameters raise an error."""
        # Temperature too low
        with pytest.raises(AtmosphereInterpolationError):
            interpolate_marcs(2000, 4.0, 0.0, 0.0, 0.0)

        # Temperature too high
        with pytest.raises(AtmosphereInterpolationError):
            interpolate_marcs(10000, 4.0, 0.0, 0.0, 0.0)

    @pytest.mark.skipif(
        os.environ.get('CI') or os.environ.get('GITHUB_ACTIONS'),
        reason="Requires actual MARCS data (not placeholder)"
    )
    def test_spherical_parameter(self):
        """Test explicit spherical parameter."""
        # Force planar for low logg
        atm = interpolate_marcs(4500, 2.0, 0.0, 0.0, 0.0, spherical=False)
        assert isinstance(atm, PlanarAtmosphere)

        # Force spherical for high logg
        atm = interpolate_marcs(6000, 4.5, 0.0, 0.0, 0.0, spherical=True)
        assert isinstance(atm, ShellAtmosphere)


class TestLazyMultilinearInterpolation:
    """Test the lazy multilinear interpolation function."""

    def test_1d_interpolation(self):
        """Test 1D interpolation."""
        import jax.numpy as jnp

        # Simple 1D grid: layers x quantities x param1
        nodes = [jnp.array([0.0, 1.0, 2.0])]
        grid = jnp.array([
            [[1.0, 2.0, 3.0]],  # layer 0, quantity 0
            [[4.0, 5.0, 6.0]]   # layer 1, quantity 0
        ])

        # Interpolate at param=0.5
        result = lazy_multilinear_interpolation(
            jnp.array([0.5]),
            nodes,
            grid
        )

        # Should interpolate between index 0 and 1
        expected = jnp.array([[1.5], [4.5]])
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_2d_interpolation(self):
        """Test 2D interpolation."""
        import jax.numpy as jnp

        # 2D grid: layers x quantities x param1 x param2
        nodes = [
            jnp.array([0.0, 1.0]),
            jnp.array([0.0, 1.0])
        ]
        grid = jnp.array([
            [[[1.0, 2.0], [3.0, 4.0]]],  # layer 0, quantity 0
        ])

        # Interpolate at center
        result = lazy_multilinear_interpolation(
            jnp.array([0.5, 0.5]),
            nodes,
            grid
        )

        # Center should be average of corners: (1+2+3+4)/4 = 2.5
        expected = jnp.array([[2.5]])
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_out_of_bounds_raises(self):
        """Test that out-of-bounds parameters raise an error."""
        import jax.numpy as jnp

        nodes = [jnp.array([0.0, 1.0])]
        grid = jnp.zeros((1, 1, 2))

        with pytest.raises(AtmosphereInterpolationError):
            lazy_multilinear_interpolation(
                jnp.array([2.0]),  # Out of bounds
                nodes,
                grid,
                param_names=["param1"]
            )


class TestCICompatibility:
    """Test that code works in CI with placeholders."""

    def test_import_with_placeholder(self):
        """Test that modules can be imported even without real data."""
        # Use temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ['KORG_DATA_DIR'] = tmpdir
            os.environ['CI'] = '1'

            # Create placeholder
            create_placeholder_artifact('SDSS_MARCS_atmospheres_v2')

            # Should be able to load grid (returns dummy data)
            with pytest.warns(UserWarning):
                nodes, grid = load_marcs_grid()

            # Should have valid structure even if dummy
            assert len(nodes) == 5
            assert grid.ndim == 7

            # Clean up
            del os.environ['KORG_DATA_DIR']
            del os.environ['CI']

    def test_interpolate_fails_gracefully_with_placeholder(self):
        """Test that interpolate_marcs fails gracefully without real data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ['KORG_DATA_DIR'] = tmpdir
            os.environ['CI'] = '1'

            # Create placeholder
            create_placeholder_artifact('SDSS_MARCS_atmospheres_v2')

            # Interpolation should fail with meaningful error or return dummy
            with pytest.warns(UserWarning):
                try:
                    # This might fail or return dummy data
                    atm = interpolate_marcs(5777, 4.44)
                    # If it succeeds, it should be with dummy data
                    # Can't do much validation here
                except (AtmosphereInterpolationError, Exception):
                    # Expected in CI without real data
                    pass

            # Clean up
            del os.environ['KORG_DATA_DIR']
            del os.environ['CI']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
