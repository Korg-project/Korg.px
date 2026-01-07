#!/usr/bin/env python3
"""
Demo script for MARCS atmosphere interpolation.

This script demonstrates how to use Korg's MARCS interpolation,
which automatically downloads atmosphere grids on first use.
"""

import numpy as np
import matplotlib.pyplot as plt

# Import Korg
import korg


def main():
    """Run MARCS interpolation demo."""
    print("=" * 60)
    print("Korg MARCS Atmosphere Interpolation Demo")
    print("=" * 60)

    # Check artifact status
    print("\nChecking artifact availability:")
    status = korg.list_artifacts()
    for name, available in status.items():
        symbol = "✓" if available else "✗"
        print(f"  {symbol} {name}")

    # If MARCS grid not available, it will be downloaded automatically
    print("\nInterpolating solar-type atmosphere (Teff=5777, logg=4.44)...")
    print("Note: First run will download ~380 MB MARCS grid from AWS S3")
    print("Subsequent runs will use cached data from ~/.korg/")

    try:
        # Interpolate a solar-type atmosphere
        atm = korg.interpolate_marcs(
            Teff=5777,      # K
            logg=4.44,      # log10(cm/s²)
            M_H=0.0,        # [M/H] = 0 (solar metallicity)
            alpha_M=0.0,    # [α/M] = 0 (solar alpha enhancement)
            C_M=0.0         # [C/metals] = 0 (solar carbon)
        )

        print(f"\n✓ Interpolation successful!")
        print(f"  Atmosphere type: {type(atm).__name__}")
        print(f"  Number of layers: {len(atm.layers)}")
        print(f"  Reference wavelength: {atm.reference_wavelength * 1e8:.0f} Å")

        # Extract atmospheric structure
        tau_ref = np.array([layer.tau_ref for layer in atm.layers])
        temperatures = np.array([layer.temperature for layer in atm.layers])
        log_tau = np.log10(tau_ref)

        print(f"\n  Temperature range: {temperatures.min():.0f} - {temperatures.max():.0f} K")
        print(f"  log(tau) range: {log_tau.min():.2f} - {log_tau.max():.2f}")

        # Plot temperature structure
        print("\nPlotting temperature structure...")
        plt.figure(figsize=(10, 6))
        plt.plot(log_tau, temperatures, 'b-', linewidth=2)
        plt.xlabel('log(τ₅₀₀₀)', fontsize=12)
        plt.ylabel('Temperature [K]', fontsize=12)
        plt.title('Solar Atmosphere Temperature Structure (MARCS)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot
        output_file = 'marcs_solar_temperature.png'
        plt.savefig(output_file, dpi=150)
        print(f"  Saved plot to {output_file}")

        # Try a metal-poor star
        print("\nInterpolating metal-poor atmosphere (Teff=6000, logg=4.0, [M/H]=-2.0)...")
        atm_poor = korg.interpolate_marcs(
            Teff=6000,
            logg=4.0,
            M_H=-2.0,    # Metal-poor
            alpha_M=0.4, # Alpha-enhanced (typical for halo stars)
            C_M=0.0
        )
        print(f"✓ Metal-poor atmosphere interpolated: {len(atm_poor.layers)} layers")

        # Try a giant star (spherical atmosphere)
        print("\nInterpolating giant star atmosphere (Teff=4500, logg=2.0)...")
        atm_giant = korg.interpolate_marcs(
            Teff=4500,
            logg=2.0,  # Low gravity triggers spherical atmosphere
            M_H=0.0
        )
        print(f"✓ Giant atmosphere interpolated: {type(atm_giant).__name__}")
        if hasattr(atm_giant, 'R_photosphere'):
            print(f"  Photospheric radius: {atm_giant.R_photosphere:.2e} cm")

        print("\n" + "=" * 60)
        print("Demo complete!")
        print("=" * 60)

        return 0

    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nThis may be because:")
        print("  1. You're in a CI environment with placeholder files")
        print("  2. The download failed")
        print("  3. There's insufficient disk space")
        return 1

    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
