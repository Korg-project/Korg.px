#!/usr/bin/env python3
"""
Demo script for MARCS atmosphere interpolation.

This script demonstrates how to use Korg's MARCS interpolation.
Note: This demo requires MARCS atmosphere data to be available.

CURRENT STATUS: The AWS S3 bucket 'korg-data' is not publicly accessible yet.
This needs to be set up before the demo can run successfully.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

# Import Korg
import korg


def create_manual_atmosphere_demo():
    """
    Demonstrate atmosphere creation without MARCS interpolation.

    This shows how to manually create a PlanarAtmosphere object
    until the MARCS grid data becomes available.
    """
    print("=" * 60)
    print("Korg Manual Atmosphere Creation Demo")
    print("=" * 60)
    print()
    print("Note: MARCS interpolation requires AWS S3 data that is")
    print("not currently accessible. This demo shows manual creation.")
    print()

    # Create a simple atmosphere structure
    # Based on typical solar values
    n_layers = 56

    # Optical depth scale (log-spaced from -5 to 2)
    log_tau = np.linspace(-5, 2, n_layers)
    tau_ref = 10**log_tau

    # Temperature structure (simple model)
    # T(tau) relationship for solar-type star
    T_eff = 5777  # K
    temperatures = T_eff * (0.5 + 0.75 * tau_ref)**(1/4)

    # Simple pressure structure (hydrostatic equilibrium approximation)
    logg = 4.44  # log10(cm/s²)
    g = 10**logg

    # Create atmosphere layers
    from korg.atmosphere import PlanarAtmosphere, PlanarAtmosphereLayer

    # Estimate heights using hydrostatic equilibrium
    # dz/dtau = -(kT/mu*m_H*g)
    heights = np.zeros(n_layers)
    for i in range(1, n_layers):
        # Simple height scale estimation
        H = 1.38e-16 * temperatures[i] / (2.3 * 1.67e-24 * g)  # Scale height
        dz = H * np.log(tau_ref[i] / tau_ref[i-1])
        heights[i] = heights[i-1] + dz

    layers = []
    for i in range(n_layers):
        layer = PlanarAtmosphereLayer(
            tau_ref=float(tau_ref[i]),
            z=float(heights[i]),
            temperature=float(temperatures[i]),
            electron_number_density=1e14,  # Placeholder
            number_density=1e17,  # Placeholder
        )
        layers.append(layer)

    # Create atmosphere object
    atm = PlanarAtmosphere(
        layers=layers,
        reference_wavelength=5000e-8,  # 5000 Angstrom in cm
    )

    print(f"Created manual atmosphere:")
    print(f"  Type: {type(atm).__name__}")
    print(f"  Number of layers: {len(atm.layers)}")
    print(f"  Reference wavelength: {atm.reference_wavelength * 1e8:.0f} Å")
    print(f"  Temperature range: {temperatures.min():.0f} - {temperatures.max():.0f} K")
    print(f"  log(tau) range: {log_tau.min():.2f} - {log_tau.max():.2f}")

    # Plot temperature structure
    print("\nPlotting temperature structure...")
    plt.figure(figsize=(10, 6))
    plt.plot(log_tau, temperatures, 'b-', linewidth=2, label='Manual model')
    plt.xlabel('log(τ₅₀₀₀)', fontsize=12)
    plt.ylabel('Temperature [K]', fontsize=12)
    plt.title('Solar-Type Atmosphere Temperature Structure (Manual)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Save plot
    output_file = 'manual_solar_temperature.png'
    plt.savefig(output_file, dpi=150)
    print(f"Saved plot to {output_file}")

    return 0


def try_marcs_interpolation():
    """Try MARCS interpolation (will fail without S3 data)."""
    print("=" * 60)
    print("Korg MARCS Atmosphere Interpolation Demo")
    print("=" * 60)

    # Check artifact status
    print("\nChecking artifact availability:")
    status = korg.list_artifacts()
    for name, available in status.items():
        symbol = "✓" if available else "✗"
        print(f"  {symbol} {name}")

    marcs_available = status.get('SDSS_MARCS_atmospheres_v2', False)

    if not marcs_available:
        print("\n" + "!" * 60)
        print("MARCS atmosphere data is not available!")
        print("!" * 60)
        print()
        print("The MARCS interpolation requires atmosphere grid data")
        print("from AWS S3. This data is not currently accessible.")
        print()
        print("To fix this issue:")
        print("  1. Create an S3 bucket named 'korg-data'")
        print("  2. Upload the MARCS atmosphere grid files")
        print("  3. Make the bucket publicly readable")
        print()
        print("Expected S3 URL:")
        print("  https://korg-data.s3.amazonaws.com/SDSS_MARCS_atmospheres_v2.tar.gz")
        print()
        print("For now, running manual atmosphere demo instead...")
        print()
        return create_manual_atmosphere_demo()

    # If data is available, try interpolation
    print("\nInterpolating solar-type atmosphere (Teff=5777, logg=4.44)...")

    try:
        atm = korg.interpolate_marcs(
            Teff=5777,
            logg=4.44,
            M_H=0.0,
            alpha_M=0.0,
            C_M=0.0
        )

        print(f"\n✓ Interpolation successful!")
        print(f"  Atmosphere type: {type(atm).__name__}")
        print(f"  Number of layers: {len(atm.layers)}")

        return 0

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nFalling back to manual atmosphere demo...")
        return create_manual_atmosphere_demo()


def main():
    """Run demo."""
    return try_marcs_interpolation()


if __name__ == '__main__':
    sys.exit(main())
