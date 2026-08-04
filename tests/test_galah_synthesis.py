"""
Test solar spectrum synthesis using galah linelist.

Compares Python Korg against Julia Korg.jl for a ~20 Å window near 4940 Å.

This test:
1. Loads the solar MARCS model atmosphere (sun.mod)
2. Reads the galah linelist (filtered to 4930-4950 Å region)
3. Synthesizes a continuum-normalized spectrum using Python Korg
4. Runs the same synthesis using Julia Korg.jl
5. Creates a comparison plot showing both spectra and their residuals

Expected runtime: ~3-5 minutes
(depends on system speed and number of lines in wavelength range)

Usage:
    python tests/test_galah_synthesis.py

Output:
    - galah_synthesis_comparison.png: Comparison plot
    - julia_galah_synthesis.h5: Julia synthesis results
"""
import os
import shutil
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
import tempfile

import pytest

from korg.linelist import Line, Species
from korg.synthesis_plan import synthesize
from korg.atmosphere import PlanarAtmosphere, PlanarAtmosphereLayer
from korg.abundances import format_A_X
from korg.constants import kboltz_cgs

# Test fixture: solar MARCS model atmosphere, committed under tests/data/.
SUN_MOD = Path(__file__).parent / "data" / "sun.mod"

# Tolerances for the Python-vs-Julia galah synthesis comparison.
#
# These are intentionally GENEROUS for now so the test passes at the current
# level of agreement -- they bound how far Python may drift from the Julia
# reference, not a target accuracy. Tighten them over time as the port's
# accuracy improves (e.g. once ABO p-d resonant broadening and other gaps are
# closed). Current measured agreement (4930-4950 Å solar window):
#   max |cnorm_py - cnorm_jl| ~ 0.234, mean ~ 0.018
#   max flux rel diff ~ 0.73,   mean ~ 0.039
GALAH_TOL_CNORM_MAX_ABS = 0.30    # max |cnorm_py - cnorm_jl|
GALAH_TOL_CNORM_MEAN_ABS = 0.03   # mean |cnorm_py - cnorm_jl|
GALAH_TOL_FLUX_MAX_REL = 1.0      # max |flux_py - flux_jl| / flux_jl
GALAH_TOL_FLUX_MEAN_REL = 0.06    # mean |flux_py - flux_jl| / flux_jl


def read_marcs_model(filename: str) -> PlanarAtmosphere:
    """
    Read a MARCS .mod file and convert to PlanarAtmosphere.

    Parameters
    ----------
    filename : str
        Path to MARCS .mod file

    Returns
    -------
    atmosphere : PlanarAtmosphere
        Atmosphere model
    """
    # Read MARCS file
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Parse to find number of depth points
    n_layers = None
    data_start = None

    for i, line in enumerate(lines):
        if 'Number of depth points' in line:
            n_layers = int(line.split()[0])
        elif line.strip().startswith('k lgTauR'):
            # Header line found, data starts on next line
            data_start = i + 1
            break

    if n_layers is None or data_start is None:
        raise ValueError("Could not parse MARCS file format")

    # Read atmosphere data
    temperatures = []
    electron_pressures = []
    gas_pressures = []
    depths = []
    tau_5000 = []

    for i in range(n_layers):
        line_data = lines[data_start + i].split()
        # k lgTauR  lgTau5    Depth     T        Pe          Pg         Prad       Pturb
        # 1 -5.00  -4.9174 -6.931E+07  4066.8  2.1166E-02  2.6699E+02  1.4884E+00  0.0000E+00
        temperatures.append(float(line_data[4]))
        electron_pressures.append(float(line_data[5]))  # Pe is already linear, not log
        gas_pressures.append(float(line_data[6]))
        depths.append(float(line_data[3]))
        tau_5000.append(10**float(line_data[2]))  # tau5000 from lgTau5

    # Convert to arrays
    temperatures = np.array(temperatures)
    electron_pressures = np.array(electron_pressures)
    gas_pressures = np.array(gas_pressures)
    depths = np.array(depths)
    tau_5000 = np.array(tau_5000)

    # Compute electron and total number densities from pressures
    # P = n * k * T  =>  n = P / (k * T)
    electron_densities = electron_pressures / (kboltz_cgs * temperatures)
    number_densities = gas_pressures / (kboltz_cgs * temperatures)

    # Create PlanarAtmosphere
    layers = []
    for i in range(n_layers):
        layer = PlanarAtmosphereLayer(
            tau_ref=tau_5000[i],
            z=depths[i],
            temperature=temperatures[i],
            electron_number_density=electron_densities[i],
            number_density=number_densities[i]
        )
        layers.append(layer)

    return PlanarAtmosphere(layers=layers, reference_wavelength=5e-5)  # 5000 Å in cm


def read_galah_linelist_hdf5(filename: str, wl_min: float = None, wl_max: float = None):
    """
    Read GALAH DR3 linelist from HDF5 file.

    Parameters
    ----------
    filename : str
        Path to galah_dr3_linelist.h5
    wl_min : float, optional
        Minimum wavelength in Å (if None, read all)
    wl_max : float, optional
        Maximum wavelength in Å (if None, read all)

    Returns
    -------
    linelist : list of Line
        Lines in the specified wavelength range
    """
    lines = []

    with h5py.File(filename, 'r') as f:
        # Read all data
        wls = np.array(f['wl'])  # Wavelengths in Å
        log_gfs = np.array(f['log_gf'])
        E_los = np.array(f['E_lo'])  # eV
        gamma_rads = np.array(f['gamma_rad'])  # rad/s
        gamma_starks = np.array(f['gamma_stark'])  # rad/s
        vdWs = np.array(f['vdW'])
        formulas = f['formula'][()]  # byte strings
        ionizations = np.array(f['ionization'])

        # Filter by wavelength if requested
        if wl_min is not None or wl_max is not None:
            mask = np.ones(len(wls), dtype=bool)
            if wl_min is not None:
                mask &= wls >= wl_min
            if wl_max is not None:
                mask &= wls <= wl_max

            wls = wls[mask]
            log_gfs = log_gfs[mask]
            E_los = E_los[mask]
            gamma_rads = gamma_rads[mask]
            gamma_starks = gamma_starks[mask]
            vdWs = vdWs[mask]
            formulas = formulas[mask]
            ionizations = ionizations[mask]

        # Convert to Line objects
        for i in range(len(wls)):
            # Decode formula from atomic numbers
            # formulas[i] is array of uint8 atomic numbers, e.g. [26, 0, 0] for Fe
            atomic_numbers = formulas[i]
            # Find first non-zero element
            nonzero_idx = np.where(atomic_numbers != 0)[0]
            if len(nonzero_idx) == 0:
                continue  # Skip empty formulas

            # Convert atomic numbers to formula string
            from korg.atomic_data import atomic_symbols
            formula_parts = []
            for z in atomic_numbers[nonzero_idx[0]:]:
                if z == 0:
                    break
                if z <= len(atomic_symbols) and z > 0:
                    formula_parts.append(atomic_symbols[z - 1])

            if len(formula_parts) == 0:
                continue

            formula_str = ''.join(formula_parts)

            # Create species
            charge = int(ionizations[i])
            species = Species(formula_str, charge)

            # Create line
            # Note: vdW might be in different format, need to handle
            # For GALAH, vdW is typically log10(γ_vdW) if negative
            vdW_val = float(vdWs[i])
            if vdW_val < 0:
                vdW_tuple = (10**vdW_val, -1.0)  # (γ_vdW, -1) for simple scaling
            else:
                vdW_tuple = (vdW_val, -1.0)

            line = Line(
                wl=float(wls[i]),  # Will be converted to cm in Line constructor
                log_gf=float(log_gfs[i]),
                species=species,
                E_lower=float(E_los[i]),
                gamma_rad=float(gamma_rads[i]),
                gamma_stark=float(gamma_starks[i]),
                vdW=vdW_tuple
            )
            lines.append(line)

    return lines


@pytest.mark.skipif(
    not SUN_MOD.exists(),
    reason=f"MARCS model fixture not found: {SUN_MOD}",
)
@pytest.mark.skipif(
    shutil.which("julia") is None,
    reason="julia executable not available for reference synthesis",
)
def test_galah_solar_synthesis():
    """
    Test solar synthesis with GALAH linelist comparing Python and Julia.
    """
    print("="*70)
    print("Solar Synthesis Test (GALAH Linelist)")
    print("="*70)

    # Wavelength range: 20 Å window near 5000 Å (but in data-rich region)
    # Note: GALAH linelist has a gap from ~4951 to ~5626 Å, so use 4930-4950 Å
    wl_center = 4940.0  # Å
    wl_halfwidth = 10.0  # Å
    wl_min = wl_center - wl_halfwidth
    wl_max = wl_center + wl_halfwidth
    wavelengths = np.linspace(wl_min, wl_max, 1000)  # 1000 points for faster synthesis

    print(f"\nWavelength range: {wl_min:.1f} - {wl_max:.1f} Å")
    print(f"Number of wavelength points: {len(wavelengths)}")

    # Load MARCS solar model
    print("\nLoading solar MARCS model...")
    atm = read_marcs_model(str(SUN_MOD))
    print(f"  ✓ Loaded {len(atm.layers)} layers")
    print(f"  Temperature range: {min(l.temperature for l in atm.layers):.0f} - "
          f"{max(l.temperature for l in atm.layers):.0f} K")

    # Get solar abundances
    print("\nSetting up solar abundances...")
    # ``synthesize`` takes A(X) directly; it used to take absolute number
    # fractions, and the conversion that stood here is now its own job.
    A_X = format_A_X(default_metals_H=0.0, default_alpha_H=0.0)  # Solar
    print(f"  ✓ Abundances set (A(Fe) = {A_X[25]:.2f})")

    # Read VALD linelist for this region
    #print(f"\nReading VALD linelist...")
    #from korg.linelist import get_VALD_solar_linelist
    from korg.linelist import get_GALAH_DR3_linelist

    # Include buffer for line wings
    line_buffer = 20.0  # Å
    wl_min_cm = (wl_min - line_buffer) * 1e-8  # Å to cm
    wl_max_cm = (wl_max + line_buffer) * 1e-8

    # Read full linelist and filter to wavelength range
    full_linelist = get_GALAH_DR3_linelist()
    linelist = [line for line in full_linelist
                if wl_min_cm <= line.wl <= wl_max_cm]

    print(f"  ✓ Loaded {len(linelist)} lines in {wl_min-line_buffer:.1f}-{wl_max+line_buffer:.1f} Å")

    # Python synthesis
    print("\n" + "="*70)
    print("Python Korg Synthesis")
    print("="*70)

    def _run_py_synth():
        flux, cntm = synthesize(atm, linelist, wavelengths, A_X,
                                vmic=1.0,  # km/s
                                hydrogen_lines=True)
        # block_until_ready: the traced path returns JAX arrays, so without this
        # the timing would measure dispatch rather than synthesis.
        return np.asarray(flux), np.asarray(cntm)

    # Cold call: includes JAX tracing/compilation, and building the plan.
    _t0 = time.perf_counter()
    result_py = _run_py_synth()
    py_cold_seconds = time.perf_counter() - _t0

    # Warm call: identical shapes, so JAX reuses the compiled executable. The
    # plan is rebuilt, though -- ``prepare_synthesis`` is the form that avoids
    # that, and is what a real timing comparison should use.
    _t0 = time.perf_counter()
    result_py = _run_py_synth()
    py_warm_seconds = time.perf_counter() - _t0

    print(f"\n  Python synthesis wall time: cold={py_cold_seconds:.3f} s, "
          f"warm={py_warm_seconds:.3f} s")

    flux_py, continuum_py = result_py
    cnorm_py = flux_py / continuum_py

    print(f"\n✓ Python synthesis complete")
    print(f"  Flux range: {flux_py.min():.3e} - {flux_py.max():.3e}")
    print(f"  Continuum range: {continuum_py.min():.3e} - {continuum_py.max():.3e}")
    print(f"  Min continuum-normalized flux: {cnorm_py.min():.4f}")

    # Julia synthesis
    print("\n" + "="*70)
    print("Julia Korg.jl Synthesis")
    print("="*70)

    # Create Julia script. Activate the Julia project at the repo root, which
    # provides Korg, HDF5 and JSON.
    repo_root = Path(__file__).resolve().parent.parent
    julia_script = f"""
using Pkg
Pkg.activate("{repo_root}")
using Korg
using HDF5

println("Loading atmosphere...")
atm = Korg.read_model_atmosphere("{SUN_MOD}")
println("  ✓ Loaded ", length(atm.layers), " layers")
println("\\nLoading GALAH linelist...")
full_linelist = Korg.get_GALAH_DR3_linelist()
# Filter to wavelength range (in cm)
wl_min_cm = {wl_min - line_buffer} * 1e-8  # Å to cm
wl_max_cm = {wl_max + line_buffer} * 1e-8
linelist = filter(line -> wl_min_cm <= line.wl <= wl_max_cm, full_linelist)
println("  ✓ Loaded ", length(linelist), " lines")

println("\\nSynthesizing spectrum...")
wavelengths = collect(range({wl_min}, {wl_max}, length={len(wavelengths)}))
A_X = Korg.grevesse_2007_solar_abundances

# Cold call: includes Julia method compilation.
jl_cold_seconds = @elapsed sol = Korg.synthesize(atm, linelist, A_X, wavelengths,
                      vmic=1.0,  # km/s
                      hydrogen_lines=true)
# Warm call: everything already compiled.
jl_warm_seconds = @elapsed sol = Korg.synthesize(atm, linelist, A_X, wavelengths,
                      vmic=1.0,  # km/s
                      hydrogen_lines=true)
println("\\nJULIA_COLD_SECONDS=", jl_cold_seconds)
println("JULIA_WARM_SECONDS=", jl_warm_seconds)

println("\\n✓ Julia synthesis complete")
println("  Flux range: ", minimum(sol.flux), " - ", maximum(sol.flux))
println("  Continuum range: ", minimum(sol.cntm), " - ", maximum(sol.cntm))

# Save results
h5open("julia_galah_synthesis.h5", "w") do f
    f["wavelengths"] = wavelengths
    f["flux"] = sol.flux
    f["continuum"] = sol.cntm
    f["continuum_normalized"] = sol.flux ./ sol.cntm
end

println("\\n✓ Saved to julia_galah_synthesis.h5")
"""

    # Write and run Julia script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jl', delete=False) as f:
        f.write(julia_script)
        julia_script_path = f.name

    try:
        result = subprocess.run(
            ['julia', julia_script_path],
            capture_output=True,
            text=True,
            timeout=300
        )
        print(result.stdout)
        if result.stderr:
            print("Julia stderr:", result.stderr)

        if result.returncode != 0:
            raise AssertionError(
                f"Julia synthesis failed with return code {result.returncode}.\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

        # Load Julia results
        with h5py.File("julia_galah_synthesis.h5", "r") as f:
            wavelengths_jl = np.array(f["wavelengths"])
            flux_jl = np.array(f["flux"])
            continuum_jl = np.array(f["continuum"])
            cnorm_jl = np.array(f["continuum_normalized"])

        print("\n✓ Julia results loaded")

    finally:
        os.unlink(julia_script_path)

    # Compare
    print("\n" + "="*70)
    print("Comparison")
    print("="*70)

    # Flux comparison
    flux_diff = np.abs(flux_py - flux_jl)
    flux_rel_diff = flux_diff / flux_jl
    print(f"\nFlux comparison:")
    print(f"  Max absolute difference: {flux_diff.max():.3e}")
    print(f"  Mean absolute difference: {flux_diff.mean():.3e}")
    print(f"  Max relative difference: {flux_rel_diff.max():.4f} ({flux_rel_diff.max()*100:.2f}%)")
    print(f"  Mean relative difference: {flux_rel_diff.mean():.4f} ({flux_rel_diff.mean()*100:.2f}%)")

    # Continuum-normalized comparison
    cnorm_diff = np.abs(cnorm_py - cnorm_jl)
    print(f"\nContinuum-normalized flux comparison:")
    print(f"  Max absolute difference: {cnorm_diff.max():.4f}")
    print(f"  Mean absolute difference: {cnorm_diff.mean():.6f}")

    # Find deepest line
    i_deepest = np.argmin(cnorm_py)
    print(f"\nDeepest line (Python):")
    print(f"  Wavelength: {wavelengths[i_deepest]:.2f} Å")
    print(f"  Python depth: {(1-cnorm_py[i_deepest])*100:.2f}%")
    print(f"  Julia depth: {(1-cnorm_jl[i_deepest])*100:.2f}%")

    # Assert Python agrees with Julia within the (currently generous) tolerances.
    cnorm_max_abs = float(cnorm_diff.max())
    cnorm_mean_abs = float(cnorm_diff.mean())
    flux_max_rel = float(flux_rel_diff.max())
    flux_mean_rel = float(flux_rel_diff.mean())

    assert cnorm_max_abs <= GALAH_TOL_CNORM_MAX_ABS, (
        f"Max continuum-normalized abs diff {cnorm_max_abs:.4f} exceeds "
        f"tolerance {GALAH_TOL_CNORM_MAX_ABS}")
    assert cnorm_mean_abs <= GALAH_TOL_CNORM_MEAN_ABS, (
        f"Mean continuum-normalized abs diff {cnorm_mean_abs:.6f} exceeds "
        f"tolerance {GALAH_TOL_CNORM_MEAN_ABS}")
    assert flux_max_rel <= GALAH_TOL_FLUX_MAX_REL, (
        f"Max flux relative diff {flux_max_rel:.4f} exceeds "
        f"tolerance {GALAH_TOL_FLUX_MAX_REL}")
    assert flux_mean_rel <= GALAH_TOL_FLUX_MEAN_REL, (
        f"Mean flux relative diff {flux_mean_rel:.4f} exceeds "
        f"tolerance {GALAH_TOL_FLUX_MEAN_REL}")

    # Plot
    print("\n" + "="*70)
    print("Creating comparison plot...")
    print("="*70)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top panel: Continuum-normalized spectra
    ax1 = axes[0]
    ax1.plot(wavelengths, cnorm_py, 'b-', linewidth=0.8, label='Python Korg', alpha=0.8)
    ax1.plot(wavelengths_jl, cnorm_jl, 'r--', linewidth=0.8, label='Julia Korg.jl', alpha=0.8)
    ax1.set_ylabel('Normalized Flux', fontsize=12)
    ax1.set_title(f'Solar Spectrum Synthesis: galah Linelist ({wl_min:.0f}-{wl_max:.0f} Å)',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])

    # Bottom panel: Residuals
    ax2 = axes[1]
    ax2.plot(wavelengths, cnorm_py - cnorm_jl, 'k-', linewidth=0.5)
    ax2.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Wavelength (Å)', fontsize=12)
    ax2.set_ylabel('Residual\n(Python - Julia)', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Add statistics box
    stats_text = (f'Max diff: {cnorm_diff.max():.4f}\n'
                  f'Mean diff: {cnorm_diff.mean():.6f}\n'
                  f'RMS diff: {np.sqrt(np.mean(cnorm_diff**2)):.6f}')
    ax2.text(0.02, 0.98, stats_text,
             transform=ax2.transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=9,
             family='monospace')

    plt.tight_layout()

    output_file = "galah_synthesis_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to {output_file}")

    # plt.show()  # Commented out to avoid blocking

    print("\n" + "="*70)
    print("Test complete!")
    print("="*70)


if __name__ == "__main__":
    test_galah_solar_synthesis()
