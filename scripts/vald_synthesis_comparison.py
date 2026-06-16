"""
Compare Python Korg vs Julia Korg.jl for a 100 Å synthesis with VALD linelist.

Synthesizes wl = linspace(5000, 5100, 2000) with the solar VALD linelist
using both Python and Julia, then creates a comparison plot.
"""
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import tempfile
import time
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import jax.numpy as jnp
import korg
from korg.linelist import read_vald_linelist
from korg.abundances import A_X_to_absolute
from korg.synthesis import (precompute_synthesis_data, preprocess_linelist, synthesize_jit)
from korg.data_loader import (ionization_energies, default_partition_funcs,
                               default_log_equilibrium_constants)

VALD_PATH = os.path.join(os.path.dirname(__file__), '..', 'src', 'korg', 'data',
                         'linelists', 'vald_extract_stellar_solar_threshold001.vald')
ATMOSPHERE_PATH = os.path.join(os.path.dirname(__file__), '..', 'sun.mod')
OUTPUT_H5 = '/tmp/julia_vald_synthesis.h5'
OUTPUT_PNG = os.path.join(os.path.dirname(__file__), '..', 'vald_synthesis_comparison.png')

WL_MIN = 5000.0  # Å
WL_MAX = 5100.0  # Å
N_WL = 2000


def run_python_synthesis():
    print("=== Python Korg Synthesis (synthesize_jit) ===")
    linelist = read_vald_linelist(VALD_PATH)
    print(f"  Loaded {len(linelist)} lines from VALD")

    atm = korg.read_model_atmosphere(ATMOSPHERE_PATH)
    A_X = korg.format_A_X()
    wavelengths = np.linspace(WL_MIN, WL_MAX, N_WL)
    wavelengths_cm = jnp.array(wavelengths * 1e-8)

    # Pre-compute static data (not timed — one-time setup)
    print("  Pre-computing synthesis data...")
    data = precompute_synthesis_data(ionization_energies, default_partition_funcs,
                                     default_log_equilibrium_constants)

    # Filter linelist to synthesis range (+10 Å buffer) before JIT preprocessing.
    # synthesize_jit cannot filter at runtime; all pre-filtered lines are processed.
    line_buffer_ang = 10.0
    wl_lo_cm = (WL_MIN - line_buffer_ang) * 1e-8
    wl_hi_cm = (WL_MAX + line_buffer_ang) * 1e-8
    linelist_filtered = [l for l in linelist if wl_lo_cm <= l.wl <= wl_hi_cm]
    print(f"  Filtered linelist: {len(linelist_filtered)} lines in range")
    linelist_data = preprocess_linelist(linelist_filtered)
    abundances = jnp.array(A_X_to_absolute(A_X))

    T_layers = jnp.array(atm.T)
    n_total = jnp.array(atm.n_total)
    ne_layers = jnp.array(atm.ne)
    z_layers = jnp.array(atm.z)
    log_tau_ref = jnp.array(atm.log_tau_ref)

    # Warmup call to trigger JAX compilation
    print("  Warming up (JIT compile)...")
    flux_w, cont_w = synthesize_jit(
        wavelengths_cm=wavelengths_cm,
        T_layers=T_layers, n_total_layers=n_total, ne_layers=ne_layers,
        z_layers=z_layers, log_tau_ref=log_tau_ref,
        abundances=abundances, vmic_cm_s=1.0e5, data=data, linelist_data=linelist_data
    )
    _ = float(flux_w[0])  # force device sync

    t0 = time.perf_counter()
    flux_jit, cont_jit = synthesize_jit(
        wavelengths_cm=wavelengths_cm,
        T_layers=T_layers, n_total_layers=n_total, ne_layers=ne_layers,
        z_layers=z_layers, log_tau_ref=log_tau_ref,
        abundances=abundances, vmic_cm_s=1.0e5, data=data, linelist_data=linelist_data
    )
    _ = float(flux_jit[0])  # force device sync
    elapsed = time.perf_counter() - t0

    flux = np.array(flux_jit)
    continuum = np.array(cont_jit)
    cnorm = flux / continuum

    print(f"  Elapsed: {elapsed*1000:.1f} ms")
    print(f"  Min normalized flux: {cnorm.min():.4f}")
    return wavelengths, flux, continuum, cnorm, elapsed


def run_julia_synthesis():
    print("\n=== Julia Korg.jl Synthesis ===")

    vald_abs = os.path.abspath(VALD_PATH)
    atm_abs = os.path.abspath(ATMOSPHERE_PATH)

    line_buffer_cm = 10.0 * 1e-8  # 10 Å in cm
    wl_min_cm = WL_MIN * 1e-8
    wl_max_cm = WL_MAX * 1e-8

    julia_script = f"""
using Pkg
Pkg.activate("/tmp/Korg.jl")
using Korg, HDF5

println("  Loading atmosphere...")
atm = Korg.read_model_atmosphere("{atm_abs}")
println("  Loading VALD linelist...")
all_lines = Korg.read_linelist("{vald_abs}", format="vald")
# Pre-filter to synthesis range (matching Python's line_buffer=10Å default)
linelist = filter(l -> ({wl_min_cm} - {line_buffer_cm}) <= l.wl <= ({wl_max_cm} + {line_buffer_cm}), all_lines)
println("  Using ", length(linelist), " lines in range")

A_X = Korg.format_A_X()
wavelengths = collect(range({WL_MIN}, {WL_MAX}, length={N_WL}))

# Warmup call
Korg.synthesize(atm, linelist, A_X, wavelengths, vmic=1.0)

println("  Synthesizing (timed)...")
t0 = time()
sol = Korg.synthesize(atm, linelist, A_X, wavelengths, vmic=1.0)
elapsed = time() - t0
println("  Elapsed: ", round(elapsed*1000, digits=1), " ms")
println("  Min normalized flux: ", minimum(sol.flux ./ sol.cntm))

h5open("{OUTPUT_H5}", "w") do f
    f["wavelengths"] = wavelengths
    f["flux"]        = sol.flux
    f["continuum"]   = sol.cntm
    f["cnorm"]       = sol.flux ./ sol.cntm
    f["elapsed_ms"]  = elapsed * 1000
end
println("  Saved to {OUTPUT_H5}")
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jl', delete=False) as f:
        f.write(julia_script)
        script_path = f.name

    try:
        t0 = time.perf_counter()
        result = subprocess.run(['julia', script_path],
                                capture_output=True, text=True, timeout=600)
        wall = time.perf_counter() - t0
        print(result.stdout)
        if result.returncode != 0:
            print("Julia stderr:", result.stderr[-2000:])
            raise RuntimeError(f"Julia exited {result.returncode}")
    finally:
        os.unlink(script_path)

    import h5py
    with h5py.File(OUTPUT_H5, 'r') as f:
        wl_jl = np.array(f['wavelengths'])
        flux_jl = np.array(f['flux'])
        cntm_jl = np.array(f['continuum'])
        cnorm_jl = np.array(f['cnorm'])
        julia_ms = float(f['elapsed_ms'][()])

    print(f"  Julia internal time: {julia_ms:.1f} ms  (wall {wall:.1f} s incl. startup)")
    return wl_jl, flux_jl, cntm_jl, cnorm_jl, julia_ms


def make_plot(wl_py, cnorm_py, py_ms,
              wl_jl, cnorm_jl, julia_ms):
    diff = cnorm_py - np.interp(wl_py, wl_jl, cnorm_jl)
    abs_diff = np.abs(diff)
    rms = np.sqrt(np.mean(diff**2))
    speedup = julia_ms / (py_ms * 1000)

    fig, axes = plt.subplots(2, 1, figsize=(18, 4),
                             gridspec_kw={'height_ratios': [1, 3]},
                             sharex=True)
    fig.subplots_adjust(hspace=0.04)

    # Spectrum panel (bottom)
    ax = axes[1]
    ax.plot(wl_jl, cnorm_jl, '-', color='#333333', lw=0.6)
    ax.plot(wl_py, cnorm_py, '-', color='#999999', lw=0.5)
    # Direct labels — no legend box
    ax.text(WL_MAX - 0.4, 1.058, f'Korg.jl  {julia_ms:.0f} ms',
            ha='right', va='center', fontsize=8, color='#333333')
    ax.text(WL_MAX - 0.4, 1.038, f'Korg.py  {py_ms*1000:.0f} ms  ·  {speedup:.1f}× faster',
            ha='right', va='center', fontsize=8, color='#999999')
    ax.set_ylabel('Normalized flux')
    ax.set_ylim(-0.02, 1.09)
    ax.set_xlim(WL_MIN, WL_MAX)
    ax.set_xlabel('Wavelength (Å)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Residual panel (top) — auto-scaled to actual differences
    ax = axes[0]
    ax.plot(wl_py, diff, '-', color='#555555', lw=0.5)
    ax.axhline(0, color='#cccccc', lw=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(bottom=False)
    ax.set_ylabel('Δ flux', fontsize=8)
    ax.autoscale_view()
    ymin, ymax = ax.get_ylim()
    ax.text(WL_MAX - 0.4, ymax,
            f'RMS={rms:.4f}  max|Δ|={abs_diff.max():.4f}',
            ha='right', va='top', fontsize=7, color='#777777', family='monospace')

    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {OUTPUT_PNG}")
    return fig


if __name__ == '__main__':
    wl_py, flux_py, cntm_py, cnorm_py, py_elapsed = run_python_synthesis()
    wl_jl, flux_jl, cntm_jl, cnorm_jl, julia_ms = run_julia_synthesis()
    make_plot(wl_py, cnorm_py, py_elapsed,
              wl_jl, cnorm_jl, julia_ms)
    print("\nDone.")
