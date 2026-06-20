"""
Compare Python Korg vs Julia Korg.jl for a 100 Å synthesis with VALD linelist.

Synthesizes wl = linspace(5000, 5100, 2000) with the solar VALD linelist
using both Python (JIT and non-JIT) and Julia, then creates a six-panel
comparison plot.
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
from korg.abundances import A_X_to_absolute, format_A_X
from korg.synthesis import (precompute_synthesis_data, preprocess_linelist,
                             synthesize_jit, synthesize,
                             precompute_atmosphere, PrecomputedAtmosphereData)
from korg.data_loader import (ionization_energies, default_partition_funcs,
                               default_log_equilibrium_constants)

VALD_PATH = os.path.join(os.path.dirname(__file__), '..', 'src', 'korg', 'data',
                         'linelists', 'vald_extract_stellar_solar_threshold001.vald')
ATMOSPHERE_PATH = os.path.join(os.path.dirname(__file__), '..', 'sun.mod')
OUTPUT_H5 = '/tmp/julia_vald_synthesis.h5'
OUTPUT_PNG = os.path.join(os.path.dirname(__file__), '..', 'vald_synthesis_comparison.png')

WL_MIN = 5100.0  # Å
WL_MAX = 5200.0  # Å
N_WL = 2_000


def run_python_synthesis_jit():
    print("=== Python Korg Synthesis (synthesize_jit) ===")
    linelist = read_vald_linelist(VALD_PATH)
    print(f"  Loaded {len(linelist)} lines from VALD")

    atm = korg.read_model_atmosphere(ATMOSPHERE_PATH)
    A_X = korg.format_A_X()
    wavelengths = np.linspace(WL_MIN, WL_MAX, N_WL)
    wavelengths_cm = jnp.array(wavelengths * 1e-8)

    print("  Pre-computing synthesis data...")
    data = precompute_synthesis_data(ionization_energies, default_partition_funcs,
                                     default_log_equilibrium_constants)

    line_buffer_ang = 10.0
    wl_lo_cm = (WL_MIN - line_buffer_ang) * 1e-8
    wl_hi_cm = (WL_MAX + line_buffer_ang) * 1e-8
    linelist_filtered = [l for l in linelist if wl_lo_cm <= l.wl <= wl_hi_cm]
    print(f"  Filtered linelist: {len(linelist_filtered)} lines in range")
    linelist_data = preprocess_linelist(
        linelist_filtered, chem_eq_data=data.chem_eq_data,
        wavelengths_cm=np.asarray(wavelengths_cm)
    )
    abundances = jnp.array(A_X_to_absolute(A_X))

    T_layers = jnp.array(atm.T)
    n_total = jnp.array(atm.n_total)
    ne_layers = jnp.array(atm.ne)
    z_layers = jnp.array(atm.z)
    log_tau_ref = jnp.array(atm.log_tau_ref)

    print("  Warming up (JIT compile)...")
    kw = dict(wavelengths_cm=wavelengths_cm, T_layers=T_layers, n_total_layers=n_total,
              ne_layers=ne_layers, z_layers=z_layers, log_tau_ref=log_tau_ref,
              abundances=abundances, vmic_cm_s=1.0e5, data=data, linelist_data=linelist_data)
    _ = float(synthesize_jit(**kw)[0][0])
    _ = float(synthesize_jit(**kw)[0][0])

    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        flux_jit, cont_jit = synthesize_jit(**kw)
        _ = float(flux_jit[0])
        times.append(time.perf_counter() - t0)
    elapsed = min(times)

    flux = np.array(flux_jit)
    continuum = np.array(cont_jit)
    cnorm = flux / continuum

    print(f"  Elapsed: {elapsed*1000:.1f} ms")
    print(f"  Min normalized flux: {cnorm.min():.4f}")

    # --- Precomputed atmosphere workflow ---
    print("\n  Pre-computing atmosphere (one-time cost)...")
    atm_precomputed = precompute_atmosphere(
        wavelengths_cm, T_layers, n_total, ne_layers, z_layers, log_tau_ref,
        abundances, 1.0e5, data, linelist_data
    )
    print("  Timing synthesize_jit WITH precomputed atmosphere...")
    kw_fast = dict(wavelengths_cm=wavelengths_cm, T_layers=T_layers, n_total_layers=n_total,
                   ne_layers=ne_layers, z_layers=z_layers, log_tau_ref=log_tau_ref,
                   abundances=abundances, vmic_cm_s=1.0e5, data=data,
                   linelist_data=linelist_data, precomputed_atm=atm_precomputed)
    _ = float(synthesize_jit(**kw_fast)[0][0])
    _ = float(synthesize_jit(**kw_fast)[0][0])
    times_fast = []
    for _ in range(3):
        t0 = time.perf_counter()
        flux_fast, cont_fast = synthesize_jit(**kw_fast)
        _ = float(flux_fast[0])
        times_fast.append(time.perf_counter() - t0)
    elapsed_fast = min(times_fast)
    print(f"  Elapsed (precomputed): {elapsed_fast*1000:.1f} ms  ({elapsed/elapsed_fast:.1f}x vs non-precomputed)")

    return wavelengths, flux, continuum, cnorm, elapsed, elapsed_fast


def run_python_synthesis_nonjit():
    print("\n=== Python Korg Synthesis (synthesize, non-JIT) ===")
    linelist = read_vald_linelist(VALD_PATH)
    print(f"  Loaded {len(linelist)} lines from VALD")

    atm = korg.read_model_atmosphere(ATMOSPHERE_PATH)
    A_X = korg.format_A_X()
    wavelengths = np.linspace(WL_MIN, WL_MAX, N_WL)
    abundances = np.array(A_X_to_absolute(A_X))

    t0 = time.perf_counter()
    result = synthesize(
        atmosphere=atm,
        linelist=linelist,
        wavelengths_angstrom=wavelengths,
        abundances=abundances,
        vmic=1.0,
        verbose=False,
    )
    elapsed = time.perf_counter() - t0

    flux = np.array(result.flux)
    continuum = np.array(result.continuum)
    cnorm = flux / continuum

    print(f"  Elapsed: {elapsed*1000:.1f} ms")
    print(f"  Min normalized flux: {cnorm.min():.4f}")
    return wavelengths, flux, continuum, cnorm, elapsed


def run_julia_synthesis():
    print("\n=== Julia Korg.jl Synthesis ===")

    vald_abs = os.path.abspath(VALD_PATH)
    atm_abs = os.path.abspath(ATMOSPHERE_PATH)

    line_buffer_cm = 10.0 * 1e-8
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


def _ms(elapsed_s):
    return elapsed_s * 1000


def make_plot(wl_jit, cnorm_jit, jit_elapsed, wl_jl, cnorm_jl, julia_ms):
    jit_ms = _ms(jit_elapsed)

    cnorm_jl_i = np.interp(wl_jit, wl_jl, cnorm_jl)
    diff = cnorm_jit - cnorm_jl_i
    rms = np.sqrt(np.mean(diff**2))

    C_JL = '#111111'
    C_PY = '#777777'

    C_PY = "tab:blue"
    C_JL = "tab:red"

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.08, wspace=0.08)

    ax_jl = fig.add_subplot(gs[0, 0])
    ax_py = fig.add_subplot(gs[0, 1], sharey=ax_jl)
    ax_res = fig.add_subplot(gs[1, :])

    # --- Korg.jl spectrum ---
    ax_jl.plot(wl_jl, cnorm_jl, '-', color=C_JL, lw=0.6)
    ax_jl.set_title(f'Korg.jl  ({julia_ms:.0f} ms)', fontsize=9, loc='left', pad=3)
    ax_jl.set_xlim(WL_MIN, WL_MAX)
    ax_jl.set_ylim(-0.02, 1.09)
    ax_jl.set_ylabel('Normalized flux')
    ax_jl.tick_params(labelbottom=False)
    ax_jl.spines['top'].set_visible(False)
    ax_jl.spines['right'].set_visible(False)

    # --- Korg.py JIT spectrum ---
    ax_py.plot(wl_jit, cnorm_jit, '-', color=C_PY, lw=0.6)
    ax_py.set_title(f'Korg.py  ({jit_ms:.0f} ms)', fontsize=9, loc='left', pad=3)
    ax_py.set_xlim(WL_MIN, WL_MAX)
    ax_py.tick_params(labelbottom=False, labelleft=False)
    ax_py.spines['top'].set_visible(False)
    ax_py.spines['right'].set_visible(False)

    # --- Residuals (JIT − Julia) ---
    d_abs_max = np.abs(diff).max() * 1.15
    ax_res.plot(wl_jit, diff, '-', color='#444444', lw=0.5)
    ax_res.axhline(0, color='#cccccc', lw=0.6, zorder=0)
    ax_res.set_xlim(WL_MIN, WL_MAX)
    ax_res.set_ylim(-d_abs_max, d_abs_max)
    ax_res.set_xlabel('Wavelength (Å)')
    ax_res.set_ylabel('Δ (JIT − Julia)', fontsize=8)
    ax_res.text(0.99, 0.97, f'RMS {rms:.4f}  max|Δ| {np.abs(diff).max():.4f}',
                transform=ax_res.transAxes,
                ha='right', va='top', fontsize=7, color='#777777', family='monospace')
    ax_res.spines['top'].set_visible(False)
    ax_res.spines['right'].set_visible(False)

    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {OUTPUT_PNG}")
    return fig


if __name__ == '__main__':
    wl_jit, flux_jit, cntm_jit, cnorm_jit, jit_elapsed, jit_fast_elapsed = run_python_synthesis_jit()
    wl_jl, flux_jl, cntm_jl, cnorm_jl, julia_ms = run_julia_synthesis()
    speedup = julia_ms / 1000 / jit_fast_elapsed
    print(f"\n  Speedup (precomputed vs Julia): {speedup:.1f}×  ({jit_fast_elapsed*1000:.1f} ms vs {julia_ms:.1f} ms)")
    make_plot(wl_jit, cnorm_jit, jit_elapsed, wl_jl, cnorm_jl, julia_ms)
    print("\nDone.")
