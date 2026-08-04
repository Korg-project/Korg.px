"""Quick timing test for chemical equilibrium and synthesize."""
import time
import numpy as np

print("Importing korg...", flush=True)
t0 = time.perf_counter()
from korg.atmosphere import create_solar_test_atmosphere
from korg.abundances import format_A_X, A_X_to_absolute
from korg.synthesis_plan import prepare_synthesis, synthesize
from korg.data_loader import (load_ionization_energies,
                              setup_partition_funcs_and_equilibrium_constants)
from korg.statmech import reference_chemical_equilibrium
print(f"  import done in {time.perf_counter()-t0:.2f}s", flush=True)

pf, log_Keq = setup_partition_funcs_and_equilibrium_constants()
ion_e = load_ionization_energies()
A_X = format_A_X()
abs_abund = A_X_to_absolute(A_X)

# --- reference_chemical_equilibrium in isolation ---
print("\n--- reference_chemical_equilibrium (statmech) ---", flush=True)
T = 5777.0
n_total = 1e17
ne_model = 3e13
print(f"  T={T}K, n_total={n_total:.1e}, ne_model={ne_model:.1e}", flush=True)

t0 = time.perf_counter()
ne, nd = reference_chemical_equilibrium(T, n_total, ne_model, abs_abund, ion_e, pf, log_Keq)
print(f"  1st call: {time.perf_counter()-t0:.3f}s  (ne={ne:.3e})", flush=True)

t0 = time.perf_counter()
ne, nd = reference_chemical_equilibrium(T*1.01, n_total, ne_model, abs_abund, ion_e, pf, log_Keq)
print(f"  2nd call: {time.perf_counter()-t0:.3f}s", flush=True)

t0 = time.perf_counter()
ne, nd = reference_chemical_equilibrium(T*0.99, n_total*2, ne_model, abs_abund, ion_e, pf, log_Keq)
print(f"  3rd call: {time.perf_counter()-t0:.3f}s", flush=True)

# Time 10 calls to get stable estimate
t0 = time.perf_counter()
for i in range(10):
    ne, nd = reference_chemical_equilibrium(T*(1+0.01*i), n_total, ne_model, abs_abund, ion_e, pf, log_Keq)
print(f"  avg of 10 calls: {(time.perf_counter()-t0)/10:.3f}s each", flush=True)

# --- synthesize ---
# Note that this times the *one-shot* form, which rebuilds the plan on every
# call. prepare_synthesis + repeated calls is what a real workload should do,
# and is timed separately below.
print("\n--- synthesize (full pipeline, no lines) ---", flush=True)
atm = create_solar_test_atmosphere()
wl = np.linspace(5000, 5010, 200)

t0 = time.perf_counter()
result = synthesize(atm, [], wl, A_X, vmic=1.0)
print(f"  1st call (JIT compile): {time.perf_counter()-t0:.3f}s", flush=True)

t0 = time.perf_counter()
result = synthesize(atm, [], wl, A_X, vmic=1.0)
print(f"  2nd call:               {time.perf_counter()-t0:.3f}s", flush=True)

t0 = time.perf_counter()
result = synthesize(atm, [], wl, A_X, vmic=1.0)
print(f"  3rd call:               {time.perf_counter()-t0:.3f}s", flush=True)

# --- prepare_synthesis, then repeated calls ---
print("\n--- prepare_synthesis + repeated calls ---", flush=True)
t0 = time.perf_counter()
plan = prepare_synthesis(wl, [], geometry="plane-parallel",
                         n_layers=len(atm.layers))
print(f"  plan build:             {time.perf_counter()-t0:.3f}s", flush=True)

import jax.numpy as jnp
args = (jnp.asarray(atm.T), jnp.asarray(atm.n_total), jnp.asarray(atm.ne),
        jnp.asarray(atm.z), jnp.asarray(atm.log_tau_ref),
        jnp.asarray(abs_abund))
for i in range(3):
    t0 = time.perf_counter()
    f, c = plan.from_atmosphere(*args)
    float(f[0])   # force the device computation to finish
    print(f"  call {i+1}:                 {time.perf_counter()-t0:.3f}s", flush=True)

print("\nDone.", flush=True)
