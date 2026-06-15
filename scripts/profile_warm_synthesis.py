"""
Warm-run flamegraph using pyinstrument.
Loads everything, warms up JIT, then profiles only the warm synthesis loop.
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import korg
from korg.linelist import read_vald_linelist

VALD = os.path.join(os.path.dirname(__file__), '..', 'src', 'korg', 'data',
                    'linelists', 'vald_extract_stellar_solar_threshold001.vald')
ATM  = os.path.join(os.path.dirname(__file__), '..', 'sun.mod')
OUT  = os.path.join(os.path.dirname(__file__), '..', 'synthesis_flamegraph.html')
N_WARM_RUNS = 30

print("Loading inputs...", flush=True)
linelist = read_vald_linelist(VALD)
atm      = korg.read_model_atmosphere(ATM)
A_X      = korg.format_A_X()
wls      = np.linspace(5000.0, 5100.0, 2000)

print("Warming up JIT...", flush=True)
korg.synthesize(atm, linelist, wls, A_X, vmic=1.0, verbose=False)
korg.synthesize(atm, linelist, wls, A_X, vmic=1.0, verbose=False)  # second warmup

print(f"Profiling {N_WARM_RUNS} warm runs...", flush=True)
from pyinstrument import Profiler
profiler = Profiler(interval=0.0005)  # 0.5 ms sampling interval
profiler.start()
for _ in range(N_WARM_RUNS):
    korg.synthesize(atm, linelist, wls, A_X, vmic=1.0, verbose=False)
profiler.stop()

profiler.print()
with open(OUT, 'w') as f:
    f.write(profiler.output_html())
print(f"\nFlamegraph saved to {OUT}")
