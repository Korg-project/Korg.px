# `voigt_hjerting` has up to ~15% error near the α = 1.4 branch boundary

Draft of an issue to file against Korg.px (and worth raising upstream on Korg.jl, which
shares the implementation). Everything below is reproducible from this repository.

## Summary

`voigt_hjerting(α, v)` departs from the exact Voigt function by up to **15.5%** in a band
approaching the `α = 1.4` branch boundary from below. The error is not a smooth
approximation error: it is discontinuous across the regime seams of the Hunger (1965)
piecewise fit, so both the profile *value* and its derivative jump there.

Checked against `real(w(v + iα))`, where `w` is the Faddeeva function — that is the
Voigt–Hjerting function by definition.

This is **not a porting artefact**. Korg.jl v1.2.1 produces the same numbers to every
printed digit; Korg.px reproduces the identical coefficients and seams. Fixing it in the
port alone would break the Julia reference tests.

## Reproduce — Python

```python
import numpy as np
from scipy.special import wofz
from korg.line_profiles import voigt_hjerting

for a, v in [(0.19, 1.0), (0.21, 1.0), (1.39, 1.0), (1.40, 1.0), (1.41, 1.0),
             (0.50, 3.0), (3.00, 6.0)]:
    H, E = float(voigt_hjerting(a, v)), wofz(v + 1j * a).real
    print(f"alpha={a} v={v}  korg={H:.8f}  exact={E:.8f}  rel err={abs(H - E) / E:.2%}")
```

```
alpha=0.19 v=1.0  korg=0.37105628  exact=0.37334546  rel err=0.61%
alpha=0.21 v=1.0  korg=0.37261867  exact=0.37292320  rel err=0.08%
alpha=1.39 v=1.0  korg=0.29696067  exact=0.26687088  rel err=11.28%
alpha=1.4  v=1.0  korg=0.29676343  exact=0.26596654  rel err=11.58%
alpha=1.41 v=1.0  korg=0.25064021  exact=0.26506589  rel err=5.44%
alpha=0.5  v=3.0  korg=0.03645910  exact=0.03712637  rel err=1.80%
alpha=3.0  v=6.0  korg=0.03855472  exact=0.03855460  rel err=0.00%
```

## Reproduce — Julia

```julia
using Korg, SpecialFunctions

exact(α, v) = real(faddeeva(complex(v, α)))

println("Korg.jl v", pkgversion(Korg))
for (α, v) in [(0.19, 1.0), (0.21, 1.0), (1.39, 1.0), (1.40, 1.0), (1.41, 1.0),
               (0.50, 3.0), (3.00, 6.0)]
    H, E = Korg.voigt_hjerting(α, v), exact(α, v)
    println("α=$α v=$v  Korg=$(round(H, digits=8))  exact=$(round(E, digits=8))  ",
            "rel err=$(round(100*abs(H-E)/E, digits=2))%")
end
```

```
Korg.jl v1.2.1
α=0.19 v=1.0  Korg=0.37105628  exact=0.37334546  rel err=0.61%
α=0.21 v=1.0  Korg=0.37261867  exact=0.3729232   rel err=0.08%
α=1.39 v=1.0  Korg=0.29696067  exact=0.26687088  rel err=11.28%
α=1.4  v=1.0  Korg=0.29676343  exact=0.26596654  rel err=11.58%
α=1.41 v=1.0  Korg=0.25064021  exact=0.26506589  rel err=5.44%
α=0.5  v=3.0  Korg=0.0364591   exact=0.03712637  rel err=1.8%
α=3.0  v=6.0  Korg=0.03855472  exact=0.0385546   rel err=0.0%
```

Note that `α = 1.39` and `α = 1.41` straddle the case-3/case-4 boundary: the returned
value drops by 15% across it, while the true function changes by 0.7%.

## Extent in (α, v)

Scanning `α ∈ [0.001, 4]` × `v ∈ [0, 8]` on a 400×400 grid:

| region | max relative error |
|---|---|
| `α < 0.2` | 1.09% |
| `0.2 ≤ α < 1.4` | **15.53%** |
| `1.4 ≤ α ≤ 4` | 9.93% |

Worst point **α = 1.394, v = 1.143 → 15.5%**. 7.6% of the grid exceeds 1% error, 1.1%
exceeds 5%. The large errors sit in a band approaching `α = 1.4` from below, i.e. in
case 3 rather than case 4 — at `(1.4, 1.0)` case 3 is 11.6% high while case 4 is 5.4%
low, so neither side is accurate there and the discontinuity is roughly the sum of the
two.

## Extent in practice

The question that decides whether this matters is how often a real synthesis lands in
that band. Measured over **840,000 (line, layer) evaluations** — 15,000 lines sampled
from the GALAH DR3 linelist against the MARCS solar model, with radiative, Stark and van
der Waals broadening all included:

| α band | share of evaluations | error there |
|---|---|---|
| `< 0.2` | 98.84% | < 1.1% |
| `0.2 – 1.2` | 0.63% | up to ~5% |
| **`1.2 – 1.6`** | **0.073%** | **up to 15.5%** |
| `1.6 – 4.0` | 0.16% | up to 9.9% |
| `> 4.0` | 0.29% | ~0% (asymptotically Lorentzian) |

Mapping the error curve onto that distribution:

| error exceeds | fraction of evaluations |
|---|---|
| 1% | 0.88% |
| 2% | 0.74% |
| 5% | 0.16% |
| 10% | 0.036% |

So roughly 1 in 1,100 evaluations is wrong by more than 5%, and 1 in 2,800 by more than
10%. The median α is 0.0009 — the solar photosphere is overwhelmingly Doppler-dominated.

Three caveats stop that being negligible:

- **It is not random which lines are affected.** `α = γ_L / (σ_D √2)` is largest for the
  strongest, most pressure-broadened lines and the deepest layers — precisely the lines
  carrying the most equivalent width, and the ones used for abundance work. A 15% error
  scattered uniformly would be noise; concentrated on the strong-line tail it is a
  systematic.
- **The p99 is already inside it.** At α = 0.27 the error is 3.4%.
- **The Sun is the easy case.** Higher-gravity or cooler atmospheres push α up, so this
  is a lower bound over Korg's parameter space.

The far tail is safe: above α ≈ 4 the profile is asymptotically Lorentzian and case 4 is
accurate to ~0%, so the 0.29% of evaluations at α > 4 cost nothing. The damage is
entirely in the middle.

## Derivative

The seams also make the derivative discontinuous, which matters for gradient-based
fitting. One-sided relative jumps in `∂H/∂α`:

| boundary | jump |
|---|---|
| `α = 0.2` (case 2 ↔ 3) | 4.4% |
| `v = 5` (case 2 ↔ 1) | 7.0% |
| `α + v = 3.2` (case 3 ↔ 4) | 6.4% |
| `α = 1.4` (case 3 ↔ 4) | 240% |

## Possible resolutions

1. **Call a real Faddeeva implementation.** `scipy.special.wofz` in Python,
   `SpecialFunctions.faddeeva` in Julia. Machine precision everywhere and a smooth
   derivative. Cost is the open question — `voigt_hjerting` is in the inner loop of line
   absorption — so this needs benchmarking before being dismissed as too slow.
2. **Refit case 3, or move the case-3/case-4 boundary**, keeping the piecewise structure
   but reducing the seam error.
3. **Document it.** If the current accuracy is acceptable, say so in the `voigt_hjerting`
   docstring so downstream users know the profile is unreliable near α ≈ 1.4.

Option 1 with a benchmark seems the natural first step.

## Context

Found while adding automatic-differentiation tests to Korg.px. The seams are now pinned
by characterization tests in `tests/test_autodiff_line_profiles.py`, with per-boundary
bounds, so a regression that *widens* one will fail — but the tests deliberately encode
the current behaviour rather than asserting correctness against the exact function.
