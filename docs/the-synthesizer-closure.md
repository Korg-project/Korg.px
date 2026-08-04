# The synthesizer closure

This is the part of Korg.px that has no counterpart in Korg.jl.

```python
import numpy as np, korg
from korg.synthesis_plan import prepare_synthesis

wavelengths = np.arange(5000.0, 5002.0, 0.01)          # Angstroms
linelist = korg.get_VALD_solar_linelist()

synth = prepare_synthesis(wavelengths, linelist, geometry="plane-parallel")
flux, continuum = synth(5777.0, 4.44, 0.0)
```

`prepare_synthesis` does not synthesize anything. It returns a `Synthesizer` — a callable that
takes stellar parameters and returns a spectrum. Use it for anything beyond a single spectrum.

## Why it exists

JAX needs every array shape to be known when it traces a function. A synthesis has several
quantities that decide a shape or a selection rather than a value:

| Quantity | Decides |
|---|---|
| the first and last wavelength | which hydrogen Stark transitions are in range; whether Brackett lines are; which pixel is the 5000 Å reference |
| the coarse continuum grid | how many continuum evaluation points there are |
| each line's window in pixels | which lines share a fixed-width bucket, and how wide the buckets are |
| the MARCS interpolation's NaN mask | how many atmospheric layers there are |

Every one of them is a function of the wavelength grid, the linelist and the atmosphere grid —
all fixed before any stellar parameter is varied. If they are recomputed inside the traced
region from traced values, the function cannot be traced at all.

`prepare_synthesis` computes them once, on the host, in ordinary NumPy, and captures them in a
closure. Anything a closure captures is already a compile-time constant to JAX, so there is no
pytree to register, no `static_argnames` at the call site, and no way for a shape to change
without building a new plan. What comes back is one XLA program that can be placed on a GPU,
`vmap`ped, and differentiated end to end.

This is also why `prepare_synthesis` is a better default than `synthesize` even without
autodiff. Building the plan — partitioning the linelist into buckets, measuring the line windows
at the reference parameters, selecting the hydrogen transitions, loading the MARCS grid — costs
substantially more than a synthesis does, and
[`synthesize`](getting-started.md#synthesizing-a-solar-spectrum) pays it again on every call. On
one CPU measurement (500 pixels, 166 lines) the plan took 93 s and a call took 7 s. The ratio
varies with backend, grid size and linelist, but the plan is always the part you want to build
once.

## Building a plan

```python
prepare_synthesis(wavelengths, linelist, data=None, *,
                  geometry=None, cntm_step_cm=1e-8, window_safety=2.0,
                  reference=(5777.0, 4.44, 0.0), n_layers=56, n_mu=20)
```

- **`wavelengths_angstrom`** — the synthesis grid, in **Angstroms**, as a concrete 1-D array. This is
  host code, so it may not be a tracer. At least two points are required.
- **`linelist`** — a list of `korg.Line`, or an already-preprocessed `LinelistData`. A list is
  trimmed to the synthesis range using `line_buffer_cm`; a `LinelistData` is used as-is, since its
  extent was fixed when it was built.
- **`line_buffer_cm`** — discard lines further than this outside the grid, in cm. Default
  `10.0e-8` (10 Å), matching Korg.jl's `line_buffer`. `None` keeps every line.
- **`data`** — a `SynthesisData`; loaded from disk if omitted. Pass one you already have to avoid
  re-reading the tables when building many plans.
- **`geometry`** — `None`, `'spherical'`, or `'plane-parallel'` (also spelled `'planar'`,
  `'pp'`, `'plane_parallel'`, case-insensitive). See [below](#geometry).
- **`cntm_step_cm`** — spacing of the coarse grid the continuum is evaluated on before being
  interpolated onto the synthesis grid. Default 1e-8 cm = 1 Å, matching Korg.jl's `cntm_step`.
- **`window_safety`** — headroom on each line's pixel window. See [below](#window-truncation).
- **`reference`** — the `(Teff, log g, [M/H])` at which the line windows are *measured*. Pick
  something near the middle of the range you intend to explore.
- **`n_layers`** — the layer count the plan assumes. 56 for the MARCS grid, which is the count at
  every grid point checked. If you are calling `from_atmosphere` with a model of a different
  size, set this to match it.
- **`n_mu`** — number of Gauss-Legendre μ points for the spherical surface-flux integral;
  Korg.jl's `mu_values`. Unused in plane-parallel geometry.

The resulting object reports what it fixed:

```python
>>> synth
Synthesizer(n_wl=500, n_layers=56, n_lines=166, buckets=(32, 64, 128, 256, 500, 500), geometry='plane-parallel')
>>> synth.stark_keys          # hydrogen transitions in range
('2_4',)
>>> synth.brackett_in_range   # see "Brackett lines" below
False
>>> synth.ref_pixel           # index of 5000 A in the grid, or None if not covered
0
```

A plan is valid only for the wavelength grid, linelist, geometry and layer count it was built
with. A different grid means a different plan.

## Calling it

There are two entry points, both traced and both differentiable in every argument.

### From stellar parameters

```python
synth(Teff, logg, m_H=0.0, alpha_m=0.0, C_m=0.0, abundances=None, vmic_cm_s=1e5)
```

The MARCS interpolation runs *inside* the traced region, so `jax.grad(..., argnums=0)` is
d(flux)/d(Teff) through interpolation, chemical equilibrium, opacity and radiative transfer in
one pass.

`abundances` is a 92-element vector of number fractions (`n_X / n_total`), not `A(X)`. Convert
with `korg.abundances.A_X_to_absolute`. If you leave it `None`, it is built from `(m_H, alpha_m,
C_m)` inside the traced region, so `d/d[M/H]` picks up both the change in atmospheric structure
and the change in composition. Passing `abundances` explicitly overrides that and leaves `m_H`
acting on structure alone — which is what you want when differentiating with respect to
individual element abundances.

`vmic_cm_s` is microturbulence in **cm/s** (1e5 cm/s = 1 km/s), a scalar.

### From an atmosphere you already have

```python
synth.from_atmosphere(T, n_total, ne, z, log_tau_ref, abundances,
                      vmic_cm_s=1e5, R_photosphere=None, logg=None)
```

Use this when the atmosphere does not come from the MARCS grid. All six leading arguments are
`(n_layers,)` arrays and all are differentiable.

`log_tau_ref` here is **base-10**, the same convention as `atmosphere.log_tau_ref`, so pass that
attribute directly. See
[Differences](differences-from-korg-jl.md#log_tau_ref-is-base-10-everywhere).

`logg` is required when the plan was built with `geometry=None`, because that is what the
geometry is chosen from; it raises otherwise. `R_photosphere` defaults to
`sqrt(G M_sun / g)`, Korg.jl's convention, computed from `logg`.

Both entry points return `(flux, continuum)` in Korg.jl's flux units.

## `jax.jit`

The closure traces, so `jit` works with no `static_argnames`:

```python
import jax

fast = jax.jit(lambda Teff, logg, m_H: synth(Teff, logg, m_H))
flux, cntm = fast(5777.0, 4.44, 0.0)     # compiles
flux, cntm = fast(5800.0, 4.40, -0.5)    # does not
```

Retracing happens exactly when a new plan is built, which is exactly when a shape can change.

`jit` and eager evaluation agree to about 1.3e-8 relative, not exactly. The bucketed Voigt
kernel evaluates its profile in float32 and XLA fuses that differently under `jit`, so do not
assert bitwise equality. See [Precision floors](differences-from-korg-jl.md#precision-floors).

## `jax.vmap`

A grid of stellar parameters in one dispatch:

```python
import jax.numpy as jnp

teffs = jnp.array([5600.0, 5777.0, 5900.0])
fluxes = jax.vmap(lambda t: synth(t, 4.44, 0.0)[0])(teffs)     # (3, n_wl)
```

Over several parameters at once:

```python
Teff  = jnp.array([5600.0, 5777.0, 6100.0])
logg  = jnp.array([4.20,   4.44,   4.10])
m_H   = jnp.array([-0.5,   0.0,    0.2])
fluxes = jax.vmap(lambda t, g, m: synth(t, g, m)[0])(Teff, logg, m_H)
```

Over abundance vectors:

```python
from korg.abundances import A_X_to_absolute
base = jnp.asarray(A_X_to_absolute(korg.format_A_X()))
batch = jnp.stack([base, base * 1.01])
fluxes = jax.vmap(lambda a: synth(5777.0, 4.44, abundances=a)[0])(batch)
```

`vmap` and `jit` compose in either order. Note that `vmap` turns the `geometry=None` dispatch
into a `select` and evaluates **both** branches — see [Geometry](#geometry).

## `jax.grad`

Reverse-mode differentiation of a scalar function of the spectrum:

```python
def rectified_sum(Teff, logg, m_H):
    flux, cntm = synth(Teff, logg, m_H)
    return jnp.sum(flux / cntm)

dTeff = jax.grad(rectified_sum, argnums=0)(5777.0, 4.44, 0.0)
dlogg = jax.grad(rectified_sum, argnums=1)(5777.0, 4.44, 0.0)
dm_H  = jax.grad(rectified_sum, argnums=2)(5777.0, 4.44, 0.0)
```

With respect to all 92 abundances in one reverse pass:

```python
from korg.abundances import A_X_to_absolute
base = jnp.asarray(A_X_to_absolute(korg.format_A_X()))

def rect_sum_of_abundances(a):
    flux, cntm = synth(5777.0, 4.44, abundances=a)
    return jnp.sum(flux / cntm)

g = jax.grad(rect_sum_of_abundances)(base)     # shape (92,)
```

Note that this is the derivative with respect to *number fractions*, not with respect to `A(X)`
or `[X/H]`, so trace elements — whose number fractions are tiny — dominate the magnitudes. Chain
through `A_X_to_absolute` yourself if you want sensitivities in dex.

`jit` and `grad` compose:

```python
g = jax.jit(jax.grad(lambda t: jnp.sum(synth(t, 4.44, 0.0)[0])))(5777.0)
```

A Jacobian of the whole spectrum with respect to a few parameters is cheapest in forward mode:

```python
J = jax.jacfwd(lambda p: synth(p[0], p[1], p[2])[0])(jnp.array([5777.0, 4.44, 0.0]))
# J.shape == (n_wl, 3)
```

### What the gradients are, and are not

**Finite differences are not a reliable check for the stellar parameters.** The MARCS
interpolation is *multilinear*, so the model is only piecewise smooth in `(Teff, logg, m_H,
alpha_m, C_m)`, and a central difference does not converge under step refinement — the measured
relative difference between AD and FD wanders in the region of a few times 1e-3 as the step
shrinks, instead of decreasing. The test suite therefore asserts sign and order of magnitude
(agreement to 5%) rather than many digits. The AD value is the derivative of the model that is
actually being evaluated; the FD value is a difference quotient over a piecewise-linear
interpolant. Establishing a tighter reference is open work.

**Line windows do not contribute a derivative.** The half-width of each line's window is
recomputed inside the kernel from the traced amplitudes and broadening parameters, but it reaches
the output only through `mask = |delta| <= win` and through an integer `jnp.searchsorted`. A step
function and an integer index have zero derivative, so the window's contribution to the gradient
is structurally zero — as it is in any hard-cutoff scheme, Korg.jl's included. What recomputing
the window buys is that it *tracks* the parameters under `vmap` and away from the plan's
reference point, rather than being frozen at plan-time conditions. That is a correctness
property, not a gradient one.

**`geometry=None` is not differentiable across log g = 3.5.** See below.

## Geometry

```python
prepare_synthesis(..., geometry="plane-parallel")   # or "planar", "pp", "plane_parallel"
prepare_synthesis(..., geometry="spherical")
prepare_synthesis(..., geometry=None)               # decide from log g at call time
```

With an explicit geometry the branch is baked into the plan, and the two compile separately so
neither pays for the other's branches.

With `geometry=None`, the choice is made inside the traced region from `log g < 3.5`, which is
what Korg.jl does. It costs two things:

- Under plain `jit`, `lax.cond` runs only the selected branch, but under `vmap` JAX converts it
  to a `select` and evaluates **both**. That is a cost, not a correctness problem, but it roughly
  doubles the work when you `vmap` over log g.
- The derivative is the *selected* branch's. The switch contributes nothing and cannot:
  log g = 3.5 is a discontinuous change of model, not a smooth transition, so d(flux)/d(log g)
  genuinely does not exist there. Differentiating or `vmap`ping across the boundary is yours to
  avoid.

If you know which regime you are in, name it.

The photospheric radius used for spherical transfer is `sqrt(G M_sun / g)` computed from the
`logg` you *call* with, not from the plan's reference — one plan built at solar parameters gives
a giant the giant's radius, and `d/dlogg` sees the radius dependence.

## Window truncation

Each line is assigned to a bucket whose pixel capacity `W` is fixed at plan time, measured from a
concrete synthesis at `reference` and then rounded up to the next power of two past
`window_safety * W`. If a traced window grows past its bucket's `W` — because you moved far from
the reference parameters, or raised an abundance a long way — the profile truncates early and the
line is slightly too shallow in the wings.

The default `window_safety=2.0` covers roughly a dex. The kernel computes how many pixels each
bucket actually wanted, but the `Synthesizer` does not currently return it, so this cannot be
checked from the outside — see
[Differences](differences-from-korg-jl.md#max_needed_px-is-computed-but-not-returned). Until it
is exposed, guard against truncation by building the plan with `reference` near the middle of the
range you will explore, and by raising `window_safety` if you intend to move a long way in
abundance or pressure broadening.

## Brackett lines

Hydrogen Brackett lines are **not** implemented in the traced path. Their implementation walks
atmospheric layers in Python and would have to be rewritten to trace. The plan exposes
`synth.brackett_in_range`, which is `True` when any Brackett transition falls within 150 Å of the
synthesis window, so you can refuse rather than quietly drop them:

```python
synth = prepare_synthesis(wavelengths, linelist)
if synth.brackett_in_range:
    raise RuntimeError("Brackett lines fall in this window and are not modelled")
```

This matters in the infrared: Brackett-α is near 4.05 μm and the series head near 1.46 μm.

## A worked example

Driving an optimizer with exact gradients, with the plan built and the loss compiled once. The
loop below is deliberately the crudest possible one — a fixed-step descent, which does *not*
converge in ten steps at these step sizes; it is here to show the mechanics of building
`value_and_grad` on top of a plan. For a real fit, hand `loss_and_grad` to
`scipy.optimize.minimize(..., jac=True)` or to `optax`, or use `korg.fit_spectrum`.

```python
import numpy as np, korg, jax, jax.numpy as jnp
from korg.synthesis_plan import prepare_synthesis

wavelengths = np.arange(5000.0, 5002.0, 0.01)          # Angstroms
lines = korg.get_VALD_solar_linelist()
synth = prepare_synthesis(wavelengths, lines, geometry="plane-parallel",
                          reference=(5750.0, 4.4, 0.0))

def model(params):
    flux, cntm = synth(params[0], params[1], 0.0)
    return flux / cntm

observed = model(jnp.array([5777.0, 4.44]))          # stand-in for real data

@jax.jit
def loss_and_grad(params):
    return jax.value_and_grad(lambda p: jnp.sum((model(p) - observed) ** 2))(params)

params = jnp.array([5700.0, 4.30])
for _ in range(10):
    loss, g = loss_and_grad(params)
    params = params - jnp.array([1e2, 1e-4]) * g
```

The plan and the compile happen once; each iteration after that is a single XLA program. Note
that the compile of a `value_and_grad` over the whole synthesis is itself substantial — several
minutes on a CPU backend for this 200-pixel, 147-line problem.
