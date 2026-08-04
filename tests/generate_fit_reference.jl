#!/usr/bin/env julia
"""
Generate high-precision Korg.jl reference data for `src/korg/fit.py`.

Writes `tests/fit_reference_data.json`, consumed by
`tests/test_fit_julia_reference.py`.

This generator deliberately covers the parts of `Korg.Fit` that are *exactly*
comparable between the Julia and Python implementations — i.e. the pure
algorithm, with no dependence on the synthesis port:

  * `tan_scale` / `tan_unscale` and the `vmic`/`vsini` sqrt variants
  * `Korg.merge_bounds` (the window merger behind `_merge_windows`)
  * `get_slope` / `get_slope_uncertainty` (excitation/ionization balance)
  * `linear_continuum_adjustment!` (the weighted linear continuum solve)
  * the *equivalent-width extraction algorithm* of `calculate_EWs`: the raw
    synthesis grid and absorption depths are dumped alongside the resulting
    EWs, so the Python port can be fed byte-identical inputs and its boundary
    finding + trapezoid integration compared at rtol 1e-10.

It also dumps the end-to-end `calculate_EWs` values on the committed solar
MARCS model.  Those depend on the whole synthesis pipeline, so they are
compared at a much looser, separately-documented tolerance.

Run with:
    export PATH="/mnt/sw/nix/store/yr11xz204lj9ah1irz61lbqh1dk0hcif-julia-1.11.2/bin:\$PATH"
    julia --project=. tests/generate_fit_reference.jl
"""

using Pkg
project_dir = dirname(dirname(@__FILE__))
Pkg.activate(project_dir)
Pkg.instantiate()

using Korg
using JSON

const OUT = joinpath(@__DIR__, "fit_reference_data.json")

data = Dict{String,Any}()
data["korg_version"] = string(pkgversion(Korg))

# ---------------------------------------------------------------------------
# 1. tan_scale / tan_unscale
# ---------------------------------------------------------------------------
println("tan_scale / tan_unscale ...")

scale_cases = []
for (name, (lo, hi)) in [("Teff", (2800.0, 8000.0)),
                         ("logg", (-0.5, 5.5)),
                         ("M_H", (-5.0, 1.0)),
                         ("alpha_H", (-3.5, 2.0)),
                         ("epsilon", (0.0, 1.0)),
                         ("cntm_offset", (-0.5, 0.5)),
                         ("cntm_slope", (-0.1, 0.1)),
                         ("Fe", (-10.0, 4.0))]
    for frac in [0.01, 0.1, 0.25, 0.5, 0.5000001, 0.75, 0.9, 0.99]
        p = lo + frac * (hi - lo)
        push!(scale_cases,
              Dict("name" => name, "lower" => lo, "upper" => hi, "p" => p,
                   "scaled" => Korg.Fit.tan_scale(p, lo, hi)))
    end
end
data["tan_scale_cases"] = scale_cases

unscale_cases = []
for (name, (lo, hi)) in [("Teff", (2800.0, 8000.0)), ("M_H", (-5.0, 1.0))]
    for s in [-100.0, -3.0, -1.0, -0.25, 0.0, 0.25, 1.0, 3.0, 100.0]
        push!(unscale_cases,
              Dict("name" => name, "lower" => lo, "upper" => hi, "scaled" => s,
                   "unscaled" => Korg.Fit.tan_unscale(s, lo, hi)))
    end
end
data["tan_unscale_cases"] = unscale_cases

# vmic/vsini use sqrt(p) scaled onto (0, sqrt(250))
sqrt_cases = []
for p in [1e-6, 0.001, 0.5, 1.0, 1.5, 5.0, 20.0, 100.0, 249.0]
    push!(sqrt_cases,
          Dict("p" => p,
               "scaled" => Korg.Fit.tan_scale(sqrt(p), 0, sqrt(250)),
               "unscaled_roundtrip" => Korg.Fit.tan_unscale(Korg.Fit.tan_scale(sqrt(p), 0,
                                                                               sqrt(250)),
                                                            0, sqrt(250))^2))
end
data["sqrt_scale_cases"] = sqrt_cases

# The full scale()/unscale() dict interface
let params = Dict("Teff" => 5777.0, "logg" => 4.44, "M_H" => -0.13, "vmic" => 1.32,
                  "vsini" => 7.5, "epsilon" => 0.62, "cntm_offset" => 0.03,
                  "cntm_slope" => -0.004, "Fe" => 0.11, "alpha_H" => 0.21)
    data["scale_dict_input"] = params
    data["scale_dict_output"] = Korg.Fit.scale(params)
    data["unscale_dict_output"] = Korg.Fit.unscale(Korg.Fit.scale(params))
end

# ---------------------------------------------------------------------------
# 2. merge_bounds
# ---------------------------------------------------------------------------
println("merge_bounds ...")

merge_cases = []
raw_sets = [
    [(5000.0, 5100.0)],
    [(5000.0, 5100.0), (5050.0, 5200.0)],
    [(5000.0, 5050.0), (5200.0, 5300.0)],
    [(5200.0, 5300.0), (5000.0, 5050.0)],            # unsorted input
    [(5000.0, 5050.0), (5050.0, 5100.0)],            # touching exactly
    [(5000.0, 5400.0), (5100.0, 5200.0)],            # fully contained
    [(4000.0, 4001.0), (4000.5, 4002.0), (4010.0, 4011.0)],
]
for raw in raw_sets, buffer in [0.0, 1.0, 100.0]
    expanded = [(lo - buffer, hi + buffer) for (lo, hi) in raw]
    merged, idxs = Korg.merge_bounds(expanded, 0.0)
    push!(merge_cases,
          Dict("raw" => [[lo, hi] for (lo, hi) in raw],
               "buffer" => buffer,
               "merged" => [[lo, hi] for (lo, hi) in merged],
               "indices" => [collect(g) for g in idxs]))
end
data["merge_bounds_cases"] = merge_cases

# ---------------------------------------------------------------------------
# 3. get_slope / get_slope_uncertainty
# ---------------------------------------------------------------------------
println("get_slope ...")

slope_cases = []
xy_sets = [
    ([1.0, 2.0, 3.0, 4.0], [7.0, 9.0, 11.0, 13.0]),
    ([0.0, 0.5, 1.0, 2.0, 3.5], [7.51, 7.49, 7.55, 7.42, 7.61]),
    ([-5.3, -4.9, -4.4, -4.0, -3.1], [7.30, 7.44, 7.51, 7.62, 7.79]),
    ([2.0, 2.0, 2.0000001, 2.0], [1.0, 1.1, 0.9, 1.05]),
]
for (xs, ys) in xy_sets
    push!(slope_cases,
          Dict("xs" => xs, "ys" => ys,
               "slope" => Korg.Fit.get_slope(xs, ys),
               "slope_uncertainty" => Korg.Fit.get_slope_uncertainty(xs)))
end
data["get_slope_cases"] = slope_cases

# ---------------------------------------------------------------------------
# 4. linear_continuum_adjustment!
# ---------------------------------------------------------------------------
println("linear_continuum_adjustment ...")

cont_cases = []
let
    obs_wls = collect(range(5000.0, 5010.0; length=101))
    obs_flux = @. 1.0 - 0.4 * exp(-((obs_wls - 5003.0)^2) / 0.09) -
                  0.25 * exp(-((obs_wls - 5007.5)^2) / 0.04)
    obs_err = fill(0.01, length(obs_wls))
    obs_err[1:10] .= 0.05   # non-uniform weights so the ivar weighting matters

    for (label, windows) in [("nothing", nothing),
                             ("full", [(5000.0, 5010.0)]),
                             ("two", [(5000.5, 5004.0), (5006.0, 5009.0)])]
        model_flux = @. 0.97 * obs_flux + 0.004 * (obs_wls - 5005.0)
        Korg.Fit.linear_continuum_adjustment!(obs_wls, windows, model_flux, obs_flux, obs_err)
        push!(cont_cases,
              Dict("label" => label,
                   "windows" => isnothing(windows) ? nothing :
                                [[lo, hi] for (lo, hi) in windows],
                   "obs_wls" => obs_wls, "obs_flux" => obs_flux, "obs_err" => obs_err,
                   "model_flux_in" => @.(0.97 * obs_flux + 0.004 * (obs_wls - 5005.0)),
                   "model_flux_out" => model_flux))
    end
end
data["linear_continuum_adjustment_cases"] = cont_cases

# ---------------------------------------------------------------------------
# 5. calculate_EWs — algorithm isolation + end-to-end
# ---------------------------------------------------------------------------
println("calculate_EWs ...")

atm = Korg.read_model_atmosphere(joinpath(@__DIR__, "data", "sun.mod"))
A_X = Korg.format_A_X()

function ew_case(label, linelist; ew_window_size=2.0, wl_step=0.01)
    merged_windows, lines_per_window = Korg.merge_bounds([(line.wl * 1e8 - ew_window_size,
                                                           line.wl * 1e8 + ew_window_size)
                                                          for line in linelist], 0.0)
    wl_ranges = map(merged_windows) do (wl1, wl2)
        wl1:wl_step:wl2
    end
    sol = Korg.synthesize(atm, linelist, A_X, wl_ranges; line_buffer=0.0, hydrogen_lines=false)
    depth = 1 .- sol.flux ./ sol.cntm

    EWs = Korg.Fit.calculate_EWs(atm, linelist, A_X; ew_window_size=ew_window_size,
                                 wl_step=wl_step, blend_warn_threshold=Inf)

    Dict("label" => label,
         "ew_window_size" => ew_window_size,
         "wl_step" => wl_step,
         "line_wl_angstrom" => [l.wl * 1e8 for l in linelist],
         "line_log_gf" => [l.log_gf for l in linelist],
         "line_species" => [string(l.species) for l in linelist],
         "line_E_lower" => [l.E_lower for l in linelist],
         # the *exact* grid and absorption the Julia EW extraction saw
         "wl_grid" => collect(sol.wavelengths),
         "depth" => collect(depth),
         "window_lengths" => [length(r) for r in wl_ranges],
         "lines_per_window" => [collect(g) for g in lines_per_window],
         "EWs" => collect(EWs))
end

mkline(wl, log_gf, species, E_lower) = Korg.Line(wl, log_gf, Korg.Species(species), E_lower)

data["ew_isolated"] = [
    ew_case("three_isolated_FeI",
            [mkline(5000.0, -1.5, "Fe I", 1.0),
             mkline(5100.0, -1.0, "Fe I", 2.2),
             mkline(5200.0, -0.5, "Fe I", 0.9)]),
    ew_case("blended_pair",
            [mkline(5000.0, -1.0, "Fe I", 1.0),
             mkline(5001.2, -0.8, "Fe I", 1.2)]),
    ew_case("single_line_fine_step",
            [mkline(5050.0, -1.0, "Fe I", 1.5)]; wl_step=0.005),
    ew_case("mixed_ionization",
            [mkline(5000.0, -1.5, "Fe I", 1.0),
             mkline(5018.4, -1.1, "Fe II", 2.9),
             mkline(5100.0, -1.0, "Fe I", 2.2)]),
]

# ---------------------------------------------------------------------------
# 6. ews_to_abundances_approx (pure arithmetic given fixed synthetic EWs)
# ---------------------------------------------------------------------------
println("ews_to_abundances_approx ...")

let
    A0 = [7.5, 7.5, 7.5]
    synth = [102.3, 55.7, 18.2]
    measured = [204.6, 27.85, 18.2]
    data["ews_to_abundances_approx_case"] = Dict(
        "A0" => A0, "synth_EWs" => synth, "measured_EWs" => measured,
        "expected" => @.(A0 + (log10(measured) - log10(synth))))
end

# ---------------------------------------------------------------------------

open(OUT, "w") do io
    JSON.print(io, data)
end
println("Wrote $OUT")
