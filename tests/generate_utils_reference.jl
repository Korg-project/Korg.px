#!/usr/bin/env julia
#
# Generate high-precision reference values for Korg.px's *utility* modules:
#   species.py, wavelengths.py, cubic_splines.py, abundances.py
#
# Writes tests/utils_reference_data.json.
#
# This file is deliberately separate from generate_julia_reference.jl so that
# regenerating utility references can never clobber tests/julia_reference_data.json.
#
# Usage:
#   export PATH="/mnt/sw/nix/store/yr11xz204lj9ah1irz61lbqh1dk0hcif-julia-1.11.2/bin:$PATH"
#   julia --project=. tests/generate_utils_reference.jl

using Korg
using JSON

const OUT = joinpath(@__DIR__, "utils_reference_data.json")

ref = Dict{String,Any}()

# ---------------------------------------------------------------------------
# Species parsing
# ---------------------------------------------------------------------------
println("  - Species parsing...")

species_codes = [
    # spectroscopic notation, every separator Korg.jl accepts
    "H I", "H 1", "H     1", "H_1", "H.I", "H 2", "H2", "H",
    "He I", "He II", "He III",
    "Fe I", "Fe II", "Fe III", "Fe IV", "Fe V",
    "Ca II", "Na I", "Mg_I", "Ti.II", "Ba II", "La II", "U I",
    # Kurucz/MOOG-style numeric codes (incl. leading/trailing zeros)
    "01.00", "02.01", "02.1000", "26.00", "26.01", "26.02",
    "0608", "0801", "0106", "20.0", "607.0", "812",
    # anion / cation suffixes
    "H-", "OH+", "CH+", "H2+", "CN-",
    # molecules
    "CO", "H2O", "C2", "FeH", "TiO", "MgH", "CaH", "SiO", "CN", "OH", "CH",
    "C2H2", "H2O2",
    # trailing-zero charge tags (Julia strips them)
    "H 10", "Fe 20",
]

species_out = Dict{String,Any}()
for code in species_codes
    s = Korg.Species(code)
    species_out[code] = Dict(
        "string" => string(s),
        "charge" => Int(s.charge),
        "atoms" => Int.(collect(s.formula.atoms)),
        "is_molecule" => Korg.ismolecule(s),
        "n_atoms" => Korg.n_atoms(s),
        "mass" => Korg.get_mass(s),
    )
end
ref["species_parsing"] = Dict("inputs" => species_codes, "outputs" => species_out)

# Codes that must be rejected
println("  - Species errors...")
bad_codes = ["Fe I II", "H.1.2", "Qx I", "1.2.3", "Zz"]
bad_out = Dict{String,Bool}()
for code in bad_codes
    bad_out[code] = try
        Korg.Species(code)
        false
    catch
        true
    end
end
ref["species_errors"] = bad_out

# Species from a Float64 (MOOG codes come in as floats)
println("  - Species from float...")
float_codes = [1.0, 2.01, 20.0, 26.01, 26.0, 8.01]
float_out = Dict{String,Any}()
for f in float_codes
    s = Korg.Species(f)
    float_out[string(f)] = Dict("string" => string(s), "charge" => Int(s.charge),
                                "atoms" => Int.(collect(s.formula.atoms)))
end
ref["species_from_float"] = float_out

# ---------------------------------------------------------------------------
# Formula parsing
# ---------------------------------------------------------------------------
println("  - Formula parsing...")

formula_codes = ["H", "He", "Fe", "U", "CO", "H2O", "FeH", "C2", "TiO", "OH",
                 "CN", "MgH", "C2H2", "0608", "0801", "0106", "812", "26", "01",
                 "060606", "080808", "10608", "0106080808", "H2", "N2", "O2",
                 "SiO", "CaH", "H2O2"]
formula_out = Dict{String,Any}()
for code in formula_codes
    f = Korg.Formula(code)
    formula_out[code] = Dict(
        "string" => string(f),
        "atoms" => Int.(collect(f.atoms)),
        "n_atoms" => Korg.n_atoms(f),
        "is_molecule" => Korg.ismolecule(f),
        "mass" => Korg.get_mass(f),
    )
end
ref["formula_parsing"] = Dict("inputs" => formula_codes, "outputs" => formula_out)

# all_atomic_species
println("  - all_atomic_species...")
aas = collect(Korg.all_atomic_species())
ref["all_atomic_species"] = Dict(
    "count" => length(aas),
    "strings" => string.(aas),
)

# ---------------------------------------------------------------------------
# Wavelengths
# ---------------------------------------------------------------------------
println("  - Wavelengths...")

function wl_summary(w)
    Dict(
        "length" => length(w),
        "first_cm" => w[1],
        "last_cm" => w[end],
        "sample_cm" => [w[i] for i in unique(clamp.([1, 2, 3,
                                                     cld(length(w), 2),
                                                     length(w) - 1, length(w)],
                                                    1, length(w)))],
        "sample_idx" => unique(clamp.([1, 2, 3, cld(length(w), 2),
                                       length(w) - 1, length(w)], 1, length(w))),
        "n_ranges" => length(w.wl_ranges),
        "range_lengths" => [length(r) for r in w.wl_ranges],
        # NOTE: in Korg.jl 1.2.1, `all_wls` is built *before* the air->vacuum
        # conversion and is never rebuilt, so with air_wavelengths=true only
        # `wl_ranges` carries the vacuum values.  These are the authoritative
        # numbers to compare against.
        "range_starts" => [first(r) for r in w.wl_ranges],
        "range_stops" => [last(r) for r in w.wl_ranges],
        "first_freq" => w.all_freqs[1],
        "last_freq" => w.all_freqs[end],
    )
end

wl_out = Dict{String,Any}()
# (start, stop) with the default 0.01 Å step
wl_out["5000_5001"] = wl_summary(Korg.Wavelengths((5000, 5001)))
# (start, stop, step) where the step divides the interval exactly
wl_out["5000_5500_1"] = wl_summary(Korg.Wavelengths((5000, 5500, 1.0)))
wl_out["4000_4010_0.1"] = wl_summary(Korg.Wavelengths((4000, 4010, 0.1)))
# a step that does NOT divide the interval exactly
wl_out["5000_5500_0.03"] = wl_summary(Korg.Wavelengths((5000, 5500, 0.03)))
# multiple windows
wl_out["multi"] = wl_summary(Korg.Wavelengths([(5000, 5010, 1.0), (6000, 6010, 1.0)]))
wl_out["multi3"] = wl_summary(Korg.Wavelengths([(4000, 4002, 0.5), (5000, 5002, 0.5),
                                                (6000, 6002, 0.5)]))
# cm input (values < 1 are cm)
wl_out["cm_input"] = wl_summary(Korg.Wavelengths((5.0e-5, 5.001e-5, 1.0e-8)))
# from an explicit vector
wl_out["from_vector"] = wl_summary(Korg.Wavelengths(collect(range(5000.0, 5010.0; length=11))))
wl_out["single_value"] = wl_summary(Korg.Wavelengths([5000.0]))
# air wavelengths
wl_out["air"] = wl_summary(Korg.Wavelengths((5000, 5010, 1.0); air_wavelengths=true))

ref["wavelengths"] = wl_out

# subspectrum indices (Julia is 1-based; Python's are 0-based half-open)
w = Korg.Wavelengths([(5000, 5010, 1.0), (6000, 6010, 1.0)])
ref["wavelengths_subspectrum"] = Dict(
    "first_last" => [[first(idx), last(idx)] for idx in Korg.subspectrum_indices(w)],
)
ref["wavelengths_eachwindow"] = Dict(
    "windows" => [[lo, hi] for (lo, hi) in Korg.eachwindow(w)],
)

# ---------------------------------------------------------------------------
# Abundances
# ---------------------------------------------------------------------------
println("  - Abundances...")

abund_out = Dict{String,Any}()
abund_out["solar"] = Korg.format_A_X()
abund_out["mh_-1"] = Korg.format_A_X(-1.0)
abund_out["mh_-2.5_alpha_-2.0"] = Korg.format_A_X(-2.5, -2.0)
abund_out["mh_+0.3"] = Korg.format_A_X(0.3)
abund_out["fe_-0.5"] = Korg.format_A_X(0.0, 0.0, Dict("Fe" => -0.5))
abund_out["fe_Z_-0.5"] = Korg.format_A_X(0.0, 0.0, Dict(26 => -0.5))
abund_out["He_absolute"] = Korg.format_A_X(0.0, 0.0, Dict("He" => 10.5);
                                           solar_relative=false)
abund_out["mixed"] = Korg.format_A_X(-1.0, -0.6, Dict("C" => 0.3, 22 => 0.1))
abund_out["asplund09"] = Korg.format_A_X(-0.5;
                                         solar_abundances=Korg.asplund_2009_solar_abundances)
abund_out["asplund20"] = Korg.format_A_X(-0.5;
                                         solar_abundances=Korg.asplund_2020_solar_abundances)
abund_out["grevesse07"] = Korg.format_A_X(-0.5;
                                          solar_abundances=Korg.grevesse_2007_solar_abundances)
abund_out["custom_alpha_elements"] = Korg.format_A_X(-1.0, -0.4; alpha_elements=[8, 12, 14])
ref["format_A_X"] = abund_out

# get_metals_H / get_alpha_H
mh_out = Dict{String,Any}()
for (name, A_X) in abund_out
    mh_out[name] = Dict(
        "metals_H_ignore_alpha" => Korg.get_metals_H(A_X),
        "metals_H_all" => Korg.get_metals_H(A_X; ignore_alpha=false),
        "alpha_H" => Korg.get_alpha_H(A_X),
    )
end
ref["metals_alpha_H"] = mh_out

ref["default_alpha_elements"] = collect(Korg.default_alpha_elements)
ref["solar_abundance_sets"] = Dict(
    "bergemann_2025" => Korg.bergemann_2025_solar_abundances,
    "asplund_2020" => Korg.asplund_2020_solar_abundances,
    "asplund_2009" => Korg.asplund_2009_solar_abundances,
    "grevesse_2007" => Korg.grevesse_2007_solar_abundances,
    "default_is" => Korg.default_solar_abundances == Korg.bergemann_2025_solar_abundances ?
                    "bergemann_2025" : "other",
)

# ---------------------------------------------------------------------------
# Cubic splines
# ---------------------------------------------------------------------------
println("  - Cubic splines...")

spline_out = Dict{String,Any}()

function spline_case(name, t, u; extrapolate=false, evals=nothing)
    sp = Korg.CubicSplines.CubicSpline(t, u; extrapolate=extrapolate)
    ev = evals === nothing ? collect(range(t[1], t[end]; length=21)) : evals
    spline_out[name] = Dict(
        "t" => collect(float.(t)),
        "u" => collect(float.(u)),
        "z" => collect(float.(sp.z)),
        "h" => collect(float.(sp.h)),
        "eval_x" => collect(float.(ev)),
        "eval_y" => [sp(x) for x in ev],
        "extrapolate" => extrapolate,
    )
end

spline_case("quadratic", [0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 4.0, 9.0, 16.0])
spline_case("wiggly", [0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 0.5, 2.0, 1.0])
spline_case("nonuniform", [0.0, 0.3, 1.7, 2.1, 5.0, 9.0],
            [1.0, -2.0, 0.5, 3.25, -1.0, 0.125])
spline_case("exp_like", collect(range(-2.0, 3.0; length=12)),
            [exp(x) for x in range(-2.0, 3.0; length=12)])
# flat extrapolation outside the domain
spline_case("extrap", [1.0, 2.0, 3.0, 4.0], [1.0, 4.0, 9.0, 16.0];
            extrapolate=true,
            evals=[-3.0, 0.0, 0.5, 1.0, 1.5, 2.5, 3.5, 4.0, 4.5, 10.0])

ref["cubic_splines"] = spline_out

# cumulative_integral
println("  - Cubic spline cumulative integral...")
let t = [0.0, 1.0, 2.0, 3.0, 4.0], u = [0.0, 1.0, 4.0, 9.0, 16.0]
    sp = Korg.CubicSplines.CubicSpline(t, u)
    out = zeros(length(t))
    Korg.CubicSplines.cumulative_integral!(out, sp, 0.0, 4.0)
    ref["cubic_spline_cumulative_integral"] = Dict(
        "t" => t, "u" => u, "t1" => 0.0, "t2" => 4.0, "out" => out,
    )
end

# ---------------------------------------------------------------------------

println("Writing $OUT")
open(OUT, "w") do io
    JSON.print(io, ref, 2)
end
println("Done.")
