#!/usr/bin/env julia
"""
Generate Julia reference data for the *linelist I/O* cluster of Korg.px.

This is deliberately a **separate** generator from tests/generate_julia_reference.jl
(which is owned by another workstream) so that the two can be regenerated
independently without collisions.

Run with:

    export PATH="/mnt/sw/nix/store/yr11xz204lj9ah1irz61lbqh1dk0hcif-julia-1.11.2/bin:\$PATH"
    cd /mnt/home/acasey/software/Korg.px
    julia --project=. tests/generate_linelist_reference.jl

Output: tests/linelist_reference_data.json

The input linelists live in tests/data/linelists and were copied verbatim from
Korg.jl v1.2.1's own test suite (test/data/linelists), so Python and Julia parse
byte-identical inputs.
"""

using Pkg
project_dir = dirname(dirname(@__FILE__))
Pkg.activate(project_dir)
using Korg
using JSON

const DATA = joinpath(@__DIR__, "data", "linelists")
const OUT = joinpath(@__DIR__, "linelist_reference_data.json")

"""Serialise a Korg.Line to a plain Dict (full Float64 precision via JSON.jl)."""
function line_dict(l)
    Dict{String,Any}(
        "wl" => l.wl,
        "log_gf" => l.log_gf,
        "species" => string(l.species),
        "species_charge" => l.species.charge,
        "E_lower" => l.E_lower,
        "gamma_rad" => l.gamma_rad,
        "gamma_stark" => l.gamma_stark,
        "vdW" => [l.vdW[1], l.vdW[2]],
    )
end

ref = Dict{String,Any}()
ref["korg_version"] = string(pkgversion(Korg))

# ---------------------------------------------------------------------------
# MOOG
# ---------------------------------------------------------------------------
println("  - MOOG linelist...")
moog_file = joinpath(DATA, "s5eqw_short.moog")
ref["moog"] = Dict(
    "file" => "s5eqw_short.moog",
    "vacuum" => [line_dict(l)
                 for l in open(f -> Korg.parse_moog_linelist(f, Korg.isotopic_abundances, true),
                               moog_file)],
    "air" => [line_dict(l)
              for l in open(f -> Korg.parse_moog_linelist(f, Korg.isotopic_abundances, false),
                            moog_file)],
)

# ---------------------------------------------------------------------------
# Kurucz
#
# parse_kurucz_linelist is captured unfiltered and in file order; read_linelist
# is captured on top of it because that is where Korg.jl drops triply+ ionized
# species and hydrogen and sorts by wavelength.
# ---------------------------------------------------------------------------
println("  - Kurucz linelists...")
const KURUCZ = joinpath(DATA, "kurucz")
kurucz_out = Dict{String,Any}()
for (key, fname) in [
    "head" => "gfallvac08oct17.head.dat",
    "head_missing_col" => "gfallvac08oct17-missing-col.head.dat",
    "short_lines" => "gfallvac08oct17-short-lines.stub.dat",
    "ba" => "gfallvac08oct17_ba",
    "filtered_species" => "gfallvac08oct17-filtered-species.dat",
]
    path = joinpath(KURUCZ, fname)
    kurucz_out[key] = Dict(
        "file" => "kurucz/" * fname,
        # air wavelengths, Korg's NIST isotopic abundances
        "air" => [line_dict(l)
                  for l in open(f -> Korg.parse_kurucz_linelist(f;
                                                                isotopic_abundances=Korg.isotopic_abundances),
                                path)],
        # vacuum wavelengths (format="kurucz_vac")
        "vac" => [line_dict(l)
                  for l in open(f -> Korg.parse_kurucz_linelist(f; vacuum=true,
                                                                isotopic_abundances=Korg.isotopic_abundances),
                                path)],
        # isotopic_abundances=nothing: use the adjustment Kurucz wrote into the file
        "kurucz_iso" => [line_dict(l)
                         for l in open(f -> Korg.parse_kurucz_linelist(f;
                                                                       isotopic_abundances=nothing),
                                       path)],
        # the full read_linelist path: parse, filter, sort
        "read_linelist" => [line_dict(l)
                            for l in Korg.read_linelist(path; format="kurucz")],
        "read_linelist_vac" => [line_dict(l)
                                for l in Korg.read_linelist(path; format="kurucz_vac")],
        "read_linelist_kurucz_iso" => [line_dict(l)
                                       for l in Korg.read_linelist(path; format="kurucz",
                                                                   isotopic_abundances=nothing)],
    )
end
ref["kurucz"] = kurucz_out

# ---------------------------------------------------------------------------
# Turbospectrum
# ---------------------------------------------------------------------------
println("  - Turbospectrum linelist...")
ts_file = joinpath(DATA, "Turbospectrum", "goodlist")
ref["turbospectrum"] = Dict(
    "file" => "Turbospectrum/goodlist",
    "air" => [line_dict(l)
              for l in Korg.parse_turbospectrum_linelist(ts_file, Korg.isotopic_abundances, false)],
    "vac" => [line_dict(l)
              for l in Korg.parse_turbospectrum_linelist(ts_file, Korg.isotopic_abundances, true)],
)

# ---------------------------------------------------------------------------
# VALD (all four format combinations, plus isotopic scaling)
# ---------------------------------------------------------------------------
println("  - VALD linelists...")
vald_cases = Dict(
    "short_extract_stellar" => "short-extract-stellar.vald",
    "short_extract_all" => "short-extract-all.vald",
    "long_extract_stellar" => "long-extract-stellar.vald",
    "long_extract_all_air_wavenumber" => "long-extract-all-air-wavenumber.vald",
    "long_extract_all_noquotes" => "long-extract-all-air-wavenumber-noquotes.vald",
    "linelist_vald" => "linelist.vald",
    "vald_5000_5005" => "5000-5005.vald",
    "iso_short_all_unscaled" => "isotopic_scaling/short-all-unscaled.vald",
    "iso_short_stellar_unscaled" => "isotopic_scaling/short-stellar-unscaled.vald",
    "iso_long_all_unscaled" => "isotopic_scaling/long-all-unscaled.vald",
    "iso_long_stellar_unscaled" => "isotopic_scaling/long-stellar-unscaled.vald",
    "iso_scaled" => "isotopic_scaling/scaled.vald",
)
vald_out = Dict{String,Any}()
for (key, fname) in vald_cases
    path = joinpath(DATA, fname)
    lines = open(f -> Korg.parse_vald_linelist(f, Korg.isotopic_abundances), path)
    vald_out[key] = Dict("file" => fname, "lines" => [line_dict(l) for l in lines])
end
ref["vald"] = vald_out

# ---------------------------------------------------------------------------
# ExoMol
# ---------------------------------------------------------------------------
println("  - ExoMol linelist...")
exomol_states = joinpath(DATA, "ExoMol", "40Ca-1H__XAB_abridged.states")
exomol_trans = joinpath(DATA, "ExoMol", "40Ca-1H__XAB_abridged.trans")
exomol_out = Dict{String,Any}()
exomol_out["default_6800_6810"] = [line_dict(l)
                                   for l in Korg.load_ExoMol_linelist(Korg.species"CaH",
                                                                      exomol_states, exomol_trans,
                                                                      6800, 6810; verbose=false)]
exomol_out["default_6800_6804"] = [line_dict(l)
                                   for l in Korg.load_ExoMol_linelist(Korg.species"CaH",
                                                                      exomol_states, exomol_trans,
                                                                      6800, 6804; verbose=false)]
exomol_out["explicit_isotopes_6800_6810"] = [line_dict(l)
                                             for l in Korg.load_ExoMol_linelist(Korg.species"CaH",
                                                                                exomol_states,
                                                                                exomol_trans, 6800,
                                                                                6810;
                                                                                isotopes=[(20, 40),
                                                                                          (1, 1)],
                                                                                verbose=false)]
exomol_out["deuterated_6800_6810"] = [line_dict(l)
                                      for l in Korg.load_ExoMol_linelist(Korg.species"CaH",
                                                                         exomol_states,
                                                                         exomol_trans, 6800, 6810;
                                                                         isotopes=[(20, 40), (1, 2)],
                                                                         verbose=false)]
exomol_out["n_empty_5500_6000"] = length(Korg.load_ExoMol_linelist(Korg.species"CaH", exomol_states,
                                                                  exomol_trans, 5500, 6000;
                                                                  verbose=false))
ref["exomol"] = exomol_out

# ---------------------------------------------------------------------------
# approximate_line_strength
# ---------------------------------------------------------------------------
println("  - approximate_line_strength...")
als_inputs = [
    # (wl_angstrom, log_gf, species, E_lower, T)
    (5000.0, -1.5, "Fe I", 1.01, 3500.0),
    (5000.0, -1.5, "Fe I", 1.01, 5777.0),
    (15000.0, 0.5, "CaH", 0.25, 3500.0),
    (3000.0, -3.0, "Ca II", 3.15, 8000.0),
    (6563.0, 0.71, "H I", 10.2, 6000.0),
]
ref["approximate_line_strength"] = Dict(
    "inputs" => [[a, b, c, d, e] for (a, b, c, d, e) in als_inputs],
    "outputs" => [Korg.approximate_line_strength(Korg.Line(wl, log_gf, Korg.Species(sp), E), T)
                  for (wl, log_gf, sp, E, T) in als_inputs],
)

# ---------------------------------------------------------------------------
# merge_close_lines
# ---------------------------------------------------------------------------
println("  - merge_close_lines...")
mcl_inputs = [
    # (wl_angstrom, log_gf, species, E_lower)
    (5000.00, -1.0, "Fe I", 1.0),
    (5000.10, -0.5, "Fe I", 1.0),   # merges with the previous Fe I line
    (5000.05, -2.0, "Ca I", 1.0),   # different species -> own group
    (5000.50, -1.0, "Fe I", 1.0),   # too far -> new Fe I group
    (5000.55, 0.0, "Fe I", 1.0),
    (5001.00, -1.0, "Ca I", 1.0),
    (5002.00, -3.0, "Ti II", 2.0),  # singleton
]
mcl_lines = [Korg.Line(wl, log_gf, Korg.Species(sp), E) for (wl, log_gf, sp, E) in mcl_inputs]
mcl_merge_distances = [0.2, 0.05, 1.0]
ref["merge_close_lines"] = Dict(
    "inputs" => [[a, b, c, d] for (a, b, c, d) in mcl_inputs],
    "outputs" => Dict(string(d) => [[t[1], t[2], t[3], t[4]]
                                    for t in Korg.merge_close_lines(mcl_lines; merge_distance=d)]
                      for d in mcl_merge_distances),
)

# ---------------------------------------------------------------------------
# save_linelist / read_korg_linelist round trip
# ---------------------------------------------------------------------------
println("  - Korg HDF5 linelist round trip...")
rt_lines = [Korg.Line(5000.0, -1.5, Korg.Species("Fe I"), 1.01),
    Korg.Line(5001.0, -0.5, Korg.Species("Ca II"), 3.15),
    Korg.Line(5002.0, 0.25, Korg.Species("CaH"), 0.5),
    Korg.Line(5003.0, -2.0, Korg.Species("Fe I"), 2.0, 1.0e8, 1.0e-5, -7.5),
    Korg.Line(5004.0, -2.0, Korg.Species("Fe I"), 2.0, 1.0e8, 1.0e-5, 234.23)]
h5path = joinpath(@__DIR__, "data", "linelists", "korg_roundtrip.h5")
Korg.save_linelist(h5path, rt_lines)
ref["korg_h5_roundtrip"] = Dict(
    "file" => "korg_roundtrip.h5",
    "lines" => [line_dict(l) for l in Korg.read_linelist(h5path)],
)

# ---------------------------------------------------------------------------
# approximate_gammas: branches not exercised by julia_reference_data.json
# (autoionizing lines, Z > 3, molecules, singly/doubly ionized)
# ---------------------------------------------------------------------------
println("  - approximate_gammas edge cases...")
ag_inputs = [
    (5.0e-5, "Fe I", 7.5),    # E_upper > χ(Fe I) = 7.902 eV -> autoionizing
    (2.0e-5, "Fe I", 1.0),    # 2000 Å: E_upper = 1 + 6.2 eV -> autoionizing
    (5.0e-5, "Fe II", 1.0),   # Z = 2 branch of Cowley
    (5.0e-5, "Fe III", 1.0),  # Z = 3 branch
    (5.0e-5, "Fe IV", 1.0),   # Z = 4 > 3 -> (0, 0)
    (5.0e-5, "CN", 0.5),      # molecule -> (0, 0)
    (5.0e-5, "H I", 10.2),
    (5.0e-5, "Ca II", 3.15),
]
ref["approximate_gammas_edge_cases"] = Dict(
    "inputs" => [[wl, sp, E] for (wl, sp, E) in ag_inputs],
    "outputs" => [collect(Korg.approximate_gammas(wl, Korg.Species(sp), E))
                  for (wl, sp, E) in ag_inputs],
)

# ---------------------------------------------------------------------------
# Line constructor: every branch of the scalar vdW decoding
# ---------------------------------------------------------------------------
println("  - Line vdW decoding...")
vdw_cases = Any[
    -7.5,     # negative -> log10(γ_vdW)
    0.0,      # exactly zero -> no vdW broadening
    1.0,      # 0 < x < 20 -> Unsöld fudge factor
    2.5,      # ditto
    20.0,     # >= 20 -> packed ABO
    234.23,   # ditto, with a fractional α
    318.245,  # a real VALD value
]
ref["line_vdW_decoding"] = Dict(
    "inputs" => vdw_cases,
    "outputs" => [collect(Korg.Line(5000.0, -1.5, Korg.species"Fe I", 1.01, missing, missing, v).vdW)
                  for v in vdw_cases],
    # NB: gamma_stark must be given explicitly here.  Korg.jl v1.2.1 calls
    # isnan(vdW) on line 70 of linelist.jl without the `!(vdW isa Tuple)` guard
    # it uses on line 67, so Line(...; gamma_stark=missing, vdW=(σ, α)) throws a
    # MethodError.  korg.linelist.create_line guards both, so it accepts it.
    "tuple_input" => collect(Korg.Line(5000.0, -1.5, Korg.species"Fe I", 1.01, missing, 1.0e-5,
                                       (1.0e-14, 0.3)).vdW),
)

# ---------------------------------------------------------------------------
# MolecularCrossSection
# ---------------------------------------------------------------------------
println("  - MolecularCrossSection...")
mol_lines = [Korg.Line(5000.0, -1.0, Korg.species"CN", 0.5),
    Korg.Line(5000.4, -0.5, Korg.species"CN", 0.8),
    Korg.Line(5001.2, -1.5, Korg.species"CN", 0.2)]
mol_wls = collect(range(4999.5, 5001.5; length=41))          # Å
mol_vmic = [0.0, 1.0, 2.0]
mol_logT = [3.4, 3.6, 3.8]
mcs = Korg.MolecularCrossSection(mol_lines, mol_wls; vmic_vals=mol_vmic,
                                 log_temp_vals=mol_logT)
# α[i_vmic, i_logT, i_wl] as nested arrays (JSON has no ndarray type)
mol_grid = [[[mcs.itp(v, lt, wl * 1e-8) for wl in mol_wls] for lt in mol_logT] for v in mol_vmic]
ref["molecular_cross_section"] = Dict(
    "lines" => [[5000.0, -1.0, "CN", 0.5], [5000.4, -0.5, "CN", 0.8], [5001.2, -1.5, "CN", 0.2]],
    "wavelengths_angstrom" => mol_wls,
    "vmic_vals" => mol_vmic,
    "log_temp_vals" => mol_logT,
    "grid" => mol_grid,
    "species" => string(mcs.species),
    # a few off-grid interpolation queries plus one out-of-bounds query
    "queries" => [[0.5, 3.5, 5000.2], [1.5, 3.7, 5000.9], [0.0, 3.4, 4999.5],
        [2.0, 3.8, 5001.5], [1.0, 3.6, 4000.0]],
    "query_values" => [mcs.itp(q[1], q[2], q[3] * 1e-8)
                       for q in [[0.5, 3.5, 5000.2], [1.5, 3.7, 5000.9], [0.0, 3.4, 4999.5],
        [2.0, 3.8, 5001.5], [1.0, 3.6, 4000.0]]],
)

# ---------------------------------------------------------------------------
# prune_linelist
# ---------------------------------------------------------------------------
println("  - prune_linelist...")
atm = Korg.read_model_atmosphere(joinpath(@__DIR__, "data", "sun.mod"))
A_X = Korg.format_A_X()
prune_lines = [Korg.Line(5000.0, -1.5, Korg.species"Fe I", 1.0),
    Korg.Line(5000.5, -6.0, Korg.species"Fe I", 4.0),
    Korg.Line(5001.0, -0.5, Korg.species"Fe I", 0.9),
    Korg.Line(5010.0, -1.0, Korg.species"Ca I", 2.0)]
pruned_unsorted = Korg.prune_linelist(atm, prune_lines, A_X, (4999.0, 5002.0);
                                      threshold=0.1, sort_by_EW=false, verbose=false)
pruned_sorted = Korg.prune_linelist(atm, prune_lines, A_X, (4999.0, 5002.0);
                                    threshold=0.1, sort_by_EW=true, verbose=false)
pruned_loose = Korg.prune_linelist(atm, prune_lines, A_X, (4999.0, 5002.0);
                                   threshold=1e-8, sort_by_EW=false, verbose=false)
pruned_far = Korg.prune_linelist(atm, prune_lines, A_X, (4999.0, 5002.0);
                                 threshold=1e-8, sort_by_EW=false, max_distance=20.0,
                                 verbose=false)
ref["prune_linelist"] = Dict(
    "lines" => [[5000.0, -1.5, "Fe I", 1.0], [5000.5, -6.0, "Fe I", 4.0],
        [5001.0, -0.5, "Fe I", 0.9], [5010.0, -1.0, "Ca I", 2.0]],
    "window" => [4999.0, 5002.0],
    "unsorted_wls" => [l.wl * 1e8 for l in pruned_unsorted],
    "sorted_by_EW_wls" => [l.wl * 1e8 for l in pruned_sorted],
    "loose_threshold_wls" => [l.wl * 1e8 for l in pruned_loose],
    "max_distance_20_wls" => [l.wl * 1e8 for l in pruned_far],
)

# ---------------------------------------------------------------------------
open(OUT, "w") do io
    JSON.print(io, ref)
end
println("Wrote $OUT")
