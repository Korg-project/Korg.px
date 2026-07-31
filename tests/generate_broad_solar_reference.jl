#!/usr/bin/env julia
"""
Generate the broad solar synthesis reference used by tests/test_broad_solar_synthesis.py.

Synthesizes 3000–10000 Å (300–1000 nm) with hydrogen lines only, no metal lines,
using the solar MARCS model atmosphere and Julia Korg.jl.

Run locally:
    julia --project=. tests/generate_broad_solar_reference.jl
"""

using Pkg
project_dir = dirname(dirname(@__FILE__))
Pkg.activate(project_dir)
Pkg.instantiate()
using Korg
using HDF5

println("="^60)
println("Creating Broad Solar Synthesis Reference (3000–10000 Å)")
println("="^60)

atm = Korg.read_model_atmosphere(joinpath(@__DIR__, "data", "sun.mod"))
println("Loaded atmosphere: $(length(atm.layers)) layers")

# 3000–10000 Å, 10 Å spacing → 701 points
wavelengths = collect(3000.0:10.0:10000.0)
println("Wavelength grid: $(length(wavelengths)) points, $(minimum(wavelengths))–$(maximum(wavelengths)) Å")

A_X = Korg.grevesse_2007_solar_abundances

println("Synthesizing with hydrogen lines, no metal lines...")
sol = Korg.synthesize(atm, [], A_X, wavelengths; hydrogen_lines=true)
println("Synthesizing continuum (no lines)...")
cntm = Korg.synthesize(atm, [], A_X, wavelengths; hydrogen_lines=false)
cnorm = sol.flux ./ cntm.flux
println("Min cnorm: $(minimum(cnorm)) (deepest H line)")

output_file = joinpath(project_dir, "julia_broad_solar_synthesis.h5")
println("Saving to $output_file...")
h5open(output_file, "w") do f
    f["wavelengths"] = wavelengths
    f["flux"] = sol.flux
    f["continuum"] = cntm.flux
    f["continuum_normalized_flux"] = cnorm
    f["atmosphere/temperature"] = [layer.temp for layer in atm.layers]
    f["atmosphere/electron_number_density"] = [layer.electron_number_density for layer in atm.layers]
    f["atmosphere/number_density"] = [layer.number_density for layer in atm.layers]
    f["atmosphere/z"] = [layer.z for layer in atm.layers]
    f["atmosphere/tau_ref"] = [layer.tau_ref for layer in atm.layers]
    f["atmosphere/vmic"] = 1.0
    f["abundances"] = A_X
    # Provenance: which Korg.jl produced this reference.
    attributes(f)["korg_version"] = string(pkgversion(Korg))
    attributes(f)["julia_version"] = string(VERSION)
end

println("✓ Saved: $output_file")
