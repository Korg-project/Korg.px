#!/usr/bin/env julia
"""
Generate the solar synthesis reference used by tests/test_solar_synthesis_integration.py.

Synthesizes a single Na I line at 6000 Å in the solar MARCS model with Julia
Korg.jl and writes julia_solar_synthesis.h5 to the repository root (where the
test's REFERENCE_FILE expects it).

Run locally:
    julia --project=. tests/generate_solar_synthesis_reference.jl

In CI this runs in the generate-reference job and the resulting .h5 is uploaded
as an artifact that the test job downloads, mirroring julia_reference_data.json.
"""

using Pkg
# Activate the current project (Korg.jl), matching tests/generate_julia_reference.jl
project_dir = dirname(dirname(@__FILE__))
Pkg.activate(project_dir)
Pkg.instantiate()
using Korg
using HDF5

println("="^60)
println("Creating Solar Synthesis Reference Data")
println("="^60)

# Load the solar MARCS model (committed test fixture)
println("\nLoading solar MARCS model...")
atm = Korg.read_model_atmosphere(joinpath(@__DIR__, "data", "sun.mod"))
println("  Number of layers: $(length(atm.layers))")

# Single Na I line at 6000 Å
println("\nCreating line list...")
na_species = Korg.Species("Na I")
linelist = [
    Korg.Line(
        6000.0,       # wavelength in Å
        -1.0,         # log(gf)
        na_species,   # species
        2.1,          # E_lower in eV
        1e7,          # gamma_rad (rad/s)
        0.0,          # gamma_stark (rad/s)
    )
]

# Wavelength grid: 6000 ± 10 Å
λ_center = 6000.0
λ_range = 10.0
λ_step = 0.01
wavelengths = collect(λ_center - λ_range : λ_step : λ_center + λ_range)
println("  $(length(wavelengths)) wavelengths over $(minimum(wavelengths)) - $(maximum(wavelengths)) Å")

# Solar abundances
A_X = Korg.grevesse_2007_solar_abundances

# Synthesize spectrum (line) and continuum (no lines), no hydrogen lines
println("\nSynthesizing spectrum...")
sol = Korg.synthesize(atm, linelist, A_X, wavelengths; hydrogen_lines=false)
println("Computing continuum...")
continuum_sol = Korg.synthesize(atm, [], A_X, wavelengths; hydrogen_lines=false)
cnorm_flux = sol.flux ./ continuum_sol.flux
println("  Line depth: $(1.0 - minimum(cnorm_flux))")

# Save to HDF5 at the repo root, where the test looks for it.
output_file = joinpath(project_dir, "julia_solar_synthesis.h5")
println("\nSaving to $output_file...")
h5open(output_file, "w") do f
    # Wavelengths and fluxes
    f["wavelengths"] = wavelengths
    f["flux"] = sol.flux
    f["continuum"] = continuum_sol.flux
    f["continuum_normalized_flux"] = cnorm_flux

    # Line information
    line = linelist[1]
    f["line/wavelength_angstrom"] = 6000.0
    f["line/wavelength_cm"] = 6000.0 * 1e-8
    f["line/log_gf"] = -1.0
    f["line/species_formula"] = "Na"
    f["line/species_charge"] = 0
    f["line/E_lower_eV"] = 2.1
    f["line/gamma_rad"] = 1e7
    f["line/gamma_stark"] = 0.0
    f["line/vdW_sigma"] = line.vdW[1]
    f["line/vdW_alpha"] = line.vdW[2]

    # Atmosphere information
    f["atmosphere/temperature"] = [layer.temp for layer in atm.layers]
    f["atmosphere/electron_number_density"] = [layer.electron_number_density for layer in atm.layers]
    f["atmosphere/number_density"] = [layer.number_density for layer in atm.layers]
    f["atmosphere/z"] = [layer.z for layer in atm.layers]
    f["atmosphere/tau_ref"] = [layer.tau_ref for layer in atm.layers]
    f["atmosphere/vmic"] = 1.0

    # Abundances
    f["abundances"] = A_X

    # Na I number densities via Saha (matches the original reference generator)
    println("Computing Na I number densities...")
    partition_fn = Korg.default_partition_funcs[na_species]
    na_number_densities = zeros(length(atm.layers))
    for (i, layer) in enumerate(atm.layers)
        nₑ = layer.electron_number_density
        T = layer.temp
        n_total = layer.number_density

        χI_Na = Korg.ionization_energies[11][1]

        U_I = partition_fn(log(T))
        U_II = Korg.default_partition_funcs[Korg.Species("Na II")](log(T))
        transU = (2π * Korg.electron_mass_cgs * Korg.kboltz_cgs * T / Korg.hplanck_cgs^2)^1.5
        wII = 2.0 / nₑ * (U_II / U_I) * transU * exp(-χI_Na / (Korg.kboltz_eV * T))

        A_Na = A_X[11]
        n_Na_total = n_total * 10^(A_Na - 12)
        na_number_densities[i] = n_Na_total / (1.0 + wII)
    end
    f["atmosphere/na_I_number_density"] = na_number_densities
end

println("\n✓ Reference data saved to $output_file")
