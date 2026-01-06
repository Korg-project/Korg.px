#!/usr/bin/env julia
"""
Generate reference test data from Julia Korg.jl for Python comparison tests.

Run this script to generate reference data:
    julia --project=. tests/generate_julia_reference.jl

The output is saved to tests/julia_reference_data.json
"""

using Pkg
# Activate the current project (Korg.jl)
Pkg.activate(dirname(dirname(@__FILE__)))
using Korg
using JSON

# Output file
output_file = joinpath(@__DIR__, "julia_reference_data.json")

println("Generating Julia reference data for Python comparison tests...")

# Dictionary to store all test data
reference_data = Dict{String, Any}()

# =============================================================================
# Constants
# =============================================================================
println("  - Constants...")
reference_data["constants"] = Dict(
    "c_cgs" => Korg.c_cgs,
    "hplanck_cgs" => Korg.hplanck_cgs,
    "kboltz_cgs" => Korg.kboltz_cgs,
    "electron_mass_cgs" => Korg.electron_mass_cgs,
    "electron_charge_cgs" => Korg.electron_charge_cgs,
    "amu_cgs" => Korg.amu_cgs,
    "Rydberg_eV" => Korg.Rydberg_eV,
    "kboltz_eV" => Korg.kboltz_eV,
    "hplanck_eV" => Korg.hplanck_eV,
)

# =============================================================================
# Electron Scattering
# =============================================================================
println("  - Electron scattering...")
electron_densities = [1e8, 1e10, 1e12, 1e14, 1e16]
electron_scattering_results = Dict{String, Float64}()
for ne in electron_densities
    result = Korg.ContinuumAbsorption.electron_scattering(ne)
    electron_scattering_results[string(ne)] = result
end
reference_data["electron_scattering"] = Dict(
    "inputs" => Dict("electron_densities" => electron_densities),
    "outputs" => electron_scattering_results
)

# =============================================================================
# Rayleigh Scattering
# =============================================================================
println("  - Rayleigh scattering...")
wavelengths_angstrom = [3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0, 10000.0]
nH_I = 1e15
nHe_I = 1e14
nH2 = 1e10

rayleigh_results = Dict{String, Float64}()
for wl in wavelengths_angstrom
    nu = Korg.c_cgs / (wl * 1e-8)  # Convert Angstrom to cm, then to frequency
    result = Korg.ContinuumAbsorption.rayleigh([nu], nH_I, nHe_I, nH2)[1]
    rayleigh_results[string(wl)] = result
end
reference_data["rayleigh_scattering"] = Dict(
    "inputs" => Dict(
        "wavelengths_angstrom" => wavelengths_angstrom,
        "nH_I" => nH_I,
        "nHe_I" => nHe_I,
        "nH2" => nH2
    ),
    "outputs" => rayleigh_results
)

# =============================================================================
# Translational Partition Function
# =============================================================================
println("  - Translational U...")
temperatures = [1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 5777.0, 6000.0, 8000.0, 10000.0, 15000.0]
translational_U_results = Dict{String, Float64}()
for T in temperatures
    result = Korg.translational_U(Korg.electron_mass_cgs, T)
    translational_U_results[string(T)] = result
end
reference_data["translational_U"] = Dict(
    "inputs" => Dict("temperatures" => temperatures),
    "outputs" => translational_U_results
)

# =============================================================================
# Gaunt Factor (free-free)
# =============================================================================
println("  - Gaunt factor...")
# Test at various (log_u, log_gamma2) combinations
gaunt_test_cases = [
    (-2.0, -1.0),
    (-1.5, -0.5),
    (-1.0, 0.0),
    (-0.5, 0.5),
    (0.0, 1.0),
    (0.5, 1.5),
    (1.0, 2.0),
]
gaunt_results = Dict{String, Float64}()
for (log_u, log_gamma2) in gaunt_test_cases
    result = Korg.ContinuumAbsorption.gaunt_ff_vanHoof(log_u, log_gamma2)
    key = "$(log_u)_$(log_gamma2)"
    gaunt_results[key] = result
end
reference_data["gaunt_ff"] = Dict(
    "inputs" => Dict("test_cases" => gaunt_test_cases),
    "outputs" => gaunt_results
)

# =============================================================================
# Hydrogenic Free-Free Absorption
# =============================================================================
println("  - Hydrogenic ff absorption...")
T_ff = 5777.0
Z = 1
ni = 1e14
ne = 1e13
wavelengths_ff = [3000.0, 4000.0, 5000.0, 6000.0, 8000.0, 10000.0]

hydrogenic_ff_results = Dict{String, Float64}()
for wl in wavelengths_ff
    nu = Korg.c_cgs / (wl * 1e-8)
    result = Korg.ContinuumAbsorption.hydrogenic_ff_absorption(nu, T_ff, Z, ni, ne)
    hydrogenic_ff_results[string(wl)] = result
end
reference_data["hydrogenic_ff"] = Dict(
    "inputs" => Dict(
        "T" => T_ff,
        "Z" => Z,
        "ni" => ni,
        "ne" => ne,
        "wavelengths_angstrom" => wavelengths_ff
    ),
    "outputs" => hydrogenic_ff_results
)

# =============================================================================
# Species Parsing
# =============================================================================
println("  - Species parsing...")
species_codes = ["H I", "H II", "He I", "He II", "Fe I", "Fe II", "Ca II", "CO", "H2O", "FeH"]
species_results = Dict{String, Any}()
for code in species_codes
    sp = Korg.Species(code)
    species_results[code] = Dict(
        "charge" => sp.charge,
        "atoms" => collect(sp.formula.atoms),
        "is_molecule" => Korg.ismolecule(sp)
    )
end
reference_data["species"] = Dict(
    "inputs" => Dict("codes" => species_codes),
    "outputs" => species_results
)

# =============================================================================
# Formula Parsing
# =============================================================================
println("  - Formula parsing...")
formula_codes = ["H", "He", "Fe", "CO", "H2O", "FeH", "C2", "TiO"]
formula_results = Dict{String, Any}()
for code in formula_codes
    f = Korg.Formula(code)
    formula_results[code] = Dict(
        "atoms" => collect(f.atoms),
        "mass" => Korg.get_mass(f)
    )
end
reference_data["formula"] = Dict(
    "inputs" => Dict("codes" => formula_codes),
    "outputs" => formula_results
)

# =============================================================================
# Atomic Data
# =============================================================================
println("  - Atomic data...")
reference_data["atomic_data"] = Dict(
    "atomic_masses" => [Korg.atomic_masses[i] for i in 1:92],
    "first_ionization_energies" => [Korg.ionization_energies[i][1] for i in 1:92],
)

# =============================================================================
# Save to JSON
# =============================================================================
println("\nSaving to $output_file...")
open(output_file, "w") do f
    JSON.print(f, reference_data, 2)  # 2-space indentation
end

println("Done! Reference data saved.")
println("\nTo run Python comparison tests:")
println("  pytest tests/test_julia_comparison.py -v")
