#!/usr/bin/env julia
"""
Generate reference test data from Julia Korg.jl for Python comparison tests.

Run this script to generate reference data:
    julia --project=. tests/generate_julia_reference.jl

The output is saved to tests/julia_reference_data.json
"""

using Pkg
# Activate the current project (Korg.jl)
project_dir = dirname(dirname(@__FILE__))
Pkg.activate(project_dir)
Pkg.instantiate()
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
    "atomic_symbols" => Korg.atomic_symbols,
    "atomic_masses" => [Korg.atomic_masses[i] for i in 1:92],
    # Ionization energies: Dict Z => [χ₁, χ₂, χ₃] in eV
    "ionization_energies" => Dict(
        string(Z) => Korg.ionization_energies[Z] for Z in 1:92
    ),
)

# =============================================================================
# Solar Abundances
# =============================================================================
println("  - Solar abundances...")
reference_data["solar_abundances"] = Dict(
    "grevesse_2007" => Korg.grevesse_2007_solar_abundances,
    "asplund_2009" => Korg.asplund_2009_solar_abundances,
    "asplund_2020" => Korg.asplund_2020_solar_abundances,
    "bergemann_2025" => Korg.bergemann_2025_solar_abundances,
    "magg_2022" => Korg.magg_2022_solar_abundances,
    "default" => Korg.default_solar_abundances,
)

# =============================================================================
# Level 1: Wavelength Utilities
# =============================================================================
println("  - Wavelength utilities...")
wavelengths_A = [3000.0, 4000.0, 4500.0, 5000.0, 5500.0, 6000.0, 7000.0, 8000.0, 10000.0, 15000.0, 20000.0]

air_to_vacuum_results = Dict{String, Float64}()
vacuum_to_air_results = Dict{String, Float64}()
for wl in wavelengths_A
    air_to_vacuum_results[string(wl)] = Korg.air_to_vacuum(wl)
    vacuum_to_air_results[string(wl)] = Korg.vacuum_to_air(wl)
end
reference_data["wavelength_utils"] = Dict(
    "inputs" => Dict("wavelengths_angstrom" => wavelengths_A),
    "air_to_vacuum" => air_to_vacuum_results,
    "vacuum_to_air" => vacuum_to_air_results,
)

# =============================================================================
# Level 1: Line Physics Functions
# =============================================================================
println("  - Line physics functions...")

# sigma_line: cross-section factor
sigma_line_wavelengths_A = [3000.0, 4000.0, 5000.0, 6000.0, 8000.0, 10000.0]
sigma_line_results = Dict{String, Float64}()
for wl_A in sigma_line_wavelengths_A
    wl_cm = wl_A * 1e-8
    sigma_line_results[string(wl_A)] = Korg.sigma_line(wl_cm)
end

# doppler_width: Doppler broadening parameter
# Test at different temperatures, masses, microturbulence values
doppler_test_cases = [
    # (wl_A, T, mass_amu, xi_km/s) -> result
    (5000.0, 5777.0, 55.85, 1.0),   # Fe at solar temp
    (5000.0, 10000.0, 55.85, 1.0),  # Fe at hot temp
    (5000.0, 5777.0, 55.85, 2.0),   # Fe with higher xi
    (5000.0, 5777.0, 1.008, 1.0),   # H at solar temp
    (4000.0, 5777.0, 55.85, 1.0),   # Fe at different wavelength
]
doppler_results = Dict{String, Float64}()
for (wl_A, T, mass_amu, xi_kms) in doppler_test_cases
    wl_cm = wl_A * 1e-8
    mass_g = mass_amu * Korg.amu_cgs
    xi_cgs = xi_kms * 1e5
    result = Korg.doppler_width(wl_cm, T, mass_g, xi_cgs)
    key = "$(wl_A)_$(T)_$(mass_amu)_$(xi_kms)"
    doppler_results[key] = result
end

# scaled_stark: Stark broadening temperature scaling
stark_test_cases = [
    # (gamma_stark, T) -> result
    (1e-6, 5777.0),
    (1e-6, 10000.0),
    (1e-5, 5777.0),
    (1e-5, 10000.0),
]
scaled_stark_results = Dict{String, Float64}()
for (gamma, T) in stark_test_cases
    result = Korg.scaled_stark(gamma, T)
    key = "$(gamma)_$(T)"
    scaled_stark_results[key] = result
end

reference_data["line_physics"] = Dict(
    "sigma_line" => Dict(
        "inputs" => Dict("wavelengths_angstrom" => sigma_line_wavelengths_A),
        "outputs" => sigma_line_results,
    ),
    "doppler_width" => Dict(
        "inputs" => doppler_test_cases,
        "outputs" => doppler_results,
    ),
    "scaled_stark" => Dict(
        "inputs" => stark_test_cases,
        "outputs" => scaled_stark_results,
    ),
)

# =============================================================================
# Level 1: normal_pdf (LSF utility)
# =============================================================================
println("  - Normal PDF...")
normal_pdf_test_cases = [
    # (delta, sigma) -> result
    (0.0, 1.0),
    (0.5, 1.0),
    (1.0, 1.0),
    (2.0, 1.0),
    (0.0, 0.5),
    (0.5, 0.5),
    (0.0, 2.0),
    (1.0, 2.0),
]
normal_pdf_results = Dict{String, Float64}()
for (delta, sigma) in normal_pdf_test_cases
    # Julia uses Distributions.Normal, but the PDF is 1/(σ√(2π)) * exp(-Δ²/(2σ²))
    result = exp(-0.5 * delta^2 / sigma^2) / sqrt(2 * pi) / sigma
    key = "$(delta)_$(sigma)"
    normal_pdf_results[key] = result
end
reference_data["normal_pdf"] = Dict(
    "inputs" => normal_pdf_test_cases,
    "outputs" => normal_pdf_results,
)

# =============================================================================
# Level 1: exponential_integral_1 (E1)
# =============================================================================
println("  - Exponential integral E1...")
# Test at various x values covering all branches of the piecewise approximation
e1_test_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 35.0]
e1_results = Dict{String, Float64}()
for x in e1_test_values
    result = Korg.exponential_integral_1(x)
    e1_results[string(x)] = result
end
reference_data["exponential_integral_1"] = Dict(
    "inputs" => Dict("test_values" => e1_test_values),
    "outputs" => e1_results,
)

# =============================================================================
# Level 1: Interval utilities
# =============================================================================
println("  - Interval utilities...")

# Test contained() function with exclusive interval
interval_test_cases = [
    # (value, lower, upper, expected_result)
    (5.0, 3.0, 10.0, true),   # value inside
    (3.0, 3.0, 10.0, false),  # value at lower bound (exclusive)
    (10.0, 3.0, 10.0, false), # value at upper bound (exclusive)
    (2.0, 3.0, 10.0, false),  # value below lower bound
    (11.0, 3.0, 10.0, false), # value above upper bound
]
contained_results = Dict{String, Bool}()
for (value, lower, upper, _) in interval_test_cases
    interval = Korg.Interval(lower, upper)
    result = Korg.contained(value, interval)
    key = "$(value)_$(lower)_$(upper)"
    contained_results[key] = result
end

# Test closed_interval (inclusive bounds)
closed_interval_test_cases = [
    # (value, lower, upper, expected_result)
    (3.0, 3.0, 10.0, true),   # value at lower bound (inclusive)
    (10.0, 3.0, 10.0, true),  # value at upper bound (inclusive)
    (5.0, 3.0, 10.0, true),   # value inside
    (2.0, 3.0, 10.0, false),  # value below lower bound
    (11.0, 3.0, 10.0, false), # value above upper bound
]
closed_contained_results = Dict{String, Bool}()
for (value, lower, upper, _) in closed_interval_test_cases
    interval = Korg.closed_interval(lower, upper)
    result = Korg.contained(value, interval)
    key = "$(value)_$(lower)_$(upper)"
    closed_contained_results[key] = result
end

# Test contained_slice
contained_slice_test_vals = [1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 9.0, 12.0]
contained_slice_results = Dict{String, Any}()
# Exclusive interval (3, 10)
interval_exclusive = Korg.Interval(3.0, 10.0)
slice_exclusive = Korg.contained_slice(contained_slice_test_vals, interval_exclusive)
contained_slice_results["exclusive_3_10"] = Dict(
    "first" => first(slice_exclusive),
    "last" => last(slice_exclusive),
    "values" => contained_slice_test_vals[slice_exclusive]
)
# Closed interval [3, 10]
interval_closed = Korg.closed_interval(3.0, 10.0)
slice_closed = Korg.contained_slice(contained_slice_test_vals, interval_closed)
contained_slice_results["closed_3_10"] = Dict(
    "first" => first(slice_closed),
    "last" => last(slice_closed),
    "values" => contained_slice_test_vals[slice_closed]
)

reference_data["interval_utils"] = Dict(
    "contained" => Dict(
        "inputs" => interval_test_cases,
        "outputs" => contained_results,
    ),
    "closed_interval_contained" => Dict(
        "inputs" => closed_interval_test_cases,
        "outputs" => closed_contained_results,
    ),
    "contained_slice" => Dict(
        "test_vals" => contained_slice_test_vals,
        "outputs" => contained_slice_results,
    ),
)

# =============================================================================
# Isotopic Data
# =============================================================================
println("  - Isotopic data...")
# Convert isotopic_abundances to JSON-serializable format: Dict(Z => Dict(A => abundance))
isotopic_abundances_json = Dict{String, Dict{String, Float64}}()
for (Z, isotopes) in Korg.isotopic_abundances
    isotopic_abundances_json[string(Z)] = Dict{String, Float64}(
        string(A) => abund for (A, abund) in isotopes
    )
end

# Convert isotopic_nuclear_spin_degeneracies: Dict(Z => Dict(A => degeneracy))
isotopic_spin_json = Dict{String, Dict{String, Int}}()
for (Z, isotopes) in Korg.isotopic_nuclear_spin_degeneracies
    isotopic_spin_json[string(Z)] = Dict{String, Int}(
        string(A) => deg for (A, deg) in isotopes
    )
end

reference_data["isotopic_data"] = Dict(
    "isotopic_abundances" => isotopic_abundances_json,
    "isotopic_nuclear_spin_degeneracies" => isotopic_spin_json,
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
