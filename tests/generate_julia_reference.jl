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
# Level 2: Voigt Profile Functions
# =============================================================================
println("  - Voigt profile functions...")

# harris_series test cases (v < 5)
harris_test_values = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 4.9]
harris_results = Dict{String, Any}()
for v in harris_test_values
    H0, H1, H2 = Korg.harris_series(v)
    harris_results[string(v)] = Dict("H0" => H0, "H1" => H1, "H2" => H2)
end
reference_data["harris_series"] = Dict(
    "inputs" => Dict("test_values" => harris_test_values),
    "outputs" => harris_results,
)

# voigt_hjerting test cases covering all branches
voigt_test_cases = [
    # (alpha, v) - covering different regions of the approximation
    (0.1, 0.5),   # alpha <= 0.2, v < 5 (case 2)
    (0.1, 1.0),   # alpha <= 0.2, v < 5 (case 2)
    (0.1, 2.0),   # alpha <= 0.2, v < 5 (case 2)
    (0.1, 6.0),   # alpha <= 0.2, v >= 5 (case 1)
    (0.1, 10.0),  # alpha <= 0.2, v >= 5 (case 1)
    (0.5, 0.5),   # alpha <= 1.4, alpha + v < 3.2 (case 3)
    (0.5, 2.0),   # alpha <= 1.4, alpha + v < 3.2 (case 3)
    (1.0, 1.5),   # alpha <= 1.4, alpha + v < 3.2 (case 3)
    (1.0, 3.0),   # alpha <= 1.4, alpha + v > 3.2 (case 4)
    (2.0, 1.0),   # alpha > 1.4 (case 4)
    (2.0, 5.0),   # alpha > 1.4 (case 4)
]
voigt_results = Dict{String, Float64}()
for (alpha, v) in voigt_test_cases
    result = Korg.voigt_hjerting(alpha, v)
    key = "$(alpha)_$(v)"
    voigt_results[key] = result
end
reference_data["voigt_hjerting"] = Dict(
    "inputs" => voigt_test_cases,
    "outputs" => voigt_results,
)

# line_profile test cases
# (λ₀, σ, γ, amplitude, λ) -> result
line_profile_test_cases = [
    # Line center
    (5000e-8, 0.01e-8, 0.001e-8, 1.0, 5000e-8),
    # Doppler wing
    (5000e-8, 0.01e-8, 0.001e-8, 1.0, 5000.02e-8),
    # Lorentz wing
    (5000e-8, 0.01e-8, 0.001e-8, 1.0, 5000.05e-8),
    # Different parameters
    (6000e-8, 0.02e-8, 0.005e-8, 2.0, 6000e-8),
    (6000e-8, 0.02e-8, 0.005e-8, 2.0, 6000.01e-8),
]
line_profile_results = Dict{String, Float64}()
for (i, (wl0, sigma, gamma, amp, wl)) in enumerate(line_profile_test_cases)
    result = Korg.line_profile(wl0, sigma, gamma, amp, wl)
    line_profile_results[string(i)] = result
end
reference_data["line_profile"] = Dict(
    "inputs" => line_profile_test_cases,
    "outputs" => line_profile_results,
)

# =============================================================================
# Level 2: Line Window Functions
# =============================================================================
println("  - Line window functions...")

# inverse_gaussian_density test cases
# (rho, sigma) -> result
inverse_gaussian_test_cases = [
    (0.1, 1.0),   # Normal case
    (0.2, 1.0),   # Normal case
    (0.3, 1.0),   # Normal case
    (0.01, 0.5),  # Small sigma
    (0.01, 2.0),  # Large sigma
    (0.5, 1.0),   # rho > max_density -> 0
]
inverse_gaussian_results = Dict{String, Float64}()
for (rho, sigma) in inverse_gaussian_test_cases
    result = Korg.inverse_gaussian_density(rho, sigma)
    key = "$(rho)_$(sigma)"
    inverse_gaussian_results[key] = result
end
reference_data["inverse_gaussian_density"] = Dict(
    "inputs" => inverse_gaussian_test_cases,
    "outputs" => inverse_gaussian_results,
)

# inverse_lorentz_density test cases
# (rho, gamma) -> result
inverse_lorentz_test_cases = [
    (0.1, 1.0),   # Normal case
    (0.2, 1.0),   # Normal case
    (0.05, 0.5),  # Small gamma
    (0.05, 2.0),  # Large gamma
    (0.5, 1.0),   # rho > max_density -> 0
]
inverse_lorentz_results = Dict{String, Float64}()
for (rho, gamma) in inverse_lorentz_test_cases
    result = Korg.inverse_lorentz_density(rho, gamma)
    key = "$(rho)_$(gamma)"
    inverse_lorentz_results[key] = result
end
reference_data["inverse_lorentz_density"] = Dict(
    "inputs" => inverse_lorentz_test_cases,
    "outputs" => inverse_lorentz_results,
)

# =============================================================================
# Level 2: VdW Broadening
# =============================================================================
println("  - VdW broadening...")

# scaled_vdW test cases
# (vdW, mass, T) -> result
# vdW is either (gamma_vdW, -1) for simple scaling or (sigma, alpha) for ABO
scaled_vdW_test_cases = [
    # Simple scaling: gamma_vdW * (T/10000)^0.3
    ((1e-7, -1.0), 55.85 * Korg.amu_cgs, 5777.0),
    ((1e-7, -1.0), 55.85 * Korg.amu_cgs, 10000.0),
    ((1e-7, -1.0), 55.85 * Korg.amu_cgs, 4000.0),
    # ABO theory with typical σ and α values
    ((300.0, 0.25), 55.85 * Korg.amu_cgs, 5777.0),
    ((300.0, 0.25), 55.85 * Korg.amu_cgs, 10000.0),
    ((250.0, 0.30), 24.305 * Korg.amu_cgs, 5777.0),  # Mg
]
scaled_vdW_results = Dict{String, Float64}()
for (i, (vdW, mass, T)) in enumerate(scaled_vdW_test_cases)
    result = Korg.scaled_vdW(vdW, mass, T)
    scaled_vdW_results[string(i)] = result
end
reference_data["scaled_vdW"] = Dict(
    "inputs" => scaled_vdW_test_cases,
    "outputs" => scaled_vdW_results,
)

# =============================================================================
# Level 2: Species and Formula
# =============================================================================
println("  - Species and Formula details...")

# More detailed Species tests using constructor
species_detail_tests = Dict{String, Any}()
# Test atomic species
for (code, Z, charge) in [("Fe I", 26, 0), ("Fe II", 26, 1), ("Ca II", 20, 1), ("H I", 1, 0), ("He I", 2, 0)]
    sp = Korg.Species(code)
    species_detail_tests[code] = Dict(
        "Z" => sp.formula.atoms[end],
        "charge" => sp.charge,
        "mass" => Korg.get_mass(sp),
        "is_molecule" => Korg.ismolecule(sp),
        "n_atoms" => Korg.n_atoms(sp),
    )
end
# Test molecular species
for code in ["CO", "H2O", "FeH", "TiO", "C2"]
    sp = Korg.Species(code)
    species_detail_tests[code] = Dict(
        "charge" => sp.charge,
        "mass" => Korg.get_mass(sp),
        "is_molecule" => Korg.ismolecule(sp),
        "n_atoms" => Korg.n_atoms(sp),
        "atoms" => collect(sp.formula.atoms),
    )
end
reference_data["species_details"] = species_detail_tests

# Formula tests
formula_detail_tests = Dict{String, Any}()
for code in ["H", "Fe", "CO", "H2O", "FeH", "C2", "TiO"]
    f = Korg.Formula(code)
    formula_detail_tests[code] = Dict(
        "mass" => Korg.get_mass(f),
        "atoms" => collect(f.atoms),
        "n_atoms" => Korg.n_atoms(f),
        "is_molecule" => Korg.ismolecule(f),
    )
end
reference_data["formula_details"] = formula_detail_tests

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
# Saha Ion Weights
# =============================================================================
println("  - Saha ion weights...")
saha_outputs = Dict{String, Any}()
for (T, ne, Z) in [
    (5000.0, 1e13, 1), (5777.0, 1e14, 1), (5777.0, 1e14, 26),
    (8000.0, 1e14, 26), (4000.0, 1e12, 20), (6000.0, 1e13, 26),
    (10000.0, 1e15, 1), (4500.0, 1e12, 12)
]
    wII, wIII = Korg.saha_ion_weights(T, ne, Z, Korg.ionization_energies, Korg.default_partition_funcs)
    saha_outputs["$(T)_$(ne)_$(Z)"] = Dict("wII" => Float64(wII), "wIII" => Float64(wIII))
end
reference_data["saha_ion_weights"] = Dict("outputs" => saha_outputs)

# =============================================================================
# Get log nK (molecular equilibrium constants)
# =============================================================================
println("  - get_log_nK...")
get_log_nK_outputs = Dict{String, Any}()
mol_names = ["CO", "H2", "OH", "CN", "MgH", "CH"]
for mol_str in mol_names
    try
        mol = Korg.Species(mol_str)
        if mol in keys(Korg.default_log_equilibrium_constants)
            get_log_nK_outputs[mol_str] = Dict{String, Float64}()
            for T in [4000.0, 5000.0, 5777.0, 7000.0, 8000.0]
                log_nK = Korg.get_log_nK(mol, T, Korg.default_log_equilibrium_constants)
                get_log_nK_outputs[mol_str][string(T)] = Float64(log_nK)
            end
        end
    catch e
        println("    Warning: could not compute get_log_nK for $mol_str: $e")
    end
end
reference_data["get_log_nK"] = Dict("outputs" => get_log_nK_outputs)

# =============================================================================
# Exponential Integral E2
# =============================================================================
println("  - exponential_integral_2...")
e2_test_values = [0.01, 0.1, 0.5, 1.0, 1.1, 2.0, 2.5, 5.0, 10.0, 20.0, 50.0, 100.0]
e2_outputs = Dict{String, Float64}()
for x in e2_test_values
    e2_outputs[string(x)] = Float64(Korg.RadiativeTransfer.exponential_integral_2(x))
end
reference_data["exponential_integral_2"] = Dict(
    "inputs" => Dict("test_values" => e2_test_values),
    "outputs" => e2_outputs
)

# =============================================================================
# Expint Transfer Integral Core
# =============================================================================
println("  - expint_transfer_integral_core...")
expint_core_outputs = Dict{String, Float64}()
for (tau, m, b) in [
    (0.1, 0.5, 1.0), (1.0, 0.5, 1.0), (5.0, 0.5, 1.0),
    (0.1, 1.0, 2.0), (1.0, 1.0, 2.0), (2.0, 0.3, 0.5),
    (0.5, 0.8, 1.5), (3.0, 0.2, 0.8)
]
    val = Korg.RadiativeTransfer.expint_transfer_integral_core(tau, m, b)
    expint_core_outputs["$(tau)_$(m)_$(b)"] = Float64(val)
end
reference_data["expint_transfer_integral_core"] = Dict("outputs" => expint_core_outputs)

# =============================================================================
# H I bound-free absorption
# =============================================================================
println("  - H_I_bf...")
let
    T_hi = 5778.0
    nH_I_hi = 1e17
    nHe_I_hi = 9e15
    ne_hi = 1.5e14
    h_neutral = Korg.Species("H I")
    invU_H_hi = 1.0 / Korg.default_partition_funcs[h_neutral](log(T_hi))

    hi_bf_outputs = Dict{String, Float64}()
    for wl_A in [3000.0, 3646.0, 4000.0, 5000.0, 8204.0, 10000.0, 20000.0]
        nu = Korg.c_cgs / (wl_A * 1e-8)
        val = Korg.ContinuumAbsorption.H_I_bf([nu], T_hi, nH_I_hi, nHe_I_hi, ne_hi, invU_H_hi; n_max_MHD=6)[1]
        hi_bf_outputs[string(wl_A)] = Float64(val)
    end

    reference_data["H_I_bf"] = Dict(
        "inputs" => Dict("T" => T_hi, "nH_I" => nH_I_hi, "nHe_I" => nHe_I_hi,
                         "ne" => ne_hi, "invU_H" => invU_H_hi),
        "outputs" => hi_bf_outputs
    )
end

# =============================================================================
# H2+ bound-free and free-free absorption
# =============================================================================
println("  - H2plus_bf_and_ff...")
let
    T_h2p = 6000.0
    nH_I_h2p = 1e16
    nH_II_h2p = 1e11

    # Wavelength variation at fixed T
    wl_outputs = Dict{String, Float64}()
    for wl_A in [2500.0, 5000.0, 10000.0, 20000.0, 50000.0, 100000.0]
        nu = Korg.c_cgs / (wl_A * 1e-8)
        val = Korg.ContinuumAbsorption._H2plus_bf_and_ff(nu, T_h2p, nH_I_h2p, nH_II_h2p)
        wl_outputs[string(wl_A)] = Float64(val)
    end

    # Temperature variation at fixed wavelength (5000 A)
    wl_test_A = 5000.0
    nu_test = Korg.c_cgs / (wl_test_A * 1e-8)
    T_outputs = Dict{String, Float64}()
    for T in [3000.0, 4000.0, 5000.0, 6000.0, 8000.0, 10000.0]
        val = Korg.ContinuumAbsorption._H2plus_bf_and_ff(nu_test, T, nH_I_h2p, nH_II_h2p)
        T_outputs[string(T)] = Float64(val)
    end

    reference_data["H2plus_bf_and_ff"] = Dict(
        "inputs" => Dict("T" => T_h2p, "nH_I" => nH_I_h2p, "nH_II" => nH_II_h2p,
                         "temperature_test_wavelength_angstrom" => wl_test_A),
        "wavelength_outputs" => wl_outputs,
        "temperature_outputs" => T_outputs
    )
end

# =============================================================================
# Line class construction (approximate broadening)
# =============================================================================
println("  - line_class...")
let
    line_inputs = [
        (5000.0, -1.5, "Fe 1", 1.01),
        (6563.0, 0.71, "H 1", 10.2),
        (3933.0, 0.135, "Ca 2", 0.0),
        (5172.0, -0.45, "Mg 1", 2.712),
        (6707.0, 0.174, "Li 1", 0.0),
    ]

    line_outputs = Dict{String, Any}()
    for (i, (wl_A, log_gf, spec_str, E_lower)) in enumerate(line_inputs)
        wl_cm = wl_A * 1e-8
        try
            sp = Korg.Species(spec_str)
            line = Korg.Line(wl_cm, log_gf, sp, E_lower)
            line_outputs[string(i)] = Dict(
                "wl" => Float64(line.wl),
                "log_gf" => Float64(line.log_gf),
                "E_lower" => Float64(line.E_lower),
                "species_charge" => Int(sp.charge),
                "gamma_rad" => Float64(line.gamma_rad),
                "gamma_stark" => Float64(line.gamma_stark),
                "vdW" => [Float64(line.vdW[1]), Float64(line.vdW[2])],
            )
        catch e
            println("    Warning: line_class failed for $spec_str: $e")
        end
    end

    # Store inputs as list of lists (JSON-friendly)
    line_inputs_json = [[wl_A, log_gf, spec_str, E_lower] for (wl_A, log_gf, spec_str, E_lower) in line_inputs]

    reference_data["line_class"] = Dict(
        "inputs" => line_inputs_json,
        "outputs" => line_outputs
    )
end

# =============================================================================
# approximate_radiative_gamma
# =============================================================================
println("  - approximate_radiative_gamma...")
let
    radgamma_outputs = Dict{String, Float64}()
    for (wl_cm, log_gf) in [
        (5.0e-5, -1.5), (5.0e-5, 0.0), (5.0e-5, 1.0),
        (4.0e-5, -2.0), (6.0e-5, 0.5), (3.0e-5, -0.5),
        (8.0e-5, -1.0), (1.0e-4, -3.0),
    ]
        val = Korg.approximate_radiative_gamma(wl_cm, log_gf)
        radgamma_outputs["$(wl_cm)_$(log_gf)"] = Float64(val)
    end
    reference_data["approximate_radiative_gamma"] = Dict("outputs" => radgamma_outputs)
end

# =============================================================================
# approximate_gammas
# =============================================================================
println("  - approximate_gammas...")
let
    approx_gammas_outputs = Dict{String, Any}()
    test_cases = [
        (5.0e-5, "Fe I", 1.01),
        (5.0e-5, "Fe II", 3.0),
        (6.5e-5, "Ca I", 0.0),
        (4.0e-5, "Mg I", 2.712),
        (3.9e-5, "Ca II", 1.69),
        (5.0e-5, "Mn I", 0.0),
    ]
    for (wl_cm, spec_str, E_lower) in test_cases
        try
            sp = Korg.Species(spec_str)
            gamma_stark, log_gamma_vdW = Korg.approximate_gammas(wl_cm, sp, E_lower)
            # Key format must be parseable as: rsplit("_",1) -> [wl_species, E_lower]
            # then split("_",1) -> [wl, species]
            # Species with space like "Fe I" works fine since we split on "_" not " "
            key = "$(wl_cm)_$(spec_str)_$(E_lower)"
            approx_gammas_outputs[key] = Dict(
                "gamma_stark" => Float64(gamma_stark),
                "log_gamma_vdW" => Float64(log_gamma_vdW)
            )
        catch e
            println("    Warning: approximate_gammas failed for $spec_str: $e")
        end
    end
    reference_data["approximate_gammas"] = Dict("outputs" => approx_gammas_outputs)
end

# =============================================================================
# Line with explicit broadening parameters
# =============================================================================
println("  - line_explicit_broadening...")
let
    # Each test case: [wl_A, log_gf, species_str, E_lower, gamma_rad, gamma_stark, vdW]
    # vdW is a tuple [gamma, alpha]; gamma_rad in s^-1, gamma_stark in s^-1
    explicit_inputs = [
        [5000.0, -1.5, "Fe 1", 1.01, 1e8, 1e-6, [-7.5, 0.3]],
        [4227.0, 0.265, "Ca 1", 0.0, 2e8, 4e-7, [-7.3, 0.3]],
        [3933.0, 0.135, "Ca 2", 0.0, 1.5e8, 3e-7, [-7.2, 0.25]],
    ]

    explicit_outputs = Dict{String, Any}()
    for (i, tc) in enumerate(explicit_inputs)
        wl_A, log_gf, spec_str, E_lower, gamma_rad, gamma_stark, vdW_arr = tc
        wl_cm = wl_A * 1e-8
        try
            sp = Korg.Species(spec_str)
            vdW_tup = (vdW_arr[1], vdW_arr[2])
            line = Korg.Line(wl_cm, log_gf, sp, E_lower, gamma_rad, gamma_stark, vdW_tup)
            explicit_outputs[string(i)] = Dict(
                "wl" => Float64(line.wl),
                "gamma_rad" => Float64(line.gamma_rad),
                "gamma_stark" => Float64(line.gamma_stark),
                "vdW" => [Float64(line.vdW[1]), Float64(line.vdW[2])],
            )
        catch e
            println("    Warning: line_explicit_broadening failed: $e")
        end
    end

    reference_data["line_explicit_broadening"] = Dict(
        "inputs" => explicit_inputs,
        "outputs" => explicit_outputs
    )
end

# =============================================================================
# Chemical Equilibrium
# =============================================================================
println("  - chemical_equilibrium (this may take a moment)...")
let
    A_X = Korg.format_A_X()
    # Julia convention: abs_abundances = 10^(A_X - 12) = N_X/N_H (H=1)
    abs_abundances = @. 10.0^(A_X - 12)

    # Use normalized convention (N_X/N_total, sum=1) to match Python's A_X_to_absolute
    abs_abundances_normalized = abs_abundances ./ sum(abs_abundances)

    cases = [
        ("solar_tau1",  5778.0,  2.0e17, 1.5e14),
        ("solar_deep",  9000.0,  5.0e17, 5.0e15),
    ]

    chem_eq_data = Dict{String, Any}()
    for (label, T, n_total, ne_model) in cases
        try
            nₑ, number_densities = Korg.chemical_equilibrium(
                T, n_total, ne_model, abs_abundances_normalized,
                Korg.ionization_energies, Korg.default_partition_funcs,
                Korg.default_log_equilibrium_constants
            )
            chem_eq_data[label] = Dict(
                "T" => T, "n_total" => n_total, "ne_model" => ne_model,
                "ne_result" => Float64(nₑ),
                "n_H_I"  => Float64(number_densities[Korg.Species("H I")]),
                "n_H_II" => Float64(number_densities[Korg.Species("H II")]),
                "n_Fe_I" => Float64(number_densities[Korg.Species("Fe I")]),
                "n_Fe_II"=> Float64(number_densities[Korg.Species("Fe II")]),
            )
        catch e
            println("    Warning: chemical_equilibrium failed for $label: $e")
        end
    end
    reference_data["chemical_equilibrium"] = chem_eq_data
end

# =============================================================================
# Total Continuum Absorption
# =============================================================================
println("  - total_continuum_absorption...")
let
    T_cntm = 5778.0
    ne_cntm = 1.5e14
    nH_I_cntm  = 1.8e17
    nH_II_cntm = 1.0e11
    nHe_I_cntm = 1.6e16
    nHe_II_cntm = 1.0e9
    nH2_cntm   = 1.0e12

    number_densities_cntm = Dict(
        Korg.species"H_I"  => nH_I_cntm,
        Korg.species"H_II" => nH_II_cntm,
        Korg.species"He_I" => nHe_I_cntm,
        Korg.species"He_II"=> nHe_II_cntm,
        Korg.species"H2"   => nH2_cntm,
    )

    wavelengths_A = [3000.0, 4000.0, 5000.0, 6000.0, 8000.0, 10000.0]
    cntm_outputs = Dict{String, Float64}()
    for wl_A in wavelengths_A
        nu = Korg.c_cgs / (wl_A * 1e-8)
        val = Korg.ContinuumAbsorption.total_continuum_absorption(
            [nu], T_cntm, ne_cntm, number_densities_cntm, Korg.default_partition_funcs
        )[1]
        cntm_outputs[string(wl_A)] = Float64(val)
    end

    reference_data["total_continuum_absorption"] = Dict(
        "solar_layer" => Dict(
            "T" => T_cntm, "ne" => ne_cntm,
            "nH_I" => nH_I_cntm, "nH_II" => nH_II_cntm,
            "nHe_I" => nHe_I_cntm, "nHe_II" => nHe_II_cntm, "nH2" => nH2_cntm,
            "wavelengths_A" => wavelengths_A,
            "outputs" => cntm_outputs
        )
    )
end

# =============================================================================
# Hydrogen Line Absorption
# =============================================================================
println("  - hydrogen_line_absorption...")
let
    # Brackett oscillator strengths (n=4, m=5..12)
    brackett_outputs = Dict{String, Float64}()
    for m in 5:12
        f = Korg.brackett_oscillator_strength(4, m)
        brackett_outputs[string(m)] = Float64(f)
    end

    # Hummer-Mihalas occupation probability
    T_hmw = 5778.0
    nH_hmw = 1.8e17
    nHe_hmw = 1.6e16
    ne_hmw = 1.5e14
    hmw_outputs = Dict{String, Float64}()
    for n_eff in [2.0, 3.0, 5.0, 10.0, 20.0, 50.0]
        w = Korg.hummer_mihalas_w(T_hmw, n_eff, nH_hmw, nHe_hmw, ne_hmw)
        hmw_outputs[string(n_eff)] = Float64(w)
    end

    # Griem 1960 Knm constants
    greim_outputs = Dict{String, Float64}()
    for (n, m) in [(2,3), (2,4), (3,4), (3,5), (4,5), (4,6), (2,5)]
        K = Korg.greim_1960_Knm(n, m)
        greim_outputs["$(n)_$(m)"] = Float64(K)
    end

    # Holtsmark profile
    P_holt = 0.5
    holt_outputs = Dict{String, Float64}()
    for beta in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
        H = Korg.holtsmark_profile(beta, P_holt)
        holt_outputs[string(beta)] = Float64(H)
    end

    reference_data["hydrogen_line_absorption"] = Dict(
        "brackett_oscillator_strength" => brackett_outputs,
        "hummer_mihalas_w" => Dict(
            "T" => T_hmw, "nH" => nH_hmw, "nHe" => nHe_hmw, "ne" => ne_hmw,
            "outputs" => hmw_outputs
        ),
        "griem_1960_Knm" => greim_outputs,
        "holtsmark_profile" => Dict("P" => P_holt, "outputs" => holt_outputs)
    )
end

# =============================================================================
# LSF and Rotation
# =============================================================================
println("  - lsf_rotation...")
let
    wl_start = 5000.0
    wl_stop  = 5050.0
    wl_step  = 0.1

    # Synthetic spectrum: mostly flat with a few absorption features
    n_pts = round(Int, (wl_stop - wl_start) / wl_step) + 1
    wls = range(wl_start, wl_stop, length=n_pts)
    flux = ones(n_pts)
    # Add a Gaussian absorption feature at 5025 A
    for (i, wl) in enumerate(wls)
        flux[i] = 1.0 - 0.5 * exp(-0.5 * ((wl - 5025.0) / 1.0)^2)
        flux[i] -= 0.3 * exp(-0.5 * ((wl - 5010.0) / 0.5)^2)
    end
    flux = clamp.(flux, 0.0, 1.0)

    synth_wls = (wl_start, wl_stop, wl_step)

    # apply_LSF at various R values
    lsf_outputs = Dict{String, Vector{Float64}}()
    for R in [5000.0, 10000.0, 50000.0]
        result = Korg.apply_LSF(flux, synth_wls, R)
        lsf_outputs[string(R)] = Float64.(result)
    end

    # apply_rotation at various vsini values
    rot_outputs = Dict{String, Vector{Float64}}()
    for vsini in [5.0, 20.0, 50.0]
        result = Korg.apply_rotation(flux, synth_wls, vsini)
        rot_outputs[string(vsini)] = Float64.(result)
    end

    # compute_LSF_matrix
    obs_wls = collect(range(5010.0, 5040.0, length=31))
    lsf_R = 10000.0
    lsf_matrix = Korg.compute_LSF_matrix(synth_wls, obs_wls, lsf_R; verbose=false)
    lsf_matrix_result = Float64.(lsf_matrix * flux)

    reference_data["lsf_rotation"] = Dict(
        "inputs" => Dict(
            "wl_start" => wl_start, "wl_stop" => wl_stop, "wl_step" => wl_step,
            "flux" => Float64.(flux),
            "obs_wls" => Float64.(obs_wls),
            "lsf_matrix_R" => lsf_R
        ),
        "apply_lsf" => lsf_outputs,
        "apply_rotation" => rot_outputs,
        "lsf_matrix_result" => lsf_matrix_result
    )
end

# =============================================================================
# Radiative Transfer Utilities
# =============================================================================
println("  - radiative_transfer utilities...")
let
    # generate_mu_grid — returns (μ_grid, μ_weights)
    mu_grid_outputs = Dict{String, Any}()
    for n in [2, 3, 5, 7]
        μ_grid, μ_weights = Korg.RadiativeTransfer.generate_mu_grid(n)
        mu_grid_outputs[string(n)] = Dict(
            "mu" => Float64.(μ_grid),
            "weights" => Float64.(μ_weights)
        )
    end
    reference_data["generate_mu_grid"] = Dict("outputs" => mu_grid_outputs)

    # compute_I_linear_flux_only and compute_F_flux_only_expint
    # Build simple test case: increasing optical depth with linear source function
    n_layers = 20
    tau_vals = Float64.(10 .^ range(-2, 1, length=n_layers))  # τ from 0.01 to 10
    S_vals = Float64.(1.0 .+ 0.5 .* tau_vals)  # S = 1 + 0.5*τ

    F_linear = Korg.RadiativeTransfer.compute_I_linear_flux_only(tau_vals, S_vals)
    F_expint = Korg.RadiativeTransfer.compute_F_flux_only_expint(tau_vals, S_vals)

    reference_data["rt_formal_solution"] = Dict(
        "tau" => tau_vals,
        "S" => S_vals,
        "F_linear_flux_only" => Float64(F_linear),
        "F_expint_flux_only" => Float64(F_expint)
    )
end

# =============================================================================
# Blackbody / Planck function
# =============================================================================
println("  - blackbody...")
let
    bb_outputs = Dict{String, Float64}()
    for (T, wl_cm) in [
        (5778.0, 5.0e-5), (5778.0, 3.0e-5), (5778.0, 1.0e-4),
        (4000.0, 5.0e-5), (8000.0, 5.0e-5), (3000.0, 1.0e-4),
    ]
        val = Korg.blackbody(T, wl_cm)
        bb_outputs["$(T)_$(wl_cm)"] = Float64(val)
    end
    reference_data["blackbody"] = Dict("outputs" => bb_outputs)
end

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
