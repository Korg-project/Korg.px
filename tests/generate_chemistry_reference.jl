#
# Generates tests/chemistry_reference_data.json
#
# Reference values for the chemistry (statmech) and continuum-opacity modules of
# Korg.px, taken from Korg.jl v1.2.1.
#
# This file is deliberately separate from tests/generate_julia_reference.jl — do
# not merge them, and do not regenerate both at once.
#
# Run with:
#   export PATH="/mnt/sw/nix/store/yr11xz204lj9ah1irz61lbqh1dk0hcif-julia-1.11.2/bin:$PATH"
#   julia --project=. tests/generate_chemistry_reference.jl
#

using Korg
using JSON

const CA = Korg.ContinuumAbsorption

reference_data = Dict{String,Any}()

reference_data["metadata"] = Dict(
    "korg_version" => string(pkgversion(Korg)),
    "julia_version" => string(VERSION),
    "description" => "Chemistry + continuum bounds reference for Korg.px",
)

# =============================================================================
# 1. Interval / bounds-checking primitives
# =============================================================================
println("  - bounds ...")
let
    bounds = Dict{String,Any}()

    function dump_interval(iv)
        Dict("lower" => iv.lower, "upper" => iv.upper)
    end

    hminus_bf_nu = Korg.closed_interval(0.0, 2.417989242625068e19)
    hminus_ff_nu = Korg.λ_to_ν_bound(Korg.closed_interval(1823e-8, 151890e-8))
    h2plus_nu = Korg.λ_to_ν_bound(Korg.closed_interval(7e-6, 2e-3))
    heminus_ff_nu = Korg.λ_to_ν_bound(Korg.closed_interval(5.063e-5, 1.518780e-03))
    temp_1400 = Korg.closed_interval(1400, 10080)
    temp_3150 = Korg.closed_interval(3150, 25200)
    unbounded = Korg.Interval(0, Inf)

    bounds["intervals"] = Dict(
        "Hminus_bf_nu" => dump_interval(hminus_bf_nu),
        "Hminus_bf_temp" => dump_interval(unbounded),
        "Hminus_ff_nu" => dump_interval(hminus_ff_nu),
        "Hminus_ff_temp" => dump_interval(temp_1400),
        "H2plus_nu" => dump_interval(h2plus_nu),
        "H2plus_temp" => dump_interval(temp_3150),
        "Heminus_ff_nu" => dump_interval(heminus_ff_nu),
        "Heminus_ff_temp" => dump_interval(temp_1400),
    )

    # contained() at and around every endpoint that Korg.px will be asked about
    probes = Dict{String,Any}()
    for (name, iv, vals) in [
        ("Hminus_ff_nu", hminus_ff_nu,
         [Korg.c_cgs / 1823e-8, Korg.c_cgs / 151890e-8, Korg.c_cgs / 1822e-8,
          Korg.c_cgs / 151891e-8, Korg.c_cgs / 5000e-8]),
        ("Heminus_ff_nu", heminus_ff_nu,
         [Korg.c_cgs / 5.063e-5, Korg.c_cgs / 1.518780e-03, Korg.c_cgs / 5064e-8,
          Korg.c_cgs / 5062e-8, Korg.c_cgs / 1e-4]),
        ("H2plus_nu", h2plus_nu,
         [Korg.c_cgs / 7e-6, Korg.c_cgs / 2e-3, Korg.c_cgs / 6.9e-6, Korg.c_cgs / 2.1e-3]),
        ("Hminus_bf_nu", hminus_bf_nu,
         [0.0, 1.0e14, 2.417989242625068e19, 2.5e19]),
        ("Hminus_ff_temp", temp_1400, [1399.0, 1400.0, 5000.0, 10080.0, 10081.0, 12000.0]),
        ("H2plus_temp", temp_3150, [3149.0, 3150.0, 6000.0, 25200.0, 25201.0]),
        ("unbounded_temp", unbounded, [0.0, 1.0e-8, 1.0, 1e6]),
    ]
        probes[name] = Dict(string(v) => Korg.contained(v, iv) for v in vals)
    end
    bounds["contained"] = probes

    # contained_slice on a grid straddling the H⁻ ff wavelength bounds
    let
        vals = sort([Korg.c_cgs / (λ * 1e-8)
                     for λ in [1000.0, 1823.0, 5000.0, 100000.0, 151890.0, 200000.0]])
        sl = Korg.contained_slice(vals, hminus_ff_nu)
        bounds["contained_slice"] = Dict(
            "values" => vals,
            "first" => first(sl),      # 1-based, inclusive
            "last" => last(sl),        # 1-based, inclusive
            "selected" => vals[sl],
        )
    end

    reference_data["bounds"] = bounds
end

# =============================================================================
# 2. Continuum sources evaluated in *and* out of bounds (public, wrapped API)
# =============================================================================
println("  - continuum out-of-bounds ...")
let
    nH_I_div_U = 1.0e16
    nHe_I_div_U = 1.0e15
    ne = 1.0e13
    nH_I = 1.0e17
    nH_II = 1.0e12
    n_Hminus = 1.0e6

    # (label, T) pairs bracketing the table temperature limits
    temps = [1000.0, 1399.0, 1400.0, 3000.0, 3150.0, 5000.0, 5778.0, 9000.0,
             10080.0, 10081.0, 12000.0, 25200.0, 25201.0, 30000.0]
    # wavelengths in Å bracketing the table wavelength limits
    lambdas = [500.0, 700.0, 1000.0, 1823.0, 3000.0, 5000.0, 5063.0, 5064.0, 8000.0,
               10000.0, 20000.0, 151878.0, 151890.0, 200000.0, 250000.0]

    out = Dict{String,Any}()
    for src in ("Hminus_bf", "Hminus_ff", "Heminus_ff", "H2plus_bf_and_ff")
        out[src] = Dict{String,Any}()
    end

    for T in temps, λ in lambdas
        νs = [Korg.c_cgs / (λ * 1e-8)]
        k = string(T) * "_" * string(λ)
        out["Hminus_bf"][k] = Float64(CA.Hminus_bf(νs, T, n_Hminus, ne)[1])
        out["Hminus_ff"][k] = Float64(CA.Hminus_ff(νs, T, nH_I_div_U, ne)[1])
        out["Heminus_ff"][k] = Float64(CA.Heminus_ff(νs, T, nHe_I_div_U, ne)[1])
        out["H2plus_bf_and_ff"][k] = Float64(CA.H2plus_bf_and_ff(νs, T, nH_I, nH_II)[1])
    end

    reference_data["continuum_oob"] = Dict(
        "temperatures" => temps,
        "wavelengths_A" => lambdas,
        "inputs" => Dict("nH_I_div_U" => nH_I_div_U, "nHe_I_div_U" => nHe_I_div_U,
                         "ne" => ne, "nH_I" => nH_I, "nH_II" => nH_II,
                         "n_Hminus" => n_Hminus),
        "outputs" => out,
    )
end

# =============================================================================
# 3. A multi-frequency vector spanning the bound, to pin the partial-truncation
#    behaviour of bounds_checked_absorption (some entries in, some out).
# =============================================================================
println("  - continuum vector truncation ...")
let
    λs = [1000.0, 1500.0, 1823.0, 4000.0, 8000.0, 100000.0, 151890.0, 300000.0]
    νs = sort([Korg.c_cgs / (λ * 1e-8) for λ in λs])
    T = 5000.0
    reference_data["continuum_vector"] = Dict(
        "wavelengths_A_sorted_by_nu" => [Korg.c_cgs / ν * 1e8 for ν in νs],
        "nus" => νs,
        "T" => T,
        "Hminus_ff" => Vector{Float64}(CA.Hminus_ff(νs, T, 1.0e16, 1.0e13)),
        "Heminus_ff" => Vector{Float64}(CA.Heminus_ff(νs, T, 1.0e15, 1.0e13)),
    )
end

# =============================================================================
# 4. statmech scalars
# =============================================================================
println("  - statmech ...")
let
    sm = Dict{String,Any}()

    temps = [1000.0, 3500.0, 5778.0, 9000.0, 12000.0, 30000.0]

    sm["translational_U"] = Dict(
        string(T) => Korg.translational_U(Korg.electron_mass_cgs, T) for T in temps)
    sm["Hminus_nK"] = Dict(string(T) => Korg.Hminus_nK(T) for T in temps)

    # hummer_mihalas_w over a grid of n_eff and densities, both generalizations
    hm = Dict{String,Any}()
    for use_hubeny in (false, true)
        d = Dict{String,Float64}()
        for T in [3500.0, 5778.0, 9000.0]
            for n_eff in [1.0, 2.0, 3.0, 3.5, 4.0, 10.0, 40.0]
                for (nH, nHe, ne) in [(1.0e16, 1.0e15, 1.0e13),
                                      (1.0e12, 1.0e11, 1.0e9),
                                      (1.0e18, 1.0e17, 1.0e15)]
                    k = join(string.([T, n_eff, nH, nHe, ne]), "_")
                    d[k] = Korg.hummer_mihalas_w(T, n_eff, nH, nHe, ne;
                                                 use_hubeny_generalization=use_hubeny)
                end
            end
        end
        hm[use_hubeny ? "hubeny" : "standard"] = d
    end
    sm["hummer_mihalas_w"] = hm

    # hummer_mihalas_U_H
    uh = Dict{String,Any}()
    for use_hubeny in (false, true)
        d = Dict{String,Float64}()
        for T in [3500.0, 5778.0, 9000.0]
            for (nH, nHe, ne) in [(1.0e16, 1.0e15, 1.0e13), (1.0e12, 1.0e11, 1.0e9)]
                k = join(string.([T, nH, nHe, ne]), "_")
                d[k] = Korg.hummer_mihalas_U_H(T, nH, nHe, ne;
                                               use_hubeny_generalization=use_hubeny)
            end
        end
        uh[use_hubeny ? "hubeny" : "standard"] = d
    end
    sm["hummer_mihalas_U_H"] = uh

    # saha_ion_weights across elements, including H (which has no wIII)
    saha = Dict{String,Any}()
    for Z in [1, 2, 6, 26, 92]
        d = Dict{String,Any}()
        for T in [3500.0, 5778.0, 9000.0, 12000.0]
            for ne in [1.0e10, 1.0e13, 1.0e16]
                wII, wIII = Korg.saha_ion_weights(T, ne, Z, Korg.ionization_energies,
                                                  Korg.default_partition_funcs)
                d[string(T) * "_" * string(ne)] = Dict("wII" => Float64(wII),
                                                       "wIII" => Float64(wIII))
            end
        end
        saha[string(Z)] = d
    end
    sm["saha_ion_weights"] = saha

    # get_log_nK for a handful of molecules (diatomic + polyatomic)
    nk = Dict{String,Any}()
    for mol_str in ["H2", "CO", "OH", "CN", "H2O", "CO2", "TiO"]
        mol = Korg.Species(mol_str)
        if haskey(Korg.default_log_equilibrium_constants, mol)
            nk[mol_str] = Dict(
                string(T) => Float64(Korg.get_log_nK(mol, T,
                                                     Korg.default_log_equilibrium_constants))
                for T in [3500.0, 5778.0, 9000.0])
        end
    end
    sm["get_log_nK"] = nk

    reference_data["statmech"] = sm
end

# =============================================================================
# 5. Full chemical equilibrium at a few conditions
# =============================================================================
println("  - chemical equilibrium ...")
let
    ce = Dict{String,Any}()
    A_X = Korg.format_A_X()
    # N_X/N_total, normalised to sum to 1 — matches Korg.px's A_X_to_absolute
    abs_abund = @. 10.0^(A_X - 12)
    abs_abund = abs_abund ./ sum(abs_abund)

    for (label, T, nt, ne_guess) in [("cool", 3500.0, 1.0e16, 1.0e10),
                                     ("solar", 5778.0, 1.0e16, 1.0e12),
                                     ("hot", 9000.0, 1.0e15, 1.0e13)]
        ne_sol, nds = Korg.chemical_equilibrium(T, nt, ne_guess, abs_abund,
                                                Korg.ionization_energies,
                                                Korg.default_partition_funcs,
                                                Korg.default_log_equilibrium_constants)
        wanted = ["H I", "H II", "H-", "He I", "He II", "C I", "C II", "N I", "O I",
                  "Na I", "Mg I", "Mg II", "Si I", "Ca I", "Ca II", "Fe I", "Fe II",
                  "H2", "CO", "OH", "H2O", "CN", "TiO"]
        dens = Dict{String,Float64}()
        for w in wanted
            sp = Korg.Species(w)
            if haskey(nds, sp)
                dens[w] = Float64(nds[sp])
            end
        end
        ce[label] = Dict("T" => T, "n_total" => nt, "ne_guess" => ne_guess,
                         "ne" => Float64(ne_sol), "number_densities" => dens)
    end
    reference_data["chemical_equilibrium"] = ce
end

# =============================================================================

# JSON has no representation for Inf/NaN; emit them as strings so the Python side
# can turn them back into floats with float().
sanitize(x::AbstractFloat) = isfinite(x) ? x : string(x)
sanitize(x::AbstractDict) = Dict(k => sanitize(v) for (k, v) in x)
sanitize(x::AbstractVector) = [sanitize(v) for v in x]
sanitize(x) = x

open(joinpath(@__DIR__, "chemistry_reference_data.json"), "w") do f
    JSON.print(f, sanitize(reference_data), 2)
end
println("Wrote tests/chemistry_reference_data.json")
