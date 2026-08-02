# Generate high-precision Korg.jl reference data for the Python synthesis and
# atmosphere modules.
#
#   export PATH="/mnt/sw/nix/store/yr11xz204lj9ah1irz61lbqh1dk0hcif-julia-1.11.2/bin:$PATH"
#   cd /mnt/home/acasey/software/Korg.px
#   julia --project=. tests/generate_synthesis_reference.jl
#
# Writes tests/synthesis_reference_data.json.
#
# This file is owned by the synthesis/atmosphere test suite.  It deliberately
# does NOT touch tests/generate_julia_reference.jl or
# tests/julia_reference_data.json, which are maintained separately.

using Korg
using JSON

const OUT = joinpath(@__DIR__, "synthesis_reference_data.json")
const SUNMOD = joinpath(@__DIR__, "data", "sun.mod")

ref = Dict{String,Any}()
ref["korg_version"] = string(pkgversion(Korg))

# ---------------------------------------------------------------------------
# 1. MARCS model atmosphere structure (deterministic file parse)
# ---------------------------------------------------------------------------
atm = Korg.read_model_atmosphere(SUNMOD)
ref["marcs_sun"] = Dict(
    "type" => string(nameof(typeof(atm))),
    "n_layers" => length(atm.layers),
    "reference_wavelength" => atm.reference_wavelength,
    "tau_ref" => [l.tau_ref for l in atm.layers],
    "z" => [l.z for l in atm.layers],
    "temp" => [l.temp for l in atm.layers],
    "electron_number_density" => [l.electron_number_density for l in atm.layers],
    "number_density" => [l.number_density for l in atm.layers],
)

# ---------------------------------------------------------------------------
# 2. blackbody / Planck function
# ---------------------------------------------------------------------------
bb_T = [3000.0, 4066.8, 5777.0, 9000.0, 25000.0]
bb_lambda = [1.5e-5, 3.0e-5, 5.0e-5, 6.5e-5, 1.2e-4, 5.0e-4]   # cm
ref["blackbody"] = Dict(
    "T" => bb_T,
    "lambda_cm" => bb_lambda,
    # B[i][j] = blackbody(T[i], lambda[j])
    "B" => [[Korg.blackbody(T, l) for l in bb_lambda] for T in bb_T],
)

# ---------------------------------------------------------------------------
# 3. Shell atmosphere geometry + photosphere correction
# ---------------------------------------------------------------------------
shell_R = [2.31e8, 3.0e9, 3.0e10, 6.957e10]
ref["shell_geometry"] = [
    begin
        sh = Korg.ShellAtmosphere(atm, R)
        radii = [sh.R + l.z for l in sh.layers]
        Dict("R" => R,
             "radii" => radii,
             "photosphere_correction" => radii[1]^2 / sh.R^2)
    end
    for R in shell_R
]

# ---------------------------------------------------------------------------
# 4. mu grids (Gauss-Legendre), used by the spherical solver
# ---------------------------------------------------------------------------
ref["mu_grids"] = Dict(
    string(n) => begin
        g, w = Korg.RadiativeTransfer.generate_mu_grid(n)
        Dict("mu" => collect(g), "weights" => collect(w))
    end
    for n in (5, 20)
)

# ---------------------------------------------------------------------------
# 5. End-to-end synthesis: planar and spherical, same layer data
# ---------------------------------------------------------------------------
A_X = Korg.format_A_X()
ref["A_X"] = A_X

wls = collect(5000.0:0.01:5001.0)          # Å, 101 points
ref["wavelengths"] = wls

# a single Fe I line inside the window
fe_line = Korg.Line(5000.5e-8, -1.5, Korg.species"Fe I", 3.0)
ref["line"] = Dict(
    "wl_cm" => fe_line.wl,
    "log_gf" => fe_line.log_gf,
    "species" => string(fe_line.species),
    "E_lower" => fe_line.E_lower,
    "gamma_rad" => fe_line.gamma_rad,
    "gamma_stark" => fe_line.gamma_stark,
    "vdW" => collect(fe_line.vdW),
)

function run_case(model, linelist)
    r = Korg.synthesize(model, linelist, A_X, (wls[1], wls[end]);
                        hydrogen_lines=false, vmic=1.0)
    Dict("flux" => collect(r.flux), "cntm" => collect(r.cntm),
         "wavelengths" => collect(r.wavelengths))
end

cases = Dict{String,Any}()
cases["planar_cntm"] = run_case(atm, typeof(fe_line)[])
cases["planar_line"] = run_case(atm, [fe_line])

# t/R ≈ 0.3 — a strongly extended giant-like geometry.  The photosphere
# correction is ~1.69 here, so an implementation that drops it is off by ~70%.
sh_extended = Korg.ShellAtmosphere(atm, 2.31e8)
cases["shell_extended_cntm"] = run_case(sh_extended, typeof(fe_line)[])
cases["shell_extended_line"] = run_case(sh_extended, [fe_line])
cases["shell_extended_R"] = 2.31e8

# A nearly-planar shell: the flux must approach the plane-parallel answer.
sh_thin = Korg.ShellAtmosphere(atm, 3.0e10)
cases["shell_thin_cntm"] = run_case(sh_thin, typeof(fe_line)[])
cases["shell_thin_R"] = 3.0e10

ref["synthesis"] = cases

open(OUT, "w") do io
    JSON.print(io, ref)
end
println("wrote $OUT")
