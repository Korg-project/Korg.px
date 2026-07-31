#!/usr/bin/env julia
# Generate Julia reference data for the Balmer ABO p-d broadening case.
# Calls Korg.hydrogen_line_absorption! directly for Hα and Hβ windows.
# Output: tests/data/balmer_abo_reference.h5

using Pkg
project_dir = dirname(dirname(@__FILE__))
Pkg.activate(project_dir)
using Korg
using HDF5

const c_cgs = Korg.c_cgs

function compute_case(λ0_cm, halfwidth_cm, npts, T, ne, nH_I, nHe_I, UH_I, ξ)
    λstart = λ0_cm - halfwidth_cm
    λend = λ0_cm + halfwidth_cm
    # Korg.Wavelengths expects cm bounds
    λs = Korg.Wavelengths(range(λstart, λend; length=npts))
    αs = zeros(length(λs))
    Korg.hydrogen_line_absorption!(αs, λs, T, ne, nH_I, nHe_I, UH_I, ξ,
                                   150e-8)  # window_size 150 Å in cm
    return collect(λs), αs
end

outfile = joinpath(@__DIR__, "data", "balmer_abo_reference.h5")
mkpath(dirname(outfile))

# centers (ABO line centers, cm)
Hbeta_λ0 = 4.8626810200000004e-5
Halpha_λ0 = 6.56460998e-5
Hgamma_λ0 = 4.34168232e-5
halfwidth = 30e-8  # 30 Å each side
npts = 401

cases = [
    ("Hbeta_T6000",  Hbeta_λ0,  6000.0,  1e14),
    ("Hbeta_T10000", Hbeta_λ0,  10000.0, 1e14),
    ("Halpha_T6000", Halpha_λ0, 6000.0,  1e14),
    ("Halpha_T10000",Halpha_λ0, 10000.0, 1e14),
    ("Hbeta_T6000_ne1e13", Hbeta_λ0, 6000.0, 1e13),
    ("Hbeta_T6000_ne1e15", Hbeta_λ0, 6000.0, 1e15),
    ("Hgamma_T6000",  Hgamma_λ0, 6000.0,  1e14),
    ("Hgamma_T10000", Hgamma_λ0, 10000.0, 1e14),
    ("Hgamma_T6000_ne1e13", Hgamma_λ0, 6000.0, 1e13),
    ("Hgamma_T6000_ne1e15", Hgamma_λ0, 6000.0, 1e15),
]

nH_I = 1e16
nHe_I = 1e15
UH_I = 2.0
ξ = 1e5

h5open(outfile, "w") do fid
    for (name, λ0, T, ne) in cases
        wls, αs = compute_case(λ0, halfwidth, npts, T, ne, nH_I, nHe_I, UH_I, ξ)
        g = create_group(fid, name)
        g["wavelengths"] = wls
        g["alpha"] = αs
        attributes(g)["T"] = T
        attributes(g)["ne"] = ne
        attributes(g)["nH_I"] = nH_I
        attributes(g)["nHe_I"] = nHe_I
        attributes(g)["UH_I"] = UH_I
        attributes(g)["xi"] = ξ
        attributes(g)["lambda0"] = λ0
        println("  $name: max alpha = ", maximum(αs))
    end
    # Provenance: which Korg.jl produced this reference.
    attributes(fid)["korg_version"] = string(pkgversion(Korg))
    attributes(fid)["julia_version"] = string(VERSION)
end
println("Wrote ", outfile)
