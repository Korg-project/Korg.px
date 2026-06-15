"""
Regression test for Barklem, Piskunov & O'Mara (2000) ABO p-d resonant
(self-)broadening of the lower Balmer lines (Hα/Hβ/Hγ, lower==2, upper in
{3,4,5}), which Korg.jl adds to the Stehlé & Hutcheon (1999) Stark profile.

Without this contribution the Python port underestimates the Balmer line cores
by roughly a factor of two. The reference HDF5 file was produced from Korg.jl
(see tests/gen_balmer_abo_reference.jl) by calling
``Korg.hydrogen_line_absorption!`` directly over Hα and Hβ windows.
"""

import os

import h5py
import numpy as np
import pytest

from korg.hydrogen_line_absorption import hydrogen_line_absorption

_REF = os.path.join(os.path.dirname(__file__), "data", "balmer_abo_reference.h5")

# Machine-precision agreement is expected; this is a comfortable bound.
_TOL = 1e-8


def _cases():
    with h5py.File(_REF, "r") as fid:
        names = list(fid.keys())
    return names


@pytest.mark.skipif(not os.path.exists(_REF), reason="Balmer ABO reference HDF5 missing")
@pytest.mark.parametrize("name", _cases())
@pytest.mark.parametrize("use_jit", [True, False])
def test_balmer_abo_matches_julia(name, use_jit):
    with h5py.File(_REF, "r") as fid:
        g = fid[name]
        wls = g["wavelengths"][:]
        alpha_jl = g["alpha"][:]
        T = float(g.attrs["T"])
        ne = float(g.attrs["ne"])
        nH_I = float(g.attrs["nH_I"])
        nHe_I = float(g.attrs["nHe_I"])
        UH_I = float(g.attrs["UH_I"])
        xi = float(g.attrs["xi"])
        lam0 = float(g.attrs["lambda0"])

    alpha_py = np.asarray(
        hydrogen_line_absorption(
            wls, T, ne, nH_I, nHe_I, UH_I, xi, 150e-8,
            use_MHD=True, use_jit=use_jit,
        )
    )

    # Core (nearest the ABO line centre): the previously-confirmed ~2x deficit
    # must be gone.
    icore = int(np.argmin(np.abs(wls - lam0)))
    assert alpha_jl[icore] > 0
    core_rel = abs(alpha_py[icore] - alpha_jl[icore]) / abs(alpha_jl[icore])
    assert core_rel < _TOL, f"{name}: core rel diff {core_rel:.2e}"

    # Whole window.
    denom = np.where(np.abs(alpha_jl) > 1e-30, np.abs(alpha_jl), np.nan)
    max_rel = np.nanmax(np.abs(alpha_py - alpha_jl) / denom)
    assert max_rel < _TOL, f"{name}: window max rel diff {max_rel:.2e}"
