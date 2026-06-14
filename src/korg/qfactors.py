"""
Q-factor and RV precision utilities.

Port of Korg.jl qfactors.jl. Computes spectral quality factors (Q) and
radial-velocity precision estimates following Bouchy et al. (2001).
"""

import numpy as np
from .constants import c_cgs


def Qfactor(synth_flux, synth_wl, obs_wl, LSF_mat, obs_mask=None):
    """
    Compute the Q factor from a high-resolution theoretical spectrum.

    Based on Bouchy et al. (2001, A&A, 374, 733).

    Parameters
    ----------
    synth_flux : array, shape (n_synth,)
        High-resolution theoretical (normalised) flux.
    synth_wl : array, shape (n_synth,)
        High-resolution wavelength grid in Å.
    obs_wl : array, shape (n_obs,)
        Low-resolution (observed) wavelength grid in Å.
    LSF_mat : array, shape (n_obs, n_synth)
        LSF matrix (see compute_LSF_matrix).
    obs_mask : array of bool, shape (n_obs,), optional
        Mask selecting pixels used in the computation.

    Returns
    -------
    float
        Q factor (dimensionless).
    """
    synth_flux = np.asarray(synth_flux, dtype=float)
    synth_wl = np.asarray(synth_wl, dtype=float)
    obs_wl = np.asarray(obs_wl, dtype=float)
    LSF_mat = np.asarray(LSF_mat, dtype=float)

    nvecLSF = LSF_mat.sum(axis=1)
    spec_lres = (LSF_mat @ synth_flux) / nvecLSF

    dspec_dlam = np.zeros(len(synth_flux))
    dspec_dlam[1:] = np.diff(synth_flux) / np.diff(synth_wl) * 1e-8  # Å → cm

    Wvec = ((obs_wl * (LSF_mat @ dspec_dlam) / nvecLSF) ** 2) / spec_lres

    if obs_mask is None:
        return float(np.sqrt(np.sum(Wvec) / np.sum(spec_lres)))
    else:
        obs_mask = np.asarray(obs_mask, dtype=bool)
        return float(np.sqrt(np.sum(Wvec[obs_mask]) / np.sum(spec_lres[obs_mask])))


def RV_prec_from_Q(Q, RMS_SNR, Npix):
    """
    Compute RV precision (m/s) from Q factor, SNR, and number of pixels.

    Parameters
    ----------
    Q : float
        Q factor of the spectrum.
    RMS_SNR : float
        Root-mean-squared per-pixel SNR.
    Npix : int or float
        Number of pixels in the spectrum.

    Returns
    -------
    float
        RV precision in m/s.
    """
    c_m_s = c_cgs * 1e-2  # cm/s → m/s
    return c_m_s / (Q * np.sqrt(Npix) * RMS_SNR)


def RV_prec_from_noise(synth_flux, synth_wl, obs_wl, LSF_mat, obs_err, obs_mask=None):
    """
    Compute best achievable RV precision given a spectrum with uncertainties.

    Parameters
    ----------
    synth_flux : array, shape (n_synth,)
        High-resolution theoretical (normalised) flux.
    synth_wl : array, shape (n_synth,)
        High-resolution wavelength grid in Å.
    obs_wl : array, shape (n_obs,)
        Low-resolution (observed) wavelength grid in Å.
    LSF_mat : array, shape (n_obs, n_synth)
        LSF matrix (see compute_LSF_matrix).
    obs_err : array, shape (n_obs,)
        Noise (1-σ uncertainty) in the continuum-normalised spectrum.
    obs_mask : array of bool, shape (n_obs,), optional
        Mask selecting pixels used in the computation.

    Returns
    -------
    float
        RV precision in m/s.
    """
    synth_flux = np.asarray(synth_flux, dtype=float)
    synth_wl = np.asarray(synth_wl, dtype=float)
    obs_wl = np.asarray(obs_wl, dtype=float)
    LSF_mat = np.asarray(LSF_mat, dtype=float)
    obs_err = np.asarray(obs_err, dtype=float)

    nvecLSF = LSF_mat.sum(axis=1)

    dspec_dlam = np.zeros(len(synth_flux))
    dspec_dlam[1:] = np.diff(synth_flux) / np.diff(synth_wl) * 1e-8  # Å → cm

    Wvec = ((obs_wl * (LSF_mat @ dspec_dlam) / nvecLSF) ** 2) / obs_err**2

    c_m_s = c_cgs * 1e-2  # cm/s → m/s
    if obs_mask is None:
        return c_m_s / np.sqrt(np.sum(Wvec))
    else:
        obs_mask = np.asarray(obs_mask, dtype=bool)
        return c_m_s / np.sqrt(np.sum(Wvec[obs_mask]))
