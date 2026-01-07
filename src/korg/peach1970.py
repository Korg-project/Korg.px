"""
Peach 1970 free-free departure coefficients.

This module contains departure coefficients from Peach+ 1970 for correcting
the hydrogenic free-free absorption coefficient for specific species.

The free-free absorption coefficient (including stimulated emission) is:
α_ff = α_hydrogenic(ν, T, n_i, n_e; Z) × (1 + D(T, σ))

where:
- α_hydrogenic should include stimulated emission correction
- n_i is the number density of the ion that participates in the interaction
- n_e is the number density of free electrons
- D(T, σ) is the departure coefficient
- σ is the photon energy in units of Rydberg × Z_eff²

Reference
---------
Peach+ 1970: https://ui.adsabs.harvard.edu/abs/1970MmRAS..73....1P
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


def _bilinear_interpolate_table(T, sigma, T_vals, sigma_vals, table_vals):
    """
    Bilinear interpolation of a 2D table with extrapolation to 0 outside bounds.

    Parameters
    ----------
    T : float or array
        Temperature in K
    sigma : float or array
        Photon energy in units of Rydberg × Z_eff²
    T_vals : array
        Temperature grid points
    sigma_vals : array
        Sigma grid points
    table_vals : 2D array
        Table values, shape (len(T_vals), len(sigma_vals))

    Returns
    -------
    float or array
        Interpolated departure coefficient (0 outside table bounds)
    """
    T = jnp.asarray(T)
    sigma = jnp.asarray(sigma)

    # Check if we're outside the table bounds
    T_min, T_max = float(T_vals[0]), float(T_vals[-1])
    sigma_min, sigma_max = float(sigma_vals[0]), float(sigma_vals[-1])

    out_of_bounds = (T < T_min) | (T > T_max) | (sigma < sigma_min) | (sigma > sigma_max)

    # Find indices in the grid
    i_T = jnp.searchsorted(T_vals, T, side='right') - 1
    i_sigma = jnp.searchsorted(sigma_vals, sigma, side='right') - 1

    # Clip to valid range
    i_T = jnp.clip(i_T, 0, len(T_vals) - 2)
    i_sigma = jnp.clip(i_sigma, 0, len(sigma_vals) - 2)

    # Get grid points
    T0 = T_vals[i_T]
    T1 = T_vals[i_T + 1]
    s0 = sigma_vals[i_sigma]
    s1 = sigma_vals[i_sigma + 1]

    # Compute fractional positions
    t_T = jnp.where(T1 != T0, (T - T0) / (T1 - T0), 0.0)
    t_sigma = jnp.where(s1 != s0, (sigma - s0) / (s1 - s0), 0.0)

    # Get corner values (table is [T, sigma])
    D00 = table_vals[i_T, i_sigma]
    D01 = table_vals[i_T, i_sigma + 1]
    D10 = table_vals[i_T + 1, i_sigma]
    D11 = table_vals[i_T + 1, i_sigma + 1]

    # Bilinear interpolation
    D_interp = ((1.0 - t_T) * (1.0 - t_sigma) * D00 +
                (1.0 - t_T) * t_sigma * D01 +
                t_T * (1.0 - t_sigma) * D10 +
                t_T * t_sigma * D11)

    # Return 0 outside bounds
    return jnp.where(out_of_bounds, 0.0, D_interp)


# He II departure coefficients (neutral Helium free-free)
# From Table III of Peach 1970
_He_II_sigma_vals = jnp.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30], dtype=jnp.float64)

_He_II_T_vals = jnp.array([
    10000.0, 11000.0, 12000.0, 13000.0, 14000.0, 15000.0, 16000.0, 17000.0, 18000.0,
    19000.0, 20000.0, 21000.0, 22000.0, 23000.0, 24000.0, 25000.0, 26000.0, 27000.0,
    28000.0, 29000.0, 30000.0, 32000.0, 34000.0, 36000.0, 38000.0, 40000.0, 42000.0,
    44000.0, 46000.0, 48000.0
], dtype=jnp.float64)

_He_II_table = jnp.array([
    [0.016, 0.039, 0.069, 0.100, 0.135, 0.169],
    [0.018, 0.041, 0.071, 0.103, 0.137, 0.172],
    [0.020, 0.043, 0.073, 0.105, 0.139, 0.174],
    [0.022, 0.045, 0.075, 0.107, 0.142, 0.176],
    [0.024, 0.047, 0.078, 0.109, 0.144, 0.179],
    [0.026, 0.050, 0.080, 0.112, 0.146, 0.181],
    [0.028, 0.052, 0.082, 0.114, 0.148, 0.183],
    [0.029, 0.054, 0.084, 0.116, 0.151, 0.185],
    [0.031, 0.056, 0.086, 0.118, 0.153, 0.187],
    [0.033, 0.058, 0.088, 0.120, 0.155, 0.190],
    [0.035, 0.060, 0.090, 0.122, 0.157, 0.192],
    [0.037, 0.062, 0.092, 0.125, 0.159, 0.194],
    [0.039, 0.064, 0.095, 0.127, 0.162, 0.196],
    [0.041, 0.066, 0.097, 0.129, 0.164, 0.198],
    [0.043, 0.068, 0.099, 0.131, 0.166, 0.201],
    [0.045, 0.070, 0.101, 0.133, 0.168, 0.203],
    [0.047, 0.072, 0.103, 0.135, 0.170, 0.205],
    [0.049, 0.074, 0.105, 0.138, 0.173, 0.207],
    [0.050, 0.076, 0.107, 0.140, 0.175, 0.209],
    [0.052, 0.079, 0.109, 0.142, 0.177, 0.211],
    [0.054, 0.081, 0.111, 0.144, 0.179, 0.214],
    [0.058, 0.085, 0.115, 0.148, 0.183, 0.218],
    [0.062, 0.089, 0.119, 0.153, 0.188, 0.222],
    [0.065, 0.093, 0.124, 0.157, 0.102, 0.226],
    [0.069, 0.096, 0.128, 0.161, 0.196, 0.230],
    [0.072, 0.100, 0.132, 0.165, 0.200, 0.235],
    [0.076, 0.104, 0.135, 0.169, 0.204, 0.239],
    [0.079, 0.108, 0.139, 0.173, 0.208, 0.243],
    [0.082, 0.111, 0.143, 0.177, 0.212, 0.247],
    [0.085, 0.115, 0.147, 0.181, 0.216, 0.251]
], dtype=jnp.float64)


def D_He_II(T, sigma):
    """
    Departure coefficient for He II (neutral Helium) free-free absorption.

    Parameters
    ----------
    T : float or array
        Temperature in K
    sigma : float or array
        Photon energy in units of Rydberg × Z_eff²

    Returns
    -------
    float or array
        Departure coefficient D (0 outside valid range)

    Notes
    -----
    Valid for T = 10000-48000 K, σ = 0.05-0.30
    """
    return _bilinear_interpolate_table(T, sigma, _He_II_T_vals, _He_II_sigma_vals, _He_II_table)


# C II departure coefficients (neutral Carbon free-free)
# From Table III of Peach 1970
_C_II_sigma_vals = jnp.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30], dtype=jnp.float64)

_C_II_T_vals = jnp.array([
    4000.0, 5000.0, 6000.0, 7000.0, 8000.0, 9000.0, 10000.0, 11000.0, 12000.0, 13000.0,
    14000.0, 15000.0, 16000.0, 17000.0, 18000.0, 19000.0, 20000.0, 21000.0, 22000.0,
    23000.0, 24000.0, 25000.0, 26000.0, 27000.0, 28000.0, 29000.0, 30000.0, 32000.0,
    34000.0, 36000.0
], dtype=jnp.float64)

_C_II_table = jnp.array([
    [-0.145, -0.144, -0.068, 0.054, 0.200, 0.394],
    [-0.132, -0.124, -0.045, 0.077, 0.222, 0.415],
    [-0.121, -0.109, -0.027, 0.097, 0.244, 0.438],
    [-0.112, -0.095, -0.010, 0.115, 0.264, 0.461],
    [-0.104, -0.082, 0.005, 0.133, 0.284, 0.484],
    [-0.095, -0.070, 0.020, 0.150, 0.303, 0.507],
    [-0.087, -0.058, 0.034, 0.166, 0.321, 0.529],
    [-0.079, -0.047, 0.048, 0.181, 0.339, 0.550],
    [-0.071, -0.036, 0.061, 0.196, 0.356, 0.570],
    [-0.063, -0.025, 0.074, 0.210, 0.372, 0.590],
    [-0.055, -0.015, 0.086, 0.223, 0.388, 0.609],
    [-0.047, -0.005, 0.098, 0.237, 0.403, 0.628],
    [-0.040, 0.005, 0.109, 0.249, 0.418, 0.646],
    [-0.032, 0.015, 0.120, 0.261, 0.432, 0.664],
    [-0.025, 0.024, 0.131, 0.273, 0.446, 0.680],
    [-0.017, 0.034, 0.141, 0.285, 0.459, 0.697],
    [-0.010, 0.043, 0.152, 0.296, 0.472, 0.713],
    [-0.003, 0.051, 0.161, 0.307, 0.485, 0.728],
    [0.004, 0.060, 0.171, 0.317, 0.497, 0.744],
    [0.011, 0.069, 0.181, 0.327, 0.509, 0.758],
    [0.018, 0.077, 0.100, 0.337, 0.521, 0.773],
    [0.025, 0.085, 0.109, 0.347, 0.532, 0.787],
    [0.032, 0.093, 0.208, 0.356, 0.543, 0.800],
    [0.039, 0.101, 0.216, 0.365, 0.554, 0.814],
    [0.046, 0.109, 0.225, 0.374, 0.564, 0.827],
    [0.052, 0.117, 0.233, 0.383, 0.574, 0.839],
    [0.059, 0.124, 0.241, 0.391, 0.585, 0.852],
    [0.072, 0.139, 0.257, 0.408, 0.604, 0.876],
    [0.085, 0.154, 0.273, 0.424, 0.623, 0.900],
    [0.097, 0.168, 0.288, 0.439, 0.641, 0.923]
], dtype=jnp.float64)


def D_C_II(T, sigma):
    """
    Departure coefficient for C II (neutral Carbon) free-free absorption.

    Parameters
    ----------
    T : float or array
        Temperature in K
    sigma : float or array
        Photon energy in units of Rydberg × Z_eff²

    Returns
    -------
    float or array
        Departure coefficient D (0 outside valid range)

    Notes
    -----
    Valid for T = 4000-36000 K, σ = 0.05-0.30
    """
    return _bilinear_interpolate_table(T, sigma, _C_II_T_vals, _C_II_sigma_vals, _C_II_table)


# Si II departure coefficients (neutral Silicon free-free)
# From Table III of Peach 1970
_Si_II_sigma_vals = jnp.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30], dtype=jnp.float64)

_Si_II_T_vals = jnp.array([
    4000.0, 5000.0, 6000.0, 7000.0, 8000.0, 9000.0, 10000.0, 11000.0, 12000.0, 13000.0,
    14000.0, 15000.0, 16000.0, 17000.0, 18000.0, 19000.0, 20000.0, 21000.0, 22000.0,
    23000.0, 24000.0, 25000.0, 26000.0, 27000.0, 28000.0, 29000.0, 30000.0, 32000.0,
    34000.0, 36000.0
], dtype=jnp.float64)

_Si_II_table = jnp.array([
    [-0.079, 0.033, 0.214, 0.434, 0.650, 0.973],
    [-0.066, 0.042, 0.216, 0.429, 0.642, 0.062],
    [-0.056, 0.050, 0.220, 0.430, 0.643, 0.965],
    [-0.048, 0.057, 0.224, 0.433, 0.648, 0.974],
    [-0.040, 0.063, 0.229, 0.436, 0.653, 0.081],
    [-0.033, 0.069, 0.233, 0.440, 0.659, 0.995],
    [-0.027, 0.074, 0.238, 0.444, 0.666, 1.007],
    [-0.021, 0.080, 0.242, 0.448, 0.672, 1.019],
    [-0.015, 0.085, 0.246, 0.452, 0.679, 1.031],
    [-0.010, 0.089, 0.250, 0.456, 0.685, 1.042],
    [-0.004, 0.094, 0.254, 0.459, 0.692, 1.054],
    [0.001, 0.009, 0.258, 0.463, 0.698, 1.065],
    [0.006, 0.103, 0.262, 0.467, 0.705, 1.076],
    [0.011, 0.107, 0.265, 0.471, 0.711, 1.087],
    [0.016, 0.112, 0.269, 0.474, 0.717, 1.097],
    [0.021, 0.116, 0.273, 0.478, 0.724, 1.108],
    [0.026, 0.120, 0.277, 0.482, 0.730, 1.118],
    [0.030, 0.125, 0.281, 0.486, 0.736, 1.127],
    [0.035, 0.129, 0.285, 0.490, 0.742, 1.137],
    [0.040, 0.134, 0.289, 0.493, 0.747, 1.146],
    [0.045, 0.138, 0.293, 0.497, 0.753, 1.155],
    [0.050, 0.143, 0.297, 0.501, 0.759, 1.164],
    [0.055, 0.147, 0.301, 0.505, 0.765, 1.173],
    [0.060, 0.152, 0.305, 0.509, 0.770, 1.181],
    [0.065, 0.156, 0.310, 0.513, 0.776, 1.189],
    [0.071, 0.161, 0.314, 0.517, 0.781, 1.197],
    [0.076, 0.166, 0.318, 0.520, 0.787, 1.205],
    [0.087, 0.176, 0.328, 0.528, 0.798, 1.221],
    [0.008, 0.186, 0.317, 0.537, 0.809, 1.236],
    [0.109, 0.196, 0.346, 0.545, 0.819, 1.251]
], dtype=jnp.float64)


def D_Si_II(T, sigma):
    """
    Departure coefficient for Si II (neutral Silicon) free-free absorption.

    Parameters
    ----------
    T : float or array
        Temperature in K
    sigma : float or array
        Photon energy in units of Rydberg × Z_eff²

    Returns
    -------
    float or array
        Departure coefficient D (0 outside valid range)

    Notes
    -----
    Valid for T = 4000-36000 K, σ = 0.05-0.30
    """
    return _bilinear_interpolate_table(T, sigma, _Si_II_T_vals, _Si_II_sigma_vals, _Si_II_table)


# Mg II departure coefficients (neutral Magnesium free-free)
# From Table III of Peach 1970
_Mg_II_sigma_vals = jnp.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30], dtype=jnp.float64)

_Mg_II_T_vals = jnp.array([
    4000.0, 5000.0, 6000.0, 7000.0, 8000.0, 9000.0, 10000.0, 11000.0, 12000.0, 13000.0,
    14000.0, 15000.0, 16000.0, 17000.0, 18000.0, 19000.0, 20000.0, 21000.0, 22000.0,
    23000.0, 24000.0, 25000.0, 26000.0, 27000.0, 28000.0, 29000.0, 30000.0, 32000.0,
    34000.0
], dtype=jnp.float64)

_Mg_II_table = jnp.array([
    [-0.070, 0.008, 0.121, 0.221, 0.274, 0.356],
    [-0.067, 0.003, 0.104, 0.105, 0.244, 0.325],
    [-0.066, -0.002, 0.091, 0.175, 0.221, 0.302],
    [-0.065, -0.007, 0.080, 0.157, 0.201, 0.282],
    [-0.065, -0.012, 0.069, 0.141, 0.183, 0.264],
    [-0.065, -0.016, 0.059, 0.126, 0.166, 0.248],
    [-0.065, -0.020, 0.049, 0.113, 0.151, 0.232],
    [-0.066, -0.024, 0.040, 0.100, 0.137, 0.218],
    [-0.066, -0.028, 0.032, 0.088, 0.124, 0.205],
    [-0.066, -0.032, 0.025, 0.077, 0.112, 0.194],
    [-0.066, -0.035, 0.018, 0.067, 0.101, 0.183],
    [-0.066, -0.037, 0.012, 0.058, 0.091, 0.173],
    [-0.066, -0.040, 0.006, 0.049, 0.082, 0.164],
    [-0.066, -0.042, 0.001, 0.042, 0.074, 0.157],
    [-0.066, -0.044, -0.004, 0.036, 0.067, 0.150],
    [-0.065, -0.045, -0.007, 0.030, 0.061, 0.144],
    [-0.064, -0.046, -0.011, 0.025, 0.056, 0.139],
    [-0.063, -0.047, -0.014, 0.020, 0.051, 0.135],
    [-0.062, -0.048, -0.016, 0.017, 0.048, 0.131],
    [-0.061, -0.048, -0.018, 0.014, 0.045, 0.128],
    [-0.059, -0.047, -0.019, 0.011, 0.042, 0.126],
    [-0.057, -0.047, -0.020, 0.009, 0.040, 0.124],
    [-0.055, -0.046, -0.020, 0.008, 0.039, 0.123],
    [-0.053, -0.045, -0.021, 0.007, 0.038, 0.123],
    [-0.051, -0.044, -0.020, 0.006, 0.038, 0.123],
    [-0.048, -0.042, -0.020, 0.006, 0.038, 0.123],
    [-0.045, -0.040, -0.019, 0.006, 0.039, 0.124],
    [-0.039, -0.035, -0.016, 0.008, 0.042, 0.128],
    [-0.032, -0.030, -0.012, 0.011, 0.046, 0.133]
], dtype=jnp.float64)


def D_Mg_II(T, sigma):
    """
    Departure coefficient for Mg II (neutral Magnesium) free-free absorption.

    Parameters
    ----------
    T : float or array
        Temperature in K
    sigma : float or array
        Photon energy in units of Rydberg × Z_eff²

    Returns
    -------
    float or array
        Departure coefficient D (0 outside valid range)

    Notes
    -----
    Valid for T = 4000-34000 K, σ = 0.05-0.30
    """
    return _bilinear_interpolate_table(T, sigma, _Mg_II_T_vals, _Mg_II_sigma_vals, _Mg_II_table)


# Dictionary mapping species names to departure coefficient functions
DEPARTURE_COEFFICIENTS = {
    'He_II': D_He_II,
    'C_II': D_C_II,
    'Si_II': D_Si_II,
    'Mg_II': D_Mg_II
}
