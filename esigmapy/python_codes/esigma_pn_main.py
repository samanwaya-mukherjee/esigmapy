# Translated from LALSimESIGMA.c by Samanwaya Mukherjee, 2026

import sys
import numpy as np
from scipy.interpolate import CubicSpline
from numba import njit, cfunc, carray
from numbalsoda import lsoda_sig, lsoda, dop853
import math
from numba import njit
from .esigma_pn_inspiral import *
from .esigma_go_terms import *
import lal

# Constants (LAL equivalents)
LAL_PI = lal.PI 
LAL_MTSUN_SI = lal.MTSUN_SI #4.925491025543576e-6, Solar mass in seconds
LAL_MRSUN_SI = lal.MRSUN_SI #1.476625061404649e3   # solar mass in metres

RadiationPNOrderDefault = 8  # Default radiation reaction PN order (3PN)

import os


@cfunc(lsoda_sig)
def rhs_cfunc(t, u, du, p):
    params = carray(p, (8,))
    eta_val = params[0]
    m1_val = params[1]
    m2_val = params[2]
    S1z_val = params[3]
    S2z_val = params[4]
    rad_pn_order_val = int(params[5])
    x_dot_4pn_SF_val = params[6]
    x_final = params[7]

    # Smoothly cap x to x_final to prevent domain errors (like sqrt(negative)) past ISCO,
    # while keeping the derivative continuous (it will just be constant past x_final).
    u_eval = np.empty(4, dtype=np.float64)
    u_eval[0] = u[0] if u[0] < x_final else x_final
    u_eval[1] = u[1]
    u_eval[2] = u[2]
    u_eval[3] = u[3]

    dydt = eccentric_x_model_odes(
        t,
        u_eval,
        eta_val,
        m1_val,
        m2_val,
        S1z_val,
        S2z_val,
        rad_pn_order_val,
        x_dot_4pn_SF_val,
    )

    du[0] = dydt[0]
    du[1] = dydt[1]
    du[2] = dydt[2]
    du[3] = dydt[3]


@njit(cache=True)
def integrate_to_isco_numbalsoda(funcptr, y0, params_array, dt, x_final, max_samples, ode_eps):
    chunk_size = 1024

    t_arr = np.empty(max_samples, dtype=np.float64)
    y_arr = np.empty((max_samples, 4), dtype=np.float64)

    t_arr[0] = 0.0
    y_arr[0, 0] = y0[0]
    y_arr[0, 1] = y0[1]
    y_arr[0, 2] = y0[2]
    y_arr[0, 3] = y0[3]

    t_curr = 0.0
    u_curr = y0.copy()

    idx = 1
    bad_number = False

    while idx < max_samples:
        t_eval = np.empty(chunk_size + 1, dtype=np.float64)
        for i in range(chunk_size + 1):
            t_eval[i] = t_curr + i * dt

        usol, success = lsoda(
            funcptr,
            u_curr,
            t_eval,
            data=params_array,
            rtol=ode_eps,
            atol=ode_eps,
            mxstep=5000000,
        )

        if not success:
            bad_number = True
            break

        crossed = False
        for i in range(1, chunk_size + 1):
            t_arr[idx] = t_eval[i]
            y_arr[idx, 0] = usol[i, 0]
            y_arr[idx, 1] = usol[i, 1]
            y_arr[idx, 2] = usol[i, 2]
            y_arr[idx, 3] = usol[i, 3]

            if np.isnan(usol[i, 0]) or np.isnan(usol[i, 1]):
                bad_number = True
                crossed = True
                break

            idx += 1
            if usol[i, 0] >= x_final:
                crossed = True
                break

        if crossed:
            break

        u_curr = usol[-1].copy()
        t_curr = t_eval[-1]

    return t_arr[:idx], y_arr[:idx], bad_number

@njit(cache=True)
def integrate_to_isco_numbalsoda_dop853(funcptr, y0, params_array, dt, x_final, max_samples, ode_eps):
    chunk_size = 1024

    t_arr = np.empty(max_samples, dtype=np.float64)
    y_arr = np.empty((max_samples, 4), dtype=np.float64)

    t_arr[0] = 0.0
    y_arr[0, 0] = y0[0]
    y_arr[0, 1] = y0[1]
    y_arr[0, 2] = y0[2]
    y_arr[0, 3] = y0[3]

    t_curr = 0.0
    u_curr = y0.copy()

    idx = 1
    bad_number = False

    while idx < max_samples:
        t_eval = np.empty(chunk_size + 1, dtype=np.float64)
        for i in range(chunk_size + 1):
            t_eval[i] = t_curr + i * dt

        usol, success = dop853(
            funcptr,
            u_curr,
            t_eval,
            data=params_array,
            rtol=ode_eps,
            atol=ode_eps,
            mxstep=5000000,
        )

        if not success:
            bad_number = True
            break

        crossed = False
        for i in range(1, chunk_size + 1):
            t_arr[idx] = t_eval[i]
            y_arr[idx, 0] = usol[i, 0]
            y_arr[idx, 1] = usol[i, 1]
            y_arr[idx, 2] = usol[i, 2]
            y_arr[idx, 3] = usol[i, 3]

            if np.isnan(usol[i, 0]) or np.isnan(usol[i, 1]):
                bad_number = True
                crossed = True
                break

            idx += 1
            if usol[i, 0] >= x_final:
                crossed = True
                break

        if crossed:
            break

        u_curr = usol[-1].copy()
        t_curr = t_eval[-1]

    return t_arr[:idx], y_arr[:idx], bad_number



# from dataclasses import dataclass, field

# ------------------------------------------------------------------ #
# Constants / defaults (set these to match your C macros)
# ------------------------------------------------------------------ #

from dataclasses import dataclass


@dataclass
class Params:
    eta: float
    radiation_pn_order: int
    m1: float
    m2: float
    S1z: float
    S2z: float


# ------------------------------------------------------------------ #
# 1.  JIT Helpers
# ------------------------------------------------------------------ #


@njit(cache=True)
def compute_state_arrays(x_arr, e_arr, l_arr, eta, mass1, mass2, S1z, S2z):
    n = len(x_arr)
    u_arr = np.empty(n, dtype=np.float64)
    r_arr = np.empty(n, dtype=np.float64)
    phi_dot_arr = np.empty(n, dtype=np.float64)
    for i in range(n):
        ui = pn_kepler_equation(eta, x_arr[i], e_arr[i], l_arr[i])
        ri = separation(ui, eta, x_arr[i], e_arr[i], mass1, mass2, S1z, S2z)
        u_arr[i] = ui
        r_arr[i] = ri
        phi_dot_arr[i] = dphi_dt(ui, eta, mass1, mass2, S1z, S2z, x_arr[i], e_arr[i])
    return u_arr, r_arr, phi_dot_arr


@njit(cache=True)
def compute_mode_from_dynamics(
    l: int,
    m: int,
    x_vec: np.ndarray,  # PN expansion parameter (length N)
    phi_vec: np.ndarray,  # orbital phase
    phi_dot_vec: np.ndarray,  # d(phi)/dt
    r_vec: np.ndarray,  # orbital separation
    r_dot_vec: np.ndarray,  # d(r)/dt
    mass1: float,
    mass2: float,
    S1z: float,
    S2z: float,
    R: float,  # source distance (m)
    vpnorder: int,
) -> np.ndarray:  # complex128, length N
    """
    Compute the (l, m) spin-weighted spherical harmonic mode h_lm
    at every time step from the orbital dynamics arrays.

    Mirrors the static C function compute_mode_from_dynamics().
    """
    total_mass = mass1 + mass2
    eta = (mass1 * mass2) / total_mass**2

    # kv = _build_kepler_vars(eta, total_mass, S1z, S2z)

    length = len(x_vec)
    h_lm = np.zeros(length, dtype=np.complex128)

    for i in range(length):
        # populate_kepler_params(
        #     kv,
        #     e=0.0,
        #     x=x_vec[i],
        #     r=r_vec[i] * total_mass,
        #     r_dot=r_dot_vec[i],
        #     phi_dot=phi_dot_vec[i] / total_mass,
        # )
        h_lm[i] = (
            hlmGOresult(
                l,
                m,
                total_mass,
                eta,
                r_vec[i] * total_mass,
                r_dot_vec[i],
                phi_vec[i],
                phi_dot_vec[i] / total_mass,
                R,
                vpnorder,
                S1z,
                S2z,
                x_vec[i],
            )
            * LAL_MRSUN_SI
        )

    return h_lm


# ------------------------------------------------------------------ #
# compute_strain_from_dynamics
# ------------------------------------------------------------------ #
def compute_strain_from_dynamics(
    x_vec: np.ndarray,
    phi_vec: np.ndarray,
    phi_dot_vec: np.ndarray,
    r_vec: np.ndarray,
    r_dot_vec: np.ndarray,
    mass1: float,
    mass2: float,
    S1z: float,
    S2z: float,
    euler_iota: float,
    euler_beta: float,
    R:          float,
    vpnorder:   int,
    ONLY_LeqM_MODES: bool,
    L_MIN: int = 2,
    L_MAX: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sum all (l, m) modes weighted by spin-weighted spherical harmonics
    to produce the + and x strain polarizations.

    Mirrors the static C function compute_strain_from_dynamics().
    """
    length = len(x_vec)
    h_plus = np.zeros(length, dtype=float)
    h_cross = np.zeros(length, dtype=float)

    for ell in range(L_MIN, L_MAX + 1):
        for em in range(-ell, ell + 1):

            if ONLY_LeqM_MODES and ell != abs(em):
                continue

            hlm = compute_mode_from_dynamics(
                ell,
                em,
                x_vec,
                phi_vec,
                phi_dot_vec,
                r_vec,
                r_dot_vec,
                mass1,
                mass2,
                S1z,
                S2z,
                R,
                vpnorder,
            )

            ylm = lal.SpinWeightedSphericalHarmonic(euler_iota, euler_beta, -2, ell, em)

            hlm_times_ylm = hlm * ylm
            h_plus += hlm_times_ylm.real
            h_cross -= hlm_times_ylm.imag

    return h_plus, h_cross


# ------------------------------------------------------------------ #
# XLALSimInspiralesigmaModeFromDynamics  (public)
# ------------------------------------------------------------------ #
def inspiral_esigma_mode_from_dynamics(
    l: int,
    m: int,
    t_vector: np.ndarray,
    x_vector: np.ndarray,
    phi_vector: np.ndarray,
    phi_dot_vector: np.ndarray,
    r_vector: np.ndarray,
    r_dot_vector: np.ndarray,
    mass1: float,
    mass2: float,
    S1z:   float,
    S2z:   float,
    R:     float,
    mode_pn_order: int,
) -> np.ndarray:               # complex128
    """
    Public wrapper: compute a single (l, m) waveform mode from dynamics.

    Note: in the C version this modifies t_vec, r_vec, phi_dot_vec in place
    by scaling by total_mass.  Here we keep the arrays immutable and do the
    scaling inside compute_mode_from_dynamics (matches the C end result).
    """
    return compute_mode_from_dynamics(
        l,
        m,
        x_vector,
        phi_vector,
        phi_dot_vector,
        r_vector,
        r_dot_vector,
        mass1,
        mass2,
        S1z,
        S2z,
        R,
        mode_pn_order,
    )


# ------------------------------------------------------------------ #
#  XLALSimInspiralesigmaStrainFromDynamics  (public)
# ------------------------------------------------------------------ #
def esigma_strain_from_dynamics(
    t_vector: np.ndarray,
    x_vector: np.ndarray,
    phi_vector: np.ndarray,
    phi_dot_vector: np.ndarray,
    r_vector: np.ndarray,
    r_dot_vector: np.ndarray,
    mass1: float,
    mass2: float,
    S1z: float,
    S2z: float,
    euler_iota: float,
    euler_beta: float,
    R:          float,
    mode_pn_order: int,
    ONLY_LeqM_MODES: bool = False,
    L_MIN: int = 2,
    L_MAX: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Public wrapper: compute h+ and hx polarizations from dynamics arrays.
    Returns (h_plus, h_cross) as 1-D float64 arrays.
    """
    # mode_pn_order = int(os.environ.get("ModePNOrder", ModePNOrderDefault))

    return compute_strain_from_dynamics(
        x_vector, phi_vector, phi_dot_vector,
        r_vector, r_dot_vector,
        mass1, mass2, S1z, S2z,
        euler_iota, euler_beta, R, mode_pn_order, ONLY_LeqM_MODES, L_MIN, L_MAX
    )


# ------------------------------------------------------------------ #
# x_model_eccbbh_inspiral_waveform  (internal, called by esigma)
# ------------------------------------------------------------------ #
def x_model_eccbbh_inspiral_waveform(
    mass1: float,  # solar masses
    mass2: float,
    S1z: float,
    S2z: float,
    e_init: float,
    f_gw_init: float,  # Hz
    distance: float,  # metres
    mean_anom_init: float,
    ode_eps:        float,
    euler_iota:     float,
    euler_beta:     float,
    sampling_rate:  float,   # Hz
    ONLY_LeqM_MODES: bool,
    L_MIN: int,
    L_MAX: int,
    mode_pn_order: int,
    integrator: str = "lsoda",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Drive the full esigma inspiral:
      1. integrate orbital dynamics,
      2. project onto polarizations.

    Returns (h_plus, h_cross) as float64 arrays.
    """
    # --- Step 1: orbital dynamics ----------------------------------- #
    dyn = inspiral_esigma_dynamics(
        mass1,
        mass2,
        S1z,
        S2z,
        e_init,
        f_gw_init,
        mean_anom_init,
        ode_eps,
        sampling_rate,
        integrator,
    )

    # --- Step 2: strain from dynamics ------------------------------- #
    h_plus, h_cross = esigma_strain_from_dynamics(
        dyn["time_evol"],
        dyn["x_evol"],
        dyn["phi_evol"],
        dyn["phi_dot_evol"],
        dyn["r_evol"],
        dyn["r_dot_evol"],
        mass1, mass2, S1z, S2z,
        euler_iota, euler_beta, distance,
        mode_pn_order,
        ONLY_LeqM_MODES,
        L_MIN,
        L_MAX
    )

    return h_plus, h_cross


def inspiral_esigma_dynamics(
    mass1,  # mass1 in solar mass
    mass2,  # mass2 in solar mass
    S1z,  # z-component of spin of companion 1
    S2z,  # z-component of spin of companion 2
    e_init,  # initial eccentricity
    f_gw_init,  # initial GW frequency
    mean_anom_init,  # initial mean anomaly
    ode_eps,  # tolerance (relative)
    sampling_rate,  # sample rate in Hz
    abs_tol = 1e-17, # absolute tolerance
    solve_ivp_method = "RK45", # method for solve_ivp module
    rad_pn_order = 8, # Radiation PN order
    inspiral_end_radius = 4.0, # Inspiral end radius (in units of total mass)   
    integrator="lsoda",
):
    """
    Compute ESIGMA orbital dynamics via ODE integration, then interpolate
    to a uniform time grid.

    Returns a dict with keys:
        time_evol, x_evol, eccentricity_evol, mean_ano_evol,
        phi_evol, phi_dot_evol, r_evol, r_dot_evol
    Each value is a 1D numpy array sampled at 1/sampling_rate intervals.
    """

    # ------------------------------------------------------------------ #
    # Input validation
    # ------------------------------------------------------------------ #
    if not (0.0 <= e_init < 1.0):
        raise ValueError("Invalid eccentricity, must be in range [0, 1)")

    # ------------------------------------------------------------------ #
    # Mass / PN bookkeeping
    # ------------------------------------------------------------------ #
    total_mass = mass1 + mass2
    reduced_mass = mass1 * mass2 / total_mass
    eta = reduced_mass / total_mass  # symmetric mass ratio

    omega_init = LAL_PI * f_gw_init * LAL_MTSUN_SI
    x_init = (total_mass * omega_init) ** (2.0 / 3.0)

    # PN / radiation order (mirror the C env-var logic; default hardcoded)
    # rad_pn_order = int(os.environ.get("RadiationPNOrder", RadiationPNOrderDefault))
    params = Params(
        eta=eta,
        radiation_pn_order=rad_pn_order,
        m1=mass1,
        m2=mass2,
        S1z=S1z,
        S2z=S2z,
    )

    # ------------------------------------------------------------------ #
    # Termination condition: ISCO
    # ------------------------------------------------------------------ #
    TRANS = inspiral_end_radius
    f_gw_isco = 1.0 / (TRANS * math.sqrt(TRANS) * LAL_PI * total_mass)
    x_final = (LAL_PI * total_mass * f_gw_isco) ** (2.0 / 3.0)

    # ------------------------------------------------------------------ #
    # Time step in geometric units
    # ------------------------------------------------------------------ #
    dt_sec = 1.0 / sampling_rate  # seconds
    dt = dt_sec / (total_mass * LAL_MTSUN_SI)  # geometric (M)

    # Precompute the 4PN Self-Force term (constant over integration)
    from esigmapy.python_codes.esigma_pn_inspiral import x_dot_4pn_SF

    x_dot_4pn_SF_val = x_dot_4pn_SF(e_init, eta, S1z)

    # ------------------------------------------------------------------ #
    # Initial conditions  y = [x, e, l (mean anomaly), phi]
    # ------------------------------------------------------------------ #

    y0 = np.array([x_init, e_init, mean_anom_init, 0.0])

    MAX_SAMPLES = (
        2048 * 16384
    )  # maximum number of samples to prevent infinite loops; adjust as needed

    # =========== ODE solver using numbalsoda =========================#

    params_array = np.array(
        [eta, mass1, mass2, S1z, S2z, rad_pn_order, x_dot_4pn_SF_val, x_final],
        dtype=np.float64,
    )

    if integrator not in ["lsoda", "dop853"]:
        raise ValueError("Invalid integrator. Must be 'lsoda' or 'dop853'")

    if integrator == "lsoda":
        t_arr, y_arr, bad_number = integrate_to_isco_numbalsoda(
            rhs_cfunc.address, y0, params_array, dt, x_final, MAX_SAMPLES, ode_eps
        )
    else:
        t_arr, y_arr, bad_number = integrate_to_isco_numbalsoda_dop853(
            rhs_cfunc.address, y0, params_array, dt, x_final, MAX_SAMPLES, ode_eps
        )

    # Unpack variables
    x_arr = y_arr[:, 0]
    e_arr = y_arr[:, 1]
    l_arr = y_arr[:, 2]
    phi_arr = y_arr[:, 3]

    u_arr, r_arr, phi_dot_arr = compute_state_arrays(
        x_arr, e_arr, l_arr, eta, mass1, mass2, S1z, S2z
    )

    final_i = len(t_arr)
    if final_i < 4:
        raise RuntimeError(
            "Integration produced fewer than 4 points; cannot interpolate."
        )
    elif bad_number:
        raise ValueError("Infinity or nan encountered!")

    # ------------------------------------------------------------------ #
    # Uniform-grid interpolation
    # ------------------------------------------------------------------ #
    # Since numbalsoda natively returns exactly dt-spaced points,
    # we don't need any CubicSpline interpolation for state variables!

    uniform_x = x_arr
    uniform_phi = phi_arr
    uniform_phi_dot = phi_dot_arr
    uniform_r = r_arr
    uniform_e = e_arr
    uniform_l = l_arr

    # For r_dot, we can compute the derivative of the uniform array
    cs_r = CubicSpline(t_arr, r_arr)
    uniform_r_dot = cs_r(t_arr, 1)

    # Convert time back to seconds for the caller
    uniform_t_sec = t_arr * (total_mass * LAL_MTSUN_SI)

    return {
        "time_evol": uniform_t_sec,
        "x_evol": uniform_x,
        "eccentricity_evol": uniform_e,
        "mean_ano_evol": uniform_l,
        "phi_evol": uniform_phi,
        "phi_dot_evol": uniform_phi_dot,
        "r_evol": uniform_r,
        "r_dot_evol": uniform_r_dot,
    }
