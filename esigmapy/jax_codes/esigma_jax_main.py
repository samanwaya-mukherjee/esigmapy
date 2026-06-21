"""
esigma_jax_main.py
==================
JAX port of the ESIGMA inspiral driver: diffrax ODE integration,
post-processing, GW mode computation, and high-level waveform API.

All functions through ``integrate_esigma_dynamics_jax`` are differentiable
via ``jax.grad`` / ``jax.jacfwd``.  The high-level API wrappers return plain
NumPy arrays and are not themselves JIT-traced end-to-end (use the lower-level
primitives for gradient-based inference).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

import diffrax

from .esigma_jax_kepler import (
    solve_kepler_jax,
    separation_jax,
)
from .esigma_jax_inspiral import (
    eccentric_x_model_odes_jax,
    dphi_dt_jax,
)

# ---------------------------------------------------------------------------
# Physical constants (matching LAL values)
# ---------------------------------------------------------------------------

LAL_MTSUN_SI = 4.925491025543576e-6  # solar mass in seconds
LAL_MRSUN_SI = 1.476625061404649e3  # solar mass in metres
LAL_PI = jnp.pi
LAL_PC_SI = 3.085677581491367e16  # parsec in metres

# Default PN orders (matching python_codes defaults)
_RAD_PN_ORDER_DEFAULT = 8
_MODE_PN_ORDER_DEFAULT = 8

# ---------------------------------------------------------------------------
# T_max estimator
# ---------------------------------------------------------------------------


def estimate_T_max_jax(x_init: float, eta: float, safety: float = 4.0) -> float:
    """
    Analytical upper bound on the inspiral duration in geometric units (M=1).

    Leading-order (0PN) Peters formula:
        T ~ (5/256) * (1/eta) * x_init^{-4}

    Multiplied by ``safety`` to cover higher-PN corrections.
    """
    return safety * (5.0 / 256.0) / (eta * x_init**4)


# ---------------------------------------------------------------------------
# Low-level diffrax integration
# ---------------------------------------------------------------------------


def _make_rhs(rad_pn_order: int, vpnorder: int, x_final: float):
    """
    Return a diffrax-compatible RHS function with PN orders baked in.

    ``rad_pn_order`` and ``vpnorder`` are Python ints: captured in the closure
    so that the Python-level ``if radiation_pn_order >= N:`` branches in the
    ODE are resolved at trace time (not treated as JAX dynamic values).
    """

    def rhs(t, y, args):
        eta, m1, m2, S1z, S2z = args
        x, e, l, phi = y[0], y[1], y[2], y[3]
        # Smoothly cap x at x_final (mirrors the C / numba implementation)
        past_isco = x >= x_final
        # Cap x so the ODE terms don't blow up beyond ISCO
        x_capped = jnp.where(past_isco, x_final, x)
        y_capped = jnp.array([x_capped, e, l, phi])
        dydt = eccentric_x_model_odes_jax(
            t,
            y_capped,
            (eta, m1, m2, S1z, S2z, rad_pn_order, vpnorder),
        )
        # Freeze all derivatives once ISCO is reached so the adaptive solver
        # can take large steps through the post-ISCO portion of the time grid.
        return jnp.where(past_isco, jnp.zeros_like(dydt), dydt)

    return rhs


def integrate_esigma_dynamics_jax(
    y0: jax.Array,
    eta: float,
    m1: float,
    m2: float,
    S1z: float,
    S2z: float,
    dt_M: float,
    T_max_M: float,
    x_final: float,
    rad_pn_order: int = 8,
    vpnorder: int = 8,
    ode_rtol: float = 1e-8,
    ode_atol: float = 1e-8,
    max_steps: int = 100_000_000,
) -> tuple[jax.Array, jax.Array]:
    """
    Integrate the ESIGMA ODE system using diffrax (Tsit5 solver).

    State vector ``y = [x, e, l, phi]`` where
      x   = PN parameter (M*Omega)^{2/3}
      e   = orbital eccentricity
      l   = mean anomaly
      phi = orbital phase

    Parameters
    ----------
    y0 : jnp.array, shape (4,)
        Initial conditions [x0, e0, l0, phi0].
    eta : float
        Symmetric mass ratio.
    m1, m2 : float
        Component masses (solar masses, used inside the ODE as dimensionless
        mass ratios: m1/(m1+m2) etc.).
    S1z, S2z : float
        Dimensionless z-spins.
    dt_M : float
        Output time step in geometric units (M=1).
    T_max_M : float
        Maximum integration time in geometric units.
    x_final : float
        Termination condition: integration stops (x is capped) when x ≥ x_final.
    rad_pn_order : int
        Radiation-reaction PN order (static, determines ODE terms included).
    vpnorder : int
        Waveform-mode PN order (static, passed through to phi_dot and modes).
    ode_rtol, ode_atol : float
        Relative and absolute tolerances for the adaptive step-size controller.
    max_steps : int
        Maximum number of internal solver steps.

    Returns
    -------
    t_arr : jnp.array, shape (N,)
        Uniform time grid in geometric units.
    y_arr : jnp.array, shape (N, 4)
        Solution [x, e, l, phi] at each grid point.
    """
    n_steps = int(T_max_M / dt_M) + 1
    t_arr = jnp.linspace(0.0, T_max_M, n_steps)

    rhs = _make_rhs(rad_pn_order, vpnorder, x_final)

    sol = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(rhs),
        solver=diffrax.Tsit5(),
        t0=0.0,
        t1=T_max_M,
        dt0=dt_M,
        y0=y0,
        args=(eta, m1, m2, S1z, S2z),
        saveat=diffrax.SaveAt(ts=t_arr),
        stepsize_controller=diffrax.PIDController(rtol=ode_rtol, atol=ode_atol),
        max_steps=max_steps,
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
    )

    return t_arr, sol.ys


# ---------------------------------------------------------------------------
# High-level dynamics driver (mirrors inspiral_esigma_dynamics)
# ---------------------------------------------------------------------------


def inspiral_esigma_dynamics_jax(
    mass1: float,
    mass2: float,
    S1z: float,
    S2z: float,
    e_init: float,
    f_gw_init: float,
    mean_anom_init: float = 0.0,
    ode_eps: float = 1e-8,
    sampling_rate: float = 4096.0,
    rad_pn_order: int = 8,
    vpnorder: int = 8,
    inspiral_end_radius: float = 4.0,
) -> dict:
    """
    Compute ESIGMA orbital dynamics via diffrax ODE integration.

    Parameters
    ----------
    mass1, mass2 : float
        Component masses in solar masses.
    S1z, S2z : float
        Dimensionless z-spins.
    e_init : float
        Initial eccentricity.
    f_gw_init : float
        Initial GW frequency in Hz.
    mean_anom_init : float
        Initial mean anomaly in radians.
    ode_eps : float
        ODE tolerance (relative and absolute).
    sampling_rate : float
        Output sample rate in Hz.
    rad_pn_order : int
        Radiation-reaction PN order (default 8 = 3PN + 3.5PN).
    vpnorder : int
        Waveform-mode PN order.
    inspiral_end_radius : float
        Termination radius in units of total mass (default 4M).

    Returns
    -------
    dict with keys:
        time_evol, x_evol, eccentricity_evol, mean_ano_evol,
        phi_evol, phi_dot_evol, r_evol, r_dot_evol
    All values are 1-D NumPy arrays.  time_evol is in seconds.
    The arrays are truncated at the ISCO crossing (x >= x_final).
    """
    if not (0.0 <= e_init < 1.0):
        raise ValueError("eccentricity must be in [0, 1)")

    total_mass = mass1 + mass2
    eta = (mass1 * mass2) / total_mass**2

    omega_init = LAL_PI * f_gw_init * LAL_MTSUN_SI
    x_init = float((total_mass * omega_init) ** (2.0 / 3.0))

    # ISCO: x_final corresponds to r = inspiral_end_radius * M
    # For r=4M (Schwarzschild-like): omega_isco = 1/(r^{3/2}) -> x_final = r^{-1}
    # From Kepler: x = (M*omega)^{2/3} = (M/(r^{3/2}))^{2/3} = M^{2/3}/r
    # In geometric units (M=1): x_final = 1/inspiral_end_radius
    x_final = float(1.0 / inspiral_end_radius)

    dt_sec = 1.0 / sampling_rate
    dt_M = dt_sec / (total_mass * LAL_MTSUN_SI)

    T_max_M = estimate_T_max_jax(x_init, eta)

    y0 = jnp.array([x_init, e_init, mean_anom_init, 0.0])

    t_arr_M, y_arr = integrate_esigma_dynamics_jax(
        y0=y0,
        eta=eta,
        m1=mass1,
        m2=mass2,
        S1z=S1z,
        S2z=S2z,
        dt_M=dt_M,
        T_max_M=T_max_M,
        x_final=x_final,
        rad_pn_order=rad_pn_order,
        vpnorder=vpnorder,
        ode_rtol=ode_eps,
        ode_atol=ode_eps,
    )

    # Convert to NumPy for post-processing
    t_arr_M = np.asarray(t_arr_M)
    y_np = np.asarray(y_arr)

    x_arr = y_np[:, 0]
    e_arr = y_np[:, 1]
    l_arr = y_np[:, 2]
    phi_arr = y_np[:, 3]

    # Truncate at ISCO
    isco_idx = int(np.searchsorted(x_arr, x_final))
    if isco_idx < 4:
        isco_idx = 4
    isco_idx = min(isco_idx, len(x_arr))

    t_arr_M = t_arr_M[:isco_idx]
    x_arr = x_arr[:isco_idx]
    e_arr = e_arr[:isco_idx]
    l_arr = l_arr[:isco_idx]
    phi_arr = phi_arr[:isco_idx]

    # Post-process: u, r, phi_dot via JAX scan
    dphi_fn = partial(_dphi_baked, vpnorder=vpnorder)
    u_arr, r_arr, phi_dot_arr = _compute_state_arrays_np(
        x_arr, e_arr, l_arr, eta, mass1, mass2, S1z, S2z, dphi_fn, vpnorder
    )

    # r_dot via numerical differentiation
    r_dot_arr = np.gradient(r_arr, t_arr_M)

    # Convert time to seconds
    t_sec = t_arr_M * (total_mass * LAL_MTSUN_SI)

    return {
        "time_evol": t_sec,
        "x_evol": x_arr,
        "eccentricity_evol": e_arr,
        "mean_ano_evol": l_arr,
        "phi_evol": phi_arr,
        "phi_dot_evol": phi_dot_arr,
        "r_evol": r_arr,
        "r_dot_evol": r_dot_arr,
    }


def _dphi_baked(u, eta, m1, m2, S1z, S2z, x, e, *, vpnorder):
    """Wrapper that bakes vpnorder into dphi_dt_jax for use in compute_state_arrays_jax."""
    return dphi_dt_jax(u, eta, m1, m2, S1z, S2z, x, e, vpnorder)


def _compute_state_arrays_np(
    x_arr, e_arr, l_arr, eta, m1, m2, S1z, S2z, dphi_fn, vpnorder
):
    """
    Vectorised post-ODE state array computation using jax.vmap.
    Called after ISCO truncation (variable-length arrays).
    """
    x_jnp = jnp.asarray(x_arr, dtype=jnp.float64)
    e_jnp = jnp.asarray(e_arr, dtype=jnp.float64)
    l_jnp = jnp.asarray(l_arr, dtype=jnp.float64)

    # Vectorise kepler solve over all timesteps
    u_arr = jax.vmap(lambda li, ei: solve_kepler_jax(li, ei))(l_jnp, e_jnp)

    # Vectorise separation
    r_arr = jax.vmap(
        lambda ui, xi, ei: separation_jax(ui, eta, xi, ei, m1, m2, S1z, S2z)
    )(u_arr, x_jnp, e_jnp)

    # Vectorise phi_dot
    phi_dot_arr = jax.vmap(
        lambda ui, xi, ei: dphi_dt_jax(ui, eta, m1, m2, S1z, S2z, xi, ei, vpnorder)
    )(u_arr, x_jnp, e_jnp)

    return np.asarray(u_arr), np.asarray(r_arr), np.asarray(phi_dot_arr)


# ---------------------------------------------------------------------------
# GW mode computation (mirrors compute_mode_from_dynamics)
# ---------------------------------------------------------------------------


def _make_vectorized_mode_kernel(l: int, m: int, vpnorder: int):
    """Build a JIT-compiled, vmapped mode kernel for a given (l, m, vpnorder).

    Returns a function: (r_vec, rdot_vec, phi_vec, phidot_vec, x_vec,
                          total_mass, eta, R, S1z, S2z) -> h_lm_vec
    """
    from .esigma_jax_go_terms import generate_hlm_jax, CommonVars

    @jax.jit
    def _vectorized_kernel(r_vec, rdot_vec, phi_vec, phidot_vec, x_vec,
                           total_mass, eta, R, S1z, S2z):
        b0 = 2.0 * total_mass / jnp.exp(0.5)
        logb0 = jnp.log(b0)
        delta = jnp.sqrt(1.0 - 4.0 * eta)

        def single_step(r, rDOT, Phi, PhiDOT, x):
            params = CommonVars(
                xp5=jnp.sqrt(x), logx=jnp.log(x),
                b0=b0, r0=b0, logb0=logb0, logr0=logb0, delta=delta,
            )
            hlm = 0.0 + 0.0j
            for pno in range(vpnorder, -1, -1):
                hlm = hlm + generate_hlm_jax(
                    l, m, total_mass, eta,
                    r, rDOT, Phi, PhiDOT, R, pno, S1z, S2z, x, params,
                )
            return hlm

        return jax.vmap(single_step)(r_vec, rdot_vec, phi_vec, phidot_vec, x_vec)

    return _vectorized_kernel


_MODE_KERNEL_CACHE: dict = {}


def compute_mode_from_dynamics_jax(
    l: int,
    m: int,
    x_vec: np.ndarray,
    phi_vec: np.ndarray,
    phi_dot_vec: np.ndarray,
    r_vec: np.ndarray,
    r_dot_vec: np.ndarray,
    mass1: float,
    mass2: float,
    S1z: float,
    S2z: float,
    R: float,
    vpnorder: int,
) -> np.ndarray:
    """
    Compute the (l, m) GW mode h_lm from orbital dynamics arrays.

    Uses jax.vmap to vectorize over all timesteps in a single JIT kernel.

    Parameters
    ----------
    l, m : int
        Spherical harmonic indices.
    x_vec : array (N,)
        PN parameter (dimensionless).
    phi_vec : array (N,)
        Orbital phase (rad) in geometric units.
    phi_dot_vec : array (N,)
        d(phi)/dt in geometric units (1/M).
    r_vec : array (N,)
        Orbital separation in units of total mass (dimensionless).
    r_dot_vec : array (N,)
        dr/dt in geometric units (dimensionless/M).
    mass1, mass2 : float
        Masses in solar masses.
    S1z, S2z : float
        Dimensionless z-spins.
    R : float
        Luminosity distance in metres.
    vpnorder : int
        Waveform PN order.

    Returns
    -------
    h_lm : np.ndarray, complex128, shape (N,)
    """
    total_mass = mass1 + mass2
    eta = (mass1 * mass2) / total_mass**2

    # Get or build the vectorized kernel for this (l, m, vpnorder)
    key = (l, m, vpnorder)
    if key not in _MODE_KERNEL_CACHE:
        _MODE_KERNEL_CACHE[key] = _make_vectorized_mode_kernel(l, m, vpnorder)
    kernel = _MODE_KERNEL_CACHE[key]

    # Vectorized scaling (no Python loop)
    r_scaled = jnp.asarray(r_vec) * total_mass
    rdot_arr = jnp.asarray(r_dot_vec)
    phi_arr = jnp.asarray(phi_vec)
    phidot_scaled = jnp.asarray(phi_dot_vec) / total_mass
    x_arr = jnp.asarray(x_vec)

    # ONE vectorized JIT call for all N timesteps
    h_lm = kernel(r_scaled, rdot_arr, phi_arr, phidot_scaled, x_arr,
                  total_mass, eta, R, S1z, S2z)

    return np.asarray(h_lm * LAL_MRSUN_SI)


# ---------------------------------------------------------------------------
# High-level API: modes
# ---------------------------------------------------------------------------


def get_inspiral_esigma_modes_jax(
    mass1: float,
    mass2: float,
    f_lower: float,
    delta_t: float,
    spin1z: float = 0.0,
    spin2z: float = 0.0,
    eccentricity: float = 0.0,
    mean_anomaly: float = 0.0,
    distance: float = 1.0,
    modes_to_use: list = None,
    include_conjugate_modes: bool = True,
    rad_pn_order: int = 8,
    mode_pn_order: int = 8,
    ode_eps: float = 1e-8,
    inspiral_end_radius: float = 4.0,
    return_orbital_params: bool = False,
) -> dict:
    """
    Generate ESIGMA GW modes using the JAX ODE backend.

    Parameters
    ----------
    mass1, mass2 : float
        Component masses in solar masses.
    f_lower : float
        Starting GW frequency in Hz.
    delta_t : float
        Output time step in seconds.
    spin1z, spin2z : float
        Dimensionless z-spins.
    eccentricity : float
        Initial eccentricity.
    mean_anomaly : float
        Initial mean anomaly in radians.
    distance : float
        Luminosity distance in Mpc.
    modes_to_use : list of (l, |m|) tuples
        GW modes to compute. Defaults to [(2,2), (3,3), (4,4)].
    include_conjugate_modes : bool
        If True, include (l, -|m|) modes as well.
    rad_pn_order : int
        Radiation-reaction PN order.
    mode_pn_order : int
        Waveform-mode PN order.
    ode_eps : float
        ODE tolerance.
    inspiral_end_radius : float
        Termination radius in units of M.
    return_orbital_params : bool
        If True, also return the orbital evolution dict.

    Returns
    -------
    modes : dict mapping (l, m) → np.ndarray complex128
    (optional) dyn : dict of orbital dynamics arrays
    """
    if modes_to_use is None:
        modes_to_use = [(2, 2), (3, 3), (4, 4)]

    dyn = inspiral_esigma_dynamics_jax(
        mass1=mass1,
        mass2=mass2,
        S1z=spin1z,
        S2z=spin2z,
        e_init=eccentricity,
        f_gw_init=f_lower,
        mean_anom_init=mean_anomaly,
        ode_eps=ode_eps,
        sampling_rate=1.0 / delta_t,
        rad_pn_order=rad_pn_order,
        vpnorder=mode_pn_order,
        inspiral_end_radius=inspiral_end_radius,
    )

    x = dyn["x_evol"]
    phi = dyn["phi_evol"]
    phidot = dyn["phi_dot_evol"]
    r = dyn["r_evol"]
    rdot = dyn["r_dot_evol"]

    if include_conjugate_modes:
        full_modes = list(modes_to_use)
        for el, em in modes_to_use:
            if (el, -em) not in full_modes:
                full_modes.append((el, -em))
    else:
        full_modes = list(modes_to_use)

    R_SI = distance * 1.0e6 * float(LAL_PC_SI)
    modes = {}
    for el, em in full_modes:
        modes[(el, em)] = compute_mode_from_dynamics_jax(
            el,
            em,
            x,
            phi,
            phidot,
            r,
            rdot,
            mass1,
            mass2,
            spin1z,
            spin2z,
            R_SI,
            mode_pn_order,
        )

    if return_orbital_params:
        return dyn, modes
    return modes


# ---------------------------------------------------------------------------
# High-level API: polarizations
# ---------------------------------------------------------------------------


def _ylm_spin_weighted(iota: float, beta: float, s: int, l: int, m: int) -> complex:
    """
    Spin-weighted spherical harmonic Y^s_{lm}(iota, beta).
    Delegates to lal if available; otherwise raises.
    """
    try:
        import lal

        return complex(lal.SpinWeightedSphericalHarmonic(iota, beta, s, l, m))
    except ImportError:
        raise ImportError(
            "lal is required to compute spin-weighted spherical harmonics. "
            "Install it or supply polarizations manually from modes."
        )


def get_inspiral_esigma_waveform_jax(
    mass1: float,
    mass2: float,
    f_lower: float,
    delta_t: float,
    spin1z: float = 0.0,
    spin2z: float = 0.0,
    eccentricity: float = 0.0,
    mean_anomaly: float = 0.0,
    inclination: float = 0.0,
    coa_phase: float = 0.0,
    distance: float = 1.0,
    modes_to_use: list = None,
    rad_pn_order: int = 8,
    mode_pn_order: int = 8,
    ode_eps: float = 1e-8,
    inspiral_end_radius: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate ESIGMA h+ and hx polarizations using the JAX backend.

    Parameters
    ----------
    mass1, mass2 : float
        Masses in solar masses.
    f_lower : float
        Starting GW frequency in Hz.
    delta_t : float
        Time step in seconds.
    spin1z, spin2z : float
        Dimensionless z-spins.
    eccentricity : float
        Initial eccentricity.
    mean_anomaly : float
        Initial mean anomaly in radians.
    inclination : float
        Source inclination (Euler angle iota) in radians.
    coa_phase : float
        Coalescence phase (Euler angle beta) in radians.
    distance : float
        Luminosity distance in Mpc.
    modes_to_use : list of (l, |m|) tuples
        Modes to include.  Defaults to [(2,2), (3,3), (4,4)].
    rad_pn_order, mode_pn_order : int
        PN orders.
    ode_eps : float
        ODE tolerance.
    inspiral_end_radius : float
        ISCO termination radius in units of M.

    Returns
    -------
    h_plus, h_cross : np.ndarray, float64
    """
    if modes_to_use is None:
        modes_to_use = [(2, 2), (3, 3), (4, 4)]

    modes = get_inspiral_esigma_modes_jax(
        mass1=mass1,
        mass2=mass2,
        f_lower=f_lower,
        delta_t=delta_t,
        spin1z=spin1z,
        spin2z=spin2z,
        eccentricity=eccentricity,
        mean_anomaly=mean_anomaly,
        distance=distance,
        modes_to_use=modes_to_use,
        include_conjugate_modes=True,
        rad_pn_order=rad_pn_order,
        mode_pn_order=mode_pn_order,
        ode_eps=ode_eps,
        inspiral_end_radius=inspiral_end_radius,
    )

    # Determine output length from first available mode
    n = len(next(iter(modes.values())))
    h_plus = np.zeros(n)
    h_cross = np.zeros(n)

    for (el, em), hlm in modes.items():
        ylm = _ylm_spin_weighted(inclination, coa_phase, -2, el, em)
        h_plus += (hlm * ylm).real
        h_cross -= (hlm * ylm).imag

    return h_plus, h_cross
