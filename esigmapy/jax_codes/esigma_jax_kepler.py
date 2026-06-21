"""
esigma_jax_kepler.py
====================
JAX port of the Kepler equation solver and orbital element computation
from esigma_pn_inspiral.py.

All functions are differentiable via jax.grad / jax.jacfwd.
"""

import jax
import jax.numpy as jnp
from functools import partial

# ---------------------------------------------------------------------------
# Kepler equation solver
# ---------------------------------------------------------------------------


def _mikkola_finder_jax(e: float, l_pos: float) -> float:
    """
    Mikkola (1987) algebraic initial guess for the eccentric anomaly.
    l_pos must be non-negative (caller is responsible for sign tracking).
    """
    sgn_l = jnp.where(l_pos >= 0.0, 1.0, -1.0)
    l_pos = jnp.abs(l_pos)

    a = (1.0 - e) / (4.0 * e + 0.5)
    b = 0.5 * l_pos / (4.0 * e + 0.5)
    sgn_b = jnp.where(b >= 0.0, 1.0, -1.0)
    z = (b + sgn_b * jnp.sqrt(b * b + a * a * a)) ** (1.0 / 3.0)
    s = z - a / z
    s = s - 0.078 * s**5 / (1.0 + e)
    ecc = l_pos + e * (3.0 * s - 4.0 * s**3)
    return ecc * sgn_l


def solve_kepler_jax(l: float, e: float, n_iter: int = 12) -> float:
    """
    Solve Kepler's equation  u - e*sin(u) = l  for the eccentric anomaly u.

    Uses the Mikkola (1987) algebraic initial guess followed by n_iter
    Newton–Raphson steps. Fully differentiable.

    Parameters
    ----------
    l : float
        Mean anomaly (arbitrary range; internally range-reduced to [-π, π]).
    e : float
        Orbital eccentricity, 0 ≤ e < 1.
    n_iter : int
        Number of Newton iterations (12 achieves machine precision for e < 0.95).

    Returns
    -------
    u : float
        Eccentric anomaly.
    """
    # Range-reduce l to (-π, π)
    l_red = l - 2.0 * jnp.pi * jnp.round(l / (2.0 * jnp.pi))

    # Track sign so we can work with positive l
    sgn_l = jnp.sign(l_red)
    l_pos = jnp.abs(l_red)

    # Mikkola initial guess
    u_mik = _mikkola_finder_jax(e, l_pos)

    # High-eccentricity correction (e > 0.8, l < π/3)
    trial1 = l_pos / jnp.where(jnp.abs(1.0 - e) > 1e-12, jnp.abs(1.0 - e), 1e-12)
    trial2 = jnp.where(
        trial1 * trial1 > 6.0 * jnp.abs(1.0 - e), (6.0 * l_pos) ** (1.0 / 3.0), trial1
    )
    use_high_ecc = (e > 0.8) & (l_pos < jnp.pi / 3.0)
    u_init = jnp.where(use_high_ecc, trial2, u_mik)

    # Handle l == 0 → u = 0
    u = jnp.where(l_pos < 1e-15, 0.0, u_init)

    # Newton–Raphson iterations (unrolled; all differentiable)
    for _ in range(n_iter):
        f = u - e * jnp.sin(u) - l_pos
        fp = 1.0 - e * jnp.cos(u)
        u = u - f / fp

    # Restore sign (handle l == 0 edge case)
    return jnp.where(l_pos < 1e-15, 0.0, u * sgn_l)


# ---------------------------------------------------------------------------
# PN-corrected relative separation  r(u; x, e, spins)
# ---------------------------------------------------------------------------
# Individual PN terms (translated from esigma_pn_inspiral.py: rel_sep_*pn)


def _rel_sep_0pn(e, u):
    return 1.0 - e * jnp.cos(u)


def _rel_sep_1pn(e, u, eta):
    ef = 1.0 - e * e
    b1 = 2.0 * (1.0 - e * jnp.cos(u)) / ef
    b2 = (-18.0 + 2.0 * eta - e * (6.0 - 7.0 * eta) * jnp.cos(u)) / 6.0
    return b1 + b2


def _rel_sep_1_5pn(e, u, m1, m2, S1z, S2z):
    """1.5PN spin-orbit (Klein et al. arXiv:1801.08542, Eq. B1a)"""
    e2 = e * e
    ef = 1.0 - e2
    M = m1 + m2
    return (
        -1.0
        / 3.0
        * (
            2 * m1 * m1 * (S1z + 3 * e2 * S1z)
            + 2 * (1 + 3 * e2) * m2 * m2 * S2z
            + 3 * (1 + e2) * m1 * m2 * (S1z + S2z)
            - 2
            * e
            * (4 * m1 * m1 * S1z + 4 * m2 * m2 * S2z + 3 * m1 * m2 * (S1z + S2z))
            * jnp.cos(u)
        )
        / (ef**1.5 * M**2)
    )


def _rel_sep_2pn(e, u, eta):
    eta2 = eta * eta
    e2 = e * e
    ef = 1.0 - e2
    n1 = (-48.0 + 28.0 * eta + e2 * (-51.0 + 26.0 * eta)) * (-1.0 + e * jnp.cos(u))
    d1 = 6.0 * ef**2
    n2 = (
        72.0 * (-4.0 + 7.0 * eta)
        + 36.0 * jnp.sqrt(ef) * (-5.0 + 2.0 * eta) * (2.0 + e * jnp.cos(u))
        + ef
        * (
            72.0
            + 30.0 * eta
            + 8.0 * eta2
            + e * (-72.0 + 7 * (33.0 - 5.0 * eta) * eta) * jnp.cos(u)
        )
    )
    d2 = 72.0 * ef
    return n1 / d1 + n2 / d2


def _rel_sep_2pn_SS(e, u, m1, m2, S1z, S2z):
    """2PN spin-spin relative separation."""
    kappa1 = 1.0
    kappa2 = 1.0
    return (
        (m1 * S1z * (2 * m2 * S2z + m1 * S1z * kappa1) + m2 * m2 * S2z * S2z * kappa2)
        * (1 + e * e - 2 * e * jnp.cos(u))
    ) / (2.0 * (-1 + e**2) ** 2 * (m1 + m2) ** 2)


def _rel_sep_2_5pn_SO(e, u, m1, m2, S1z, S2z):
    """2.5PN SO relative separation."""
    e2 = e * e
    e4 = e2 * e2
    ef = 1.0 - e2
    ef_sqrt = jnp.sqrt(ef)
    M2 = (m1 + m2) ** 2
    M4 = M2 * M2
    return (
        (
            2
            * (-1 + e2) ** 2
            * (
                -12 * m1**4 * S1z
                - 12 * m2**4 * S2z
                - 21 * m1 * m1 * m2 * m2 * (S1z + S2z)
                - 2 * m1**3 * m2 * (13 * S1z + 3 * S2z)
                - 2 * m1 * m2**3 * (3 * S1z + 13 * S2z)
            )
            * (2 + e * jnp.cos(u))
            + 2
            * ef_sqrt
            * (
                12 * (2 + 7 * e2 + e4) * m1**4 * S1z
                + 12 * (2 + 7 * e2 + e4) * m2**4 * S2z
                + 2 * (22 + 85 * e2 + 10 * e4) * m1 * m1 * m2 * m2 * (S1z + S2z)
                + m1**3
                * m2
                * (
                    2 * (28 + 96 * e2 + 11 * e4) * S1z
                    + 3 * (4 + 19 * e2 + 2 * e4) * S2z
                )
                + m1
                * m2**3
                * (
                    3 * (4 + 19 * e2 + 2 * e4) * S1z
                    + 2 * (28 + 96 * e2 + 11 * e4) * S2z
                )
                - e
                * (
                    60 * (1 + e2) * m1**4 * S1z
                    + 60 * (1 + e2) * m2**4 * S2z
                    + 3 * (35 + 43 * e2) * m1 * m1 * m2 * m2 * (S1z + S2z)
                    + m1**3
                    * m2
                    * (133 * S1z + 137 * e2 * S1z + 30 * S2z + 45 * e2 * S2z)
                    + m1
                    * m2**3
                    * (30 * S1z + 45 * e2 * S1z + 133 * S2z + 137 * e2 * S2z)
                )
                * jnp.cos(u)
            )
        )
    ) / (6.0 * (-1 + e2) ** 3 * M4)


def _rel_sep_3pn(e, u, eta):
    pi2 = jnp.pi**2
    ef = 1.0 - e * e
    eta2 = eta * eta
    e2 = e * e
    e4 = e2 * e2

    term1 = (
        (-665280.0 * eta2 + 1753920.0 * eta - 1814400.0) * e4
        + ((725760.0 * eta2 - 77490.0 * pi2 + 5523840.0) * eta - 3628800.0) * e2
        + ((544320.0 * eta2 + 154980.0 * pi2 - 14132160.0) * eta + 7257600.0)
    ) * e2

    term2 = -604800.0 * eta2 + 6854400.0 * eta

    term3_cos = (
        (
            (302400.0 * eta2 - 1254960.0 * eta + 453600.0) * e4
            + ((-1542240.0 * eta2 - 38745.0 * pi2 + 6980400.0) * eta - 453600.0) * e2
            + ((2177280.0 * eta2 + 77490.0 * pi2 - 12373200.0) * eta + 4989600.0)
        )
        * e
        * e2
        + ((-937440.0 * eta2 - 37845.0 * pi2 + 6647760.0) * eta - 4989600.0) * e
    ) * jnp.cos(u)

    term4_sqrt = jnp.sqrt(ef) * (
        (
            ((-4480.0 * eta2 - 25200.0 * eta + 22680.0) * eta - 120960.0) * e4
            + (
                13440.0 * eta2 * eta
                + 4404960.0 * eta * eta
                + 116235.0 * pi2
                - 12718296.0 * eta
                + 5261760.0
            )
            * e2
        )
        * (2.0 + e * jnp.cos(u))
        + (
            ((-17920.0 * eta2 - 100800.0 * eta + 90720.0) * eta - 483840.0) * e4
            + (
                53760.0 * eta2 * eta
                + 17619840.0 * eta * eta
                + 464940.0 * pi2
                - 50873184.0 * eta
                + 21047040.0
            )
            * e2
            + ((-17920.0 * eta2 - 100800.0 * eta + 90720.0) * eta - 483840.0)
        )
    )

    return (term1 + term2 + term3_cos + term4_sqrt) / 3628800.0 / ef**3.5


def _rel_sep_3pn_SS(e, u, m1, m2, S1z, S2z):
    """3PN SS relative separation."""
    kappa1 = 1.0
    kappa2 = 1.0
    e2 = e * e
    ef = 1.0 - e2
    M = m1 + m2
    ef_sq = jnp.sqrt(ef)
    return -(
        2
        * (
            (2 + kappa1) * m1**3 * S1z**2
            + m1**2 * m2 * (2 * S1z**2 + (4 + kappa1 + kappa2) * S1z * S2z)
            + m1 * m2**2 * ((4 + kappa1 + kappa2) * S1z * S2z + 2 * S2z**2)
            + (2 + kappa2) * m2**3 * S2z**2
        )
        * (-3 * ef_sq * (2 + e * jnp.cos(u)) + (2 + e2) * (1 - e * jnp.cos(u)))
    ) / (3.0 * ef**3 * M**4)


def separation_jax(u, eta, x, e, m1, m2, S1z, S2z):
    """
    PN-corrected relative separation r(u) at 3PN order including spin effects.
    Translated from esigma_pn_inspiral.py: separation().

    Returns r in units of total_mass (geometric units M=1).
    """
    sqx = jnp.sqrt(x)
    return (
        (1.0 / x) * _rel_sep_0pn(e, u)
        + _rel_sep_1pn(e, u, eta)
        + _rel_sep_1_5pn(e, u, m1, m2, S1z, S2z) * sqx
        + _rel_sep_2pn_SS(e, u, m1, m2, S1z, S2z) * x
        + _rel_sep_2pn(e, u, eta) * x
        + _rel_sep_2_5pn_SO(e, u, m1, m2, S1z, S2z) * x * sqx
        + _rel_sep_3pn(e, u, eta) * x * x
        + _rel_sep_3pn_SS(e, u, m1, m2, S1z, S2z) * x * x
    )


# ---------------------------------------------------------------------------
# Batch computation of (u, r, phi_dot) arrays for post-ODE processing
# The phi_dot computation is imported from esigma_jax_inspiral at call time
# to avoid circular imports.
# ---------------------------------------------------------------------------


def compute_state_arrays_jax(x_arr, e_arr, l_arr, eta, m1, m2, S1z, S2z, dphi_dt_fn):
    """
    For each time step, compute:
      - eccentric anomaly u via Kepler solver
      - orbital separation r (in units of M)
      - instantaneous orbital phase rate dφ/dt

    Parameters
    ----------
    x_arr, e_arr, l_arr : 1-D jnp arrays of shape (N,)
    dphi_dt_fn : callable (u, eta, m1, m2, S1z, S2z, x, e) → float
        dphi/dt dispatcher from esigma_jax_inspiral.

    Returns
    -------
    u_arr, r_arr, phi_dot_arr : 1-D jnp arrays of shape (N,)
    """

    def _step(carry, state):
        xi, ei, li = state
        ui = solve_kepler_jax(li, ei)
        ri = separation_jax(ui, eta, xi, ei, m1, m2, S1z, S2z)
        phidoti = dphi_dt_fn(ui, eta, m1, m2, S1z, S2z, xi, ei)
        return carry, (ui, ri, phidoti)

    _, (u_arr, r_arr, phi_dot_arr) = jax.lax.scan(_step, None, (x_arr, e_arr, l_arr))
    return u_arr, r_arr, phi_dot_arr
