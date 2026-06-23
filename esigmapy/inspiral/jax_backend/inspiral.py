"""
esigma_jax_inspiral.py
======================
JAX port of all eccentric enhancement functions and ODE RHS PN terms
from esigma_pn_inspiral.py.

All functions are fully differentiable via jax.grad / jax.jacfwd.
PN order is a static compile-time argument for the dispatcher functions.
"""

import jax
import jax.numpy as jnp
from functools import partial

from .kepler import solve_kepler_jax, separation_jax

# Physical / mathematical constants
EULER_GAMMA = 0.5772156649015329
LOG2 = jnp.log(2.0)
LOG3 = jnp.log(3.0)
LOG4 = jnp.log(4.0)


# ============ Eccentric enhancement functions ================================


def phi_e_jax(e):
    e2 = e * e
    ef = 1.0 - e2
    e4 = e2 * e2
    e6 = e4 * e2
    e8 = e6 * e2
    e10 = e8 * e2
    e12 = e10 * e2
    num = (
        1.0
        + (18970894028.0 * e2) / 2649026657.0
        + (157473274.0 * e4) / 30734301.0
        + (48176523.0 * e6) / 177473701.0
        + (9293260.0 * e8) / 3542508891.0
        - (5034498.0 * e10) / 7491716851.0
        + (428340.0 * e12) / 9958749469.0
    )
    den = ef**5
    return num / den


def psi_e_jax(e):
    e2 = e * e
    ef = 1.0 - e2
    e4 = e2 * e2
    e6 = e4 * e2
    num = 1.0 - (185.0 * e2) / 21.0 - (3733.0 * e4) / 99.0 - (1423.0 * e6) / 104.0
    den = ef**6
    return num / den


def zed_e_jax(e):
    e2 = e * e
    ef = 1.0 - e2
    e4 = e2 * e2
    e6 = e4 * e2
    num = 1.0 + (2095.0 * e2) / 143.0 + (1590.0 * e4) / 59.0 + (977.0 * e6) / 113.0
    den = ef**6
    return num / den


def kappa_e_jax(e):
    e2 = e * e
    ef = 1.0 - e2
    e4 = e2 * e2
    e6 = e4 * e2
    e8 = e6 * e2
    e10 = e8 * e2
    num = (
        1.0
        + (1497.0 * e2) / 79.0
        + (7021.0 * e4) / 143.0
        + (997.0 * e6) / 98.0
        + (463.0 * e8) / 51.0
        - (3829.0 * e10) / 120.0
    )
    den = ef**7
    return num / den


def phi_e_tilde_jax(e):
    e2 = e * e
    ef = 1.0 - e2
    e4 = e2 * e2
    e6 = e4 * e2
    e8 = e6 * e2
    e10 = e8 * e2
    num = (
        1.0
        + (413137256.0 * e2) / 136292703.0
        + (37570495.0 * e4) / 98143337.0
        - (2640201.0 * e6) / 993226448.0
        - (4679700.0 * e8) / 6316712563.0
        - (328675.0 * e10) / 8674876481.0
    )
    den = ef**3 * jnp.sqrt(ef)
    return num / den


def psi_e_tilde_jax(e):
    e2 = e * e
    ef = 1.0 - e2
    e4 = e2 * e2
    e6 = e4 * e2
    e8 = e6 * e2
    e10 = e8 * e2
    num = (
        1.0
        - (2022.0 * e2) / 305.0
        - (249.0 * e4) / 26.0
        - (193.0 * e6) / 239.0
        + (23.0 * e8) / 43.0
        - (102.0 * e10) / 463.0
    )
    den = ef**4 * jnp.sqrt(ef)
    return num / den


def zed_e_tilde_jax(e):
    e2 = e * e
    ef = 1.0 - e2
    e4 = e2 * e2
    e6 = e4 * e2
    e8 = e6 * e2
    num = (
        1.0
        + (1563.0 * e2) / 194.0
        + (1142.0 * e4) / 193.0
        + (123.0 * e6) / 281.0
        - (27.0 * e8) / 328.0
    )
    den = ef**4 * jnp.sqrt(ef)
    return num / den


def kappa_e_tilde_jax(e):
    e2 = e * e
    ef = 1.0 - e2
    e4 = e2 * e2
    e6 = e4 * e2
    e8 = e6 * e2
    e10 = e8 * e2
    num = (
        1.0
        + (1789.0 * e2) / 167.0
        + (5391.0 * e4) / 340.0
        + (2150.0 * e6) / 219.0
        - (1007.0 * e8) / 320.0
        + (2588.0 * e10) / 189.0
    )
    den = ef**5
    return num / den


def phi_e_rad_jax(e):
    e2 = e * e
    safe_e2 = jnp.where(e2 > 1e-30, e2, 1.0)
    ef = 1.0 - e2
    sqef = jnp.sqrt(ef)
    pre = 192.0 * sqef / (985.0 * safe_e2)
    result = pre * (sqef * phi_e_jax(e) - phi_e_tilde_jax(e))
    return jnp.where(e2 > 1e-30, result, 0.0)


def psi_e_rad_jax(e):
    e2 = e * e
    safe_e2 = jnp.where(e2 > 1e-30, e2, 1.0)
    ef = 1.0 - e2
    sqef = jnp.sqrt(ef)
    pf1 = 18816.0 / (55691.0 * safe_e2 * sqef)
    pf2 = 16382.0 * sqef / (55691.0 * safe_e2)
    b1 = sqef * (1.0 - (11.0 / 7.0) * e2) * phi_e_jax(e) - (
        1.0 - (3.0 / 7.0) * e2
    ) * phi_e_tilde_jax(e)
    b2 = sqef * psi_e_jax(e) - psi_e_tilde_jax(e)
    result = pf1 * b1 + pf2 * b2
    return jnp.where(e2 > 1e-30, result, 0.0)


def zed_e_rad_jax(e):
    e2 = e * e
    safe_e2 = jnp.where(e2 > 1e-30, e2, 1.0)
    ef = 1.0 - e2
    sqef = jnp.sqrt(ef)
    pf1 = 924.0 / (19067.0 * safe_e2 * sqef)
    pf2 = 12243.0 * sqef / (76268.0 * safe_e2)
    b1 = -ef * sqef * phi_e_jax(e) + (1.0 - (5.0 / 11.0) * e2) * phi_e_tilde_jax(e)
    b2 = sqef * zed_e_jax(e) - zed_e_tilde_jax(e)
    result = pf1 * b1 + pf2 * b2
    return jnp.where(e2 > 1e-30, result, 0.0)


def kappa_e_rad_jax(e):
    e2 = e * e
    safe_e2 = jnp.where(e2 > 1e-30, e2, 1.0)
    ef = 1.0 - e2
    sqef = jnp.sqrt(ef)
    den = 769.0 / 96.0 - 3059665.0 * LOG2 / 700566.0 + 8190315.0 * LOG3 / 1868176.0
    pf = sqef / safe_e2
    b1 = sqef * kappa_e_jax(e) - kappa_e_tilde_jax(e)
    result = pf * b1 / den
    return jnp.where(e2 > 1e-30, result, 0.0)


def f_e_jax(e):
    ef = 1.0 - e * e
    e2 = e * e
    e4 = e2 * e2
    e6 = e4 * e2
    e8 = e6 * e2
    denom = jnp.sqrt(ef) * ef**6
    num = (
        1.0
        + 85.0 * e2 / 6.0
        + 5171.0 * e4 / 192.0
        + 1751.0 * e6 / 192.0
        + 297.0 * e8 / 1024.0
    )
    return num / denom


def capital_f_e_jax(e):
    ef = 1.0 - e * e
    e2 = e * e
    e4 = e2 * e2
    e6 = e4 * e2
    denom = jnp.sqrt(ef) * ef**5
    num = 1.0 + 2782.0 * e2 / 769.0 + 10721.0 * e4 / 6152.0 + 1719.0 * e6 / 24608.0
    return num / denom


def psi_n_jax(e):
    ef = 1.0 - e * e
    e2 = e * e
    pf1 = 1344.0 * (7.0 - 5.0 * e2) / (17599.0 * ef)
    pf2 = 8191.0 / 17599.0
    return pf1 * phi_e_jax(e) + pf2 * psi_e_jax(e)


def zed_n_jax(e):
    pf1 = 583.0 / 567.0
    pf2 = 16.0 / 567.0
    return pf1 * zed_e_jax(e) - pf2 * phi_e_jax(e)


# ============ x_dot (dx/dt) terms ==========================================


def x_dot_0pn_jax(e, eta):
    """Eq. (A26)"""
    e0 = 2.0 * eta * 96.0 / 15.0
    e2 = e * e
    e4 = e2 * e2
    ef = 1.0 - e2
    num = 2.0 * eta * (37.0 * e4 + 292.0 * e2 + 96.0)
    den = 15.0 * ef**3 * jnp.sqrt(ef)
    full_expr = num / den
    return jnp.where(jnp.abs(e) < 1e-12, e0, full_expr)


def x_dot_1pn_jax(e, eta):
    """Eq. (A27)"""
    e0 = -eta * (2972.0 / 105.0 + 176.0 * eta / 5.0)
    e2 = e * e
    e4 = e2 * e2
    e6 = e4 * e2
    ef = 1.0 - e2
    t0 = 11888.0 + 14784.0 * eta
    t2 = e2 * (-87720.0 + 159600.0 * eta)
    t4 = e4 * (-171038.0 + 141708.0 * eta)
    t6 = e6 * (-11717.0 + 8288.0 * eta)
    den = 420.0 * ef**4 * jnp.sqrt(ef)
    num = -eta * (t0 + t2 + t4 + t6)
    full_expr = num / den
    return jnp.where(jnp.abs(e) < 1e-12, e0, full_expr)


def x_dot_1_5_pn_jax(e, eta, m1, m2, S1z, S2z):
    """1.5PN spin-orbit term"""
    M = m1 + m2
    e2 = e * e
    e4 = e2 * e2
    e6 = e4 * e2
    ef = 1.0 - e2
    ef5 = ef * ef * ef * ef * ef
    M4 = M * M * M * M

    e0 = (64.0 * eta / 5.0) * (
        -1.0
        / 12.0
        * (113 * m1 * m1 * S1z + 113 * m2 * m2 * S2z + 75 * m1 * m2 * (S1z + S2z))
        / M**2
    )

    full_expr = -(
        m1
        * m2
        * (
            (5424 + 27608 * e2 + 16694 * e4 + 585 * e6) * m1 * m1 * S1z
            + (5424 + 27608 * e2 + 16694 * e4 + 585 * e6) * m2 * m2 * S2z
            + 3 * (1200 + 6976 * e2 + 4886 * e4 + 207 * e6) * m1 * m2 * (S1z + S2z)
        )
    ) / (45.0 * ef5 * M4)

    return jnp.where(jnp.abs(e) < 1e-12, e0, full_expr)


def x_dot_hereditary_1_5_jax(e, eta, x):
    """Eq. (A28)"""
    pre = eta * x * x * x * x * x * x * jnp.sqrt(x)
    full_expr = pre * 256.0 * jnp.pi * phi_e_jax(e) / 5.0
    e0 = pre * 256.0 * jnp.pi / 5.0
    return jnp.where(jnp.abs(e) > 1e-12, full_expr, e0)


def x_dot_2pn_jax(e, eta, x):
    """Eq. (A29)"""
    eta2 = eta * eta
    e0_lim = eta * (68206.0 / 2835.0 + 27322.0 * eta / 315.0 + 1888.0 * eta2 / 45.0)

    e2 = e * e
    e4 = e2 * e2
    e6 = e4 * e2
    e8 = e4 * e4
    ef = 1.0 - e2
    sqrt_ef = jnp.sqrt(ef)
    den = 45360.0 * ef**5 * sqrt_ef

    e0_term = -360224.0 + 4514976.0 * eta + 1903104.0 * eta2
    e2_term = e2 * (-92846560.0 + 15464736.0 * eta + 61282032.0 * eta2)
    e4_term = e4 * (783768.0 - 207204264.0 * eta + 166506060.0 * eta2)
    e6_term = e6 * (83424402.0 - 123108426.0 * eta + 64828848.0 * eta2)
    e8_term = e8 * (3523113.0 - 3259980.0 * eta + 1964256.0 * eta2)

    rt_e0_term = 1451520.0 - 580608.0 * eta
    rt_e2_term = e2 * (64532160.0 - 25812864.0 * eta)
    rt_e4_term = e4 * (66316320.0 - 26526528.0 * eta)
    rt_e6_term = e6 * (2646000.0 - 1058400.0 * eta)

    num = eta * (
        e0_term
        + e2_term
        + e4_term
        + e6_term
        + e8_term
        + sqrt_ef * (rt_e0_term + rt_e2_term + rt_e4_term + rt_e6_term)
    )
    full_expr = num / den
    return jnp.where(jnp.abs(e) < 1e-12, e0_lim, full_expr)


def x_dot_2pn_SS_jax(e, eta, m1, m2, S1z, S2z):
    """2PN spin-spin term"""
    kappa1 = 1.0
    kappa2 = 1.0
    M2 = (m1 + m2) ** 2
    pre = 64.0 * eta / 5.0

    e0 = (
        pre
        * (
            (
                (1 + 80 * kappa1) * m1 * m1 * S1z * S1z
                + 158 * m1 * m2 * S1z * S2z
                + (1 + 80 * kappa2) * m2 * m2 * S2z * S2z
            )
        )
        / (16.0 * M2)
    )

    e2 = e * e
    e4 = e2 * e2
    e6 = e4 * e2
    ef = 1.0 - e2
    ef_sqrt = jnp.sqrt(ef)
    ef_55 = ef**4 * ef * ef_sqrt
    M4 = M2**2
    full_expr = (
        m1
        * m2
        * (
            (
                48 * (1 + 80 * kappa1)
                + 3 * e6 * (9 + 236 * kappa1)
                + 8 * e2 * (57 + 2692 * kappa1)
                + 2 * e4 * (207 + 7472 * kappa1)
            )
            * m1
            * m1
            * S1z
            * S1z
            + 2 * (3792 + 21080 * e2 + 14530 * e4 + 681 * e6) * m1 * m2 * S1z * S2z
            + (
                48 * (1 + 80 * kappa2)
                + 3 * e6 * (9 + 236 * kappa2)
                + 8 * e2 * (57 + 2692 * kappa2)
                + 2 * e4 * (207 + 7472 * kappa2)
            )
            * m2
            * m2
            * S2z
            * S2z
        )
    ) / (60.0 * ef_55 * M4)

    return jnp.where(jnp.abs(e) < 1e-12, e0, full_expr)


def x_dot_hereditary_2_5_jax(e, eta, x):
    """2.5PN hereditary"""
    pre = eta * x**7 * jnp.sqrt(x)
    ef = 1.0 - e * e
    ef2 = ef * ef
    e2 = e * e
    b1 = 256.0 * jnp.pi * phi_e_jax(e) / ef
    b2 = (
        -17599.0 * jnp.pi * psi_n_jax(e) / 35.0
        - 2268.0 * eta * jnp.pi * zed_n_jax(e) / 5.0
        - 788.0 * e2 * jnp.pi * phi_e_rad_jax(e) / ef2
    )
    full_expr = pre * (b1 + 2.0 * b2 / 3.0)

    b1e0 = 256.0 * jnp.pi
    b2e0 = -17599.0 * jnp.pi / 35.0 - 2268.0 * eta * jnp.pi / 5.0
    e0 = pre * (b1e0 + 2.0 * b2e0 / 3.0)

    return jnp.where(jnp.abs(e) > 1e-12, full_expr, e0)


def x_dot_2_5pn_SO_jax(e, eta, m1, m2, S1z, S2z):
    """2.5PN spin-orbit"""
    M2 = (m1 + m2) ** 2
    M4 = M2 * M2
    pre = 64.0 * eta / 5.0

    e2 = e * e
    e4 = e2 * e2
    e6 = e4 * e2
    e8 = e6 * e2
    ef = 1.0 - e2
    ef_sqrt = jnp.sqrt(ef)
    M6 = M4 * M2

    full_expr = (
        -0.0000992063492063492
        * m1
        * m2
        * (
            (
                4008832
                + 808515 * e8
                + 896 * e4 * (126373 + 26748 * ef_sqrt)
                + 16 * e6 * (2581907 + 30576 * ef_sqrt)
                + 384 * e2 * (111403 + 61264 * ef_sqrt)
            )
            * m1**4
            * S1z
            + (
                4008832
                + 808515 * e8
                + 896 * e4 * (126373 + 26748 * ef_sqrt)
                + 16 * e6 * (2581907 + 30576 * ef_sqrt)
                + 384 * e2 * (111403 + 61264 * ef_sqrt)
            )
            * m2**4
            * S2z
            + (
                1962112
                + 1803645 * e8
                + 32 * e6 * (2542879 + 26754 * ef_sqrt)
                + 896 * e4 * (202075 + 46809 * ef_sqrt)
                + 768 * e2 * (51189 + 53606 * ef_sqrt)
            )
            * m1
            * m1
            * m2
            * m2
            * (S1z + S2z)
            + m1**3
            * m2
            * (
                (
                    3029504
                    + 1807155 * e8
                    + 64 * e6 * (1314169 + 16562 * ef_sqrt)
                    + 128 * e2 * (340325 + 398216 * ef_sqrt)
                    + 112 * e4 * (1732273 + 463632 * ef_sqrt)
                )
                * S1z
                + 3
                * (
                    414208
                    + 281775 * e8
                    + 112 * e6 * (103639 + 728 * ef_sqrt)
                    + 336 * e4 * (77531 + 11888 * ef_sqrt)
                    + 128 * e2 * (55999 + 30632 * ef_sqrt)
                )
                * S2z
            )
            + m1
            * m2**3
            * (
                3
                * (
                    414208
                    + 281775 * e8
                    + 112 * e6 * (103639 + 728 * ef_sqrt)
                    + 336 * e4 * (77531 + 11888 * ef_sqrt)
                    + 128 * e2 * (55999 + 30632 * ef_sqrt)
                )
                * S1z
                + (
                    3029504
                    + 1807155 * e8
                    + 64 * e6 * (1314169 + 16562 * ef_sqrt)
                    + 128 * e2 * (340325 + 398216 * ef_sqrt)
                    + 112 * e4 * (1732273 + 463632 * ef_sqrt)
                )
                * S2z
            )
        )
        / ((-1 + e2) ** 6 * M6)
    )

    e0 = pre * (
        -0.000992063492063492
        * (
            31319 * m1**4 * S1z
            + 31319 * m2**4 * S2z
            + 15329 * m1 * m1 * m2 * m2 * (S1z + S2z)
            + 4 * m1**3 * m2 * (5917 * S1z + 2427 * S2z)
            + 4 * m1 * m2**3 * (2427 * S1z + 5917 * S2z)
        )
        / M4
    )

    return jnp.where(jnp.abs(e) > 1e-12, full_expr, e0)


def x_dot_2_5pn_SF_jax(e, eta, S1z):
    """2.5PN self-force (horizon flux). Same value for all e."""
    pre_factor = 64.0 * eta / 5.0
    return pre_factor * (-504.0 * S1z - 1512.0 * S1z**3) / 2016.0


def x_dot_hereditary_3_jax(e, eta, x):
    """3PN hereditary"""
    pi2 = jnp.pi * jnp.pi
    pre = eta * x**8

    ef = 1.0 - e * e
    sqrt_ef = jnp.sqrt(ef)

    full_expr = (
        64.0
        * (
            -116761.0 * kappa_e_jax(e)
            + (
                19600.0 * pi2
                - 59920.0 * EULER_GAMMA
                - 59920.0 * LOG4
                - 89880.0 * jnp.log(x)
            )
            * f_e_jax(e)
        )
    ) / 18375.0

    e0 = (
        64.0
        * (
            -59920.0 * EULER_GAMMA
            - 116761.0
            + 19600.0 * pi2
            - 59920.0 * LOG4
            - 89880.0 * jnp.log(x)
        )
    ) / 18375.0

    x_3_term = jnp.where(jnp.abs(e) > 1e-12, full_expr, e0)
    return pre * x_3_term


def x_dot_3pn_jax(e, eta, x):
    """3PN non-spinning"""
    eta2 = eta * eta
    pi2 = jnp.pi * jnp.pi

    e0_lim = eta * (
        426247111.0 / 222750.0
        - 56198689.0 * eta / 17010.0
        + 541.0 * eta2 / 70.0
        - 2242.0 * eta2 * eta / 81.0
        + 1804.0 * eta * pi2 / 15.0
        + 109568.0 * jnp.log(x) / 525.0
    )

    e2 = e * e
    e4 = e2 * e2
    e6 = e4 * e2
    e8 = e4 * e4
    e10 = e8 * e2
    ef = 1.0 - e2
    sqrt_ef = jnp.sqrt(ef)
    den = 598752000.0 * ef**7

    bit_1 = (
        -25.0
        * e10
        * (
            81.0 * (99269280.0 - 33332681.0 * sqrt_ef)
            + 176.0
            * eta
            * (
                9.0 * (-5132796.0 + 1874543.0 * sqrt_ef)
                + 4.0
                * eta
                * (3582684.0 - 2962791.0 * sqrt_ef + 2320640.0 * sqrt_ef * eta)
            )
        )
    )

    bit_2 = -128.0 * (
        3950984268.0
        - 12902173599.0 * sqrt_ef
        + 275.0
        * eta
        * (
            -1066392.0
            + 57265081.0 * sqrt_ef
            + 81.0 * (-17696.0 + 16073.0 * sqrt_ef) * eta
            + 470820.0 * sqrt_ef * eta2
            - 46494.0 * (-1.0 + 45.0 * sqrt_ef) * pi2
        )
    )

    bit_3 = (
        -32.0
        * e2
        * (
            -18.0 * (2603019496.0 + 19904214811.0 * sqrt_ef)
            + 55.0
            * eta
            * (
                8147179440.0
                - 5387647438.0 * sqrt_ef
                + 270.0 * (-6909392.0 + 9657701.0 * sqrt_ef) * eta
                + 901169500.0 * sqrt_ef * eta2
                - 193725.0 * (229.0 + 237.0 * sqrt_ef) * pi2
            )
        )
    )

    bit_4 = (
        -8.0
        * e4
        * (
            -6.0 * (312191560692.0 + 8654689873.0 * sqrt_ef)
            + 55.0
            * eta
            * (
                42004763280.0
                - 88628306866.0 * sqrt_ef
                - 1350.0 * (8601376.0 + 1306589.0 * sqrt_ef) * eta
                + 23638717900.0 * sqrt_ef * eta2
                + 891135.0 * (-2.0 + 627.0 * sqrt_ef) * pi2
            )
        )
    )

    bit_5 = (
        -2.0
        * e8
        * (
            4351589277552.0
            - 1595548875627.0 * sqrt_ef
            + 550.0
            * eta
            * (
                432.0 * (6368264.0 - 10627167.0 * sqrt_ef) * eta
                + 2201124800.0 * sqrt_ef * eta2
                + 9.0
                * (
                    8.0 * (-134041982.0 + 65136045.0 * sqrt_ef)
                    + 861.0 * (14.0 + 891.0 * sqrt_ef) * pi2
                )
            )
        )
    )

    bit_6 = (
        -12.0
        * e6
        * (
            589550775792.0
            - 6005081022.0 * sqrt_ef
            + 55.0
            * eta
            * (
                90.0 * (90130656.0 - 311841025.0 * sqrt_ef) * eta
                + 17925404000.0 * sqrt_ef * eta2
                + 3.0
                * (
                    -2.0 * (5546517920.0 + 383583403.0 * sqrt_ef)
                    + 4305.0 * (9046.0 + 19113.0 * sqrt_ef) * pi2
                )
            )
        )
    )

    bit_7 = (
        -40677120.0
        * sqrt_ef
        * (3072.0 + 43520.0 * e2 + 82736.0 * e4 + 28016.0 * e6 + 891.0 * e8)
        * (LOG2 - jnp.log((1.0 / ef) + (1.0 / sqrt_ef)) - jnp.log(x))
    )

    num = eta * (bit_1 + bit_2 + bit_3 + bit_4 + bit_5 + bit_6 + bit_7)
    full_expr = num / den

    return jnp.where(jnp.abs(e) < 1e-12, e0_lim, full_expr)


def x_dot_3pn_SO_jax(e, eta, m1, m2, S1z, S2z):
    """3PN spin-orbit"""
    M2 = (m1 + m2) ** 2
    M4 = M2 * M2
    pre = 64.0 * eta / 5.0

    e2 = e * e
    e4 = e2 * e2
    e6 = e4 * e2
    e8 = e6 * e2
    ef = 1.0 - e2
    ef_sqrt = jnp.sqrt(ef)
    ef_65 = ef**6 * ef_sqrt

    full_expr = (
        -9.645061728395062e-6
        * m1
        * m2
        * jnp.pi
        * (
            (49766400 + 528887808 * e2 + 814424832 * e4 + 213166272 * e6 + 3911917 * e8)
            * m1
            * m1
            * S1z
            + (
                49766400
                + 528887808 * e2
                + 814424832 * e4
                + 213166272 * e6
                + 3911917 * e8
            )
            * m2
            * m2
            * S2z
            + 3
            * (
                11132928
                + 133936128 * e2
                + 232455168 * e4
                + 67616624 * e6
                + 1479919 * e8
            )
            * m1
            * m2
            * (S1z + S2z)
        )
        / (ef_65 * M4)
    )

    e0 = pre * (
        -1.0
        / 6.0
        * jnp.pi
        * (225 * m1 * m1 * S1z + 225 * m2 * m2 * S2z + 151 * m1 * m2 * (S1z + S2z))
        / M2
    )

    return jnp.where(jnp.abs(e) > 1e-12, full_expr, e0)


def x_dot_3pn_SS_jax(e, eta, m1, m2, S1z, S2z):
    """3PN spin-spin"""
    kappa1 = 1.0
    kappa2 = 1.0
    pre_factor = 64.0 * eta / 5.0
    M = m1 + m2
    M2 = M * M
    M4 = M2 * M2

    e0 = pre_factor * (
        (
            (36995 + 16358 * kappa1) * m1**4 * S1z * S1z
            + (36995 + 16358 * kappa2) * m2**4 * S2z * S2z
            + m1**3 * m2 * S1z * ((34377 + 5864 * kappa1) * S1z + 59554 * S2z)
            + m1 * m2**3 * S2z * (59554 * S1z + (34377 + 5864 * kappa2) * S2z)
            + 3
            * m1**2
            * m2**2
            * (
                (1841 + 1318 * kappa1) * S1z * S1z
                + 35498 * S1z * S2z
                + (1841 + 1318 * kappa2) * S2z * S2z
            )
        )
        / (672.0 * M4)
    )

    e2 = e * e
    e4 = e2 * e2
    e6 = e4 * e2
    e8 = e6 * e2
    e_fact = 1.0 - e2
    e_fact_sqrt = jnp.sqrt(e_fact)
    e_fact_65 = (e_fact**6) * e_fact_sqrt
    M6 = M4 * M2

    term1 = (
        27 * e8 * (15141 + 109852 * kappa1)
        + 8
        * e4
        * (
            22676605
            + 3044160 * e_fact_sqrt
            + 32290722 * kappa1
            + 5327280 * e_fact_sqrt * kappa1
        )
        + 84
        * e6
        * (437079 + 448 * e_fact_sqrt + 14 * (89253 + 56 * e_fact_sqrt) * kappa1)
        - 576 * (-37891 + 896 * e_fact_sqrt + 2 * (-8963 + 784 * e_fact_sqrt) * kappa1)
        + 224
        * e2
        * (633619 + 107616 * e_fact_sqrt + 42 * (10029 + 4484 * e_fact_sqrt) * kappa1)
    )

    term2 = term1 + (kappa2 - kappa1) * (
        27 * e8 * 109852
        + 8 * e4 * (32290722 + 5327280 * e_fact_sqrt)
        + 84 * e6 * (14 * (89253 + 56 * e_fact_sqrt))
        - 576 * (2 * (-8963 + 784 * e_fact_sqrt))
        + 224 * e2 * (42 * (10029 + 4484 * e_fact_sqrt))
    )

    part1 = term1 * m1**4 * S1z * S1z
    part2 = term2 * m2**4 * S2z * S2z

    mix_common = (
        521613 * e8
        + 96 * (31457 - 1680 * e_fact_sqrt)
        + 294 * e6 * (72619 + 40 * e_fact_sqrt)
        + 112 * e2 * (247661 + 67260 * e_fact_sqrt)
        + e4 * (60534556 + 7610400 * e_fact_sqrt)
    )

    part3 = (
        6
        * m1**3
        * m2
        * S1z
        * (
            (
                3 * e8 * (47103 + 202177 * kappa1)
                - 96
                * (
                    -34713
                    + 336 * e_fact_sqrt
                    - 8440 * kappa1
                    + 2576 * e_fact_sqrt * kappa1
                )
                + 112
                * e2
                * (
                    278677
                    + 13452 * e_fact_sqrt
                    + 53216 * kappa1
                    + 103132 * e_fact_sqrt * kappa1
                )
                + 14
                * e6
                * (
                    772539
                    + 168 * e_fact_sqrt
                    + 4 * (369781 + 322 * e_fact_sqrt) * kappa1
                )
                + 4
                * e4
                * (
                    7 * (1655621 + 54360 * e_fact_sqrt)
                    + 460 * (22397 + 6342 * e_fact_sqrt) * kappa1
                )
            )
            * S1z
            + 2 * mix_common * S2z
        )
    )

    part4 = (
        6
        * m1
        * m2**3
        * S2z
        * (
            2 * mix_common * S1z
            + (
                3 * e8 * (47103 + 202177 * kappa2)
                - 96
                * (
                    -34713
                    + 336 * e_fact_sqrt
                    - 8440 * kappa2
                    + 2576 * e_fact_sqrt * kappa2
                )
                + 112
                * e2
                * (
                    278677
                    + 13452 * e_fact_sqrt
                    + 53216 * kappa2
                    + 103132 * e_fact_sqrt * kappa2
                )
                + 14
                * e6
                * (
                    772539
                    + 168 * e_fact_sqrt
                    + 4 * (369781 + 322 * e_fact_sqrt) * kappa2
                )
                + 4
                * e4
                * (
                    7 * (1655621 + 54360 * e_fact_sqrt)
                    + 460 * (22397 + 6342 * e_fact_sqrt) * kappa2
                )
            )
            * S2z
        )
    )

    part5 = (
        m1**2
        * m2**2
        * (
            9
            * (
                -86016 * e_fact_sqrt * kappa1
                + 192 * (1729 + 112 * e_fact_sqrt + 1766 * kappa1)
                + e8 * (58359 + 325902 * kappa1)
                + 28
                * e6
                * (121449 - 56 * e_fact_sqrt + (388890 + 224 * e_fact_sqrt) * kappa1)
                + 224
                * e2
                * (
                    31367
                    - 4484 * e_fact_sqrt
                    + 2 * (12189 + 8968 * e_fact_sqrt) * kappa1
                )
                + 8
                * e4
                * (
                    1541617
                    - 126840 * e_fact_sqrt
                    + 6 * (482187 + 84560 * e_fact_sqrt) * kappa1
                )
            )
            * S1z
            * S1z
            + 8
            * (
                8135280
                + 1021401 * e8
                - 467712 * e_fact_sqrt
                + 147 * e6 * (305971 + 232 * e_fact_sqrt)
                + 56 * e2 * (1084885 + 390108 * e_fact_sqrt)
                + 2 * e4 * (65163991 + 11035080 * e_fact_sqrt)
            )
            * S1z
            * S2z
            + 9
            * (
                -86016 * e_fact_sqrt * kappa2
                + 192 * (1729 + 112 * e_fact_sqrt + 1766 * kappa2)
                + e8 * (58359 + 325902 * kappa2)
                + 28
                * e6
                * (121449 - 56 * e_fact_sqrt + (388890 + 224 * e_fact_sqrt) * kappa2)
                + 224
                * e2
                * (
                    31367
                    - 4484 * e_fact_sqrt
                    + 2 * (12189 + 8968 * e_fact_sqrt) * kappa2
                )
                + 8
                * e4
                * (
                    1541617
                    - 126840 * e_fact_sqrt
                    + 6 * (482187 + 84560 * e_fact_sqrt) * kappa2
                )
            )
            * S2z
            * S2z
        )
    )

    numerator = m1 * m2 * (part1 + part2 + part3 + part4 + part5)
    denominator = 30240.0 * e_fact_65 * M6
    full_expr = numerator / denominator

    return jnp.where(jnp.abs(e) < 1e-12, e0, full_expr)


def x_dot_3_5pnSO_jax(e, eta, m1, m2, S1z, S2z):
    """3.5PN spin-orbit"""
    pre = 64.0 * eta / 5.0
    M = m1 + m2
    val_e0 = pre * (
        -0.00005511463844797178
        * (
            3127800 * m1**6 * S1z
            + 3127800 * m2**6 * S2z
            + 4914306 * m1**3 * m2**3 * (S1z + S2z)
            + m1**5 * m2 * (6542338 * S1z + 1195759 * S2z)
            + m1**4 * m2**2 * (6694579 * S1z + 3284422 * S2z)
            + m1 * m2**5 * (1195759 * S1z + 6542338 * S2z)
            + m1**2 * m2**4 * (3284422 * S1z + 6694579 * S2z)
        )
        / M**6
    )
    # e-dependent expression not yet implemented; same value for all e
    return val_e0


def x_dot_3_5pn_SS_jax(e, eta, m1, m2, S1z, S2z):
    """3.5PN spin-spin"""
    kappa1 = 1.0
    kappa2 = 1.0
    pre = 64.0 * eta / 5.0
    M2 = (m1 + m2) ** 2
    return (
        pre
        * (
            jnp.pi
            * (
                (1 + 160 * kappa1) * m1 * m1 * S1z * S1z
                + 318 * m1 * m2 * S1z * S2z
                + (1 + 160 * kappa2) * m2 * m2 * S2z * S2z
            )
        )
        / (8.0 * M2)
    )


def x_dot_3_5pn_cubicSpin_jax(e, eta, m1, m2, S1z, S2z):
    """3.5PN cubic-in-spin"""
    kappa1 = 1.0
    kappa2 = 1.0
    lambda1 = 1.0
    lambda2 = 1.0
    pre = 64.0 * eta / 5.0
    M = m1 + m2
    return pre * (
        -0.020833333333333332
        * (
            2 * (15 + 1016 * kappa1 + 528 * lambda1) * m1**4 * S1z**3
            + 2 * (15 + 1016 * kappa2 + 528 * lambda2) * m2**4 * S2z**3
            + m1**3
            * m2
            * S1z
            * S1z
            * (
                (21 + 536 * kappa1 + 1056 * lambda1) * S1z
                + (4033 + 3712 * kappa1) * S2z
            )
            + 12
            * m1
            * m1
            * m2
            * m2
            * S1z
            * S2z
            * ((89 + 434 * kappa1) * S1z + (89 + 434 * kappa2) * S2z)
            + m1
            * m2**3
            * S2z
            * S2z
            * (
                (4033 + 3712 * kappa2) * S1z
                + (21 + 536 * kappa2 + 1056 * lambda2) * S2z
            )
        )
        / M**4
    )


def x_dot_3_5_pn_jax(e, eta):
    """3.5PN non-spinning"""
    return (
        64.0
        * eta
        * jnp.pi
        * (-4415.0 / 4032.0 + 358675.0 * eta / 6048.0 + 91495.0 * eta * eta / 1512.0)
        / 5.0
    )


def x_dot_3_5pn_SF_jax(e, eta, S1z):
    """3.5PN self-force (horizon flux). Same value for all e."""
    pre_factor = 64.0 * eta / 5.0
    return pre_factor * ((-16632.0 * S1z - 38556.0 * S1z**3) / 12096.0)


def x_dot_4pn_jax(e, eta, x):
    """4PN non-spinning (e=0 limit used for all e)"""
    euler = EULER_GAMMA
    pre = 64.0 * eta / 5.0
    eta2 = eta * eta
    return pre * (
        (3959271176713 - 20643291551545 * eta) / 2.54270016e10
        + eta2 * (2016887396 + 21 * eta * (-1909807 + 49518 * eta)) / 1.306368e6
        - 896 * eta * euler / 3.0
        + (124741 + 734620 * eta) * euler / 4410.0
        - 361 * jnp.pi**2 / 126.0
        - eta * (1472377 + 928158 * eta) * jnp.pi**2 / 16128.0
        + 127751 * LOG2 / 1470.0
        - 47385 * LOG3 / 1568.0
        + eta * (-850042 * LOG2 / 2205.0 + 47385 * LOG3 / 392.0)
        + (124741 - 582500 * eta) * jnp.log(x) / 8820.0
    )


def x_dot_4pnSO_jax(e, eta, m1, m2, S1z, S2z):
    """4PN spin-orbit (e=0 limit used for all e)"""
    pre = 64.0 * eta / 5.0
    M = m1 + m2
    return pre * (
        -0.000496031746031746
        * jnp.pi
        * (
            307708 * m1**4 * S1z
            + 307708 * m2**4 * S2z
            + 93121 * m1 * m1 * m2 * m2 * (S1z + S2z)
            + m1 * m2**3 * (119880 * S1z + 75131 * S2z)
            + m1**3 * m2 * (75131 * S1z + 119880 * S2z)
        )
        / M**4
    )


def x_dot_4pnSS_jax(e, eta, m1, m2, S1z, S2z):
    """4PN spin-spin (e=0 limit used for all e)"""
    kappa1 = 1.0
    kappa2 = 1.0
    pre = 64.0 * eta / 5.0
    M = m1 + m2
    return pre * (
        (
            (41400957 + 10676336 * kappa1) * m1**6 * S1z * S1z
            + (41400957 + 10676336 * kappa2) * m2**6 * S2z * S2z
            + m1**5
            * m2
            * S1z
            * (5 * (16862889 + 4890568 * kappa1) * S1z + 53648974 * S2z)
            + m1
            * m2**5
            * S2z
            * (53648974 * S1z + 5 * (16862889 + 4890568 * kappa2) * S2z)
            + m1**4
            * m2**2
            * (
                (82037757 + 30833184 * kappa1) * S1z * S1z
                + 168293278 * S1z * S2z
                + (8950581 + 4778168 * kappa2) * S2z * S2z
            )
            + m1**2
            * m2**4
            * (
                (8950581 + 4778168 * kappa1) * S1z * S1z
                + 168293278 * S1z * S2z
                + 3 * (27345919 + 10277728 * kappa2) * S2z * S2z
            )
            + m1**3
            * m2**3
            * (
                (43557309 + 18675776 * kappa1) * S1z * S1z
                + 226571670 * S1z * S2z
                + (43557309 + 18675776 * kappa2) * S2z * S2z
            )
        )
        / (72576.0 * M**6)
    )


def x_dot_4_5_pn_jax(e, eta, x):
    """4.5PN non-spinning (e=0 limit used for all e)"""
    euler = EULER_GAMMA
    pre = 64.0 * eta / 5.0
    return pre * (
        451 * eta * jnp.pi**3 / 12.0
        - jnp.pi
        * (
            700 * eta * (3098001198 + eta * (525268513 + 289286988 * eta))
            + 145786798080 * euler
            + 3 * (-343801320119 + 97191198720 * LOG2)
        )
        / 2.2353408e9
        - 3424 * jnp.pi * jnp.log(x) / 105.0
    )


# ============ e_dot (de/dt) terms ==========================================


def e_dot_0pn_jax(e, eta):
    """Eq. (A31)"""
    e2 = e * e
    ef = 1.0 - e2
    num = -e * eta * (121.0 * e2 + 304.0)
    den = 15.0 * ef**2 * jnp.sqrt(ef)
    full_expr = num / den
    return jnp.where(jnp.abs(e) < 1e-12, 0.0, full_expr)


def e_dot_1pn_jax(e, eta):
    """Eq. (A32)"""
    e2 = e * e
    e4 = e2 * e2
    ef = 1.0 - e2
    t0 = 8.0 * (28588.0 * eta + 8451.0)
    t2 = 12.0 * (54271.0 * eta - 59834.0) * e2
    t4 = (93184.0 * eta - 125361.0) * e4
    pre = e * eta / (2520.0 * ef**3 * jnp.sqrt(ef))
    full_expr = pre * (t0 + t2 + t4)
    return jnp.where(jnp.abs(e) < 1e-12, 0.0, full_expr)


def e_dot_1_5pn_SO_jax(e, m1, m2, S1z, S2z):
    """1.5PN SO eccentricity"""
    e2 = e * e
    e4 = e2 * e2
    ef = 1.0 - e2
    ef4 = ef * ef * ef * ef
    e_pre = e / ef4
    pre = e_pre * ((m1 * m2) / (90.0 * (m1 + m2) * (m1 + m2) * (m1 + m2) * (m1 + m2)))
    full_expr = pre * (
        (19688 + 28256 * e2 + 2367 * e4) * m1 * m1 * S1z
        + (19688 + 28256 * e2 + 2367 * e4) * m2 * m2 * S2z
        + 3 * (4344 + 8090 * e2 + 835 * e4) * m1 * m2 * (S1z + S2z)
    )
    return jnp.where(jnp.abs(e) < 1e-12, 0.0, full_expr)


def e_rad_hereditary_1_5_jax(e, eta, x):
    """1.5PN hereditary e_dot"""
    pre = 32.0 * eta * e * x**4 * x * jnp.sqrt(x) / 5.0
    full_expr = pre * (-985.0 * jnp.pi * phi_e_rad_jax(e) / 48.0)
    return jnp.where(jnp.abs(e) < 1e-12, 0.0, full_expr)


def e_dot_2pn_jax(e, eta):
    """2PN eccentricity"""
    eta_pow_2 = eta * eta
    e_pow_2 = e * e
    e_pow_4 = e_pow_2 * e_pow_2
    e_pow_6 = e_pow_4 * e_pow_2
    e_fact = 1.0 - e_pow_2

    zero_term = -952397.0 / 1890.0 + 5937.0 * eta / 14.0 + 752.0 * eta_pow_2 / 5.0
    e_2_term = e_pow_2 * (
        -3113989.0 / 2520.0 - 388419.0 * eta / 280.0 + 64433.0 * eta_pow_2 / 40.0
    )
    e_4_term = e_pow_4 * (
        4656611.0 / 3024.0 - 13057267.0 * eta / 5040.0 + 127411.0 * eta_pow_2 / 90.0
    )
    e_6_term = e_pow_6 * (
        420727.0 / 3360.0 - 362071.0 * eta / 2520.0 + 821.0 * eta_pow_2 / 9.0
    )
    zero_rt = 1336.0 / 3.0 - 2672.0 * eta / 15.0
    e_2_rt = e_pow_2 * (2321.0 / 2.0 - 2321.0 * eta / 5.0)
    e_4_rt = e_pow_4 * (565.0 / 6.0 - 113.0 * eta / 3.0)

    pre_factor = -e * eta / (e_fact**4 * jnp.sqrt(e_fact))
    full_expr = pre_factor * (
        zero_term
        + e_2_term
        + e_4_term
        + e_6_term
        + jnp.sqrt(e_fact) * (zero_rt + e_2_rt + e_4_rt)
    )
    return jnp.where(jnp.abs(e) < 1e-12, 0.0, full_expr)


def e_dot_2pn_SS_jax(e, m1, m2, S1z, S2z):
    """2PN SS eccentricity"""
    kappa1 = 1.0
    kappa2 = 1.0
    e2 = e * e
    e4 = e2 * e2
    ef = 1.0 - e2
    ef_45 = ef**4 * jnp.sqrt(ef)
    M2 = (m1 + m2) ** 2
    M4 = M2 * M2
    full_expr = (
        -0.008333333333333333
        * e
        * m1
        * m2
        * (
            (45 * (8 + 12 * e2 + e4) + 4 * (3752 + 5950 * e2 + 555 * e4) * kappa1)
            * m1
            * m1
            * S1z
            * S1z
            + 2 * (14648 + 23260 * e2 + 2175 * e4) * m1 * m2 * S1z * S2z
            + (45 * (8 + 12 * e2 + e4) + 4 * (3752 + 5950 * e2 + 555 * e4) * kappa2)
            * m2
            * m2
            * S2z
            * S2z
        )
        / (ef_45 * M4)
    )
    return jnp.where(jnp.abs(e) < 1e-12, 0.0, full_expr)


def e_dot_2_5pn_SO_jax(e, m1, m2, S1z, S2z):
    """2.5PN SO eccentricity"""
    e_2 = e * e
    e_4 = e_2 * e_2
    e_6 = e_4 * e_2
    e_fact = 1.0 - e_2
    e_fact_2 = e_fact * e_fact
    e_fact_sqrt = jnp.sqrt(e_fact)
    e_fact_55 = e_fact_2 * e_fact_2 * e_fact * e_fact_sqrt

    M = m1 + m2
    M_fact_2 = M * M
    M_fact_4 = M_fact_2 * M_fact_2
    M_fact_6 = M_fact_4 * M_fact_2

    common_factor = e * m1 * m2

    term_A = 3 * (
        192 * (82880 + 36083 * e_fact_sqrt)
        + 168 * e_4 * (-211648 + 487675 * e_fact_sqrt)
        + 3 * e_6 * (-560896 + 1037433 * e_fact_sqrt)
        + 16 * e_2 * (1332912 + 6885449 * e_fact_sqrt)
    )

    term_B = 3 * (
        27847680
        - 9476416 * e_fact_sqrt
        + 7 * e_6 * (-420672 + 929363 * e_fact_sqrt)
        + 24 * e_4 * (-2592688 + 6508535 * e_fact_sqrt)
        + 16 * e_2 * (2332596 + 9517267 * e_fact_sqrt)
    )

    term_C1 = (
        128 * (808080 - 259453 * e_fact_sqrt)
        + 3 * e_6 * (-3645824 + 6571731 * e_fact_sqrt)
        + 48 * e_2 * (2887976 + 10533923 * e_fact_sqrt)
        + 32 * e_4 * (-7222488 + 15232045 * e_fact_sqrt)
    )

    term_C2 = 9 * (
        206464 * e_fact_sqrt
        + 21830032 * e_2 * e_fact_sqrt
        + 22413824 * e_4 * e_fact_sqrt
        + 1071519 * e_6 * e_fact_sqrt
        - 896 * (-1 + e) * (1 + e) * (2960 + 6927 * e_2 + 313 * e_4)
    )

    term_D1 = 9 * (
        128 * (20720 + 1613 * e_fact_sqrt)
        + 256 * e_4 * (-23149 + 87554 * e_fact_sqrt)
        + 112 * e_2 * (31736 + 194911 * e_fact_sqrt)
        + e_6 * (-280448 + 1071519 * e_fact_sqrt)
    )

    term_D2 = term_C1

    numerator = common_factor * (
        term_A * (m1**4) * S1z
        + term_A * (m2**4) * S2z
        + term_B * (m1**2) * (m2**2) * (S1z + S2z)
        + (m1**3) * m2 * (term_C1 * S1z + term_C2 * S2z)
        + m1 * (m2**3) * (term_D1 * S1z + term_D2 * S2z)
    )

    full_expr = numerator / (60480.0 * e_fact_55 * M_fact_6)
    return jnp.where(jnp.abs(e) < 1e-12, 0.0, full_expr)


def e_rad_hereditary_2_5_jax(e, eta, x):
    """2.5PN hereditary e_dot"""
    pre = 32.0 * eta * e * x**4 * x**2 * jnp.sqrt(x) / 5.0
    a2 = (
        55691.0 * psi_e_rad_jax(e) / 1344.0 + 19067.0 * eta * zed_e_rad_jax(e) / 126.0
    ) * jnp.pi
    full_expr = pre * a2
    return jnp.where(jnp.abs(e) < 1e-12, 0.0, full_expr)


def e_rad_hereditary_3_jax(e, eta, x):
    """3PN hereditary e_dot"""
    pre = 32.0 * eta * e * x**7 / 5.0
    x32 = x * jnp.sqrt(x)
    a3 = (
        89789209.0 / 352800.0 - 87419.0 * LOG2 / 630.0 + 78003.0 * LOG3 / 560.0
    ) * kappa_e_rad_jax(e)
    a4 = (
        (-769.0 / 96.0)
        * (
            16.0 * jnp.pi**2 / 3.0
            - 1712.0 * EULER_GAMMA / 105.0
            - 1712.0 * jnp.log(4.0 * x32) / 105.0
        )
        * capital_f_e_jax(e)
    )
    full_expr = pre * (a3 + a4)
    return jnp.where(jnp.abs(e) < 1e-12, 0.0, full_expr)


def e_dot_3pn_jax(e, eta, x):
    """Eq. (B5d) of Ebersold et al. — direct translation of LALSimESIGMA_PNInspiral.c"""
    eta2 = eta * eta
    e2 = e * e
    e4 = e2 * e2
    e6 = e4 * e2
    e8 = e6 * e2
    ef = 1.0 - e2
    efsqrt = jnp.sqrt(ef)
    pi2 = jnp.pi * jnp.pi

    pre = -e * eta / (ef * ef * ef * ef * ef * efsqrt)

    t_e8 = (
        25.0
        * e8
        * (
            18490461597.0
            - 8162698563.0 * efsqrt
            + 176.0
            * eta
            * (
                27.0 * (-3872433.0 + 1921387.0 * efsqrt)
                + 28.0
                * eta
                * (
                    -45.0 * (2815.0 + 41179.0 * efsqrt)
                    + 1547168.0 * (1.0 + efsqrt) * eta
                )
            )
        )
    )

    t_64 = 64.0 * (
        -3.0 * (45406954567.0 + 45214190215.0 * efsqrt)
        + 55.0
        * (1.0 + efsqrt)
        * eta
        * (1839419691.0 + 5.0 * eta * (28250883.0 + 8540140.0 * eta) - 42232050.0 * pi2)
    )

    t_e2 = (
        16.0
        * e2
        * (
            -12.0 * (126022071521.0 + 117027704117.0 * efsqrt)
            + 55.0
            * eta
            * (
                -288.0 * (31771481.0 + 4136526.0 * efsqrt)
                + 90.0 * (10100935.0 - 8154617.0 * efsqrt) * eta
                + 6083735630.0 * (1.0 + efsqrt) * eta2
                + 38745.0 * (3265.0 + 1641.0 * efsqrt) * pi2
            )
        )
    )

    t_e4 = (
        12.0
        * e4
        * (
            429883524894.0
            - 702938620770.0 * efsqrt
            + 55.0
            * eta
            * (
                -46370859158.0
                + 8774742922.0 * efsqrt
                - 90.0 * (253550327.0 + 406315863.0 * efsqrt) * eta
                + 22410269280.0 * (1.0 + efsqrt) * eta2
                + 12915.0 * (33553.0 + 19771.0 * efsqrt) * pi2
            )
        )
    )

    t_e6 = (
        4.0
        * e6
        * (
            2616262495497.0
            - 1598322429999.0 * efsqrt
            + 275.0
            * eta
            * (
                -432.0 * (12599311.0 + 25205247.0 * efsqrt) * eta
                + 5282846912.0 * (1.0 + efsqrt) * eta2
                + 9.0
                * (
                    -962621272.0
                    + 1155643608.0 * efsqrt
                    + 861.0 * (1553.0 + 1431.0 * efsqrt) * pi2
                )
            )
        )
    )

    log_t = (
        -40677120.0
        * (24608.0 + 89024.0 * e2 + 42884.0 * e4 + 1719.0 * e6)
        * (1.0 + efsqrt)
        * jnp.log(((1.0 + efsqrt) * x) / (2.0 - 2.0 * e2))
    )

    full_expr = pre * (
        -8.350702795147239e-10
        * (t_e8 + t_64 + t_e2 + t_e4 + t_e6 + log_t)
        / (1.0 + efsqrt)
    )
    return jnp.where(jnp.abs(e) < 1e-12, 0.0, full_expr)


def e_dot_3pn_SO_jax(e, m1, m2, S1z, S2z):
    """3PN SO eccentricity"""
    e2 = e * e
    e4 = e2 * e2
    e6 = e4 * e2
    ef = 1.0 - e2
    ef_sqrt = jnp.sqrt(ef)
    ef_55 = ef**4 * ef * ef_sqrt
    M2 = (m1 + m2) ** 2
    M4 = M2 * M2
    full_expr = (
        e
        * m1
        * m2
        * jnp.pi
        * (
            (64622592 + 238783104 * e2 + 96887280 * e4 + 2313613 * e6) * m1 * m1 * S1z
            + (64622592 + 238783104 * e2 + 96887280 * e4 + 2313613 * e6) * m2 * m2 * S2z
            + 24
            * (1744704 + 8150400 * e2 + 3941409 * e4 + 122714 * e6)
            * m1
            * m2
            * (S1z + S2z)
        )
    ) / (51840.0 * ef_55 * M4)
    return jnp.where(jnp.abs(e) < 1e-12, 0.0, full_expr)


def e_dot_3pn_SS_jax(e, m1, m2, S1z, S2z):
    """3PN SS eccentricity"""
    kappa1 = 1.0
    kappa2 = 1.0

    e_2 = e * e
    e_4 = e_2 * e_2
    e_6 = e_4 * e_2

    e_fact = 1.0 - e_2
    e_fact_2 = e_fact * e_fact
    e_fact_sqrt = jnp.sqrt(e_fact)
    e_fact_55 = e_fact_2 * e_fact_2 * e_fact * e_fact_sqrt

    M = m1 + m2
    M_fact_2 = M * M
    M_fact_4 = M_fact_2 * M_fact_2
    M_fact_6 = M_fact_4 * M_fact_2

    prefactor_const = -0.000016534391534391536

    term = (
        e
        * m1
        * m2
        * (
            (
                27 * e_6 * (53837 + 368940 * kappa1)
                + 12
                * e_4
                * (
                    6350925
                    + 27328 * e_fact_sqrt
                    + 16634634 * kappa1
                    + 47824 * e_fact_sqrt * kappa1
                )
                + 32
                * (
                    2226847
                    + 545664 * e_fact_sqrt
                    + 538758 * kappa1
                    + 954912 * e_fact_sqrt * kappa1
                )
                + 32
                * e_2
                * (
                    7292761
                    + 1157688 * e_fact_sqrt
                    + 9 * (847619 + 225106 * e_fact_sqrt) * kappa1
                )
            )
            * m1**4
            * S1z
            * S1z
            + (
                27 * e_6 * (53837 + 368940 * kappa2)
                + 12
                * e_4
                * (
                    6350925
                    + 27328 * e_fact_sqrt
                    + 16634634 * kappa2
                    + 47824 * e_fact_sqrt * kappa2
                )
                + 32
                * (
                    2226847
                    + 545664 * e_fact_sqrt
                    + 538758 * kappa2
                    + 954912 * e_fact_sqrt * kappa2
                )
                + 32
                * e_2
                * (
                    7292761
                    + 1157688 * e_fact_sqrt
                    + 9 * (847619 + 225106 * e_fact_sqrt) * kappa2
                )
            )
            * m2**4
            * S2z
            * S2z
            + 6
            * m1**3
            * m2
            * S1z
            * (
                (
                    9 * e_6 * (55293 + 221219 * kappa1)
                    + 2
                    * e_4
                    * (
                        11036963
                        + 10248 * e_fact_sqrt
                        + 19572200 * kappa1
                        + 78568 * e_fact_sqrt * kappa1
                    )
                    + 16
                    * (
                        798819
                        + 68208 * e_fact_sqrt
                        - 336460 * kappa1
                        + 522928 * e_fact_sqrt * kappa1
                    )
                    + 8
                    * e_2
                    * (
                        7063231
                        + 289422 * e_fact_sqrt
                        + 6 * (698816 + 369817 * e_fact_sqrt) * kappa1
                    )
                )
                * S1z
                + 2
                * (
                    1818549 * e_6
                    + 240 * (31249 + 22736 * e_fact_sqrt)
                    + 2 * e_4 * (20863999 + 51240 * e_fact_sqrt)
                    + 8 * e_2 * (7764719 + 1447110 * e_fact_sqrt)
                )
                * S2z
            )
            + 6
            * m1
            * m2**3
            * S2z
            * (
                2
                * (
                    1818549 * e_6
                    + 240 * (31249 + 22736 * e_fact_sqrt)
                    + 2 * e_4 * (20863999 + 51240 * e_fact_sqrt)
                    + 8 * e_2 * (7764719 + 1447110 * e_fact_sqrt)
                )
                * S1z
                + (
                    9 * e_6 * (55293 + 221219 * kappa2)
                    + 2
                    * e_4
                    * (
                        11036963
                        + 10248 * e_fact_sqrt
                        + 19572200 * kappa2
                        + 78568 * e_fact_sqrt * kappa2
                    )
                    + 16
                    * (
                        798819
                        + 68208 * e_fact_sqrt
                        - 336460 * kappa2
                        + 522928 * e_fact_sqrt * kappa2
                    )
                    + 8
                    * e_2
                    * (
                        7063231
                        + 289422 * e_fact_sqrt
                        + 6 * (698816 + 369817 * e_fact_sqrt) * kappa2
                    )
                )
                * S2z
            )
            + m1**2
            * m2**2
            * (
                9
                * (
                    3 * e_6 * (63301 + 361786 * kappa1)
                    + 4
                    * e_4
                    * (
                        1667883
                        - 3416 * e_fact_sqrt
                        + 5133138 * kappa1
                        + 13664 * e_fact_sqrt * kappa1
                    )
                    + 32
                    * (
                        69895
                        - 22736 * e_fact_sqrt
                        - 37046 * kappa1
                        + 90944 * e_fact_sqrt * kappa1
                    )
                    + 32
                    * e_2
                    * (
                        439124
                        - 48237 * e_fact_sqrt
                        + 616472 * kappa1
                        + 192948 * e_fact_sqrt * kappa1
                    )
                )
                * S1z
                * S1z
                + 8
                * (
                    3553929 * e_6
                    + 3 * e_4 * (29583761 + 99064 * e_fact_sqrt)
                    + 116 * e_2 * (1176325 + 289422 * e_fact_sqrt)
                    + 8 * (2057131 + 1978032 * e_fact_sqrt)
                )
                * S1z
                * S2z
                + 9
                * (
                    3 * e_6 * (63301 + 361786 * kappa2)
                    + 4
                    * e_4
                    * (
                        1667883
                        - 3416 * e_fact_sqrt
                        + 5133138 * kappa2
                        + 13664 * e_fact_sqrt * kappa2
                    )
                    + 32
                    * (
                        69895
                        - 22736 * e_fact_sqrt
                        - 37046 * kappa2
                        + 90944 * e_fact_sqrt * kappa2
                    )
                    + 32
                    * e_2
                    * (
                        439124
                        - 48237 * e_fact_sqrt
                        + 616472 * kappa2
                        + 192948 * e_fact_sqrt * kappa2
                    )
                )
                * S2z
                * S2z
            )
        )
    )

    full_expr = prefactor_const * term / (e_fact_55 * M_fact_6)
    return jnp.where(jnp.abs(e) < 1e-12, 0.0, full_expr)


def e_dot_3_5pn_jax(e, eta):
    """3.5PN eccentricity — zero."""
    return 0.0


# ============ l_dot (dl/dt) terms ==========================================


def l_dot_1pn_jax(e, eta):
    """Eq. (A2)"""
    return 3.0 / (e * e - 1.0)


def l_dot_1_5pn_SO_jax(e, m1, m2, S1z, S2z):
    """1.5PN SO correction"""
    e2 = e * e
    ef = 1.0 - e2
    pf = 1.0 / (ef * jnp.sqrt(ef) * (m1 + m2) ** 2)
    return pf * (4 * m1 * m1 * S1z + 4 * m2 * m2 * S2z + 3 * m1 * m2 * (S1z + S2z))


def l_dot_2pn_SS_jax(e, m1, m2, S1z, S2z):
    """2PN SS correction"""
    kappa1 = 1.0
    kappa2 = 1.0
    return (
        -3.0
        * (m1 * S1z * (2 * m2 * S2z + m1 * S1z * kappa1) + m2 * m2 * S2z * S2z * kappa2)
    ) / (2.0 * (-1 + e**2) ** 2 * (m1 + m2) ** 2)


def l_dot_2pn_jax(e, eta):
    """Eq. (A3)"""
    ef = 1.0 - e * e
    ef2 = ef * ef
    return ((26.0 * eta - 51.0) * e * e + 28.0 * eta - 18.0) / (4.0 * ef2)


def l_dot_2_5pn_SO_jax(e, m1, m2, S1z, S2z):
    """2.5PN SO correction"""
    e2 = e * e
    ef = 1.0 - e2
    ef2 = ef * ef
    ef_sqrt = jnp.sqrt(ef)
    ef_25 = ef2 * ef_sqrt
    M2 = (m1 + m2) ** 2
    M4 = M2 * M2
    return (
        (
            20 * m1**4 * (S1z + 3 * e2 * S1z)
            + 20 * (1 + 3 * e2) * m2**4 * S2z
            + (5 + 129 * e2) * m1 * m1 * m2 * m2 * (S1z + S2z)
            + m1**3 * m2 * ((23 + 137 * e2) * S1z + 45 * e2 * S2z)
            + m1 * m2**3 * (23 * S2z + e2 * (45 * S1z + 137 * S2z))
        )
    ) / (2.0 * ef_25 * M4)


def l_dot_3pn_jax(e, eta):
    """Eq. (A4)"""
    ef = 1.0 - e * e
    ef3 = ef * ef * ef
    pre = -1.0 / (128.0 * jnp.sqrt(ef) * ef3)
    eta2 = eta * eta
    pi2 = jnp.pi * jnp.pi
    e2 = e * e
    e4 = e2 * e2
    t0 = 1920.0 - 768.0 * eta
    t2 = (1920.0 - 768.0 * eta) * e2
    t4 = (1536.0 * eta - 3840.0) * e4
    r0 = 896.0 * eta2 - 14624.0 * eta + 492.0 * pi2 * eta - 192.0
    r2 = (5120.0 * eta2 + 123.0 * pi2 * eta - 17856.0 * eta + 8544.0) * e2
    r4 = (1040.0 * eta2 - 1760.0 * eta + 2496.0) * e4
    return pre * (t0 + t2 + t4 + jnp.sqrt(ef) * (r0 + r2 + r4))


def l_dot_3pn_SS_jax(e, m1, m2, S1z, S2z):
    """3PN SS correction"""
    kappa1 = 1.0
    kappa2 = 1.0
    e2 = e * e
    M2 = (m1 + m2) ** 2
    M4 = M2 * M2
    return (
        (
            m1
            * m1
            * (
                (-8 + 42 * kappa1 + 6 * e2 * (4 + 13 * kappa1)) * m1 * m1
                + (-60 + 50 * kappa1 + 11 * e2 * (3 + 11 * kappa1)) * m1 * m2
                + 3 * (-14 + 6 * kappa1 + 3 * e2 * (1 + 8 * kappa1)) * m2 * m2
            )
            * S1z
            * S1z
            + 2
            * m1
            * m2
            * (
                (6 + 93 * e2) * m1 * m1
                + (12 + 157 * e2) * m1 * m2
                + 3 * (2 + 31 * e2) * m2 * m2
            )
            * S1z
            * S2z
            + m2
            * m2
            * (
                3 * (-14 + 6 * kappa2 + 3 * e2 * (1 + 8 * kappa2)) * m1 * m1
                + (-60 + 50 * kappa2 + 11 * e2 * (3 + 11 * kappa2)) * m1 * m2
                + 2 * (-4 + 21 * kappa2 + 3 * e2 * (4 + 13 * kappa2)) * m2 * m2
            )
            * S2z
            * S2z
        )
    ) / (4.0 * (-1 + e2) ** 3 * M4)


def l_dot_4pn_jax(e, eta):
    """4PN mean anomaly rate (arXiv:2508.08618).
    Translated from LALSimESIGMA_PNInspiral.c l_dot_4pn."""
    e2 = e * e
    e4 = e2 * e2
    e6 = e4 * e2
    ef = 1.0 - e2
    ef_sqrt = jnp.sqrt(ef)
    ef_pow_4_5 = ef * ef * ef * ef * ef_sqrt
    eta2 = eta * eta
    eta3 = eta2 * eta
    pi2 = jnp.pi * jnp.pi
    term_e6 = (
        8960.0
        * e6
        * (
            -1080.0 * (-99.0 + 7.0 * ef_sqrt)
            + 27.0 * (-3712.0 + 71.0 * ef_sqrt) * eta
            - 945.0 * (-32.0 + 5.0 * ef_sqrt) * eta2
            + 6422.0 * ef_sqrt * eta3
        )
    )
    term_e2 = (
        12.0
        * e2
        * (
            -241920.0 * (1130.0 + 261.0 * ef_sqrt)
            + eta
            * (
                384.0 * (874160.0 + 3031751.0 * ef_sqrt)
                + 59745280.0 * ef_sqrt * eta2
                - 315.0 * (2624.0 + 137381.0 * ef_sqrt) * pi2
                + 120960.0 * eta * (-608.0 - 7222.0 * ef_sqrt + 205.0 * ef_sqrt * pi2)
            )
        )
    )
    term_e4 = (
        105.0
        * e4
        * (
            -86400.0 * (-320.0 + 191.0 * ef_sqrt)
            + eta
            * (
                1536.0 * (-29072.0 + 30809.0 * ef_sqrt)
                + 8018944.0 * ef_sqrt * eta2
                + 9.0 * (20992.0 - 72961.0 * ef_sqrt) * pi2
                + 288.0 * eta * (29952.0 - 119984.0 * ef_sqrt + 1107.0 * ef_sqrt * pi2)
            )
        )
    )
    term_e0 = 280.0 * (
        51840.0 * (-40.0 + 13.0 * ef_sqrt)
        + eta
        * (
            5566464.0
            - 9343104.0 * ef_sqrt
            + 100352.0 * ef_sqrt * eta2
            + 9.0 * (-3936.0 + 26777.0 * ef_sqrt) * pi2
            + 288.0 * eta * (-3648.0 - 34504.0 * ef_sqrt + 1353.0 * ef_sqrt * pi2)
        )
    )
    return (term_e6 + term_e2 + term_e4 + term_e0) / (7.74144e6 * ef_pow_4_5)


# ============ phi_dot (dphi/dt) terms ======================================


def _cosu_factor_jax(e, u):
    return e * jnp.cos(u) - 1.0


def phi_dot_0pn_jax(e, eta, u):
    """Eq. (A11)"""
    cf = _cosu_factor_jax(e, u)
    return jnp.sqrt(1.0 - e * e) / (cf * cf)


def phi_dot_1pn_jax(e, eta, u):
    """Eq. (A12)"""
    cf = _cosu_factor_jax(e, u)
    return -(e * (eta - 4.0) * (e - jnp.cos(u))) / (jnp.sqrt(1.0 - e * e) * cf**3)


def phi_dot_1_5_pnSO_ecc_jax(e, m1, m2, S1z, S2z, u):
    """1.5PN SO phi_dot"""
    cf = -1.0 + e * jnp.cos(u)
    full_expr = (2 * e * (m1 * S1z + m2 * S2z) * (e - jnp.cos(u))) / (
        (-1 + e * e) * (m1 + m2) * cf**3
    )
    return jnp.where(jnp.abs(e) < 1e-12, 0.0, full_expr)


def phi_dot_2pn_jax(e, eta, u):
    """Eq. (A13)"""
    cf = _cosu_factor_jax(e, u)
    cf5 = cf**5
    e2 = e * e
    e3 = e2 * e
    e4 = e2 * e2
    e5 = e4 * e
    e6 = e4 * e2
    ef = 1.0 - e2
    eta2 = eta * eta
    cu = jnp.cos(u)
    cu2 = cu * cu
    cu3 = cu2 * cu

    t0 = 90.0 - 36.0 * eta
    t2 = (-2.0 * eta2 + 50.0 * eta + 75.0) * e2
    t4 = (20.0 * eta2 - 26.0 * eta - 60.0) * e4
    t6 = (-12.0 * eta - 18.0) * eta * e6
    c1 = (
        (-eta2 + 97.0 * eta + 12.0) * e5
        + (-16.0 * eta2 - 74.0 * eta - 81.0) * e3
        + (-eta2 + 67.0 * eta - 246.0) * e
    ) * cu
    c2 = (
        (17.0 * eta2 - 17.0 * eta + 48.0) * e6
        + (-4.0 * eta2 - 38.0 * eta + 153.0) * e4
        + (5.0 * eta2 - 35.0 * eta + 114.0) * e2
    ) * cu2
    c3 = (
        (-14.0 * eta2 + 8.0 * eta - 147.0) * e5 + (8.0 * eta2 + 22.0 * eta + 42.0) * e3
    ) * cu3

    r0 = (180.0 - 72.0 * eta) * e2 + 36.0 * eta - 90.0
    rc1 = ((144.0 * eta - 360.0) * e3 + (90.0 - 36.0 * eta) * e) * cu
    rc2 = ((180.0 - 72.0 * eta) * e4 + (90.0 - 36.0 * eta) * e2) * cu2
    rc3 = e3 * (36.0 * eta - 90.0) * cu3

    pre = 1.0 / (12.0 * jnp.sqrt(ef) * ef * cf5)
    return pre * (
        t0 + t2 + t4 + t6 + c1 + c2 + c3 + jnp.sqrt(ef) * (r0 + rc1 + rc2 + rc3)
    )


def phi_dot_2_pnSS_ecc_jax(e, m1, m2, S1z, S2z, u):
    """2PN SS phi_dot"""
    kappa1 = 1.0
    kappa2 = 1.0
    full_expr = (
        e
        * (kappa2 * m2 * m2 * S2z * S2z + m1 * S1z * (kappa1 * m1 * S1z + 2 * m2 * S2z))
        * (e - jnp.cos(u))
    ) / ((1 - e**2) ** 1.5 * (m1 + m2) ** 2 * (-1 + e * jnp.cos(u)) ** 3)
    return jnp.where(jnp.abs(e) < 1e-12, 0.0, full_expr)


def phi_dot_2_5pn_SO_jax(e, m1, m2, S1z, S2z, u):
    """2.5PN SO phi_dot"""
    e_2 = e * e
    e_3 = e_2 * e
    e_4 = e_2 * e_2
    e_6 = e_4 * e_2

    e_fact = 1.0 - e_2
    e_fact_sqrt = jnp.sqrt(e_fact)

    M = m1 + m2
    M_fact_2 = M * M
    M_fact_4 = M_fact_2 * M_fact_2

    cos_u = jnp.cos(u)

    numerator = (
        (m1 - m2)
        * M
        * (
            24 * (m1 * m2 - 3 * M_fact_2)
            - 6 * e_6 * (m1 * m2 - 2 * M_fact_2)
            + 18 * e_4 * (3 * m1 * m2 - 2 * M_fact_2)
            - e_2 * (31 * m1 * m2 + 56 * M_fact_2)
        )
        * (S1z - S2z)
        + (
            e_2 * (50 * m1 * m1 * m2 * m2 - 57 * m1 * m2 * M_fact_2 - 56 * M_fact_4)
            + 6 * e_6 * (2 * m1 * m1 * m2 * m2 - 3 * m1 * m2 * M_fact_2 + 2 * M_fact_4)
            - 6 * e_4 * (8 * m1 * m1 * m2 * m2 - 27 * m1 * m2 * M_fact_2 + 6 * M_fact_4)
            - 12 * (m1 * m1 * m2 * m2 - 8 * m1 * m2 * M_fact_2 + 6 * M_fact_4)
        )
        * (S1z + S2z)
        - e
        * (
            -8 * (56 + 49 * e_2 + 9 * e_4) * m1**4 * S1z
            - 8 * (56 + 49 * e_2 + 9 * e_4) * m2**4 * S2z
            + 4 * (-217 - 200 * e_2 + 9 * e_4) * m1 * m1 * m2 * m2 * (S1z + S2z)
            - 2
            * m1**3
            * m2
            * ((530 + 496 * e_2 + 6 * e_4) * S1z + 9 * (14 + 13 * e_2) * S2z)
            - 2
            * m1
            * m2**3
            * (9 * (14 + 13 * e_2) * S1z + 2 * (265 + 248 * e_2 + 3 * e_4) * S2z)
        )
        * cos_u
        + e_2
        * (
            -8 * (2 + e_2) * (29 + 9 * e_2) * m1**4 * S1z
            - 8 * (2 + e_2) * (29 + 9 * e_2) * m2**4 * S2z
            - 8 * (2 + e_2) * (47 + 21 * e_2) * m1 * m1 * m2 * m2 * (S1z + S2z)
            - 2
            * m1**3
            * m2
            * ((514 + 440 * e_2 + 78 * e_4) * S1z + 9 * (2 + e_2) * (5 + 4 * e_2) * S2z)
            - 2
            * m1
            * m2**3
            * (
                9 * (2 + e_2) * (5 + 4 * e_2) * S1z
                + 2 * (257 + 220 * e_2 + 39 * e_4) * S2z
            )
        )
        * (cos_u**2)
        - e_3
        * (
            -8 * (17 + 21 * e_2) * m1**4 * S1z
            - 8 * (17 + 21 * e_2) * m2**4 * S2z
            - 8 * (11 + 57 * e_2) * m1 * m1 * m2 * m2 * (S1z + S2z)
            - 2 * m1**3 * m2 * (4 * (29 + 57 * e_2) * S1z - 6 * S2z + 87 * e_2 * S2z)
            - 2 * m1 * m2**3 * ((-6 + 87 * e_2) * S1z + 4 * (29 + 57 * e_2) * S2z)
        )
        * (cos_u**3)
        - 12
        * e_fact_sqrt
        * (
            -12 * m1**4 * S1z
            - 12 * m2**4 * S2z
            - 21 * m1 * m1 * m2 * m2 * (S1z + S2z)
            - 2 * m1**3 * m2 * (13 * S1z + 3 * S2z)
            - 2 * m1 * m2**3 * (3 * S1z + 13 * S2z)
        )
        * ((-1 + e * cos_u) ** 2)
        * (1 - 2 * e_2 + e * cos_u)
    )

    denominator = 12.0 * ((-1 + e_2) ** 2) * M_fact_4 * ((-1 + e * cos_u) ** 5)

    return numerator / denominator


def phi_dot_3pn_jax(e, eta, u):
    """3PN non-spinning phi_dot"""
    u_factor = _cosu_factor_jax(e, u)
    u_factor_pow_7 = u_factor**7
    pi_pow_2 = jnp.pi**2
    eta_pow_2 = eta**2
    eta_pow_3 = eta**3
    e_pow_2 = e**2
    e_fact = 1.0 - e_pow_2
    e_pow_3 = e**3
    e_pow_4 = e**4
    e_pow_5 = e**5
    e_pow_6 = e**6
    e_pow_7 = e**7
    e_pow_8 = e**8
    e_pow_9 = e**9
    e_pow_10 = e**10
    e_factor = 1.0 - e_pow_2
    cos_u = jnp.cos(u)
    cos_u_pow_2 = cos_u**2
    cos_u_pow_3 = cos_u**3
    cos_u_pow_4 = cos_u**4
    cos_u_pow_5 = cos_u**5

    pre_factor = 1.0 / (13440.0 * jnp.sqrt(e_factor) * e_factor**2 * u_factor_pow_7)

    e_0_term = 67200.0 * eta_pow_2 - 761600.0 * eta + 8610.0 * eta * pi_pow_2 + 201600.0
    e_2_term = (
        4480.0 * eta_pow_3
        - 412160.0 * eta_pow_2
        - 30135.0 * pi_pow_2 * eta
        + 553008.0 * eta
        + 342720.0
    ) * e_pow_2
    e_4_term = (
        -52640.0 * eta_pow_3
        + 516880.0 * eta_pow_2
        + 68880.0 * pi_pow_2 * eta
        - 1916048.0 * eta
        + 262080.0
    ) * e_pow_4
    e_6_term = (
        84000.0 * eta_pow_3
        - 190400.0 * eta_pow_2
        - 17220.0 * pi_pow_2 * eta
        - 50048.0 * eta
        - 241920.0
    ) * e_pow_6
    e_8_term = (-52640.0 * eta_pow_2 - 13440.0 * eta + 483280.0) * eta * e_pow_8
    e_10_term = (10080.0 * eta_pow_2 + 40320.0 * eta - 15120.0) * eta * e_pow_10

    cosu_1 = (
        (-2240.0 * eta_pow_3 - 168000.0 * eta_pow_2 - 424480.0 * eta) * e_pow_9
        + (
            28560.0 * eta_pow_3
            + 242480.0 * eta_pow_2
            + 34440.0 * pi_pow_2 * eta
            - 1340224.0 * eta
            + 725760.0
        )
        * e_pow_7
        + (
            -33040.0 * eta_pow_3
            - 754880.0 * eta_pow_2
            - 172200.0 * pi_pow_2 * eta
            + 5458480.0 * eta
            - 221760.0
        )
        * e_pow_5
        + (
            40880.0 * eta_pow_3
            + 738640.0 * eta_pow_2
            + 30135.0 * pi_pow_2 * eta
            + 1554048.0 * eta
            - 2936640.0
        )
        * e_pow_3
        + (
            -560.0 * eta_pow_3
            - 100240.0 * eta_pow_2
            - 43050.0 * pi_pow_2 * eta
            + 3284816.0 * eta
            - 389760.0
        )
        * e
    ) * cos_u

    cosu_2 = (
        (4480.0 * eta_pow_3 - 20160.0 * eta_pow_2 + 16800.0 * eta) * e_pow_10
        + (
            3920.0 * eta_pow_3
            + 475440.0 * eta_pow_2
            - 17220.0 * pi_pow_2 * eta
            + 831952.0 * eta
            - 7257600.0
        )
        * e_pow_8
        + (
            -75600.0 * eta_pow_3
            + 96880.0 * eta_pow_2
            + 154980.0 * pi_pow_2 * eta
            - 3249488.0 * eta
            - 685440.0
        )
        * e_pow_6
        + (
            5040.0 * eta_pow_3
            - 659120.0 * eta_pow_2
            + 25830.0 * pi_pow_2 * eta
            - 7356624.0 * eta
            + 6948480.0
        )
        * e_pow_4
        + (
            -5040.0 * eta_pow_3
            + 190960.0 * eta_pow_2
            + 137760.0 * pi_pow_2 * eta
            - 7307920.0 * eta
            + 107520.0
        )
        * e_pow_2
    ) * cos_u_pow_2

    cosu_3 = (
        (560.0 * eta_pow_3 - 137200.0 * eta_pow_2 + 388640.0 * eta + 241920.0) * e_pow_9
        + (
            30800.0 * eta_pow_3
            - 264880.0 * eta_pow_2
            - 68880.0 * pi_pow_2 * eta
            + 624128.0 * eta
            + 766080.0
        )
        * e_pow_7
        + (
            66640.0 * eta_pow_3
            + 612080.0 * eta_pow_2
            - 8610.0 * pi_pow_2 * eta
            + 6666080.0 * eta
            - 6652800.0
        )
        * e_pow_5
        + (
            -30800.0 * eta_pow_3
            - 294000.0 * eta_pow_2
            - 223860.0 * pi_pow_2 * eta
            + 9386432.0 * eta
        )
        * e_pow_3
    ) * cos_u_pow_3

    cosu_4 = (
        (-16240.0 * eta_pow_3 + 12880.0 * eta_pow_2 + 18480.0 * eta) * e_pow_10
        + (
            16240.0 * eta_pow_3
            - 91840.0 * eta_pow_2
            + 17220.0 * pi_pow_2 * eta
            - 652192.0 * eta
            + 100800.0
        )
        * e_pow_8
        + (
            -55440.0 * eta_pow_3
            + 34160.0 * eta_pow_2
            - 30135.0 * pi_pow_2 * eta
            - 2185040.0 * eta
            + 2493120.0
        )
        * e_pow_6
        + (
            21480.0 * eta_pow_3
            + 86800.0 * eta_pow_2
            + 163590.0 * pi_pow_2 * eta
            - 5713888.0 * eta
            + 228480.0
        )
        * e_pow_4
    ) * cos_u_pow_4

    cosu_5 = (
        (13440.0 * eta_pow_3 + 94640.0 * eta_pow_2 - 113680.0 * eta - 221760.0)
        * e_pow_9
        + (
            -11200.0 * eta_pow_3
            - 112000.0 * eta_pow_2
            + 12915.0 * pi_pow_2 * eta
            + 692928.0 * eta
            - 194880.0
        )
        * e_pow_7
        + (
            4480.0 * eta_pow_3
            + 8960.0 * eta_pow_2
            - 43050.0 * pi_pow_2 * eta
            + 1127280.0 * eta
            - 147840.0
        )
        * e_pow_5
    ) * cos_u_pow_5

    rt_zero = (
        -67200.0 * eta_pow_2
        + 761600.0 * eta
        + e_pow_4 * (40320.0 * eta_pow_2 + 309120.0 * eta - 672000.0)
        + e_pow_2
        * (
            208320.0 * eta_pow_2
            + 17220.0 * pi_pow_2 * eta
            - 2289280.0 * eta
            + 1680000.0
        )
        - 8610.0 * pi_pow_2 * eta
        - 201600.0
    )

    rt_cosu_1 = (
        (-282240.0 * eta_pow_2 - 450240.0 * eta + 1478400.0) * e_pow_5
        + (
            -719040.0 * eta_pow_2
            - 68880.0 * pi_pow_2 * eta
            + 8128960.0 * eta
            - 5040000.0
        )
        * e_pow_3
        + (94080.0 * eta_pow_2 + 25830.0 * pi_pow_2 * eta - 1585920.0 * eta - 470400.0)
        * e
    ) * cos_u

    rt_cosu_2 = (
        (604800.0 * eta_pow_2 - 504000.0 * eta - 403200.0) * e_pow_6
        + (
            1034880.0 * eta_pow_2
            + 103320.0 * pi_pow_2 * eta
            - 11195520.0 * eta
            + 5779200.0
        )
        * e_pow_4
        + (174720.0 * eta_pow_2 - 17220.0 * pi_pow_2 * eta - 486080.0 * eta + 2688000.0)
        * e_pow_2
    ) * cos_u_pow_2

    rt_cosu_3 = (
        (-524160.0 * eta_pow_2 + 1122240.0 * eta - 940800.0) * e_pow_7
        + (
            -873600.0 * eta_pow_2
            - 68880.0 * pi_pow_2 * eta
            + 7705600.0 * eta
            - 3897600.0
        )
        * e_pow_5
        + (
            -416640.0 * eta_pow_2
            - 17220.0 * pi_pow_2 * eta
            + 3357760.0 * eta
            - 3225600.0
        )
        * e_pow_3
    ) * cos_u_pow_3

    rt_cosu_4 = (
        (161280.0 * eta_pow_2 - 477120.0 * eta + 537600.0) * e_pow_8
        + (
            477120.0 * eta_pow_2
            + 17220.0 * pi_pow_2 * eta
            - 2894080.0 * eta
            + 2217600.0
        )
        * e_pow_6
        + (
            268800.0 * eta_pow_2
            + 25830.0 * pi_pow_2 * eta
            - 2721600.0 * eta
            + 1276800.0
        )
        * e_pow_4
    ) * cos_u_pow_4

    rt_cosu_5 = (
        (-127680.0 * eta_pow_2 + 544320.0 * eta - 739200.0) * e_pow_7
        + (-53760.0 * eta_pow_2 - 8610.0 * pi_pow_2 * eta + 674240.0 * eta - 67200.0)
        * e_pow_5
    ) * cos_u_pow_5

    return pre_factor * (
        e_0_term
        + e_2_term
        + e_4_term
        + e_6_term
        + e_8_term
        + e_10_term
        + cosu_1
        + cosu_2
        + cosu_3
        + cosu_4
        + cosu_5
        + jnp.sqrt(e_fact)
        * (rt_zero + rt_cosu_1 + rt_cosu_2 + rt_cosu_3 + rt_cosu_4 + rt_cosu_5)
    )


def phi_dot_3pn_SS_jax(e, m1, m2, S1z, S2z, u):
    """3PN SS phi_dot"""
    kappa1 = 1.0
    kappa2 = 1.0
    e_2 = e * e
    e_3 = e_2 * e
    e_4 = e_2 * e_2
    e_6 = e_4 * e_2
    e_fact = 1.0 - e_2
    e_fact_sqrt = jnp.sqrt(e_fact)
    M_fact_2 = (m1 + m2) ** 2
    M_fact_4 = M_fact_2 * M_fact_2
    cos_u = jnp.cos(u)
    cos_u_2 = cos_u * cos_u
    cos_u_3 = cos_u_2 * cos_u

    s1z2_m1_4 = (
        6
        * (
            4 * e_6 * e_fact_sqrt * kappa1
            - 2 * (-1 + e_fact_sqrt) * (4 + 7 * kappa1)
            - e_2 * (24 + 20 * e_fact_sqrt + 42 * kappa1 + 19 * e_fact_sqrt * kappa1)
            - 4 * e_4 * (-4 + (-7 + 3 * e_fact_sqrt) * kappa1)
        )
        * m1**4
        * S1z
        * S1z
    )

    s2z2_m2_4 = (
        6
        * (
            4 * e_6 * e_fact_sqrt * kappa2
            - 2 * (-1 + e_fact_sqrt) * (4 + 7 * kappa2)
            - e_2 * (24 + 20 * e_fact_sqrt + 42 * kappa2 + 19 * e_fact_sqrt * kappa2)
            - 4 * e_4 * (-4 + (-7 + 3 * e_fact_sqrt) * kappa2)
        )
        * m2**4
        * S2z
        * S2z
    )

    cross_m1_3_m2 = (
        m1**3
        * m2
        * S1z
        * (
            (
                48 * e_6 * e_fact_sqrt * (-1 + kappa1)
                - 6 * (-1 + e_fact_sqrt) * (3 + 23 * kappa1)
                - 4
                * e_4
                * (-9 - 60 * e_fact_sqrt - 69 * kappa1 + 32 * e_fact_sqrt * kappa1)
                - e_2
                * (54 + 297 * e_fact_sqrt + 414 * kappa1 + 145 * e_fact_sqrt * kappa1)
            )
            * S1z
            + 6
            * (
                60 * e_4
                + 4 * e_6 * e_fact_sqrt
                - 30 * (-1 + e_fact_sqrt)
                - e_2 * (90 + 67 * e_fact_sqrt)
            )
            * S2z
        )
    )

    cross_m1_m2_3 = (
        m1
        * m2**3
        * S2z
        * (
            6
            * (
                60 * e_4
                + 4 * e_6 * e_fact_sqrt
                - 30 * (-1 + e_fact_sqrt)
                - e_2 * (90 + 67 * e_fact_sqrt)
            )
            * S1z
            + (
                48 * e_6 * e_fact_sqrt * (-1 + kappa2)
                - 6 * (-1 + e_fact_sqrt) * (3 + 23 * kappa2)
                - 4
                * e_4
                * (-9 - 60 * e_fact_sqrt - 69 * kappa2 + 32 * e_fact_sqrt * kappa2)
                - e_2
                * (54 + 297 * e_fact_sqrt + 414 * kappa2 + 145 * e_fact_sqrt * kappa2)
            )
            * S2z
        )
    )

    cross_m1_2_m2_2 = (
        m1**2
        * m2**2
        * (
            3
            * (
                4 * e_6 * e_fact_sqrt * (-4 + 3 * kappa1)
                - 6 * (-1 + e_fact_sqrt) * (-1 + 4 * kappa1)
                - e_2 * (-18 + 55 * e_fact_sqrt + 4 * (18 + e_fact_sqrt) * kappa1)
                + e_4 * (-12 + 68 * e_fact_sqrt - 8 * (-6 + 5 * e_fact_sqrt) * kappa1)
            )
            * S1z
            * S1z
            + 2
            * (
                12 * e_6 * e_fact_sqrt
                - 174 * (-1 + e_fact_sqrt)
                + 4 * e_4 * (87 + 7 * e_fact_sqrt)
                - e_2 * (522 + 409 * e_fact_sqrt)
            )
            * S1z
            * S2z
            + 3
            * (
                4 * e_6 * e_fact_sqrt * (-4 + 3 * kappa2)
                - 6 * (-1 + e_fact_sqrt) * (-1 + 4 * kappa2)
                - e_2 * (-18 + 55 * e_fact_sqrt + 4 * (18 + e_fact_sqrt) * kappa2)
                + e_4 * (-12 + 68 * e_fact_sqrt - 8 * (-6 + 5 * e_fact_sqrt) * kappa2)
            )
            * S2z
            * S2z
        )
    )

    term_const = s1z2_m1_4 + s2z2_m2_4 + cross_m1_3_m2 + cross_m1_m2_3 + cross_m1_2_m2_2

    cosu_m1_4_S1z2 = (
        6
        * (
            -8
            + 40 * e_fact_sqrt
            + 2 * (-7 + 25 * e_fact_sqrt) * kappa1
            + 4 * e_4 * (-8 + (-14 + 3 * e_fact_sqrt) * kappa1)
            + e_2 * (40 + 44 * e_fact_sqrt + (70 + 61 * e_fact_sqrt) * kappa1)
        )
        * m1**4
        * S1z
        * S1z
    )

    cosu_m2_4_S2z2 = (
        6
        * (
            -8
            + 40 * e_fact_sqrt
            + 2 * (-7 + 25 * e_fact_sqrt) * kappa2
            + 4 * e_4 * (-8 + (-14 + 3 * e_fact_sqrt) * kappa2)
            + e_2 * (40 + 44 * e_fact_sqrt + (70 + 61 * e_fact_sqrt) * kappa2)
        )
        * m2**4
        * S2z
        * S2z
    )

    cosu_m1_3_m2 = (
        m1**3
        * m2
        * S1z
        * (
            (
                -18
                + 246 * e_fact_sqrt
                + 46 * (-3 + 10 * e_fact_sqrt) * kappa1
                + 2
                * e_4
                * (-36 - 60 * e_fact_sqrt - 276 * kappa1 + 47 * e_fact_sqrt * kappa1)
                + e_2 * (90 + 243 * e_fact_sqrt + (690 + 535 * e_fact_sqrt) * kappa1)
            )
            * S1z
            + 18
            * (
                -10
                + 42 * e_fact_sqrt
                + 4 * e_4 * (-10 + e_fact_sqrt)
                + e_2 * (50 + 47 * e_fact_sqrt)
            )
            * S2z
        )
    )

    cosu_m1_m2_3 = (
        m1
        * m2**3
        * S2z
        * (
            18
            * (
                -10
                + 42 * e_fact_sqrt
                + 4 * e_4 * (-10 + e_fact_sqrt)
                + e_2 * (50 + 47 * e_fact_sqrt)
            )
            * S1z
            + (
                -18
                + 246 * e_fact_sqrt
                + 46 * (-3 + 10 * e_fact_sqrt) * kappa2
                + 2
                * e_4
                * (-36 - 60 * e_fact_sqrt - 276 * kappa2 + 47 * e_fact_sqrt * kappa2)
                + e_2 * (90 + 243 * e_fact_sqrt + (690 + 535 * e_fact_sqrt) * kappa2)
            )
            * S2z
        )
    )

    cosu_m1_2_m2_2 = (
        m1**2
        * m2**2
        * (
            3
            * (
                6
                + 14 * e_fact_sqrt
                + 8 * (-3 + 8 * e_fact_sqrt) * kappa1
                + 4 * e_4 * (6 - 7 * e_fact_sqrt + 8 * (-3 + e_fact_sqrt) * kappa1)
                + e_2 * (5 * (-6 + e_fact_sqrt) + 24 * (5 + 3 * e_fact_sqrt) * kappa1)
            )
            * S1z
            * S1z
            + 2
            * (
                -174
                + 760 * e_fact_sqrt
                + e_4 * (-696 + 34 * e_fact_sqrt)
                + e_2 * (870 + 835 * e_fact_sqrt)
            )
            * S1z
            * S2z
            + 3
            * (
                6
                + 14 * e_fact_sqrt
                + 8 * (-3 + 8 * e_fact_sqrt) * kappa2
                + 4 * e_4 * (6 - 7 * e_fact_sqrt + 8 * (-3 + e_fact_sqrt) * kappa2)
                + e_2 * (5 * (-6 + e_fact_sqrt) + 24 * (5 + 3 * e_fact_sqrt) * kappa2)
            )
            * S2z
            * S2z
        )
    )

    term_cosu = (
        e
        * (
            cosu_m1_4_S1z2
            + cosu_m2_4_S2z2
            + cosu_m1_3_m2
            + cosu_m1_m2_3
            + cosu_m1_2_m2_2
        )
        * cos_u
    )

    cosu2_m1_4_S1z2 = (
        6
        * (
            8
            + 56 * e_fact_sqrt
            + 14 * kappa1
            + 58 * e_fact_sqrt * kappa1
            + 4 * e_4 * (-4 - 7 * kappa1 + 3 * e_fact_sqrt * kappa1)
            + e_2 * (8 + 28 * e_fact_sqrt + 14 * kappa1 + 53 * e_fact_sqrt * kappa1)
        )
        * m1**4
        * S1z
        * S1z
    )

    cosu2_m2_4_S2z2 = (
        6
        * (
            8
            + 56 * e_fact_sqrt
            + 14 * kappa2
            + 58 * e_fact_sqrt * kappa2
            + 4 * e_4 * (-4 - 7 * kappa2 + 3 * e_fact_sqrt * kappa2)
            + e_2 * (8 + 28 * e_fact_sqrt + 14 * kappa2 + 53 * e_fact_sqrt * kappa2)
        )
        * m2**4
        * S2z
        * S2z
    )

    cosu2_m1_3_m2 = (
        m1**3
        * m2
        * S1z
        * (
            (
                2
                * e_4
                * (-18 - 12 * e_fact_sqrt - 138 * kappa1 + 55 * e_fact_sqrt * kappa1)
                + 2 * (9 + 111 * e_fact_sqrt + 69 * kappa1 + 262 * e_fact_sqrt * kappa1)
                + e_2
                * (18 + 171 * e_fact_sqrt + 138 * kappa1 + 455 * e_fact_sqrt * kappa1)
            )
            * S1z
            + 18
            * (
                10
                + 46 * e_fact_sqrt
                + 4 * e_4 * (-5 + 2 * e_fact_sqrt)
                + e_2 * (10 + 39 * e_fact_sqrt)
            )
            * S2z
        )
    )

    cosu2_m1_m2_3 = (
        m1
        * m2**3
        * S2z
        * (
            18
            * (
                10
                + 46 * e_fact_sqrt
                + 4 * e_4 * (-5 + 2 * e_fact_sqrt)
                + e_2 * (10 + 39 * e_fact_sqrt)
            )
            * S1z
            + (
                2
                * e_4
                * (-18 - 12 * e_fact_sqrt - 138 * kappa2 + 55 * e_fact_sqrt * kappa2)
                + 2 * (9 + 111 * e_fact_sqrt + 69 * kappa2 + 262 * e_fact_sqrt * kappa2)
                + e_2
                * (18 + 171 * e_fact_sqrt + 138 * kappa2 + 455 * e_fact_sqrt * kappa2)
            )
            * S2z
        )
    )

    cosu2_m1_2_m2_2 = (
        m1**2
        * m2**2
        * (
            3
            * (
                -6
                - 14 * e_fact_sqrt
                + 24 * kappa1
                + 68 * e_fact_sqrt * kappa1
                + 4
                * e_4
                * (3 - 2 * e_fact_sqrt - 12 * kappa1 + 7 * e_fact_sqrt * kappa1)
                + e_2
                * (-6 + 13 * e_fact_sqrt + 24 * kappa1 + 72 * e_fact_sqrt * kappa1)
            )
            * S1z
            * S1z
            + 2
            * (
                174
                + 872 * e_fact_sqrt
                + e_4 * (-348 + 98 * e_fact_sqrt)
                + e_2 * (174 + 659 * e_fact_sqrt)
            )
            * S1z
            * S2z
            + 3
            * (
                -6
                - 14 * e_fact_sqrt
                + 24 * kappa2
                + 68 * e_fact_sqrt * kappa2
                + 4
                * e_4
                * (3 - 2 * e_fact_sqrt - 12 * kappa2 + 7 * e_fact_sqrt * kappa2)
                + e_2
                * (-6 + 13 * e_fact_sqrt + 24 * kappa2 + 72 * e_fact_sqrt * kappa2)
            )
            * S2z
            * S2z
        )
    )

    term_cosu2 = (
        -e_2
        * (
            cosu2_m1_4_S1z2
            + cosu2_m2_4_S2z2
            + cosu2_m1_3_m2
            + cosu2_m1_m2_3
            + cosu2_m1_2_m2_2
        )
        * cos_u_2
    )

    cosu3_m1_4_S1z2 = (
        6
        * (
            8
            + 24 * e_fact_sqrt
            + 2 * (7 + 9 * e_fact_sqrt) * kappa1
            + e_2 * (-8 + 4 * e_fact_sqrt - 14 * kappa1 + 23 * e_fact_sqrt * kappa1)
        )
        * m1**4
        * S1z
        * S1z
    )

    cosu3_m2_4_S2z2 = (
        6
        * (
            8
            + 24 * e_fact_sqrt
            + 2 * (7 + 9 * e_fact_sqrt) * kappa2
            + e_2 * (-8 + 4 * e_fact_sqrt - 14 * kappa2 + 23 * e_fact_sqrt * kappa2)
        )
        * m2**4
        * S2z
        * S2z
    )

    cosu3_m1_3_m2 = (
        m1**3
        * m2
        * S1z
        * (
            (
                18
                + 42 * e_fact_sqrt
                + 2 * (69 + 77 * e_fact_sqrt) * kappa1
                + e_2
                * (-18 + 81 * e_fact_sqrt - 138 * kappa1 + 209 * e_fact_sqrt * kappa1)
            )
            * S1z
            + 6 * (30 + 38 * e_fact_sqrt + 5 * e_2 * (-6 + 11 * e_fact_sqrt)) * S2z
        )
    )

    cosu3_m1_m2_3 = (
        m1
        * m2**3
        * S2z
        * (
            6 * (30 + 38 * e_fact_sqrt + 5 * e_2 * (-6 + 11 * e_fact_sqrt)) * S1z
            + (
                18
                + 42 * e_fact_sqrt
                + 2 * (69 + 77 * e_fact_sqrt) * kappa2
                + e_2
                * (-18 + 81 * e_fact_sqrt - 138 * kappa2 + 209 * e_fact_sqrt * kappa2)
            )
            * S2z
        )
    )

    cosu3_m1_2_m2_2 = (
        m1**2
        * m2**2
        * (
            3
            * (
                -6
                - 18 * e_fact_sqrt
                + 8 * (3 + 2 * e_fact_sqrt) * kappa1
                + e_2 * (6 + 15 * e_fact_sqrt + 8 * (-3 + 5 * e_fact_sqrt) * kappa1)
            )
            * S1z
            * S1z
            + 2
            * (174 + 274 * e_fact_sqrt + e_2 * (-174 + 269 * e_fact_sqrt))
            * S1z
            * S2z
            + 3
            * (
                -6
                - 18 * e_fact_sqrt
                + 8 * (3 + 2 * e_fact_sqrt) * kappa2
                + e_2 * (6 + 15 * e_fact_sqrt + 8 * (-3 + 5 * e_fact_sqrt) * kappa2)
            )
            * S2z
            * S2z
        )
    )

    term_cosu3 = (
        e_3
        * (
            cosu3_m1_4_S1z2
            + cosu3_m2_4_S2z2
            + cosu3_m1_3_m2
            + cosu3_m1_m2_3
            + cosu3_m1_2_m2_2
        )
        * cos_u_3
    )

    denominator = 12.0 * (e_2 - 1.0) ** 3 * M_fact_4 * (-1.0 + e * cos_u) ** 5

    return (term_const + term_cosu + term_cosu2 + term_cosu3) / denominator


def phi_dot_4pn_SS_jax(e, m1, m2, S1z, S2z):
    """4PN SS phi_dot — returns 0."""
    return 0.0


def phi_dot_4_5_pn_jax(e, eta, x):
    """4.5PN phi_dot — returns 0."""
    return 0.0


# ============ Dispatcher functions ==========================================


def dx_dt_jax(e, eta, m1, m2, S1z, S2z, x, radiation_pn_order, x_dot_4pn_SF_val=0.0):
    """Compute dx/dt at given PN order. radiation_pn_order is static (Python int)."""
    x2 = x * x
    x3 = x2 * x
    x5 = x3 * x2
    sqx = jnp.sqrt(x)
    xsqx = x * sqx
    x2sqx = x2 * sqx
    x3sqx = x3 * sqx

    inst = x_dot_0pn_jax(e, eta)

    if radiation_pn_order >= 2:
        inst = inst + x_dot_1pn_jax(e, eta) * x

    if radiation_pn_order >= 3:
        inst = inst + x_dot_1_5_pn_jax(e, eta, m1, m2, S1z, S2z) * xsqx

    if radiation_pn_order >= 4:
        inst = inst + x_dot_2pn_jax(e, eta, x) * x2
        inst = inst + x_dot_2pn_SS_jax(e, eta, m1, m2, S1z, S2z) * x2

    if radiation_pn_order >= 5:
        inst = inst + x_dot_2_5pn_SO_jax(e, eta, m1, m2, S1z, S2z) * x2sqx
        inst = inst + x_dot_2_5pn_SF_jax(e, eta, S1z) * x2sqx

    if radiation_pn_order >= 6:
        inst = inst + x_dot_3pn_jax(e, eta, x) * x3
        inst = inst + x_dot_3pn_SO_jax(e, eta, m1, m2, S1z, S2z) * x3
        inst = inst + x_dot_3pn_SS_jax(e, eta, m1, m2, S1z, S2z) * x3

    if radiation_pn_order >= 7:
        inst = inst + x_dot_3_5pnSO_jax(e, eta, m1, m2, S1z, S2z) * x3sqx
        inst = inst + x_dot_3_5_pn_jax(e, eta) * x3sqx
        inst = inst + x_dot_3_5pn_SS_jax(e, eta, m1, m2, S1z, S2z) * x3sqx
        inst = inst + x_dot_3_5pn_cubicSpin_jax(e, eta, m1, m2, S1z, S2z) * x3sqx
        inst = inst + x_dot_3_5pn_SF_jax(e, eta, S1z) * x3sqx

    if radiation_pn_order >= 8:
        inst = inst + (
            x_dot_4pn_jax(e, eta, x)
            + x_dot_4pnSO_jax(e, eta, m1, m2, S1z, S2z)
            + x_dot_4pnSS_jax(e, eta, m1, m2, S1z, S2z)
            + x_dot_4pn_SF_val
        ) * (x2 * x2)

    if radiation_pn_order >= 9:
        inst = inst + x_dot_4_5_pn_jax(e, eta, x) * (x2 * x2) * sqx

    result = inst * x5

    # Hereditary part (not multiplied by x^5)
    if radiation_pn_order >= 3:
        result = result + x_dot_hereditary_1_5_jax(e, eta, x)

    if radiation_pn_order >= 5:
        result = result + x_dot_hereditary_2_5_jax(e, eta, x)

    if radiation_pn_order >= 6:
        result = result + x_dot_hereditary_3_jax(e, eta, x)

    return result


def de_dt_jax(e, eta, m1, m2, S1z, S2z, x, radiation_pn_order):
    """Compute de/dt at given PN order. radiation_pn_order is static (Python int)."""
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    sqx = jnp.sqrt(x)
    xsqx = x * sqx
    x2sqx = x2 * sqx
    x3sqx = x3 * sqx

    inst = e_dot_0pn_jax(e, eta)

    if radiation_pn_order >= 1:
        inst = inst + e_dot_1pn_jax(e, eta) * x

    if radiation_pn_order >= 3:
        inst = inst + e_dot_1_5pn_SO_jax(e, m1, m2, S1z, S2z) * xsqx

    if radiation_pn_order >= 4:
        inst = inst + e_dot_2pn_jax(e, eta) * x2
        inst = inst + e_dot_2pn_SS_jax(e, m1, m2, S1z, S2z) * x2

    if radiation_pn_order >= 5:
        inst = inst + e_dot_2_5pn_SO_jax(e, m1, m2, S1z, S2z) * x2sqx

    if radiation_pn_order >= 6:
        inst = inst + e_dot_3pn_jax(e, eta, x) * x3
        inst = inst + e_dot_3pn_SO_jax(e, m1, m2, S1z, S2z) * x3
        inst = inst + e_dot_3pn_SS_jax(e, m1, m2, S1z, S2z) * x3

    if radiation_pn_order >= 7:
        inst = inst + e_dot_3_5pn_jax(e, eta) * x3sqx

    result = inst * x4

    # Hereditary part
    if radiation_pn_order >= 3:
        result = result + e_rad_hereditary_1_5_jax(e, eta, x)

    if radiation_pn_order >= 5:
        result = result + e_rad_hereditary_2_5_jax(e, eta, x)

    if radiation_pn_order >= 6:
        result = result + e_rad_hereditary_3_jax(e, eta, x)

    return result


def dl_dt_jax(e, eta, m1, m2, S1z, S2z, x, radiation_pn_order):
    """Compute dl/dt — 4PN accurate with spin corrections (matches LALSimESIGMA C code)."""
    x32 = jnp.sqrt(x) * x
    return (
        1.0
        + x * l_dot_1pn_jax(e, eta)
        + x32 * l_dot_1_5pn_SO_jax(e, m1, m2, S1z, S2z)
        + x * x * l_dot_2pn_jax(e, eta)
        + x * x * l_dot_2pn_SS_jax(e, m1, m2, S1z, S2z)
        + l_dot_2_5pn_SO_jax(e, m1, m2, S1z, S2z) * x32 * x
        + x**3 * l_dot_3pn_jax(e, eta)
        + x**3 * l_dot_3pn_SS_jax(e, m1, m2, S1z, S2z)
        + l_dot_4pn_jax(e, eta) * x**4
    ) * x32


def dphi_dt_jax(u, eta, m1, m2, S1z, S2z, x, e, vpnorder):
    """Compute dphi/dt — 4PN accurate."""
    x32 = jnp.sqrt(x) * x
    return (
        phi_dot_0pn_jax(e, eta, u)
        + x * phi_dot_1pn_jax(e, eta, u)
        + x32 * phi_dot_1_5_pnSO_ecc_jax(e, m1, m2, S1z, S2z, u)
        + x * x * phi_dot_2_pnSS_ecc_jax(e, m1, m2, S1z, S2z, u)
        + x * x * phi_dot_2pn_jax(e, eta, u)
        + x32 * x * phi_dot_2_5pn_SO_jax(e, m1, m2, S1z, S2z, u)
        + x**3 * phi_dot_3pn_jax(e, eta, u)
        + x32 * x32 * phi_dot_3pn_SS_jax(e, m1, m2, S1z, S2z, u)
        + phi_dot_4pn_SS_jax(e, m1, m2, S1z, S2z) * x32 * x * x32
        + phi_dot_4_5_pn_jax(e, eta, x) * x32**3
    ) * x32


# ============ ODE system ===================================================


def eccentric_x_model_odes_jax(t, y, args):
    """
    ODE RHS for eccentric gravitational-wave inspiral.

    State:  y = [x, e, l, phi]
    Args:   tuple (eta, m1, m2, S1z, S2z, rad_pn_order, vpnorder[, x_dot_4pn_SF_val])

    Returns jnp.array([xdot, edot, ldot, phidot])
    """
    if len(args) == 8:
        eta, m1, m2, S1z, S2z, rad_pn_order, vpnorder, x_dot_4pn_SF_val = args
    else:
        eta, m1, m2, S1z, S2z, rad_pn_order, vpnorder = args
        x_dot_4pn_SF_val = 0.0

    x = y[0]
    e = y[1]
    l = y[2]

    u = solve_kepler_jax(l, e)

    xdot = dx_dt_jax(e, eta, m1, m2, S1z, S2z, x, rad_pn_order, x_dot_4pn_SF_val)
    edot = de_dt_jax(e, eta, m1, m2, S1z, S2z, x, rad_pn_order)
    ldot = dl_dt_jax(e, eta, m1, m2, S1z, S2z, x, rad_pn_order)
    phidot = dphi_dt_jax(u, eta, m1, m2, S1z, S2z, x, e, vpnorder)

    return jnp.array([xdot, edot, ldot, phidot])
