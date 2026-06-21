"""
JAX backend for ESIGMA inspiral waveforms.

Fully differentiable implementation using diffrax ODE integration
and jax.vmap-vectorized mode computation.
"""

import jax
jax.config.update("jax_enable_x64", True)

from .kepler import (
    solve_kepler_jax,
    separation_jax,
    compute_state_arrays_jax,
)

from .inspiral import (
    phi_e_jax,
    psi_e_jax,
    zed_e_jax,
    kappa_e_jax,
    phi_e_tilde_jax,
    psi_e_tilde_jax,
    zed_e_tilde_jax,
    kappa_e_tilde_jax,
    phi_e_rad_jax,
    psi_e_rad_jax,
    zed_e_rad_jax,
    kappa_e_rad_jax,
    f_e_jax,
    capital_f_e_jax,
    psi_n_jax,
    zed_n_jax,
    dx_dt_jax,
    de_dt_jax,
    dl_dt_jax,
    dphi_dt_jax,
    eccentric_x_model_odes_jax,
)

from .generator import (
    estimate_T_max_jax,
    integrate_esigma_dynamics_jax,
    inspiral_esigma_dynamics_jax,
    compute_mode_from_dynamics_jax,
    get_inspiral_esigma_modes_jax,
    get_inspiral_esigma_waveform_jax,
)

try:
    from .go_terms import (
        CommonVars,
        generate_hlm_jax,
        hlmGOresult_jax,
    )
except ImportError:
    pass
