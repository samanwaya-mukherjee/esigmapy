"""
esigmapy.jax_codes
==================
Fully differentiable JAX implementation of the ESIGMA gravitational-wave model.

Public API
----------
Dynamics / integration
  inspiral_esigma_dynamics_jax  — integrate ODE, return dynamics dict
  integrate_esigma_dynamics_jax — low-level diffrax call, returns (t, y)

Waveform
  get_inspiral_esigma_modes_jax     — return dict of complex h_lm arrays
  get_inspiral_esigma_waveform_jax  — return (h_plus, h_cross)

Primitives
  solve_kepler_jax     — eccentric anomaly from mean anomaly
  separation_jax       — 3PN orbital separation
  eccentric_x_model_odes_jax — ODE RHS for use with diffrax / jax.odeint
  hlmGOresult_jax      — single (l, m) mode from instantaneous orbital state
"""

import jax
# Required: ESIGMA uses float64 throughout; without this JAX silently
# downgrades float64 inputs to float32, breaking ODE convergence.
jax.config.update("jax_enable_x64", True)

from .esigma_jax_kepler import (
    solve_kepler_jax,
    separation_jax,
    compute_state_arrays_jax,
)

from .esigma_jax_inspiral import (
    # Enhancement functions
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
    # ODE dispatcher and system
    dx_dt_jax,
    de_dt_jax,
    dl_dt_jax,
    dphi_dt_jax,
    eccentric_x_model_odes_jax,
)

from .esigma_jax_main import (
    estimate_T_max_jax,
    integrate_esigma_dynamics_jax,
    inspiral_esigma_dynamics_jax,
    compute_mode_from_dynamics_jax,
    get_inspiral_esigma_modes_jax,
    get_inspiral_esigma_waveform_jax,
)

# esigma_jax_go_terms exports (available after Phase 5 is written)
try:
    from .esigma_jax_go_terms import (
        CommonVars,
        generate_hlm_jax,
        hlmGOresult_jax,
    )
except ImportError:
    pass  # go_terms not yet available; dynamics-only features still work
