"""
esigmapy.inspiral
=================
Unified inspiral waveform generation API with backend dispatch.

Supports single backends ("numba", "jax", "lalsim", "surrogate")
and hybrid backends ("numba:jax" for numba dynamics + JAX modes).
"""


def _resolve_backend(backend_str):
    """Parse 'numba:jax' -> ('numba', 'jax') or 'numba' -> ('numba', 'numba')."""
    from ..config import get_config

    backend_str = backend_str or get_config().default_backend
    if ":" in backend_str:
        dyn_backend, modes_backend = backend_str.split(":", 1)
    else:
        dyn_backend = modes_backend = backend_str
    return dyn_backend, modes_backend


def _get_backend_module(name):
    if name == "lalsim":
        from .lalsimulation_backend import generator as mod
    elif name == "numba":
        from .numba_backend import generator as mod
    elif name == "jax":
        from .jax_backend import generator as mod
    elif name == "surrogate":
        from .surrogate_backend import generator as mod
    else:
        raise ValueError(
            f"Unknown backend: {name!r}. "
            f"Choose from: 'lalsim', 'numba', 'jax', 'surrogate', "
            f"or a hybrid like 'numba:jax'."
        )
    return mod


def get_modes(mass1, mass2, f_lower, delta_t, *, backend=None, **kwargs):
    """Generate inspiral GW modes.

    Parameters
    ----------
    mass1, mass2 : float
        Component masses in solar masses.
    f_lower : float
        Starting GW frequency in Hz.
    delta_t : float
        Output time step in seconds.
    backend : str, optional
        Backend to use. Single ("numba", "jax", "lalsim", "surrogate")
        or hybrid ("numba:jax"). Defaults to config.default_backend.
    **kwargs
        Backend-specific options (eccentricity, spin1z, etc.)

    Returns
    -------
    dict : mapping (l, m) -> complex array
    """
    dyn_back, modes_back = _resolve_backend(backend)
    if dyn_back == modes_back:
        return _get_backend_module(dyn_back).get_modes(
            mass1, mass2, f_lower, delta_t, **kwargs
        )
    else:
        dyn = _get_backend_module(dyn_back).get_dynamics(
            mass1, mass2, f_lower, delta_t, **kwargs
        )
        return _get_backend_module(modes_back).get_modes_from_dynamics(
            dyn, mass1, mass2, **kwargs
        )


def get_dynamics(mass1, mass2, f_lower, delta_t, *, backend=None, **kwargs):
    """Generate inspiral orbital dynamics.

    Returns
    -------
    dict with keys: time_evol, x_evol, eccentricity_evol, mean_ano_evol,
                    phi_evol, phi_dot_evol, r_evol, r_dot_evol
    """
    dyn_back, _ = _resolve_backend(backend)
    return _get_backend_module(dyn_back).get_dynamics(
        mass1, mass2, f_lower, delta_t, **kwargs
    )


def get_waveform(mass1, mass2, f_lower, delta_t, *, backend=None, **kwargs):
    """Generate inspiral GW polarizations (h_plus, h_cross).

    Returns
    -------
    tuple : (h_plus, h_cross) arrays
    """
    dyn_back, modes_back = _resolve_backend(backend)
    if dyn_back == modes_back:
        return _get_backend_module(dyn_back).get_waveform(
            mass1, mass2, f_lower, delta_t, **kwargs
        )
    else:
        dyn = _get_backend_module(dyn_back).get_dynamics(
            mass1, mass2, f_lower, delta_t, **kwargs
        )
        return _get_backend_module(modes_back).get_waveform_from_dynamics(
            dyn, mass1, mass2, **kwargs
        )
