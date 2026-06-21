"""
Global configuration for the esigmapy package.

Usage:
    import esigmapy
    esigmapy.configure(default_backend="numba:jax", n_max=80000)
"""


class ESIGMAConfig:
    default_backend: str = "lalsim"
    n_max: int = 80000
    ode_eps: float = 1e-8
    rad_pn_order: int = 8
    mode_pn_order: int = 8


_config = ESIGMAConfig()


def configure(**kwargs):
    """Update global esigmapy configuration.

    Parameters
    ----------
    default_backend : str
        Default backend for waveform generation. Options:
        "lalsim", "numba", "jax", "surrogate", or hybrid "numba:jax".
    n_max : int
        Maximum array length for JAX padding (avoids recompilation).
    ode_eps : float
        Default ODE tolerance.
    rad_pn_order : int
        Default radiation-reaction PN order.
    mode_pn_order : int
        Default waveform mode PN order.
    """
    for k, v in kwargs.items():
        if not hasattr(_config, k):
            raise ValueError(f"Unknown config key: {k}")
        setattr(_config, k, v)


def get_config() -> ESIGMAConfig:
    """Return the current global configuration."""
    return _config
