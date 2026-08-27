from __future__ import absolute_import

from . import legacy, utils, condition, config, blend
from .config import configure, get_config

from .inspiral.lalsimulation_backend.generator import (
    eccentricity_at_extremum_frequency,
    eccentricity_at_reference_frequency,
    get_imr_esigma_modes,
    get_imr_esigma_waveform,
    get_inspiral_esigma_modes,
    get_inspiral_esigma_waveform,
)

from .inspiral import get_inspiral_modes, get_inspiral_dynamics, get_inspiral_waveform


def get_version_information():
    import os

    version_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "esigmapy/.version"
    )
    try:
        with open(version_file, "r") as f:
            return f.readline().rstrip()
    except EnvironmentError:
        print("No version information file '.version' found")


# pycbc wrapper
import inspect

_params = inspect.signature(get_imr_esigma_waveform).parameters.keys()


def pycbc_esigma(**params):
    from pycbc.waveform.waveform import parse_mode_array

    # Pass all parameters the model suppports
    pt = {p: params[p] for p in _params if p in params}
    pt["f_ref"] = pt["f_lower"] if pt["f_ref"] == 0 else pt["f_ref"]

    gen_wav = get_imr_esigma_waveform
    if "skip_merger" in params:
        gen_wav = get_inspiral_esigma_waveform

    modes = params.pop("mode_array", [(2, 2), (2, -2)])
    modes = parse_mode_array({"mode_array": modes})["mode_array"]

    return gen_wav(
        pt.pop("mass1"),
        pt.pop("mass2"),
        pt.pop("f_lower"),
        pt.pop("delta_t"),
        **pt,
        modes_to_use=modes
    )

def get_imr_modes(**params):

    # Pass all parameters the model suppports
    pt = {p: params[p] for p in _params if p in params}
    pt["f_ref"] = pt.get("f_ref") or pt["f_lower"]

    backend = params.pop("backend","numba")

    import importlib

    module = importlib.import_module(
        f"esigmapy.inspiral.{backend}_backend"
    )

    gen_wav = getattr(module, "get_imr_esigma_modes_py")

    if "skip_merger" in params:
        gen_wav = get_inspiral_esigma_waveform

    return gen_wav(
        pt.pop("mass1"),
        pt.pop("mass2"),
        pt.pop("f_lower"),
        pt.pop("delta_t"),
        **pt,
    )

def get_imr_waveform(**params):

    # Pass all parameters the model suppports
    pt = {p: params[p] for p in _params if p in params}
    pt["f_ref"] = pt.get("f_ref") or pt["f_lower"]

    backend = params.pop("backend","numba")

    import importlib

    module = importlib.import_module(
        f"esigmapy.inspiral.{backend}_backend"
    )

    gen_wav = getattr(module, "get_imr_esigma_waveform_py")

    if "skip_merger" in params:
        gen_wav = get_inspiral_esigma_waveform

    return gen_wav(
        pt.pop("mass1"),
        pt.pop("mass2"),
        pt.pop("f_lower"),
        pt.pop("delta_t"),
        **pt,
    )


__version__ = get_version_information()
