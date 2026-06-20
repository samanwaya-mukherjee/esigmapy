#!/usr/bin/env python3
"""
test_ode_rhs_agreement.py
=========================
Compare ODE RHS (xdot, edot, ldot, phidot) across three implementations:

    LALSim C  — via ultra-fine finite-difference on SimInspiralESIGMADynamics
    Python    — direct call to eccentric_x_model_odes (numba)
    JAX       — direct call to eccentric_x_model_odes_jax

Each system is evaluated at the *initial* state (x0, e0, l=0, phi=0).
Pass/fail thresholds distinguish equation-level errors from finite-difference
noise in the LALSim estimate.

Usage:
    conda run -n lalsuite-dev python tests/test_ode_rhs_agreement.py
    conda run -n lalsuite-dev python tests/test_ode_rhs_agreement.py --verbose
    conda run -n lalsuite-dev python tests/test_ode_rhs_agreement.py --tol 1e-4
"""
import argparse
import os
import sys
import textwrap

ROOT_DIR = os.environ.get(
    "ESIGMAPY_ROOT_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)
sys.path.insert(0, ROOT_DIR)

import numpy as np

# ---------------------------------------------------------------------------
# Test configurations — a mix of mass ratios, spins, and eccentricities
# ---------------------------------------------------------------------------
SYSTEMS = [
    {"label": "equal-mass nonspinning e=0.3",
     "m1": 20.0, "m2": 20.0, "S1z": 0.0,  "S2z": 0.0,  "ecc": 0.3,  "f_lower": 20.0},
    {"label": "equal-mass nonspinning e=0.1",
     "m1": 20.0, "m2": 20.0, "S1z": 0.0,  "S2z": 0.0,  "ecc": 0.1,  "f_lower": 20.0},
    {"label": "equal-mass spinning e=0.2",
     "m1": 15.0, "m2": 15.0, "S1z": 0.3,  "S2z": -0.2, "ecc": 0.2,  "f_lower": 20.0},
    {"label": "unequal-mass nonspinning e=0.1",
     "m1": 30.0, "m2": 10.0, "S1z": 0.0,  "S2z": 0.0,  "ecc": 0.1,  "f_lower": 15.0},
    {"label": "unequal-mass spinning e=0.3",
     "m1": 25.0, "m2": 8.0,  "S1z": 0.5,  "S2z": 0.1,  "ecc": 0.3,  "f_lower": 15.0},
    {"label": "high-spin e=0.05",
     "m1": 20.0, "m2": 20.0, "S1z": 0.7,  "S2z": -0.7, "ecc": 0.05, "f_lower": 20.0},
    {"label": "low-ecc e=0.01",
     "m1": 20.0, "m2": 15.0, "S1z": 0.0,  "S2z": 0.0,  "ecc": 0.01, "f_lower": 20.0},
    {"label": "high-ecc e=0.5",
     "m1": 20.0, "m2": 20.0, "S1z": 0.0,  "S2z": 0.0,  "ecc": 0.5,  "f_lower": 15.0},
    {"label": "anti-aligned spins e=0.15",
     "m1": 18.0, "m2": 12.0, "S1z": -0.4, "S2z": 0.4,  "ecc": 0.15, "f_lower": 20.0},
    {"label": "heavy equal-mass e=0.2",
     "m1": 40.0, "m2": 40.0, "S1z": 0.1,  "S2z": 0.1,  "ecc": 0.2,  "f_lower": 10.0},
]


def rel_diff(a, b):
    denom = max(abs(a), abs(b), 1e-30)
    return abs(a - b) / denom


# ---------------------------------------------------------------------------
# LALSim RHS via ultra-fine finite difference
# ---------------------------------------------------------------------------
def lalsim_rhs_fd(m1, m2, S1z, S2z, ecc, f_lower, dt_fine_sec=1e-7):
    """Return (xdot, edot, ldot, phidot) at the initial state from LALSim,
    estimated via forward finite-difference at a very small dt."""
    import lalsimulation as ls

    total_mass = m1 + m2

    retval = ls.SimInspiralESIGMADynamics(
        m1, m2, S1z, S2z, ecc, f_lower, 0.0, 1e-12, 1.0 / dt_fine_sec
    )
    t = np.array(retval[0].data.data)
    x = np.array(retval[1].data.data)
    e = np.array(retval[2].data.data)
    l = np.array(retval[3].data.data)
    phi = np.array(retval[4].data.data)

    dt_geom = t[1] - t[0]  # geometric time step (M=1 units)

    return {
        "xdot": (x[1] - x[0]) / dt_geom,
        "edot": (e[1] - e[0]) / dt_geom,
        "ldot": (l[1] - l[0]) / dt_geom,
        "phidot": (phi[1] - phi[0]) / dt_geom,
        "x0": x[0],
        "dt_geom": dt_geom,
    }


# ---------------------------------------------------------------------------
# Python (numba) RHS — direct evaluation
# ---------------------------------------------------------------------------
def python_rhs(m1, m2, S1z, S2z, ecc, f_lower, rad_pn_order=8):
    import lal

    from esigmapy.python_codes.esigma_pn_inspiral import (
        eccentric_x_model_odes,
        x_dot_4pn_SF,
    )

    total_mass = m1 + m2
    eta = (m1 * m2) / total_mass**2
    omega_init = lal.PI * f_lower * lal.MTSUN_SI
    x0 = (total_mass * omega_init) ** (2.0 / 3.0)

    x_dot_4pn_SF_val = x_dot_4pn_SF(ecc, eta, S1z)
    y0 = np.array([x0, ecc, 0.0, 0.0])
    dydt = eccentric_x_model_odes(
        0.0, y0, eta, m1, m2, S1z, S2z, rad_pn_order, x_dot_4pn_SF_val
    )

    return {
        "xdot": dydt[0],
        "edot": dydt[1],
        "ldot": dydt[2],
        "phidot": dydt[3],
        "x0": x0,
    }


# ---------------------------------------------------------------------------
# JAX RHS — direct evaluation
# ---------------------------------------------------------------------------
def jax_rhs(m1, m2, S1z, S2z, ecc, f_lower, rad_pn_order=8):
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    import lal

    from esigmapy.jax_codes.esigma_jax_inspiral import eccentric_x_model_odes_jax

    total_mass = m1 + m2
    eta = (m1 * m2) / total_mass**2
    omega_init = lal.PI * f_lower * lal.MTSUN_SI
    x0 = (total_mass * omega_init) ** (2.0 / 3.0)

    args = (eta, m1, m2, S1z, S2z, rad_pn_order, rad_pn_order)
    y0 = jnp.array([x0, ecc, 0.0, 0.0])
    dydt = eccentric_x_model_odes_jax(0.0, y0, args)

    return {
        "xdot": float(dydt[0]),
        "edot": float(dydt[1]),
        "ldot": float(dydt[2]),
        "phidot": float(dydt[3]),
        "x0": float(x0),
    }


# ---------------------------------------------------------------------------
# Run comparison for a single system
# ---------------------------------------------------------------------------
def compare_system(sys_params, tol_py_jax, tol_vs_lalsim, verbose=False):
    """Compare RHS across all three implementations. Returns (passed, details)."""
    label = sys_params["label"]
    m1, m2 = sys_params["m1"], sys_params["m2"]
    S1z, S2z = sys_params["S1z"], sys_params["S2z"]
    ecc, f_lower = sys_params["ecc"], sys_params["f_lower"]

    py = python_rhs(m1, m2, S1z, S2z, ecc, f_lower)
    jx = jax_rhs(m1, m2, S1z, S2z, ecc, f_lower)

    try:
        lal_fd = lalsim_rhs_fd(m1, m2, S1z, S2z, ecc, f_lower)
        has_lalsim = True
    except Exception as exc:
        lal_fd = None
        has_lalsim = False
        lalsim_err = str(exc)

    fields = ["xdot", "edot", "ldot", "phidot"]
    results = {}
    all_pass = True

    for f in fields:
        rd_pj = rel_diff(py[f], jx[f])
        ok_pj = rd_pj <= tol_py_jax

        if has_lalsim:
            rd_pl = rel_diff(py[f], lal_fd[f])
            rd_jl = rel_diff(jx[f], lal_fd[f])
            ok_l = rd_pl <= tol_vs_lalsim and rd_jl <= tol_vs_lalsim
        else:
            rd_pl = rd_jl = None
            ok_l = True  # skip LALSim check when unavailable

        passed = ok_pj and ok_l
        if not passed:
            all_pass = False

        results[f] = {
            "py": py[f], "jax": jx[f],
            "lal": lal_fd[f] if has_lalsim else None,
            "rd_pj": rd_pj, "rd_pl": rd_pl, "rd_jl": rd_jl,
            "ok_pj": ok_pj, "ok_l": ok_l, "passed": passed,
        }

    # Build output
    lines = []
    status = "PASS" if all_pass else "FAIL"
    lines.append(f"  [{status}] {label}")
    lines.append(f"         m1={m1}, m2={m2}, S1z={S1z}, S2z={S2z}, e={ecc}, f={f_lower} Hz")
    lines.append(f"         x0 = {py['x0']:.10e}")

    if verbose or not all_pass:
        hdr = f"    {'':>8} {'Python':>16} {'JAX':>16}"
        sep = "    " + "-" * 50
        if has_lalsim:
            hdr += f" {'LALSim(FD)':>16} {'Py-JAX':>10} {'Py-LAL':>10} {'JAX-LAL':>10}"
            sep = "    " + "-" * 100
        else:
            hdr += f" {'Py-JAX':>10}"
        lines.append(hdr)
        lines.append(sep)

        for f in fields:
            r = results[f]
            mark = "  " if r["passed"] else "**"
            line = f"  {mark}{f:>6} {r['py']:>16.8e} {r['jax']:>16.8e}"
            if has_lalsim:
                line += f" {r['lal']:>16.8e} {r['rd_pj']:>10.2e} {r['rd_pl']:>10.2e} {r['rd_jl']:>10.2e}"
            else:
                line += f" {r['rd_pj']:>10.2e}"
            lines.append(line)

    if not has_lalsim:
        lines.append(f"         (LALSim not available: {lalsim_err})")

    return all_pass, "\n".join(lines), results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compare ODE RHS across LALSim C, Python-numba, and JAX.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Tolerances:
              --tol-py-jax    Python vs JAX relative tolerance (default: 1e-12).
                              These share identical source expressions, so any
                              nonzero difference is a bug.
              --tol-vs-lalsim Python/JAX vs LALSim(FD) tolerance (default: 1e-3).
                              This is looser because the LALSim values come from
                              finite differences and the expressions may use
                              algebraically equivalent but numerically different
                              forms.
        """),
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print per-field details for passing systems too.",
    )
    parser.add_argument(
        "--tol-py-jax", type=float, default=1e-12,
        help="Relative tolerance for Python vs JAX (default: 1e-12).",
    )
    parser.add_argument(
        "--tol-vs-lalsim", type=float, default=1e-3,
        help="Relative tolerance for Python/JAX vs LALSim FD (default: 1e-3).",
    )
    args = parser.parse_args()

    # Try to import LALSim — if unavailable, we still compare Python vs JAX
    lalsim_path = os.environ.get("LALSIM_PYTHON_PATH",
        "/home/prayush/local/lalsuite/esigma_github/lib/python3.13/site-packages")
    if lalsim_path not in sys.path:
        sys.path.insert(0, lalsim_path)

    has_lalsim = True
    try:
        import lalsimulation as ls
        if not hasattr(ls, "SimInspiralESIGMADynamics"):
            has_lalsim = False
            print("WARNING: lalsimulation found but SimInspiralESIGMADynamics not available.")
            print("         LALSim comparisons will be skipped.\n")
    except ImportError:
        has_lalsim = False
        print("WARNING: lalsimulation not importable. LALSim comparisons will be skipped.\n")

    print("=" * 80)
    print("ODE RHS agreement test: LALSim C vs Python-numba vs JAX")
    print("=" * 80)
    print(f"  Python-JAX tolerance: {args.tol_py_jax:.0e}")
    print(f"  vs-LALSim tolerance:  {args.tol_vs_lalsim:.0e}")
    print(f"  Systems to test:      {len(SYSTEMS)}")
    print(f"  LALSim available:     {has_lalsim}")
    print()

    n_pass = 0
    n_fail = 0
    all_results = {}

    for sys_params in SYSTEMS:
        try:
            passed, detail, results = compare_system(
                sys_params, args.tol_py_jax, args.tol_vs_lalsim, args.verbose,
            )
        except Exception as exc:
            passed = False
            detail = f"  [ERROR] {sys_params['label']}: {exc}"
            results = {}

        print(detail)
        all_results[sys_params["label"]] = results

        if passed:
            n_pass += 1
        else:
            n_fail += 1

    # Summary
    print()
    print("=" * 80)
    total = n_pass + n_fail
    if n_fail == 0:
        print(f"RESULT: ALL {total} SYSTEMS PASSED")
    else:
        print(f"RESULT: {n_fail}/{total} SYSTEMS FAILED")
    print("=" * 80)

    # Aggregate worst-case relative differences
    if all_results:
        print("\nWorst-case relative differences across all systems:")
        for f in ["xdot", "edot", "ldot", "phidot"]:
            worst_pj = max(
                (r[f]["rd_pj"] for r in all_results.values() if f in r), default=0
            )
            worst_pl = max(
                (r[f]["rd_pl"] for r in all_results.values() if f in r and r[f]["rd_pl"] is not None),
                default=None,
            )
            line = f"  {f:>6}  Py-JAX: {worst_pj:.2e}"
            if worst_pl is not None:
                line += f"   Py-LALSim: {worst_pl:.2e}"
            print(line)

    return 1 if n_fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
