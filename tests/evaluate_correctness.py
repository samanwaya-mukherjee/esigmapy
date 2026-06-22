#!/usr/bin/env python3
"""
evaluate_correctness.py
=======================
Comprehensive correctness evaluation of ESIGMA backends against the LALSim C
reference implementation.

Tests:
  1. ODE RHS comparison at initial state (200 random systems)
  2. Full dynamics evolution comparison (20 systems)
  3. GW mode + waveform comparison (20 systems)

All comparisons use the C backend as the reference. Results and figures are
written to correctness_results/.

Usage:
    conda run -n lalsuite-dev python tests/evaluate_correctness.py
    conda run -n lalsuite-dev python tests/evaluate_correctness.py --n-rhs 500 --n-evol 50
"""
import os
import sys
import argparse
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "correctness_results"
)


# ---------------------------------------------------------------------------
# Parameter generation
# ---------------------------------------------------------------------------

def generate_params(n, seed=42):
    """Generate n random binary parameter sets."""
    rng = np.random.default_rng(seed)
    params = []
    for _ in range(n):
        m_tot = rng.uniform(10, 80)
        q = rng.uniform(1, 8)
        m1 = m_tot * q / (1 + q)
        m2 = m_tot / (1 + q)
        params.append({
            "mass1": m1, "mass2": m2,
            "spin1z": rng.uniform(-0.8, 0.8),
            "spin2z": rng.uniform(-0.8, 0.8),
            "eccentricity": rng.uniform(0.0, 0.4),
            "f_lower": rng.choice([10.0, 15.0, 20.0]),
        })
    return params


# ---------------------------------------------------------------------------
# Test 1: ODE RHS comparison
# ---------------------------------------------------------------------------

def compare_ode_rhs(params_list, ode_eps=1e-12):
    """Compare ODE RHS at initial state: Python/JAX vs LALSim (fine FD)."""
    import lal
    import lalsimulation as ls
    from esigmapy.inspiral.numba_backend.pn_inspiral import (
        eccentric_x_model_odes, x_dot_4pn_SF,
    )
    from esigmapy.inspiral.jax_backend.inspiral import eccentric_x_model_odes_jax
    import jax.numpy as jnp

    results = []
    dt_fine_sec = 1e-5  # coarser than 1e-7 to avoid OOM; still gives ~1e-4 FD accuracy

    for i, p in enumerate(params_list):
        m1, m2 = p["mass1"], p["mass2"]
        S1z, S2z = p["spin1z"], p["spin2z"]
        ecc, f_lower = p["eccentricity"], p["f_lower"]
        total_mass = m1 + m2
        eta = (m1 * m2) / total_mass**2
        omega_init = lal.PI * f_lower * lal.MTSUN_SI
        x0 = (total_mass * omega_init) ** (2.0 / 3.0)

        # Python RHS
        x_dot_4pn_SF_val = x_dot_4pn_SF(ecc, eta, S1z)
        y0 = np.array([x0, ecc, 0.0, 0.0])
        dydt_py = eccentric_x_model_odes(0.0, y0, eta, m1, m2, S1z, S2z, 8, x_dot_4pn_SF_val)

        # JAX RHS
        args_jax = (eta, m1, m2, S1z, S2z, 8, 8)
        y0_jax = jnp.array([x0, ecc, 0.0, 0.0])
        dydt_jax = eccentric_x_model_odes_jax(0.0, y0_jax, args_jax)

        # LALSim fine FD — only keep first 2 steps for derivative, then free
        try:
            retval = ls.SimInspiralESIGMADynamics(
                m1, m2, S1z, S2z, ecc, f_lower, 0.0, ode_eps, 1.0 / dt_fine_sec
            )
            dt_geom = retval[0].data.data[1] - retval[0].data.data[0]
            dydt_lal = np.array([
                (retval[1].data.data[1] - retval[1].data.data[0]) / dt_geom,
                (retval[2].data.data[1] - retval[2].data.data[0]) / dt_geom,
                (retval[3].data.data[1] - retval[3].data.data[0]) / dt_geom,
                (retval[4].data.data[1] - retval[4].data.data[0]) / dt_geom,
            ])
            del retval
            has_lal = True
        except Exception:
            has_lal = False
            dydt_lal = np.zeros(4)

        def reldiff(a, b):
            d = max(abs(a), abs(b), 1e-30)
            return abs(a - b) / d

        r = {"params": p, "has_lal": has_lal}
        for j, name in enumerate(["xdot", "edot", "ldot", "phidot"]):
            r[f"py_{name}"] = float(dydt_py[j])
            r[f"jax_{name}"] = float(dydt_jax[j])
            r[f"lal_{name}"] = float(dydt_lal[j])
            r[f"rd_py_jax_{name}"] = reldiff(dydt_py[j], float(dydt_jax[j]))
            if has_lal:
                r[f"rd_py_lal_{name}"] = reldiff(dydt_py[j], dydt_lal[j])
        results.append(r)

        if (i + 1) % 50 == 0:
            print(f"  ODE RHS: {i+1}/{len(params_list)} done")

    return results


# ---------------------------------------------------------------------------
# Test 2: Full dynamics evolution comparison
# ---------------------------------------------------------------------------

def compare_dynamics(params_list, ode_eps=1e-12):
    """Compare full orbital dynamics (x, e, phi) across backends."""
    from esigmapy.inspiral import get_dynamics

    results = []
    for i, p in enumerate(params_list):
        kw = dict(
            spin1z=p["spin1z"], spin2z=p["spin2z"],
            eccentricity=p["eccentricity"], mean_anomaly=0.0,
            ode_eps=ode_eps,
        )

        try:
            dyn_lal = get_dynamics(p["mass1"], p["mass2"], p["f_lower"], 1/4096.0,
                                   backend="lalsim", **kw)
        except Exception as e:
            results.append({"params": p, "error": f"lalsim: {e}"})
            continue

        try:
            dyn_numba = get_dynamics(p["mass1"], p["mass2"], p["f_lower"], 1/4096.0,
                                     backend="numba", integrator="dop853", **kw)
        except Exception as e:
            results.append({"params": p, "error": f"numba: {e}"})
            continue

        N = min(len(dyn_lal["phi_evol"]), len(dyn_numba["phi_evol"]))
        N_cmp = int(0.9 * N)

        phi_diff = np.abs(dyn_lal["phi_evol"][:N_cmp] - dyn_numba["phi_evol"][:N_cmp])
        x_diff = np.abs(dyn_lal["x_evol"][:N_cmp] - dyn_numba["x_evol"][:N_cmp])

        results.append({
            "params": p,
            "N_lal": len(dyn_lal["phi_evol"]),
            "N_numba": len(dyn_numba["phi_evol"]),
            "N_cmp": N_cmp,
            "max_phi_diff_rad": float(np.max(phi_diff)),
            "max_gw_phase_diff_rad": float(2 * np.max(phi_diff)),
            "max_x_rel_diff": float(np.max(x_diff) / np.max(dyn_lal["x_evol"][:N_cmp])),
            "phi_lal_final": float(dyn_lal["phi_evol"][N_cmp - 1]),
        })

        print(f"  Dynamics: {i+1}/{len(params_list)} M={p['mass1']+p['mass2']:.1f} "
              f"GW_phase_diff={2*np.max(phi_diff):.4f} rad")

    return results


# ---------------------------------------------------------------------------
# Test 3: Mode + waveform comparison
# ---------------------------------------------------------------------------

def compare_modes(params_list, ode_eps=1e-12):
    """Compare GW modes across backends."""
    from esigmapy.inspiral import get_modes

    results = []
    for i, p in enumerate(params_list):
        kw = dict(
            spin1z=p["spin1z"], spin2z=p["spin2z"],
            eccentricity=p["eccentricity"], mean_anomaly=0.0,
            distance=100.0, ode_eps=ode_eps,
            modes_to_use=[(2, 2), (3, 3), (4, 4)],
        )

        try:
            modes_lal = get_modes(p["mass1"], p["mass2"], p["f_lower"], 1/4096.0,
                                   backend="lalsim", **kw)
        except Exception as e:
            results.append({"params": p, "error": f"lalsim: {e}"})
            continue

        try:
            modes_numba = get_modes(p["mass1"], p["mass2"], p["f_lower"], 1/4096.0,
                                     backend="numba", integrator="dop853", **kw)
        except Exception as e:
            results.append({"params": p, "error": f"numba: {e}"})
            continue

        try:
            modes_hybrid = get_modes(p["mass1"], p["mass2"], p["f_lower"], 1/4096.0,
                                      backend="numba:jax", **kw)
        except Exception as e:
            modes_hybrid = None

        r = {"params": p, "modes": {}}
        for lm in [(2, 2), (3, 3), (4, 4)]:
            if lm not in modes_lal or lm not in modes_numba:
                continue
            h_lal = modes_lal[lm]
            h_numba = modes_numba[lm]
            N = min(len(h_lal), len(h_numba))
            N_cmp = int(0.9 * N)
            max_abs = max(np.max(np.abs(h_lal[:N_cmp])), 1e-30)
            rd_numba = float(np.max(np.abs(h_lal[:N_cmp] - h_numba[:N_cmp])) / max_abs)

            rd_hybrid = None
            if modes_hybrid and lm in modes_hybrid:
                h_hyb = modes_hybrid[lm]
                N_h = min(len(h_lal), len(h_hyb))
                N_h_cmp = int(0.9 * N_h)
                rd_hybrid = float(np.max(np.abs(h_lal[:N_h_cmp] - h_hyb[:N_h_cmp])) / max_abs)

            r["modes"][f"{lm[0]},{lm[1]}"] = {
                "rd_numba_vs_lal": rd_numba,
                "rd_hybrid_vs_lal": rd_hybrid,
                "N": N,
            }

        results.append(r)
        mode_str = " ".join(f"({k})={v['rd_numba_vs_lal']:.2e}"
                           for k, v in r["modes"].items())
        print(f"  Modes: {i+1}/{len(params_list)} M={p['mass1']+p['mass2']:.1f} {mode_str}")

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_rhs_results(rhs_results):
    """Plot ODE RHS relative differences."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Py vs JAX
    for name, color in [("xdot", "C0"), ("edot", "C1"), ("ldot", "C2"), ("phidot", "C3")]:
        vals = [r[f"rd_py_jax_{name}"] for r in rhs_results]
        axes[0].hist(np.log10(np.array(vals) + 1e-16), bins=30, alpha=0.5, label=name, color=color)
    axes[0].set_xlabel("log10(relative difference)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Python vs JAX (should be ~machine precision)")
    axes[0].legend()

    # Py vs LALSim
    for name, color in [("xdot", "C0"), ("edot", "C1"), ("ldot", "C2"), ("phidot", "C3")]:
        vals = [r[f"rd_py_lal_{name}"] for r in rhs_results if r["has_lal"]]
        if vals:
            axes[1].hist(np.log10(np.array(vals) + 1e-16), bins=30, alpha=0.5, label=name, color=color)
    axes[1].set_xlabel("log10(relative difference)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Python vs LALSim C (FD estimate)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "ode_rhs_comparison.png"), dpi=150)
    plt.close()


def plot_dynamics_results(dyn_results):
    """Plot dynamics comparison results."""
    valid = [r for r in dyn_results if "error" not in r]
    if not valid:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    masses = [r["params"]["mass1"] + r["params"]["mass2"] for r in valid]
    gw_diffs = [r["max_gw_phase_diff_rad"] for r in valid]
    eccs = [r["params"]["eccentricity"] for r in valid]

    sc = axes[0].scatter(masses, gw_diffs, c=eccs, cmap="viridis", s=40)
    axes[0].set_xlabel("Total mass (Msun)")
    axes[0].set_ylabel("Max GW phase difference (rad)")
    axes[0].set_title("numba(dop853) vs LALSim C dynamics")
    plt.colorbar(sc, ax=axes[0], label="eccentricity")
    axes[0].axhline(0.1, color="r", ls="--", lw=0.8, label="0.1 rad")
    axes[0].legend()

    x_diffs = [r["max_x_rel_diff"] for r in valid]
    axes[1].scatter(masses, x_diffs, c=eccs, cmap="viridis", s=40)
    axes[1].set_xlabel("Total mass (Msun)")
    axes[1].set_ylabel("Max |x| relative difference")
    axes[1].set_title("x(t) relative difference")
    axes[1].set_yscale("log")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "dynamics_comparison.png"), dpi=150)
    plt.close()


def plot_modes_results(modes_results):
    """Plot mode comparison results."""
    valid = [r for r in modes_results if "error" not in r]
    if not valid:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    masses = [r["params"]["mass1"] + r["params"]["mass2"] for r in valid]

    for lm_key, marker in [("2,2", "o"), ("3,3", "s"), ("4,4", "^")]:
        rd_numba = []
        rd_hybrid = []
        m_vals = []
        for r, m in zip(valid, masses):
            if lm_key in r["modes"]:
                rd_numba.append(r["modes"][lm_key]["rd_numba_vs_lal"])
                rd_hybrid.append(r["modes"][lm_key]["rd_hybrid_vs_lal"])
                m_vals.append(m)
        if rd_numba:
            ax.scatter(m_vals, rd_numba, marker=marker, s=40, alpha=0.7,
                      label=f"({lm_key}) numba vs C")
            if any(v is not None for v in rd_hybrid):
                rd_h = [v for v in rd_hybrid if v is not None]
                m_h = [m for m, v in zip(m_vals, rd_hybrid) if v is not None]
                ax.scatter(m_h, rd_h, marker=marker, s=20, alpha=0.5, facecolors="none",
                          edgecolors="red", label=f"({lm_key}) hybrid vs C")

    ax.set_xlabel("Total mass (Msun)")
    ax.set_ylabel("Max relative mode difference")
    ax.set_title("GW modes: numba/hybrid vs LALSim C")
    ax.set_yscale("log")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "modes_comparison.png"), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def write_summary(rhs_results, dyn_results, modes_results):
    """Write a text summary."""
    lines = ["ESIGMA Correctness Evaluation Summary", "=" * 50, ""]

    # RHS
    py_jax_max = {name: max(r[f"rd_py_jax_{name}"] for r in rhs_results)
                  for name in ["xdot", "edot", "ldot", "phidot"]}
    py_lal_max = {name: max((r[f"rd_py_lal_{name}"] for r in rhs_results if r["has_lal"]), default=0)
                  for name in ["xdot", "edot", "ldot", "phidot"]}

    lines.append(f"1. ODE RHS comparison ({len(rhs_results)} systems)")
    lines.append(f"   Python vs JAX (worst-case relative diff):")
    for name in ["xdot", "edot", "ldot", "phidot"]:
        lines.append(f"     {name}: {py_jax_max[name]:.2e}")
    lines.append(f"   Python vs LALSim C (worst-case relative diff):")
    for name in ["xdot", "edot", "ldot", "phidot"]:
        lines.append(f"     {name}: {py_lal_max[name]:.2e}")
    lines.append("")

    # Dynamics
    valid_dyn = [r for r in dyn_results if "error" not in r]
    if valid_dyn:
        gw_diffs = [r["max_gw_phase_diff_rad"] for r in valid_dyn]
        lines.append(f"2. Full dynamics comparison ({len(valid_dyn)} systems)")
        lines.append(f"   GW phase diff (numba-dop853 vs C):")
        lines.append(f"     mean: {np.mean(gw_diffs):.4f} rad")
        lines.append(f"     max:  {np.max(gw_diffs):.4f} rad")
        lines.append(f"     median: {np.median(gw_diffs):.4f} rad")
    lines.append("")

    # Modes
    valid_modes = [r for r in modes_results if "error" not in r]
    if valid_modes:
        lines.append(f"3. GW mode comparison ({len(valid_modes)} systems)")
        for lm_key in ["2,2", "3,3", "4,4"]:
            rd_vals = [r["modes"][lm_key]["rd_numba_vs_lal"]
                      for r in valid_modes if lm_key in r["modes"]]
            if rd_vals:
                lines.append(f"   ({lm_key}) numba vs C: mean={np.mean(rd_vals):.4e} max={np.max(rd_vals):.4e}")
            rd_hyb = [r["modes"][lm_key]["rd_hybrid_vs_lal"]
                     for r in valid_modes if lm_key in r["modes"] and r["modes"][lm_key]["rd_hybrid_vs_lal"] is not None]
            if rd_hyb:
                lines.append(f"   ({lm_key}) hybrid vs C: mean={np.mean(rd_hyb):.4e} max={np.max(rd_hyb):.4e}")

    summary = "\n".join(lines)
    with open(os.path.join(RESULTS_DIR, "summary.txt"), "w") as f:
        f.write(summary)
    print()
    print(summary)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate ESIGMA correctness across backends.")
    parser.add_argument("--n-rhs", type=int, default=200, help="Number of systems for ODE RHS test")
    parser.add_argument("--n-evol", type=int, default=20, help="Number of systems for dynamics/modes test")
    parser.add_argument("--ode-eps", type=float, default=1e-12, help="ODE tolerance for all backends")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Ensure LALSim is available
    lalsim_path = os.environ.get("LALSIM_PYTHON_PATH",
        "/home/prayush/local/lalsuite/esigma_github/lib/python3.13/site-packages")
    if lalsim_path not in sys.path:
        sys.path.insert(0, lalsim_path)

    print("=" * 60)
    print("ESIGMA Correctness Evaluation")
    print("=" * 60)
    print(f"  ODE RHS systems: {args.n_rhs}")
    print(f"  Dynamics/modes systems: {args.n_evol}")
    print(f"  ODE tolerance: {args.ode_eps}")
    print(f"  Output: {RESULTS_DIR}/")
    print()

    # Generate parameters
    rhs_params = generate_params(args.n_rhs, seed=args.seed)
    evol_params = generate_params(args.n_evol, seed=args.seed + 1000)

    # Test 1: ODE RHS
    print("--- Test 1: ODE RHS comparison ---")
    t0 = time.perf_counter()
    rhs_results = compare_ode_rhs(rhs_params, ode_eps=args.ode_eps)
    print(f"  Done in {time.perf_counter()-t0:.1f}s")
    with open(os.path.join(RESULTS_DIR, "rhs_results.json"), "w") as f:
        json.dump(rhs_results, f, indent=2)

    # Test 2: Dynamics
    print("\n--- Test 2: Full dynamics comparison ---")
    t0 = time.perf_counter()
    dyn_results = compare_dynamics(evol_params, ode_eps=args.ode_eps)
    print(f"  Done in {time.perf_counter()-t0:.1f}s")
    with open(os.path.join(RESULTS_DIR, "dynamics_results.json"), "w") as f:
        json.dump(dyn_results, f, indent=2)

    # Test 3: Modes
    print("\n--- Test 3: GW mode comparison ---")
    t0 = time.perf_counter()
    modes_results = compare_modes(evol_params, ode_eps=args.ode_eps)
    print(f"  Done in {time.perf_counter()-t0:.1f}s")
    with open(os.path.join(RESULTS_DIR, "modes_results.json"), "w") as f:
        json.dump(modes_results, f, indent=2)

    # Plots
    print("\n--- Generating figures ---")
    plot_rhs_results(rhs_results)
    plot_dynamics_results(dyn_results)
    plot_modes_results(modes_results)

    # Summary
    write_summary(rhs_results, dyn_results, modes_results)

    print(f"\nAll results written to {RESULTS_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
