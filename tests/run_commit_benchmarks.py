import argparse
import glob
import json
import subprocess
import sys
import os

ROOT_DIR = os.environ.get(
    "ESIGMAPY_ROOT_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
sys.path.insert(0, ROOT_DIR)
import shutil
import time
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt


def generate_params(n_points=20):
    params_file = os.path.join(ROOT_DIR, "speed_test_params.json")
    if not os.path.exists(params_file):
        np.random.seed(42)
        m_tot = np.random.uniform(10, 50, n_points)
        q = np.random.uniform(1, 6, n_points)
        m1 = m_tot * q / (1 + q)
        m2 = m_tot / (1 + q)
        s1z = np.random.uniform(-0.8, 0.8, n_points)
        s2z = np.random.uniform(-0.8, 0.8, n_points)
        ecc = np.random.uniform(0, 0.4, n_points)

        params_list = []
        for i in range(n_points):
            params_list.append(
                {
                    "mass1": m1[i],
                    "mass2": m2[i],
                    "spin1z": s1z[i],
                    "spin2z": s2z[i],
                    "eccentricity": ecc[i],
                }
            )

        with open(params_file, "w") as f:
            json.dump(params_list, f, indent=4)


def run_single_system_wrapper(args):
    p, integrator = args
    import time
    import numpy as np
    import lal

    kwargs = {
        "mass1": p["mass1"],
        "mass2": p["mass2"],
        "spin1z": p["spin1z"],
        "spin2z": p["spin2z"],
        "eccentricity": p["eccentricity"],
        "mean_anomaly": 0.0,
        "f_lower": 10.0,
        "delta_t": 1 / 4096.0,
        "distance": 100.0,
        "return_pycbc_timeseries": False,
    }

    if integrator == "C":
        import esigmapy.generator as c_gen
        import lalsimulation as ls

        # Warmup
        try:
            c_gen.get_inspiral_esigma_modes(**kwargs)
        except Exception as e:
            pass

        times_full = []
        times_dyn = []
        times_modes = []

        for _ in range(2):
            start = time.perf_counter()
            modes = c_gen.get_inspiral_esigma_modes(**kwargs)
            times_full.append(time.perf_counter() - start)

            start = time.perf_counter()
            retval = ls.SimInspiralESIGMADynamics(
                kwargs["mass1"],
                kwargs["mass2"],
                kwargs["spin1z"],
                kwargs["spin2z"],
                kwargs["eccentricity"],
                kwargs["f_lower"],
                kwargs["mean_anomaly"],
                1e-12,
                1.0 / kwargs["delta_t"],
            )
            times_dyn.append(time.perf_counter() - start)

            t, x, e, l, phi, phidot, r, rdot = retval[:8]
            t.data.data *= (kwargs["mass1"] + kwargs["mass2"]) * lal.MTSUN_SI
            distance_m = kwargs["distance"] * 1e6 * lal.PC_SI

            start = time.perf_counter()
            for el, em in [(2, 2), (3, 3), (4, 4), (2, -2), (3, -3), (4, -4)]:
                ls.SimInspiralESIGMAModeFromDynamics(
                    el,
                    em,
                    t.data,
                    x.data,
                    phi.data,
                    phidot.data,
                    r.data,
                    rdot.data,
                    kwargs["mass1"],
                    kwargs["mass2"],
                    kwargs["spin1z"],
                    kwargs["spin2z"],
                    distance_m,
                )
            times_modes.append(time.perf_counter() - start)
    else:
        from esigmapy.python_codes.generator_python import get_inspiral_esigma_modes_py
        from esigmapy.python_codes.esigma_pn_main import (
            inspiral_esigma_dynamics,
            inspiral_esigma_mode_from_dynamics,
        )

        # Warmup
        try:
            try:
                get_inspiral_esigma_modes_py(**kwargs, integrator=integrator)
            except TypeError:
                get_inspiral_esigma_modes_py(**kwargs)
        except Exception as e:
            pass

        times_full = []
        times_dyn = []
        times_modes = []

        for _ in range(2):
            start = time.perf_counter()
            try:
                get_inspiral_esigma_modes_py(**kwargs, integrator=integrator)
            except TypeError:
                get_inspiral_esigma_modes_py(**kwargs)
            times_full.append(time.perf_counter() - start)

            start = time.perf_counter()
            try:
                retval = inspiral_esigma_dynamics(
                    kwargs["mass1"],
                    kwargs["mass2"],
                    kwargs["spin1z"],
                    kwargs["spin2z"],
                    kwargs["eccentricity"],
                    kwargs["f_lower"],
                    kwargs["mean_anomaly"],
                    1e-12,
                    1.0 / kwargs["delta_t"],
                    integrator=integrator,
                )
            except TypeError:
                retval = inspiral_esigma_dynamics(
                    kwargs["mass1"],
                    kwargs["mass2"],
                    kwargs["spin1z"],
                    kwargs["spin2z"],
                    kwargs["eccentricity"],
                    kwargs["f_lower"],
                    kwargs["mean_anomaly"],
                    1e-12,
                    1.0 / kwargs["delta_t"],
                )
            times_dyn.append(time.perf_counter() - start)

            t_arr = np.asarray(retval["time_evol"])
            x_arr = np.asarray(retval["x_evol"])
            phi_arr = np.asarray(retval["phi_evol"])
            phidot_arr = np.asarray(retval["phi_dot_evol"])
            r_arr = np.asarray(retval["r_evol"])
            rdot_arr = np.asarray(retval["r_dot_evol"])
            distance_m = kwargs["distance"] * 1e6 * lal.PC_SI

            start = time.perf_counter()
            for el, em in [(2, 2), (3, 3), (4, 4), (2, -2), (3, -3), (4, -4)]:
                inspiral_esigma_mode_from_dynamics(
                    el,
                    em,
                    t_arr,
                    x_arr,
                    phi_arr,
                    phidot_arr,
                    r_arr,
                    rdot_arr,
                    kwargs["mass1"],
                    kwargs["mass2"],
                    kwargs["spin1z"],
                    kwargs["spin2z"],
                    distance_m,
                )
            times_modes.append(time.perf_counter() - start)

    p_copy = p.copy()
    p_copy["avg_time_full"] = sum(times_full) / len(times_full)
    p_copy["avg_time_dyn"] = sum(times_dyn) / len(times_dyn)
    p_copy["avg_time_modes"] = sum(times_modes) / len(times_modes)
    p_copy["avg_time"] = p_copy["avg_time_full"]  # Fallback for backward compat
    p_copy["M"] = p["mass1"] + p["mass2"]
    return p_copy


def params_match(p1, p2):
    return (
        abs(p1["mass1"] - p2["mass1"]) < 1e-6
        and abs(p1["mass2"] - p2["mass2"]) < 1e-6
        and abs(p1["spin1z"] - p2["spin1z"]) < 1e-6
        and abs(p1["spin2z"] - p2["spin2z"]) < 1e-6
        and abs(p1["eccentricity"] - p2["eccentricity"]) < 1e-6
    )


def worker_main():
    params_file = os.path.join(ROOT_DIR, "speed_test_params.json")
    with open(params_file, "r") as f:
        params_list = json.load(f)

    out_file = sys.argv[2]
    integrator = sys.argv[3] if len(sys.argv) > 3 else "lsoda"

    print(f"Warming up JIT for {integrator}...")
    sys.stdout.flush()
    run_single_system_wrapper((params_list[0], integrator))

    existing_results = []
    if os.path.exists(out_file):
        with open(out_file, "r") as f:
            existing_results = json.load(f)

    pending_params = []
    for p in params_list:
        if not any(params_match(p, e) for e in existing_results):
            pending_params.append((p, integrator))

    if len(pending_params) > 0:
        print(f"Running benchmarks over {len(pending_params)} systems in parallel...")
        sys.stdout.flush()

        with multiprocessing.Pool() as pool:
            results = pool.map(run_single_system_wrapper, pending_params)

        existing_results.extend(results)

        with open(out_file, "w") as f:
            json.dump(existing_results, f, indent=4)
    else:
        print("All parameters already calculated. Skipping.")


def plot_results(commits_data_map, labels):
    import matplotlib.cm as cm

    colors = cm.tab10(np.linspace(0, 1, len(commits_data_map)))
    mass_bins = np.linspace(10, 50, 10)
    bin_centers = (mass_bins[:-1] + mass_bins[1:]) / 2

    base_key = list(commits_data_map.keys())[0]
    base_data = commits_data_map[base_key]

    for metric in ["full", "dyn", "modes"]:
        # Generation times plot
        plt.figure(figsize=(10, 6))
        for i, (key, c_data) in enumerate(commits_data_map.items()):
            bin_means = []
            bin_stds = []
            for j in range(len(mass_bins) - 1):
                low, high = mass_bins[j], mass_bins[j + 1]
                times = [
                    r.get(f"avg_time_{metric}", r.get("avg_time", np.nan))
                    for r in c_data
                    if low <= r["M"] <= high
                ]
                times = [t for t in times if not np.isnan(t)]
                if times:
                    bin_means.append(np.mean(times))
                    bin_stds.append(np.std(times))
                else:
                    bin_means.append(np.nan)
                    bin_stds.append(np.nan)

            if not all(np.isnan(bin_means)):
                plt.errorbar(
                    bin_centers,
                    bin_means,
                    yerr=bin_stds,
                    label=labels[key],
                    marker="o",
                    capsize=5,
                    color=colors[i],
                )

        plt.yscale("log")
        plt.xlabel("Total Mass $M_\\odot$")
        plt.ylabel("Average Generation Time (s)")
        plt.title(f"Waveform Generation Time vs Total Mass ({metric})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            os.path.join(ROOT_DIR, f"figs2/generation_times_vs_mass_{metric}.png")
        )

        # Plot speedups
        plt.figure(figsize=(10, 6))
        for i, (key, c_data) in enumerate(commits_data_map.items()):
            if key == base_key:
                continue

            system_speedups = []
            for r_opt in c_data:
                for r_base in base_data:
                    if params_match(r_opt, r_base):
                        t_opt = r_opt.get(
                            f"avg_time_{metric}", r_opt.get("avg_time", np.nan)
                        )
                        t_base = r_base.get(
                            f"avg_time_{metric}", r_base.get("avg_time", np.nan)
                        )
                        if not np.isnan(t_opt) and not np.isnan(t_base) and t_opt > 0:
                            system_speedups.append(
                                {"M": r_opt["M"], "speedup": t_base / t_opt}
                            )
                        break

            bin_means = []
            bin_stds = []
            for j in range(len(mass_bins) - 1):
                low, high = mass_bins[j], mass_bins[j + 1]
                spds = [r["speedup"] for r in system_speedups if low <= r["M"] <= high]
                if spds:
                    bin_means.append(np.mean(spds))
                    bin_stds.append(np.std(spds))
                else:
                    bin_means.append(np.nan)
                    bin_stds.append(np.nan)

            if not all(np.isnan(bin_means)):
                plt.errorbar(
                    bin_centers,
                    bin_means,
                    yerr=bin_stds,
                    label=f"{labels[key]} over {labels[base_key]}",
                    marker="o",
                    capsize=5,
                    color=colors[i],
                )

        plt.yscale("log")
        plt.xlabel("Total Mass $M_\\odot$")
        plt.ylabel("Speedup factor")
        plt.title(f"Waveform Generation Speedup vs Total Mass ({metric})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            os.path.join(ROOT_DIR, f"figs2/generation_speedups_vs_mass_{metric}.png")
        )


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker_main()
        return

    parser = argparse.ArgumentParser(
        description="Benchmark waveform generation across commits and integrators."
    )
    parser.add_argument(
        "commits",
        nargs="*",
        default=[
            "93c92f1178b85b5f9b00d6f62e9914cf4e052ed8:lsoda",
            "2859aabab918b331fdcf3359990427b45a38d1ec:lsoda",
            "HEAD:dop853",
            "HEAD:lsoda",
            "HEAD:C",
        ],
        help="List of <commit>:<integrator> to benchmark",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only generate figures from existing results data, without running any profiling tests.",
    )

    args = parser.parse_args()

    os.makedirs(os.path.join(ROOT_DIR, "figs2"), exist_ok=True)
    generate_params(20)

    parsed_configs = []
    for c in args.commits:
        if ":" in c:
            commit, integrator = c.split(":", 1)
        else:
            commit, integrator = c, "lsoda"
        parsed_configs.append((commit, integrator))

    current_branch = (
        subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        .decode()
        .strip()
    )
    if current_branch == "HEAD":
        current_branch = (
            subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        )

    shutil.copy(sys.argv[0], "/tmp/run_worker.py")

    if not args.plot_only:
        try:
            for commit, integrator in parsed_configs:
                print(f"\n===========================================")
                print(f"Testing commit {commit} with {integrator}...")
                print(f"===========================================\n")
                subprocess.run(["git", "checkout", commit], check=True)

                env = os.environ.copy()
                env["PYTHONPATH"] = ROOT_DIR
                env["ESIGMAPY_ROOT_DIR"] = ROOT_DIR
                subprocess.run(
                    [
                        "conda",
                        "run",
                        "--no-capture-output",
                        "-n",
                        "lalsuite-dev",
                        "python",
                        "-B",
                        "/tmp/run_worker.py",
                        "--worker",
                        os.path.join(
                            ROOT_DIR, f"speed_test_results_{commit}_{integrator}.json"
                        ),
                        integrator,
                    ],
                    check=True,
                    env=env,
                )
        finally:
            print("\nRestoring original branch...")
            subprocess.run(["git", "checkout", current_branch], check=True)

    print("Plotting results...")

    # Load all available results matching the pattern
    commits_data_map = {}
    labels = {}

    for config in parsed_configs:
        commit, integrator = config
        fpath = os.path.join(ROOT_DIR, f"speed_test_results_{commit}_{integrator}.json")
        if os.path.exists(fpath):
            with open(fpath, "r") as f:
                commits_data_map[f"{commit}:{integrator}"] = json.load(f)

            try:
                subject = (
                    subprocess.check_output(["git", "log", "-1", "--format=%s", commit])
                    .decode()
                    .strip()
                )
                if len(subject) > 30:
                    subject = subject[:27] + "..."
                labels[f"{commit}:{integrator}"] = (
                    f"{commit[:7]} ({subject}) [{integrator}]"
                )
            except:
                labels[f"{commit}:{integrator}"] = f"{commit[:7]} [{integrator}]"

    # Also grab any other jsons in the directory that match the format
    for fpath in glob.glob("speed_test_results_*_*.json"):
        base = (
            os.path.basename(fpath)
            .replace("speed_test_results_", "")
            .replace(".json", "")
        )
        if "_" in base:
            # Reconstruct commit and integrator
            commit = base.split("_")[0]
            integrator = base.split("_")[1]
            key = f"{commit}:{integrator}"

            if key not in commits_data_map:
                with open(fpath, "r") as f:
                    commits_data_map[key] = json.load(f)
                try:
                    subject = (
                        subprocess.check_output(
                            ["git", "log", "-1", "--format=%s", commit]
                        )
                        .decode()
                        .strip()
                    )
                    if len(subject) > 30:
                        subject = subject[:27] + "..."
                    labels[key] = f"{commit[:7]} ({subject}) [{integrator}]"
                except:
                    labels[key] = f"{commit[:7]} [{integrator}]"

    plot_results(commits_data_map, labels)
    print("Done! Plots saved in figs2/")


if __name__ == "__main__":
    main()
