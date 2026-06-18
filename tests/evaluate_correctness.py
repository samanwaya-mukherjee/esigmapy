import os
import sys

# Ensure the parent directory is in sys.path so we can import esigmapy
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import time
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import subprocess


def run_single_waveform(p):
    from esigmapy.python_codes.generator_python import get_inspiral_esigma_modes_py

    t, wfm = get_inspiral_esigma_modes_py(
        mass1=p["mass1"],
        mass2=p["mass2"],
        spin1z=p["spin1z"],
        spin2z=p["spin2z"],
        eccentricity=p["eccentricity"],
        mean_anomaly=0.0,
        distance=100.0,
        f_lower=20.0,
        delta_t=1.0 / 4096.0,
        modes_to_use=[(2, 2)],
        return_pycbc_timeseries=False,
    )
    # Just return the 22 mode
    h22 = wfm[(2, 2)]
    return {"real": h22.real.tolist(), "imag": h22.imag.tolist()}


def worker_main(params_file, out_file):
    with open(params_file, "r") as f:
        params_list = json.load(f)

    # Use 10 cores to avoid completely freezing the machine
    with Pool(processes=10) as pool:
        results = pool.map(run_single_waveform, params_list)

    with open(out_file, "w") as f:
        json.dump(results, f)


def generate_params(n=100):
    params_list = []
    np.random.seed(int(time.time()))
    for _ in range(n):
        params_list.append(
            {
                "mass1": np.random.uniform(5.0, 50.0),
                "mass2": np.random.uniform(5.0, 50.0),
                "spin1z": np.random.uniform(-0.8, 0.8),
                "spin2z": np.random.uniform(-0.8, 0.8),
                "eccentricity": np.random.uniform(0.0, 0.4),
            }
        )
    return params_list


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker_main(sys.argv[2], sys.argv[3])
        return

    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        params_file = sys.argv[2]
        opt_file = sys.argv[3]
        unopt_file = sys.argv[4]

        with open(params_file, "r") as f:
            params_list = json.load(f)
        with open(opt_file, "r") as f:
            opt_results = json.load(f)
        with open(unopt_file, "r") as f:
            unopt_results = json.load(f)

        print("Computing differences and plotting...")
        os.makedirs("figs2", exist_ok=True)
        max_diffs = []

        for i, p in enumerate(params_list):
            h22_opt = np.array(opt_results[i]["real"]) + 1j * np.array(
                opt_results[i]["imag"]
            )
            h22_unopt = np.array(unopt_results[i]["real"]) + 1j * np.array(
                unopt_results[i]["imag"]
            )

            # Ensure they are the same length (should be, but just in case)
            min_len = min(len(h22_opt), len(h22_unopt))
            h22_opt = h22_opt[:min_len]
            h22_unopt = h22_unopt[:min_len]

            diff = np.abs(h22_opt - h22_unopt)
            norm_old = np.linalg.norm(h22_unopt)
            norm_diff = np.linalg.norm(diff) / norm_old if norm_old > 0 else 0.0
            max_diffs.append(norm_diff)

            # Plotting
            plt.figure(figsize=(10, 6))

            t_arr = np.arange(min_len) * (1.0 / 4096.0)
            plt.plot(t_arr, h22_opt.real, label="Optimized (JIT)", alpha=0.7)
            plt.plot(
                t_arr,
                h22_unopt.real,
                label="Unoptimized (No-JIT)",
                linestyle="dashed",
                alpha=0.7,
            )

            plt.title(
                f"M1={p['mass1']:.2f}, M2={p['mass2']:.2f}, S1z={p['spin1z']:.2f}, S2z={p['spin2z']:.2f}, e={p['eccentricity']:.2f}\nRel. Diff Norm: {norm_diff:.2e}"
            )
            plt.xlabel("Time [s]")
            plt.ylabel("Re(h22)")
            plt.legend()

            fig_name = f"figs/m1_{p['mass1']:.1f}_m2_{p['mass2']:.1f}_s1z_{p['spin1z']:.2f}_s2z_{p['spin2z']:.2f}_e_{p['eccentricity']:.2f}.png"
            plt.savefig(fig_name, bbox_inches="tight")
            plt.close()

        print(f"Done! Evaluated 100 systems.")
        print(f"Max relative difference norm: {max(max_diffs):.2e}")
        print(f"Mean relative difference norm: {np.mean(max_diffs):.2e}")
        return

    print("Generating 100 random source parameters...")
    params_list = generate_params(100)
    with open("params_correctness.json", "w") as f:
        json.dump(params_list, f)


if __name__ == "__main__":
    main()
