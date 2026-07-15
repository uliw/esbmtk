from __future__ import annotations

import gc
import os
from copy import deepcopy

import numpy as np
import numpy.typing as npt
import pandas as pd

from esbmtk import Model, SpeciesProperties

NDArrayFloat = npt.NDArray[np.float64]

# if tp.TYPE_CHECKING:
#    from esbmtk import Model, SpeciesProperties

# the below should be generalized such that they can be used for other models also

def extract_diagnostics(M):
    """Extract final model diagnostics for experiment logging."""

    diag = {}

    # ---------------- Atmosphere ----------------
    diag["CO2_ppm"] = round(M.CO2_At.c[-1]*1e6, 1)

    # ---------------- Deep ocean carbonate chemistry ----------------
    diag["A_zsat"] = round(M.A_db.zsat.c[-1], 0)
    diag["I_zsat"] = round(M.I_db.zsat.c[-1], 0)
    diag["P_zsat"] = round(M.P_db.zsat.c[-1], 0)

    diag["A_zcc"] = round(M.A_db.zcc.c[-1], 0)
    diag["I_zcc"] = round(M.I_db.zcc.c[-1], 0)
    diag["P_zcc"] = round(M.P_db.zcc.c[-1], 0)

    # ---------------- Deep ocean carbonate ----------------
    diag["A_deep_CO3"] = round(M.A_db.CO3.c[-1] * 1e6, 2)
    diag["I_deep_CO3"] = round(M.I_db.CO3.c[-1] * 1e6, 2)
    diag["P_deep_CO3"] = round(M.P_db.CO3.c[-1] * 1e6, 2)

    # ---------------- Deep ocean carbon ----------------
    diag["A_deep_DIC"] = round(M.A_db.DIC.c[-1] * 1e6, 1)
    diag["I_deep_DIC"] = round(M.I_db.DIC.c[-1] * 1e6, 1)
    diag["P_deep_DIC"] = round(M.P_db.DIC.c[-1] * 1e6, 1)

    diag["A_deep_TA"] = round(M.A_db.TA.c[-1] * 1e6, 1)
    diag["I_deep_TA"] = round(M.I_db.TA.c[-1] * 1e6, 1)
    diag["P_deep_TA"] = round(M.P_db.TA.c[-1] * 1e6, 1)

    diag["A_deep_O2"] = round(M.A_db.O2.c[-1] * 1e6, 1)
    diag["I_deep_O2"] = round(M.I_db.O2.c[-1] * 1e6, 1)
    diag["P_deep_O2"] = round(M.P_db.O2.c[-1] * 1e6, 1)

    # ---------------- High latitude surface box ----------------
    diag["H_DIC"] = round(M.H_sb.DIC.c[-1] * 1e6, 1)
    diag["H_TA"] = round(M.H_sb.TA.c[-1] * 1e6, 1)
    diag["H_O2"] = round(M.H_sb.O2.c[-1] * 1e6, 1)

    return diag

def log_experiment(M, experiment_name, params):

    diagnostics = extract_diagnostics(M)

    row = {"experiment": experiment_name}

    # record modified parameters
    row.update(params)

    # record diagnostics
    row.update(diagnostics)

    file = "results_summary.csv"

    if os.path.exists(file):
        df = pd.read_csv(file)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(file, index=False)

    print("Experiment logged:", experiment_name)



def log_full_timeseries(M, filename: str):
    """
    Extract the full time series of key model variables and save to CSV.
    
    :param M: Model instance
    :param filename: Path to CSV file to write
    """
    # Prepare dictionary of time series
    ts_dict = {
        "time": M.time,  # model time vector
        # Atmosphere
        "CO2_ppm": M.CO2_At.c * 1e6,
        # Deep ocean carbonate chemistry
        "A_zsat": M.A_db.zsat.c,
        "I_zsat": M.I_db.zsat.c,
        "P_zsat": M.P_db.zsat.c,
        "A_zcc": M.A_db.zcc.c,
        "I_zcc": M.I_db.zcc.c,
        "P_zcc": M.P_db.zcc.c,
        # Deep ocean carbonate
        "A_deep_CO3": M.A_db.CO3.c * 1e6,
        "I_deep_CO3": M.I_db.CO3.c * 1e6,
        "P_deep_CO3": M.P_db.CO3.c * 1e6,
        # Deep ocean carbon
        "A_deep_DIC": M.A_db.DIC.c * 1e6,
        "I_deep_DIC": M.I_db.DIC.c * 1e6,
        "P_deep_DIC": M.P_db.DIC.c * 1e6,
        "A_deep_TA": M.A_db.TA.c * 1e6,
        "I_deep_TA": M.I_db.TA.c * 1e6,
        "P_deep_TA": M.P_db.TA.c * 1e6,
        # High latitude surface box
        "H_DIC": M.H_sb.DIC.c * 1e6,
        "H_TA": M.H_sb.TA.c * 1e6,
    }

    # Convert to DataFrame
    df = pd.DataFrame(ts_dict)

    # Round for readability
    df = df.round({
        "CO2_ppm": 1,
        "A_zsat": 0, "I_zsat": 0, "P_zsat": 0,
        "A_zcc": 0, "I_zcc": 0, "P_zcc": 0,
        "A_deep_CO3": 2, "I_deep_CO3": 2, "P_deep_CO3": 2,
        "A_deep_DIC": 1, "I_deep_DIC": 1, "P_deep_DIC": 1,
        "A_deep_TA": 1, "I_deep_TA": 1, "P_deep_TA": 1,
        "H_DIC": 1, "H_TA": 1,
    })

    # Write to CSV
    df.to_csv(filename, index=False)
    print(f"Full time series logged to {filename}")



def log_experiment_timeseries(M, experiment_name: str, params: dict, filename: str = "results_timeseries.csv"):
    """
    Log the full time series of an experiment, including parameters.
    
    :param M: Model instance
    :param experiment_name: Name of the experiment
    :param params: Dictionary of experiment parameters
    :param filename: CSV file to write
    """
    # ---------------- Build time series dictionary ----------------
    ts_dict = {
        "experiment": [experiment_name] * len(M.time),
        "time": M.time,  # model time vector
        # Atmosphere
        "CO2_ppm": M.CO2_At.c * 1e6,
        # Deep ocean carbonate chemistry
        "A_zsat": M.A_db.zsat.c,
        "I_zsat": M.I_db.zsat.c,
        "P_zsat": M.P_db.zsat.c,
        "A_zcc": M.A_db.zcc.c,
        "I_zcc": M.I_db.zcc.c,
        "P_zcc": M.P_db.zcc.c,
        # Deep ocean carbonate
        "A_deep_CO3": M.A_db.CO3.c * 1e6,
        "I_deep_CO3": M.I_db.CO3.c * 1e6,
        "P_deep_CO3": M.P_db.CO3.c * 1e6,
        # Deep ocean carbon
        "A_deep_DIC": M.A_db.DIC.c * 1e6,
        "I_deep_DIC": M.I_db.DIC.c * 1e6,
        "P_deep_DIC": M.P_db.DIC.c * 1e6,
        "A_deep_TA": M.A_db.TA.c * 1e6,
        "I_deep_TA": M.I_db.TA.c * 1e6,
        "P_deep_TA": M.P_db.TA.c * 1e6,
        # High latitude surface box
        "H_DIC": M.H_sb.DIC.c * 1e6,
        "H_TA": M.H_sb.TA.c * 1e6,
    }

    # Add parameter values as constant columns
    for key, val in params.items():
        ts_dict[key] = [val] * len(M.time)

    # Convert to DataFrame
    df_ts = pd.DataFrame(ts_dict)

    # ---------------- Append or create CSV ----------------
    if os.path.exists(filename):
        df_existing = pd.read_csv(filename)
        df_ts = pd.concat([df_existing, df_ts], ignore_index=True)

    df_ts.to_csv(filename, index=False)
    print(f"Experiment time series logged: {experiment_name} -> {filename}")


def sensitivity_test(
    param_name: str,
    values,
    base_params: dict,
    run_time="100 kyr",
    time_step="100 yr",
    rain_ratio=6.1,
    alpha=0.3,
    debug=False,
    ocean_names=["A", "I", "P"],
    experiment_prefix="sens"
):
    """
    Sensitivity analysis extracting key diagnostics:
    - CO2
    - zcc (all basins)
    - CO3 (all basins)
    - deep O2 (all basins)

    If a particular parameter run fails, NaNs are returned for that value.
    """

    from LOSCAR_GLACIAL import initialize_model, pp_carbonate_cs4

    results = []

    for val in values:
        print(f"\nRunning: {param_name} = {val}")

        # --- clean memory ---
        try:
            del M
        except NameError:
            pass
        gc.collect()

        # --- copy + modify parameters ---
        params = deepcopy(base_params)

        # --- special case: simultaneous mixing scaling ---
        if param_name == "mix_all":
            # Extract modern values (assumed strings like "4 Sv")
            def to_float(x):
                return float(str(x).replace("Sv","").strip())

            mix_A = to_float(base_params["mix_A_H"])
            mix_I = to_float(base_params["mix_I_H"])
            mix_P = to_float(base_params["mix_P_H"])

            # Apply scaling
            params["mix_A_H"] = f"{mix_A * val} Sv"
            params["mix_I_H"] = f"{mix_I * val} Sv"
            params["mix_P_H"] = f"{mix_P * val} Sv"

        else:
            if param_name not in params:
                raise ValueError(f"{param_name} not in base_params")
            params[param_name] = val

        try:
            # --- initialize model ---
            M = initialize_model(
                high_lat_piston=params["high_lat_piston"],
                high_lat_PO4_export=params["high_lat_PO4_export"],
                T_surf=params["T_surf"],
                T_deep=params["T_deep"],
                thc=params["thc"],
                ta=params["ta"],
                ti=params["ti"],
                mix_A_H=params["mix_A_H"],
                mix_I_H=params["mix_I_H"],
                mix_P_H=params["mix_P_H"],
                rain_ratio=rain_ratio,
                alpha=alpha,
                run_time=run_time,
                time_step=time_step,
                debug=debug,
            )

            # --- load spun-up state ---
            M.read_state("modern_state.pkl")

            # --- run model ---
            M.run()

            # --- carbonate post-processing ---
            pp_carbonate_cs4(M, ocean_names)

            # --- extract diagnostics ---
            diag = extract_diagnostics(M)

            # --- log full experiment ---
            experiment_name = f"{experiment_prefix}_{param_name}_{val}"
            try:
                log_experiment(M, experiment_name, params)
            except Exception as e:
                print(f"Logging failed for {experiment_name}: {e}")

            # --- build output row ---
            output = {
                "param_name": param_name,
                "param_value": val,
                "CO2_ppm": diag.get("CO2_ppm", np.nan),
                "A_zcc": diag.get("A_zcc", np.nan),
                "I_zcc": diag.get("I_zcc", np.nan),
                "P_zcc": diag.get("P_zcc", np.nan),
                "A_deep_CO3": diag.get("A_deep_CO3", np.nan),
                "I_deep_CO3": diag.get("I_deep_CO3", np.nan),
                "P_deep_CO3": diag.get("P_deep_CO3", np.nan),
                "A_deep_O2": diag.get("A_deep_O2", np.nan),
                "I_deep_O2": diag.get("I_deep_O2", np.nan),
                "P_deep_O2": diag.get("P_deep_O2", np.nan),
            }

        except Exception as e:
            # --- handle crash ---
            print(f"Run failed for {param_name}={val}: {e}")
            output = {
                "param_name": param_name,
                "param_value": val,
                "CO2_ppm": np.nan,
                "A_zcc": np.nan,
                "I_zcc": np.nan,
                "P_zcc": np.nan,
                "A_deep_CO3": np.nan,
                "I_deep_CO3": np.nan,
                "P_deep_CO3": np.nan,
                "A_deep_O2": np.nan,
                "I_deep_O2": np.nan,
                "P_deep_O2": np.nan,
            }

        results.append(output)

    return pd.DataFrame(results)


