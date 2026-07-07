from __future__ import annotations
import typing as tp
import numpy.typing as npt
import numpy as np
import pandas as pd
import os
from copy import deepcopy
import gc
from esbmtk import Model, SpeciesProperties


NDArrayFloat = npt.NDArray[np.float64]

# if tp.TYPE_CHECKING:
#    from esbmtk import Model, SpeciesProperties


def create_connections_from_flux_list(
    model: Model,
    flux_list: list,
    target_id: str,
    species: SpeciesProperties,
    scale: int | float,
    **kwargs: dict,  
) -> None:
    """ 
    Create Species2Species connections from an existing list of flux objects.

    This function constructs coupling relationships between reservoir species
    based on a list of reference fluxes. It interprets flux naming conventions
    to infer source/sink reservoirs unless explicitly overridden.

    Parameters
    ----------
    model : Model
        The ESBMTK model instance containing reservoir groups.
    flux_list : list
        List of flux objects used as references for constructing connections.
        Each flux is expected to have at least an ``id`` attribute following
        a naming convention such as ``A_sb_2_A_ib_POP_ex``.
    target_id : str
        Identifier used to label the resulting connection group.
    species : SpeciesProperties
        Species object defining which tracer is being coupled (e.g. DIC, PO4).
    scale : int or float
        Scaling factor applied to the reference flux when constructing the
        Species2Species coupling.

    Other Parameters
    ----------------
    source : str, optional
        Source selection mode or explicit source name.

        Options:
        - "auto" (default): infer source from flux name
        - "from_sink": infer source from sink portion of flux name
        - str: explicit model attribute name for source reservoir group
    sink : str or ReservoirGroup, optional
        Sink selection mode or explicit sink identifier.

        Options:
        - "auto" (default): infer sink from flux name
        - str: explicit sink reservoir group name
    delta : float, optional
        Isotopic fractionation passed to Species2Species.
    epsilon : float, optional
        Additional isotopic or parameter modifier passed to Species2Species.

    Returns
    -------
    None
        The function modifies the model by adding connection objects.
    """
    import logging
    from esbmtk import Species2Species

    source_arg = kwargs.get("source", "auto")
    sink_arg = kwargs.get("sink", "auto")

    delta = kwargs.get("delta", None)
    epsilon = kwargs.get("epsilon", None)

    bypass = "None"

    if len(flux_list) < 1:
        raise ValueError("Flux_list")

    logging.debug("# --- create_connections_from_flux_list ---- #")
    for f in flux_list:
        if source_arg == "auto":
            # Extract source and sink, assuming that the flux name
            # looks like: "A_sb_2_A_ib_POP_ex"
            source_name = "_".join(f.id.split("_")[:2])
        elif source_arg == "from_sink":
            source_name = "_".join(f.full_name.split("_")[4:6])
        else:
            source_name = source_arg.name

        if sink_arg == "auto":
            sink_name = "_".join(f.id.split("_")[3:5])
        else:
            sink_name = sink_arg
            bypass = "sink"

            # get reservoirgroups
        source_reservoir_group_handle = getattr(model, source_name)
        sink_reservoir_group_handle = getattr(model, sink_name)

        # Reservoir objects
        source = getattr(source_reservoir_group_handle, species.name)
        sink = getattr(sink_reservoir_group_handle, species.name)

        c = Species2Species(
            source=source,
            sink=sink,
            species=species,
            ctype="scale_with_flux",
            ref_flux=f,  # <-- indexed reference
            scale=scale,
            delta=delta,
            epsilon=epsilon,
            id=f"C_{source_name}_to_{sink_name}_{target_id}",
            bypass=bypass,
        )
        logging.debug(f"Created {c.full_name}")
    logging.debug("\n")
    # if target_id == "PIC_DIC_shelf":
    #     breakpoint()


def create_weathering_fluxes(
    model: Model,
    species: SpeciesProperties,
    area_dict: dict,
    ref_flux_name: str,
    scale: int | float,
    delta: float | None = None,
    alpha: float | None = None,
    **kwargs: dict,
) -> None:
    """
    Create Species2Species connections representing weathering fluxes.

    This function generates coupling terms between a crustal or atmospheric
    source reservoir and basin sink reservoirs, scaled by basin area and a
    reference flux.

    Parameters
    ----------
    model : Model
        ESBMTK model instance containing reservoirs and flux definitions.
    species : SpeciesProperties
        Species being transported (e.g. DIC, alkalinity).
    area_dict : dict
        Dictionary mapping basin names to their surface areas.
    ref_flux_name : str
        Name of the reference flux used.
    scale : int or float
        Scaling factor applied to all basin weathering fluxes.
    delta : float, optional
        Optional isotopic fractionation parameter passed to Species2Species.
    alpha : float, optional
        Optional isotopic fractionation parameter passed to Species2Species.

    Other Parameters
    ----------------
    source : str, optional
        Source reservoir selection mode.

        Options:
        - "crust" (default): use crustal reservoir (Fw.<species>)
        - "atmosphere": use atmospheric reservoir (e.g., CO2_At for DIC)

    Returns
    -------
    None
        The function modifies the model by adding connection objects.

    Notes
    -----
    - Weathering fluxes are constructed per basin.
    - Source selection currently supports crust and atmosphere only.

    Examples
    --------
    >>> create_weathering_fluxes(model, DIC, areas, "weathering_silicate", 1.0)
    """
    from operator import attrgetter
    import logging
    from esbmtk import Species2Species

    source_arg = kwargs.get("source", "crust")

    logging.debug("# --- create_connections_from_flux_list ---- #")
    for basin, area in area_dict.items():
        if source_arg == "crust":
            # FIXME: query list of sources
            source = attrgetter(f"Fw.{species.name}")(model)
        elif source_arg == "atmosphere":
            if species.name == "DIC":
                # FIXME: query list of gas reservoirs
                source = getattr(model, "CO2_At")

        sink = attrgetter(f"{basin}.{species.name}")(model)
        cid = f"{sink.full_name.split('.')[1]}.{species.name}_{ref_flux_name}_weathering_x"
        if model.debug:
            logging.debug(f"source = {source.full_name}, type = {type(source)}")
            logging.debug(f"sink = {sink.full_name}, type = {type(sink)}")
            logging.debug(f"species = {species.full_name}, type = {type(species)}")
            logging.debug(f"ref_flux = {ref_flux_name}")
            logging.debug(f"id = {cid}")

        # ---------------- Species2Species ---------------- #
        kwargs_s2s = dict(
            ctype="scale_with_flux",
            source=source,
            sink=sink,
            species=species,
            ref_flux=ref_flux_name,
            scale=area * scale,
            id=cid,
        )

        # only forward if provided
        if delta is not None:
            kwargs_s2s["delta"] = delta
        if alpha is not None:
            kwargs_s2s["alpha"] = alpha

        Species2Species(**kwargs_s2s)
        
        # c.name = (f"C_{source.name}_to_{sink.name}_{species.name}_{c.id}",)
        # c.full_name = (f"M.C_{source.name}_to_{sink.name}_{species.name}_{c.id}",)
        # logging.debug(f"Created {c.full_name}")
        # breakpoint()
    logging.debug("\n")

def get_matrix_coefficients(
    flux_name: str,
    CM: NDArrayFloat,
    F: NDArrayFloat,
    F_names: list[str],
    R_names: list[str],
    *,
    include_zeros: bool = False,
    atol: float = 0.0,
) -> list[tuple[str, float]]:
    """Return reservoir rows affected by a given flux (and their CM coefficients).

    A flux corresponds to one column in CM. Reservoirs correspond to rows in CM.
    This function finds the column index for `flux_name` in `F_names`, then returns
    (reservoir_name, CM[row, col]) for each reservoir row where that coefficient is
    non-zero (or all rows if include_zeros=True).

    Parameters
    ----------
    flux_name
        Name as stored in F_names (e.g., entries produced by f.full_name).
    CM
        Coefficient matrix with shape (n_reservoir_rows, n_fluxes).
    F
        Flux value vector with shape (n_fluxes,). (Not required for coefficients,
        but kept to match your provided signature and for sanity checks.)
    F_names
        Flux names aligned with flux indices (columns of CM, entries of F).
    R_names
        Reservoir row names aligned with row indices of CM.
    include_zeros
        If True, return all reservoirs with their coefficient (including 0.0).
        If False, only return reservoirs with non-zero coefficients.
    atol
        Absolute tolerance for treating very small coefficients as zero.

    Returns
    -------
    list[tuple[str, float]]
        List of (reservoir_name, coefficient) tuples.
    """
    if flux_name not in F_names:
        raise KeyError(f"Flux name not found in F_names: {flux_name!r}")

    col = F_names.index(flux_name)

    # Basic alignment checks (optional but helpful)
    if CM.shape[1] != len(F_names):
        raise ValueError(
            f"CM has {CM.shape[1]} columns but F_names has {len(F_names)} entries."
        )
    if CM.shape[0] != len(R_names):
        raise ValueError(
            f"CM has {CM.shape[0]} rows but R_names has {len(R_names)} entries."
        )
    if len(F) != len(F_names):
        raise ValueError(f"F has length {len(F)} but F_names has {len(F_names)}.")

    coeff_col = CM[:, col]

    if include_zeros:
        return [(R_names[i], float(coeff_col[i])) for i in range(len(R_names))]

    if atol > 0.0:
        rows = np.where(np.abs(coeff_col) > atol)[0]
    else:
        rows = np.nonzero(coeff_col)[0]

    return [
        (flux_name, R_names[i], f"coeff = {coeff_col[i]:.2e}, val = {F[col]:.2e}")
        for i in rows
    ]



#the above ^ should be integrated into utility functions

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


