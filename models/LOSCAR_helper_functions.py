from __future__ import annotations
import typing as tp
import numpy.typing as npt
import numpy as np
import pandas as pd
import os

NDArrayFloat = npt.NDArray[np.float64]

# if tp.TYPE_CHECKING:
#    from esbmtk import Model, SpeciesProperties

from esbmtk import Model, SpeciesProperties


def create_connections_from_flux_list(
    model: Model,
    flux_list: list,
    target_id: str,
    species: SpeciesProperties,
    scale: int | float,
    **kwargs: dict,
) -> None:
    """Create new connections based on a list of existing fluxes.

    This function will evaluate the following keywords

        - source: str = auto | from_sink_name | any str
        - sink: str | ReservoirGroup =  auto | RG

    The first value being the default.
    - "auto" will determine the source and sink reservoirs based on the flux name
       i.e., "A_sb_2_A_ib_POP_ex" will create a connection from A_sb to A_ib
    - "from_sink_name" will determine the source from the sink in the flux name,
      i.e,  "A_ib_2_A_sb_mix_up" will set the source to "A_sb". In this case the sink must be specified explicitly (i.e., sink="Fb")f
    - source_name must be a string with a valide model source name (e.g. "Fw")
    - sink_name must be a string with a valide model sink name (e.g. "Fb")
    """
    import logging
    from esbmtk import Species2Species

    source_arg = kwargs.get("source", "auto")
    sink_arg = kwargs.get("sink", "auto")
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
    **kwargs: dict,
) -> None:
    """Create the connection objects for weathering fluxes.

    This function will evaluate the following keywords

    - source: str = crust | atmosphere
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

        c = Species2Species(  # Atlantic
            ctype="scale_with_flux",
            source=source,
            sink=sink,
            species=species,
            ref_flux=ref_flux_name,
            scale=area * scale,
            id=cid,
        )
        # c.name = (f"C_{source.name}_to_{sink.name}_{species.name}_{c.id}",)
        # c.full_name = (f"M.C_{source.name}_to_{sink.name}_{species.name}_{c.id}",)
        # logging.debug(f"Created {c.full_name}")
        # breakpoint()
    logging.debug("\n")


def create_gas_exchange_connections(model, basin_list, species, piston_velocity, scale):
    """Create gas exchange connection objects."""
    from esbmtk import Species2Species

    # get reservoirgroup object
    for basin in basin_list:
        reservoir = getattr(model, basin.name)
        source = getattr(model, f"{species.name}_At")
        if species.name == "CO2":
            sink = getattr(reservoir, "DIC")
        else:
            sink = getattr(reservoir, species.name)

        cid = f"{basin.name}_{species.name}_gex"

        Species2Species(  # Pacific surface to atmosphere
            source=source,  # Reservoir Species
            sink=sink,  # Reservoir Species
            species=species,
            piston_velocity=piston_velocity,
            scale=scale,
            ctype="gasexchange",
            id=cid,
        )

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

    # ---------------- High latitude surface box ----------------
    diag["H_DIC"] = round(M.H_sb.DIC.c[-1] * 1e6, 1)
    diag["H_TA"] = round(M.H_sb.TA.c[-1] * 1e6, 1)

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

import pandas as pd

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

    # Optionally round for readability
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

import os
import pandas as pd

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
