from __future__ import annotations
import typing as tp
import numpy.typing as npt
import numpy as np

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


def create_gas_exchange_connections(model, basin_list, species, piston_velocity):
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
            ctype="gasexchange",
            id=cid,
        )


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
