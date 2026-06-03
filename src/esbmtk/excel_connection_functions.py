from __future__ import annotations

import pandas as pd

from esbmtk import Q_, create_bulk_connections
from esbmtk.extended_classes import GasReservoir
from esbmtk.base_classes import SpeciesProperties
from esbmtk.utility_functions import initialize_reservoirs

def create_reservoirs_from_excel(
    M,
    excel_file: str,
    sheet_name: str = "reservoirs",
    species_units: dict | None = None,
    default_temperature: float | None = None,
    default_salinity: float | None = None,
    default_pressure: float | None = None,
):
    """Create ESBMTK reservoirs, sources, and sinks from an Excel worksheet.

    This function reads a spreadsheet describing model boxes, converts each
    row into a dictionary key, and then calls ``initialize_reservoirs()`` to 
    create the corresponding ESBMTK objects.

    Reservoir rows must contain geometry information and may contain any
    number of species concentration columns. Species columns are detected
    automatically by matching spreadsheet column names to
    ``SpeciesProperties`` objects registered in the model.

    Parameters
    ----------
    M : Model
        ESBMTK model instance.

    excel_file : str
        Path to the Excel workbook.

    sheet_name : str, default="reservoirs"
        Worksheet containing reservoir definitions.

    species_units : dict, optional
        Dictionary mapping species names to concentration units.

        Example::

            {
                "DIC": "umol/kg",
                "TA": "umol/kg",
                "PO4": "umol/kg",
            }

        Species not listed default to ``"umol/kg"``.

    default_temperature : float, optional
        Default temperature used when a row does not specify a value.

    default_salinity : float, optional
        Default salinity used when a row does not specify a value.

    default_pressure : float, optional
        Default pressure used when a row does not specify a value.

    Returns
    -------
    list
        List of objects returned by
        ``initialize_reservoirs()``.

    Raises
    ------
    ValueError
        If no SpeciesProperties objects are found in the model.

    Notes
    -----
    Minimal required spreadsheet columns::

        "name"
        "type"

    Reservoir rows additionally require::

        "z_top"
        "z_bottom"
        "area_percentage"

    Optional columns (for reservoirs)::

        "temperature"
        "pressure"
        "salinity"

    Any additional column whose name matches a SpeciesProperties object
    registered in the model is interpreted as a species concentration.

    TO ADD: DELTA (FOR ISOTOPES)
    """

    if species_units is None:
        species_units = {}

    df = pd.read_excel(excel_file, sheet_name=sheet_name)

    model_species = {
        name: obj
        for name, obj in M.__dict__.items()
        if isinstance(obj, SpeciesProperties)
    }

    if not model_species:
        raise ValueError("No SpeciesProperties found in model.")

    metadata_columns = {
        "name",
        "type",
        "z_top",
        "z_bottom",
        "area_percentage",
        "temperature",
        "pressure",
        "salinity",
    }

    box_dict = {}

    for _, row in df.iterrows():

        name = str(row["name"]).strip()
        box_type = str(row["type"]).strip().lower()

        entry = {}

        if box_type == "reservoir":

            entry["g"] = [
                float(row["z_top"]),
                float(row["z_bottom"]),
                float(row["area_percentage"]),
            ]

            concentrations = {}

            for col in df.columns:

                if col in metadata_columns:
                    continue

                if col not in model_species:
                    continue

                if pd.isna(row[col]):
                    continue

                species = model_species[col]
                unit = species_units.get(col, "umol/kg")

                concentrations[species] = f"{row[col]} {unit}"

            entry["c"] = concentrations

            entry["T"] = (
                row["temperature"]
                if "temperature" in df.columns
                and not pd.isna(row["temperature"])
                else default_temperature
            )

            entry["P"] = (
                row["pressure"]
                if "pressure" in df.columns
                and not pd.isna(row["pressure"])
                else default_pressure
            )

            entry["S"] = (
                row["salinity"]
                if "salinity" in df.columns
                and not pd.isna(row["salinity"])
                else default_salinity
            )

        elif box_type == "source":

            entry["ty"] = "Source"
            entry["sp"] = list(model_species.values())

        elif box_type == "sink":

            entry["ty"] = "Sink"
            entry["sp"] = list(model_species.values())

        else:
            raise ValueError(
                f"Unknown box type '{box_type}' for row '{name}'"
            )

        box_dict[name] = entry

    return initialize_reservoirs(M, box_dict)

def create_transport_matrix_from_excel(
    M,
    excel_file: str,
    species_list,
    sheet_name: str = "transport_matrix",
    connection_type: str = "scale_with_concentration",
):
    """Construct transport connections for a model from an Excel sheet.

    The function reads an Excel file defining transport connections
    between reservoirs, and registers them in the model.

    Parameters
    ----------
    M : object
        Model object.

    excel_file : str
        Path to an Excel file defining the transport matrix.

    species_list : list [SpeciesProperties]
        List of species associated with each transport connection.

    sheet_name: str
        Name of the Excel worksheet containing the transport matrix. Default is "transport_matrix".

    connection_type : str, optional
        Type identifier for the connection. Default is "scale_with_concentration".

    Returns
    -------
    dict
        Dictionary of constructed connections. Keys are connection names of
        the form ``"{source}_to_{sink}@{flux_id}"`` and values are dicts
        containing:
            - ty : str
                Connection type.
            - sc : float or Quantity
                Scaling factor (evaluated expression or parsed quantity).
            - sp : list
                Species list.

    Notes
    -----
    Excel required columns format:
        - source : str
        - sink : str
        - flux_id : str
        - sc : str

    Connections with ``flux_id == "mix_up"`` automatically generate a
    corresponding reverse connection named ``mix_down`` unless that
    connection already exists in the spreadsheet.

    Examples
    --------
    Excel table:

    source | sink | flux_id     | sc
    -------|------|------------|----------------
    H_sb   | A_db | thermohaline | thc
    A_db   | A_ib | thermohaline | ta * thc
    A_ib   | A_sb | mix_up       | 21 Sverdrup
    
    TO ADD: EPSILON (FOR ISOTOPES)
    """

    # Load transport definition table from Excel
    df = pd.read_excel(excel_file, sheet_name=sheet_name)

    lookup = {
        name: getattr(M, name)
        for name in dir(M)
        if not name.startswith("_")
    }

    ct = {} # Empty dictionary for connections

    for _, row in df.iterrows():

        source = str(row["source"]).strip()
        sink = str(row["sink"]).strip()
        flux_id = str(row["flux_id"]).strip()

        # Scaling expression (either symbolic or quantity string)
        scale_expr = str(row["sc"]).strip()

        try:
            # Attempt symbolic evaluation (e.g., ta * thc)
            scale = eval(scale_expr, {}, lookup)

        except Exception:
            # Alternately interpret as physical quantity (e.g., "21 Sverdrup")
            scale = Q_(scale_expr)

        # Unique connection identifier string
        connection_name = f"{source}_to_{sink}@{flux_id}"

        ct[connection_name] = {
            "ty": connection_type,
            "sc": scale,
            "sp": species_list,
        }

    # ---------------------------------------------------------------------
    # Auto-generate reverse connections for "mix_up"
    # ---------------------------------------------------------------------
    additions = {}

    for connection_name, entry in ct.items():

        source_sink, flux_id = connection_name.split("@")

        # Only handle symmetric mixing fluxes
        if flux_id != "mix_up":
            continue

        source, sink = source_sink.split("_to_")

        # Define reverse connection name
        reverse_name = f"{sink}_to_{source}@mix_down"

        # Skip if reverse already explicitly defined in Excel
        if reverse_name in ct:
            continue

        additions[reverse_name] = {
            "ty": entry["ty"],
            "sc": entry["sc"],
            "sp": entry["sp"],
        }

    # Merge auto-generated reverse connections
    ct.update(additions)

    # Register all connections in the model
    create_bulk_connections(ct, M)

    return ct


def create_gas_reservoirs_from_excel(
    M,
    excel_file: str,
    sheet_name: str = "gas_reservoirs",
):
    """ Create atmospheric reservoirs from an Excel worksheet.

    Parameters
    ----------
    M : Model
        ESBMTK model instance containing the species definitions
        referenced by the worksheet.

    excel_file : str
        Path to the Excel workbook.

    sheet_name : str, default="gas_reservoirs"
        Worksheet containing gas reservoir definitions.

    Returns
    -------
    dict
        Dictionary mapping reservoir names to the created
        ``GasReservoir`` objects.

    Raises
    ------
    ValueError
        If no ``SpeciesProperties`` objects are found in the model.

    ValueError
        If a referenced species does not exist.

    ValueError
        If ``species_ppm`` is missing.

    Notes
    -----
    Required columns::

        name
        species
        species_ppm

    Optional columns::

        delta
        reservoir_mass
        plot

    Numeric values supplied in the ``species_ppm`` column are
    automatically interpreted as ppm and converted to strings of
    the form ``"<value> ppm"``.

    Examples
    --------
    Excel sheet::

        name      | species | species_ppm | delta
        ----------|---------|-------------|------
        CO2_At    | CO2     | 420         | 0
        O2_At     | O2      | 209000      | 0
    """

    df = pd.read_excel(excel_file, sheet_name=sheet_name)

    species_lookup = {
        name: obj
        for name, obj in M.__dict__.items()
        if isinstance(obj, SpeciesProperties)
    }

    if not species_lookup:
        raise ValueError("No SpeciesProperties found in model")

    created = {}

    for _, row in df.iterrows():

        name = str(row["name"]).strip()

        species_name = str(row["species"]).strip()
        if species_name not in species_lookup:
            raise ValueError(
                f"Unknown species '{species_name}' for reservoir '{name}'"
            )

        species = species_lookup[species_name]

        species_ppm = row.get("species_ppm")
        if pd.isna(species_ppm):
            raise ValueError(f"Missing species_ppm for '{name}'")

        # Preserve original string values exactly.
        # Only convert numeric values to ppm strings.
        if isinstance(species_ppm, (int, float)):
            species_ppm = f"{species_ppm} ppm"
        else:
            species_ppm = str(species_ppm).strip()

        kwargs = {
            "name": name,
            "species": species,
            "species_ppm": species_ppm,
        }

        # Optional arguments: only pass if present
        if "delta" in row.index and pd.notna(row["delta"]):
            kwargs["delta"] = row["delta"]

        if "reservoir_mass" in row.index and pd.notna(row["reservoir_mass"]):
            kwargs["reservoir_mass"] = Q_(str(row["reservoir_mass"]))

        if "plot" in row.index and pd.notna(row["plot"]):
            kwargs["plot"] = row["plot"]

        obj = GasReservoir(**kwargs)

        created[name] = obj

    return created

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

        Species2Species(  
            source=source,  # Reservoir Species
            sink=sink,  # Reservoir Species
            species=species,
            piston_velocity=piston_velocity,
            scale=scale,
            ctype="gasexchange",
            id=cid,
        )


def create_gas_exchange_connections_from_excel(
    M,
    excel_file: str,
    sheet_name: str = "gas_exchange",
):
    """
    Create gas exchange connections from an Excel worksheet.

    Parameters
    ----------
    M : Model
        ESBMTK model instance.

    excel_file : str
        Path to Excel workbook.

    sheet_name : str, default="gas_exchange"
        Worksheet containing gas exchange definitions.

    Returns
    -------
    None

    Notes
    -----
    Required columns::

        species
        basins
        piston_velocity

    Optional columns::

        scale

    Example
    -------
    Excel sheet::

        species | basins                  | piston_velocity | scale
        CO2     | A_sb,I_sb,P_sb,H_sb     | 4.8             | 1.0
        O2      | A_sb,I_sb,P_sb,H_sb     | 4.8             | 1.0
    """

    df = pd.read_excel(excel_file, sheet_name=sheet_name)

    species_lookup = {
        name: obj
        for name, obj in M.__dict__.items()
        if isinstance(obj, SpeciesProperties)
    }

    for _, row in df.iterrows():

        species_name = str(row["species"]).strip()

        if species_name not in species_lookup:
            raise ValueError(f"Unknown species '{species_name}'")

        species = species_lookup[species_name]

        basin_list = []

        for basin_name in str(row["basins"]).split(","):

            basin_name = basin_name.strip()

            if not hasattr(M, basin_name):
                raise ValueError(f"Unknown basin '{basin_name}'")

            basin_list.append(getattr(M, basin_name))

        piston_velocity = row["piston_velocity"]

        scale = row["scale"] if "scale" in df.columns else 1.0

        if pd.isna(scale):
            scale = 1.0

        create_gas_exchange_connections(
            model=M,
            basin_list=basin_list,
            species=species,
            piston_velocity=piston_velocity,
            scale=scale,
        )