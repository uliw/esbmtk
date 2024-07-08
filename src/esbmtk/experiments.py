"""
     esbmtk: A general purpose Earth Science box model toolkit
     Copyright (C), 2020 Ulrich G. Wortmann

     This program is free software: you can redistribute it and/or modify
     it under the terms of the GNU General Public License as published by
     the Free Software Foundation, either version 3 of the License, or
     (at your option) any later version.

     This program is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
     GNU General Public License for more details.

     You should have received a copy of the GNU General Public License
     along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
import numpy.typing as npt
from esbmtk import (
    ExternalCode,
    register_return_values,
    Flux,
    Species,
    SpeciesProperties,
    Reservoir,
)

# declare numpy types
NDArrayFloat = npt.NDArray[np.float64]

counter = 0


def calculate_burial(
    po4_export_flux: float, o2_c: float, frac_burial_params: tuple
) -> float:
    """#add an empty tuple
    Calculate burial as a function of productivity and oxygen concentration.

    :param po4_export_flux: Surface ocean productivity in umol/L
    :type po4_export_flux: float
    :param o2_con: Oxygen concentration in the deep box in umol/L
    :type o2_con: float
    :return: Burial flux in mol/year
    :rtype: float
    """
    # global counter # never ever use global variables!
    # counter += 1

    frac_burial, dbv, min_burial_fraction, max_burial_fraction = frac_burial_params
    """
    frac_burial = min_burial_fraction + (max_burial_fraction - min_burial_fraction) * (
        o2_c / 100
    )
    # )
    """
    frac_burial = min_burial_fraction

    # productivity in mol/year
    productivity_mol_year = po4_export_flux  # Convert umol/L to mol

    burial_flux = productivity_mol_year * (frac_burial)

    p_remineralisation_flux = (productivity_mol_year - frac_burial) * 138

    # print(f"bf = {-burial_flux:.2e}, rf = { p_remineralisation_flux:.2e}")

    return -burial_flux, p_remineralisation_flux


def add_my_burial(
    source: Reservoir,
    sink: Reservoir,
    species: SpeciesProperties,
    o2_c: Species,
    po4_export_flux: Flux,
    frac_burial: float,
    min_burial_fraction: float,
    max_burial_fraction: float,
    my_id1,
    my_id2,
    # pos,
) -> None:
    """
    This function initializes a user supplied function so that it can be used within the ESBMTK ecosystem.

    Parameters
    ----------
    source : Source | Species | Reservoir
        A source
    sink : Sink | Species | Reservoir
        A sink
    species : SpeciesProperties
        A model species
    po4_export_flux : float
        PO4 export flux in umol/L
    o2_c : float
        Oxygen concentration in umol/L
    frac_burial : float
        A scaling factor of burial fraction
    min_burial_fraction : float
        Minimum burial fraction
    max_burial_fraction : float
        Maximum burial fraction
    """
    # print(f"Type of source: {type(source)}")
    # print(f"Type of sink: {type(sink)}")
    # print(f"Type of species: {type(species)}")

    # ensure that the volume is in actual model units, and then strip
    # the unit information

    # print(f"po4_export_flux: {po4_export_flux}, o2_c: {o2_c}")
    model = species.mo
    dbv: float = source.volume.to(model.v_unit).magnitude

    # print(f"Source name: {source.full_name}")
    ec = ExternalCode(
        name="calculate_burial",
        species=species,
        ftype="in_sequence",
        function=calculate_burial,
        fname="calculate_burial",
        function_input_data=[po4_export_flux, o2_c],
        function_params=(
            frac_burial,
            dbv,
            min_burial_fraction,
            max_burial_fraction,
        ),
        return_values=[
            {f"F_{sink.full_name}.{species.name}": f"{my_id1}__F"},
            {f"F_{source.full_name}.{species.name}": f"{my_id2}__F"},
        ],
        register=source,
    )
    register_return_values(ec, source)
    source.mo.lpc_f.append(ec.fname)  # omitted for some reason?

    # Debug prints
    # print(f"Source name: {source.full_name}, Sink name: {sink.full_name}")
    # print(
    #   f"Function Params: {frac_burial}, {dbv}, {min_burial_fraction}, {max_burial_fraction}"
    # )
