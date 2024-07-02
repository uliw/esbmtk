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


def calculate_burial(po4_export_flux: float, o2_c: float, p: tuple) -> float:
    """#add an empty tuple
    Calculate burial as a function of productivity and oxygen concentration.

    :param po4_export_flux: Surface ocean productivity in umol/L
    :type po4_export_flux: float
    :param o2_con: Oxygen concentration in the deep box in umol/L
    :type o2_con: float
    :return: Burial flux in mol/year
    :rtype: float
    """
    frac_burial, dbv, min_burial_fraction, max_burial_fraction = p

    frac_burial = min_burial_fraction + (max_burial_fraction - min_burial_fraction) * (
        o2_c / 100
    )

    # productivity in mol/year
    productivity_mol_year = po4_export_flux * dbv * 1e-6  # Convert umol/L to mol

    burial_flux = productivity_mol_year * frac_burial
    return -burial_flux


def add_my_burial(
    source: Reservoir,
    sink: Reservoir,
    species: SpeciesProperties,
    o2_c: Species,
    po4_export_flux: Flux,
    frac_burial: float,
    min_burial_fraction: float,
    max_burial_fraction:float,
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
    print(f"Type of source: {type(source)}")
    print(f"Type of sink: {type(sink)}")
    print(f"Type of species: {type(species)}")

    model = species.mo
    dbv: float = source.volume.to(model.v_unit).magnitude

    print(f"Source name: {source.full_name}")
    ec = ExternalCode(
        name="calculate_burial",
        species=species,
        ftype="needs_flux",
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
            {f"F_{sink.full_name}.{species.name}": "burial_flux"},
        ],
        register=source,
    )
    register_return_values(ec, source)
    source.mo.lpc_f.append(ec.fname)
