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

import esbmtk

from esbmtk import (
    Model,
    Reservoir,
    ConnectionProperties,
    SourceProperties,
    SinkProperties,
    Q_,
)


def calculate_burial(po4_export_flux: float, o2_con: float) -> float:
    """#add an empty tuple
    Calculate burial as a function of productivity and oxygen concentration.

    :param po4_export_flux: Surface ocean productivity in umol/L
    :type po4_export_flux: float
    :param o2_con: Oxygen concentration in the deep box in umol/L
    :type o2_con: float
    :return: Burial flux in mol/year
    :rtype: float
    """
    # burial fraction to [oxygen] approximation of relationship from 0.01 to 0.1
    min_burial_fraction = 0.01
    max_burial_fraction = 0.1
    burial_fraction = min_burial_fraction + (max_burial_fraction - min_burial_fraction) * (o2_con / 100)

    deep_ocean_v = 1E18  # in litres

    # productivity in mol/year
    productivity_mol_year = po4_export_flux * deep_ocean_v * 1E-6  # Convert umol/L to mol

    burial_flux = productivity_mol_year * burial_fraction

    return burial_flux