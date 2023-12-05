"""esbmtk: A general purpose Earth Science box model toolkit Copyright
     (C), 2020-2021 Ulrich G. Wortmann

     This program is free software: you can redistribute it and/or
     modify it under the terms of the GNU General Public License as
     published by the Free Software Foundation, either version 3 of
     the License, or (at your option) any later version.

     This program is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
     General Public License for more details.

     You should have received a copy of the GNU General Public License
     along with this program.  If not, see
     <https://www.gnu.org/licenses/>.

"""
from __future__ import annotations
import typing as tp
import numpy as np
from numba import njit
from esbmtk.utility_functions import (
    __checkkeys__,
    __addmissingdefaults__,
    __checktypes__,
    register_return_values,
)

# if tp.TYPE_CHECKING:
#     from .esbmtk import SeawaterConstants, ReservoirGroup, Flux

# declare numpy types
# NDArrayFloat = npt.NDArray[np.float64]


# @njit(fastmath=True)
def weathering(pco2t, p) -> float:
    """Calculates weathering as a function pCO2 concentration

    :param pco2: float current pco2
    :param pco2_0: float reference pco2
    :area_fraction: float area/total area
    :param ex: exponent
    :f0: flux at pco2_0
    :returns:  F_w

    F_w = area_fraction * f0 * (pco2/pco2_0)**ex
    """

    pco2_0, area_fraction, ex, f0, isotopes = p

    if isotopes:
        pco2, pco2_0i = pco2t
    else:
        pco2 = pco2t

    return area_fraction * f0 * (pco2 / pco2_0) ** ex


def init_weathering(c, pco2, pco2_0, area_fraction, ex, f0):
    """Creates a new external code instance

    :param c: Connection
    :param pco2: float current pco2
    :param pco2_0: float reference pco2
    :area_fraction: float area/total area
    :param ex: exponent
    :f0: flux at pco2_0

    """
    from esbmtk import ExternalCode

    p = (pco2_0, area_fraction, ex, f0, c.source.isotopes)
    ec = ExternalCode(
        name="ec_weathering",
        fname="weathering",
        ftype="std",
        species=c.source.species,
        function_input_data=[pco2],
        function_params=p,
        register=c.model,
        return_values=[
            {f"F_{c.fh.full_name}": "fww"},
        ],
    )
    c.mo.lpc_f.append(ec.fname)

    return ec
