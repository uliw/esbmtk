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
from numba import njit
from esbmtk.utility_functions import (
    __checkkeys__,
    __addmissingdefaults__,
    __checktypes__,
    register_return_values,
)

if tp.TYPE_CHECKING:
    from esbmtk import Connection, Connect, Q_

# declare numpy types
# NDArrayFloat = npt.NDArray[np.float64]


# @njit(fastmath=True)
def weathering(pco2t, p) -> float | tuple:
    """Calculates weathering as a function pCO2 concentration

    :param pco2: float current pco2
    :param pco2_0: float reference pco2
    :area_fraction: float area/total area
    :param ex: exponent
    :f0: flux at pco2_0
    :returns:  F_w or F_w, F_w_i

    F_w = area_fraction * f0 * (pco2/pco2_0)**ex
    """
    pco2_0, area_fraction, ex, f0, isotopes = p
    if isotopes:
        pco2, pco2i = pco2t
        F_w = area_fraction * f0 * (pco2 / pco2_0) ** ex
        F_w_i = F_w * pco2i / pco2
        rv = F_w, F_w_i
    else:
        pco2 = pco2t
        F_w = area_fraction * f0 * (pco2 / pco2_0) ** ex
        rv = F_w

    return rv


def init_weathering(
    c: Connection | Connect,
    pco2: float,
    pco2_0: float | str | Q_,
    area_fraction: float,
    ex: float,
    f0: float | str | Q_,
):
    """Creates a new external code instance

    :param c: Connection
    :param pco2: float current pco2
    :param pco2_0: float reference pco2
    :area_fraction: float area/total area
    :param ex: exponent
    :f0: flux at pco2_0

    """
    from esbmtk import ExternalCode, check_for_quantity

    print(f" initializing weathering with isotopes = {c.sink.isotopes}")

    f0 = check_for_quantity(f0, "mol/year").magnitude
    pco2_0 = check_for_quantity(pco2_0, "ppm").magnitude
    p = (pco2_0, area_fraction, ex, f0, c.sink.isotopes)
    c.fh.ftype = "computed"
    ec = ExternalCode(
        name=f"ec_weathering_{c.id}",
        fname="weathering",
        ftype="std",
        species=c.sink.species,
        function_input_data=[pco2],
        function_params=p,
        register=c.model,
        return_values=[
            {f"F_{c.fh.full_name}": "fww"},
        ],
    )
    c.mo.lpc_f.append(ec.fname)

    return ec


def init_gas_exchange(c: Connection | Connect):
    """Create an ExternalCode instance for gas exchange reactions

    Parameters
    ----------
    c : Connection | Connect
        connection instance
    """
    from esbmtk import ExternalCode, check_for_quantity

    c.fh.ftype = "computed"
    swc = c.sink.register.swc  # sink - liquid
    piston_velocity = (
        check_for_quantity(c.piston_velocity, "m/yr").to("meter/year").magnitude
    )
    area = check_for_quantity(c.area, "m**2").to("meter**2").magnitude
    scale = area * piston_velocity
    p = (
        scale,
        c.water_vapor_pressure,
        c.solubility,
        swc.a_db,
        swc.a_dg,
        swc.a_u,
        c.source.isotopes,
    )

    ec = ExternalCode(
        name=f"gas_exchange{c.id}",
        fname="gas_exchange",
        ftype="std",
        species=c.source.species,
        function_input_data=[c.source, c.sink, c.ref_species],
        function_params=p,
        register=c.model,
        return_values=[
            {f"F_{c.fh.full_name}": "gex"},
        ],
    )
    c.mo.lpc_f.append(ec.fname)

    return ec


def gas_exchange(
    gas_c: float | tuple,
    liquid_c: float | tuple,
    gas_aq: float,
    p: tuple,
) -> float | tuple:
    """Calculate the gas exchange flux across the air sea interface
    for co2 including isotope effects.

    Parameters
    ----------
    gas_c : float | tuple
        gas concentration in atmosphere
    liquid_c : float | tuple
        reference species in liquid phase, e.g., DIC
    gas_aq: float
        dissolved gas concentration, e.g., CO2aq
    p : tuple
        parameters, see init_gas_exchange

    Returns
    -------
    float | tuple
        gas flux across the air/sea interface

    Note that the sink delta is co2aq as returned by the carbonate VR
    this equation is for mmol but esbmtk uses mol, so we need to
    multiply by 1E3

    The Total flux across interface dpends on the difference in either
    concentration or pressure the atmospheric pressure is known, as gas_c, and
    we can calculate the equilibrium pressure that corresponds to the dissolved
    gas in the water as [CO2]aq/beta.

    Conversely, we can convert the the pCO2 into the amount of dissolved CO2 =
    pCO2 * beta

    The h/c ratio in HCO3 estimated via h/c in DIC. Zeebe writes C12/C13 ratio
    but that does not work. the C13/C ratio results however in -8 permil
    offset, which is closer to observations
    """
    scale, p_H2O, solubility, a_db, a_dg, a_u, isotopes = p

    if isotopes:
        gas_c, gas_c_l = gas_c
        liquid_c, liquid_c_l = liquid_c

    # Solubility with correction for pH2O
    beta = solubility * (1 - p_H2O)
    # f as afunction of solubility difference
    f = scale * (beta * gas_c - gas_aq * 1e3)
    rv = f

    if isotopes:  # isotope ratio of DIC
        Rt = (liquid_c - liquid_c_l) / liquid_c
        # get heavy isotope concentrations in atmosphere
        gas_c_h = gas_c - gas_c_l  # gas heavy isotope concentration
        # get exchange of the heavy isotope
        f_h = scale * a_u * (a_dg * gas_c_h * beta - Rt * a_db * gas_aq * 1e3)
        f_l = f - f_h  # the corresponding flux of the light isotope
        rv = f, f_l

    return rv
