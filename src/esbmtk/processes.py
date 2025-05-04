"""esbmtk: A general purpose Earth Science box model toolkit.

Copyright(C), 2020-2021 Ulrich G. Wortmann

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
import numpy.typing as npt

if tp.TYPE_CHECKING:
    from esbmtk import Q_, Species2Species

# declare numpy types
NDArrayFloat = npt.NDArray[np.float64]


def weathering_no_isotopes(
    c_pco2: float | list[float],
    source_data: float | list[float],
    p: tuple,
) -> float | tuple:
    """Calculate weathering as a function of pCO2.

    Parameters
    ----------
    c_pco2 : float | list[float]
        current pCO2 concentration as absolute concentration, e.g.
        280 ppm = 0.00028

    p : tuple
        a tuple with the following entries:
        pco2_0 = reference pCO2
        area_fraction = fraction of total surface area
        ex = exponent used in the equation
        f0 = flux at the reference value

    Returns
    -------
    float
        a float value for the weathering flux

    Explanation
    -----------
    If the model uses isotopes, the function expects the concentration
    values for the total mass and the light isotope as a list, and
    will simiraly return the flux as a list of total flux and flux of
    the light isotope.

    The flux itself is calculated as

     F_w = area_fraction * f0 * (pco2/pco2_0)**ex

    """
    pco2_0, area_fraction, ex, f0 = p
    f = area_fraction * f0 * (c_pco2 / pco2_0) ** ex

    return f


# Note this only works for for the case where the atmosphere
# has isotopes, and the sink has none, But fails if both
# have isotopes.
# @njit(fastmath=True)
def weathering_ref_isotopes(
    c_pco2: float | list[float],
    source_data: float | list[float],
    p: tuple,
) -> float | tuple:
    """Calculate weathering as a function of pCO2.

    This is the same function as weathering_no_isotopes, but assumes that the
    data is a tuple.
    """
    pco2_0, area_fraction, ex, f0 = p
    pco2, pco2i = c_pco2
    return f0 * area_fraction * (pco2 / pco2_0) ** ex


# @njit(fastmath=True)
def weathering_isotopes(
    c_pco2: float | list[float],
    source_data: float | list[float],
    p: tuple,
) -> float | tuple:
    """Calculate weathering as a function of pCO2.

    This is the same function as weathering_no_isotopes, but assumes that we
    weather a species (e.g., carbonate) that requires fluxes for both isotopes.
    data is a tuple.
    """
    pco2, pco2_l = c_pco2  # pco2 data
    s_c, s_l = source_data
    pco2_0, area_fraction, ex, f0 = p  # constants
    w_scale = area_fraction * (pco2 / pco2_0) ** ex
    F_w = f0 * w_scale
    F_w_i = f0 * w_scale * s_l / s_c
    return (F_w, F_w_i)


# @njit(fastmath=True)
def weathering_isotopes_delta(
    c_pco2: float | list[float],
    source_data: float | list[float],
    p: tuple,
) -> float | tuple:
    """Calculate weathering as a function of pCO2.

    This is the same function as weathering_no_isotopes, but assumes that we
    weather a species (e.g., carbonate) that requires fluxes for both isotopes.
    data is a tuple.
    """
    pco2_0, area_fraction, ex, f0, delta, r = p
    pco2, pco2i = c_pco2  # pco2 data
    w_scale = area_fraction * (pco2 / pco2_0) ** ex
    F_w = f0 * w_scale
    F_w_i = f0 * w_scale * 1000 / (r * (delta + 1000) + 1000)
    return (F_w, F_w_i)


# @njit(fastmath=True)
def weathering_isotopes_alpha(
    c_pco2: float | list[float],
    source_data: float | list[float],
    p: tuple,
) -> float | tuple:
    """Calculate weathering as a function of pCO2.

    This is the same function as weathering_no_isotopes, but assumes that we
    weather a species (e.g., carbonate) that requires fluxes for both isotopes.
    data is a tuple.
    s_c = source mass
    s_l = source mass of light isotope
    """
    pco2_0, area_fraction, ex, f0, alpha, r = p
    pco2, pco2i = c_pco2  # atmosphere mass and light isotope mass
    s_c, s_l = source_data
    w_scale = area_fraction * (pco2 / pco2_0) ** ex
    F_w = f0 * w_scale  # flux at a given pco2 value
    F_w_i = f0 * w_scale * s_l / (alpha * s_c + s_l - alpha * s_l)
    return (F_w, F_w_i)


def init_weathering(
    c: Species2Species,
    pco2: float,
    pco2_0: float | str | Q_,
    area_fraction: float,
    ex: float,
    f0: float | str | Q_,
):
    """Create a new external code instance.

    :param c: Species2Species
    :param pco2: float current pco2
    :param pco2_0: float reference pco2
    :area_fraction: float area/total area
    :param ex: exponent
    :f0: flux at pco2_0

    """
    from esbmtk import ExternalCode, Sink, Source, check_for_quantity

    f0 = check_for_quantity(f0, "mol/year").magnitude
    pco2_0 = check_for_quantity(pco2_0, "ppm").magnitude
    # p = (pco2_0, area_fraction, ex, f0)
    c.fh.ftype = "computed"
    c.isotopes = c.source.isotopes  # the co may have missed this
    if c.delta != "None":
        weathering_function = "weathering_isotopes_delta"
        p = (pco2_0, area_fraction, ex, f0, c.delta, c.source.species.r)
    elif c.epsilon != "None":
        weathering_function = "weathering_isotopes_alpha"
        alpha = c.epsilon / 1000 + 1  # convert to alpha notation
        p = (
            pco2_0,  # reference pCO2
            area_fraction,  # area relative to total ocean area
            ex,  # exponent
            f0,  # flux at pCO2_0
            alpha,  # fractionation factor
            c.source.species.r,  # isotope reference species
        )
    elif isinstance(c.source, Source) and isinstance(c.sink, Sink):
        if c.reservoir_ref.isotopes:
            weathering_function = "weathering_ref_isotopes"
            p = (pco2_0, area_fraction, ex, f0)
        else:
            weathering_function = "weathering_no_isotopes"
            p = (pco2_0, area_fraction, ex, f0)
    elif (
        (c.isotopes and c.sink.isotopes)
        or (c.reservoir_ref.isotopes and c.sink.isotopes)
        or (c.isotopes and isinstance(c.sink, Sink))
    ):
        weathering_function = "weathering_isotopes"
        p = (pco2_0, area_fraction, ex, f0)
    elif c.reservoir_ref.isotopes and not c.sink.isotopes:
        weathering_function = "weathering_ref_isotopes"
        p = (pco2_0, area_fraction, ex, f0)
    else:
        weathering_function = "weathering_no_isotopes"
        p = (pco2_0, area_fraction, ex, f0)

    ec = ExternalCode(
        name=f"ec_weathering_{c.id}",
        fname=weathering_function,
        isotopes=c.reservoir_ref.isotopes,
        ftype="std",
        species=c.sink.species,
        function_input_data=[pco2, c.source],
        function_params=p,
        register=c.model,
        return_values=[
            {f"F_{c.fh.full_name}": "fww"},
        ],
    )
    c.mo.lpc_f.append(ec.fname)
    return ec


def init_gas_exchange(c: Species2Species):
    """Create ExternalCode instance for gas exchange reactions.

    Parameters
    ----------
    c : Species2Species
        connection instance

    """
    import warnings

    from esbmtk import ExternalCode, check_for_quantity

    c.fh.ftype = "computed"
    sink_reservoir = c.sink.register
    swc = sink_reservoir.swc  # sink - liquid

    if sink_reservoir.set_area_warning:
        warnings.warn(
            f"\nGEX for {sink_reservoir.full_name} is using the entire ocean area.\n"
            "Consider adjusting the area fraction parameter of the box geometry:\n"
            "g = [upper depth, lower depth, area fraction] \n"
        )

    if c.solubility == "None":  # use predefined values
        if c.species.name == "CO2":
            ref_species = sink_reservoir.CO2aq
            solubility = swc.SA_co2
            a_db = swc.co2_a_db
            a_dg = swc.co2_a_dg
            a_u = swc.co2_a_u
        elif c.species.name == "O2":
            ref_species = sink_reservoir.O2
            solubility = swc.SA_O2
            a_db = swc.o2_a_db
            a_dg = swc.o2_a_dg
            a_u = swc.o2_a_u
        else:
            raise ValueError(
                f"Gas exchange is undefined for {c.species.name}\n"
                f"consider manual setup?\n"
            )
    else:  # use user supplied values
        ref_species = sink_reservoir.O2 if c.ref_species == "None" else c.ref_species
        solubility = (
            check_for_quantity(c.solubility, "mol/(m^3 * atm)")
            .to("mol/(m^3 * atm)")
            .magnitude
        )
        a_db = c.a_db
        a_dg = c.a_dg
        a_u = c.a_u

    piston_velocity = (
        check_for_quantity(c.piston_velocity, "m/yr").to("meter/year").magnitude
    )
    area = check_for_quantity(sink_reservoir.area, "m**2").to("meter**2").magnitude

    scale = area * piston_velocity
    p = (
        scale,
        solubility,
        a_db,
        a_dg,
        a_u,
        c.isotopes,  # c.source.isotopes,
        ref_species.isotopes,
    )
    ec = ExternalCode(
        name=f"gas_exchange{c.id}",
        fname="gas_exchange",
        ftype="std",
        species=c.source.species,
        function_input_data=[c.source, c.sink, ref_species],
        function_params=p,
        isotopes=c.isotopes,
        register=c.sink,
        return_values=[
            {f"F_{c.fh.full_name}": "gex"},
        ],
    )
    c.mo.lpc_f.append(ec.fname)
    # this is now initialized through a Species2Species instance. This instance
    # will create the necessary Flux instances. Since this will compute the
    # fluxes. We need to set the compute_by flag
    for f in c.lof:
        f.ftype = "std"
    return ec


def gas_exchange(
    gas_c: float | tuple,
    liquid_c: float | tuple,
    gas_aq_c: float,
    p: tuple,
) -> float | tuple:
    """Calculate the gas exchange flux across the air sea interface.

    including isotope effects.

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
    scale, solubility, a_db, a_dg, a_u, species_isotopes, ref_species_isotopes = p

    if species_isotopes:
        gas_c, gas_c_l = gas_c
        liquid_c, liquid_c_l = liquid_c
        if ref_species_isotopes:
            gas_aq_c, _l = gas_aq_c
        else:
            gas_aq_c = gas_aq_c

    # Solubility with correction for pH2O
    # beta = solubility * (1 - p_H2O)
    beta = solubility  # solubility is already corrected for p_H20
    # f as afunction of solubility difference
    f = scale * (beta * gas_c - gas_aq_c * 1e3)
    rv = f

    if species_isotopes:  # isotope ratio of DIC
        Rt = (liquid_c - liquid_c_l) / liquid_c
        # get heavy isotope concentrations in atmosphere
        gas_c_h = gas_c - gas_c_l  # gas heavy isotope concentration
        # get exchange of the heavy isotope
        f_h = scale * a_u * (a_dg * gas_c_h * beta - Rt * a_db * gas_aq_c * 1e3)
        rv = f, f - f_h  # the corresponding flux of the light isotope

    return rv
