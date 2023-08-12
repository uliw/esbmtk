"""
esbmtk: A general purpose Earth Science box model toolkit Copyright
(C), 2020 Ulrich G.  Wortmann

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at
your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
from __future__ import annotations

import typing as tp
from esbmtk import Q_

if tp.TYPE_CHECKING:
    from esbmtk import Flux, Model, ReservoirGroup


def photosynthesis(
    o2,
    ta,
    dic,
    dic_l,
    po4,
    productivity,  # actually P flux
    volume,
    PC_ratio,
    NC_ratio,
    O2C_ratio,
    PUE,
    # rain_rate,
    # alpha,
) -> tuple:
    """Calculate the effects of photosynthesis in the surface boxes"""
    """O2 in surface box as result of photosynthesis equals the primary
    productivity export flux of organic C times the O2:C ratio
    TA increases because of nitrate uptake during photosynthesis
    Carbonate production/dissolution and burial are currently handled
    in the model definition
    """
    # OM formation
    # remove PO4 into OM
    dCdt_po4 = -productivity * PUE / volume
    # add O2 and Alkalinity from OM formation
    dCdt_o = -dCdt_po4 * PC_ratio * O2C_ratio
    dCdt_ta = -dCdt_po4 * PC_ratio * NC_ratio
    # remove DIC by OM formation
    # dCdt_dic =  dCdt_po4  *  PC_ratio
    # dCdt_dic_l =  get_li_frac(dCdt_dic,....

    # CaCO3 formation removes TA and DIC
    # dCdt_dic =  dCdt_po4  *  PC_ratio / rain_rate
    # dCdt_dic_l =  get_li_frac(dCdt_dic,....
    # dCdt_ta -=  2 * dcdt_dic

    return dCdt_o, dCdt_ta, dCdt_po4


def init_photosynthesis(surface, productivity):
    """Setup photosynthesis instances"""
    from esbmtk import ExternalCode

    M = surface.mo

    ec = ExternalCode(
        name="ps",
        species=surface.mo.Oxygen.O2,
        fname="photosynthesis",
        ftype="cs2",  # cs1 is independent of fluxes, cs2 is not
        function_input_data=[
            surface.O2,
            surface.TA,
            surface.DIC,
            surface.PO4,
            productivity,
            surface.volume.magnitude,
            M.PC_ratio,
            M.NC_ratio,
            M.O2C_ratio,
            M.PUE,
        ],
        register=surface,
        return_values=[
            surface.O2,
            surface.TA,
            surface.PO4,
            # surface.DIC,
        ],
    )

    surface.mo.lpc_f.append(ec.fname)

    return ec


def add_photosynthesis(rgs: list[ReservoirGroup], p_fluxes: list[Flux | Q_]):
    """Add process to ReservoirGroup(s) in rgs. pfluxes must be list of Flux
    objects or float values that correspond to the rgs list
    """
    from esbmtk import register_return_values

    M = rgs[0].mo
    for i, r in enumerate(rgs):
        if isinstance(p_fluxes[i], Q_):
            p_fluxes[i] = p_fluxes[i].to(M.f_unit).magnitude

        ec = init_photosynthesis(r, p_fluxes[i])
        register_return_values(ec, r)
        r.has_cs1 = True


def remineralization(
    pp_flux: list,  # export productivity P flux(es)
    remin_fraction: list,  # list of remineralization fractions
    h2s: float,  # concentration
    so4: float,  # concentration
    o2: float,  # o2 concentration in intermediate box
    po4: float,
    volume: float,  # intermediate box volume
    PC_ratio: float,
    NC_ratio: float,
    O2C_ratio: float,
    PUE: float,
) -> float:
    """Reservoirs can have multiple sources of OM with different
    remineralization efficiencies, e.g., low latidtude OM flux, vs
    high latitude OM flux.

    Carbonate dissolution is handled by carbonate_system 2
    """
    p_flux = 0
    for i, f in enumerate(pp_flux):
        p_flux += f * remin_fraction[i]

    # add PO4 from OM remineralization
    dCdt_po4 = p_flux * PUE / volume
    # remove Alkalinity from OM remineralization
    dCdt_ta = -dCdt_po4 * PC_ratio * NC_ratio
    total_OM = dCdt_po4 * PC_ratio
    # how much O2 is needed to oxidize all OM
    o2_eq = total_OM * O2C_ratio

    if o2 > o2_eq:  # box has enough oxygen
        dCdt_o2 = -o2_eq  # consume O2
        # print(f"dCdt_o2 = {dCdt_o2 * volume:.2e}")
        # dCdt_ta_ib = f_om * ib_remin * NC_ratio * -1
        dCdt_h2s = 0
        dCdt_so4 = 0
    else:  # box has not enough oxygen
        dCdt_o2 = -o2  # remove all available oxygen
        # calculate how much OM is left to oxidize
        remainig_OM = total_OM - -o2 / O2C_ratio
        # oxidize the remaining OM via sulfate reduction
        # one SO4 oxidizes 2 carbon, and add 2 mol to TA
        dCdt_so4 = -remainig_OM / 2
        dCdt_h2s = -dCdt_so4
        dCdt_ta = -dCdt_so4 * 2

    return [dCdt_ta, dCdt_h2s, dCdt_so4, dCdt_o2, dCdt_po4]


def init_remineralization(
    rg: ReservoirGroup,
    pp_fluxes: list[Flux],
    remin_fractions: list[float],
):
    """ """
    from esbmtk import ExternalCode

    M = rg.mo
    ec = ExternalCode(
        name="rm",
        species=rg.mo.Carbon.CO2,
        function=remineralization,
        fname="remineralization",
        ftype="cs2",  # cs1 is independent of fluxes, cs2 is not
        # hplus is not used but needed in post processing
        vr_datafields={"Hplus": rg.swc.hplus},
        function_input_data=[
            pp_fluxes,
            remin_fractions,
            rg.H2S,
            rg.SO4,
            rg.O2,
            rg.PO4,
            rg.volume.magnitude,
            M.PC_ratio,
            M.NC_ratio,
            M.O2C_ratio,
            M.PUE,
        ],
        register=rg,
        return_values=[
            rg.TA,
            rg.H2S,
            rg.SO4,
            rg.O2,
            rg.PO4,
        ],
    )
    # only neede
    rg.mo.lpc_f.append(ec.fname)

    return ec


def add_remineralization(M: Model, r_dict: dict) -> None:
    """ """
    from esbmtk import register_return_values

    for rg, d in r_dict.items():
        fluxes = list()
        remin = list()
        for r, v in d.items():
            fluxes.append(v[0])
            remin.append(v[1])

        # print(f"{rg.full_name}, {fluxes}, {remin}")
        ec = init_remineralization(rg, fluxes, remin)
        register_return_values(ec, r)
        r.has_cs2 = True
