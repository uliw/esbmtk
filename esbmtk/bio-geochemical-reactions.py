from __future__ import annotations

import typing as tp
from math import log, sqrt

import numpy as np
from esbmtk import Q_
from esbmtk.utility_functions import (__addmissingdefaults__, __checkkeys__,
                                      __checktypes__)

if tp.TYPE_CHECKING:
    from esbmtk import Flux, Model, ReservoirGroup


def photosynthesis(
    o2,
    ta,
    dic,
    dic_l,
    productivity,  # actually P flux
    volume,
    PC_ratio,
    NC_ratio,
    O2C_ratio,
    PUE,
) -> tuple:
    """Calculate the effects of photosynthesis in the surface boxes"""
    """O2 in surface box as result of photosynthesis equals the primary
    productivity export flux of organic C times the O2:C ratio
    TA increases because of nitrate uptake during photosynthesis
    Note that DIC is currently handled in the model
    """
    # surface box dC/dt O2 and TA as result of photosynthesis
    dCdt_o = productivity * PUE * PC_ratio * O2C_ratio / volume
    dCdt_ta = productivity * PUE * PC_ratio * NC_ratio / volume

    dCdt_ta = 0

    return dCdt_o, dCdt_ta


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
    volume: float,  # intermediate box volume
    PC_ratio: float,
    NC_ratio: float,
    O2C_ratio: float,
    PUE: float,
) -> float:
    """Reservoirs can have multiple sources of OM with different
    remineralization efficiencies, e.g., low latidtude OM flux, vs
    high latitude OM flux.
    """
    p_flux = 0
    for i, f in enumerate(pp_flux):
        p_flux += f * remin_fraction[i]

    """ This function computes dC/dt, not dM/dt -> normalize Flux
     to Reservoir volume and calculate the amount of o2 required
     to oxidize all OM """
    o2_eq = p_flux * PUE * PC_ratio * O2C_ratio / volume
    # print(f"volume = {volume:.2e}")

    if o2 > o2_eq:  # box has enough oxygen
        d_o2 = -o2_eq  # consume O2
        # print(f"d_o2 = {d_o2 * volume:.2e}")
        # d_ta_ib = f_om * ib_remin * NC_ratio * -1
    else:  # box has enough oxygen
        d_o2 = -o2  # remove all available oxygen
        # d_ta_ib = f_om * ib_remin * NC_ratio * -1

    d_h2s = 0
    d_so4 = 0
    d_ta = 0

    return [
        d_ta,
        d_h2s,
        d_so4,
        d_o2,
    ]


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
        ],
    )
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
