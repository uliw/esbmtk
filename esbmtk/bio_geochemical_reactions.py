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
import numpy as np
from math import log, sqrt
from numba import jit, njit
from esbmtk import Q_, register_return_values, get_new_ratio_from_alpha
from .utility_functions import __checkkeys__, __addmissingdefaults__, __checktypes__

if tp.TYPE_CHECKING:
    from esbmtk import Flux, Model, ReservoirGroup

import functools


def debug(func):
    """Print the function signature and return value"""

    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [f"{float(repr(a)):.2e}\n" for a in args]  # 1
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]  # 2
        signature = ", ".join(args_repr + kwargs_repr)  # 3
        print(f"Calling {func.__name__}\n({signature})")
        value = func(*args, **kwargs)
        value = [f"{v:.2e}" for v in value]
        print(f"\n{func.__name__!r} returned\n {value!r}\n")  # 4
        breakpoint()
        return value

    return wrapper_debug


@njit()
def photosynthesis(
    o2,
    ta,
    dic,
    dic_l,
    po4,
    so4,
    h2s,
    productivity,  # actually P flux
    p: tuple[float],  # parameters
) -> tuple:
    """Calculate the effects of photosynthesis in the surface boxes"""
    """O2 in surface box as result of photosynthesis equals the primary
    productivity export flux of organic C times the O2:C ratio
    TA increases because of nitrate uptake during photosynthesis

    Note that this functions returns fluxes, so we need to calculate
    dMdt, not dCdt
    """

    # unpack parameters
    (
        volume,
        PC_ratio,
        NC_ratio,
        O2C_ratio,
        PUE,
        rain_rate,
        om_fractionation_factor,
        alpha,
    ) = p

    # POM formation
    dMdt_po4 = -productivity * PUE  # remove PO4 into POM
    POM_F = -dMdt_po4 * PC_ratio  # mass of newly formed POM
    r = get_new_ratio_from_alpha(dic, dic_l, om_fractionation_factor)
    POM_F_l = POM_F * r  # mass of POM_l
    dMdt_dic = -POM_F  # remove DIC by POM formation
    dMdt_dic_l = -POM_F_l
    dMdt_ta = POM_F * NC_ratio  # add TA from nitrate uptake into POM

    # CaCO3 formation
    alpha = 1
    PIC_F = POM_F * alpha / rain_rate  # newly formed CaCO3
    PIC_F_l = PIC_F * dic_l / dic
    dMdt_dic += -PIC_F  # dic removed
    dMdt_dic_l += -PIC_F
    dMdt_ta += 2 * -PIC_F  # TA removed

    # sulfur reactions, assuming that there is alwways enough O2
    dMdt_h2s = -h2s * volume  # H2S oxidation
    dMdt_so4 = dMdt_h2s  # add S to the sulfate pool
    dMdt_ta += 2 * dMdt_so4  # adjust Alkalinity
    # add O2 from photosynthesis - h2s oxidation
    dCdt_o = POM_F * O2C_ratio - 2 * h2s * volume

    return (  # note that these are returned as fluxes
        dCdt_o,
        dMdt_ta,
        dMdt_po4,
        dMdt_so4,
        dMdt_h2s,
        dMdt_dic,
        dMdt_dic_l,
        POM_F,
        POM_F_l,
        PIC_F,
        PIC_F_l,
    )


def init_photosynthesis(rg, productivity):
    """Setup photosynthesis instances"""
    from esbmtk import ExternalCode

    M = rg.mo
    ec = ExternalCode(
        name="ps",
        species=rg.mo.Oxygen.O2,
        fname="photosynthesis",
        ftype="cs2",  # cs1 is independent of fluxes, cs2 is not
        vr_datafields={"POM": 0.0, "POM_l": 0.0},
        function_input_data=[
            rg.O2,
            rg.TA,
            rg.DIC,
            rg.PO4,
            rg.SO4,
            rg.H2S,
            productivity,
        ],
        function_params=(
            rg.volume.magnitude,
            M.PC_ratio,
            M.NC_ratio,
            M.O2C_ratio,
            M.PUE,
            M.rain_rate,
            M.OM_frac / 1000.0 + 1.0,
            M.alpha,
        ),
        register=rg,
        return_values=[
            {"F_rg.O2": "photosynthesis"},
            {"F_rg.TA": "photosynthesis"},
            {"F_rg.PO4": "photosynthesis"},
            {"F_rg.SO4": "photosynthesis"},
            {"F_rg.H2S": "photosynthesis"},
            {"F_rg.DIC": "photosynthesis"},
            {"F_rg.POM": "photosynthesis"},
            {"F_rg.PIC": "photosynthesis"},
        ],
    )
    rg.mo.lpc_f.append(ec.fname)

    return ec


def add_photosynthesis(rgs: list[ReservoirGroup], p_fluxes: list[Flux | Q_]):
    """Add process to ReservoirGroup(s) in rgs. pfluxes must be list of Flux
    objects or float values that correspond to the rgs list
    """
    from esbmtk import register_return_values

    M = rgs[0].mo
    for i, rg in enumerate(rgs):
        if isinstance(p_fluxes[i], Q_):
            p_fluxes[i] = p_fluxes[i].to(M.f_unit).magnitude
            # print(f"rg = {rg.full_name}, f = {p_fluxes[i]}")
        else:
            pass
            # print(f"rg = {rg.full_name}, f = {p_fluxes[i].full_name}")
        ec = init_photosynthesis(rg, p_fluxes[i])
        register_return_values(ec, rg)
        rg.has_cs1 = True


@njit()
def remineralization(
    pom_fluxes: list,  # POM export fluxes
    pom_fluxes_l: list,  # POM_l export fluxes
    pic_fluxes: list,
    pic_fluxes_l: list,
    pom_remin_fractions: list,  # list of remineralization fractions
    pic_remin_fractions: float,
    h2s: float,  # concentration
    so4: float,  # concentration
    o2: float,  # o2 concentration
    po4: float,  # po4 concentration
    volume: float,  # box volume
    PC_ratio: float,
    NC_ratio: float,
    O2C_ratio: float,
    alpha: float,
    CaCO3_reactions=True,
    # burial: float,
) -> float:
    """Reservoirs can have multiple sources of POM with different
    remineralization efficiencies, e.g., low latidtude POM flux, vs
    high latitude POM flux. We only add the part that is remineralized.
    Note: The CaCO3 fluxes are handled below
    """
    pom_flux = 0
    pom_flux_l = 0
    # sum all POM and dic fluxes
    for i, f in enumerate(pom_fluxes):
        pom_flux += f * pom_remin_fractions[i]
        pom_flux_l += pom_fluxes_l[i] * pom_remin_fractions[i]

    # remove Alkalinity and add dic and po4 from POM remineralization
    # this happens irrespective of oxygen levels
    dMdt_po4 = pom_flux / PC_ratio  # return PO4
    dMdt_ta = -pom_flux * NC_ratio  # remove Alkalinity from NO3
    dMdt_dic = pom_flux  # add DIC from POM
    dMdt_dic_l = pom_flux_l
    m_h2s = h2s * volume

    m_o2 = o2 * volume
    # how much O2 is needed to oxidize all POM and H2S
    m_o2_eq = pom_flux * O2C_ratio + 2 * m_h2s

    if m_o2 > m_o2_eq:  # box has enough oxygen
        dMdt_o2 = -m_o2_eq  # consume O2
        dMdt_h2s = -m_h2s  # consume all h2s
        dMdt_so4 = -m_h2s  # add sulfate

    else:  # box has not enough oxygen
        dMdt_o2 = -m_o2  # remove all available oxygen
        # calculate how much POM is left to oxidize
        pom_flux = pom_flux - m_o2 / O2C_ratio
        # oxidize the remaining POM via sulfate reduction
        dMdt_so4 = -pom_flux / 2  # one SO4 oxidizes 2 carbon, and add 2 mol to TA
        dMdt_h2s = -dMdt_so4  # move S to reduced reservoir
        dMdt_ta += 2 * -dMdt_so4  # adjust Alkalinity for changes in sulfate

    if CaCO3_reactions:
        pic_flux = 0.0
        dic_flux_l = 0.0
        for i, f in enumerate(pic_fluxes):
            pic_flux += f * pic_remin_fractions[i] * alpha
            dic_flux_l += pic_fluxes_l[i] * pic_remin_fractions[i] * alpha

        # add Alkalinity and DIC from CaCO3 dissolution. Note that
        dMdt_dic += pic_flux
        dMdt_dic_l += dic_flux_l
        dMdt_ta += 2 * pic_flux

    # note, these are returned as fluxes
    return [dMdt_dic, dMdt_dic_l, dMdt_ta, dMdt_h2s, dMdt_so4, dMdt_o2, dMdt_po4]


def init_remineralization(
    rg: ReservoirGroup,
    pom_fluxes: list[Flux],
    pom_fluxes_l: list[Flux],
    pic_fluxes: list[Flux],
    pic_fluxes_l: list[Flux],
    pom_remin_fractions: float | list[float],
    pic_remin_fractions: float | list[float],
    CaCO3_reactions: bool,
):
    """ """
    from esbmtk import ExternalCode

    if not isinstance(pom_remin_fractions, list):
        pom_remin_fractions = list(pom_remin_fractions)
    if not isinstance(pic_remin_fractions, list):
        pic_remin_fractions = list(pic_remin_fractions)

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
            pom_fluxes,
            pom_fluxes_l,
            pic_fluxes,
            pic_fluxes_l,
            pom_remin_fractions,
            pic_remin_fractions,
            rg.H2S,
            rg.SO4,
            rg.O2,
            rg.PO4,
            rg.volume.magnitude,
            M.PC_ratio,
            M.NC_ratio,
            M.O2C_ratio,
            M.alpha,
            CaCO3_reactions,
        ],
        register=rg,
        return_values=[
            {"F_rg.DIC": "remineralization"},
            {"F_rg.TA": "remineralization"},
            {"F_rg.H2S": "remineralization"},
            {"F_rg.SO4": "remineralization"},
            {"F_rg.O2": "remineralization"},
            {"F_rg.PO4": "remineralization"},
        ],
    )
    rg.mo.lpc_f.append(ec.fname)
    return ec


def add_remineralization(M: Model, f_map: dict) -> None:
    """
    Add remineralization fluxes to the model.

    Parameters:
    M (Model): The model object t
    f_map (dict): A dictionary that maps sink names to source dictionaries. The
    source dictionary should contain the source species and a list of type
    and remineralization values. For example, {M.A_ib: {M.H_sb: ["POM", 0.3]}}.

    Raises:
    ValueError: If an invalid type is specified in the source dictionary.

    Returns:
    None
    """
    # get sink name (e.g., M.A_ib) and source dict e.g. {M.H_sb: {"POM": 0.3}}
    for sink, source_dict in f_map.items():
        pom_fluxes = list()
        pom_fluxes_l = list()
        pom_remin = list()
        pic_fluxes = list()
        pic_fluxes_l = list()
        pic_remin = list()

        # create flux lists for POM and possibly CaCO3
        for source, type_dict in source_dict.items():
            # get matching fluxes for e.g., M.A_sb, and POM
            if "POM" in type_dict:
                fl = M.flux_summary(
                    filter_by=f"photosynthesis {source.name} POM",
                    return_list=True,
                )
                for f in fl:
                    if f.name[-3:] == "F_l":
                        pom_fluxes_l.append(f)
                    else:
                        pom_fluxes.append(f)
                        pom_remin.append(type_dict["POM"])

            if "PIC" in type_dict:
                fl = M.flux_summary(
                    filter_by=f"photosynthesis {source.name} PIC",
                    return_list=True,
                )
                for f in fl:
                    if f.name[-3:] == "F_l":
                        pic_fluxes_l.append(f)
                    else:
                        pic_fluxes.append(f)
                        pic_remin.append(type_dict["PIC"])

        if len(pic_fluxes) > 0:
            ec = init_remineralization(
                sink,
                pom_fluxes,
                pom_fluxes_l,
                pic_fluxes,
                pic_fluxes_l,
                pom_remin,
                pic_remin,
                True,
            )
        else:
            ec = init_remineralization(
                sink,
                pom_fluxes,
                pom_fluxes_l,
                pom_fluxes,  # numba cannot deal with empty fluxes, so we
                pom_fluxes_l,  # add the om_fluxes, but ignore them
                pom_remin,
                pom_remin,
                False,
            )
        register_return_values(ec, sink)
        sink.has_cs2 = True


def carbonate_system_3(
    rg: ReservoirGroup,  # 2 Reservoir handle
    pic_f: float,  # 3 CaCO3 export flux as DIC
    dic_db: float,  # 4 DIC in the deep box
    dic_db_l: float,  # 4 DIC in the deep box
    ta_db: float,  # 5 TA in the deep box
    dic_sb: float,  # 6 [DIC] in the surface box
    dic_sb_l: float,  # 7 [DIC_l] in the surface box
    hplus_0: float,  # 8 hplus in the deep box at t-1
    zsnow: float,  # 9 snowline in meters below sealevel at t-1
    ksp0,
    kc,
    AD,
    zsat0,
    I_caco3,
    alpha,
    zsat_min,
    zmax,
    z0,
) -> tuple:
    """Calculates and returns the fraction of the carbonate rain that is
    dissolved an returned back into the ocean. This functions returns:

    DIC_burial, DIC_burial_l, Hplus, zsnow

    LIMITATIONS:
    - Assumes all concentrations are in mol/kg
    - Assumes your Model is in mol/kg ! Otherwise, DIC and TA updating will not
    be correct.

    Calculations are based off equations from:
    Boudreau et al., 2010, https://doi.org/10.1029/2009GB003654
    Follows, 2006, doi:10.1016/j.ocemod.2005.05.004
    """

    # Parameters
    k1 = rg.swc.K1  # K1
    k1k1 = rg.swc.K1K1
    k2 = rg.swc.K2  # K2
    k1k2 = rg.swc.K1K2  # K2
    KW = rg.swc.KW  # KW
    KB = rg.swc.KB  # KB
    ca2 = rg.swc.ca2  # Ca2+
    boron = rg.swc.boron  # boron
    zsat0 = int(abs(zsat0))
    zsat_min = int(abs(zsat_min))
    zmax = int(abs(zmax))
    z0 = int(abs(z0))
    depth_area_table = rg.cs.depth_area_table
    area_dz_table = rg.cs.area_dz_table
    Csat_table = rg.cs.Csat_table
    # calc carbonate alkalinity based t-1
    oh: float = KW / hplus_0
    boh4: float = boron * KB / (hplus_0 + KB)
    fg: float = hplus_0 - oh - boh4
    ca: float = ta_db + fg

    # calculate carbon speciation
    # The following equations are after Follows et al. 2006
    gamm: float = dic_db / ca
    dummy: float = (1 - gamm) * (1 - gamm) * k1k1 - k1k2 * (4 - 8 * gamm)
    hplus: float = 0.5 * ((gamm - 1) * k1 + sqrt(dummy))
    co3 = max(dic_db / (1 + hplus / k2 + hplus * hplus / k1k2), 3.7e-05)
    # ---------- compute critical depth intervals eq after  Boudreau (2010)
    # all depths will be positive to facilitate the use of lookup_tables
    zsat = int(zsat0 * log(ca2 * co3 / ksp0))
    zsat = np.clip(zsat, zsat_min, zmax)
    zcc = int(zsat0 * log(pic_f * ca2 / (ksp0 * AD * kc) + ca2 * co3 / ksp0))  # eq3
    zcc = np.clip(zcc, zsat_min, zmax)
    # get fractional areas
    B_AD = pic_f / AD
    A_z0_zsat = depth_area_table[z0] - depth_area_table[zsat]
    A_zsat_zcc = depth_area_table[zsat] - depth_area_table[zcc]
    A_zcc_zmax = depth_area_table[zcc] - depth_area_table[zmax]
    # ------------------------Calculate Burial Fluxes----------------------------- #
    BCC = A_zcc_zmax * B_AD  # CCD dissolution
    BNS = alpha * A_z0_zsat * B_AD  # water column dissolution
    diff_co3 = Csat_table[zsat:zcc] - co3
    area_p = area_dz_table[zsat:zcc]
    BDS_under = kc * area_p.dot(diff_co3)
    BDS_resp = alpha * (A_zsat_zcc * B_AD - BDS_under)
    BDS = BDS_under + BDS_resp  # respiration
    if zsnow > zmax:
        zsnow = zmax
    diff: np.ndarray = Csat_table[zcc : int(zsnow)] - co3
    area_p: np.ndarray = area_dz_table[zcc : int(zsnow)]
    # integrate saturation difference over area
    BPDC = kc * area_p.dot(diff)  # diss if zcc < zsnow
    BPDC = max(BPDC, 0)  # prevent negative values
    d_zsnow = -BPDC / (area_dz_table[int(zsnow)] * I_caco3)

    """CaCO3_export is the flux of particulate CaCO3 into the box.
    The fraction that is buried does not affect the water chemistry,
    so here we only consider the carbonate that is dissolved.
    The isotope ratio of the dissolution flux is determined by the delta
    value of the sediments we are dissolving, and the delta of the carbonate rain.
    The currrent code, assumes that both are the same.
    """
    # add dic and TA from dissolution
    dMdt_dic = BDS + BCC + BNS + BPDC
    dMdt_ta = 2 * dMdt_dic
    dMdt_dic_l = dMdt_dic * dic_sb_l / dic_sb
    dH = hplus - hplus_0

    return dMdt_dic, dMdt_dic_l, dMdt_ta, dH, d_zsnow


def init_carbonate_system_3(
    rg: ReservoirGroup,
    pic_export_flux: Flux,
    r_sb: ReservoirGroup,
    r_db: ReservoirGroup,
    area_table: np.ndarray,
    area_dz_table: np.ndarray,
    Csat_table: np.ndarray,
    AD: float,
    kwargs: dict,
):
    from esbmtk import ExternalCode, carbonate_system_3

    ec = ExternalCode(
        name="cs",
        species=rg.mo.Carbon.CO2,
        function=carbonate_system_3,
        fname="carbonate_system_3",
        ftype="cs2",
        r_s=r_sb,  # source (RG) of CaCO3 flux,
        r_d=r_db,  # sink (RG) of CaCO3 flux,
        vr_datafields={
            "depth_area_table": area_table,
            "area_dz_table": area_dz_table,
            "Csat_table": Csat_table,
        },
        function_input_data=[
            rg,  # 0
            pic_export_flux,  # 1
            r_db.DIC,  # 2
            r_db.TA,  # 3
            r_sb.DIC,  # 4
            "Hplus",  # 5
            "zsnow",  # 6
            kwargs["Ksp0"],  # 7
            float(kwargs["kc"]),  # 8
            float(AD),  # 9
            float(abs(kwargs["zsat0"])),  # 10
            float(kwargs["I_caco3"]),  # 11
            float(kwargs["alpha"]),  # 12
            float(abs(kwargs["zsat_min"])),  # 13
            float(abs(kwargs["zmax"])),  # 14
            float(abs(kwargs["z0"])),  # 15
        ],
        return_values=[
            {"F_rg.DIC": "db_remineralization"},
            {"F_rg.TA": "db_remineralization"},
            {"Hplus": rg.swc.hplus},
            {"zsnow": float(abs(kwargs["zsnow"]))},
        ],
        register=rg,
    )
    rg.mo.lpc_f.append(ec.fname)

    return ec


def add_carbonate_system_3(**kwargs) -> None:
    """Creates a new carbonate system virtual reservoir
    which will compute carbon species, saturation, compensation,
    and snowline depth, and compute the associated carbonate burial fluxes

    Required keywords:
        r_sb: list of ReservoirGroup objects in the surface layer
        r_db: list of ReservoirGroup objects in the deep layer
        pic_export_flux: list of flux objects which must match the
                                 list of ReservoirGroup objects.
        zsat_min = depth of the upper boundary of the deep box
        z0 = upper depth limit for carbonate burial calculations
             typically zsat_min

    Optional Parameters:

        zsat = initial saturation depth (m)
        zcc = initial carbon compensation depth (m)
        zsnow = initial snowline depth (m)
        zsat0 = characteristic depth (m)
        Ksp0 = solubility product of calcite at air-water interface (mol^2/kg^2)
        kc = heterogeneous rate constant/mass transfer coefficient for calcite dissolution (kg m^-2 yr^-1)
        Ca2 = calcium ion concentration (mol/kg)
        pc = characteristic pressure (atm)
        pg = seawater density multiplied by gravity due to acceleration (atm/m)
        I = dissolvable CaCO3 inventory
        co3 = CO3 concentration (mol/kg)
        Ksp = solubility product of calcite at in situ sea water conditions (mol^2/kg^2)

    """

    from esbmtk import Reservoir, init_carbonate_system_3

    # list of known keywords
    lkk: dict = {
        "r_db": list,  # list of deep reservoirs
        "r_sb": list,  # list of corresponding surface reservoirs
        "pic_export_flux": list,
        "AD": float,
        "zsat": int,
        "zsat_min": int,
        "zcc": int,
        "zsnow": int,
        "zsat0": int,
        "Ksp0": float,
        "kc": float,
        "Ca2": float,
        "pc": (float, int),
        "pg": (float, int),
        "I_caco3": (float, int),
        "alpha": float,
        "zmax": (float, int),
        "z0": (float, int),
        "Ksp": (float, int),
        # "BM": (float, int),
    }
    # provide a list of absolutely required keywords
    lrk: list[str] = [
        "r_db",
        "r_sb",
        "pic_export_flux",
        "zsat_min",
        "z0",
    ]

    # we need the reference to the Model in order to set some
    # default values.

    reservoir = kwargs["r_db"][0]
    model = reservoir.mo
    # list of default values if none provided
    lod: dict = {
        "r_sb": [],  # empty list
        "zsat": -3715,  # m
        "zcc": -4750,  # m
        "zsnow": -5000,  # m
        "zsat0": -5078,  # m
        "Ksp0": reservoir.swc.Ksp0,  # mol^2/kg^2
        "kc": 8.84 * 1000,  # m/yr converted to kg/(m^2 yr)
        "AD": model.hyp.area_dz(-200, -6000),
        "alpha": 0.6,  # 0.928771302395292, #0.75,
        "pg": 0.103,  # pressure in atm/m
        "pc": 511,  # characteristic pressure after Boudreau 2010
        "I_caco3": 529,  # dissolveable CaCO3 in mol/m^2
        "zmax": -6000,  # max model depth
        "Ksp": reservoir.swc.Ksp,  # mol^2/kg^2
    }

    # make sure all mandatory keywords are present
    __checkkeys__(lrk, lkk, kwargs)
    # add default values for keys which were not specified
    kwargs = __addmissingdefaults__(lod, kwargs)
    # test that all keyword values are of the correct type
    __checktypes__(lkk, kwargs)

    # establish some shared parameters
    # depths_table = np.arange(0, 6001, 1)
    depths: np.ndarray = np.arange(0, 6002, 1, dtype=float)
    r_db = kwargs["r_db"]
    r_sb = kwargs["r_sb"]
    ca2 = r_db[0].swc.ca2
    pg = kwargs["pg"]
    pc = kwargs["pc"]
    z0 = kwargs["z0"]
    Ksp0 = kwargs["Ksp0"]
    # C saturation(z) after Boudreau 2010
    Csat_table: np.ndarray = (Ksp0 / ca2) * np.exp((depths * pg) / pc)
    area_table = model.hyp.get_lookup_table(0, -6002)  # area in m^2(z)
    area_dz_table = model.hyp.get_lookup_table_area_dz(0, -6002) * -1  # area'
    AD = model.hyp.area_dz(z0, -6000)  # Total Ocean Area

    for i, rg in enumerate(r_db):  # Setup the virtual reservoirs
        ec = init_carbonate_system_3(
            rg,
            kwargs["pic_export_flux"][i],
            r_sb[i],
            r_db[i],
            area_table,
            area_dz_table,
            Csat_table,
            AD,
            kwargs,
        )

        register_return_values(ec, rg)
        rg.has_cs2 = True
