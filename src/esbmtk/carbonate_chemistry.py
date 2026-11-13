"""esbmtk: A general purpose Earth Science box model toolkit.

Copyright (C), 2020-2021 Ulrich G. Wortmann

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

import typing as tp

# from functools import lru_cache
from math import log, sqrt

from numba import njit

import numpy as np
import numpy.typing as npt

from esbmtk.base_classes import Flux
from esbmtk.extended_classes import ExternalCode, Reservoir
from esbmtk.utility_functions import (
    __addmissingdefaults__,
    __checkkeys__,
    __checktypes__,
    register_return_values,
)

if tp.TYPE_CHECKING:
    pass

# declare numpy types
NDArrayFloat = npt.NDArray[np.float64]


class CarbonateSystem2Error(Exception):
    """Custom Error Class for Model-related errors."""

    def __init__(self, message):
        """Initialize Error Instance with formatted message."""
        message = f"\n\n{message}\n"
        super().__init__(message)


"""
Carbonate System 1 setup requires 3 steps: First we define the actual function,
carbonate_system_1_ode().  In the second step we create a wrapper
init_carbonate_system_1() that defines how to integrate this function into
esbmtk.  In the third step we create a functiom that uses
init_carbonate_system_1() to associates cs1 instances with the respective
reservoirs.

The process for cs2 is analogous
"""


@njit(fastmath=False, cache=True)
def get_hplus(dic, ta, h0, boron, K1, K1K2, KW, KB) -> float:
    """Calculate H+ concentration based on a previous estimate.

    [H+]. After Follows et al. 2006,
    doi:10.1016/j.ocemod.2005.05.004

    :param dic: DIC in mol/kg
    :param ta: TA in mol/kg
    :param h0: initial guess for H+ mol/kg
    :param boron: boron concentration
    :param K1: Ksp1
    :param K1K2: Ksp1 * Ksp2
    :param KW: K_water
    :param KB: K_boron

    :returns H: new H+ concentration in mol/kg
    """
    oh = KW / h0
    boh4 = boron * KB / (h0 + KB)
    fg = h0 - boh4 - oh
    cag = ta + fg
    gamm = dic / cag
    dummy = (1 - gamm) ** 2 * K1**2 - 4.0 * K1K2 * (1.0 - 2.0 * gamm)

    return 0.5 * ((gamm - 1.0) * K1 + sqrt(dummy))


@njit(fastmath=False, cache=True)
def carbonate_system_1(dic, ta, hplus_0, co2aq_0, p) -> tuple:
    """Return the H+ and carbonate alkalinity concentrations.

    :param dic: float with the dic concentration
    :param ta: float with the ta concentration
    :param hplus_0: float with the H+ concentration
    :param co2aq_0: float with the [CO2]aq concentration
    :param p: tuple with the parameter list
    :returns:  dCdt_Hplus, dCdt_co2aq

    LIMITATIONS:
    - Assumes all concentrations are in mol/kg
    - Assumes your Model is in mol/kg ! Otherwise, DIC and TA updating will not
    be correct.

    Calculations are based off equations from:
    Boudreau et al., 2010, https://doi.org/10.1029/2009GB003654
    Follows, 2006, doi:10.1016/j.ocemod.2005.05.004
    """
    k1, k2, k1k2, KW, KB, boron, _isotopes = p
    if isinstance(dic, tuple):  # isotopes: dic = (x1, x2)
        dic = dic[0]

    hplus = get_hplus(dic, ta, hplus_0, boron, k1, k1k2, KW, KB)
    co2aq = dic / (1 + k1 / hplus + k1k2 / hplus**2)
    dCdt_Hplus = hplus - hplus_0
    dCdt_co2aq = co2aq - co2aq_0

    return dCdt_Hplus, dCdt_co2aq


def init_carbonate_system_1(rg: Reservoir):
    """Create a new carbonate system virtual reservoir.

    for each reservoir in rgs. Note that rgs must be a list of reservoir groups.

    Required keywords:
        rgs: tp.List = []  of Reservoir Group objects

    These new virtual reservoirs are registered to their respective Species
    as 'cs'.

    The respective data fields are available as rgs.r.cs.xxx where xxx stands
    for a given key key in the  vr_datafields dictionary (i.e., H, CA, etc.)

    """
    from esbmtk import ExternalCode

    p = (
        rg.swc.K1,
        rg.swc.K2,
        rg.swc.K1K2,
        rg.swc.KW,
        rg.swc.KB,
        rg.swc.boron,
        rg.DIC.isotopes,
    )
    ec = ExternalCode(
        name="cs",
        species=rg.mo.Carbon.CO2,
        function=carbonate_system_1,
        fname="carbonate_system_1",
        ftype="std",
        function_input_data=[rg.DIC, rg.TA, "Hplus", "CO2aq"],
        function_params=p,
        register=rg,
        return_values=[
            {f"R_{rg.full_name}.Hplus": rg.swc.hplus},
            {f"R_{rg.full_name}.CO2aq": rg.swc.co2aq},
        ],
    )
    # rg.mo.lpc_f.append(ec.fname)
    rg.mo.lpc_f.append(ec.fname)  # list of function to be imported in ode backend

    return ec


def add_carbonate_system_1(rgs: list):
    """Create a new carbonate system virtual reservoir.

    For each reservoir in rgs. Note that rgs must be a list of reservoir groups.

    Required keywords:
        rgs: tp.List = []  of Reservoir Group objects

    These new virtual reservoirs are registered to their respective Species
    as 'cs'.

    The respective data fields are available as rgs.r.cs.xxx where xxx stands
    for a given key key in the  vr_datafields dictionary (i.e., H, CA, etc.)
    """
    for rg in rgs:
        if hasattr(rg, "DIC") and hasattr(rg, "TA"):
            ec = init_carbonate_system_1(rg)
            register_return_values(ec, rg)
            rg.has_cs1 = True
        else:
            raise AttributeError(f"{rg.full_name} must have a TA and DIC reservoir")


@njit(fastmath=False, cache=True)
def get_zsat(zsat0, zsat_min, zmax, ca2, co3, ksp0):
    """Calcualte zsat."""

    if co3 <= 0 or ca2 <= 0:
        return zsat_min
        
    zsat = int(zsat0 * log(ca2 * co3 / ksp0))
    return min(zmax, max(zsat_min, zsat))




@njit(fastmath=False, cache=True)
def get_zcc(export, zmax, zsat_min, zsat0, ca2, ksp0, AD, kc, co3):
    """Calculate zcc."""
    export = abs(export)
    zcc = int(zsat0 * log(export * ca2 / (ksp0 * AD * kc) + ca2 * co3 / ksp0))  # eq3

    return int(min(zmax, max(zsat_min, zcc)))


# @njit(fastmath=True)
# @cached(
#     cache=LRUCache(maxsize=128),
#     key=lambda CaCO3_export, dic_t_db, ta_db, dic_t_sb, hplus_0, zsnow, p: hashkey(
#         int(CaCO3_export),
#         round(dic_t_db, 5),
#         round(ta_db, 5),
#         round(dic_t_sb, 5),
#         hplus_0,
#         int(zsnow),
#     ),
# )
@njit(fastmath=False, cache=True)
def carbonate_system_2(
    CaCO3_export: float,  # 3 CaCO3 export flux as DIC
    dic_t_db: float | tuple,  # 4 DIC in the deep box
    ta_db: float,  # 5 TA in the deep box
    dic_t_sb: float | tuple,  # 6 [DIC] in the surface box
    hplus_0: float,  # 8 hplus in the deep box at t-1
    zsnow: float,  # 9 snowline in meters below sealevel at t-1
    p: tuple,
) -> tuple:
    """Return the fraction of the carbonate rain that is dissolved.

    This functions returns:

    DIC_burial, DIC_burial_l, Hplus, zsnow

    LIMITATIONS:
    - Assumes all concentrations are in mol/kg
    - Assumes your Model is in mol/kg

    Calculations are based off equations from:
    Boudreau et al., 2010, https://doi.org/10.1029/2009GB003654

    """
    sp, cp, area_table, area_dz_table, Csat_table = p
    ksp0, kc, AD, zsat0, I_caco3, alpha, zsat_min, zmax, z0 = cp
    k1, k2, k1k2, KW, KB, ca2, boron, _isotopes = sp

    # defined up front, numba cannot handle one-sided phi nodes
    dic_db_l = 0.0
    dic_sb_l = 0.0
    if isinstance(dic_t_db, tuple):
        dic_db, dic_db_l = dic_t_db
        dic_sb, dic_sb_l = dic_t_sb
    else:
        dic_db = dic_t_db
        dic_sb = dic_t_sb

    hplus = get_hplus(dic_db, ta_db, hplus_0, boron, k1, k1k2, KW, KB)

    """FIXME (2026-11-12): the equation for co3 originally was:
        >> co3 = max(dic_db / (1 + hplus / k2 + hplus**2 / k1k2), 3.7e-05)

    The 3.7e-05 mol/kg co3 value was specific to modern seawater chemistry. 
    This hardcoded constant value caused nonmodern models (e.g. PETM) with different 
    seawater chemistry to crash. I've temporarily removed it, but this is probably 
    not the best idea. It should instead be an appropriate variable. 
    """
    co3 = dic_db / (1 + hplus / k2 + hplus**2 / k1k2) 

    """ --- Compute critical depth intervals eqs after  Boudreau (2010) ---
   All depths will be positive to facilitate the use of lookup_tables.
   Note that these tables are different than the hyspometry data tables
   that expect positive and negative numbers.
    """

    zsat = get_zsat(zsat0, zsat_min, zmax, ca2, co3, ksp0)
    zcc = get_zcc(CaCO3_export, zmax, zsat_min, zsat0, ca2, ksp0, AD, kc, co3)
    B_AD = CaCO3_export / AD  # get fractional areas
    A_z0_zsat = area_table[z0] - area_table[zsat]
    A_zsat_zcc = area_table[zsat] - area_table[zcc]
    A_zcc_zmax = area_table[zcc] - area_table[zmax]
    # ------------------------Calculate Burial Fluxes----------------------------- #
    BCC = A_zcc_zmax * B_AD
    BNS = alpha * A_z0_zsat * B_AD
    diff_co3 = Csat_table[zsat:zcc] - co3
    """FIXME: Can area_sat_cc = area_dz_table[zsat:zcc]
    be rewritten as:
    
    area_sat_cc = area[zsat] - area[zcc]
    
    which would allow for a vectorized expression?
    But that also would have to work for diff_co3 as well.

    If yes, we can use Einstein summation notation
    
   result = np.einsum('ij,ij->i', U, V)
   
    `ij,ij->i` means: Multiply element-wise along each row, then sum per row.
    for  kc * area_sat_cc.dot(diff_co3)

    This would also be helpful for the post-processing step
    """
    area_sat_cc = area_dz_table[zsat:zcc]
    BDS_under = kc * area_sat_cc.dot(diff_co3)
    BDS_resp = alpha * (A_zsat_zcc * B_AD - BDS_under)
    BDS = BDS_under + BDS_resp

    # sediment dissolution if zcc is deeper than snowline
    if zsnow <= zcc:  # reset zsnow
        dzdt_zsnow = abs(zsnow - zcc)
        BPDC = 0
        zsnow = zcc
    else:  # integrate saturation difference over area
        if zsnow > zmax:  # limit zsnow to ocean depth
            zsnow = zmax

        diff: NDArrayFloat = Csat_table[zcc : int(zsnow)] - co3
        area_cc_snow: NDArrayFloat = area_dz_table[zcc : int(zsnow)]
        BPDC = max(0, kc * area_cc_snow.dot(diff))
        dzdt_zsnow = -BPDC / (area_dz_table[int(zsnow)] * I_caco3)

    """ CACO3_export is the flux of CaCO3 into the box.
    Boudreau's orginal approach is as follows.
    CACO3_export = B_diss + Fburial
    However, the model should use the bypass option and leave all flux
    calculations to the carbonate_system code. As such, ignore the burial flux
    (since it was never added), and only add the fraction of the input flux
    that dissolves back into the box
    """
    F_diss = BDS + BCC + BNS + BPDC
    dCdt_Hplus = hplus - hplus_0

    """ The isotope ratio of the dissolution flux is determined by the delta
    value of the sediments we are dissolving, and the delta of the carbonate rain.
    The currrent code, assumes that both are the same.
    """
    if isinstance(dic_t_db, tuple):
        F_diss_l = F_diss * dic_sb_l / dic_sb
        rv = (F_diss, F_diss_l, F_diss * 2, dCdt_Hplus, dzdt_zsnow)
    else:
        rv = (F_diss, F_diss * 2, dCdt_Hplus, dzdt_zsnow)

    return rv


def init_carbonate_system_2(
    export_flux: Flux,
    source_box: Reservoir,  # Surface box
    this_box: Reservoir,  # deep box
    kwargs: dict,
):
    """Initialize a carbonate system 2 instance.

    Note that the current implmentation assumes that the export flux is
    the total export flux over surface area of the mixed layer, i.e.,
    the sediment area between z0 and zmax

    Parameters
    ----------
    export_flux : Flux
        CaCO3 export flux from the surface box
    source_box : Reservoir
        Reservoir instance of the surface box
    this_box : Reservoir
        Reservoir instance of the deep box
    kwargs : dict
        dictionary of keyword value pairs


    """
    # Area between z0 and zmax
    AD = source_box.mo.hyp.area_dz(kwargs["z0"], kwargs["zmax"])
    swc = this_box.swc  # shorthand for seawater constants
    swc_p = (  # seawater parameters as tuple
        swc.K1,
        swc.K2,
        swc.K1K2,
        swc.KW,
        swc.KB,
        swc.ca2,
        swc.boron,
        source_box.DIC.isotopes,
    )
    cp = (  # other constants
        kwargs["Ksp0"],  # 7
        float(kwargs["kc"]),  # 8
        AD,  # 9
        int(abs(kwargs["zsat0"])),  # 10
        kwargs["I_caco3"],  # 11
        kwargs["alpha"],  # 12
        int(abs(kwargs["zsat_min"])),  # 13
        int(abs(kwargs["zmax"])),  # 14
        int(abs(kwargs["z0"])),  # 15
    )

    # initialize an external code instance
    ec = ExternalCode(
        name="cs2",
        species=source_box.mo.Carbon.CO2,
        function=carbonate_system_2,
        fname="carbonate_system_2",
        isotopes=source_box.DIC.isotopes,
        r_s=source_box,  # source (RG) of CaCO3 flux,
        r_d=this_box,  # sink (RG) of CaCO3 flux,
        function_input_data=[  # variable input data
            export_flux,  # 1
            this_box.DIC,  # 2
            this_box.TA,  # 3
            source_box.DIC,  # 4
            "Hplus",  # 5
            "zsnow",  # 6
        ],
        function_params=(  # constant input data
            swc_p,
            cp,
            this_box.mo.area_table,
            this_box.mo.area_dz_table,
            this_box.mo.Csat_table,
        ),
        return_values=[
            {f"F_{this_box.full_name}.DIC": "db_cs2"},
            {f"F_{this_box.full_name}.TA": "db_cs2"},
            {f"R_{this_box.full_name}.Hplus": this_box.swc.hplus},
            {f"R_{this_box.full_name}.zsnow": float(abs(kwargs["zsnow"]))},
        ],
        register=this_box,
    )
    this_box.mo.lpc_f.append(ec.fname)  # list of function to be imported in ode backend

    return ec


def add_carbonate_system_2(**kwargs) -> None:
    """Create a new carbonate system virtual reservoir.

    which will compute carbon species, saturation, compensation,
    and snowline depth, and compute the associated carbonate burial fluxes

    Required keywords:
        source_box: tp.List of Reservoir objects in the surface layer
        this_box: tp.List of Reservoir objects in the deep layer
        carbonate_export_fluxes: tp.List of flux objects which must match the
        list of Reservoir objects.
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
    # list of known keywords
    lkk: dict = {
        "this_box": list,  # list of deep reservoirs
        "source_box": list,  # list of corresponding surface reservoirs
        "r_db": list,  # list of deep reservoirs
        "r_sb": list,  # list of corresponding surface reservoirs
        "carbonate_export_fluxes": list,
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
        ["r_sb", "source_box"],
        ["r_db", "this_box"],
        "carbonate_export_fluxes",
        "z0",
    ]

    source_box = kwargs.get("source_box", kwargs.get("r_sb"))
    this_box = kwargs.get("this_box", kwargs.get("r_db"))

    # we need the reference to the Model in order to set some
    # default values.
    reservoir = this_box[0]
    model = reservoir.mo

    # list of default values if none provided
    lod: dict = {
        "source_box": [],  # empty list
        "zsat": -3715,  # m
        "zcc": -4750,  # m
        "zsnow": -4750,  # m
        "zsat0": -5078,  # m
        "Ksp0": reservoir.swc.Ksp0,  # mol^2/kg^2
        "kc": 8.84 * 1000,  # m/yr converted to kg/(m^2 yr)
        "alpha": 0.6,  # 0.928771302395292, #0.75,
        "pg": 0.103,  # pressure in atm/km
        "pc": 511,  # characteristic pressure after Boudreau 2010
        "I_caco3": 529,  # dissolveable CaCO3 in mol/m^2
        "zmax": -10999,  # max model depth
        "Ksp": reservoir.swc.Ksp_ca,  # mol^2/kg^2
    }
    __checkkeys__(lrk, lkk, kwargs)
    kwargs = __addmissingdefaults__(lod, kwargs)
    __checktypes__(lkk, kwargs)

    if "zsat_min" not in kwargs:
        kwargs["zsat_min"] = kwargs["z0"]

    if not isinstance(this_box, list):
        this_box = [this_box]

    if not isinstance(source_box, list):
        source_box = [source_box]

    if len(this_box) != len(source_box):
        raise CarbonateSystem2Error(
            f"The Number of surface boxes ({len(source_box)} does not match the number of deeb boxes ({len(this_box)}"
        )

    pg = kwargs["pg"]
    pc = kwargs["pc"]
    zmax = abs(int(kwargs["zmax"]))
    export_fluxes = kwargs["carbonate_export_fluxes"]
    n_flux = (
        int(len(export_fluxes) / 2) if this_box[0].DIC.isotopes else len(export_fluxes)
    )

    if len(this_box) != n_flux:
        raise CarbonateSystem2Error(
            f"The Number of deep boxes ({len(this_box)}) does not match the"
            f"number of export fluxes({n_flux})"
        )

    # test if corresponding surface reservoirs have been defined
    if len(source_box) == 0:
        raise CarbonateSystem2Error(
            "Please update your call to add_carbonate_system_2 and add\
            the list of corresponding surface reservoirs"
        )

    # check if we already have the hypsometry and saturation tables
    if not hasattr(model, "area_table"):
        depth_range = np.arange(0, zmax, 1, dtype=float)  # mbsl
        model.area_table = model.hyp.get_lookup_table_area()  # area in m^2(z)
        model.area_dz_table = model.hyp.get_lookup_table_area_dz() * -1  # area_dz
        model.Csat_table = (reservoir.swc.Ksp0 / reservoir.swc.ca2) * np.exp(
            (depth_range * pg) / pc
        )

    for i, rg in enumerate(this_box):  # Setup the virtual reservoirs
        if hasattr(rg, "DIC") and hasattr(rg, "TA"):
            rg.swc.update_parameters()
        else:
            raise AttributeError(f"{rg.full_name} must have a TA and DIC reservoir")

        export_flux = kwargs["carbonate_export_fluxes"][i]
        export_flux.serves_as_input = True  # flag this for ode backend
        ec = init_carbonate_system_2(
            export_flux,
            source_box[i],
            this_box[i],
            kwargs,
        )
        register_return_values(ec, rg)
        rg.has_cs2 = True


"""
Carbonate System 3:

Current vers. (last updated July 3 2025): adds a "next_box" into which 
undissolved CaCO3 (or "burial" fluxes) are explicitly transferred into 
another box. This was not explicitly handled in Carbonate System 2.

"""


@njit(fastmath=False, cache=True)
def carbonate_system_3(
    CaCO3_export: float,  # 3 CaCO3 export flux as DIC
    dic_t_db: float | tuple,  # 4 DIC in the deep box
    ta_db: float,  # 5 TA in the deep box
    dic_t_sb: float | tuple,  # 6 [DIC] in the surface box
    hplus_0: float,  # 8 hplus in the deep box at t-1
    zsnow: float,  # 9 snowline in meters below sealevel at t-1
    p: tuple,
) -> tuple:
    """Return the fraction of the carbonate rain that is dissolved.

    This functions returns:

    DIC_burial, DIC_burial_l, Hplus, zsnow

    LIMITATIONS:
    - Assumes all concentrations are in mol/kg
    - Assumes your Model is in mol/kg

    Calculations are based off equations from:
    Boudreau et al., 2010, https://doi.org/10.1029/2009GB003654

    """
    sp, cp, area_table, area_dz_table, Csat_table = p
    ksp0, kc, AD, zsat0, I_caco3, alpha, zsat_min, zmax, z0 = cp
    k1, k2, k1k2, KW, KB, ca2, boron, _isotopes = sp

    # defined up front, numba cannot handle one-sided phi nodes
    dic_db_l = 0.0
    dic_sb_l = 0.0
    if isinstance(dic_t_db, tuple):
        dic_db, dic_db_l = dic_t_db
        dic_sb, dic_sb_l = dic_t_sb
    else:
        dic_db = dic_t_db
        dic_sb = dic_t_sb

    hplus = get_hplus(dic_db, ta_db, max(hplus_0, 1e-12), boron, k1, k1k2, KW, KB)
    co3 = dic_db / (1 + hplus / k2 + hplus**2 / k1k2)

    """ --- Compute critical depth intervals eqs after  Boudreau (2010) ---
   All depths will be positive to facilitate the use of lookup_tables.
   Note that these tables are different than the hyspometry data tables
   that expect positive and negative numbers.
    """

    zsat = get_zsat(zsat0, zsat_min, zmax, ca2, co3, ksp0)
    zcc = get_zcc(CaCO3_export, zmax, zsat_min, zsat0, ca2, ksp0, AD, kc, co3)
    B_AD = CaCO3_export / AD  # get fractional areas
    A_z0_zsat = area_table[z0] - area_table[zsat]
    A_zsat_zcc = area_table[zsat] - area_table[zcc]
    A_zcc_zmax = area_table[zcc] - area_table[zmax]

    # ------------------------Calculate Burial Fluxes----------------------------- #
    BCC = A_zcc_zmax * B_AD
    BNS = alpha * A_z0_zsat * B_AD
    diff_co3 = Csat_table[zsat:zcc] - co3
    """FIXME: Can area_sat_cc = area_dz_table[zsat:zcc]
    be rewritten as:
    
    area_sat_cc = area[zsat] - area[zcc]
    
    which would allow for a vectorized expression?
    But that also would have to work for diff_co3 as well.

    If yes, we can use Einstein summation notation
    
   result = np.einsum('ij,ij->i', U, V)
   
    `ij,ij->i` means: Multiply element-wise along each row, then sum per row.
    for  kc * area_sat_cc.dot(diff_co3)

    This would also be helpful for the post-processing step
    """
    area_sat_cc = area_dz_table[zsat:zcc]
    BDS_under = kc * area_sat_cc.dot(diff_co3)
    BDS_resp = alpha * (A_zsat_zcc * B_AD - BDS_under)
    BDS = BDS_under + BDS_resp

    # sediment dissolution if zcc is deeper than snowline
    if zsnow <= zcc:  # reset zsnow
        dzdt_zsnow = abs(zsnow - zcc)
        BPDC = 0
        zsnow = zcc
    else:  # integrate saturation difference over area
        if zsnow > zmax:  # limit zsnow to ocean depth
            zsnow = zmax

        diff: NDArrayFloat = Csat_table[zcc : int(zsnow)] - co3
        area_cc_snow: NDArrayFloat = area_dz_table[zcc : int(zsnow)]
        BPDC = max(0, kc * area_cc_snow.dot(diff))
        dzdt_zsnow = -BPDC / (area_dz_table[int(zsnow)] * I_caco3)

    """ CACO3_export is the flux of CaCO3 into this_box (i.e. export from surface layer)
    Boudreau's orginal approach is as follows.
    CACO3_export = B_diss + Fburial
    However, the model should use the bypass option for the source_box --> this_box connection, 
    and leave all flux calculations to the carbonate_system code. Furthermore, the model does not 
    need to define a separate CaCO3 connection between this_box and next_box, as this connection 
    is handled entirely by the CS3 code.
    """
    F_diss = BDS + BCC + BNS + BPDC
    dCdt_Hplus = hplus - hplus_0

    F_exp = CaCO3_export - F_diss

    """ The isotope ratio of the dissolution flux is determined by the delta
    value of the sediments we are dissolving, and the delta of the carbonate rain.
    The currrent code assumes that both are the same.
    """
    if isinstance(dic_t_db, tuple):
        F_diss_l = F_diss * dic_sb_l / dic_sb
        F_exp_l = F_exp * dic_sb_l / dic_sb
        rv = (
            F_diss,
            F_diss_l,
            F_diss * 2,
            dCdt_Hplus,
            dzdt_zsnow,
            F_exp,
            F_exp_l,
            F_exp * 2,
        )
    else:
        rv = (
            F_diss,
            F_diss * 2,
            dCdt_Hplus,
            dzdt_zsnow,
            F_exp,
            F_exp * 2,
        )

    return rv


def init_carbonate_system_3(
    export_flux: Flux,
    source_box: Reservoir,  # Surface box
    this_box: Reservoir,  # intermediate box
    next_box: Reservoir, #deep box
    kwargs: dict,
):
    """Initialize a carbonate system 3 instance.

    Note that the current implmentation assumes that the export flux into
    this_box is the total export flux over surface area of the mixed layer,
    i.e., the sediment area between z0 and zmax

    Parameters
    ----------
    export_flux : Flux
        CaCO3 export flux from the surface box
    source_box : Reservoir
        Reservoir instance of the surface box
    this_box : Reservoir
        Reservoir instance of the deep box
    next_box :
        Reservoir instance of the sink box
    kwargs : dict
        dictionary of keyword value pairs


    """
    # Area between z0 and zmax
    AD = source_box.mo.hyp.area_dz(kwargs["z0"], kwargs["zmax"])
    swc = this_box.swc  # shorthand for seawater constants
    swc_p = (  # seawater parameters as tuple
        swc.K1,
        swc.K2,
        swc.K1K2,
        swc.KW,
        swc.KB,
        swc.ca2,
        swc.boron,
        source_box.DIC.isotopes,
    )
    cp = (  # other constants
        kwargs["Ksp0"],  # 7
        float(kwargs["kc"]),  # 8
        AD,  # 9
        int(abs(kwargs["zsat0"])),  # 10
        kwargs["I_caco3"],  # 11
        kwargs["alpha"],  # 12
        int(abs(kwargs["zsat_min"])),  # 13
        int(abs(kwargs["zmax"])),  # 14
        int(abs(kwargs["z0"])),  # 15
    )

    # initialize an external code instance
    ec = ExternalCode(
        name="cs3",
        species=source_box.mo.Carbon.CO2,
        function=carbonate_system_3,
        fname="carbonate_system_3",
        isotopes=source_box.DIC.isotopes,
        r_s=source_box,  # source (RG) of CaCO3 flux,
        r_d=this_box,  # sink (RG) of CaCO3 flux,
        r_n=next_box,  # sink (RG) of undissolved CaCO3
        function_input_data=[  # variable input data
            export_flux,  # 1
            this_box.DIC,  # 2
            this_box.TA,  # 3
            source_box.DIC,  # 4
            "Hplus",  # 5
            "zsnow",  # 6
        ],
        function_params=(  # constant input data
            swc_p,
            cp,
            this_box.mo.area_table,
            this_box.mo.area_dz_table,
            this_box.mo.Csat_table,
        ),
        return_values=[
            {f"F_{this_box.full_name}.DIC": "db_cs3"},
            {f"F_{this_box.full_name}.TA": "db_cs3"},
            {f"R_{this_box.full_name}.Hplus": this_box.swc.hplus},
            {f"R_{this_box.full_name}.zsnow": float(abs(kwargs["zsnow"]))},
            {f"F_{next_box.full_name}.DIC": "db_export_DIC"},
            {f"F_{next_box.full_name}.TA": "db_export_TA"},
        ],
        register=this_box,
    )
    this_box.mo.lpc_f.append(ec.fname)  # list of function to be imported in ode backend

    return ec


def add_carbonate_system_3(**kwargs) -> None:
    """Create a new carbonate system virtual reservoir.

    This function initializes carbonate system 3 (cs3) for each specified deep box.
    It computes saturation, compensation, and snowline depth, and the associated
    carbonate burial fluxes.

    Required keywords:
        r_sb / source_box: list of surface Reservoirs
        r_db / this_box: list of deep Reservoirs
        r_nb / next_box: list of sink Reservoirs
        carbonate_export_fluxes: list of CaCO3 export Flux objects
        z0: depth (m) for burial calculations

    Optional (defaulted) keywords:
        zsat, zcc, zsnow, zsat0, Ksp0, kc, alpha, pg, pc, I_caco3, zmax, Ksp
    """
    # list of known keywords
    lkk: dict = {
        "this_box": list,
        "source_box": list,
        "next_box": list,
        "r_db": list,
        "r_sb": list,
        "r_nb": list,
        "carbonate_export_fluxes": list,
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
    }

    # provide a list of absolutely required keywords:
    lrk: list = [
        ["r_sb", "source_box"],
        ["r_db", "this_box"],
        ["r_nb", "next_box"],
        "carbonate_export_fluxes",
        "z0",
    ]

    source_box = kwargs.get("source_box", kwargs.get("r_sb"))
    this_box = kwargs.get("this_box", kwargs.get("r_db"))
    next_box = kwargs.get("next_box", kwargs.get("r_nb"))
    carbonate_export_fluxes = kwargs.get("carbonate_export_fluxes")

    # we need the reference to the Model in order to set some default values

    reservoir = this_box[0]
    model = reservoir.mo

    # list of default values if none provided:
    lod: dict = {
        "source_box": [],
        "zsat": -3715,
        "zcc": -4750,
        "zsnow": -4750,
        "zsat0": -5078,
        "Ksp0": None,  # will be set later from reservoir.swc
        "kc": 8.84 * 1000,
        "alpha": 0.6,
        "pg": 0.103,
        "pc": 511,
        "I_caco3": 529,
        "zmax": -10999,
        "Ksp": None,
    }

    if lod["Ksp0"] is None:
        lod["Ksp0"] = reservoir.swc.Ksp0
    if lod["Ksp"] is None:
        lod["Ksp"] = reservoir.swc.Ksp_ca

    __checkkeys__(lrk, lkk, kwargs)
    kwargs = __addmissingdefaults__(lod, kwargs)
    __checktypes__(lkk, kwargs)

    if source_box is None or this_box is None or carbonate_export_fluxes is None:
        raise CarbonateSystem2Error(
            "Missing required inputs: source_box, this_box, or export_fluxes"
        )

    if "zsat_min" not in kwargs:
        kwargs["zsat_min"] = kwargs["z0"]

    if not isinstance(this_box, list):
        this_box = [this_box]

    if not isinstance(source_box, list):
        source_box = [source_box]

    if not isinstance(next_box, list):
        next_box = [next_box]

    if len(this_box) != len(source_box):
        raise CarbonateSystem2Error(
            f"Number of surface boxes ({len(source_box)}) does not match deep boxes ({len(this_box)})"
        )
    if len(next_box) != len(this_box):
        raise CarbonateSystem2Error(
            f"Number of next boxes ({len(next_box)}) does not match deep boxes ({len(this_box)})"
        )

    pg = kwargs["pg"]
    pc = kwargs["pc"]
    zmax = abs(int(kwargs["zmax"]))

    # check if we already have the hypsometry and saturation tables
    if not hasattr(model, "area_table"):
        depth_range = np.arange(0, zmax, 1, dtype=float)
        model.area_table = model.hyp.get_lookup_table_area()
        model.area_dz_table = model.hyp.get_lookup_table_area_dz() * -1
        model.Csat_table = (
            reservoir.swc.Ksp0
            / reservoir.swc.ca2
            * np.exp((depth_range * kwargs["pg"]) / kwargs["pc"])
        )

    # set up virtual reservoirs:
    for i, (sb, db, nb) in enumerate(zip(source_box, this_box, next_box)):
        if not (hasattr(db, "DIC") and hasattr(db, "TA")):
            raise AttributeError(f"{db.full_name} must have a DIC and TA reservoir")

        db.swc.update_parameters()

        export_flux = kwargs["carbonate_export_fluxes"][i]
        export_flux.serves_as_input = True  # flag this for ode backend
        ec = init_carbonate_system_3(
            export_flux,
            source_box[i],
            this_box[i],
            next_box[i],
            kwargs,
        )

        register_return_values(ec, db)
        db.has_cs3 = True

def carbonate_system_4(
    CaCO3_export: float,  # 3 CaCO3 export flux as DIC
    dic_t_db: float | tuple,  # 4 DIC in the deep box
    ta_db: float,  # 5 TA in the deep box
    dic_t_ib: float | tuple, #DIC in the intermediate box
    ta_ib, #TA in the intermediate box
    dic_t_sb: float | tuple,  # 6 [DIC] in the surface box
    hplus_db_0: float,  # 8 hplus in the deep box at t-1
    hplus_ib_0: float, # hplus in the intermediate box at t -1
    zsnow: float,  # 9 snowline in meters below sealevel at t-1
    p: tuple,
) -> tuple:
    """ Compute carbonate burial and dissolution fluxes for a three vertical layer ocean.

    This routine represents carbonate sediment dynamics for a system
    consisting of surface, intermediate, and deep ocean reservoirs coupled
    to a sediment reservoir. Carbonate dissolution, burial, snowline
    evolution, and alkalinity feedbacks are calculated following
    Boudreau et al. (2010).

    Parameters
    ----------
    CaCO3_export : float
        Export flux of CaCO₃ from the surface ocean.
    dic_t_db : float or tuple
        Deep-box DIC concentration. If isotopes are enabled, a tuple
        containing total and isotope-specific values.
    ta_db : float
        Deep-box total alkalinity.
    dic_t_ib : float or tuple
        Intermediate-box DIC concentration.
    ta_ib : float
        Intermediate-box total alkalinity.
    dic_t_sb : float or tuple
        Surface-box DIC concentration.
    hplus_db_0 : float
        Deep-box hydrogen ion concentration from the previous timestep.
    hplus_ib_0 : float
        Intermediate-box hydrogen ion concentration from the previous
        timestep.
    zsnow : float
        Snowline depth from the previous timestep.
    p : tuple
        Collection of constant model parameters and lookup tables:

        - seawater chemistry constants
        - carbonate-system parameters
        - area lookup table
        - area-per-depth lookup table
        - carbonate saturation lookup table

    Returns
    -------
    tuple
        Tuple containing:

        - Intermediate-box DIC dissolution flux
        - Intermediate-box TA dissolution flux
        - Deep-box hydrogen ion rate of change
        - Snowline migration rate
        - Deep-box DIC dissolution flux
        - Deep-box TA dissolution flux
        - Burial DIC flux
        - Burial TA flux

    Notes
    -----
    Assumptions:

    - All concentrations are expressed in mol kg⁻¹.
    - Carbonate chemistry follows the approximation of
      Follows et al. (2006).
    - Sediment dissolution and burial follow
      Boudreau et al. (2010).

    References
    ----------
    Boudreau, Bernard P., Jack J. Middelburg, Andreas F. Hofmann, 
    and Filip J. R. Meysman. 2010. “Ongoing Transients in Carbonate Compensation.” 
    Global Biogeochemical Cycles 24 (4). https://doi.org/10.1029/2009GB003654.
    """
    sp, cp, area_table, area_dz_table, Csat_table = p
    ksp0, kc, AD, zsat0, I_caco3, alpha, zsat_min, zmax, z0, zint = cp
    k1, k2, k1k2, KW, KB, ca2, boron, _isotopes = sp

    # isinstance() lets numba prune the dead branch at compile time
    # defined up front, numba cannot handle one-sided phi nodes
    dic_db_l = 0.0
    dic_sb_l = 0.0
    dic_ib_l = 0.0
    if isinstance(dic_t_db, tuple):
        dic_db, dic_db_l = dic_t_db
        dic_sb, dic_sb_l = dic_t_sb
        dic_ib, dic_ib_l = dic_t_ib
    else:
        dic_db = dic_t_db
        dic_sb = dic_t_sb
        dic_ib = dic_t_ib

    hplus = get_hplus(dic_db, ta_db, max(hplus_db_0, 1e-12), boron, k1, k1k2, KW, KB)
    hplus_ib = get_hplus(dic_ib, ta_ib, max(hplus_ib_0, 1e-12), boron, k1, k1k2, KW, KB)

    co3_deep = dic_db / (1 + hplus / k2 + hplus**2 / k1k2)
    co3_int = dic_ib / (1 + hplus_ib / k2 + hplus_ib**2 / k1k2)

    # taking a simple average of the CO3 of both boxes to calculate zsat 
    """FIXME: a simple avg is very crude. this should consider the vertical extent 
    of the intermediate and deep boxes actually.
    """
    co3 = 0.5 * (co3_int + co3_deep) 
    

    """ --- Compute critical depth intervals eqs after  Boudreau (2010) ---
   All depths will be positive to facilitate the use of lookup_tables.
   Note that these tables are different than the hyspometry data tables
   that expect positive and negative numbers.
    """

    zsat = get_zsat(zsat0, zsat_min, zmax, ca2, co3, ksp0)
    zcc = get_zcc(CaCO3_export, zmax, zsat_min, zsat0, ca2, ksp0, AD, kc, co3)

    # Fractional burial flux per area
    B_AD = CaCO3_export / AD

    A_zcc_zmax = area_table[zcc] - area_table[zmax]

    # BCC is always in deep box (unaffected by intermediate box)
    BCC = A_zcc_zmax * B_AD

    zsat_above_zint = zsat < zint

    # numba needs all branch-dependent variables defined up front
    BNS = 0.0
    BNS_int = 0.0
    BNS_deep = 0.0
    BDS = 0.0
    BDS_int = 0.0
    BDS_deep = 0.0

    """FIXME: The flux splitting scheme employed in carbonate_system_4 currently assumes 
    that the carbonate compensation depth (i.e. zcc) is always in the deep box. This assumption
    will obviously not hold true for every model / every box geometry. This limitation should 
    be addressed to allow CS4 to be implemented in a wider range of models.
    """
    # Calculating burial fluxes and diff_co3 
    if zsat_above_zint:
    # Surface to zsat to zint to zcc
    
        #define area tables:
        A_z0_zsat = area_table[z0] - area_table[zsat]
        A_zsat_zint = area_table[zsat] - area_table[zint]
        A_zint_zcc = area_table[zint] - area_table[zcc]

        BNS = alpha * A_z0_zsat * B_AD 
    
        #z_sat -> z_cc is split by z_int so BDS needs to be split too
        diff_co3_int = Csat_table[zsat:zint] - co3_int
        diff_co3_deep = Csat_table[zint:zcc] - co3_deep

        area_sat_int = area_dz_table[zsat:zint]
        area_int_cc = area_dz_table[zint:zcc]

        BDS_int_under = kc * area_sat_int.dot(diff_co3_int) 
        BDS_deep_under = kc * area_int_cc.dot(diff_co3_deep)

        BDS_int_resp = alpha * (A_zsat_zint * B_AD - BDS_int_under)
        BDS_deep_resp = alpha * (A_zint_zcc * B_AD - BDS_deep_under)

        BDS_int = BDS_int_under + BDS_int_resp
        BDS_deep = BDS_deep_under + BDS_deep_resp

    else:
    # Surface to zint to zsat to zcc
    
        #define area tables:
        A_z0_zint = area_table[z0] - area_table[zint]
        A_zint_zsat = area_table[zint] - area_table[zsat]
        A_zsat_zcc = area_table[zsat] - area_table[zcc]
    
        #z0 -> z_sat is split by z_int so BNS needs to be split also
        BNS_int = alpha * A_z0_zint * B_AD
        BNS_deep = alpha * A_zint_zsat * B_AD

        diff_co3 = Csat_table[zsat:zcc] - co3_deep

        area_sat_cc = area_dz_table[zsat:zcc]

        BDS_under = kc * area_sat_cc.dot(diff_co3)
        BDS_resp = alpha * (A_zsat_zcc * B_AD - BDS_under)

        BDS = BDS_under + BDS_resp

# Sediment dissolution (BPDC) if snowline is deeper than CCD
#always in deep box
    if zsnow <= zcc:
        dzdt_zsnow = abs(zsnow - zcc)
        BPDC = 0.0
        zsnow = zcc  # reset
    else:
        zsnow = min(zsnow, zmax)  # limit to ocean bottom
        diff = Csat_table[zcc:int(zsnow)] - co3_deep #FIXME: probably need to consider H+ changes in the int-box as well
        area_cc_snow = area_dz_table[zcc:int(zsnow)]
        BPDC = max(0.0, kc * np.dot(area_cc_snow, diff))
        dzdt_zsnow = -BPDC / (area_dz_table[int(zsnow)] * I_caco3)


# H+ concentration rate change 
#does this need modification re: intermediate box?
    dCdt_Hplus = hplus - hplus_db_0

#F_diss_int and F_diss_deep are going to be connected to different boxes:
    """FIXME: need to implement a way to incorporate intermediate box burial fluxes 
    within CS4 as well. At present intermediate box burial has to be handled separately.
    """
    if zsat < zint:
        F_diss_int = BNS + BDS_int
        F_diss_deep = BDS_deep + BCC + BPDC
    else:
        F_diss_int = BNS_int
        F_diss_deep = BNS_deep + BDS + BCC + BPDC

    F_burial = CaCO3_export - F_diss_int - F_diss_deep 

    if isinstance(dic_t_db, tuple):
        F_diss_int_l = F_diss_int * dic_sb_l / dic_sb
        F_diss_deep_l = F_diss_deep * dic_ib_l / dic_ib

        F_burial_l = F_burial * dic_db_l / dic_db
    
        rv = (F_diss_int, F_diss_int_l, F_diss_int * 2, dCdt_Hplus, dzdt_zsnow, F_diss_deep, F_diss_deep_l, F_diss_deep *2, F_burial, F_burial_l, F_burial*2)

    else:
        rv = (F_diss_int, F_diss_int * 2, dCdt_Hplus, dzdt_zsnow, F_diss_deep, F_diss_deep *2, F_burial, F_burial*2)
    
    return rv


def init_carbonate_system_4(
    export_flux: Flux,
    source_box: Reservoir,  # Surface box
    this_box: Reservoir,  # currently intermediate box
    next_box: Reservoir, #currently deep box
    burial_box: Reservoir,
    kwargs: dict,
):
    """Initialize a carbonate_system_4 external-code instance.

    Creates and registers an :class:`ExternalCode` object that evaluates
    carbonate dissolution, burial, and snowline dynamics for a coupled
    intermediate-deep ocean system.

    Parameters
    ----------
    export_flux : Flux
        CaCO₃ export flux from the surface reservoir.
    source_box : Reservoir
        Surface-ocean reservoir supplying exported carbonate.
    this_box : Reservoir
        Intermediate-ocean reservoir receiving dissolved carbonate.
    next_box : Reservoir
        Deep-ocean reservoir receiving dissolved carbonate and storing
        snowline state variables.
    burial_box : Reservoir
        Sediment reservoir receiving permanently buried carbonate.
    kwargs : dict
        Carbonate-system configuration parameters.

    Returns
    -------
    ExternalCode
        Configured Carbonate System 4 external-code instance.

    Notes
    -----
    The implementation assumes that the export flux represents total
    CaCO₃ export over the sediment area bounded by ``z0`` and ``zmax``.
    """
    # Area between z0 and zmax
    AD = source_box.mo.hyp.area_dz(kwargs["z0"], kwargs["zmax"])
    swc = this_box.swc  # shorthand for seawater constants
    swc_p = (  # seawater parameters as tuple
        swc.K1,
        swc.K2,
        swc.K1K2,
        swc.KW,
        swc.KB,
        swc.ca2,
        swc.boron,
        source_box.DIC.isotopes,
    )
    cp = (  # other constants
        kwargs["Ksp0"],  # 7
        float(kwargs["kc"]),  # 8
        AD,  # 9
        int(abs(kwargs["zsat0"])),  # 10
        kwargs["I_caco3"],  # 11
        kwargs["alpha"],  # 12
        int(abs(kwargs["zsat_min"])),  # 13
        int(abs(kwargs["zmax"])),  # 14
        int(abs(kwargs["z0"])),  # 15
        int(abs(kwargs["zint"])),
    )

    # initialize an external code instance
    ec = ExternalCode(
        name="cs4",
        species=source_box.mo.Carbon.CO2,
        function=carbonate_system_4,
        fname="carbonate_system_4",
        isotopes=source_box.DIC.isotopes,
        r_s=source_box,  # source (RG) of CaCO3 flux,
        r_d=this_box,  # sink (RG) of dissolved CaCO3 flux associated with intermediate box
        r_n=next_box, #sink (RG) of CaCO3 flux associated with deep box
        r_b=burial_box, #sink (RG) of undissolved CaCO3 flux
        function_input_data=[  # variable input data
            export_flux,         # CaCO3_export
            next_box.DIC,        # dic_t_db (deep box)
            next_box.TA,         # ta_db (deep box)
            this_box.DIC,        # dic_t_ib (intermediate box)
            this_box.TA,         # ta_ib (intermediate box)
            source_box.DIC,      # dic_t_sb (surface box)
            "Hplus",             # hplus_db_0 (deep box H+ at t-1)
            this_box.swc.hplus,  # hplus_ib_0 (intermediate box H+ at t-1)
            "zsnow",             # zsnow
        ],

        function_params=(  # constant input data
            swc_p,
            cp,
            next_box.mo.area_table,
            next_box.mo.area_dz_table,
            next_box.mo.Csat_table,
        ),
        return_values=[
            {f"F_{this_box.full_name}.DIC": "ib_DIC_cs4"},
            {f"F_{this_box.full_name}.TA": "ib_TA_cs4"}, 
            {f"R_{next_box.full_name}.Hplus": next_box.swc.hplus},
            {f"R_{next_box.full_name}.zsnow": float(abs(kwargs["zsnow"]))},
            {f"F_{next_box.full_name}.DIC": "db_DIC_cs4"}, 
            {f"F_{next_box.full_name}.TA": "db_TA_cs4"},
            {f"F_{burial_box.full_name}.DIC": "burial_DIC"},
            {f"F_{burial_box.full_name}.TA": "burial_TA"},
        ],
        register=next_box, 

    )
    next_box.mo.lpc_f.append(ec.fname)  # list of function to be imported in ode backend
    
    return ec

def add_carbonate_system_4(**kwargs) -> None:
    """
    Create a new carbonate system virtual reservoir.

    This function initializes carbonate system 4 for the specified set of boxes.
    It computes saturation, compensation, and snowline depth, and the associated 
    carbonate burial fluxes.

    Parameters
    ----------
    source_box, r_sb : list[ Reservoir]
        list of surface Reservoirs.
    this_box, r_db : list[ Reservoir]
        list of intermediate Reservoirs
    next_box, r_nb : list[Reservoir]
        list of deep Reservoirs.
    burial_box, r_bb : list[Reservoir]
        Burial reservoirs receiving permanently buried CaCO3.
    carbonate_export_fluxes : list[Flux]
        list of CaCO3 export Flux objects from the surface reservoir.
    z0 : float
        Depth of surface ocean box. 
    zint : float
        Depth of the intermediate ocean box.

    Returns
    -------
    None

    Other Parameters
    ----------------
    Optional (defaulted) parameters:
        zsat, zcc, zsnow, zsat0, Ksp0, kc, alpha, pg, pc, I_caco3, zmax, Ksp
    """
    # list of known keywords
    lkk: dict = {
        "this_box": list,
        "source_box": list,
        "next_box": list,
        "burial_box": list,
        "r_db": list,
        "r_sb": list,
        "r_nb": list,
        "r_bb": list,
        "carbonate_export_fluxes": list,
        "zsat": int,
        "zsat_min": int,
        "zcc": int,
        "zsnow": int,
        "zsat0": int,
        "Ksp0": float,
        "kc": float,
        "ca2": float,
        "pc": (float, int),
        "pg": (float, int),
        "I_caco3": (float, int),
        "alpha": float,
        "zmax": (float, int),
        "z0": (float, int),
        "zint": (float, int),
        "Ksp": (float, int),
    }

    # provide a list of absolutely required keywords:
    lrk: list = [
        ["r_sb", "source_box"],
        ["r_db", "this_box"],
        ["r_nb", "next_box"],
        ["r_bb","burial_box"],
        "carbonate_export_fluxes",
        "z0",
        "zint",
    ]

    source_box = kwargs.get("source_box", kwargs.get("r_sb"))
    this_box = kwargs.get("this_box", kwargs.get("r_db"))
    next_box = kwargs.get("next_box", kwargs.get("r_nb"))
    burial_box = kwargs.get("burial_box", kwargs.get("r_bb"))
    carbonate_export_fluxes = kwargs.get("carbonate_export_fluxes")

    #we need the reference to the Model in order to set some default values

    reservoir = this_box[0]
    model = reservoir.mo

    #list of default values if none provided:
    lod: dict = {
        "source_box": [],
        "zsat": -3715,
        "zcc": -4750,
        "zsnow": -4750,
        "zsat0": -5078,
        "Ksp0": None,  # will be set later from reservoir.swc
        "kc": 8.84 * 1000,
        "alpha": 0.6,
        "pg": 0.103,
        "pc": 511,
        "I_caco3": 529,
        "zmax": -10999,
        "zint": -2000,
        "Ksp": None,
    }

    if lod["Ksp0"] is None:
        lod["Ksp0"] = reservoir.swc.Ksp0
    if lod["Ksp"] is None:
        lod["Ksp"] = reservoir.swc.Ksp_ca

    __checkkeys__(lrk, lkk, kwargs)
    kwargs = __addmissingdefaults__(lod, kwargs)
    __checktypes__(lkk, kwargs)

    if source_box is None or this_box is None or carbonate_export_fluxes is None:
        raise CarbonateSystem2Error("Missing required inputs: source_box, this_box, or export_fluxes")

    if "zsat_min" not in kwargs:        
        kwargs["zsat_min"] = kwargs["z0"]
   
    if not isinstance(this_box, list):
        this_box = [this_box]

    if not isinstance(source_box, list):
        source_box = [source_box]

    if not isinstance(next_box, list):
        next_box = [next_box]

    if not isinstance(burial_box, list):
        burial_box = [burial_box]

    if len(this_box) != len(source_box):
        raise CarbonateSystem2Error(
            f"Number of surface boxes ({len(source_box)}) does not match intermediate boxes ({len(this_box)})"
        )
    if len(next_box) != len(this_box):
        raise CarbonateSystem2Error(
            f"Number of deep boxes ({len(next_box)}) does not match intermediate boxes ({len(this_box)})"
        )
    if len(burial_box) != len(next_box):
        raise CarbonateSystem2Error(
            f"Number of burial boxes ({len(burial_box)}) does not match deep boxes ({len(next_box)})"
        )


    pg = kwargs["pg"]
    pc = kwargs["pc"]
    zmax = abs(int(kwargs["zmax"]))

    #check if we already have the hypsometry and saturation tables
    if not hasattr(model, "area_table"):
        depth_range = np.arange(0, zmax, 1, dtype=float)
        model.area_table = model.hyp.get_lookup_table_area()
        model.area_dz_table = model.hyp.get_lookup_table_area_dz() * -1
        model.Csat_table = (
            reservoir.swc.Ksp0 / reservoir.swc.ca2 * np.exp(
                (depth_range * kwargs["pg"]) / kwargs["pc"]
            )
        )

    #set up virtual reservoirs:
    for i, (sb, db, nb, bb) in enumerate(zip(source_box, this_box, next_box, burial_box)):

        if not (hasattr(db, "DIC") and hasattr(db, "TA")):
            raise AttributeError(f"{db.full_name} must have a DIC and TA reservoir")

        db.swc.update_parameters()

        export_flux = kwargs["carbonate_export_fluxes"][i]
        export_flux.serves_as_input = True  # flag this for ode backend
        ec = init_carbonate_system_4(
            export_flux,
            source_box[i],
            this_box[i],
            next_box[i],
            burial_box[i],
            kwargs,
        )

        register_return_values(ec, db)
        db.has_cs4 = True
        nb.has_cs4 = True
        
    
def get_pco2(SW) -> float:
    """Calculate the concentration of pCO2."""
    dic_c: float = SW.dic
    hplus_c: float = SW.hplus
    k1: float = SW.K1
    k2: float = SW.K2
    co2: NDArrayFloat = dic_c / (1 + (k1 / hplus_c) + (k1 * k2 / (hplus_c**2)))
    pco2: NDArrayFloat = co2 / SW.K0 * 1e6
    return pco2


# define a transform function to display the Hplus concentration as pH
def phc(m: float) -> float:
    """Define a plot transform for the reservoir class.

    Here we use this to
    display the H+ concentrations as pH. After import, you can use it
    with like this in the reservoir definition

    plot_transform_c=phc,

    """
    import numpy as np

    pH = -np.log10(m)
    return pH
