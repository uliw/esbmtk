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


# @njit(fastmath=True)
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

# @lru_cache
def get_zsat(zsat0, zsat_min, zmax, ca2, co3, ksp0):
    """Calcualte zsat."""
    zsat = int(zsat0 * log(ca2 * co3 / ksp0))
    return min(zmax, max(zsat_min, zsat))


# @lru_cache
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
"""
Carbonate System 4:

Update September 2025 (testing): handles carbonate chemistry for a model with three ocean layers 
(surface, intermediate, deep) and one sediment layer. Based on Boudreau et al., (2010).

"""

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
    """

    This functions returns (in order):

    - Intermediate Box DIC
    - Intermediate Box TA
    - Hplus
    - zsnow
    - Deep Box DIC
    - Deep Box TA
    - Burial DIC 
    - Burial TA

    LIMITATIONS:
    - Assumes all concentrations are in mol/kg
    - Assumes your Model is in mol/kg

    Calculations are based off equations from:
    Boudreau et al., 2010, https://doi.org/10.1029/2009GB003654

    """
    sp, cp, area_table, area_dz_table, Csat_table = p
    ksp0, kc, AD, zsat0, I_caco3, alpha, zsat_min, zmax, z0, zint = cp
    k1, k2, k1k2, KW, KB, ca2, boron, isotopes = sp

    if isotopes:
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

    #taking a simple average of the CO3 of both boxes to calculate zsat
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
        diff = Csat_table[zcc:int(zsnow)] - co3_deep
        area_cc_snow = area_dz_table[zcc:int(zsnow)]
        BPDC = max(0.0, kc * np.dot(area_cc_snow, diff))
        dzdt_zsnow = -BPDC / (area_dz_table[int(zsnow)] * I_caco3)


# H+ concentration rate change 
#does this need modification re: intermediate box?
    dCdt_Hplus = hplus - hplus_db_0

#F_diss_int and F_diss_deep are going to be connected to different boxes:
    if zsat < zint:
        F_diss_int = BNS + BDS_int
        F_diss_deep = BDS_deep + BCC + BPDC
    else:
        F_diss_int = BNS_int
        F_diss_deep = BNS_deep + BDS + BCC + BPDC

    F_burial = CaCO3_export - F_diss_int - F_diss_deep 

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
    """Initialize a carbonate system 4 instance.

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
    burial_box :
        Reservoir instance of the sediment box
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
        r_n=next_box, #sink (RG) of CaCO3 flux associated with intermediate box
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
            this_box.mo.area_table,
            this_box.mo.area_dz_table,
            this_box.mo.Csat_table,
        ),
        return_values=[
            {f"F_{this_box.full_name}.DIC": "ib_DIC"},
            {f"F_{this_box.full_name}.TA": "ib_TA"}, 
            {f"R_{this_box.full_name}.Hplus": this_box.swc.hplus},
            {f"R_{this_box.full_name}.zsnow": float(abs(kwargs["zsnow"]))},
            {f"F_{next_box.full_name}.DIC": "db_DIC"}, 
            {f"F_{next_box.full_name}.TA": "db_TA"},
            {f"F_{burial_box.full_name}.DIC": "burial_DIC"},
            {f"F_{burial_box.full_name}.TA": "burial_TA"},
        ],
        register=this_box,
    )
    this_box.mo.lpc_f.append(ec.fname)  # list of function to be imported in ode backend
    


    return ec

def add_carbonate_system_4(**kwargs) -> None:
    """Create a new carbonate system virtual reservoir.

    This function initializes carbonate system 4 (cs4) for each specified deep box.
    It computes saturation, compensation, and snowline depth, and the associated 
    carbonate burial fluxes.

    Required keywords:
        r_sb / source_box: list of surface Reservoirs
        r_db / this_box: list of intermediate Reservoirs
        r_nb / next_box: list of deep Reservoirs
        r_bb / burial_box: list of burial Reservoirs
        carbonate_export_fluxes: list of CaCO3 export Flux objects from the surface reservoirs
        z0: depth (m) for burial calculations
        zint: depth of intermediate box 

    Optional (defaulted) keywords:
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
        "Ca2": float,
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
        next_box = [burial_box]

    if len(this_box) != len(source_box):
        raise CarbonateSystem2Error(
            f"Number of surface boxes ({len(source_box)}) does not match deep boxes ({len(this_box)})"
        )
    if len(next_box) != len(this_box):
        raise CarbonateSystem2Error(
            f"Number of next boxes ({len(next_box)}) does not match deep boxes ({len(this_box)})"
        )
    if len(burial_box) != len(this_box):
        raise CarbonateSystem2Error(
            f"Number of burial boxes ({len(burial_box)}) does not match deep boxes ({len(this_box)})"
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

        nb.swc.update_parameters()

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

        register_return_values(ec, nb)
        nb.has_cs4 = True
    