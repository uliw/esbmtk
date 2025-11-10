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
Carbonate System 5:

Update November 2025 (testing): handles carbonate chemistry for a model with four ocean layers 
(surface, intermediate, deep, abyssal) and one sediment layer. Based on Boudreau et al., (2010).

"""

def carbonate_system_5(
    CaCO3_export: float, #total CaCO3 export from the surface
    dic_t_sb: float | tuple, #[DIC] in surface box
    dic_t_ib: float | tuple, #[DIC] in intermediate box
    ta_ib: float, #[TA] in intermediate box
    dic_t_db: float | tuple, #[DIC] in deep box
    ta_db: float, #[TA] in intermediate box
    dic_t_ab: float | tuple, #[DIC] in abyssal box
    ta_ab: float, #[TA] in intermediate box
    hplus_ib_0: float,
    hplus_db_0: float,
    hplus_ab_0: float,
    zsnow: float,
    p: tuple,
) -> tuple:
    """
    5-layer carbonate system:
    surface → intermediate → deep → abyssal → burial

        This functions returns (in order):

    - Intermediate Box DIC
    - Intermediate Box TA
    - Deep Box DIC
    - Deep Box TA
    - Abyssal Box DIC 
    - Abyssal Box TA
    - Burial DIC 
    - Burial TA
    - Intermediate Box dC_dt Hplus
    - Deep Box dC_dt Hplus
    - Abyssal Box dC_dt Hplus
    - zsnow

    LIMITATIONS:
    - Assumes all concentrations are in mol/kg
    - Assumes your Model is in mol/kg
    - Only 4 ocean layers (Surface, Intermediate, Deep, Abyssal)

    we'll get to carbonate_system_n one day :')

    Calculations are based off equations from:
    Boudreau et al., 2010, https://doi.org/10.1029/2009GB003654

    """

    sp, cp, area_table, area_dz_table, Csat_table = p
    ksp0, kc, AD, zsat0, I_caco3, alpha, zsat_min, zmax, z0, zint, zdeep = cp
    k1, k2, k1k2, KW, KB, ca2, boron, isotopes = sp

    # unpack DIC concentrations for isotopes if needed
    if isotopes:
        dic_sb, dic_sb_l = dic_t_sb
        dic_ib, dic_ib_l = dic_t_ib
        dic_db, dic_db_l = dic_t_db
        dic_ab, dic_ab_l = dic_t_ab
    else:
        dic_sb = dic_t_sb
        dic_ib = dic_t_ib
        dic_db = dic_t_db
        dic_ab = dic_t_ab

    # compute H+ using seawater constants
    hplus_ib = get_hplus(dic_ib, ta_ib, max(hplus_ib_0, 1e-12), boron, k1, k1k2, KW, KB)
    hplus_db = get_hplus(dic_db, ta_db, max(hplus_db_0, 1e-12), boron, k1, k1k2, KW, KB)
    hplus_ab = get_hplus(dic_ab, ta_ab, max(hplus_ab_0, 1e-12), boron, k1, k1k2, KW, KB)

    # CO3 concentrations
    co3_ib = dic_ib / (1 + hplus_ib / k2 + hplus_ib**2 / k1k2)
    co3_db = dic_db / (1 + hplus_db / k2 + hplus_db**2 / k1k2)
    co3_ab = dic_ab / (1 + hplus_ab / k2 + hplus_ab**2 / k1k2)

    # average CO3 for zsat calculation
    co3_avg = (
        abs(zint - z0) * co3_ib
        + abs(zdeep - zint) * co3_db
        + abs(zmax - zdeep) * co3_ab
    ) / abs(z0 - zmax)

    # critical depths
    zsat = get_zsat(zsat0, zsat_min, zmax, ca2, co3_avg, ksp0)
    zcc = get_zcc(CaCO3_export, zmax, zsat_min, zsat0, ca2, ksp0, AD, kc, co3_avg)

    # Flux per unit area
    B_AD = CaCO3_export / AD

    # --- CASE SPLITTING LOGIC ---

    zsat_above_zint = zsat < zint
    zcc_above_zdeep = zcc < zdeep

    BNS_ib = BNS_db = 0.0
    BDS_ib = BDS_db = BDS_ab = 0.0
    BCC_db = BCC_ab = 0.0
    BPDC = 0.0

    if zsat_above_zint: #i.e. surface to zsat to zint

	    A_z0_zsat = area_table[z0] - area_table[zsat]
	    A_zsat_zint = area_table[zsat] - area_table[zint]
	
	    # surface → zsat (BNS)
	    BNS_ib = alpha * A_z0_zsat * B_AD
	
	# zsat → zint (BDS)
	    diff_co3_ib = Csat_table[zsat:zint] - co3_ib
	    area_sat_int = area_dz_table[zsat:zint]
	
	    BDS_ib_undersat = kc * area_sat_int.dot(diff_co3_ib)
	    BDS_ib_resp = alpha * (A_zsat_zint * B_AD - BDS_ib_undersat)
	
	 #fraction of BDS going in intermediate box:
	    BDS_ib = BDS_ib_undersat + BDS_ib_resp
	
	#remaining BDS goes into deep box OR deep + abyssal box both depending on zcc
	# zcc split:

	    if zcc_above_zdeep: #i.e. all remaining BDS goes in deep box only
		    A_zint_zcc = area_table[zint] - area_table[zcc]
	
		    area_int_cc = area_dz_table[zint:zcc]
		    diff_co3_db = Csat_table[zint:zcc] - co3_db
    
		    BDS_db_undersat = kc * area_int_cc.dot(diff_co3_db)
		    BDS_db_resp = alpha * (A_zint_zcc * B_AD - BDS_db_undersat)
	
		    BDS_db = BDS_db_undersat + BDS_db_resp
	
		# BCC: zcc → zdeep in deep box; zdeep → zmax in abyssal box
	
		    BCC_db =  B_AD * (area_table[zcc] - area_table[zdeep])
		    BCC_ab =  B_AD * (area_table[zdeep] - area_table[zmax])
		
	    else: #i.e. BDS gets split into deep and abyssal box both
		
		    A_zint_zdeep = area_table[zint] - area_table[zdeep]
		    A_zdeep_zcc = area_table[zdeep] - area_table[zcc]
			
		    diff_co3_db = Csat_table[zint:zdeep] - co3_db
		    diff_co3_ab = Csat_table[zdeep:zcc] - co3_ab
			
		    area_int_deep = area_dz_table[zint:zdeep]
		    area_deep_cc = area_dz_table[zdeep:zcc]
			
		    BDS_db_undersat = kc * area_int_deep.dot(diff_co3_db)
		    BDS_db_resp = alpha * (A_zint_zdeep * B_AD - BDS_db_undersat)
			
		    BDS_ab_undersat = kc * area_deep_cc.dot(diff_co3_ab)
		    BDS_ab_resp = alpha * (A_zdeep_zcc * B_AD - BDS_ab_undersat)
			
		    BDS_db = BDS_db_undersat + BDS_db_resp
		    BDS_ab = BDS_ab_undersat + BDS_ab_resp
			
		    BCC_ab = B_AD * (area_table[zcc] - area_table[zmax])

    else: # Surface -> zint -> zsat

	    A_z0_zint = area_table[z0] - area_table[zint]
	    A_zint_zsat = area_table[zint] - area_table[zsat]

	    BNS_ib = alpha * A_z0_zint * B_AD
	    BNS_db = alpha * A_zint_zsat * B_AD

	# BDS: zsat → zcc

	    if zcc_above_zdeep: # case: zsat < zcc < zdeep:

		    A_zsat_zcc = area_table[zsat] - area_table[zcc]
		    diff_co3_db = Csat_table[zsat:zcc] - co3_db

		    area_sat_cc = area_dz_table[zsat:zcc]
		
		    BDS_db_undersat = kc * area_sat_cc.dot(diff_co3_db)
		    BDS_db_resp = alpha * (A_zsat_zcc * B_AD - BDS_db_undersat)
		
		    BDS_db = BDS_db_undersat + BDS_db_resp

		# BCC: zcc → zdeep in deep box; zdeep → zmax in abyssal box

		    BCC_db = B_AD * (area_table[zcc] - area_table[zdeep])
		    BCC_ab = B_AD * (area_table[zdeep] - area_table[zmax])

	    else: #zsat -> zdeep -> zcc

		    A_zsat_zdeep = area_table[zsat] - area_table[zdeep]
		    A_zdeep_zcc = area_table[zdeep] - area_table[zcc]

		    diff_co3_db = Csat_table[zsat:zdeep] - co3_db
		    diff_co3_ab = Csat_table[zdeep:zcc] - co3_ab

		    area_sat_deep = area_dz_table[zsat:zdeep]
		    area_deep_cc = area_dz_table[zdeep:zcc]

		    BDS_db_undersat = kc * area_sat_deep.dot(diff_co3_db)
		    BDS_db_resp = alpha * (A_zsat_zdeep * B_AD - BDS_db_undersat)
  
		    BDS_ab_undersat = kc * area_deep_cc.dot(diff_co3_ab)
		    BDS_ab_resp = alpha * (A_zdeep_zcc * B_AD - BDS_ab_undersat)
  
		    BDS_db = BDS_db_undersat + BDS_db_resp
		    BDS_ab = BDS_ab_undersat + BDS_ab_resp

		    BCC_ab = B_AD * (area_table[zcc] - area_table[zmax])

    # Sediment dissolution if snowline deeper than CCD

    if zsnow <= zcc:
        dzdt_zsnow = abs(zsnow - zcc)
        zsnow = zcc
        BPDC = 0.0
    else:
        zsnow = min(zsnow, zmax)
        diff = Csat_table[zcc:int(zsnow)] - co3_ab
        area_cc_snow = area_dz_table[zcc:int(zsnow)]
        BPDC = max(0.0, kc * np.dot(area_cc_snow, diff))
        dzdt_zsnow = -BPDC / (area_dz_table[int(zsnow)] * I_caco3)

    # H+ concentration changes
    dCdt_Hplus_db = hplus_db - hplus_db_0
    dCdt_Hplus_ib = hplus_ib - hplus_ib_0
    dCdt_Hplus_ab = hplus_ab - hplus_ab_0

    # Total fluxes to boxes
    F_diss_ib = BNS_ib + BDS_ib
    F_diss_db = BNS_db + BDS_db + BCC_db
    F_diss_ab = BDS_ab + BCC_ab + BPDC

    F_burial = max(0.0, CaCO3_export - (F_diss_ib + F_diss_db + F_diss_ab))

    # return values: DIC, TA, Hplus, snowline, burial
    rv = (
        F_diss_ib, F_diss_ib * 2,
        F_diss_db, F_diss_db * 2,
        F_diss_ab, F_diss_ab * 2,
        F_burial, F_burial * 2,
        dCdt_Hplus_ib, dCdt_Hplus_db, dCdt_Hplus_ab,
        dzdt_zsnow,
        
    )
    return rv

def init_carbonate_system_5(
    export_flux: Flux,
    surface_box: Reservoir,
    intermediate_box: Reservoir,
    deep_box: Reservoir,
    abyssal_box: Reservoir,
    burial_box: Reservoir,
    kwargs: dict,
):
    """Initialize a carbonate system 5 instance (surface → intermediate → deep → abyssal → burial)."""
    # Area between surface and bottom
    AD = surface_box.mo.hyp.area_dz(kwargs["z0"], kwargs["zmax"])
    swc = intermediate_box.swc
    swc_p = (
        swc.K1,
        swc.K2,
        swc.K1K2,
        swc.KW,
        swc.KB,
        swc.ca2,
        swc.boron,
        surface_box.DIC.isotopes,
    )
    cp = (
        kwargs["Ksp0"],
        float(kwargs["kc"]),
        AD,
        int(abs(kwargs["zsat0"])),
        kwargs["I_caco3"],
        kwargs["alpha"],
        int(abs(kwargs["zsat_min"])),
        int(abs(kwargs["zmax"])),
        int(abs(kwargs["z0"])),
        int(abs(kwargs["zint"])),
        int(abs(kwargs["zdeep"])),
    )

    # initialize an external code instance
    ec = ExternalCode(
        name="cs5",
        species=surface_box.mo.Carbon.CO2,
        function=carbonate_system_5,
        fname="carbonate_system_5",
        isotopes=surface_box.DIC.isotopes,
        r_s=surface_box,
        r_d=intermediate_box,
        r_n=deep_box,
        r_a=abyssal_box,
        r_b=burial_box,
        function_input_data=[
            export_flux,
            surface_box.DIC,
            intermediate_box.DIC,
            intermediate_box.TA,
            deep_box.DIC,
            deep_box.TA,
            abyssal_box.DIC,
            abyssal_box.TA,
            intermediate_box.swc.hplus,
            deep_box.swc.hplus,
            abyssal_box.swc.hplus,
            "zsnow",
        ],
        function_params=(
            swc_p,
            cp,
            intermediate_box.mo.area_table,
            intermediate_box.mo.area_dz_table,
            intermediate_box.mo.Csat_table,
        ),
        return_values=[
            {f"F_{intermediate_box.full_name}.DIC": "ib_DIC"},
            {f"F_{intermediate_box.full_name}.TA": "ib_TA"},

            {f"F_{deep_box.full_name}.DIC": "db_DIC"},
            {f"F_{deep_box.full_name}.TA": "db_TA"},

            {f"F_{abyssal_box.full_name}.DIC": "ab_DIC"},
            {f"F_{abyssal_box.full_name}.TA": "ab_TA"},

            {f"F_{burial_box.full_name}.DIC": "burial_DIC"},
            {f"F_{burial_box.full_name}.TA": "burial_TA"},
            
            {f"R_{intermediate_box.full_name}.Hplus": intermediate_box.swc.hplus},
            {f"R_{deep_box.full_name}.Hplus": deep_box.swc.hplus},
            {f"R_{abyssal_box.full_name}.Hplus": abyssal_box.swc.hplus},

            {f"R_{intermediate_box.full_name}.zsnow": float(abs(kwargs["zsnow"]))},
        ],
        register=intermediate_box,
    )

    intermediate_box.mo.lpc_f.append(ec.fname)
    return ec


def add_carbonate_system_5(**kwargs) -> None:
    """Add a 5-layer carbonate system instance to the model."""
   

    # list of known keywords
    lkk: dict = {
        "intermediate_box": list,
        "surface_box": list,
        "deep_box": list,
        "abyssal_box": list,
        "burial_box": list,
        "r_db": list,
        "r_sb": list,
        "r_ib": list,
        "r_ab": list,
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
        "zdeep": (float, int),
        "Ksp": (float, int),
    }

    # provide a list of absolutely required keywords:
    lrk: list = [
        ["r_sb", "surface_box"],
        ["r_ib", "intermediate_box"],
        ["r_db", "deep_box"],
        ["r_db", "abyssal_box"],
        ["r_bb","burial_box"],
        "carbonate_export_fluxes",
        "z0",
        "zint",
        "zdeep"
    ]

    surface_box = kwargs.get("surface_box", kwargs.get("r_sb"))
    intermediate_box = kwargs.get("intermediate_box", kwargs.get("r_ib"))
    deep_box = kwargs.get("deep_box", kwargs.get("r_db"))
    abyssal_box = kwargs.get("abyssal_box", kwargs.get("r_ab"))
    burial_box = kwargs.get("burial_box", kwargs.get("r_bb"))
    export_fluxes = kwargs.get("carbonate_export_fluxes")

    # model reference
    reservoir = intermediate_box[0]
    model = reservoir.mo
    zmax = abs(int(kwargs["zmax"]))


    #list of default values if none provided:
    lod: dict = {
        "surface_box": [],
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
        "zint": -1000,
        "zdeep": -3000,
        "Ksp": None,
    }

    if lod["Ksp0"] is None:
        lod["Ksp0"] = reservoir.swc.Ksp0
    if lod["Ksp"] is None:
        lod["Ksp"] = reservoir.swc.Ksp_ca

    __checkkeys__(lrk, lkk, kwargs)
    kwargs = __addmissingdefaults__(lod, kwargs)
    __checktypes__(lkk, kwargs)

    if "zsat_min" not in kwargs:        
        kwargs["zsat_min"] = kwargs["z0"]
   
    if not isinstance(intermediate_box, list):
        intermediate_box = [intermediate_box]

    if not isinstance(surface_box, list):
        surface_box = [surface_box]

    if not isinstance(deep_box, list):
        deep_box = [deep_box]

    if not isinstance(abyssal_box, list):
        abyssal_box = [abyssal_box]

    if not isinstance(burial_box, list):
        burial_box = [burial_box]

    # single-box to list
    for name in ["surface_box", "intermediate_box", "deep_box", "abyssal_box", "burial_box"]:
        box = locals()[name]
        if not isinstance(box, list):
            locals()[name] = [box]

    pg = kwargs["pg"]
    pc = kwargs["pc"]
    zmax = abs(int(kwargs["zmax"]))

    
    # create lookup tables if not existing
    if not hasattr(model, "area_table"):
        depth_range = np.arange(0, zmax, 1, dtype=float)
        model.area_table = model.hyp.get_lookup_table_area()
        model.area_dz_table = model.hyp.get_lookup_table_area_dz() * -1
        model.Csat_table = (
            reservoir.swc.Ksp0 / reservoir.swc.ca2 * np.exp(
                (depth_range * kwargs["pg"]) / kwargs["pc"]
            )
        )

    # loop over boxes and initialize ExternalCode
    for i, (sb, ib, db, ab, bb) in enumerate(zip(surface_box, intermediate_box, deep_box, abyssal_box, burial_box)):
        ib.swc.update_parameters()
        export_flux = kwargs["carbonate_export_fluxes"][i]
        export_flux.serves_as_input = True  # flag this for ode backend

        ec = init_carbonate_system_5(
            export_flux,
            surface_box[i],
            intermediate_box[i],
            deep_box[i],
            abyssal_box[i],
            burial_box[i],
            kwargs
        )
        register_return_values(ec, ib)
        ib.has_cs5 = True
