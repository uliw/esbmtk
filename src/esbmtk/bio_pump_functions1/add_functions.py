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
from esbmtk import Q_, register_return_values
from esbmtk.utility_functions import __addmissingdefaults__, __checkkeys__, __checktypes__

if tp.TYPE_CHECKING:
    from esbmtk import Flux, Model, ReservoirGroup


def add_photosynthesis(
    rgs: list[ReservoirGroup],
    p_fluxes: list,
    piston_velocity,
    O2_At,
    CO2_At,
    CaCO3_reactions=True,
):
    """Add process to ReservoirGroup(s) in rgs. pfluxes must be list of Flux
    objects or float values that correspond to the rgs list
    """
    from esbmtk import register_return_values
    from esbmtk.reactions.init_functions import init_photosynthesis

    M = rgs[0].mo
    pv = piston_velocity.to("meter/yr").magnitude
    for i, rg in enumerate(rgs):
        if isinstance(p_fluxes[i], Q_):
            p_fluxes[i] = p_fluxes[i].to("mol/year").magnitude

        ec = init_photosynthesis(rg, p_fluxes[i], pv, O2_At, CO2_At, CaCO3_reactions)
        register_return_values(ec, rg)
        rg.has_cs1 = True


def add_OM_remineralization(M: Model, f_map: dict) -> None:
    """
    Add OM_remineralization fluxes to the model.

    Parameters:
    M (Model): The model object t
    f_map (dict): A dictionary that maps sink names to source dictionaries. The
    source dictionary should contain the source species and a list of type
    and OM_remineralization values. For example, {M.A_ib: {M.H_sb: ["POM", 0.3]}}.

    Raises:
    ValueError: If an invalid type is specified in the source dictionary.

    Returns:
    None
    """
    from esbmtk.reactions.init_functions import init_OM_remineralization

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
            ec = init_OM_remineralization(
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
            ec = init_OM_remineralization(
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


def add_carbonate_system_3(**kwargs) -> None:
    """ Creates a new carbonate system virtual reservoir
    which will compute carbon species, saturation, compensation,
    and snowline depth, and compute the associated carbonate burial fluxes

    Required keywords:

    :param r_sb : list of ReservoirGroup objects in the surface layer
    :param r_db : list of ReservoirGroup objects in the deep layer
    :param pic_export_flux : list of flux objects that match the ReservoirGroup objects
    :param zsat_min: depth of the upper boundary of the deep box
    :param z0: upper depth limit for carbonate burial calculations typically zsat_min
    
    Optional Parameters:

    :param zsat: initial saturation depth (m)
    :param zcc: initial carbon compensation depth (m)
    :param zsnow: initial snowline depth (m)
    :param zsat0: characteristic depth (m)
    :param Ksp0: solubility product of calcite at air-water interface (mol^2/kg^2)
    :param kc: heterogeneous rate constant/mass transfer coefficient for calcite dissolution (kg m^-2 yr^-1)
    :param Ca2: calcium ion concentration (mol/kg)
    :param pc: characteristic pressure (atm)
    :param pg: seawater density multiplied by gravity due to acceleration (atm/m)
    :param I: dissolvable CaCO3 inventory
    :param co3: CO3 concentration (mol/kg)
    :param Ksp: olubility product of calcite at in situ sea water conditions (mol^2/kg^2)

    """
    from esbmtk.reactions.init_functions import init_carbonate_system_3

    # list of known keywords
    lkk: dict = {
        "r_db": list,  # list of deep reservoirs
        "r_ib": list,  # list of corresponding surface reservoirs
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
        "r_ib",
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

    i = 0
    j = 0
    for rg in kwargs["r_db"]:  # Setup the virtual reservoirs
        ec = init_carbonate_system_3(
            rg,
            kwargs["pic_export_flux"][j],
            kwargs["r_sb"][i],
            kwargs["r_ib"][i],
            kwargs["r_db"][i],
            kwargs,
        )
        i += 1
        j += 2

        register_return_values(ec, rg)
        rg.has_cs2 = True


def add_co2_gas_exchange(rg_list, gas_r):
    """Add gas_exchange. Note that this will register
    the resulting flux only with the GasReservoir. So
    we need to add it to the respective surface reservoir
    manually. Approach. Scan e.c. return objects for the correct flux
    not sure yet how to deal with sign

    :param rg_list: list of reservoir group names
    """
    from esbmtk.reactions.init_functions import init_gas_exchange_with_isotopes

    species = rg_list[0].mo.CO2
    pv = Q_("4.8 m/d")

    for rg in rg_list:
        lr = getattr(rg, "DIC")
        l_ref = getattr(rg, "CO2aq")
        solubility = getattr(lr.swc, "SA_co2")
        ec = init_gas_exchange_with_isotopes(gas_r, lr, l_ref, species, pv, solubility)
        register_return_values(ec, lr)
        rg.has_cs1 = True


def add_o2_gas_exchange(rg_list, gas_r):
    """Add gas_exchange

    :param rg_list: list of reservoir group names
    """
    from esbmtk.reactions.init_functions import init_gas_exchange_no_isotopes

    species = rg_list[0].mo.O2
    pv = Q_("4.8 m/d")

    for rg in rg_list:
        lr = getattr(rg, "O2")
        l_ref = getattr(rg, "O2")
        solubility = getattr(lr.swc, "SA_o2")
        ec = init_gas_exchange_no_isotopes(gas_r, lr, l_ref, species, pv, solubility)
        register_return_values(ec, lr)
        gas_r.has_cs1 = True
