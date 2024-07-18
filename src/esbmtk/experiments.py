"""
     esbmtk: A general purpose Earth Science box model toolkit
     Copyright (C), 2020 Ulrich G. Wortmann

     This program is free software: you can redistribute it and/or modify
     it under the terms of the GNU General Public License as published by
     the Free Software Foundation, either version 3 of the License, or
     (at your option) any later version.

     This program is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
     GNU General Public License for more details.

     You should have received a copy of the GNU General Public License
     along with this program.  If not, see <https://www.gnu.org/licenses/>.

     This function add_my_burial is called externally to introduce a new flux to the esbtmk ecosystem. 
     Here it calls another function known as calculate_burial, which calculates fluxes based on oxygenation 
     of deep water in line with Van Capellan & Ingall, 1994. These fluxes are returned at every model run.
"""

import numpy as np
import numpy.typing as npt
from esbmtk import (
    ExternalCode,
    register_return_values,
    Flux,
    Species,
    SpeciesProperties,
    Reservoir,
)

# declare numpy types
NDArrayFloat = npt.NDArray[np.float64]

# debug counter
counter = 0


def calculate_burial(
    po4_export_flux: float, o2_c: float, frac_burial_params: tuple
) -> float:
    """#add an empty tuple
    Calculate burial as a function of productivity and oxygen concentration.

    :param po4_export_flux: Surface ocean productivity in umol/L
    :type po4_export_flux: float
    :param o2_con: Oxygen concentration in the deep box in umol/L
    :type o2_con: float
    :return: Burial flux in mol/year
    :rtype: float
    """
    # global debug counter
    global counter  # never ever use global variables!
    counter += 1

    (
        frac_burial,
        cp_ox,
        cp_anox,
        thc,
    ) = frac_burial_params

    productivity_mol_year = po4_export_flux  # productivity in mol/year
    NPP = 106 * productivity_mol_year

    # DOA Calc
    if o2_c < 0:
        o2_c = 0

    # koa = 6.58e14  # mol/m Koa is  an apparent constant which is proportional to the
    # average dissolved oxygen concentration of downwelling surface waters
    DOA_alt = 1 - (
        6.58e14 * ((thc / 11.448) / NPP)
    )  # 11.448 is the conversion factor between vmix and sv.

    DOA = (
        o2_c / 1.92e-4
    )  # 250 is max O2 content in umol/l in deep ocean, 150 is the max average
    DOA = 1 - DOA
    """
    DOA_alt = 1 - DOA
    """

    # Apply min and max to ensure DOA_alt is within [0, 1]
    if DOA_alt < 0:
        DOA_alt = 0
    elif DOA_alt > 1:
        DOA_alt = 1

    if DOA < 0:
        DOA = 0
    elif DOA > 1:
        DOA = 1

    # unstable yet logical funciton
    """
    frac_burial = (cp_ox * cp_anox) / (((1 - DOA_alt) * cp_anox) + ((DOA_alt) * cp_ox))

    productivity_mol_year = po4_export_flux  # productivity in mol/year

    burial_flux = 0 #need to define
    
    #POP flux more realistic
    
    NPP = 106 * productivity_mol_year

    oc_b = 1.2e-26 * (NPP**2.5)

    POP_flux = oc_b / frac_burial
    
    #POP flux less realistic but returns overall 
    POP_flux = productivity_mol_year / (frac_burial)
    burial_flux += POP_flux

    fe_p_burial = 7.60e9 * (1 - DOA_alt)  # in umol/year using k59 from van cap

    burial_flux += fe_p_burial

    p_remineralisation_flux = productivity_mol_year - burial_flux

    ap_burial = 5.56e-24 * (p_remineralisation_flux**2.5)  # from van cap in umol/year

    burial_flux += ap_burial

    p_remineralisation_flux = productivity_mol_year - burial_flux
    """

    # stable function
    frac_burial = (cp_ox * cp_anox) / (((1 - DOA) * cp_anox) + ((DOA) * cp_ox))

    burial_flux = 0  # need to define
    # POP flux more realistic

    oc_b = 1.2e-26 * (NPP**2.5)

    POP_flux = oc_b / frac_burial

    # POP flux less realistic but returns overall
    # POP_flux = productivity_mol_year / (frac_burial)

    burial_flux += POP_flux

    p_remineralisation_flux = productivity_mol_year - burial_flux

    ap_burial = 5.56e-24 * (p_remineralisation_flux**2.5)  # from van cap in umol/year

    burial_flux += ap_burial

    fe_p_burial = 7.60e9 * (1 - DOA)  # in umol/year using k59 from van cap

    burial_flux += fe_p_burial

    p_remineralisation_flux = productivity_mol_year - burial_flux

    # debugging:
    """
    flux_values = {
        "BF": -burial_flux,
        "rf": p_remineralisation_flux,
        "fe_p_burial": fe_p_burial,
        "ap_burial": ap_burial,
        "PO4_export_flux": po4_export_flux,
        "POP_flux": POP_flux,
        "O2_c": o2_c,
        "burial_fraction": frac_burial,
    }
    """
    """
    print(
        f"THC = {thc} BF = {-burial_flux:.2e}, rf = {p_remineralisation_flux:.2e}\n"
        f"fe-p_burial = {fe_p_burial:.2e}, ap_burial = {ap_burial:.2e}\n"
        f"PO4 export flux = {po4_export_flux:.2e}, POP_flux = {POP_flux:.2e}\n"
        f"O2_c = {o2_c:.2e}, DOA = {DOA}, DOA = {DOA_alt}, burial fraction = {frac_burial:.2e}\n"
    )
    """
    # if for use in debugging (first and last)

    if counter == 20000:
        print(
            f"THC = {thc} BF = {-burial_flux:.2e}, rf = {p_remineralisation_flux:.2e}\n"
            f"fe-p_burial = {fe_p_burial:.2e}, ap_burial = {ap_burial:.2e}\n"
            f"PO4 export flux = {po4_export_flux:.2e}, POP_flux = {POP_flux:.2e}\n"
            f"O2_c = {o2_c:.2e},  DOA = {DOA:.2e} DOA_alt = {DOA_alt:.2e} \n"
            f"NPP = {NPP:.2e}, burial fraction = {frac_burial:.2e}\n"
        )
        if counter == 20000:
            print("---------------------------------------------")
            counter = 0

    return -burial_flux, p_remineralisation_flux


def add_my_burial(**kwargs) -> None:
    """
    This function initializes a user-supplied function so that it can be used within the ESBMTK ecosystem.

    Parameters
    ----------
    kwargs : dict
        A dictionary of function arguments, including:
        - source : Source | Species | Reservoir
        - sink : Sink | Species | Reservoir
        - species : SpeciesProperties
        - po4_export_flux : float
        - o2_c : float
        - frac_burial : float
        - cp_ox : float
        - cp_anox : float
        - thc : float
        - my_id1
        - my_id2
    """
    # Extracting arguments from kwargs
    source = kwargs.get("source")
    sink = kwargs.get("sink")
    species = kwargs.get("species")
    o2_c = kwargs.get("o2_c")
    po4_export_flux = kwargs.get("po4_export_flux")
    frac_burial = kwargs.get("frac_burial")
    cp_ox = kwargs.get("cp_ox")
    cp_anox = kwargs.get("cp_anox")
    thc = kwargs.get("thc")
    my_id1 = kwargs.get("my_id1")
    my_id2 = kwargs.get("my_id2")

    # Useful type prints for debugging:
    """
    print(f"Type of source: {type(source)}")
    print(f"Type of sink: {type(sink)}")
    print(f"Type of species: {type(species)}")
    print(f"po4_export_flux: {po4_export_flux}, o2_c: {o2_c}")
    print(f"Source name: {source.full_name}, Sink name: {sink.full_name}")
    # print(
    #   f"Function Params: {frac_burial}, {dbv}, {min_burial_fraction}, {max_burial_fraction}"
    # )
    model = species.mo
    dbv: float = source.volume.to(model.v_unit).magnitude
    """
    # print(f"Source name: {source.full_name}")
    ec = ExternalCode(
        name="calculate_burial",
        species=species,
        ftype="in_sequence",
        function=calculate_burial,
        fname="calculate_burial",
        function_input_data=[po4_export_flux, o2_c],
        function_params=(
            frac_burial,
            cp_ox,
            cp_anox,
            thc,
        ),
        return_values=[
            {f"F_{sink.full_name}.{species.name}": f"{my_id1}__F"},  # po4 burial
            {f"F_{source.full_name}.{species.name}": f"{my_id2}__F"},  # po4 rmin
        ],
        register=source,
    )
    register_return_values(ec, source)
    source.mo.lpc_f.append(ec.fname)
