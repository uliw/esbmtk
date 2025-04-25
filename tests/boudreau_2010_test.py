"""The ESBMTK implmentation of Boudreau et al. 2010.

See https://doi.org/10.1029/2009gb003654

Authors: Uli Wortmann & Tina Tsan

Copyright (C), 2024 Ulrich G. Wortmann & Tina Tsan

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
"""


def initialize_model(rain_ratio, alpha, run_time, time_step):
    """Package the model definition inside a function."""
    from esbmtk import (
        Q_,
        ConnectionProperties,
        GasReservoir,
        Model,
        Species2Species,
        add_carbonate_system_1,
        add_carbonate_system_2,
        create_bulk_connections,
        initialize_reservoirs,
    )

    M = Model(
        stop=run_time,  # end time of model
        max_timestep=time_step,  # time step
        element=[  # list of elements we consider in the model
            "Carbon",
            "Boron",
            "Hydrogen",
            "misc_variables",  # needed for plotting depth data
        ],
        mass_unit="mol",
        concentration_unit="mol/kg",
        opt_k_carbonic=13,  # Use Millero 2006
        opt_pH_scale=3,  # 1:total, 3:free scale
    )

    # -------------------- Set up box parameters ------------------------ #
    """ Boudreau et al defined their reservoirs by area and and volume, and
    explicitly assign temperature, pressure and salinity.
    """
    box_parameters: dict = {  # name: [[geometry], T, P]
        "H_b": {  # High-Lat Box
            "c": {M.DIC: "2153 umol/kg", M.TA: "2345 umol/kg"},
            "g": {"area": "0.5e14m**2", "volume": "1.76e16 m**3"},  # geometry
            "T": 2,  # temperature in C
            "P": 17.6,  # pressure in bar
            "S": 35,  # salinity in psu
        },
        "L_b": {  # Low-Lat Box
            "c": {M.DIC: "1952 umol/kg", M.TA: "2288 umol/kg"},
            "g": {"area": "2.85e14m**2", "volume": "2.85e16 m**3"},
            "T": 21.5,
            "P": 5,
            "S": 35,
        },
        "D_b": {  # Deep Box
            "c": {M.DIC: "2291 umol/kg", M.TA: "2399 umol/kg"},
            "g": {"area": "3.36e14m**2", "volume": "1.29e18 m**3"},
            "T": 2,
            "P": 240,
            "S": 35,
        },  # sources and sinks
        "Fw": {"ty": "Source", "sp": [M.DIC, M.TA]},
        "Fb": {"ty": "Sink", "sp": [M.DIC, M.TA]},
    }

    species_list = initialize_reservoirs(M, box_parameters)

    """ Define the mixing and thermohaline fluxes 
    through a dictionary that specifies the respective source and sink
    reservoirs, connection id,  the connection type, the scaling factor
    and the list of species that will be affected. 
    """
    connection_dict = {
        # source_to_sink@id
        "H_b_to_D_b@mix_down": {  # High Lat mix down F4
            "ty": "scale_with_concentration",  # type
            "sc": "30 Sverdrup",  # scale
            "sp": species_list,  # list of affected species
        },
        "D_b_to_H_b@mix_up": {  # High Lat mix up F4
            "ty": "scale_with_concentration",
            "sc": "30 Sverdrup",
            "sp": species_list,
        },
        "L_b_to_H_b@thc": {  # thc L to H F3
            "ty": "scale_with_concentration",
            "sc": "25 Sverdrup",
            "sp": species_list,
        },
        "H_b_to_D_b@thc": {  # thc H to D F3
            "ty": "scale_with_concentration",
            "sc": "25 Sverdrup",
            "sp": species_list,
        },
        "D_b_to_L_b@thc": {  # thc D to L F3
            "ty": "scale_with_concentration",
            "sc": "25 Sverdrup",
            "sp": species_list,
        },
    }
    create_bulk_connections(connection_dict, M)

    """ Organic matter flux P - 200 Tm/yr POM = Particulate Organic
    matter. Since this model uses a fixed rate we can declare this flux with
    the rate keyword. Boudreau 2010 only considers the effect on DIC, and
    ignores the effect of POM on TA.

    Note the "bp" keyword: This specifies that the connection will remove the
    respective species form the source reservoir, but will bypass the addition
    to the sink reservoir. It is used here for the CaCO3 export (Particulate
    Inorganic Carbon, PIC) since carbonate remineralization his handled by the
    carbonate_system_2(). CS2 will calculate the amount of CaCO3 that is
    dissolved and add this to the deep-box. The amount that buried is thus
    implicit.  This is equivalent to export all CaCO3 into the deeb-box
    (i.e. bp="None"), and then creating an explicit connection that describes
    burial into the sediment.  Since these a fixed rates, they could also be
    combined into one flux for DIC and one flux for TA.
    """

    M.OM_export = Q_("200 Tmol/yr")  # convert into Quantity so we can multiply
    M.CaCO3_export = Q_("60 Tmol/yr")  # with 2 for the alkalinity term

    # Fluxes going into deep box
    connection_dict = {
        "L_b_to_D_b@POM": {  # DIC from organic matter F5
            "sp": M.DIC,
            "ty": "Fixed",
            "ra": M.OM_export,
        },
        "L_b_to_D_b@PIC_DIC": {  # DIC from CaCO3 F6
            "sp": M.DIC,
            "ty": "Fixed",
            "ra": M.CaCO3_export,
            "bp": "sink",
        },
        "L_b_to_D_b@PIC_TA": {  # TA from CaCO3 F6
            "sp": M.TA,
            "ty": "Fixed",
            "ra": M.CaCO3_export * 2,
            "bp": "sink",
        },
    }
    create_bulk_connections(connection_dict, M)

    """ #--------------- Carbonate chemisrty  -------#
    
    To setup carbonate chemistry, one needs to know the soource adn sink of the
    export production fluxes. We thus keep two lists one for the surface boxes
    and one for the deep boxes.
    
    Carbonate system ` calculates tracer like CO2aq,  CO3, and H+, for the
    surface boxes.

    Carbonate System calculates the above tracers, and additionally
    the critical depth intervals (Saturation, CCD, and snowline), as well
    as the amount of carbonate that it dissolved.
    """
    surface_boxes: list = [M.L_b, M.H_b]
    deep_boxes: list = [M.D_b]
    ef = M.flux_summary(filter_by="PIC_DIC", return_list=True)
    add_carbonate_system_1(surface_boxes)

    add_carbonate_system_2(  # F6/F2
        r_sb=surface_boxes,  # list of reservoir groups
        r_db=deep_boxes,  # list of reservoir groups
        carbonate_export_fluxes=ef,  # list of export fluxes
        z0=-200,  # depth of shelf
        alpha=alpha,  # dissolution coefficient
    )
    # -------------------- Atmosphere -------------------------
    GasReservoir(
        name="CO2_At",
        species=M.CO2,
        species_ppm="280 ppm",
    )

    """ GasExchange connections currently do not support the setup
    with the ConnectionsProperties class, since they connect CO2 to
    DIC which fools the automatic species matching logic. As such
    we use the Species2Species class to create the connection
    explicitly.
    """
    pv = "4.8 m/d"  # piston velocity

    Species2Species(  # High Latitude surface to atmosphere F8
        source=M.CO2_At,  # Reservoir Species
        sink=M.H_b.DIC,  # Reservoir Species
        species=M.CO2,
        piston_velocity=pv,
        ctype="gasexchange",
        id="H_b",
    )

    Species2Species(  # Low Latitude surface to atmosphere F7
        source=M.CO2_At,  # Reservoir Species
        sink=M.L_b.DIC,  # Reservoir Species
        species=M.CO2,
        piston_velocity=pv,
        ctype="gasexchange",
        id="L_b",
    )

    # create the weathering fluxes F1
    ConnectionProperties(
        source=M.Fw,
        sink=M.L_b,
        rate={M.DIC: "12 Tmol/a", M.TA: "24 Tmol/a"},
        species=[M.DIC, M.TA],
        ctype="fixed",
        id="weathering",
    )

    return M
