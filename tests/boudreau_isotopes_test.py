"""4 box earth system model with conventional P-cycling.

This is based on the Boudreau 2010 model, and additionally
considers organic matter (OM) burial. OM weathering is
mutiple of the PO4 weathering flux, which in turn depends
on the CaCO3 weathering rate which is a function of pCO2.
The model assumes that silicate weathering and volcanic
flux are in balance and can thus be ignored.
"""


def initialize_model_geometry(rain_ratio, alpha, run_time, time_step, hl_mixing, debug):
    """Initialize the Model Geometry."""
    from esbmtk import (
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
        element=[
            "Carbon",
            "Boron",
            "Hydrogen",
            "Phosphor",
            "Oxygen",
            "misc_variables",
        ],
        mass_unit="mol",
        volume_unit="liter",
        concentration_unit="mol/kg",
        isotopes=False,
        opt_k_carbonic=13,  # Use Millero 2006
        opt_pH_scale=3,  # 1:total, 3:free scale
        debug=debug,
    )

    # -------------------- Set up boxes  ------------------------- #
    """ Boudreau et al defined their reservoirs by area and and volume, and
    explicitly assign temperature, pressure and salinity.
    """
    box_parameters: dict = {  # name: [[geometry], T, P]
        "H_b": {  # High-Lat Box
            "c": {
                M.DIC: "2153 umol/kg",
                M.TA: "2345 umol/kg",
                M.PO4: "1.5 umol/kg",
                M.O2: "200 umol/kg",
            },
            "g": [0, -400, 0.15],
            "T": 2,
            "P": 10,
            "S": 34.7,
            "d": {M.DIC: 0},
        },
        "L_b": {  # Low-Lat Box
            "c": {
                M.DIC: "1952 umol/kg",
                M.TA: "2288 umol/kg",
                M.PO4: "1.5 umol/kg",
                M.O2: "200 umol/kg",
            },
            "g": [0, -100, 0.85],
            "T": 20,
            "P": 5,
            "S": 34.7,
            "d": {M.DIC: 0},
        },
        "D_b": {  # Deep Box
            "c": {
                M.DIC: "2291 umol/kg",
                M.TA: "2399 umol/kg",
                M.PO4: "1.5 umol/kg",
                M.O2: "200 umol/kg",
            },
            "g": [-100, -10000, 1],
            "T": 2,
            "P": 240,
            "S": 34.7,
            "d": {M.DIC: 0},
        },  # sources and sinks
        "Fw": {
            "ty": "Source",
            "sp": [M.DIC, M.TA, M.PO4, M.O2],
            "d": {M.DIC: 0},
        },
        "Fb": {
            "ty": "Sink",
            "sp": [M.DIC, M.TA, M.PO4, M.O2],
        },
    }
    species_list = initialize_reservoirs(M, box_parameters)

    # -------------------- Atmosphere ------------------------- #
    GasReservoir(
        name="CO2_At",
        species=M.CO2,
        species_ppm="280 ppm",
        delta=-7,
    )

    GasReservoir(
        name="O2_At",
        species=M.O2,
        species_ppm="21 percent",
    )

    """ Define the mixing and thermohaline fluxes 
    through a dictionary that specifies the respective source and sink
    reservoirs, connection id,  the connection type, the scaling factor
    and the list of species that will be affected. 
    """
    connection_dict = {
        # source_to_sink@id
        "H_b_to_D_b@mix_down": {  # High Lat mix down F4
            "ty": "scale_with_concentration",  # type
            "sc": hl_mixing,  # scale
            "sp": species_list,  # list of affected species
        },
        "D_b_to_H_b@mix_up": {  # High Lat mix up F4
            "ty": "scale_with_concentration",
            "sc": hl_mixing,
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

    # -------------------- Input Fluxes -------------------------------- #
    Species2Species(  # PO4 weathering
        ctype="weathering",
        source=M.Fw.PO4,  # source of flux
        sink=M.L_b.PO4,
        reservoir_ref=M.CO2_At,  # pCO2
        scale=1,  # optional, defaults to 1
        ex=0.2,  # exponent c
        pco2_0="280 ppm",  # reference pCO2
        rate="18 Gmol/a",  # rate at pco2_0
        id="weathering_po4",
    )
    Species2Species(  # DIC from weathering
        ctype="scale_with_flux",
        source=M.Fw.DIC,
        sink=M.L_b.DIC,
        ref_flux=M.flux_summary(filter_by="weathering_po4", return_list=True)[0],
        scale=12e3 / 18,  # about 12 Tmol/a
        delta=2.5,
        id="weathering_dic",
    )
    Species2Species(  # TA from weathering
        ctype="scale_with_flux",
        source=M.Fw.TA,
        sink=M.L_b.TA,
        ref_flux=M.flux_summary(filter_by="weathering_dic", return_list=True)[0],
        scale=2,
        id="weathering_TA",
    )

    # ------------------- export by primary productivity------------------
    # export of P in form of particles sinking from surface --> deep box

    M.remin_eff = 0.99  # PO4 remineralization efficiency
    M.tau = 10  # residence time in years
    M.redpc = 106  # red-field C/P ratio 1/106

    """ Volume is in liter, but concentration is mol/kg so we first
    need to convert liter to kilogram
    """
    # Phosphate
    Species2Species(  # Surface to deep box PO4 productivity
        ctype="scale_with_concentration",
        source=M.L_b.PO4,
        sink=M.D_b.PO4,
        scale=M.L_b.PO4.volume
        * M.remin_eff
        / M.tau
        * M.L_b.swc.density
        / 1000,  # concentration * volume = mass * 1/tau
        id="po4_productivity",
    )

    # Particulate organic matter POM
    Species2Species(  # DIC from POM to deep box
        ctype="scale_with_flux",
        source=M.L_b.DIC,
        sink=M.D_b.DIC,
        ref_flux=M.flux_summary(filter_by="po4_productivity", return_list=True)[0],
        scale=M.redpc / M.remin_eff,
        id="POM_DIC",
        epsilon=-28,
    )

    # Phosphate that is exported an buried
    Species2Species(  # surface to sediment
        ctype="scale_with_flux",
        source=M.L_b.PO4,
        sink=M.Fb.PO4,
        ref_flux=M.flux_summary(filter_by="po4_productivity", return_list=True)[0],
        scale=1 - M.remin_eff,  # burial of ~1% P
        id="burial_po4",
    )

    """ when OM is remineralized, O2 is consumed and it produces CO2
    CO2 is used in photosynthesis to produce OM and O2 but O2 goes from
    deep reservoirs to the surface boxes"""
    Species2Species(
        source=M.D_b.O2,
        sink=M.L_b.O2,
        ctype="scale_with_flux",
        ref_flux=M.flux_summary(filter_by="POM_DIC", return_list=True)[0],
        scale=165 / 130,  # ratio of o2 consumed per mol C mineralized (Zeebe, 2012)
        id="OM_remin",  # OM remineralization
    )

    """CaCO3 export: Remineralization and CaCO3 burial in the deep box are
    handled by carbonate systen 2. So we set the bypass option.
    Carbonate system 1 computes pH and CO2aq.
    """
    Species2Species(  # DIC from CaCO3
        source=M.L_b.DIC,
        sink=M.D_b.DIC,
        ctype="scale_with_flux",
        ref_flux=M.flux_summary(filter_by="po4_productivity", return_list=True)[0],
        scale=rain_ratio * M.redpc,
        id="PIC_DIC",
        bypass="sink",
    )
    Species2Species(  # TA from CaCO3 pp
        source=M.L_b.TA,
        sink=M.D_b.TA,
        ctype="scale_with_flux",
        ref_flux=M.flux_summary(filter_by="PIC_DIC", return_list=True)[0],
        scale=2,  # 1 mol DIC = 2 mol TA
        id="PIC_TA",
        bypass="sink",
    )
    surface_boxes: list = [M.L_b, M.H_b]
    deep_boxes: list = [M.D_b]
    add_carbonate_system_1(surface_boxes)
    add_carbonate_system_2(
        r_sb=[M.L_b],  # corresponding surface boxes
        r_db=deep_boxes,  # deep boxes where we add cs2
        carbonate_export_fluxes=M.flux_summary(filter_by="PIC_DIC", return_list=True),
        z0=-100,
        alpha=alpha,
    )

    # -------------------- Gas Exchange ------------------------ #
    Species2Species(  # CO2 High Latitude surface to atmosphere F8
        source=M.CO2_At,  # Reservoir Species
        sink=M.H_b.DIC,  # Reservoir Species
        species=M.CO2,
        piston_velocity="4.8 m/d",
        ctype="gasexchange",
        id="H_b_gex_CO2",
    )

    Species2Species(  # O2 High Latitude surface to atmosphere F8
        source=M.O2_At,  # Reservoir Species
        sink=M.H_b.O2,  # Reservoir Species
        species=M.O2,
        piston_velocity="4.8 m/d",
        ctype="gasexchange",
        id="H_b_gex_O2",
    )

    Species2Species(  # CO2 Low Latitude surface to atmosphere F7
        source=M.CO2_At,  # Reservoir Species
        sink=M.L_b.DIC,  # Reservoir Species
        species=M.CO2,
        piston_velocity="4.8 m/d",
        ctype="gasexchange",
        id="L_b_gex_CO2",
    )

    Species2Species(  # O2 Low Latitude surface to atmosphere F7
        source=M.O2_At,  # Reservoir Species
        sink=M.L_b.O2,  # Reservoir Species
        species=M.O2,
        piston_velocity="4.8 m/d",
        ctype="gasexchange",
        id="L_b_gex_O2",
    )
    return M
