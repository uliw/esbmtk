from __future__ import annotations
import typing as tp
import numpy as np

if tp.TYPE_CHECKING:
    from esbmtk import Model, ConnectionGroup

def initialize_model(rain_ratio, alpha, run_time, time_step, debug):
    """Package the model definition inside a function."""
    from esbmtk import (
        Q_,
        GasReservoir,
        Model,
        Species2Species,
        add_carbonate_system_1,
        add_carbonate_system_4,
        create_bulk_connections,
        initialize_reservoirs,
    )
    from LOSCAR_helper_functions import (
        create_connections_from_flux_list,
        create_gas_exchange_connections,
        create_weathering_fluxes,
    )

    M = Model(
        stop=run_time,  # end time of model
        max_timestep=time_step,  # time step
        element=[  # list of elements we consider in the model
            "Carbon",
            "Boron",
            "Hydrogen",
            "Phosphor",
            "Oxygen",
            "misc_variables",  # needed for plotting depth data
        ],
        mass_unit="mol",
        concentration_unit="mol/kg",
        opt_k_carbonic=13,  # Use Millero 2006
        opt_pH_scale=1,
        opt_buffers_mode=2,  # 1:total, 3:free scale
        debug=debug,
    )

    # -------------------- Set up box parameters ------------------------ #

    # weathering fluxes at steady state
    M.Fw_Ca = Q_("12 Tmol/yr")  # Carbonate weathering @280 ppm
    M.Fw_v = Q_("5 Tmol/yr")  # Volcanic flux
    M.Fw_Si = M.Fw_v  # Silicate weathering @280 ppm

    M.PC_ratio = 130
    M.OM_frac = -28
    M.PUE = 0.8
    M.NC_ratio = 15 / 130
    M.O2C_ratio = 165 / 130  # oxygen consumption per mol C

    M.ib_remin = 0.78
    M.db_remin = 1 - M.ib_remin

    M.rain = rain_ratio
    M.alpha = alpha

    # ------- setup box parameters ----------
    A_ap = 0.26  # Area percentage Atlantic ocean
    I_ap = 0.18  # Area percentage Indian ocean
    P_ap = 0.46  # Area precentage Pacific ocean
    H_ap = 0.10  # Area percentage High latidude ocean

    # ---------- initialize boxes ------------------------- #

    # initialize reservoirs
    bn: dict = {  # name: [[geometry], T, P, S]
        # Atlantic Ocean
        "A_sb": {
            "g": [0, -100, A_ap],
            "c": {
                M.DIC: "2210 umol/kg",
                M.TA: "2310 umol/kg",
                M.PO4: "2.1 umol/kg",
                M.O2: "200 umol/kg",
            },
            "T": 20,
            "P": 5,
            "S": 34.7,
        },
        "A_ib": {
            "g": [-100, -1000, A_ap],
            "c": {
                M.DIC: "2210 umol/kg",
                M.TA: "2310 umol/kg",
                M.PO4: "2.1 umol/kg",
                M.O2: "200 umol/kg",
            },
            "T": 10,
            "P": 100,
            "S": 34.7,
        },
        "A_db": {
            "g": [-1000, -6000, A_ap],
            "c": {
                M.DIC: "2210 umol/kg",
                M.TA: "2310 umol/kg",
                M.PO4: "2.1 umol/kg",
                M.O2: "200 umol/kg",
            },
            "T": 2,
            "P": 240,
            "S": 34.7,
        },
        "A_bb": { #burial box, 
            "g": [-6000, -6500, A_ap],
            "c": {
                M.DIC: "0 umol/kg",
                M.TA: "0 umol/kg",
                M.PO4: "0 umol/kg",
                M.O2: "0 umol/kg",
            },
            "T": 2,
            "P": 240,
            "S": 34.7,
        },
        # Indian Ocean
        "I_sb": {
            "g": [0, -100, I_ap],
            "c": {
                M.DIC: "2210 umol/kg",
                M.TA: "2310 umol/kg",
                M.PO4: "2.1 umol/kg",
                M.O2: "200 umol/kg",
            },
            "T": 20,
            "P": 5,
            "S": 34.7,
        },
        "I_ib": {
            "g": [-100, -1000, I_ap],
            "c": {
                M.DIC: "2210 umol/kg",
                M.TA: "2310 umol/kg",
                M.PO4: "2.1 umol/kg",
                M.O2: "200 umol/kg",
            },
            "T": 10,
            "P": 100,
            "S": 34.7,
        },
        "I_db": {
            "g": [-1000, -6000, I_ap],
            "c": {
                M.DIC: "2210 umol/kg",
                M.TA: "2310 umol/kg",
                M.PO4: "2.1 umol/kg",
                M.O2: "200 umol/kg",
            },
            "T": 2,
            "P": 240,
            "S": 34.7,
        },
        "I_bb": {
            "g": [-6000, -6500, I_ap],
            "c": {
                M.DIC: "0 umol/kg",
                M.TA: "0 umol/kg",
                M.PO4: "0 umol/kg",
                M.O2: "0 umol/kg",
            },
            "T": 2,
            "P": 240,
            "S": 34.7,
        },
        # Pacific Ocean
        "P_sb": {
            "g": [0, -100, P_ap],
            "c": {
                M.DIC: "2210 umol/kg",
                M.TA: "2310 umol/kg",
                M.PO4: "2.1 umol/kg",
                M.O2: "200 umol/kg",
            },
            "T": 20,
            "P": 5,
            "S": 34.7,
        },
        "P_ib": {
            "g": [-100, -1000, P_ap],
            "c": {
                M.DIC: "2210 umol/kg",
                M.TA: "2310 umol/kg",
                M.PO4: "2.1 umol/kg",
                M.O2: "200 umol/kg",
            },
            "T": 10,
            "P": 100,
            "S": 34.7,
        },
        "P_db": {
            "g": [-1000, -6000, P_ap],
            "c": {
                M.DIC: "2210 umol/kg",
                M.TA: "2310 umol/kg",
                M.PO4: "2.1 umol/kg",
                M.O2: "200 umol/kg",
            },
            "T": 2,
            "P": 240,
            "S": 34.7,
        },
        "P_bb": {
            "g": [-6000, -6500, P_ap],
            "c": {
                M.DIC: "0 umol/kg",
                M.TA: "0 umol/kg",
                M.PO4: "0 umol/kg",
                M.O2: "0 umol/kg",
            },
            "T": 2,
            "P": 240,
            "S": 34.7,
        },
        # High latitude box
        "H_sb": {
            "g": [0, -250, H_ap],
            "c": {
                M.DIC: "2210 umol/kg",
                M.TA: "2310 umol/kg",
                M.PO4: "2.1 umol/kg",
                M.O2: "200 umol/kg",
            },
            "T": 2,
            "P": 10,
            "S": 34.7,
        },
        # Weathering sources
        "Fw": {"ty": "Source", "sp": [M.DIC, M.TA, M.PO4, M.O2]},
        # Burial Sinks
        "Fb": {"ty": "Sink", "sp": [M.DIC, M.TA, M.PO4, M.O2]},
    }

    species_list = initialize_reservoirs(M, bn)

    # gas reservoirs
    GasReservoir(
        name="CO2_At",
        species=M.CO2,
        species_ppm="280 ppm",
    )

    GasReservoir(
        name="O2_At",
        species=M.O2,
        species_ppm="21 percent",
    )

    # ----- set up transport matrix -------------------- #
    thc = Q_("20*Sv")
    ta = 0.2
    ti = 0.2

    connection_dict = {
        # source_to_sink@id
        # thermohaline, upwelling, and advection
        "H_sb_to_A_db@thermohaline": {
            "ty": "scale_with_concentration",
            "sc": thc,
            "sp": species_list,
        },
        "A_ib_to_H_sb@thermohaline": {
            "ty": "scale_with_concentration",
            "sc": thc,
            "sp": species_list,
        },
        "A_db_to_A_ib@upwelling": {
            "ty": "scale_with_concentration",
            "sc": ta * thc,
            "sp": species_list,
        },
        "I_db_to_I_ib@upwelling": {
            "ty": "scale_with_concentration",
            "sc": ti * thc,
            "sp": species_list,
        },
        "A_db_to_I_db@thermohaline": {
            "ty": "scale_with_concentration",
            "sc": (1 - ta) * thc,
            "sp": species_list,
        },
        "I_db_to_P_db@thermohaline": {
            "ty": "scale_with_concentration",
            "sc": (1 - ta - ti) * thc,
            "sp": species_list,
        },
        "P_db_to_P_ib@thermohaline": {
            "ty": "scale_with_concentration",
            "sc": (1 - ta - ti) * thc,
            "sp": species_list,
        },
        "P_ib_to_I_ib@thermohaline": {
            "ty": "scale_with_concentration",
            "sc": (1 - ta - ti) * thc,
            "sp": species_list,
        },
        "I_ib_to_A_ib@thermohaline": {
            "ty": "scale_with_concentration",
            "sc": (1 - ta) * thc,
            "sp": species_list,
        },
        # surface/intemediate water mixing
        "A_ib_to_A_sb@mix_up": {
            "ty": "scale_with_concentration",
            "sc": "21 Sverdrup",
            "sp": species_list,
        },
        "A_sb_to_A_ib@mix_down": {
            "ty": "scale_with_concentration",
            "sc": "21 Sverdrup",
            "sp": species_list,
        },
        "I_ib_to_I_sb@mix_up": {
            "ty": "scale_with_concentration",
            "sc": "17 Sverdrup",
            "sp": species_list,
        },
        "I_sb_to_I_ib@mix_down": {
            "ty": "scale_with_concentration",
            "sc": "17 Sverdrup",
            "sp": species_list,
        },
        "P_ib_to_P_sb@mix_up": {
            "ty": "scale_with_concentration",
            "sc": "25 Sverdrup",
            "sp": species_list,
        },
        "P_sb_to_P_ib@mix_down": {
            "ty": "scale_with_concentration",
            "sc": "25 Sverdrup",
            "sp": species_list,
        },
        # deep/high box mixing
        "A_db_to_H_sb@mix_up": {
            "ty": "scale_with_concentration",
            "sc": "4 Sverdrup",
            "sp": species_list,
        },
        "H_sb_to_A_db@mix_down": {
            "ty": "scale_with_concentration",
            "sc": "4 Sverdrup",
            "sp": species_list,
        },
        "I_db_to_H_sb@mix_up": {
            "ty": "scale_with_concentration",
            "sc": "3 Sverdrup",
            "sp": species_list,
        },
        "H_sb_to_I_db@mix_down": {
            "ty": "scale_with_concentration",
            "sc": "3 Sverdrup",
            "sp": species_list,
        },
        "P_db_to_H_sb@mix_up": {
            "ty": "scale_with_concentration",
            "sc": "10 Sverdrup",
            "sp": species_list,
        },
        "H_sb_to_P_db@mix_down": {
            "ty": "scale_with_concentration",
            "sc": "10 Sverdrup",
            "sp": species_list,
        },
    }
    create_bulk_connections(connection_dict, M)

    # ---------------------  weathering fluxes ----------------- #
    # unitless weathering strength
    Species2Species(
        ctype="weathering",
        source=M.Fw.DIC,  # source of flux
        sink=M.Fb.DIC,
        species=M.DIC,
        reservoir_ref=M.CO2_At,  # pCO2
        scale=1.0,  # optional, defaults to 1
        ex=0.4,  # exponent c
        pco2_0="280 ppm",  # reference pCO2
        rate=M.Fw_Ca,  # rate at pco2_0
        id="weathering_carbonate",
    )
    Species2Species(  # CaSiO3 weathering
        ctype="weathering",
        source=M.Fw.DIC,  # source of flux
        sink=M.Fb.DIC,
        species=M.DIC,
        reservoir_ref=M.CO2_At,  # pCO2
        scale=1.00,  # optional, defaults to 1
        ex=0.2,  # exponent c
        pco2_0="280 ppm",  # reference pCO2
        rate=M.Fw_Si,  # rate at pco2_0
        id="weathering_silicate",
    )
    # volcanic flux:
    Species2Species(  # Low Latitude surface to atmosphere F7
        source=M.Fw.DIC,  # Reservoir Species
        sink=M.CO2_At,  # Reservoir Species
        species=M.CO2,
        ctype="Fixed",
        rate=M.Fw_v,
        id="volcanic_weathering",
    )

    # setup basin specific fluxes
    areas = {
        "A_sb": A_ap / (1 - H_ap),
        "P_sb": P_ap / (1 - H_ap),
        "I_sb": I_ap / (1 - H_ap),
    }

    """Carbonate weathering removes one C from the crust, and one C from the atmosphere.
    Carbonate precipitation returns one C back to the atmosphere, so there is not
    removal.  However, we keep the C from carbonate weathering as this will be removed
    through carbonate sedimentation.

    Silicate weathering takes both C from the atmosphere, but one is returned, so there
    is a net removal of 1 C, which is compensated by the volcanic flux.

    Both processes contribute 2 mol alkalinity for each mol Carbon, since calcium
    carries a double charge.
    """
    create_weathering_fluxes(M, M.DIC, areas, "weathering_carbonate", 1, source="crust")
    create_weathering_fluxes(M, M.TA, areas, "weathering_carbonate", 2, source="crust")

    create_weathering_fluxes(M, M.DIC, areas, "weathering_silicate", 1, source="atmosphere")
    create_weathering_fluxes(M, M.TA, areas, "weathering_silicate", 2, source="crust")

    # -------- biological pump particular P export ---------------------- #
    # low latitude export flux = 80% of upwelling PO4
    pfluxes = M.flux_summary(filter_by="PO4_mix_up", exclude="H_", return_list=True)

    # Export productivity in the high latidude box is fixed (after Zeebe)
    # to mimic iron limitation.
    pp_hl = Q_(f"{1.8 * M.H_sb.area.magnitude / M.PC_ratio} mol/a")

    # Particulate (OM bound) phosphate export productivity in the low latidude boxes
    ct = {  # Surface box to ib, about 78% is remineralized in the ib
        (
            "A_sb_to_A_ib@A_sb_2_A_ib_POP_ex",
            "I_sb_to_I_ib@I_sb_2_I_ib_POP_ex",
            "P_sb_to_P_ib@P_sb_2_P_ib_POP_ex",
        ): {
            "ty": "scale_with_flux",
            "sc": M.PUE * M.ib_remin,
            "re": pfluxes,
            "sp": M.PO4,
        },
        # surface box to deep box
        (
            "A_sb_to_A_db@A_sb_2_A_db_POP_ex",
            "I_sb_to_I_db@I_sb_2_I_db_POP_ex",
            "P_sb_to_P_db@P_sb_2_P_db_POP_ex",
        ): {
            "ty": "scale_with_flux",
            "sc": M.PUE * M.db_remin,
            "re": pfluxes,
            "sp": M.PO4,
        },
        # high latitude box to deep ocean boxes POP
        (
            "H_sb_to_A_db@H_sb_2_A_db_POP_ex",
            "H_sb_to_I_db@H_sb_2_I_db_POP_ex",
            "H_sb_to_P_db@H_sb_2_P_db_POP_ex",
        ): {
            # here we use a fixed rate following Zeebe's Loscar model
            "ra": [
                pp_hl * 0.3,
                pp_hl * 0.3,
                pp_hl * 0.4,
            ],
            "sp": M.PO4,
            "ty": "Regular",
        },
    }
    create_bulk_connections(ct, M)

    # -------------- biological pump particulate organic matter export -------- #
    """OM export transports DIC from the surface to the sink, and TA from the sink to
    the surface.  OM mineralization/photosynthesis also consumes/produces O2.  Since O2
    is consumed in the deep box, and produced in the surface box, we prefix the scale
    with -1.  Here we create these fluxes from the existing particulate organic phosphor
    fluxes.

    Since POP and POM remineralization is the same, we can directly use the POP export
    fluxes to calculate particulate organic matter = POP * M.PC_ratio
    """
    pfluxes = M.flux_summary(filter_by="POP_ex", return_list=True)
    # Particulate OM DIC
    create_connections_from_flux_list(
        M,  # model
        pfluxes,  # flux list
        "POM_DIC",  # new ID
        M.DIC,  # species
        M.PC_ratio,  # scale
    )
    # Particulate OM TA from Nitrate
    create_connections_from_flux_list(
        M, pfluxes, "POM_TA", M.TA, M.PC_ratio * M.NC_ratio * -1
    )
    # Particulate OM O2
    create_connections_from_flux_list(
        M, pfluxes, "POM_O2", M.O2, M.PC_ratio * M.O2C_ratio * -1
    )

    # -------------- biological pump carbonate export --------------------------- #
    """Assumptions:
    - Shallow water CaCO3 production is buried on the shelf
    - High latitude CaCO3 production reaches the deep ocean in dissolved form
    - The intermediate and deep water CaCO3 export is assumed to be buried in full, except for the
      dissolution flux computed by carbonate system 4.

    Since OM and CaCO3 mineralization behave differently, the 
    export production explicity based on the upwelling phosphate
    """
    # CaCO3 export to shelf is based on export primary productivity
    upwelling = M.flux_summary(filter_by="PO4_mix_up", exclude="H_", return_list=True)
    # get shelf fraction
    M.shelf_fraction = M.A_sb.sed_area.magnitude / M.A_sb.area.magnitude
    create_connections_from_flux_list(
        M,
        upwelling,
        "PIC_DIC_shelf",
        M.DIC,
        M.PUE * M.shelf_fraction * M.PC_ratio / M.rain,
        source="from_sink",
        sink="Fb",
    )
    create_connections_from_flux_list(
        M,
        upwelling,
        "PIC_TA_shelf",
        M.TA,
        M.PUE * 2 * M.shelf_fraction * M.PC_ratio / M.rain,
        source="from_sink",
        sink="Fb",
    )

    # CaCO3 export from low lat ocean to intermediate ocean.
    M.int_fraction = 1 - M.shelf_fraction
    create_connections_from_flux_list(
        M,
        upwelling,
        "PIC_DIC_int",
        M.DIC,
        M.PUE * M.PC_ratio * M.int_fraction / M.rain,
        source="from_sink",
        sink="Fb",
    )
    create_connections_from_flux_list(
        M,
        upwelling,
        "PIC_TA_int",
        M.TA,
        2 * M.PUE * M.PC_ratio * M.int_fraction / M.rain,
        source="from_sink",
        sink="Fb",
    )
    # CaCO3 export from high latitude to deep ocean Here we use the PO4 export
    # productivity directly, so no need to scale with the uptake efficiency.
    pfluxes = M.flux_summary(filter_by="POP_ex H_", return_list=True)
    create_connections_from_flux_list(
        M,
        pfluxes,
        "PIC_DIC_hl",
        M.DIC,
        M.PC_ratio / M.rain,
    )
    create_connections_from_flux_list(
        M,
        pfluxes,
        "PIC_TA_hl",
        M.TA,
        2 * M.PC_ratio / M.rain,
    )

    # calculate intermediate and deep sea carbonate dissolution
    # FIXME: The filtering routine needs to be more robust and better logic
    cef = M.flux_summary(filter_by="PIC_DIC_int", return_list=True)
    M.cef = cef
  
    add_carbonate_system_4(
        this_box=[M.A_ib, M.I_ib, M.P_ib],  # intermediate boxes where the carbonate export flux gets added
        source_box=[M.A_sb, M.I_sb, M.P_sb],  # corresponding surface boxes
        next_box=[M.A_db, M.I_db, M.P_db],  #deep boxes 
        burial_box=[M.A_bb, M.I_bb, M.P_bb],
        carbonate_export_fluxes=cef,
        zsat_min=-100,
        z0=-100,
        zint=-1000,
        alpha=alpha,
    )

    # calculate carbonate system parameters for the surface and intermediate boxes
    add_carbonate_system_1([M.A_sb, M.I_sb, M.P_sb, M.H_sb, M.A_ib, M.I_ib, M.P_ib])

    # ------------------ Air Sea Gas Exchange --------------------- #
    create_gas_exchange_connections(  # CO2
        M,
        [M.A_sb, M.I_sb, M.P_sb, M.H_sb],
        M.CO2,
        "4.8 m/d",  # piston velocity
    )

    create_gas_exchange_connections(  # O2
        M,
        [M.A_sb, M.I_sb, M.P_sb, M.H_sb],
        M.O2,
        "4.8 m/d",  # piston velocity
    )

    return M


def pp_carbonate_cs4(M: Model, ocean_names: list) -> None:
    """Calculate marine carbonate chemistry. Essentially a helper function for post_processing. 

    Surface and intermediate boxes use CS1, 
    deep boxes use CS4 (deep-box-only carbonate dissolution).

    :param M: Model Instance
    :param ocean_names: List of ocean names, e.g., ["A", "I", "P"]
    """
    from esbmtk import carbonate_system_1_pp, carbonate_system_4_pp
    import numpy as np

    for o in ocean_names:
        # --- Get box handles ---
        sb = eval(f"M.{o}_sb")  # surface
        ib = eval(f"M.{o}_ib")  # intermediate
        db = eval(f"M.{o}_db")  # deep

        # --- Calculate carbonate species in surface & intermediate boxes ---
        carbonate_system_1_pp(sb)
        carbonate_system_1_pp(ib)

        c_name = f"Conn_{ib.name}_to_{sb.name}_PO4_mix_up"
        C: ConnectionGroup = M.connection_summary(filter_by=c_name, return_list=True)[0]
        F_PO4 = C.scale * ib.PO4.c

        # calculate CaCO3 export productivity
        ep = F_PO4 * M.PUE * M.PC_ratio * M.int_fraction / M.rain
        carbonate_system_4_pp(db, ep)
    


if __name__ == "__main__":
    from LOSCAR_helper_functions import get_matrix_coefficients

    run_time = "1 Myr"
    time_step = "1 kyr"
    rain_ratio = 6.1
    alpha = 0.3
    debug = True

    M = initialize_model(rain_ratio, alpha, run_time, time_step, debug)

    M.debug_equations_file = False

    M.run()

    '''
    M.plot([M.CO2_At])
    M.plot([M.A_sb.PO4, M.A_ib.PO4, M.A_db.PO4])
    M.plot([M.H_sb.DIC, M.A_sb.DIC, M.A_ib.DIC, M.A_db.DIC])
    M.plot([M.H_sb.TA, M.A_sb.TA, M.A_ib.TA, M.A_db.TA])


    # ---- sanity checks ---
    search_terms = ["shelf", "slope", "deep"]
    for f_name in M.F_names:
        if any(term in f_name for term in search_terms):
            coeff = get_matrix_coefficients(f_name, M.CM, M.F, M.F_names, M.R_names)
            print(coeff)
    '''



