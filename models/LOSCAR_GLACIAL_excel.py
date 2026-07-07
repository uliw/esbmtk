from __future__ import annotations
import typing as tp


if tp.TYPE_CHECKING:
    from esbmtk import Model, ConnectionGroup

def initialize_model(high_lat_piston, high_lat_PO4_export, T_surf, T_deep, thc, ta, ti, mix_A_H, mix_I_H, mix_P_H, rain_ratio, alpha, run_time, time_step, debug):
    """Package the model definition inside a function."""
    from esbmtk import (
        Q_,
        Model,
        Species2Species,
        add_carbonate_system_1,
        add_carbonate_system_4,
        create_bulk_connections,
        create_reservoirs_from_excel,
        create_gas_reservoirs_from_excel,
        create_transport_matrix_from_excel,
        create_gas_exchange_connections_from_excel,
    )
    from LOSCAR_helper_functions import (
        create_connections_from_flux_list,
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

    # -------------------- Set up biogeochemical constants ------------------------ #

    # weathering fluxes at steady state
    M.Fw_Ca = Q_("12 Tmol/yr")  # Carbonate weathering @280 ppm
    M.Fw_v = Q_("5 Tmol/yr")  # Volcanic flux
    M.Fw_Si = M.Fw_v  # Silicate weathering @280 ppm

    M.PC_ratio = 130
    M.OM_frac = -28
    M.PUE = 0.8
    M.NC_ratio = 15 / 130
    M.O2C_ratio = 165 / 130  # oxygen consumption per mol C

    # Isotope ratios
    M.Fw_DIC_d = 1.5  # Carbonate weathering delta
    M.Fw_v_d = -4  # Volcanic flux delta
    M.OM_frac = -28  # fractionation during photosynthesis
    M.CO2_DIC_a = 8.0  # enrichment during CO2 dissolution in water
    M.Fb_DIC_d = 0 

    M.ib_remin = 0.78
    M.db_remin = 1 - M.ib_remin

    M.rain = rain_ratio
    M.alpha = alpha

    M.high_lat_piston = high_lat_piston

    M.high_lat_PO4_export = high_lat_PO4_export

    # ------- setup box parameters ----------#

    A_ap = 0.26  # Area percentage Atlantic ocean
    I_ap = 0.18  # Area percentage Indian ocean
    P_ap = 0.46  # Area precentage Pacific ocean
    H_ap = 0.10  # Area percentage High latidude ocean

    M.T_surf = T_surf
    M.T_deep = T_deep

    thc = Q_(thc)
    mix_A_H = Q_(mix_A_H)
    mix_I_H = Q_(mix_I_H)
    mix_P_H = Q_(mix_P_H)

    # Attach to model for logging
    M.thc = thc
    M.ta = ta
    M.ti = ti
    M.mix_A_H = mix_A_H
    M.mix_I_H = mix_I_H
    M.mix_P_H = mix_P_H

    species_list = create_reservoirs_from_excel(
        M, #Model object
        "/home/atlas/esbmtk/esbmtk/models/LOSCAR_sheets/LOSCAR_sheets.xlsx", #specify file path
        sheet_name="reservoirs" #specify worksheet (default = "reservoirs")
    )

    M.Fw.DIC.delta = M.Fw_DIC_d #initialize delta for Source object
    M.Fb.DIC.delta = M.Fb_DIC_d #initialize delta fro Sink object

    create_gas_reservoirs_from_excel(
        M, #Model object
        "/home/atlas/esbmtk/esbmtk/models/LOSCAR_sheets/LOSCAR_sheets.xlsx", #specify file path
        sheet_name="gas_reservoirs" #specify worksheet (default = "gas_reservoirs")
    )

    create_transport_matrix_from_excel(
        M, #Model object
        "/home/atlas/esbmtk/esbmtk/models/LOSCAR_sheets/LOSCAR_sheets.xlsx", #specify file path
        species_list, #list of species being transported via advection and mixing
        sheet_name="transport_matrix" #specify worksheet (default = "transport_matrix")
    )

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
        delta=M.Fw_v_d,
        id="volcanic_weathering",
    )

    # setup basin specific fluxes
    areas = {
        "A_sb": A_ap / (1 - H_ap),
        "P_sb": P_ap / (1 - H_ap),
        "I_sb": I_ap / (1 - H_ap),
    }

    """Carbonate weathering removes one C from the crust, and one C from the atmosphere.
    Carbonate precipitation returns one C back to the atmosphere, so there is no
    removal.  However, we keep the C from the crust as this will be removed
    through carbonate sedimentation.

    Silicate weathering takes both C from the atmosphere, but one is returned, so there
    is a net removal of 1 C, which is compensated by the volcanic flux.

    Both processes contribute 2 mol alkalinity for each mol Carbon, since calcium
    carries a double charge.
    """
    create_weathering_fluxes(M, M.DIC, areas, "weathering_carbonate", 1, source="crust", delta=M.Fw_DIC_d) 
    create_weathering_fluxes(M, M.TA, areas, "weathering_carbonate", 2, source="crust")

    create_weathering_fluxes(M, M.DIC, areas, "weathering_silicate", 1, source="atmosphere", alpha=M.CO2_DIC_a)
    create_weathering_fluxes(M, M.TA, areas, "weathering_silicate", 2, source="crust")

    # -------- biological pump particulate P export ---------------------- #
    # low latitude export flux = 80% of upwelling PO4
    pfluxes = M.flux_summary(filter_by="PO4_mix_up", exclude="H_", return_list=True)
    print(pfluxes)
   

    # Export productivity in the high latitude box is fixed (after Zeebe)
    # to mimic iron limitation.

    pp_hl = Q_(f"{high_lat_PO4_export * M.H_sb.area.magnitude / M.PC_ratio} mol/a")

    # Particulate (OM bound) phosphate export productivity in the low latitude boxes
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
    }

    create_bulk_connections(ct, M)
    
    # choose limitation regime: "iron" or "phosphate"
    limitation_regime = "iron"   # toggle this

    if limitation_regime == "iron":
        # iron-limited: fixed export rates (Zeebe, 2012)
        ct = {
            (
                "H_sb_to_A_db@H_sb_2_A_db_POP_ex",
                "H_sb_to_I_db@H_sb_2_I_db_POP_ex",
                "H_sb_to_P_db@H_sb_2_P_db_POP_ex",
            ): {
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

    elif limitation_regime == "phosphate":
        # phosphate-limited: scale with PO4 flux
        highlat_pfluxes = M.flux_summary(
            filter_by="H_sb_PO4_mix_up",
            return_list=True
        )

        ct = {
            (
                "H_sb_to_A_db@H_sb_2_A_db_POP_ex",
                "H_sb_to_I_db@H_sb_2_I_db_POP_ex",
                "H_sb_to_P_db@H_sb_2_P_db_POP_ex",
            ): {
                "ty": "scale_with_flux",
                "sc": M.PUE,
                "re": highlat_pfluxes,
                "sp": M.PO4,
            },
        }
        create_bulk_connections(ct, M)

    else:
        raise ValueError("limitation_regime must be 'iron' or 'phosphate'")

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
        delta=M.OM_frac, #fractionation of OM
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
    cef = [
    M.flux_summary(filter_by="A_sb_to_Fb_PIC_DIC_int", return_list=True)[0],
    M.flux_summary(filter_by="I_sb_to_Fb_PIC_DIC_int", return_list=True)[0],
    M.flux_summary(filter_by="P_sb_to_Fb_PIC_DIC_int", return_list=True)[0],
    ]
    M.cef = cef
    

    # calculate carbonate system parameters for the surface and intermediate boxes
    add_carbonate_system_1([M.A_sb, M.I_sb, M.P_sb, M.H_sb, M.A_ib, M.I_ib, M.P_ib])
  
    add_carbonate_system_4(
        this_box=[M.A_ib, M.I_ib, M.P_ib],  # intermediate boxes where the carbonate export flux gets added
        source_box=[M.A_sb, M.I_sb, M.P_sb],  # corresponding surface boxes
        next_box=[M.A_db, M.I_db, M.P_db],  #deep boxes 
        burial_box=[M.A_bb, M.I_bb, M.P_bb],
        carbonate_export_fluxes=cef,
        zsat_min=-100,
        z0=-100,
        zint=-1000,
        zmax=-6000,
        alpha=alpha,
    )

    #--------Air-Sea Gas Exchange----------#
    """Requires the initialization of carbonate_system_1 to work, and therefore 
    must be placed after add_carbonate_system_1 in the model definition.
    """
    create_gas_exchange_connections_from_excel(
        M, #Model object
        "/home/atlas/esbmtk/esbmtk/models/LOSCAR_sheets/LOSCAR_sheets.xlsx", #specify file path
        sheet_name="gas_exchange" #specify worksheet (default = "gas_exchange")
    )

    return M


def cs4_pp_helper(M: Model, ocean_names: list) -> None:
    """Model-specific helper function for carbonate_system_4 post-processing.

    carbonate_system_4_pp requires a CaCO3 export flux as input and therefore 
    cannot be applied directly to a reservoir without additional model-specific
    calculations for obtaining the export flux outside the model definition.

    This function applies carbonate_system_1_pp to surface and intermediate boxes for 
    each specified basin, calculates CaCO3 export fluxes for each basin, and then uses 
    carbonate_system_4_pp to obtain carbonate system diagnostics for the deep boxes.

    Parameters
    ----------
    M : Model
        ESBMTK model instance.
    ocean_names : list[str]
        Ocean basin identifiers (e.g. ``["A", "I", "P"]``).

    Returns
    -------
    None
        Carbonate diagnostics are attached to the corresponding
        reservoirs as ``VectorData`` objects.
    """
    from esbmtk import carbonate_system_1_pp, carbonate_system_4_pp

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

    run_time = "1 kyr"
    time_step = "100 yr"
    rain_ratio = 6.1
    alpha = 0.3
    debug = False
    T_surf = 20
    T_deep = 2
    thc = "20 Sv"
    ta = 0.2
    ti = 0.2
    mix_A_H = "4 Sv"
    mix_I_H = "3 Sv"
    mix_P_H = "10 Sv"
    high_lat_PO4_export=1.8
    high_lat_piston= "4.8m/d"

    M_glacial = initialize_model(
        high_lat_piston, high_lat_PO4_export, 
        T_surf, T_deep, thc, ta, ti, 
        mix_A_H, mix_I_H, mix_P_H, 
        rain_ratio, alpha, run_time, time_step, debug)


    M_glacial.read_state("modern_state.pkl")
    M_glacial.run()

    M_glacial.plot(M_glacial.CO2_At)

    
    

    




    
    
    
    

    
    







