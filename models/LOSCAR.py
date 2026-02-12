def initialize_model(rain_ratio, alpha, run_time, time_step):
    """Package the model definition inside a function."""
    from esbmtk import (
        Q_,
        ConnectionProperties,
        GasReservoir,
        Model,
        Species2Species,
        add_carbonate_system_1,
        add_carbonate_system_3,
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
            "Phosphor",
            "Oxygen",
            "misc_variables",  # needed for plotting depth data
        ],
        mass_unit="mol",
        concentration_unit="mol/kg",
        opt_k_carbonic=13,  # Use Millero 2006
        opt_pH_scale=1, 
        opt_buffers_mode=2, # 1:total, 3:free scale
    )

    # -------------------- Set up box parameters ------------------------ #

    # weathering fluxes at steady state
    M.Fw_v = Q_("5 Tmol/yr")  # Volcanic flux
    M.Fw_Si = M.Fw_v  # Silicate weathering @280 ppm
    M.Fw_Ca = Q_("12 Tmol/yr")  # Carbonate weathering @280 ppm
    M.Fw_P = Q_("30 Gmol/a")  # Phosphate weathering at 280 ppm (Filipelli 2002)

    M.PC_ratio = 130
    M.OM_frac = -28
    M.PUE = 0.8
    M.P_burial = 0.01  # about 1% of the exported P is buried in the deep ocean
    M.NC_ratio = 15 / 130

    M.ib_remin = 0.78 - M.P_burial / 2
    M.db_remin = 1 - M.ib_remin - M.P_burial / 2

    M.rain = rain_ratio
    M.alpha = alpha

    # ------- setup box parameters ----------
    A_ap = 0.26  # Area percentage Atlantic ocean
    I_ap = 0.18  # Area percentage Indian ocean
    P_ap = 0.46  # Area precentage Pacific ocean
    H_ap = 0.10  # Area percentage High latidude ocean
    
    box_parameters: dict = {  # name: [[geometry], T, P]
        

        "H_b": {  # High-Lat Box
            "c": {M.DIC: "2210 umol/kg", M.TA: "2310 umol/kg", M.PO4: "2.1 umol/kg"},
            "g": {"area": "0.349e14m**2", "volume": "87.5e14 m**3"},  # geometry
            "T": 2,  # temperature in C
            "P": 10,  # pressure in bar
            "S": 34.7,  # salinity in psu
        },

        "A_sb": {  # Atlantic low
            "c": {M.DIC: "2210 umol/kg", M.TA: "2310 umol/kg", M.PO4: "2.1 umol/kg"},
            "g": {"area": "0.907e14m**2", "volume": "0.907e16 m**3"},  # geometry
            "T": 20,  # temperature in C
            "P": 5,  # pressure in bar
            "S": 34.7,  # salinity in psu
        },
        "A_ib": {  # Atlantic intermediate
            "c": {M.DIC: "2210 umol/kg", M.TA: "2310 umol/kg", M.PO4: "2.1 umol/kg"},
            "g": {"area": "0.907e14m**2", "volume": "0.817e17 m**3"},
            "T": 10,
            "P": 80,
            "S": 34.7,
        },
        "A_db": {  #Atlantic deep
            "c": {M.DIC: "2210 umol/kg", M.TA: "2310 umol/kg", M.PO4: "2.1 umol/kg"},
            "g": {"area": "0.907e14m**2", "volume": "2.853e17 m**3"}, #z_int = 1000 m
            "T": 2,
            "P": 240,
            "S": 34.7,
        }, 
        
        "A_bb": { #Atlantic Burial box - defined as an ocean reservoir because CS3 cannot currently work with Sink objects
            "c":{M.DIC: "0 umol/kg", M.TA: "0 umol/kg", M.PO4: "0 umol/kg"},
            "g":{"area": "0.907e14m**2", "volume": "3.628e16 m**3"}, #Based on a sediment depth of approx 400m 
            "T": 2,
            "P": 240,
            "S": 34.7,
        },

        #PACIFIC:

        "P_sb": {  # Pacific low
            "c": {M.DIC: "2210 umol/kg", M.TA: "2310 umol/kg", M.PO4: "2.1 umol/kg"},
            "g": {"area": "1.605e14m**2", "volume": "1.605e16 m**3"},  # geometry
            "T": 20,  # temperature in C
            "P": 5,  # pressure in bar
            "S": 34.7,  # salinity in psu
        },
        "P_ib": {  # Pacific intermediate
            "c": {M.DIC: "2210 umol/kg", M.TA: "2310 umol/kg", M.PO4: "2.1 umol/kg"},
            "g": {"area": "1.605e14m**2", "volume": "1.445e17 m**3"},
            "T": 10,
            "P": 80,
            "S": 34.7,
        },
        "P_db": {  # Pacific deep
            "c": {M.DIC: "2210 umol/kg", M.TA: "2310 umol/kg", M.PO4: "2.1 umol/kg"}, 
            "g": {"area": "1.605e14m**2", "volume": "4.739e17 m**3"}, #z_int = 1000 m
            "T": 2,
            "P": 240,
            "S": 34.7,
        }, 
        
        "P_bb": { # Pacific Burial box - defined as an ocean reservoir because CS3 cannot currently work with Sink objects
            "c":{M.DIC: "0 umol/kg", M.TA: "0 umol/kg", M.PO4: "0 umol/kg"},
            "g":{"area": "1.605e14m**2", "volume": "6.42e16 m**3"}, #Based on a sediment depth of approx 400m 
            "T": 2,
            "P": 240,
            "S": 34.7,
        },

        #INDIAN: 

        "I_sb": {  # Indian low
            "c": {M.DIC: "2210 umol/kg", M.TA: "2310 umol/kg", M.PO4: "2.1 umol/kg"},
            "g": {"area": "0.628e14m**2", "volume": "0.628e16 m**3"},  # geometry
            "T": 20,  # temperature in C
            "P": 5,  # pressure in bar
            "S": 34.7,  # salinity in psu
        },
        "I_ib": {  # Indian intermediate
            "c": {M.DIC: "2210 umol/kg", M.TA: "2310 umol/kg", M.PO4: "2.1 umol/kg"},
            "g": {"area": "0.628e14m**2", "volume": "0.565e17 m**3"},
            "T": 10,
            "P": 80,
            "S": 34.7,
        },
        "I_db": {  # Indian deep
            "c": {M.DIC: "2210 umol/kg", M.TA: "2310 umol/kg", M.PO4: "2.1 umol/kg"}, 
            "g": {"area": "0.628e14m**2", "volume": "2.099e17 m**3"}, #z_int = 1000 m
            "T": 2,
            "P": 240,
            "S": 34.7,
        }, 
        
        "I_bb": { # Indian Burial box - defined as an ocean reservoir because CS5 cannot currently work with Sink objects
            "c":{M.DIC: "0 umol/kg", M.TA: "0 umol/kg", M.PO4: "0 umol/kg"},
            "g":{"area": "0.628e14m**2", "volume": "2.512e16 m**3"}, #Based on a sediment depth of approx 400m 
            "T": 2,
            "P": 240,
            "S": 34.7,
        },

         # sources and sinks
        "Fw": {"ty": "Source", "sp": [M.DIC, M.TA, M.PO4]},
        "Fb": {"ty": "Sink", "sp": [M.DIC, M.TA, M.PO4]},
    }

    species_list = initialize_reservoirs(M, box_parameters)

    connection_dict = {
        # source_to_sink@id

        "H_b_to_A_db@thermohaline": {"ty": "scale_with_concentration", "sc": "20 Sverdrup", "sp": species_list,},
        "A_db_to_A_ib@upwelling": {"ty": "scale_with_concentration", "sc": "4 Sverdrup", "sp": species_list,},
        "I_db_to_I_ib@upwelling": {"ty": "scale_with_concentration", "sc": "4 Sverdrup", "sp": species_list,},
        "A_db_to_I_db@thermohaline": {"ty": "scale_with_concentration", "sc": "16 Sverdrup","sp": species_list,},
        "I_db_to_P_db@thermohaline": {"ty": "scale_with_concentration", "sc": "12 Sverdrup", "sp": species_list,},
        "P_db_to_P_ib@thermohaline": {"ty": "scale_with_concentration", "sc": "12 Sverdrup", "sp": species_list,},
        "P_ib_to_I_ib@thermohaline": {"ty": "scale_with_concentration", "sc": "12 Sverdrup", "sp": species_list,},
        "I_ib_to_A_ib@thermohaline": {"ty": "scale_with_concentration", "sc": "16 Sverdrup", "sp": species_list,},
        "A_ib_to_H_b@thermohaline": {"ty": "scale_with_concentration", "sc": "20 Sverdrup", "sp": species_list,},
        "A_ib_to_A_sb@mix_up": {"ty": "scale_with_concentration", "sc": "21 Sverdrup", "sp": species_list,},
        "A_sb_to_A_ib@mix_down": {"ty": "scale_with_concentration", "sc": "21 Sverdrup", "sp": species_list,},
        "I_ib_to_I_sb@mix_up": {"ty": "scale_with_concentration", "sc": "17 Sverdrup", "sp": species_list,},
        "I_sb_to_I_ib@mix_down": {"ty": "scale_with_concentration",  "sc": "17 Sverdrup",  "sp": species_list, },
        "P_ib_to_P_sb@mix_up": {"ty": "scale_with_concentration", "sc": "25 Sverdrup", "sp": species_list,},
        "P_sb_to_P_ib@mix_down": {"ty": "scale_with_concentration", "sc": "25 Sverdrup", "sp": species_list,},
        "A_db_to_H_b@mix_up": { "ty": "scale_with_concentration", "sc": "4 Sverdrup","sp": species_list, },
        "H_b_to_A_db@mix_down": { "ty": "scale_with_concentration", "sc": "4 Sverdrup", "sp": species_list,},
        "I_db_to_H_b@mix_up": {"ty": "scale_with_concentration", "sc": "3 Sverdrup","sp": species_list, },
        "H_b_to_I_db@mix_down": {"ty": "scale_with_concentration","sc": "3 Sverdrup", "sp": species_list,},
        "P_db_to_H_b@mix_up": {"ty": "scale_with_concentration", "sc": "10 Sverdrup","sp": species_list,},
        "H_b_to_P_db@mix_down": {"ty": "scale_with_concentration","sc": "10 Sverdrup","sp": species_list,},

    }
    create_bulk_connections(connection_dict, M)

    #Phosphorus cycling and organic matter:

    """P-cycling: Export production is a function of the P-Export.  In this
    model we describe these processes by quantyfying how much P is exported from
    the photic zone as function of the P upwelling flux into surface box *
    uptake efficiency.

    The P fluxes into the surface box are the mixing fluxes from the
    intermediate waters for A/I/P and the thermohaline upwelling for the high
    latitude box.  We set this up as follows

        1. Export P as part of OM in the form of particulate organic matter
           (POM_P).  This is a function of the mixing flux.

        2. Add the POM_P which is remineralized to the dissolved P in each
           respective box.

    So the particulate P transport describes the removal of P from the surface
    box via particular P in OM, and the addition P into the underlying box.  In
    this box model, not all P stays in the intermediate box, some goes into the
    underlying deep box as well.  As such, multiply the fluxes with factor
    decscibing how much P ends up in the respective box.

    Get the list of P fluxes which mix upwards from the intermediate box.  Since
    we treat the high latidude box different, remove all H_sb values by only
    selecting the first 3 entries.  THis is not particularly robust.  Would be
    better to add exclude option to the flux_summmary method

    Note that the downward fluxes are particulate P, so he density terms do not
    apply
    """

    """ Settling fluxes from the High Latitude box to the various deep ocean boxes
    are scaled according to their mixing fluxes (4, 3, 10)
    """

    pfluxes = M.flux_summary(filter_by="PO4_mix_up", exclude="H_", return_list=True)
    
    # export productivity in the high latidude box
    PO4_ex = Q_(
        f"{1.8 * M.H_b.area / M.PC_ratio} mol/a"
    )  # Export productivity in the H box
    

    ct = {  # Surface box to ib, about 78% is remineralized in the ib
        ("A_sb_to_A_ib@POM_P", "I_sb_to_I_ib@POM_P", "P_sb_to_P_ib@POM_P"): {
            "ty": "scale_with_flux",
            "sc": M.PUE * M.ib_remin,
            "re": pfluxes,
            "sp": M.PO4,
        },  
        # surface box to deep box
        ("A_sb_to_A_db@POM_P", "I_sb_to_I_db@POM_P", "P_sb_to_P_db@POM_P"): {
            "ty": "scale_with_flux",
            "sc": M.PUE * M.db_remin,
            "re": pfluxes,
            "sp": M.PO4,
        },

        # high latitude box to deep ocean boxes POM_P
        ("H_b_to_A_db@POM_P", "H_b_to_I_db@POM_P", "H_b_to_P_db@POM_P"): {
            # here we use a fixed rate following Zeebe's Loscar model
            "ra": [
                PO4_ex * 0.3,
                PO4_ex * 0.3,
                PO4_ex * 0.4,
            ],
            "sp": M.PO4,
            "ty": "Regular",
        },
    
    }
    create_bulk_connections(ct, M)

    """Organic Matter settling
    and remineralization follows the P fluxes with a constant factor
    given by the Redfield Ratio.

    For each mol P we create 130 mol OM, which consumes 130 mol DIC
    from the surface box. Remineralization injects the equivalent
    amoundt of DIC into the intermediate and deep waters.

    Since OM formation and remineralization also involves nitrogen
    fixation, photosynthesis increases alkalinity in the surface
    boxes, and organic matter remineralization decreases alkalinity in
    the intermediate and deep boxes
    
    Here we first use a helper function to generate the list which
    contains all fluxes with the POP label, and the corresponding list of
    connections we want to create for the particulate OM flux (POM) and
    then we use this list to create the connection dictionary.
    """

    pomp_fluxes = M.flux_summary(filter_by="POM_P", return_list=True)

    ct = {}

    for f in pomp_fluxes:
        fname = f.full_name  # <-- STRING, e.g. "A_sb_to_A_db@POM_P

        if not fname.endswith("@POM_P"):
            continue

    # Construct new flux IDs 
        dic_flux_id = fname.split("@")[0] + "@POM_DIC"
        ta_flux_id  = fname.split("@")[0] + "@POM_TA"

    # DIC connection
        ct[dic_flux_id] = {
            "re": f,                     # reference = Flux object
            "sp": M.DIC,
            "ty": "scale_with_flux",
            "sc": M.PC_ratio,
            "al": M.OM_frac,
        }

    # TA connection
        ct[ta_flux_id] = {
            "re": f,
            "sp": M.TA,
            "ty": "scale_with_flux",
            "sc": M.PC_ratio * M.NC_ratio * -1,
        }
        
    create_bulk_connections(ct, M)

    #Particulate Inorganic Matter and Carbonate System: 

    """particulate inorganic carbon as function of export productivity (EP),
    where EP is equal to the PO4 export efficiency (M.PUE) * the P/C ratio *
    rain ratio.

    The LOSCAR model assumes that part of CaCO3 is dissolved in the upper 200
    meters (i.e., the alpha parameter), while the rest travels straight to the
    deep box before dissolution.

    As such, we cannot simply scale the POM fluxes.  Also, the carbonate
    calculations after Boudreau implement their own alpha, so this is not a 1:1
    mapping with LOSCAR.  Boudreau also calculates the calcite dissolution from
    oxic respiration, which will also occur in the intermediate (and surface?)
    box.  So we need to add these terms, and remove them from the flux of
    carbonate buried.

    It is likely that no high latitude CaCO3 reaches the deep ocean so we do not
    include the H_b CaCO3 export production

    LOSCARs rain ratio is Corg/CaCO3 = 6.1, so we need to devide by their rain
    rate

    Note that burial and dissolution in the deep box are fully handled by the
    cs2 function, so we need to bypass these fluxes

    CaCO3 rain burial in the intermediate and shallow water equals
    export production = M.PUE * M.PC_ratio * area_fraction * (1 - alpha) / rain rate %

    CaCO3 dissolution inthe intermediate box equals
    M.PUE * M.PC_ratio * area_fraction * alpha / rain rate
    
    Deeb box fluxes are M.PUE * M.PC_ratio * af_fraction / rain
    alpha will be calculated by cs2

    CaCO3 burial in the shallow water = production - water column dissolution
    M.PUE * M.PC_ratio *  (1 - alpha) / rain rate * area of shallow water
    """

    # get P-fluxes that drive export productivity
    pfluxes = M.flux_summary(filter_by="PO4_mix_up", exclude="H_b", return_list=True)
    af_sb = 0.05
    #af_sb = M.A_sb.sed_area / M.A_sb.area  # shelf/area vs total area
    sb = M.PUE * M.PC_ratio * (af_sb) * (1 - M.alpha) / M.rain
    # surface box burial and dissolution
    ct = {  # DIC
        (
            "A_sb_to_Fb@DIC_burial_sb_A",
            "I_sb_to_Fb@DIC_burial_sb_I",
            "P_sb_to_Fb@DIC_burial_sb_P",
        ): {  # surface box burial PIC DIC
            "ty": ["scale_with_flux", "scale_with_flux", "scale_with_flux"],
            "sc": [sb, sb, sb],
            "re": pfluxes,
            "sp": M.DIC,
        },
        # TA
        (
            "A_sb_to_Fb@TA_burial_sb_A",
            "I_sb_to_Fb@TA_burial_sb_I",
            "P_sb_to_Fb@TA_burial_sb_P",
        ): {  # surface box burial PIC TA
            "ty": ["scale_with_flux", "scale_with_flux", "scale_with_flux"],
            "sc": [2 * sb, 2 * sb, 2 * sb],
            "re": pfluxes,
            "sp": M.TA,
        },
    }
    create_bulk_connections(ct, M)

    """intermediate box burial: This removes DIC and TA from the surface box and
    deposits it on the slope.  As before this depends on the export production
    flux, the slope area and the water column dissolution.
    """
    af_ib = 0.05
    #af_ib = M.A_ib.area_dz / M.A_sb.area  # slope area/total area
    ib = M.PUE * M.PC_ratio * af_ib * (1 - M.alpha) / M.rain
    ct = {
        (  # DIC
            "A_sb_to_Fb@DIC_burial_ib_A",
            "I_sb_to_Fb@DIC_burial_ib_I",
            "P_sb_to_Fb@DIC_burial_ib_P",
        ): {  # surface box to deep box PIC_DIC
            "ty": ["scale_with_flux", "scale_with_flux", "scale_with_flux"],
            "sc": [ib, ib, ib],
            "re": pfluxes,
            "sp": M.DIC,
        },  #  TA
        (
            "A_sb_to_Fb@TA_burial_ib_A",
            "I_sb_to_Fb@TA_burial_ib_I",
            "P_sb_to_Fb@TA_burial_ib_P",
        ): {
            "ty": ["scale_with_flux", "scale_with_flux", "scale_with_flux"],
            "sc": [2 * ib, 2 * ib, 2 * ib],
            "re": pfluxes,
            "sp": M.TA,
        },
    }
    create_bulk_connections(ct, M)

    """Deep box export production: CaCO3 export into the deep box removes 1
    mol C and 2 mol TA for each mol CaCO3 removed from the surface water.

    However, only part of this will be added to the deep box, since some will
    end up as sediment.  the actual ratios will be calculated by the
    carbonate_system 2 routine.  As such, we need to specify the bypass option
    so that we do not add the entire PIC_DIC and PIC_TA fluxes to the deep box.
    Unlike before, water column dissolution (i.e. alpha) will be handled by the
    carbonate system module, so here we just specify the the export
    production/area
    """
    af_db = 0.9 #arbitrary 

    #af_db = M.A_db.area_dz / M.A_sb.area
    db = M.PUE * M.PC_ratio * af_db / M.rain

    print(f"areas = {[af_sb, af_ib, af_db]}")

    ct = {  # surface box to deep box  PIC_DIC
        ("A_sb_to_A_db@PIC_DIC_A", "I_sb_to_I_db@PIC_DIC_I", "P_sb_to_P_db@PIC_DIC_P"): {
            "ty": ["scale_with_flux", "scale_with_flux", "scale_with_flux"],
            "sc": [db, db, db],
            "re": pfluxes,
            "sp": M.DIC,
            "bp": ["sink", "sink", "sink"],  # bypass the deep box!
        },
        # surface box to deep box  PIC_TA
        ("A_sb_to_A_db@PIC_TA_A", "I_sb_to_I_db@PIC_TA_I", "P_sb_to_P_db@PIC_TA_P"): {
            "ty": ["scale_with_flux", "scale_with_flux", "scale_with_flux"],
            "sc": [2 * db, 2 * db, 2 * db],
            "re": pfluxes,
            "sp": M.TA,
            "bp": ["sink", "sink", "sink"],  # bypass the deep box!
        },
    }
    create_bulk_connections(ct, M)

    #Not needed because of CS3:
    """Deep box carbonate burial: This removes DIC and TA from the deep box.
    Here we only initialize the fluxes, so that they exist in the model
    namespace.  We thus set the connection type to "ignore" since the actual
    fluxes will be computed by the carbonate-system module.
    
    ct = {
        (  # DIC
            "A_db_to_Fb@DIC_burial_db_A",
            "I_db_to_Fb@DIC_burial_db_I",
            "P_db_to_Fb@DIC_burial_db_P",
        ): {
            "ty": ["ignore", "ignore", "ignore"],
            "ra": [0, 0, 0],
            "sp": M.DIC,
        },
        (  # TA
            "A_db_to_Fb@TA_burial_db_A",
            "I_db_to_Fb@TA_burial_db_I",
            "P_db_to_Fb@TA_burial_db_P",
        ): {
            "ty": ["ignore", "ignore", "ignore"],
            "ra": [0, 0, 0],
            "sp": M.TA,
        },
    }
    create_bulk_connections(ct, M)
    """
    

    """ Carbonate Chemistry calculations
    are done through two additional modules. CC1 calculates a variety
    of carbonate species like CO2aq and CO3-, whereas CS3 additionally
    calculates carbonate solubility, the depth where c-solubility
    changes from positive to negative, i.e., carbonate saturation
    horizon (top of the lysocline, zsat), the carbonate compensation
    depth (zcc) and the depth of the snow line (zsnow). It also
    calculates the carbonate burial flux as function of the incoming
    PIC, and CO3- concentration in the deep box (including sediment
    dissolution).  Parametrization of these processes after Boudreau
    et al. 2010
    """

    # All surface and intermediate boxes use cs 1
    add_carbonate_system_1([M.A_sb, M.I_sb, M.P_sb, M.A_ib, M.I_ib, M.P_ib, M.H_b])

    # The deep boxes will additionally calculate carbonate burial and
    # dissolution fluxes
    cef = M.flux_summary(filter_by="PIC_DIC", return_list=True)

    add_carbonate_system_3(
        this_box=[M.A_db, M.I_db, M.P_db],  # deep boxes where we add cs2
        source_box=[M.A_sb, M.I_sb, M.P_sb],  # corresponding surface boxes
        next_box = [M.A_bb, M.I_bb, M.P_bb],
        carbonate_export_fluxes=cef,
        zsat_min=-1000,  # zsat_max
        z0=-1000,
        alpha=alpha,
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

    Species2Species(  # High box to atmosphere
        source=M.CO2_At,  # Reservoir Species
        sink=M.H_b.DIC,  # Reservoir Species
        species=M.CO2,
        piston_velocity=pv,
        ctype="gasexchange",
        id="H_b",
    )

    Species2Species(  # Atlantic surface to atmosphere
        source=M.CO2_At,  # Reservoir Species
        sink=M.A_sb.DIC,  # Reservoir Species
        species=M.CO2,
        piston_velocity=pv,
        ctype="gasexchange",
        id="A_sb",
    )

    Species2Species(  # Indian surface to atmosphere
        source=M.CO2_At,  # Reservoir Species
        sink=M.I_sb.DIC,  # Reservoir Species
        species=M.CO2,
        piston_velocity=pv,
        ctype="gasexchange",
        id="I_sb",
    )

    Species2Species(  # Pacific surface to atmosphere
        source=M.CO2_At,  # Reservoir Species
        sink=M.P_sb.DIC,  # Reservoir Species
        species=M.CO2,
        piston_velocity=pv,
        ctype="gasexchange",
        id="P_sb",
    )

    #weathering fluxes:

    #volcanic flux:

    Species2Species(  # Low Latitude surface to atmosphere F7
        source=M.Fw.DIC,  # Reservoir Species
        sink=M.CO2_At,  # Reservoir Species
        species=M.CO2,
        ctype="Fixed",
        rate=M.Fw_v,
        id="volcanic_weathering",
    )

    # CaCO3:

     # unitless weathering strength
    Species2Species(
        ctype="weathering",
        source=M.Fw.DIC,  # source of flux
        sink=M.Fb.DIC,
        reservoir_ref=M.CO2_At,  # pCO2
        scale=1.0,  # optional, defaults to 1
        ex=0.4,  # exponent c
        pco2_0="280 ppm",  # reference pCO2
        rate=1.0,  # rate at pco2_0
        id="weathering_carbonate",
    )
    
    Species2Species(  # CaSiO3 weathering
        ctype="weathering",
        source=M.Fw.DIC,  # source of flux
        sink=M.Fb.DIC,
        reservoir_ref=M.CO2_At,  # pCO2
        scale= 1.00,  # optional, defaults to 1
        ex=0.2,  # exponent c
        pco2_0="280 ppm",  # reference pCO2
        rate= M.Fw_Si,  # rate at pco2_0
        id="weathering_silicate",
    )


     # DIC fluxes from carbonate weathering:

    Species2Species(  # Atlantic
        ctype="scale_with_flux",
        source=M.Fw.DIC,
        sink=M.A_sb.DIC,
        ref_flux="weathering_carbonate",
        scale= M.Fw_Ca * A_ap/(1 - H_ap),
        id="weathering_caco3_A",
    )

    Species2Species(  # Pacific
        ctype="scale_with_flux",
        source=M.Fw.DIC,
        sink=M.P_sb.DIC,
        ref_flux="weathering_carbonate",
        scale= M.Fw_Ca * P_ap/(1 - H_ap),
        id="weathering_caco3_P",
    )

    Species2Species(  # Indian
        ctype="scale_with_flux",
        source=M.Fw.DIC,
        sink=M.I_sb.DIC,
        ref_flux="weathering_carbonate",
        scale= M.Fw_Ca * I_ap/(1 - H_ap),
        id="weathering_caco3_I",
    )

    # CaCO3 weathering TA:

    Species2Species(  
        source=M.Fw.TA,  # source of flux
        sink=M.A_sb.TA,
        ctype="scale_with_flux",
        ref_flux="weathering_caco3_A",  
        scale= 2, 
        id="caco3_TA_A",
    )

    Species2Species(  
        source=M.Fw.TA,  
        sink=M.P_sb.TA,
        ctype="scale_with_flux",
        ref_flux="weathering_caco3_P",  
        scale= 2, 
        id="caco3_TA_P",
    )

    Species2Species(  
        source=M.Fw.TA,  
        sink=M.I_sb.TA,
        ctype="scale_with_flux",
        ref_flux="weathering_caco3_I",  
        scale= 2, 
        id="caco3_TA_I",
    )
    
    #CaSiO3: DIC
    
    Species2Species(  # Atlantic
        ctype="scale_with_flux",
        source=M.CO2_At,
        sink=M.A_sb.DIC,
        ref_flux="weathering_silicate",
        scale=  A_ap/(1 - H_ap),
        id="wsi_A",
    )
    
    Species2Species(  # Pacific
        ctype="scale_with_flux",
        source=M.CO2_At,
        sink=M.P_sb.DIC,
        ref_flux="weathering_silicate",
        scale= P_ap/(1 - H_ap),
        id="wsi_P",
    )

    Species2Species(  # Indian
        ctype="scale_with_flux",
        source=M.CO2_At,
        sink=M.I_sb.DIC,
        ref_flux="weathering_silicate",
        scale=  I_ap/(1 - H_ap),
        id="wsi_I",
    )

    #CaSiO3 TA:

    Species2Species(  # CaSiO3 weathering
        source=M.Fw.TA,  # source of flux
        sink=M.A_sb.TA,
        ctype="scale_with_flux",
        ref_flux="wsi_A",  
        scale= 2, 
        id="wsi_A_TA",
    )

    Species2Species(  # CaSiO3 weathering
        source=M.Fw.TA,  # source of flux
        sink=M.P_sb.TA,
        ctype="scale_with_flux",
        ref_flux="wsi_P",  
        scale= 2,  # optional, defaults to 1
        id="wsi_P_TA",
    )
    
    Species2Species(  # CaSiO3 weathering
        source=M.Fw.TA,  # source of flux
        sink=M.I_sb.TA,
        ctype="scale_with_flux",
        ref_flux="wsi_I",  
        scale= 2,  # optional, defaults to 1
        id="wsi_I_TA",
    
    )

    return M

run_time = "10000 kyr"
time_step = "1000 yr"  
rain_ratio = 5.1
alpha = 0.45

M = initialize_model(rain_ratio, alpha, run_time, time_step)

#M.debug_equations_file=True
M.run()

"""
M.plot([M.CO2_At])
print(f"CO2 {M.CO2_At.c[-2]:.6f}")

print(f"DIC Atlantic surface {M.A_sb.DIC.c[-2]:.6f}")
print(f"DIC Atlantic intermediate {M.A_ib.DIC.c[-2]:.6f}")
print(f"DIC Atlantic deep {M.A_db.DIC.c[-2]:.6f}")

print(f"DIC Pacific surface {M.P_sb.DIC.c[-2]:.6f}")
print(f"DIC Pacific intermediate {M.P_ib.DIC.c[-2]:.6f}")
print(f"DIC Pacific deep {M.P_db.DIC.c[-2]:.6f}")

print(f"DIC Indian surface {M.I_sb.DIC.c[-2]:.6f}")
print(f"DIC Indian intermediate {M.I_ib.DIC.c[-2]:.6f}")
print(f"DIC Indian deep {M.I_db.DIC.c[-2]:.6f}")

M.plot([M.H_b.DIC, M.A_sb.DIC, M.A_ib.DIC, M.A_db.DIC])
M.plot([M.H_b.TA, M.A_sb.TA, M.A_ib.TA, M.A_db.TA])
M.plot([M.P_sb.DIC, M.P_ib.DIC, M.P_db.DIC])
M.plot([M.I_sb.DIC, M.I_ib.DIC, M.I_db.DIC])
M.plot([M.A_bb.DIC, M.I_bb.DIC, M.P_bb.DIC,])

"""
