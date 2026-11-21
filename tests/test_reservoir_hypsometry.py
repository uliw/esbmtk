"""Test dedicated hypsometry per Reservoir / ReservoirGroup."""

import numpy as np
import pytest
from esbmtk import (
    Model,
    Reservoir,
    add_carbonate_system_1,
    add_carbonate_system_2,
    hypsometry,
)


def test_reservoir_dedicated_hypsometry():
    """Verify that a reservoir group can have a dedicated hypsometry instance."""
    M = Model(
        stop="10 a",
        max_timestep="1 a",
        element=["Carbon", "Boron", "Hydrogen", "misc_variables"],
    )

    # Global hypsometry created by default on M.hyp
    default_hyp = M.hyp

    # Create a dedicated hypsometry instance
    custom_hyp = hypsometry(
        name="custom_hyp",
        model=M,
        register=M,
        smooth_step=0,
    )

    # Surface reservoir
    R_surf = Reservoir(
        name="Surf_Box",
        geometry=[0, 100, 1.0],
        concentration={M.DIC: "2000 umol/kg", M.TA: "2300 umol/kg"},
        seawater_parameters={"temperature": 20, "salinity": 35, "pressure": 5},
    )

    # 1. Reservoir with default hypsometry
    R_default = Reservoir(
        name="Default_Box",
        geometry=[100, 1000, 1.0],
        concentration={M.DIC: "2000 umol/kg", M.TA: "2300 umol/kg"},
        seawater_parameters={"temperature": 4, "salinity": 35, "pressure": 50},
    )

    # 2. Reservoir with dedicated hypsometry
    R_custom = Reservoir(
        name="Custom_Box",
        geometry=[100, 1000, 1.0],
        concentration={M.DIC: "2000 umol/kg", M.TA: "2300 umol/kg"},
        seawater_parameters={"temperature": 4, "salinity": 35, "pressure": 50},
        hypsometry=custom_hyp,
    )

    assert R_default.hyp is None
    assert R_custom.hyp is custom_hyp

    # Verify sed_area uses dedicated hypsometry
    expected_default_sed_area = default_hyp.area_dz(100, 1000)
    expected_custom_sed_area = custom_hyp.area_dz(100, 1000)
    assert R_default.sed_area.magnitude == pytest.approx(expected_default_sed_area)
    assert R_custom.sed_area.magnitude == pytest.approx(expected_custom_sed_area)

    add_carbonate_system_1([R_surf])

    # Now test cs2
    from esbmtk import create_bulk_connections

    connection_dict = {
        "Surf_Box_to_Default_Box@PIC_DIC": {
            "sp": M.DIC,
            "ty": "Fixed",
            "ra": "1e12 mol/a",
            "bp": "sink",
        },
        "Surf_Box_to_Custom_Box@PIC_DIC": {
            "sp": M.DIC,
            "ty": "Fixed",
            "ra": "1e12 mol/a",
            "bp": "sink",
        },
    }
    create_bulk_connections(connection_dict, M)
    ef = M.flux_summary(filter_by="PIC_DIC", return_list=True)

    add_carbonate_system_2(
        this_box=[R_default, R_custom],
        source_box=[R_surf, R_surf],
        carbonate_export_fluxes=ef,
        z0=-100,
    )

    # Verify function_params of each cs2 instance
    # cs2 params tuple: (swc_p, cp, area_table, area_dz_table, Csat_table)
    # Default box should have model.area_table and model.area_dz_table
    p_default = R_default.cs2.function_params
    assert p_default[2] is M.area_table
    assert p_default[3] is M.area_dz_table

    # Custom box should have custom_hyp.area_table and custom_hyp.area_dz_table
    p_custom = R_custom.cs2.function_params
    assert p_custom[2] is custom_hyp.area_table
    assert p_custom[3] is custom_hyp.area_dz_table
    # Area between z0 and zmax in cp (cp[2]) should match custom_hyp
    assert p_custom[1][2] == pytest.approx(custom_hyp.area_dz(-100, -10099))
