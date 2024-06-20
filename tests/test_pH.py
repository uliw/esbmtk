import pytest
from math import log10
from esbmtk import Model, Reservoir, get_hplus
import PyCO2SYS as pyco2


def test_manual_ph_calculation():
    """Test convergence of interative pH calculation"""
    M = Model(
        stop="1 yr",
        max_timestep="1 d",
        element=[
            "Carbon",
            "Boron",
            "Hydrogen",
            "Phosphor",
            "Oxygen",
            "misc_variables",
        ],
        mass_unit="mol",
        volume_unit="l",
        concentration_unit="mol/kg",
        opt_k_carbonic=13,
        opt_pH_scale=3,
        opt_buffers_mode=2,
    )

    Reservoir(
        name="S_b",
        geometry={"area": "0.5e14m**2", "volume": "1.76e16 m**3"},
        concentration={
            M.DIC: "1.9728038446966216 mmol/kg",
            M.TA: "2.3146405168630797 mmol/kg",
        },
        seawater_parameters={
            "T": 21.5,
            "P": 5,
            "S": 35,
        },
    )

    hplus0 = M.S_b.swc.hplus
    hplus = hplus0 * 10
    for i in range(10):
        hplus = get_hplus(
            M.S_b.DIC.c[0],
            M.S_b.TA.c[0],
            hplus,
            M.S_b.swc.boron,
            M.S_b.swc.K1,
            M.S_b.swc.K1K2,
            M.S_b.swc.KW,
            M.S_b.swc.KB,
        )

    assert abs(hplus0 - hplus) < 1e-12


def test_pyco2sys_ph_calculation():
    """Compare result of pH computation with the value
    provided by pyCO2sys
    """
    M = Model(
        stop="1 yr",
        max_timestep="1 d",
        element=[
            "Carbon",
            "Boron",
            "Hydrogen",
            "Phosphor",
            "Oxygen",
            "misc_variables",
        ],
        mass_unit="mol",
        volume_unit="l",
        concentration_unit="mol/kg",
        opt_k_carbonic=13,
        opt_pH_scale=3,
        opt_buffers_mode=2,
    )

    Reservoir(
        name="S_b",
        geometry={"area": "0.5e14m**2", "volume": "1.76e16 m**3"},
        concentration={
            M.DIC: "1.9728038446966216 mmol/kg",
            M.TA: "2.3146405168630797 mmol/kg",
        },
        seawater_parameters={
            "T": 21.5,
            "P": 5,
            "S": 35,
        },
    )

    params = dict(
        salinity=M.S_b.swc.salinity,
        temperature=M.S_b.swc.temperature,
        pressure=M.S_b.swc.pressure * 10,
        par1_type=1,
        par1=M.S_b.TA.c[0] * 1e6,
        par2_type=2,
        par2=M.S_b.DIC.c[0] * 1e6,
        opt_k_carbonic=M.opt_k_carbonic,
        opt_pH_scale=M.opt_pH_scale,
        opt_buffers_mode=M.opt_buffers_mode,
    )
    results = pyco2.sys(**params)
    pH = results["pH"]

    hplus0 = M.S_b.swc.hplus
    hplus = hplus0 * 10
    for i in range(10):
        hplus = get_hplus(
            M.S_b.DIC.c[0],
            M.S_b.TA.c[0],
            hplus,
            M.S_b.swc.boron,
            M.S_b.swc.K1,
            M.S_b.swc.K1K2,
            M.S_b.swc.KW,
            M.S_b.swc.KB,
        )

    assert abs(pH - -log10(hplus)) < 1e-4
