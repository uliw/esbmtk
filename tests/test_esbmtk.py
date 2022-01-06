def test_delta_conversion():
    from esbmtk import get_imass, get_delta, get_flux_data

    r = 0.0112372
    m = 1
    d = 0

    l, h = get_imass(m, d, r)
    d2 = get_delta(l, h, r)
    assert abs(d - d2) < 1e-10, "deltas should be similar"
    A = get_flux_data(m, d, r)

    assert A[0] == m, "mass should be equal"
    assert A[1] == l, "light mass should be equal"
    assert A[2] == h, "heavy mass should be equal"
    assert A[3] == d, "deltas should be equal"


def test_delta_conversion_with_vector():
    from esbmtk import get_imass, get_delta, get_flux_data
    import numpy as np

    r = 0.0112372
    m = np.zeros(4) + 3001
    d = 0
    l, h = get_imass(m, d, r)
    d2 = get_delta(l, h, r)
    assert abs(max(d2 - d)) < 1e-10


def test_isotope_mass_calculation():
    from esbmtk import get_imass

    r = 0.0112372

    d = 21
    m = 10

    li, hi = get_imass(m, d, r)

    assert m - li - hi == 0


def test_fractionation_calculation():
    from esbmtk import get_imass, get_delta, get_frac

    r = 0.044162589
    m = 10
    d = 21
    a = 1.003

    l, h = get_imass(m, d, r)
    l2, h2 = get_frac(m, l, a)

    a2 = (l * h2) / (h * l2)

    diff = abs(a - a2)
    assert diff < 1e-15


import pytest


@pytest.fixture
def create_model():

    # from module import symbol
    from esbmtk import Model, Reservoir

    c0 = 3.1
    d0 = 0
    v0 = 1025
    # create model
    M1 = Model(
        name="M1",  # model name
        stop="1 kyrs",  # end time of model
        timestep=" 1 yr",  # base unit for time
        mass_unit="mol",  # base unit for mass
        volume_unit="l",  # base unit for volume
        element="Carbon",  # load default element and species definitions
        m_type="both",
    )
    Reservoir(
        name="R1",  # Name of reservoir
        species=M1.DIC,  # Species handle
        delta=d0,  # initial delta
        concentration=f"{c0} mol/l",  # concentration
        volume=f"{v0} l",  # reservoir size (m^3)
    )

    return v0, c0, d0, M1

def test_reservoir_creation(create_model):
    """Test that reservoir is initialized correctly"""
    v0, c0, d0, M1 = create_model

    mass = v0 * c0
    assert M1.R1.c[2] == c0
    assert M1.R1.m[2] == mass
    assert abs(M1.R1.d[2] - d0) < 1e-10


def test_fractionation(create_model):
    """Test that isotope fractionation from a flux out of a reserevoir
    results in the corrrect fractionation

    """

    from esbmtk import Source, Sink, Connect, Reservoir
    
    v0, c0, d0, M1 = create_model
    
    Source(name = "SO1",species =M1.DIC)
    Reservoir(
        name="R1",  # Name of reservoir
        species=M1.DIC,  # Species handle
        delta=d0,  # initial delta
        concentration=f"{c0} mol/l",  # concentration
        volume=f"{v0} l",  # reservoir size (m^3)
    )
    
    Sink(name = "SI1",species =M1.DIC)

    Connect(
        source=M1.SO1,  # source of flux
        sink=M1.R1,  # target of flux
        rate="100 mol/yr",  # weathering flux in
        delta=d0,  # set a default flux
    )

    Connect(
        source=M1.R1,  # source of flux
        sink=M1.SI1,  # target of flux
        ctype="Regular",
        rate="100 mol/yr",  # weathering flux in
        alpha=-28,  # set a default flux
    )

    M1.run()

    assert round(M1.R1.d[-2],6) == 28

def test_fractionation_numba(create_model):
    """Test that isotope fractionation from a flux out of a reserevoir
    results in the corrrect fractionation, when using the numba solver
    
    """
    
    from esbmtk import Source, Sink, Connect, Reservoir
    
    v0, c0, d0, M1 = create_model
    
    Source(name = "SO1",species =M1.DIC)
    Reservoir(
        name="R1",  # Name of reservoir
        species=M1.DIC,  # Species handle
        delta=d0,  # initial delta
        concentration=f"{c0} mol/l",  # concentration
        volume=f"{v0} l",  # reservoir size (m^3)
    )
    
    Sink(name = "SI1",species =M1.DIC)

    Connect(
        source=M1.SO1,  # source of flux
        sink=M1.R1,  # target of flux
        rate="100 mol/yr",  # weathering flux in
        delta=d0,  # set a default flux
    )

    Connect(
        source=M1.R1,  # source of flux
        sink=M1.SI1,  # target of flux
        ctype="Regular",
        rate="100 mol/yr",  # weathering flux in
        alpha=-28,  # set a default flux
    )

    M1.run(solver="numba")

    assert round(M1.R1.d[-2],6) == 28


def test_external_data(create_model):
    """test the creation of an external data object"""

    from esbmtk import ExternalData

    v0, c0, d0, M1 = create_model

    ExternalData(
        name="ED1",
        filename="measured_c_isotopes.csv",
        legend="ED1",
        reservoir=M1.R1,
        offset="0.1 kyrs",
    )
    assert M1.ED1.x[0] == 100
    assert "Unnamed" in M1.ED1.df.columns[1]
    assert round(M1.ED1.z[0], 10) == 2.09512
    assert round(M1.ED1.z[-1], 10) == 0.968293


def test_sum_fluxes(create_model):
    """
    test that the code which adds fluxes to a reservoir yields the expected results
    """

    from esbmtk import Connect, Source, Sink

    v0, c0, d0, M1 = create_model

    Source(name="SO1", species=M1.CO2)
    Source(name="SO2", species=M1.CO2)
    Sink(name="SI1", species=M1.CO2)
    Sink(name="SI2", species=M1.CO2)

    Connect(
        source=M1.SO1,  # source of flux
        sink=M1.R1,  # target of flux
        rate="100 mol/yr",  # weathering flux in
        delta=d0,  # set a default flux
    )
    Connect(
        source=M1.SO2,  # source of flux
        sink=M1.R1,  # target of flux
        rate="300 mol/yr",  # weathering flux in
        delta=d0,  # set a default flux
    )

    Connect(
        source=M1.R1,  # source of flux
        sink=M1.SI1,  # target of flux
        rate="250 mol/yr",  # weathering flux in
        delta=d0,  # set a default flux
    )

    Connect(
        source=M1.R1,  # source of flux
        sink=M1.SI2,  # target of flux
        rate="150 mol/yr",  # weathering flux in
        delta=d0,  # set a default flux
    )

    M1.run()
    assert M1.R1.c[-2] == c0
    M1.R1.d[-2] == d0

    # strangely, this fails at the moment
    # assert R1.d[-]2 == 1


def test_passive_sum(create_model):
    """
    test that the code which adds fluxes to a reservoir yields the expected results
    """

    from esbmtk import Connect, Source, Sink

    v0, c0, d0, M1 = create_model

    Source(name="SO1", species=M1.CO2)
    Source(name="SO2", species=M1.CO2)
    Sink(name="SI1", species=M1.CO2)
    Sink(name="SI2", species=M1.CO2)

    Connect(
        source=M1.SO1,  # source of flux
        sink=M1.R1,  # target of flux
        rate="100 mol/yr",  # weathering flux in
        delta=d0,  # set a default flux
    )
    Connect(
        source=M1.SO2,  # source of flux
        sink=M1.R1,  # target of flux
        rate="300 mol/yr",  # weathering flux in
        delta=d0,  # set a default flux
    )

    Connect(
        source=M1.R1,  # source of flux
        sink=M1.SI1,  # target of flux
        rate="250 mol/yr",  # weathering flux in
        delta=0,  # set a default flux
    )

    Connect(
        source=M1.R1,  # source of flux
        sink=M1.SI2,  # target of flux
    )

    M1.run()
    assert M1.R1.c[-2] == c0
    assert abs(M1.R1.d[-2] - d0) < 1e-10
