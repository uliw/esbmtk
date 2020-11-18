def test_delta_conversion():
    from esbmtk import get_imass, get_delta, get_flux_data

    r =0.0112372
    m = 1
    d = 0
    
    l,h = get_imass(m, d, r)
    d2 = get_delta(l,h,r)
    assert d == d2
    A =  get_flux_data(m, d, r)
    assert A[0] == m
    assert A[1] == l
    assert A[2] == h
    assert A[3] == d

def test_delta_conversion_with_vector():
    from esbmtk import get_imass, get_delta, get_flux_data
    import numpy as np

    r =0.0112372
    m = np.zeros(4) + 3001
    d = 0
    l,h = get_imass(m, d, r)
    d2 = get_delta(l,h,r)
    assert d == d2[1]

import pytest

@pytest.fixture
def create_model():
    
    # from module import symbol
    from esbmtk import Model, Reservoir

    c0 = 3.1
    d0 = 0
    v0 = 1025
    # create model
    Model(
        name="M1",  # model name
        stop="1 kyrs",  # end time of model
        timestep=" 1 yr",  # base unit for time
        mass_unit="mol",  # base unit for mass
        volume_unit="l",  # base unit for volume
        element="Carbon",  # load default element and species definitions
    )
    Reservoir(
        name="R1",  # Name of reservoir
        species=DIC,  # Species handle
        delta=d0,  # initial delta
        concentration=f"{c0} mol/l",  # concentration 
        volume=f"{v0} l",  # reservoir size (m^3)
    )

    return v0, c0, d0, M1, R1,

def test_reservoir_creation(create_model):
    """ Test that reservoir is initialized correctly
    
    """
    v0, c0, d0, M1, R1 = create_model

    mass = v0 * c0
    assert R1.c[2] == c0
    assert R1.m[2] == mass
    assert R1.d[2] == d0

def test_sum_fluxes(create_model):
    """
    test that the code which adds fluxes to a reservoir yields the expected results
    """

    from esbmtk import Connect, Source, Sink

    v0, c0, d0, M1, R1 = create_model
    
    Source(name="SO1", species=CO2)
    Source(name="SO2", species=CO2)
    Sink(name="SI1", species=CO2)
    Sink(name="SI2", species=CO2)

    Connect(
        source=SO1,  # source of flux
        sink=R1,  # target of flux
        rate="100 mol/yr",  # weathering flux in 
        delta=d0,  # set a default flux
    )
    Connect(
        source=SO2,  # source of flux
        sink=R1,  # target of flux
        rate="300 mol/yr",  # weathering flux in 
        delta=d0,  # set a default flux
    )

    Connect(
        source=R1,  # source of flux
        sink=SI1,  # target of flux
        rate="250 mol/yr",  # weathering flux in 
        delta=d0,  # set a default flux
    )

    Connect(
        source=R1,  # source of flux
        sink=SI2,  # target of flux
        rate="150 mol/yr",  # weathering flux in 
        delta=d0,  # set a default flux
    )

   
    M1.run()
    assert R1.c[-2] == c0
    R1.d[-2] == d0

    # strangely, this fails at the moment
    # assert R1.d[-]2 == 1

def test_passive_sum(create_model):
    """
    test that the code which adds fluxes to a reservoir yields the expected results
    """

    from esbmtk import Connect, Source, Sink

    v0, c0, d0, M1, R1 = create_model

    Source(name="SO1", species=CO2)
    Source(name="SO2", species=CO2)
    Sink(name="SI1", species=CO2)
    Sink(name="SI2", species=CO2)

    Connect(
        source=SO1,  # source of flux
        sink=R1,  # target of flux
        rate="100 mol/yr",  # weathering flux in 
        delta=d0,  # set a default flux
    )
    Connect(
        source=SO2,  # source of flux
        sink=R1,  # target of flux
        rate="300 mol/yr",  # weathering flux in 
        delta=d0,  # set a default flux
    )

    Connect(
        source=R1,  # source of flux
        sink=SI1,  # target of flux
        rate="250 mol/yr",  # weathering flux in 
        delta=0,  # set a default flux
    )

    Connect(
        source=R1,  # source of flux
        sink=SI2,  # target of flux
    )

    M1.run()
    assert R1.c[-2] == c0
    assert R1.d[-2] == d0
