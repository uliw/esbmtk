import pytest


@pytest.fixture(params=[True, False])
def create_model(request):

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
        save_flux_data=request.param,
    )
    Reservoir(
        name="R1",  # Name of reservoir
        species=M1.DIC,  # Species handle
        delta=d0,  # initial delta
        concentration=f"{c0} mol/l",  # concentration
        volume=f"{v0} l",  # reservoir size (m^3)
    )

    return v0, c0, d0, M1


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


def test_reservoir_creation(create_model):
    """Test that reservoir is initialized correctly"""
    v0, c0, d0, M1 = create_model

    mass = v0 * c0
    assert M1.R1.c[2] == c0
    assert M1.R1.m[2] == mass
    M1.get_delta_values()
    assert abs(M1.R1.d[2] - d0) < 1e-10


@pytest.mark.parametrize(
    "idx", [[0, 200], [100, 200], [200, 400], [200, 1200], [1100, 1000]]
)
@pytest.mark.parametrize("stype", ["square", "pyramid", "bell"])
def test_signal_indexing(idx, stype):
    """
    test that signal indexing works
    """

    from esbmtk import Source, Sink, Reservoir, Connect, Signal, Model
    import numpy as np

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
        offset="100 yrs",
    )

    Source(name="SO1", species=M1.DIC)

    Reservoir(
        name="R1",  # Name of reservoir
        species=M1.DIC,  # Species handle
        delta=d0,  # initial delta
        concentration=f"{c0} mol/l",  # concentration
        volume=f"{v0} l",  # reservoir size (m^3)
    )

    Sink(name="SI1", species=M1.DIC)
    Sink(name="SI2", species=M1.DIC)

    Connect(
        source=M1.SO1,  # source of flux
        sink=M1.R1,  # target of flux
        rate="100 mol/yr",  # weathering flux in
        delta=d0,  # set a default flux
    )

    Signal(
        name="foo",
        species=M1.DIC,
        start=f"{idx[0]} yrs",
        duration=f"{idx[1]} yrs",
        shape=stype,
        magnitude="50 mol/year",
        delta=-28,
    )

    Signal(
        name="foo2",
        species=M1.DIC,
        start=f"{idx[0]} yrs",
        duration=f"{idx[1]} yrs",
        shape=stype,
        mass="5000 mol",
        delta=-28,
    )

    Connect(
        source=M1.R1,  # source of flux
        sink=M1.SI1,  # target of flux
        ctype="Regular",
        rate="50 mol/yr",  # weathering flux in
        delta=0,
        signal=M1.foo,
        # alpha=-28,  # set a default flux
    )

    M1.run()


@pytest.mark.parametrize("solver", ["numba", "python"])
def test_fractionation(create_model, solver):
    """Test that isotope fractionation from a flux out of a reserevoir
    results in the corrrect fractionation

    """

    from esbmtk import Source, Sink, Connect, Reservoir

    v0, c0, d0, M1 = create_model

    Source(name="SO1", species=M1.DIC)
    Reservoir(
        name="R1",  # Name of reservoir
        species=M1.DIC,  # Species handle
        delta=d0,  # initial delta
        concentration=f"{c0} mol/l",  # concentration
        volume=f"{v0} l",  # reservoir size (m^3)
    )

    Sink(name="SI1", species=M1.DIC)

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
    M1.run(solver=solver)
    M1.get_delta_values()
    assert round(M1.R1.d[-2], 4) == 28.8066


@pytest.mark.parametrize("solver", ["numba", "python"])
def test_scale_flux(create_model, solver):
    """Test that isotope fractionation from a flux out of a reserevoir
    results in the corrrect fractionation, when using the numba solver

    """

    from esbmtk import Source, Sink, Connect, Reservoir

    v0, c0, d0, M1 = create_model

    Source(name="SO1", species=M1.DIC)
    Reservoir(
        name="R1",  # Name of reservoir
        species=M1.DIC,  # Species handle
        delta=d0,  # initial delta
        concentration=f"{c0} mol/l",  # concentration
        volume=f"{v0} l",  # reservoir size (m^3)
    )

    Sink(name="SI1", species=M1.DIC)
    Sink(name="SI2", species=M1.DIC)

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
        rate="50 mol/yr",  # weathering flux in
        alpha=-28,  # set a default flux
    )

    Connect(
        source=M1.R1,  # source of flux
        sink=M1.SI2,  # target of flux
        ctype="scale_with_flux",
        ref_flux=M1.C_R1_2_SI1.R1_2_SI1_F,
        scale=1,  # weathering flux in
    )
    M1.run(solver=solver)
    M1.get_delta_values()
    assert round(M1.R1.d[-2], 2) == 14.2


@pytest.mark.parametrize("solver", ["numba", "python"])
def test_scale_with_concentration_empty(create_model, solver):

    from esbmtk import Source, Sink, Connect, Reservoir
    import numpy as np

    v0, c0, d0, M1 = create_model
    Source(name="SO1", species=M1.DIC)
    Sink(name="SI1", species=M1.DIC)
    Reservoir(
        name="R2",  # Name of reservoir
        species=M1.DIC,  # Species handle
        delta=d0,  # initial delta
        concentration=f"0 mol/l",  # concentration
        volume=f"{v0} l",  # reservoir size (m^3)
    )
    Connect(
        source=M1.R2,  # source of flux
        sink=M1.R1,  # target of flux
        ctype="scale_with_concentration",
        alpha=-70,
    )
    M1.run(solver="solver")
    M1.get_delta_values()

    assert M1.R2.m[500] == 0
    assert M1.R2.l[500] == 0
    assert M1.R2.c[500] == 0
    assert M1.R1.c[500] == 3.1
    assert round(M1.R1.d[500], 10) == 0
    assert M1.C_R2_2_R1.R2_2_R1_F.fa[0] == 0
    assert M1.C_R2_2_R1.R2_2_R1_F.fa[1] == 0


@pytest.mark.parametrize("solver", ["numba", "python"])
def test_scale_with_concentration(create_model, solver):
    """Test that isotope fractionation from a flux out of a reserevoir
    results in the corrrect fractionation, when using the numba solver

    """

    from esbmtk import Source, Sink, Connect, Reservoir

    v0, c0, d0, M1 = create_model

    Source(name="SO1", species=M1.DIC)
    Reservoir(
        name="R1",  # Name of reservoir
        species=M1.DIC,  # Species handle
        delta=d0,  # initial delta
        concentration=f"{c0} mol/l",  # concentration
        volume=f"{v0} l",  # reservoir size (m^3)
    )

    Sink(name="SI1", species=M1.DIC)
    Sink(name="SI2", species=M1.DIC)

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
        rate="50 mol/yr",  # weathering flux in
        alpha=-28,  # set a default flux
    )

    Connect(
        source=M1.R1,  # source of flux
        sink=M1.SI2,  # target of flux
        ctype="scale_with_concentration",
        scale=1,
    )
    M1.run(solver=solver)
    M1.get_delta_values()

    assert round(M1.R1.d[-2]) == 14
    assert round(M1.R1.c[-2]) == 32


@pytest.mark.parametrize("solver", ["numba", "python"])
def test_scale_with_mass(create_model, solver):
    """Test that isotope fractionation from a flux out of a reserevoir
    results in the corrrect fractionation, when using the numba solver

    """

    from esbmtk import Source, Sink, Connect, Reservoir

    v0, c0, d0, M1 = create_model

    Source(name="SO1", species=M1.DIC)
    Reservoir(
        name="R1",  # Name of reservoir
        species=M1.DIC,  # Species handle
        delta=d0,  # initial delta
        concentration=f"{c0} mol/l",  # concentration
        volume=f"{v0} l",  # reservoir size (m^3)
    )

    Sink(name="SI1", species=M1.DIC)
    Sink(name="SI2", species=M1.DIC)

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
        rate="50 mol/yr",  # weathering flux in
        alpha=-28,  # set a default flux
    )

    Connect(
        source=M1.R1,  # source of flux
        sink=M1.SI2,  # target of flux
        ctype="scale_with_mass",
        scale=1 / 1025,
    )
    M1.run(solver=solver)
    M1.get_delta_values()

    assert round(M1.R1.d[-2]) == 14
    assert round(M1.R1.c[-2]) == 32


# should add tests for magnitude vs mass
@pytest.mark.parametrize("solver", ["numba", "python"])
def test_square_signal(create_model, solver):
    from esbmtk import Source, Sink, Reservoir, Connect, Signal
    import numpy as np

    v0, c0, d0, M1 = create_model

    Source(name="SO1", species=M1.DIC)
    Reservoir(
        name="R1",  # Name of reservoir
        species=M1.DIC,  # Species handle
        delta=d0,  # initial delta
        concentration=f"{c0} mol/l",  # concentration
        volume=f"{v0} l",  # reservoir size (m^3)
    )

    Sink(name="SI1", species=M1.DIC)
    Sink(name="SI2", species=M1.DIC)

    Connect(
        source=M1.SO1,  # source of flux
        sink=M1.R1,  # target of flux
        rate="100 mol/yr",  # weathering flux in
        delta=d0,  # set a default flux
    )

    Signal(
        name="foo",
        species=M1.DIC,
        start="100 yrs",
        duration="800 yrs",
        shape="square",
        magnitude="50 mol/year",
        delta=-28,
        register=M1,
    )

    Connect(
        source=M1.R1,  # source of flux
        sink=M1.SI1,  # target of flux
        ctype="Regular",
        rate="50 mol/yr",  # weathering flux in
        delta=0,
        signal=M1.foo,
        # alpha=-28,  # set a default flux
    )

    M1.run(solver=solver)
    M1.get_delta_values()

    assert round(M1.R1.d[90], 0) == 0
    assert round(max(M1.R1.d)) == 277
    assert round(M1.R1.c[200]) == 8
    assert round(max(M1.R1.c)) == 13
    # M1.plot([M1.R1, M1.foo, M1.C_R1_2_SI1.R1_2_SI1_F])


# should add tests for magnitude vs mass
@pytest.mark.parametrize("solver", ["numba", "python"])
def test_pyramid_signal(create_model, solver):
    v0, c0, d0, M1 = create_model

    from esbmtk import Source, Sink, Reservoir, Connect, Signal
    import numpy as np

    Source(name="SO1", species=M1.DIC)

    Reservoir(
        name="R1",  # Name of reservoir
        species=M1.DIC,  # Species handle
        delta=d0,  # initial delta
        concentration=f"{c0} mol/l",  # concentration
        volume=f"{v0} l",  # reservoir size (m^3)
    )

    Sink(name="SI1", species=M1.DIC)
    Sink(name="SI2", species=M1.DIC)

    Connect(
        source=M1.SO1,  # source of flux
        sink=M1.R1,  # target of flux
        rate="100 mol/yr",  # weathering flux in
        delta=d0,  # set a default flux
    )

    Signal(
        name="foo",
        species=M1.DIC,
        start="100 yrs",
        duration="800 yrs",
        shape="pyramid",
        magnitude="50 mol/year",
        delta=-28,
        register=M1,
    )

    Connect(
        source=M1.R1,  # source of flux
        sink=M1.SI1,  # target of flux
        ctype="Regular",
        rate="50 mol/yr",  # weathering flux in
        delta=0,
        signal=M1.foo,
        # alpha=-28,  # set a default flux
    )
    M1.run(solver=solver)
    M1.get_delta_values()

    assert round(max(M1.R1.d)) == 41
    assert round(max(M1.R1.c)) == 32
    assert np.argmax(M1.R1.d) == 702
    # M1.plot([M1.R1, M1.foo, M1.C_R1_2_SI1.R1_2_SI1_F])


@pytest.mark.parametrize("solver", ["numba", "python"])
def test_pyramid_signal_multiplication(create_model, solver):
    from esbmtk import Source, Sink, Reservoir, Connect, Signal
    import numpy as np

    v0, c0, d0, M1 = create_model

    Source(name="SO1", species=M1.DIC)
    Sink(name="SI1", species=M1.DIC)
    Sink(name="SI2", species=M1.DIC)

    Connect(
        source=M1.SO1,  # source of flux
        sink=M1.R1,  # target of flux
        rate="100 mol/yr",  # weathering flux in
        delta=d0,  # set a default flux
    )

    Signal(
        name="foo",
        species=M1.DIC,
        start="100 yrs",
        duration="800 yrs",
        shape="pyramid",
        magnitude="1 mol/year",
        stype="multiplication",
        register=M1,
    )

    Connect(
        source=M1.R1,  # source of flux
        sink=M1.SI1,  # target of flux
        ctype="Regular",
        rate="50 mol/yr",  # weathering flux in
        delta=10,
        signal=M1.foo,
        # alpha=-28,  # set a default flux
    )

    M1.run(solver=solver)
    M1.get_delta_values()

    assert round(M1.R1.c[-2]) == 81
    assert np.argmin(M1.R1.d) == 708
    assert round(M1.R1.d[-2], 2) == -2.41
    # M1.plot([M1.R1, M1.foo, M1.C_R1_2_SI1.R1_2_SI1_F])


@pytest.mark.parametrize("solver", ["numba", "python"])
def test_pyramid_signal_multiply(create_model, solver):
    from esbmtk import Source, Sink, Reservoir, Connect, Signal
    import numpy as np

    v0, c0, d0, M1 = create_model

    Source(name="SO1", species=M1.DIC)
    Sink(name="SI1", species=M1.DIC)
    Sink(name="SI2", species=M1.DIC)

    Connect(
        source=M1.SO1,  # source of flux
        sink=M1.R1,  # target of flux
        rate="100 mol/yr",  # weathering flux in
        delta=d0,  # set a default flux
    )

    Signal(
        name="foo",
        species=M1.DIC,
        start="100 yrs",
        duration="800 yrs",
        shape="pyramid",
        magnitude="1 mol/year",
        stype="multiplication",
        register=M1,
    )

    Connect(
        source=M1.R1,  # source of flux
        sink=M1.SI1,  # target of flux
        ctype="Regular",
        rate="50 mol/yr",  # weathering flux in
        alpha=-70,
        signal=M1.foo,
        # alpha=-28,  # set a default flux
    )

    M1.run(solver="solver")
    M1.get_delta_values()
    diff = M1.C_R1_2_SI1.R1_2_SI1_F.d[500] - M1.R1.d[500]
    assert round(M1.R1.c[-2]) == 81
    assert np.argmax(M1.R1.d) == 692
    assert round(diff) == -71
    # M1.plot([M1.R1, M1.foo, M1.C_R1_2_SI1.R1_2_SI1_F])


@pytest.mark.parametrize("solver", ["numba", "python"])
def test_scale_with_flux_and_signal_multiplication(create_model, solver):

    from esbmtk import Source, Sink, Reservoir, Connect, Signal
    import numpy as np

    v0, c0, d0, M1 = create_model

    Source(name="SO1", species=M1.DIC)
    Sink(name="SI1", species=M1.DIC)

    Connect(
        source=M1.SO1,  # source of flux
        sink=M1.R1,  # target of flux
        rate="100 mol/yr",  # weathering flux in
        delta=d0,  # set a default flux
    )
    Signal(
        name="foo",
        species=M1.DIC,
        start="100 yrs",
        duration="800 yrs",
        shape="pyramid",
        magnitude="1 mol/year",
        stype="multiplication",
    )

    Connect(
        source=M1.R1,  # source of flux
        sink=M1.SI1,  # target of flux
        ctype="scale_with_flux",
        ref_flux=M1.C_SO1_2_R1.SO1_2_R1_F,
        alpha=-72,
        signal=M1.foo,
        # alpha=-28,  # set a default flux
    )
    M1.run(solver=solver)
    M1.get_delta_values()

    diff = M1.C_R1_2_SI1.R1_2_SI1_F.d[500] - M1.R1.d[500]
    assert round(diff) == -74
    assert round(M1.R1.c[-2]) == 61
    assert np.argmax(M1.R1.d) == 672
    assert round(max(M1.R1.d)) == 43

    # M1.plot([M1.R1, M1.foo, M1.C_R1_2_SI1.R1_2_SI1_F])


@pytest.mark.parametrize("solver", ["numba", "python"])
def test_scale_with_flux_and_signal_addition(create_model, solver):

    from esbmtk import Source, Sink, Reservoir, Connect, Signal
    import numpy as np

    v0, c0, d0, M1 = create_model

    Source(name="SO1", species=M1.DIC)
    Sink(name="SI1", species=M1.DIC)

    Connect(
        source=M1.SO1,  # source of flux
        sink=M1.R1,  # target of flux
        rate="100 mol/yr",  # weathering flux in
        delta=d0,  # set a default flux
    )
    Signal(
        name="foo",
        species=M1.DIC,
        start="100 yrs",
        duration="800 yrs",
        shape="pyramid",
        magnitude="100 mol/year",
        stype="addition",
    )

    Connect(
        source=M1.R1,  # source of flux
        sink=M1.SI1,  # target of flux
        ctype="scale_with_flux",
        ref_flux=M1.C_SO1_2_R1.SO1_2_R1_F,
        alpha=-72,
        signal=M1.foo,
        scale=0.1
        # alpha=-28,  # set a default flux
    )
    # M1.run(solver='python')
    M1.run(solver="solver")
    M1.get_delta_values()

    diff = M1.C_R1_2_SI1.R1_2_SI1_F.d[500] - M1.R1.d[500]
    assert round(diff) == -75
    assert round(M1.R1.c[-2]) == 52
    assert np.argmax(M1.R1.d) == 658
    assert round(max(M1.R1.d)) == 53

    # M1.plot([M1.R1, M1.foo,
    #          M1.C_SO1_2_R1.SO1_2_R1_F,
    #          M1.C_R1_2_SI1.R1_2_SI1_F])


def test_seawaterconstants(create_model):
    from esbmtk import Source, Sink, Reservoir, Connect, Signal
    from esbmtk import ExternalCode, ReservoirGroup, add_carbonate_system_1
    from esbmtk import GasReservoir, AirSeaExchange
    import numpy as np
    import sys

    v0, c0, d0, M1 = create_model
    Source(name="SO1", species=M1.DIC)
    Sink(name="SI1", species=M1.DIC)

    ReservoirGroup(name = "S",        # Name of reservoir group
                        #volume = "1E5 l",       # see below
                        geometry = [0, 6000, 1],
                        delta   = {M1.DIC:0, M1.TA:0},  # dict of delta values
                        concentration = {M1.DIC:"2.1 mmol/l", M1.TA: "2.36 mmol/l"},
                        isotopes = {M1.DIC: True},
                        seawater_parameters = {"temperature": 5, "pressure": 0, "salinity" : 0},
                        #@carbonate_system= True,
                        register=M1,
                   )
    assert round(M1.S.swc.density, 6) ==  999.966751

    ReservoirGroup(name = "S",        # Name of reservoir group
                        #volume = "1E5 l",       # see below
                        geometry = [0, 6000, 1],
                        delta   = {M1.DIC:0, M1.TA:0},  # dict of delta values
                        concentration = {M1.DIC:"2.1 mmol/l", M1.TA: "2.36 mmol/l"},
                        isotopes = {M1.DIC: True},
                        seawater_parameters = {"temperature": 25, "pressure": 1000, "salinity" : 35},
                        #@carbonate_system= True,
                        register=M1,
                   )

    assert round(M1.S.swc.density, 6) ==  1062.538172 # kg/m^3
    assert round(M1.S.swc.so4,6) == 0.03002 # mol/liter

@pytest.mark.parametrize("solver", ["numba", "python"])
def test_carbonate_system1_constants(create_model, solver):
    """Test that isotope fractionation from a flux out of a reserevoir
    results in the corrrect fractionation, when using the numba solver

    """

    from esbmtk import Source, Sink, Connect, Reservoir

    v0, c0, d0, M1 = create_model

    from esbmtk import ExternalCode, ReservoirGroup, add_carbonate_system_1
    import numpy as np

    ReservoirGroup(
        name="S",  # Name of reservoir group
        volume="1E5 l",  # see below
        delta={M1.DIC: 0, M1.TA: 0},  # dict of delta values
        concentration={M1.DIC: "2.1 mmol/l", M1.TA: "2.4 mmol/l"},
        isotopes={M1.DIC: True},
        seawater_parameters={"temperature": 25, "pressure": 1, "salinity": 35},
        # @carbonate_system= True,
        register=M1,
    )

    add_carbonate_system_1([M1.S])

    Source(name="SO1", species=M1.DIC)

    Sink(name="SI1", species=M1.DIC)

    # DIC influx
    Connect(
        source=M1.SO1,  # source of flux
        sink=M1.S.DIC,  # target of flux
        rate="100 mol/yr",  # weathering flux in
    )
    # TA influx
    Connect(
        source=M1.SO1,  # source of flux
        sink=M1.S.TA,  # target of flux
        rate="100 mol/yr",  # weathering flux in
    )

    # DIC outflux
    Connect(
        source=M1.S.DIC,  # source of flux
        sink=M1.SI1,  # target of flux
        ctype="Regular",
        rate="100 mol/yr",  # weathering flux in
    )
    # TA out
    Connect(
        source=M1.S.TA,  # source of flux
        sink=M1.SI1,  # target of flux
        ctype="Regular",
        rate="100 mol/yr",  # weathering flux in
        # alpha=-28,  # set a default flux
    )

    M1.run(solver=solver)
    i = 200
    assert round(M1.S.DIC.c[i] * 1000, 2) == 2.1  # DIC
    assert round(M1.S.TA.c[i] * 1000, 2) == 2.4  # TA
    assert round(-np.log10(M1.S.cs.H[i]), 2) == 7.99  # pH
    assert round(M1.S.cs.HCO3[i] * 1000, 2) == 1.87
    assert round(M1.S.cs.CO3[i] * 1000, 2) == 0.22
    assert round(M1.S.cs.CO2aq[i] * 1000, 4) == 0.0134

@pytest.mark.parametrize("solver", ["numba", "python"])
def test_carbonate_system2_(create_model, solver):
    """Test that isotope fractionation from a flux out of a reserevoir
    results in the corrrect fractionation, when using the numba solver

    """

    from esbmtk import Source, Sink, Connect, Reservoir

    v0, c0, d0, M1 = create_model

    from esbmtk import ExternalCode, ReservoirGroup, add_carbonate_system_2
    import numpy as np

    ReservoirGroup(name = "S",        # Name of reservoir group
                    #volume = "1E5 l",       # see below
                    geometry = [0, 6000, 1],
                    delta   = {M1.DIC:0, M1.TA:0},  # dict of delta values
                    concentration = {M1.DIC:"2.3 mmol/l", M1.TA: "2.4 mmol/l"},
                    isotopes = {M1.DIC: True},
                    seawater_parameters = {"temperature": 2, "pressure": 240, "salinity" : 35},
                    #@carbonate_system= True,
                    register=M1,
               )



    Source(name="SO1", species=M1.DIC)

    Sink(name="SI1", species=M1.DIC)

    # DIC influx
    Connect(
        source=M1.SO1,  # source of flux
        sink=M1.S.DIC,  # target of flux
        rate="100 mol/yr",  # weathering flux in
    )
    # TA influx
    Connect(
        source=M1.SO1,  # source of flux
        sink=M1.S.TA,  # target of flux
        rate="100 mol/yr",  # weathering flux in
    )

    # DIC outflux
    Connect(
        source=M1.S.DIC,  # source of flux
        sink=M1.SI1,  # target of flux
        ctype="Regular",
        rate="100 mol/yr",  # weathering flux in
    )
    # TA x
    Connect(
        source=M1.S.TA,  # source of flux
        sink=M1.SI1,  # target of flux
        ctype="Regular",
        rate="100 mol/yr",  # weathering flux in
        #alpha=-28,  # set a default flux
    )
    exp = M1.C_SO1_2_DIC.SO1_2_DIC_F
    exp.fa = np.array([60E12, 60E12])
    add_carbonate_system_2(rgs=[M1.S],
                           carbonate_export_fluxes=[exp],
                           alpha=0.6,
                           zsat_min = -200,
                           z0 = -200)
    M1.run(solver=solver)
    i = 997
    assert round(M1.S.DIC.c[i] * 1000, 2) == 2.36  # DIC
    assert round(M1.S.TA.c[i] * 1000, 2) == 2.42  # TA
    assert round(-np.log10(M1.S.cs.H[i]), 2) == 7.72  # pH
    assert round(M1.S.cs.HCO3[i] * 1000, 2) == 2.25
    assert round(M1.S.cs.CO3[i] * 1000, 4) == 0.0661
    assert round(M1.S.cs.CO2aq[i] * 1000, 4) == 0.0386
    assert round(M1.S.cs.zsat[i]) == 2335
    assert round(M1.S.cs.zsnow[i]) == 4605

@pytest.mark.parametrize("solver", ["numba", "python"])
def test_gas_exchange(create_model, solver):
    """Test that isotope fractionation from a flux out of a reserevoir
    results in the corrrect fractionation, when using the numba solver

    """
   
    from esbmtk import Source, Sink, Connect, Reservoir, GasReservoir
    from esbmtk import ExternalCode, ReservoirGroup, AirSeaExchange
    from esbmtk import add_carbonate_system_1
    import numpy as np

    v0, c0, d0, M1 = create_model

    Source(name="SO1", species=M1.DIC)
    Sink(name="SI1", species=M1.DIC)

    ReservoirGroup(name = "S",        # Name of reservoir group
                        #volume = "1E5 l",       # see below
                        geometry = [0, 6000, 1],
                        delta   = {M1.DIC:0, M1.TA:0},  # dict of delta values
                        concentration = {M1.DIC:"2.1 mmol/l", M1.TA: "2.36 mmol/l"},
                        isotopes = {M1.DIC: True},
                        seawater_parameters = {"temperature": 25, "pressure": 1, "salinity" : 35},
                        #@carbonate_system= True,
                        register=M1,
                   )

    add_carbonate_system_1(rgs=[M1.S])

    GasReservoir(
            name="CO2_At",
            species=M1.CO2,
            reservoir_mass="1.833E20 mol",
            species_ppm="280 ppm",
            isotopes=True,
            delta = -7,
        )

    # DIC influx
    Connect(
        source=M1.SO1,  # source of flux
        sink=M1.S.DIC,  # target of flux
        rate="0 mol/yr",  # weathering flux in
        delta=0,
    )

    AirSeaExchange(
            gas_reservoir=M1.CO2_At,  # Reservoir
            liquid_reservoir=M1.S.DIC,  # ReservoirGroup
            species=M1.CO2,
            ref_species=M1.S.cs.CO2aq,
            solubility=M1.S.swc.SA_co2,  # float
            area=M1.S.area,
            piston_velocity="4.8 m/d",
            water_vapor_pressure=M1.S.swc.p_H2O,
            id ="A_sb",
        )   

    M1.run(solver=solver)

    M1.get_delta_values()
    # calculate theoretical equlibrium pCO2 based on pCO2
    epco2_at = M1.CO2_At.c*(1- M1.S.swc.p_H2O) *1e6
    # calculate equilibrium pCO2 based on CO2aq
    epco2_aq = M1.S.cs.CO2aq/M1.S.swc.K0 * 1e6
    
    diff_c = epco2_at[-3] - epco2_aq[-3]
    diff_d = M1.S.DIC.d[-3] - M1.CO2_At.d[-3]

    assert round(abs(diff_c),3) == 1.344 # ppm
    assert  round(diff_d,1) == 8.0
    
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


# the following do currently not work with numba

# @pytest.mark.parametrize("solver", ["numba","python"])
# def test_sum_fluxes(create_model,solver):
#     """
#     test that the code which adds fluxes to a reservoir yields the expected results
#     """

#     from esbmtk import Connect, Source, Sink

#     v0, c0, d0, M1 = create_model

#     Source(name="SO1", species=M1.CO2)
#     Source(name="SO2", species=M1.CO2)
#     Sink(name="SI1", species=M1.CO2)
#     Sink(name="SI2", species=M1.CO2)

#     Connect(
#         source=M1.SO1,  # source of flux
#         sink=M1.R1,  # target of flux
#         rate="100 mol/yr",  # weathering flux in
#         delta=d0,  # set a default flux
#     )
#     Connect(
#         source=M1.SO2,  # source of flux
#         sink=M1.R1,  # target of flux
#         rate="300 mol/yr",  # weathering flux in
#         delta=d0,  # set a default flux
#     )

#     Connect(
#         source=M1.R1,  # source of flux
#         sink=M1.SI1,  # target of flux
#         rate="250 mol/yr",  # weathering flux in
#         delta=d0,  # set a default flux
#     )

#     Connect(
#         source=M1.R1,  # source of flux
#         sink=M1.SI2,  # target of flux
#         rate="150 mol/yr",  # weathering flux in
#         delta=d0,  # set a default flux
#     )

#     M1.run(solver=solver)
# M1.get_delta_values()

#     assert M1.R1.c[-2] == c0
#     M1.R1.d[-2] == d0

#     # strangely, this fails at the moment
#     # assert R1.d[-]2 == 1

# @pytest.mark.parametrize("solver", ["numba","python"])
# def test_passive_sum(create_model,solver):
#     """
#     test that the code which adds fluxes to a reservoir yields the expected results
#     """

#     from esbmtk import Connect, Source, Sink

#     v0, c0, d0, M1 = create_model

#     Source(name="SO1", species=M1.CO2)
#     Source(name="SO2", species=M1.CO2)
#     Sink(name="SI1", species=M1.CO2)
#     Sink(name="SI2", species=M1.CO2)

#     Connect(
#         source=M1.SO1,  # source of flux
#         sink=M1.R1,  # target of flux
#         rate="100 mol/yr",  # weathering flux in
#         delta=d0,  # set a default flux
#     )
#     Connect(
#         source=M1.SO2,  # source of flux
#         sink=M1.R1,  # target of flux
#         rate="300 mol/yr",  # weathering flux in
#         delta=d0,  # set a default flux
#     )

#     Connect(
#         source=M1.R1,  # source of flux
#         sink=M1.SI1,  # target of flux
#         rate="250 mol/yr",  # weathering flux in
#         delta=0,  # set a default flux
#     )

#     Connect(
#         source=M1.R1,  # source of flux
#         sink=M1.SI2,  # target of flux
#     )

#     M1.run(solver=solver)
# M1.get_delta_values()

#     assert M1.R1.c[-2] == c0
#     assert abs(M1.R1.d[-2] - d0) < 1e-10
