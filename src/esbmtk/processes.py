from __future__ import annotations
import numpy as np
import typing as tp
import numpy.typing as npt
from . import Q_
from .esbmtk_base import esbmtkBase  # , Reservoir, Flux, Source, Sink
from .solver import get_l_mass

# declare numpy types
NDArrayFloat = npt.NDArray[np.float64]

np.set_printoptions(precision=4)
# from .connections import ConnnectionGroup

if tp.TYPE_CHECKING:
    from esbmtk import Flux, Model
#     from .esbmtk import Source, Reservoir, Sink, Flux, Model
#     from .extended_classes import GasReservoir, ReservoirGroup
#     from .connections import ConnectionGroup


class Process(esbmtkBase):
    """This class defines template for proycess which acts on one or more
    reservoir flux combinations. To use it, you need to create an
    subclass which defines the actual process implementation in their
    call method. See 'PassiveFlux as example'

    """

    from .esbmtk import Source, Reservoir, Sink

    def __init__(self, **kwargs: dict[str, any]) -> None:
        """
        Create a new process object with a given process type and options
        """

        self.__defaultnames__()  # default kwargs names
        self.__initerrormessages__()  # default error messages
        self.bem.update({"rate": "a string"})
        self.bem.update({"scale": "Number or quantity"})
        self.__initialize_keyword_variables__(kwargs)  # initialize keyword values

        self.__postinit__()  # do some housekeeping
        self.parent = self.register
        self.__register_name_new__()

    def __postinit__(self) -> None:
        """Do some housekeeping for the process class"""

        from esbmtk import Reservoir

        # legacy name aliases
        self.n: str = self.name  # display name of species
        if "reservoir" in self.kwargs:
            self.r: Reservoir = self.reservoir

        self.mo: Model = self.model
        self.f: Flux = self.flux

        # self.rm0: float = self.r.m[0]  # the initial reservoir mass

        if "reservoir" in self.kwargs and isinstance(self.r, Reservoir):
            self.direction: dict[Flux, int] = self.r.lio[self.f]

        self.delta = self.kwargs["delta"] if "delta" in self.kwargs else "None"
        self.__misc_init__()

    def __delayed_init__(self) -> None:
        """
        Initialize stuff which is only known after the entire model has been defined.
        This will be executed just before running the model. You need to add the following
        two lines somewhere in the init procedure (preferably by redefining __misc_init__)
        and redefine __delayed_init__ to actually execute any code below

        # this process requires delayed init
        self.mo.lto.append(self)

        """

        pass

    def __misc_init__(self) -> None:
        """This is just a place holder method which will be called by default
        in __post_init__() This can be overloaded to add additional
        code to the init procedure without the need to redefine
        init. This useful for processes which only define a call method.

        """

        pass

    def __defaultnames__(self) -> None:
        """Set up the default names and dicts for the process class. This
        allows us to extend these values without modifying the entire init process

        """

        from esbmtk import (
            Reservoir,
            ReservoirGroup,
            Source,
            Sink,
            GasReservoir,
            Flux,
            ConnectionGroup,
            Model,
        )

        # provide a dict of known keywords and types
        self.defaults: dict[str, list[any, tuple]] = {
            "name": ["None", (str)],
            "reservoir": ["None", (str, Reservoir, Source, Sink, GasReservoir)],
            "flux": ["None", (str, Flux)],
            "ref_flux": ["None", (str, Flux)],
            "rate": [0, (int, float, np.ndarray)],
            "delta": ["None", (int, float, np.ndarray, str)],
            "lt": ["None", (Flux)],
            "alpha": [0, (int, float, np.ndarray)],
            "scale": [1, (int, float, np.ndarray)],
            "ref_reservoirs": ["None", (Flux, Reservoir, GasReservoir, list, str)],
            "model": ["None", (str, Model)],
            "source": ["None", (str, Source, Reservoir, GasReservoir)],
            "register": [
                "None",
                (
                    str,
                    ConnectionGroup,
                    Reservoir,
                    ReservoirGroup,
                    GasReservoir,
                    Flux,
                    Model,
                ),
            ],
        }

        # provide a list of absolutely required keywords
        self.lrk: list[str] = ["name", "register"]

    def __register__(self, reservoir, flux: Flux) -> None:
        """Register the flux/reservoir pair we are acting upon, and register
        the process with the reservoir

        """

        from esbmtk import Reservoir

        # register the reservoir flux combination we are acting on
        self.f: Flux = flux
        self.r: Reservoir = reservoir
        # add this process to the list of processes acting on this reservoir
        reservoir.lop.append(self)
        flux.lop.append(self)
        # Add to model flux list
        reservoir.mo.lop.append(self)

    def show_figure(self, x, y) -> None:
        """Apply the current process to the vector x, and show the result as y.
        The resulting figure will be automatically saved.

        Example::
             process_name.show_figure(x,y)

        """
        pass


class GenericFunction(Process):
    """This Process class creates a GenericFunction instance which is
    typically used with the VirtualReservoir, and
    ExternalCode classes. This class is not user facing,
    please see the ExternalCode class docstring for the
    function template of a user provided function.

    see calc_carbonates in the carbonate chemistry for an example how
    to write a function for this class.

    """

    def __init__(self, **kwargs: dict[str, any]) -> None:
        """
        Create a new process object with a given process type and options

        """

        from esbmtk import ReservoirGroup
        from typing import Callable
        
        self.__defaultnames__()  # default kwargs names

        # list of allowed keywords
        defaults: dict[str, list[any, tuple]] = {
            "function": ["None", (str, Callable)],
            "input_data": ["None", (list, str)],
            "vr_data": ["None", (list, str)],
            "function_params": ["None", (list, str)],
            "model": ["None", (str, Model)],
            "ftype": ["None", (str)],
            "r_s": ["None", (str, ReservoirGroup)],
            "r_g": ["None", (str, ReservoirGroup)],
        }

        self.defaults.update(defaults)
        self.__initialize_keyword_variables__(kwargs)
        # required arguments
        self.lrk: list = ["name", "input_data", "vr_data", "function_params", "model"]

        self.mo = self.model
        self.parent = self.register

        self.__postinit__()  # do some housekeeping

        if self.mo.register == "local" and self.register == "None":
            self.register = self.mo

        self.__register_name_new__()  #

    def __call__(self, i: int) -> None:
        """Here we execute the user supplied function
        Where i = index of the current timestep

        """

        self.function(i, self.input_data, self.vr_data, self.function_params)


class AddSignal(Process):
    """This process adds values to the current flux based on the values provided by
    the signal object. This class is typically invoked through the connector object:

    Example::

     AddSignal(name = "name",
               reservoir = upstream_reservoir_handle,
               flux = flux_to_act_upon,
               lt = flux with lookup values,
     )

    where - the upstream reservoir is the reservoir the process belongs too,
    the flux is the flux to act upon, and lt= contains the flux object we lookup from

    """

    def __init__(self, **kwargs: dict[str, any]) -> None:
        """
        Create a new process object with a given process type and options
        """

        # get default names and update list for this Process
        self.__defaultnames__()  # default kwargs names
        self.lrk.extend(["lt", "flux", "reservoir"])  # new required keywords
        self.__initialize_keyword_variables__(kwargs)  # initialize keyword values

        # legacy variables
        self.mo = self.reservoir.mo
        self.__postinit__()  # do some housekeeping
        self.parent = self.register
        self.__register_name_new__()

        # defaults
        self.__execute__ = self.__add_with_fi__

    # setup a placeholder call function
    def __call__(self, i: int):
        return self.__execute__(i)

    # use this when we do isotopes
    def __add_with_fi__(self, i) -> None:
        """Each process is associated with a flux (self.f). Here we replace
        the flux value with the value from the signal object which
        we use as a lookup-table (self.lt)

        Note that the signal may also specify a delta value. Thus we
        need to
        1) get the mass of flux + signal
        2) add the delta of flux and signal
        3) calculate the new li and hi

        """

        # add signal mass to flux mass
        r = self.f.species.r
        fm = self.f.m[i]  # flux rate
        fl = self.f.l[i]  # flux rate light isotope
        sm = self.lt.m[i]  # signal mass
        sl = self.lt.l[i]  # signal li

        # get signal delta
        sd = 1000 * ((sm - sl) / sl - r) / r if sm > 0 else 0
        # print(f"Signal delta = {sd}")

        fd = 1000 * ((fm - fl) / fl - r) / r + sd if fm > 0 else sd
        fm += sm  # add signal mass
        fl = 1000.0 * fm / ((fd + 1000.0) * r + 1000.0)

        self.f.fa = np.array([fm, fl])
        # print(f"fa = {self.f.fa}\n")

    def __add_with_fa__(self, i) -> None:
        """same as above but use fa instead of flux data"""

        # add signal mass to flux mass
        r = self.f.species.r
        fm = self.f.fa[0]  # flux rate
        fl = self.f.fa[1]  # flux rate light isotope
        sm = self.lt.m[i]  # signal mass
        sl = self.lt.l[i]  # signal li

        # get signal delta
        sd = 1000 * ((sm - sl) / sl - r) / r

        fd = 1000 * ((fm - fl) / fl - r) / r + sd if fm > 0 else sd
        # set new flux rate
        fm += sm  # add signal mass
        fl = 1000.0 * fm / ((fd + 1000.0) * r + 1000.0)

        self.f.fa = np.array([fm, fl])

    def p_add_signal_fi(data, params, i) -> None:
        r: float = params[0]
        fm: float = data[0][i]  # fm
        fl: float = data[1][i]  # fl
        sm: float = data[2][i]  # sm
        sl: float = data[3][i]  # sd

        # get signal delta
        sd = 1000 * ((sm - sl) / sl - r) / r if sm > 0 else 0
        fd = 1000 * ((fm - fl) / fl - r) / r + sd if fm > 0 else sd
        # set new flux rate
        fm += sm  # add signal mass
        fl = 1000.0 * fm / ((fd + 1000.0) * r + 1000.0)

        data[4][:] = [fm, fl]

    def p_add_signal_fa(data, params, i) -> None:
        r: float = params[0]
        fm: float = data[2][0]
        fl: float = data[2][1]
        sm: float = data[0][i]
        sl: float = data[1][i]

        # get signal delta
        sd = 1000 * ((sm - sl) / sl - r) / r if sm > 0 else 0
        fd = 1000 * ((fm - fl) / fl - r) / r + sd if fm > 0 else sd
        fm += sm  # add signal mass
        fl = 1000.0 * fm / ((fd + 1000.0) * r + 1000.0)
        data[2][:] = [fm, fl]


class SaveFluxData(Process):
    """
    This process stores the flux data from each iteration into a vector
    Example::

        SaveFluxData(name = "Name",
                     flux = Flux Handle,
                     )

    """

    def __init__(self, **kwargs: dict[str, any]) -> None:
        """Initialize this Process"""
        # get default names and update list for this Process
        self.__defaultnames__()  # default kwargs names
        self.lrk.extend(["flux"])  # new required keywords

        self.__initialize_keyword_variables__(kwargs)  # initialize keyword values

        # legacy variables
        self.parent = self.register
        self.__postinit__()  # do some housekeeping
        self.__register_name_new__()

    # setup a placeholder call function
    def __call__(self, i: int):
        self.f[i] = self.f.fa

    def p_save_flux(data, params, i) -> None:
        data[0][i] = data[2][0]
        data[1][i] = data[2][1]


class ScaleFlux(Process):
    """This process scales the mass of a flux (m,l,h) relative to
    another flux. Delta is either taken from the upstream reservoir
    or set to a fixed value.  The scale factor "scale" and flux
    reference must be present when the object is being initalized

    Example::
         ScaleFlux(name = "Name",
                   reservoir = reservoir_handle (upstream or downstream)
                   scale = 1
                   ref_flux = flux we use for scale
                   )

    """

    def __init__(self, **kwargs: dict[str, any]) -> None:
        """Initialize this Process"""

        from esbmtk import Source

        # get default names and update list for this Process
        self.__defaultnames__()  # default kwargs names
        self.lrk.extend(["reservoir", "flux", "scale"])  # new required keywords
        self.__initialize_keyword_variables__(kwargs)  # initialize keyword values

        if "ref_reservoirs" in kwargs:
            self.ref_flux = kwargs["ref_reservoirs"]
        elif "ref_flux" not in kwargs:
            raise ValueError("You need to specify a value for ref_flux")

        # legacy variables
        self.mo = self.reservoir.mo
        self.__postinit__()  # do some housekeeping
        self.parent = self.register
        self.__register_name_new__()

        """ Decide how to calculate isotopes (if)
        If the connection specifies delta explicitly, this will
        override any settings in the source.
        """
        self.c = 1
        if self.delta != "None":
            # delta explicitly provided
            l = get_l_mass(1, self.delta, self.reservoir.species.element.r)
            self.c = l / (1 - l)
            self.__execute__ = self.__with_fixed_delta__
        elif isinstance(self.source, Source):
            if self.source.delta != "None":
                self.delta = self.source.delta
                l = get_l_mass(1, self.delta, self.reservoir.species.element.r)
                self.c = l / (1 - l)
                self.__execute__ = self.__with_fixed_delta__
            else:
                self.__execute__ = self.__without_isotopes__
        elif self.source.isotopes:
            self.__execute__ = self.__with_isotopes__
        else:
            self.__execute__ = self.__without_isotopes__

    # create stubs
    def __with_source__():
        raise NotImplementedError()

    # setup a placeholder call function
    def __call__(self, i: int):
        return self.__execute__(i)

    def __with_isotopes__(self, i: int) -> None:
        """Apply the scale factor. This is typically done through the the
        model execute method.
        Note that this will use the mass of the flux we use for scaling, but that we will set the
        delta according to reservoir this flux derives from

        """

        # get reference flux
        rm: float = self.ref_flux.fa[0] * self.scale

        # get the target isotope ratio based on upstream delta
        c = self.reservoir.l[i - 1] / (
            self.reservoir.c[i - 1] - self.reservoir.l[i - 1]
        )

        fl: float = rm * c / (c + 1)

        self.flux.fa[:] = [rm, fl]

    def __with_fixed_delta__(self, i: int) -> None:
        """similar to with isotopes, but this time we take the
        isotope value of the source (which has no array, just a fixed
        value"""

        # get reference flux
        rm: float = self.ref_flux.fa[0] * self.scale

        # get the target isotope ratio based on upstream delta
        c = self.c

        fl: float = rm * c / (c + 1)

        self.flux.fa[:] = [rm, fl]

    def __without_isotopes__(self, i: int) -> None:
        """Apply the scale factor. This is typically done through the the
        model execute method.
        Note that this will use the mass of the reference object, but that we will set the
        delta according to the reservoir (or the flux?)

        """

        self.f.fa = np.array(
            [
                self.ref_flux.fa[0] * self.scale,
                0,
            ]
        )

    def p_scale_flux_r(data, params, i) -> None:
        """delta is derived from upstream reservoir"""

        # params
        s: float = params[1]  # scale

        # data
        mf: float = data[6][0] * s  # mass of reference flux
        mr: float = data[3][i - 1]  # mass upstream reservoir
        lr: float = data[4][i - 1]  # li upstream reservoir

        # get the target isotope ratio based on upstream delta
        c = lr / (mr - lr)
        l = mf * c / (c + 1)

        data[5][:] = [mf, l]

    def p_scale_flux_fd(data, params, i) -> None:
        """delta is set to a fixed value"""

        # params
        s: float = params[1]  # scale
        c: float = params[2]  # l/h ratio

        # data
        mf: float = data[6][0] * s  # mass of reference flux

        # get the target isotope ratio based on upstream delta
        l = mf * c / (c + 1)

        data[5][:] = [mf, l]


class Fractionation(Process):
    """This process offsets the isotopic ratio of the flux by a given
       delta value. In other words, we add a fractionation factor

    Example::
         Fractionation(name = "Name",
                       reservoir = upstream_reservoir_handle,
                       flux = flux handle
                       alpha = 12 in permil (e.f)

    """

    def __init__(self, **kwargs: dict[str, any]) -> None:
        """Initialize this Process"""
        # get default names and update list for this Process
        self.__defaultnames__()  # default kwargs names
        self.lrk.extend(["reservoir", "flux", "alpha"])  # new required keywords

        self.__initialize_keyword_variables__(kwargs)  # initialize keyword values
        self.__postinit__()  # do some housekeeping

        # alpha is given in permil, but the fractionation routine expects
        # it as 1 + permil, i.e., 70 permil would 1.007

        self.alp = 1 + self.alpha / 1000
        self.mo = self.reservoir.mo
        self.parent = self.register
        self.__register_name_new__()

    def __call__(self, i: int) -> None:
        """
        Set flux isotope masses based on fractionation factor
        relative to reservoir

        """

        # print(f"self.f.m[i] =  {self.f.m[i]}")
        fm = self.f.fa[0]
        if fm != 0:
            rm = self.reservoir.m[i - 1]
            rl = self.reservoir.l[i - 1]
            a = self.alp

            fl = rl * fm / (a * rm + rl - a * rl)
            self.f.fa[:] = [fm, fl]
        else:
            self.f.fa[:] = [
                0,
                0,
            ]

        return

    def p_fractionation(data, params, i) -> None:
        # params
        a: float = params[1]  # alpha

        # data
        fm: float = data[4][0]  # flux mass
        rm: float = data[2][i - 1]  # 4 reservoir mass
        rl: float = data[3][i - 1]  # 4 reservoir light isotope

        fl = rl * fm / (a * rm + rl - a * rl)
        data[4][:] = [fm, fl]


class RateConstant(Process):
    """This is a wrapper for a variety of processes which depend on rate constants
    Please see the below class definitions for details on how to call them
    At present, the following processes are defined



    """

    def __init__(self, **kwargs: dict[str, any]) -> None:
        """Initialize this Process"""

        from esbmtk import (
            SeawaterConstants,
            GasReservoir,
            Reservoir,
            Source,
            Sink,
            Q_,
        )
        from typing import Callable

        # Note that self.lkk values also need to be added to the lkk
        # list of the connector object.

        # get default names and update list for this Process
        self.__defaultnames__()  # default kwargs names

        # update the allowed keywords
        defaults: dict[str, list[any, tuple]] = {
            "scale": [1, (int, float, np.ndarray)],
            "k_value": [1, (int, float, np.ndarray)],
            "ref_reservoirs": ["None", (str, list)],
            "reservoir_ref": ["None", (str, Reservoir, GasReservoir)],
            "left": [
                "None",
                (str, list, Reservoir, int, float, np.ndarray, np.ndarray),
            ],
            "right": ["None", (str, list, Reservoir, int, float, np.ndarray)],
            "gas": [
                "None",
                (str, Reservoir, GasReservoir, Source, Sink, np.ndarray, float),
            ],
            "liquid": ["None", (Reservoir, Source, Sink, float)],
            "solubility": ["None", (str, int, float, np.ndarray)],
            "piston_velocity": ["None", (str, int, float, np.ndarray)],
            "water_vapor_pressure": ["None", (str, int, float, np.ndarray)],
            "ref_species": ["None", (str, np.ndarray, float)],
            "seawaterconstants": ["None", (str, SeawaterConstants)],
            "isotopes": [False, (bool)],
            "function_reference": ["None", (str, Callable)],
            "f_0": ["None", (str, Q_, float, int)],
            "pco2_0": ["280 ppm", (str, Q_, float)],
            "ex": [0.2, (int, float)],
            "source": ["None", (str, Source, GasReservoir)],
        }

        self.defaults.update(defaults)
        # new required keywords
        # self.lrk.extend([["reservoir", "atmosphere"], ["scale", "k_value"], "register"])

        self.__initialize_keyword_variables__(kwargs)

        self.parent = self.register
        if "reservoir" in kwargs:
            self.mo = self.reservoir.mo
        elif "gas_reservoir" in kwargs:
            self.mo = self.gas_reservoir

        self.__misc_init__()
        self.__postinit__()  # do some housekeeping
        # legacy variables

        self.__register_name_new__()

        self.c = 1
        if self.reservoir.isotopes or self.isotopes:
            self.__execute__ = self.__with_isotopes__
            if self.delta != "None":
                self.__execute__ = self.__with_fixed_delta__
                l = get_l_mass(1, self.delta, self.reservoir.species.r)
                self.c = l
                # print(f"delta = {self.delta} setting c = {self.c}")
        else:
            self.__execute__ = self.__without_isotopes__

    def __postinit__(self) -> "None":
        self.mo = self.reservoir.mo

    # setup a placeholder call function
    def __call__(self, i: int):
        return self.__execute__(i)


class weathering(RateConstant):
    """This process calculates the flux as a function of the upstream
     reservoir concentration C and a constant which describes thet
     strength of relation between the reservoir concentration and
     the flux scaling

     F = f_0 * (scale * C/pco2_0)**ncc

     where C denotes the concentration in the ustream reservoir, k is a
     constant. This process is typically called by the connector
     instance. However you can instantiate it manually as

     weathering(
                       name = "Name",
                       reservoir = upstream_reservoir_handle,
                       reservoir_ref = reference_reservoir (Atmosphere)
                       flux = flux handle,
                       ex = exponent
                       pco2_0 = 280,
                       f_0 = 12 / 17e12
                       Scale =  1000,
                       delta = 0,

    )

    """

    def __misc_init__(self):
        """
        Scale the flux relative to the pco2_0 concentration.
        The delta values are derived from either:

        - the upstream reservoir
        - the upstream source
        - set to a fixed value

        """

        from esbmtk import Source

        if isinstance(self.f_0, str):
            self.f_0: float = Q_(self.f_0).to(self.mo.f_unit).magnitude
        elif isinstance(self.f_0, Q_):
            self.f_0: float = self.f_0.to(self.mo.f_unit).magnitude

        if isinstance(self.pco2_0, str):
            self.pco2_0 = Q_(self.pco2_0).to("ppm").magnitude * 1e-6
        elif isinstance(self.pco2_0, Q_):
            self.pco2_0 = self.pco2_0.magnitude.to("ppm").magnitude * 1e-6

        # if self.delta == "None" and  self.source.isotopes == False:
        #      self.fixed = True
        # delta provided explicitly
        if self.delta != "None":
            self.d = self.delta
            self.isotopes = True
            l = get_l_mass(1, self.d, self.source.species.element.r)
            self.c = l / (1 - l)
            self.func_name = self.p_weathering_fd
            self.__execute__ = self.__with_fixed_delta__
            self.fixed = True

        # source is Source and has delta
        elif self.source.isotopes and isinstance(self.source, Source):
            self.isotopes = True
            self.delta = self.source.delta
            self.func_name = self.p_weathering_fd
            self.__execute__ = self.__with_fixed_delta__
            self.c = self.source.c
            self.fixed = True

        # source is reservoir
        elif self.source.isotopes:
            self.isotopes = True
            self.func_name = self.p_weathering
            self.__execute__ = self.__with_isotopes__
            self.fixed = False

        # source is source, has no delta, and delta is not provided
        else:
            self.c = 1
            self.isotopes = False
            self.func_name = self.p_weathering_fd
            self.__execute__ = self.__without_isotopes__
            self.fixed = True

    def __without_isotopes__(self, i: int) -> None:
        f = self.scale * (
            self.f_0 * (self.reservoir_ref.c[i - 1] / self.pco2_0) ** self.ex
        )
        self.flux.fa = np.array([f, 0])

    def __with_isotopes__(self, i: int) -> None:
        """
        C = M/V so we express this as relative to mass which allows us to
        use the isotope data.

        The below calculates the flux as function of reservoir concentration,
        rather than scaling the flux.
        """

        c = self.flux.fa[1] / self.flux.fa[0]
        f = self.scale * (
            self.f_0 * (self.reservoir_ref.c[i - 1] / self.pco2_0) ** self.ex
        )
        fl = f * c
        self.flux.fa = np.array([f, fl])

    def __with_fixed_delta__(self, i: int) -> None:
        """ """

        f = self.scale * (
            self.f_0 * (self.reservoir_ref.c[i - 1] / self.pco2_0) ** self.ex
        )
        fl = f * self.c
        self.flux.fa = np.array([f, fl])

    def p_weathering(data, params, i) -> None:
        """delta value depends on upstream reservoir"""
        # params
        s: float = params[1]
        f_0: float = params[2]
        pco2_0: float = params[3]
        ex: float = params[4]

        # data
        cr = data[3][i - 1] / data[2][i - 1]
        c: float = data[1][i - 1]

        f: float = s * f_0 * (c / pco2_0) ** ex
        fl: float = f * cr
        data[0][:] = [f, fl]

    def p_weathering_fd(data, params, i) -> None:
        """delta value is fixed"""
        # params
        s: float = params[1]
        f_0: float = params[2]
        pco2_0: float = params[3]
        ex: float = params[4]
        cr = params[5]

        c: float = data[1][i - 1]
        f: float = s * f_0 * (c / pco2_0) ** ex
        fl = f * cr
        data[0][:] = [f, fl]


class MultiplySignal(Process):
    """This process mulitplies a given flux with the the data in the
    signal.  This class is typically invoked through the connector
    object: Note that this process will not modify the delta value of
    a given flux.  If you needto vary the delta value it is best to
    add a second signal which uses the add signal type.

    Example::
    
       MultiplySignal(name = "name",
               reservoir = upstream_reservoir_handle,
               flux = flux_to_act_upon,
               lt = flux with lookup values,
          )

    where the upstream reservoir is the reservoir the process belongs
    too the flux is the flux to act upon lt= contains the
    flux object we lookup from

    """

    def __init__(self, **kwargs: dict[str, any]) -> None:
        """
        Create a new process object with a given process type and options
        """

        # get default names and update list for this Process
        self.__defaultnames__()  # default kwargs names
        self.lrk.extend(["lt", "flux", "reservoir"])  # new required keywords
        self.__initialize_keyword_variables__(kwargs)  # initialize keyword values

        # legacy variables
        self.mo = self.reservoir.mo
        self.__postinit__()  # do some housekeeping
        self.parent = self.register
        self.__register_name_new__()

        # decide whichh call function to use
        # if self.mo.m_type == "both":

        # default
        self.__execute__ = self.__multiply_with_flux_fi__

    # setup a placeholder call function
    def __call__(self, i: int):
        return self.__execute__(i)

    # use this when we do isotopes
    def __multiply_with_flux_fi__(self, i) -> None:
        """Each process is associated with a flux (self.f). Here we replace
        the flux value with the value from the signal object which
        we use as a lookup-table (self.lt)
        """

        # multiply flux mass with signal
        c = self.lt.m[i]
        m = self.f.m[i] * c
        l = self.f.l[i] * c
        # h = self.f.h[i] * c
        # d = self.f.d[i]
        self.flux.fa = np.array([m, l])
        print(f"multiply fa {self.flux.fa}, f. = {self.f.m[i]}, c = {c} ")

    def __multiply_with_flux_fa__(self, i) -> None:
        """Each process is associated with a flux (self.f). Here we replace
        the flux value with the value from the signal object which
        we use as a lookup-table (self.lt)
        """

        # multiply flux mass with signal
        c = self.lt.m[i]
        m = self.f.fa[0] * c
        l = self.f.fa[1] * c
        self.flux.fa = np.array([m, l])

    def p_multiply_signal_fi(data, params, i) -> None:
        c = data[2][i]
        m = data[0][i] * c
        l = data[1][i] * c

        data[3][:] = [m, l]

    def p_multiply_signal_fa(data, params, i) -> None:
        c = data[0][i]
        m = data[1][0] * c  # m
        l = data[1][1] * c  # l

        data[1][:] = [m, l]


class VarDeltaOut(Process):
    """Unlike a passive flux, this process sets the flux istope ratio
    equal to the isotopic ratio of the reservoir. The
    init and register methods are inherited from the process
    class::
    
     VarDeltaOut(name = "name",
                reservoir = upstream_reservoir_handle,
                flux = flux handle,
                rate = rate,
     )
     
    """

    def __init__(self, **kwargs: dict[str, any]) -> None:
        """Initialize this Process"""

        from esbmtk import (
            Reservoir,
            Source,
        )

        # get default names and update list for this Process
        self.__defaultnames__()

        self.lrk.extend(["reservoir", "flux", "register"])  # required keywords

        self.__initialize_keyword_variables__(kwargs)
        self.mo = self.reservoir.mo
        self.parent = self.register
        self.__postinit__()  # do some housekeeping
        self.__register_name_new__()

        # decide which call function to use
        # if self.mo.m_type == "both":
        if self.reservoir.isotopes:
            # print(
            #    f"vardeltaout with isotopes for {self.reservoir.register.name}.{self.reservoir.name}"
            # )
            if isinstance(self.reservoir, Reservoir):
                # print("Using reservoir")
                self.__execute__ = self.__with_isotopes_reservoir__
            elif isinstance(self.reservoir, Source):
                # print("Using Source")
                self.__execute__ = self.__with_isotopes_source__
            else:
                raise ValueError(
                    f"{self.name}, reservoir must be of type Source or Reservoir, not {type(self.reservoir)}"
                )
        else:
            self.__execute__ = self.__without_isotopes__

    # setup a placeholder call function
    def __call__(self, i: int):
        return self.__execute__(i)

    def __with_isotopes_reservoir__(self, i: int) -> None:
        """Here we re-balance the flux. This code will be called by the
        apply_flux_modifier method of a reservoir which itself is
        called by the model execute method
        """

        m: float = self.flux.m[i]
        if m != 0:
            c = self.reservoir.l[i - 1] / (
                self.reservoir.m[i - 1] - self.reservoir.l[i - 1]
            )
            self.flux.fa = [m, m * c]

    def p_vardeltaout(data, params, i) -> None:
        # concentration times scale factor

        mf: float = data[0][i - 1]  # flux mass
        mr: float = data[1][i - 1]  # reservoir mass
        lr: float = data[2][i - 1]  # reservoir l

        c = lr / (mr - lr)
        data[3][:] = [mf, mf * c]

    def __with_isotopes_source__(self, i: int) -> None:
        """If the source of the flux is a source, there is only a single delta value.
        Changes to the flux delta are applied through the Signal class.
        """

        m: float = self.flux.m[i]
        if m != 0:
            c = self.reservoir.l[i - 1] / (
                self.reservoir.m[i - 1] - self.reservoir.l[i - 1]
            )
            self.flux.fa = [m, m * c]

    def __without_isotopes__(self, i: int) -> None:
        """Here we re-balance the flux. This code will be called by the
        apply_flux_modifier method of a reservoir which itself is
        called by the model execute method
        """

        raise NotImplementedError("vardeltaout w/o isotopes is not defined")


class ScaleRelativeToMass(RateConstant):
    """This process scales the flux as a function of the upstream
    reservoir Mass M and a constant which describes the
    strength of relation between the reservoir mass and
    the flux scaling

    F = F0 *  M * k

    where M denotes the mass in the ustream reservoir, k is a
    constant and F0 is the initial unscaled flux. This process is
    typically called by the connector instance. However you can
    instantiate it manually as

    Note that we scale the flux, rather than compute the flux!

    This is faster than setting a new flux, computing the isotope
    ratio and setting delta. So you either have to set the initial
    flux F0 to 1, or calculate the scale accordingly::

     ScaleRelativeToMass(
                       name = "Name",
                       reservoir= upstream_reservoir_handle,
                       flux = flux handle,
                       Scale =  1000,
     )

    """

    def __without_isotopes__(self, i: int) -> None:
        m: float = self.reservoir.m[i - 1] * self.scale
        self.flux.fa = [m, 0, 0, 0]

    def __with_isotopes__(self, i: int) -> None:
        """
        this will be called by the Model.run() method

        """
        m: float = self.reservoir.m[i - 1] * self.scale
        # d: float = self.reservoir.d[i - 1]
        c = self.reservoir.l[i - 1] / (
            self.reservoir.m[i - 1] - self.reservoir.l[i - 1]
        )
        l = m * c / (c + 1)
        self.flux.fa[:] = [m, l]

    def p_scale_relative_to_mass(data, params, i) -> None:
        # concentration times scale factor
        # params
        s: float = params[1]  # scale factor

        # data
        rm = data[2][i - 1]
        rl = data[3][i - 1]
        # rd = data[6][i - 1]

        m: float = rm * s  # new flux
        c: float = rl / (rm - rl)
        l: float = m * c / (c + 1)
        # d: float = data[1][i - 1]  # flux delta

        data[4][:] = [m, l]


class GasExchange(RateConstant):
    """
    Example::
    
      GasExchange(
          gas =GasReservoir, #
          liquid = Reservoir, #,
          ref_species = array of concentrations #
          solubility=Atmosphere.swc.SA_co2 [mol/(m^3 atm)],
          area = area, # m^2
          piston_velocity = m/year
          seawaterconstants = Ocean.swc
          water_vapor_pressure=Ocean.swc.p_H2O,
      )

    """

    # redefine misc_init which is being called by post-init
    def __misc_init__(self):
        """Set up input variables"""

        self.p_H2O = self.seawaterconstants.p_H2O
        self.a_dg = self.seawaterconstants.a_dg
        self.a_db = self.seawaterconstants.a_db
        self.a_u = self.seawaterconstants.a_u
        self.rvalue = self.liquid.sp.r
        self.volume = self.gas.volume

    def ode(self) -> None:
        return self.scale * (
            self.gas.c * (1 - self.p_H2O) * self.solubility - self.ref_species * 1000
        )  # area in m^2  # Atmosphere  # p_H2O  # SA_co2 = mol/(m^3 atm)  # [CO2]aq mol

    def __without_isotopes__(self, i: int) -> None:
        # set flux
        # note that the sink delta is co2aq as returned by the carbonate VR
        # this equation is for mmol but esbmtk uses mol, so we need to
        # multiply by 1E3

        f = self.scale * (  # area in m^2
            self.gas.c[i - 1]  # Atmosphere
            * (1 - self.p_H2O)  # p_H2O
            * self.solubility  # SA_co2 = mol/(m^3 atm)
            - self.ref_species[i - 1] * 1000  # [CO2]aq mol
        )
        # print(self.gas.c[i - 1])

        # changes in the mass of CO2 also affect changes in the total mass
        # of the atmosphere. So we need to update the reservoir volume
        # variable which we use the store the total atmospheric mass
        """The solver will use f and f12 to update mass, light
        isotope and concentration values in the Gas
        reservoir. However, Gas reservoirs track total gas pressure
        in the volume variable. This will not be updated by the
        solver. So we need to do it ourseleves

        """
        # self.gas.v[i] = self.gas.v[i - 1] + f * self.gas.mo.dt
        self.flux.fa[:] = [f, 1]

    def __with_isotopes__(self, i: int) -> None:
        """
        In the following I assume near neutral pH between 7 and 9, so that
        the isotopic composition of HCO3- is approximately equal to the isotopic
        ratio of DIC. The isotopic ratio of [CO2]aq can then be obtained from DIC via
        swc.e_db (swc.a_db)

        The fractionation factor subscripts denote the following:

        g = gaseous
        d = dissolved
        b = bicarbonate ion
        c = carbonate ion

        a_db is thus the fractionation factor between dissolved CO2aq and HCO3-
        and a_dg between CO2aq and CO2g

        ref_species = DIC.cs.CO2aq
        liquid.m = DIC.m (used to get the isotope ratio)

        gas.m = mass of CO2
        gas.l = mass of 12CO2
        gas.c = CO2 concentration
        gas.v = total mass of atmosphere

        """

        # equilibrium concentration of CO2 in water based on pCO2
        eco2_at = (
            self.gas.c[i - 1]  # p Atmosphere
            * (1 - self.p_H2O)  # p_H2O
            * self.solubility
        )
        # equilibrium concentration of CO2 in water based on CO2aq
        eco2_aq = self.ref_species[i - 1] * 1000

        #  flux
        f = self.scale * (eco2_at - eco2_aq)

        c13g = 1 - self.gas.l[i - 1] / self.gas.m[i - 1]
        c13aq = 1 - self.liquid.l[i - 1] / self.liquid.m[i - 1]
        # get 13C CO2 equlibrium concentration  CO2 in water based on pCO2
        eco2_at_13 = (
            self.gas.c[i - 1]
            * c13g
            * (1 - self.p_H2O)  # p_H2O
            * self.solubility
            * self.a_dg
        )

        # get 13C equilibrium  CO2 in water based on DIC m & l
        eco2_aq_13 = self.a_db * eco2_aq * c13aq

        # 13C flux
        f13 = self.scale * self.a_u * (eco2_at_13 - eco2_aq_13)
        f12 = f - f13

        self.flux.fa = [f, f12]
        """The solver will use f and f12 to update mass, light
        isotope and concentration values in the Gas
        reservoir. However, Gas reservoirs track total gas pressure
         in the volume variable. This will not be updated by the
        solver. So we need to do it ourseleves

        not working, see note below
        """
        self.gas.v[i] = self.gas.v[i - 1] + f * self.gas.mo.dt
        return [f, f12]
        # print()

    def __postinit__(self) -> None:
        """Do some housekeeping for the process class"""

        # legacy name aliases
        self.n: str = self.name  # display name of species
        self.r = self.liquid
        self.reservoir = self.liquid


class ScaleRelativeToConcentration(RateConstant):
    """This process calculates the flux as a function of the upstream
    reservoir concentration C and a constant which describes thet
    strength of relation between the reservoir concentration and
    the flux scaling:

    F = C * k

    where C denotes the concentration in the ustream reservoir, k is a
    constant. This process is typically called by the connector
    instance. However you can instantiate it manually as::

     ScaleRelativeToConcentration(
                       name = "Name",
                       reservoir= upstream_reservoir_handle,
                       flux = flux handle,
                       Scale =  1000,
                       delta = "None",
     )

    If delta is given the fluxes uses a ficed delta, If not it uses
    the upestream delta

    """

    def __without_isotopes__(self, i: int) -> None:
        m: float = self.reservoir.m[i - 1]
        if m > 0:  # otherwise there is no flux
            # convert to concentration
            c = m / self.reservoir.volume
            f = c * self.scale
            self.flux.fa = [
                f,
                0,
            ]

    def __with_isotopes__(self, i: int) -> None:
        """
        C = M/V so we express this as relative to mass which allows us to
        use the isotope data.

        The below calculates the flux as function of self.reservoir concentration,
        rather than scaling the flux.
        """

        rc: float = self.reservoir.c[i - 1]
        if rc > 0:  # otherwise there is no flux
            c = self.reservoir.l[i - 1] / (
                self.reservoir.m[i - 1] - self.reservoir.l[i - 1]
            )
            m = rc * self.scale
            l = m * c / (c + 1)
            self.f.fa = [m, l]

    def __with_fixed_delta__(self, i: int) -> None:
        """
        C = M/V so we express this as relative to mass which allows us to
        use the isotope data.

        The below calculates the flux as function of self.reservoir concentration,
        rather than scaling the flux.
        """

        rc: float = self.reservoir.c[i - 1]
        if rc > 0:  # otherwise there is no flux
            c = self.c
            m = rc * self.scale
            l = m * c / (c + 1)
            self.f.fa = [m, l]

    def p_scale_relative_to_concentration(data, params, i) -> None:
        """delta depends on upstream delta"""
        # data
        rm: float = data[0][i - 1]  # res mass
        if rm > 0:
            # params
            s: float = params[0]  # scale factor
            v: float = params[1]  # res volume
            fm: float = rm / v * s
            rl: float = data[1][i - 1]  # res li

            c: float = rl / (rm - rl)
            fl: float = fm * c / (c + 1)
            data[2][:] = [fm, fl]

        else:
            data[2][:] = [0.0, 0.0]

    def p_scale_relative_to_concentration_fd(data, params, i) -> None:
        """delta depends fixed value"""
        # data
        rm: float = data[0][i - 1]  # res mass

        if rm > 0:
            # params
            s: float = params[1]  # scale factor
            v: float = params[2]  # res volume
            fm: float = rm / v * s
            c: float = params[3]  # isotope ratio
            fl: float = fm * c / (c + 1)
            data[2][:] = [fm, fl]

        else:
            data[2][:] = [0.0, 0.0]
