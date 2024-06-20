"""
     esbmtk.connections

     Classes which handle the connections and fluxes between esbmtk objects
     like Species, Sources, and Sinks.

     esbmtk: A general purpose Earth Science box model toolkit
     Copyright (C), 2020 Ulrich G. Wortmann

     This program is free software: you can redistribute it and/or modify
     it under the terms of the GNU General Public License as published by
     the Free Software Foundation, either version 3 of the License, or
     (at your option) any later version.

     This program is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
     GNU General Public License for more details.

     You should have received a copy of the GNU General Public License
     along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import annotations
import uuid
import typing as tp
import numpy as np
import numpy.typing as npt
from .esbmtk import esbmtkBase
from .utility_functions import map_units
from .utility_functions import check_for_quantity

np.set_printoptions(precision=4)
# declare numpy types
NDArrayFloat = npt.NDArray[np.float64]


class Species2SpeciesError(Exception):
    def __init__(self, message):
        message = f"\n\n{message}\n"
        super().__init__(message)


class ScaleFluxError(Exception):
    def __init__(self, message):
        message = f"\n\n{message}\n"
        super().__init__(message)


class KeywordError(Exception):
    def __init__(self, message):
        message = f"\n\n{message}\n"
        super().__init__(message)


class Species2Species(esbmtkBase):
    """Two reservoirs connect to each other via at least one flux. This
     module creates the connecting flux and creates a connector object
     which stores all connection properties.

     For simple connections, the type flux type is derived implcitly
     from the specified parameters.  For complex connections, the flux
     type must be set explicitly. See the examples below:

     Parameters:
         - source: An object handle for a Source or Species
         - sink: An object handle for a Sink or Species
         - rate: A quantity (e.g., "1 mol/s"), optional
         - delta: The isotope ratio, optional
         - ref_reservoirs: Species or flux reference
         - epsilon: A fractionation factor, optional
         - id: A string wich will become part of the object name, it will override
           automatic name creation
         - signal: An object handle of signal, optional
         - ctype: connection type, see below
         - bypass :str optional defaults to "None" see scale with flux

    The connection name is derived automatically, see the documentation of
    __set_name__() for details

    Connect Types:
    -----------------
    Basic Connects (the advanced ones are below):

     - If both =rate= and =delta= are given, the flux is treated as a
        fixed flux with a given isotope ratio. This is usually the case for
        most source objects (they can still be affected by a signal, see
        above), but makes little sense for reservoirs and sinks.

     - If both the =rate= and =epsilon= are given, the flux rate is fixed
       (subject to any signals), but the isotopic ratio of the output
       flux depends on the isotopic ratio of the upstream reservoir
       plus any isotopic fractionation specified by =epsilon=. This is
       typically the case for fluxes which include an isotopic
       fractionation (i.e., pyrite burial). This combination is not
       particularly useful for source objects.

     - If the connection specifies only =delta= the flux is treated as a
       variable flux which is computed in such a way that the reservoir
       maintains steady state with respect to it's mass.

     - If the connection specifies only =rate= the flux is treated as a
       fixed flux which is computed in such a way that the reservoir
       maintains steady state with respect to it's isotope ratio.

    Connecting a Source to a Species
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Unless you use a Signal, a source typically provides a steady stream with a given isotope ratio (if used)

     Example::

        Species2Species(source =  Source,
                sink = downstrean reservoir,
                rate = "1 mol/s",
                delta = optional,
                signal = optional, see the signal documentation
                )

    Connecting a Species to Sink or another Species
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Here we can distinguish between cases where we use fixed flux, or a flux that reacts to in some way to the
     upstream reservoir (see the Species to Species section for a more complete treatment):

    Fixed outflux, with no isotope fractionation

    Example::

          Species2Species(source =  upstream reservoir,
                sink = Sink,
                rate = "1 mol/s",
                )

    Fixed outflux, with isotope fractionation

    Example::

          Species2Species(source =  upstream reservoir,
                sink = Sink,
                epsilon = -28,
                rate = "1 mol/s",
                )

    Advanced Connects
    --------------------

    You can aditionally define connection properties via the ctype
    keyword. This requires additional keyword parameters. The following values are
    recognized

    ctype = "scale_with_flux"
    ~~~~~~~~~~~~~~~~~~~~~~~~~

    This will scale a flux relative to another flux:

    Example::

        Species2Species(source =  upstream reservoir,
                sink = downstream reservoir,
                ctype = "scale_with_flux",
                ref_flux = flux handle,
                scale = 1, #
                )


    ctype = "scale_with_concentration"
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    This will scale a flux relative to the mass or concentration of a reservoir

    Example::

         Species2Species(source =  upstream reservoir,
                sink = downstream reservoir,
                ctype = "scale_with_concentration",
                ref_reservoirs = reservoir handle,
                scale = 1, # scaling factor
                )

    Useful methods in this class
    ----------------------------

    The following methods might prove useful:

      - info() will provide a short description of the connection objects.
      - list_processes() which will list all the processes which are associated with this connection.
      - update() which allows you to update connection properties after the connection has been created

    """

    def __init__(self, **kwargs):
        """The init method of the connector obbjects performs sanity checks e.g.:
               - whether the reservoirs exist
               - correct flux properties (this will be handled by the process object)
               - whether the processes do exist (hmmh, that implies that the optional processes do get registered with the model)
               - creates the correct default processes
               - and connects the reservoirs

        see the class documentation for details and examples

        """

        from esbmtk import (
            Q_,
            Source,
            Sink,
            SpeciesProperties,
            Species,
            GasReservoir,
            Model,
            Species2Species,
            Species2Species,
            ConnectionProperties,
            Flux,
            Signal,
        )

        # provide a dict of all known keywords and their type
        self.defaults: dict[str, list[any, tuple]] = {
            "id": ["", str],
            "source": ["None", (str, Source, Species, GasReservoir)],
            "sink": ["None", (str, Sink, Species, GasReservoir)],
            "delta": ["None", (int, float, str, dict)],
            "rate": ["None", (str, int, float, Q_, dict)],
            "pl": ["None", (list, str)],
            "epsilon": ["None", (int, float, str, dict)],
            "species": ["None", (SpeciesProperties, str)],
            "ctype": ["regular", (str)],
            "ref_reservoirs": ["None", (Species, GasReservoir, str, list)],
            "reservoir_ref": ["None", (GasReservoir, str)],
            "ref_flux": ["None", (Flux, str, list)],
            "ratio": ["None", (int, float, str)],
            "scale": [1, (int, float, Q_, str)],
            "ref_value": ["None", (str, int, float, Q_)],
            "k_value": ["None", (int, float, str, Q_)],
            "a_value": ["None", (int, float)],
            "b_value": ["None", (int, float)],
            "left": ["None", (list, int, float, Species, GasReservoir)],
            "right": ["None", (list, int, float, Species, GasReservoir)],
            "plot": ["yes", (str)],
            "groupname": [False, (bool)],
            "register": ["None", (str, Model, Species2Species, ConnectionProperties)],
            "signal": ["None", (Signal, str)],
            "bypass": ["None", (str, Species, GasReservoir)],
            "isotopes": [False, (bool)],
            "solubility": ["None", (str, int, float)],
            "area": ["None", (str, int, float, Q_)],
            "ex": [1, (int, float)],
            "pco2_0": ["280 ppm", (str, Q_)],
            "piston_velocity": ["None", (str, int, float)],
            "function_ref": ["None", (str, callable)],
            "ref_species": ["None", (str, Species)],
            "water_vapor_pressure": ["None", (str, float)],
        }

        # provide a list of absolutely required keywords
        self.lrk: tp.List = ["source", "sink"]

        self.lop: tp.List = []
        self.lof: tp.List = []

        self.__initialize_keyword_variables__(kwargs)

        if self.register == "None":
            self.register = self.source.register
        if self.id == "None":
            self.id = str(uuid.uuid4())[:8]

        # legacy names
        self.influx: int = 1
        self.outflux: int = -1
        self.mo = self.source.sp.mo
        self.model = self.mo
        self.sp = self.source.species
        self.p = 0  # the default process handle
        self.r1 = self.source
        self.r2 = self.sink
        self.parent = self.register

        if isinstance(self.pco2_0, str):
            self.pco2_0 = Q_(self.pco2_0).to("ppm").magnitude * 1e-6
        elif isinstance(self.pco2_0, Q_):
            self.pco2_0 = self.pco2_0.magnitude.to("ppm").magnitude * 1e-6

        self.lop: tp.List = self.pl if "pl" in kwargs else []
        if self.signal != "None":
            self.lop.append(self.signal)

        if self.rate != "None":
            if isinstance(self.rate, str):
                self._rate: float = Q_(self.rate).to(self.mo.f_unit).magnitude
            elif isinstance(self.rate, Q_):
                self._rate: float = self.rate.to(self.mo.f_unit).magnitude
            elif isinstance(self.rate, (int, float)):
                self._rate: float = self.rate

        # if no reference reservoir is specified, default to the upstream
        # reservoir
        if self.ref_reservoirs == "None":
            self.ref_reservoirs = kwargs["source"]

        # decide if this connection needs isotope calculations
        # if self.ctype == "scale_with_flux":
        #     pass
        if self.source.isotopes and self.sink.isotopes:
            self.isotopes = True

        self.get_species(self.r1, self.r2)  #
        self.mo: Model = self.sp.mo  # the current model handle
        self.lof: tp.List[Flux] = []  # list of fluxes in this connection
        # get a list of all reservoirs registered for this species
        self.lor: tp.List[Species] = self.mo.lor

        if self.scale == "None":
            self.scale = 1.0

        if isinstance(self.scale, str):
            self.scale = Q_(self.scale)

        if isinstance(self.scale, Q_):  # test what type of Quantity we have
            if self.scale.check(["volume]/[time"]):  # flux
                self.scale = self.scale.to(self.mo.r_unit)
            elif self.scale.check(["mass] / [time"]):  # flux
                self.scale = self.scale.to(self.mo.f_unit)
            elif self.scale.check("[mass]/[volume]"):  # concentration
                self.scale = self.scale.to(self.mo.c_unit)
            else:
                Species2SpeciesError(
                    f"No conversion to model units for {self.scale} specified"
                )

        self.__set_name__()  # get name of connection
        self.__register_name_new__()  # register connection in namespace
        self.__create_flux__()  # Source/Sink/Fixed
        self.__set_process_type__()  # derive flux type and create flux(es)

        # print(f"mo.reg = {self.mo.register}, slf reg = {self.register}")
        if self.mo.register == "local" and self.register == "None":
            self.register = self.mo

        self.source.loc.add(self)  # register connector with reservoir
        self.sink.loc.add(self)  # register connector with reservoir
        self.mo.loc.add(self)  # register connector with model

        # update ode constants
        self.s_index = self.__update_ode_constants__(self.scale)
        self.r_index = self.__update_ode_constants__(self.rate)
        self.d_index = self.__update_ode_constants__(self.delta)
        self.a_index = self.__update_ode_constants__(self.epsilon)

    def __set_name__(self):
        """The connection name is derived according to the following scheme:

        if manual connection
            if sink and source species are equal
               name = C_source2sink_species name
            otherwise
               name = C_source.species_name2sink.species_name

        if parent == ConnectionProperties
           if sink and source species are equal
               name = species name
            otherwise
               name = source.species2sink.species

        if id is set and id contains the species name, id will be
        taken as as connection name, otherwise, append id to the name

        """

        from esbmtk import Reservoir, Source, SourceProperties

        # same species?
        if self.sink.species.name == self.source.species.name:
            self.name = f"{self.source.species.name}"
        else:
            self.name = f"{self.source.species.name}_to_{self.sink.species.name}"

        # Connect by itself
        if not isinstance(self.parent, ConnectionProperties):  # manual connection
            if isinstance(self.source.parent, Reservoir):
                so = self.source.parent.name
            elif isinstance(self.source.parent, Source):
                so = self.source.parent.name
            elif isinstance(self.source.parent, SourceProperties):
                so = self.source.parent.name
            else:
                so = self.source.name

            si = (
                self.sink.parent.name
                if isinstance(self.sink.parent, Reservoir)
                else self.sink.name
            )

            self.name = f"C_{so}_to_{si}_{self.source.sp.name}"
        elif self.sink.species.name == self.source.species.name:
            self.name = f"{self.source.species.name}"
        else:
            self.name = f"{self.source.species.name}_to_{self.sink.species.name}"

        # always overide name with id for manual connections
        if self.ctype == "weathering":
            self.name = f"{self.name}_{self.id}"
        elif self.id != "None":
            if (self.source.species.name in self.id) or (
                self.sink.species.name in self.id
            ):
                self.name = f"{self.id}"
            else:
                self.name = f"{self.name}_{self.id}"

    def update(self, **kwargs):
        """Update connection properties. This will delete existing processes
        and fluxes, replace existing key-value pairs in the
        self.kwargs dict, and then re-initialize the connection.

        """

        raise NotImplementedError
        self.__delete_process__()
        self.__delete_flux__()
        self.kwargs.update(kwargs)
        self.__set__name__()  # get name of connection
        self.__init_connection__(self.kwargs)

    def get_species(self, r1, r2) -> None:
        """In most cases the species is set by r2. However, if we have
        backward fluxes the species depends on the r2

        """
        from esbmtk import Source

        self.r = r1 if isinstance(self.r1, Source) else r2
        self.sp = self.kwargs["species"] if "species" in self.kwargs else self.r.sp

    def __create_flux__(self) -> None:
        """Create flux object, and register with reservoir and global
        namespace"""

        from esbmtk import Flux, Source, Sink

        # test if default arguments present
        d = 0 if self.delta == "None" else self.delta
        r = f"0 {self.sp.mo.f_unit}" if self.rate == "None" else self.rate

        if self.sink.isotopes and self.source.isotopes:
            isotopes = True
        else:
            isotopes = False

        # if self.ctype == "weathering" and self.sp.name == "DIC":
        #    breakpoint()
        self.fh = Flux(
            species=self.sp,  # SpeciesProperties handle
            delta=d,  # delta value of flux
            rate=r,  # flux value
            plot=self.plot,  # display this flux?
            register=self,  # is this part of a group?
            isotopes=isotopes,
            id=self.id,
        )
        # register flux with its reservoirs
        if isinstance(self.r1, Source):
            # add the flux name direction/pair
            self.r2.lio[self.fh] = self.influx
            # add the handle to the list of fluxes
            self.r2.lof.append(self.fh)
            # register flux and element in the reservoir.
            self.__register_species__(self.r2, self.r1.sp)

        elif isinstance(self.r2, Sink):
            # add the flux name direction/pair
            self.r1.lio[self.fh] = self.outflux
            # add flux to the upstream reservoir
            self.r1.lof.append(self.fh)
            # register flux and element in the reservoir.
            self.__register_species__(self.r1, self.r2.sp)

        elif isinstance(self.r1, Sink):
            raise Species2SpeciesError(
                "The Sink must be specified as a destination (i.e., as second argument"
            )

        elif isinstance(self.r2, Source):
            raise Species2SpeciesError("The Source must be specified as first argument")

        else:  # add the flux name direction/pair
            self.r1.lio[self.fh] = self.outflux
            self.r2.lio[self.fh] = self.influx
            self.r1.lof.append(self.fh)  # add flux to the upstream reservoir
            self.r2.lof.append(self.fh)  # add flux to the downstream reservoir
            self.__register_species__(self.r1, self.r1.sp)
            self.__register_species__(self.r2, self.r2.sp)

        self.lof.append(self.fh)

    def __register_species__(self, r, sp) -> None:
        """Add flux to the correct element dictionary"""

        if sp.eh in r.doe:  # test if element key is present in reservoir
            r.doe[sp.eh].append(self.fh)  # add flux handle to dictionary list
        else:  # add key and first list value
            r.doe[sp.eh] = [self.fh]

    def __set_process_type__(self) -> None:
        """Deduce flux type based on the provided flux properties. The method calls the
        appropriate method init routine
        """

        from esbmtk import (
            Source,
            Sink,
        )

        self.r = self.r2 if isinstance(self.r1, Source) else self.r1
        # if signal is provided but rate is omitted
        if self.signal != "None" and self.rate == "None":
            self._rate = 0

        # if connection type is not set explicitly
        if (
            self.ctype == "None"
            or self.ctype.casefold() == "regular"
            or self.ctype.casefold() == "fixed"
        ):
            self.ctype = "regular"
            if self.delta == "None" and self.epsilon == "None" and self.isotopes:
                self._epsilon = 0

        elif self.ctype == "ignore":
            pass
        elif self.ctype == "scale_with_flux":
            self.__scaleflux__()
        elif self.ctype == "weathering":
            self.__weathering__()
        elif self.ctype == "gasexchange":
            self.__gasexchange__()
        elif self.ctype == "scale_with_concentration":
            self.__rateconstant__()
        elif self.ctype != "manual":
            print(f"Species2Species Type {self.ctype} is unknown")
            raise Species2SpeciesError(f"Unknown connection type {self.ctype}")

        # check if flux should bypass any reservoirs
        if self.bypass == "source" and not isinstance(self.source, Source):
            self.source.lof.remove(self.fh)
        elif self.bypass == "sink" and not isinstance(self.sink, Sink):
            self.sink.lof.remove(self.fh)
            print(f"bypassing {self.fh.full_name} in {self.sink.full_name}")

    def __scaleflux__(self) -> None:
        """Scale a flux relative to another flux"""

        from esbmtk import Flux

        if not isinstance(self.ref_flux, Flux):
            raise Species2SpeciesError("Scale reference must be a flux")

        if self.isotopes == "None":
            raise ScaleFluxError(f"{self.name}: You need to set the isotope keyword")

        if self.k_value != "None":
            self.scale = self.k_value
            print(f"\n Warning: use scale instead of k_value for scaleflux type\n")

    def __weathering__(self):
        from esbmtk import init_weathering, register_return_values

        ec = init_weathering(
            self,  # connection object
            self.reservoir_ref,  # current pCO2
            self.pco2_0,  # reference pCO2
            self.scale,  # area fraction
            self.ex,  # exponent
            self.rate,  # initial flux
        )
        register_return_values(ec, self.sink)

    def __gasexchange__(self):
        from esbmtk import init_gas_exchange, register_return_values

        ec = init_gas_exchange(self)
        register_return_values(ec, self.sink)

    def __rateconstant__(self) -> None:
        """Add rate constant type process"""

        if self.ctype == "scale_with_concentration":
            if self.k_value != "None":
                self.scale = self.k_value
                print(
                    f"\n Warning: use scale instead of k_value for scale with concentration type\n"
                )

            self.scale = map_units(
                self,
                self.scale,
                self.mo.c_unit,
                self.mo.f_unit,
                self.mo.r_unit,
                self.mo.v_unit,
            )

    def info(self, **kwargs) -> None:
        """Show an overview of the object properties.
        Optional arguments are
        index  :int = 0 this will show data at the given index
        indent :int = 0 indentation

        """
        index = 0 if "index" not in kwargs else kwargs["index"]
        if "indent" not in kwargs:
            indent = 0
            ind = ""
        else:
            indent = kwargs["indent"]
            ind = " " * indent

        # print basic data bout this Species2Species
        print(f"{ind}{self.__str__(kwargs)}")

        print(f"{ind}Fluxes:")
        for f in sorted(self.lof):
            f.info(indent=indent, index=index)

    # ---- Property definitions to allow for connection updates --------
    """ Changing the below properties requires that we delete all
    associated objects (processes), and determines the new flux type,
    and initialize/register these with the connection and model.
    We also have to update the keyword arguments as these are used
    for the log entry

    """

    # ---- epsilon ----
    @property
    def epsilon(self) -> tp.Union[float, int]:
        return self._epsilon

    @epsilon.setter
    def epsilon(self, a: tp.Union[float, int]) -> None:
        if self.update and a != "None":
            self.__delete_process__()
            self.__delete_flux__()
            self._epsilon = a
            self.__set_process_type__()  # derive flux type and create flux(es)

    # ---- rate  ----
    @property
    def rate(self) -> tp.Union[float, int]:
        return self._rate

    @rate.setter
    def rate(self, r: str) -> None:
        from . import Q_

        if self.update and r != "None":
            self.__delete_process__()
            self.__delete_flux__()
            self._rate = Q_(r).to(self.model.f_unit).magnitude
            self.__create_flux__()  # Source/Sink/Fixed
            self.__set_process_type__()  # derive flux type and create flux(es)

    # ---- delta  ----
    @property
    def delta(self) -> tp.Union[float, int]:
        return self._delta

    @delta.setter
    def delta(self, d: tp.Union[float, int]) -> None:
        if self.update and d != "None":
            self.__delete_process__()
            self.__delete_flux__()
            self._delta = d
            self.kwargs["delta"] = d
            self.__create_flux__()  # Source/Sink/Fixed
            self.__set_process_type__()  # derive flux type and create flux(es)


class ConnectionProperties(esbmtkBase):
    """ConnectionProperties

        Connect reservoir/sink/source groups when at least one of the
        arguments is a reservoirs_group object. This method will
        create regular connections for each matching species.

        Use the connection.update() method to fine tune connections
        after creation

    Example::

        ConnectionProperties(source =  upstream reservoir / upstream reservoir group
           sink = downstrean reservoir / downstream reservoirs_group
           delta = defaults to zero and has to be set manually
           epsilon =  defaults to zero and has to be set manually
           rate = shared between all connections
           ref_reservoirs = shared between all connections
           ref_flux = shared between all connections
           species = list, optional, if present, only these species will be connected
           ctype = needs to be set for all connections. Use "Fixed"
                   unless you require a specific connection type
           pl = [list]) process list. optional, shared between all connections
           id = optional identifier, passed on to individual connection
           plot = "yes/no" # defaults to yes, shared between all connections
        )

        ConnectionProperties(
                  source=OM_Weathering,
                  sink=Ocean,
                  rate={DIC: f"{OM_w} Tmol/yr" ,
                        ALK: f"{0} Tmol/yr"},
                  ctype = {DIC: "Fixed",
                           ALK: "Fixed"},
                )


    """

    def __init__(self, **kwargs) -> None:
        from esbmtk import (
            SourceProperties,
            Reservoir,
            Species,
            GasReservoir,
            SinkProperties,
            Signal,
            SpeciesProperties,
            Flux,
            Model,
            Q_,
        )

        self.defaults: dict[str, any] = {
            "id": ["None", (str)],
            "source": [
                "None",
                (str, SourceProperties, Species, Reservoir, GasReservoir),
            ],
            "sink": ["None", (str, SinkProperties, Species, Reservoir, GasReservoir)],
            "delta": ["None", (str, dict, tuple, int, float)],
            "rate": ["None", (Q_, str, dict, tuple, int, float)],
            "pl": ["None", (str, dict, tuple)],
            "signal": ["None", (str, Signal, dict)],
            "epsilon": ["None", (str, dict, tuple, int, float)],
            "species": ["None", (str, dict, tuple, list, SpeciesProperties)],
            "ctype": ["None", (str, dict, tuple)],
            "ref_reservoirs": ["None", (str, dict, tuple, Species)],
            "ref_flux": ["None", (str, dict, tuple, Flux)],
            "plot": ["yes", (str, dict, tuple)],
            "scale": [1, (str, dict, tuple, int, float, Q_)],
            "bypass": ["None", (dict, tuple, str)],
            "register": ["None", (str, tuple, Model)],
            "save_flux_data": [False, (bool, tuple)],
            "ref_species": ["None", (str, Species)],
            "water_vapor_pressure": ["None", (str, float)],
            "piston_velocity": ["None", (str, int, float)],
            "solubility": ["None", (str, int, float)],
            "area": ["None", (str, int, float, Q_)],
            "ex": [1, (int, float)],
            "pco2_0": ["280 ppm", (str, Q_)],
        }

        # provide a list of absolutely required keywords
        self.lrk: tp.List = ["source", "sink", "ctype"]
        self.__initialize_keyword_variables__(kwargs)

        if self.register == "None":
            self.register = self.source.register

        # # self.source.lor is a  list with the object names in the group
        self.mo = self.sink.lor[0].mo
        self.model = self.mo
        self.loc: tp.List = []  # list of connection objects

        self.name = f"CG_{self.source.name}_to_{self.sink.name}_{self.id}"
        # fixme this results in duplicate names in the model namespace.
        # probably related to the create connection function
        # if self.id != "None":
        #     self.name = f"{self.name}@{self.id}"

        self.base_name = self.name
        self.parent = self.register
        self.__register_name_new__()
        self.__create_connections__()

    def add_connections(self, **kwargs) -> None:
        """Add connections to the connection group"""

        self.__initialize_keyword_variables__(kwargs)
        self.__create_connections__()

    def __create_connections__(self) -> None:
        """Create Species2Species"""

        from esbmtk import Reservoir, SinkProperties, SourceProperties

        self.connections: tp.List = []

        if isinstance(self.ctype, str):
            if isinstance(self.source, (Reservoir, SinkProperties, SourceProperties)):
                if self.species == "None":
                    for s in self.source.lor:
                        self.connections.append(s.species)
                else:
                    for s in self.species:
                        self.connections.append(s)

        elif isinstance(self.ctype, dict):
            # find all sub reservoirs which have been specified by the ctype keyword
            for r, t in self.ctype.items():
                self.connections.append(r)

        # now we need to create defaults for all connections
        self.c_defaults: dict = {}  # connection dictionary with defaults
        # loop over species
        for sp in self.connections:  # ["SO4", "H2S"]
            # print(f"found species: ----------- {sp.name} ------------")
            self.c_defaults[sp.n] = {
                # "cid": self.id,
                "cid": "None",
                "plot": "yes",
                "delta": "None",
                "epsilon": "None",
                "rate": "None",
                "scale": "None",
                "ctype": "None",
                "ref_reservoirs": "None",
                "ref_flux": "None",
                "bypass": "None",
                "signal": "None",
            }

            # loop over entries in defaults dict
            for key, value in self.c_defaults[sp.name].items():
                # test if key in default dict is also specified as connection keyword
                # test if rate in kwargs, if sp in rate dict
                if key in self.kwargs and isinstance(self.kwargs[key], dict):
                    if key in self.kwargs and sp in self.kwargs[key]:
                        self.c_defaults[sp.n][key] = self.kwargs[key][sp]
                elif key in self.kwargs and self.kwargs[key] != "None":
                    # if value was supplied, update defaults dict
                    if self.kwargs[key] != "None":
                        self.c_defaults[sp.n][key] = self.kwargs[key]

            a = Species2Species(
                source=getattr(self.source, sp.n),
                sink=getattr(self.sink, sp.n),
                rate=self.c_defaults[sp.n]["rate"],
                delta=self.c_defaults[sp.n]["delta"],
                epsilon=self.c_defaults[sp.n]["epsilon"],
                plot=self.c_defaults[sp.n]["plot"],
                ctype=self.c_defaults[sp.n]["ctype"],
                scale=self.c_defaults[sp.n]["scale"],
                bypass=self.c_defaults[sp.n]["bypass"],
                signal=self.c_defaults[sp.n]["signal"],
                ref_reservoirs=self.c_defaults[sp.n]["ref_reservoirs"],
                ref_flux=self.c_defaults[sp.n]["ref_flux"],
                groupname=True,
                id=self.id,
                register=self,
            )

            self.loc.append(a)  # add connection to list of connections
            if self.mo.debug:
                print(
                    f"created connection with full name {a.full_name}, registered to {self.name} "
                    f"fn = {self.full_name}"
                )

    def info(self) -> None:
        """List all connections in this group"""

        print(f"Group Connect from {self.source.name} to {self.sink.name}\n")
        print("The following Species2Species are part of this group\n")
        print(f"You can query the details of each connection like this:\n")
        for c in self.loc:
            print(f"{c.name}: {self.name}.{c.name}.info()")

        print("")
