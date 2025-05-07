"""esbmtk: A general purpose Earth Science box model toolkit.

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
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from .esbmtk_base import esbmtkBase
from .utility_functions import map_units

np.set_printoptions(precision=4)
# declare numpy types
NDArrayFloat = npt.NDArray[np.float64]


class Species2SpeciesError(Exception):
    """Custom Error Class."""

    def __init__(self, message):
        """Initialize Error Instance."""
        message = f"\n\n{message}\n"
        super().__init__(message)


class ScaleFluxError(Exception):
    """Custom Error Class."""

    def __init__(self, message):
        """Initialize Error Instance."""
        message = f"\n\n{message}\n"
        super().__init__(message)


class KeywordError(Exception):
    """Custom Error Class."""

    def __init__(self, message):
        """Initialize Error Instance."""
        message = f"\n\n{message}\n"
        super().__init__(message)


class Species2Species(esbmtkBase):
    """Connect two reservoir species to each other.

     This module creates the connecting flux and creates a connector object
     which stores all connection properties.

     For simple connections, the type flux type is derived implcitly
     from the specified parameters.  For complex connections, the flux
     type must be set explicitly. See the examples below:

    Parameters
    ----------
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

    The connection name is derived automatically, and can be queried with
    M.connection_summary()

    Connection Types:
    -----------------
    Connecting two reservoirs with a fixed rate:
    >>> Species2Species(  # deep box to sediment
    >>>    ctype="fixed",
    >>>    source=M.L_b.PO4,
    >>>    sink=M.Fb.PO4,
    >>>    scale="108 Gmol/year",
    >>>    id="shelf_burial",
    >>> )

    Connecting two reservoirs with a concentration dependent flux:
    >>> Species2Species(  # Surface to deep box
    >>>    source=M.L_b.PO4,
    >>>    sink=M.D_b.PO4,
    >>>    ctype="scale_with_concentration",
    >>>    scale=M.L_b.PO4.volume
    >>>    / M.tau
    >>>    * M.L_b.swc.density
    >>>    / 1000,  # concentration * volume = mass * 1/tau
    >>>    id="po4_productivity",
    >>> )

    Connecting two reservoirs using another flux as reference:
    The reference flux can either be given as a flux object, e.g., as
    returned from a flux lookup:

    M.flux_summary(filter_by="po4_productivity", return_list=True)[0]

    or simply as the id string:

    "po4_productivity"

    >>> Species2Species(  # deep box to sediment
    >>>    source=M.D_b.PO4,
    >>>    sink=M.Fb.PO4,
    >>>    ctype="scale_with_flux",
    >>>    ref_flux=""po4_productivity",
    >>>    # increase p_burial
    >>>    scale=(1 - M.remin_eff),  # burial of ~1% P
    >>>    id="burial",
    >>> )


    Useful methods in this class
    ----------------------------

      - info() will provide a short description of the connection objects.
      - list_processes() which will list all the processes which are associated with this connection.
      - update() which allows you to update connection properties after the connection has been created

    """

    def __init__(self, **kwargs):
        """Perform sanity checks.

               - whether the reservoirs exist
               - correct flux properties (this will be handled by the process object)
               - whether the processes do exist (hmmh, that implies that the optional processes do get registered with the model)
               - creates the correct default processes
               - and connects the reservoirs

        see the class documentation for details and examples
        """
        from esbmtk import (
            Q_,
            ConnectionProperties,
            Flux,
            GasReservoir,
            Model,
            Signal,
            Sink,
            Source,
            Species,
            Species2Species,
            SpeciesProperties,
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
            "alpha": ["None", (str, float)],
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
            "a_db": [1, (Callable, int, float)],
            "a_dg": [1, (Callable, int, float)],
            "a_u": [1, (Callable, int, float)],
            "area": ["None", (str, int, float, Q_)],
            "ex": [1, (int, float)],
            "pco2_0": ["280 ppm", (str, Q_)],
            "piston_velocity": ["None", (str, int, float)],
            "function_ref": ["None", (str, callable)],
            "ref_species": ["None", (str, Species)],
            "water_vapor_pressure": ["None", (str, float)],
        }

        # FIXME: We need a mechanism to first check for the
        # connection type (ctype) and then set the lrk list
        # depending on the connection type.!
        # provide a list of absolutely required keywords
        # the __scaleflux__ method (and similarm could return the
        # necessary information).
        self.lrk: list = ["source", "sink"]
        self.lop: list = []
        self.lof: list = []

        self.__initialize_keyword_variables__(kwargs)

        if self.register == "None":
            self.register = self.source.model
        elif isinstance(self.register, str):
            breakpoint()
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
        self.source_name = self.source.full_name
        self.sink_name = self.sink.full_name
        self.target = self.sink.full_name

        if isinstance(self.pco2_0, str):
            self.pco2_0 = Q_(self.pco2_0).to("ppm").magnitude * 1e-6
        elif isinstance(self.pco2_0, Q_):
            self.pco2_0 = self.pco2_0.magnitude.to("ppm").magnitude * 1e-6

        self.lop: list = self.pl if "pl" in kwargs else []
        if self.signal != "None":
            self.lop.append(self.signal)

        if self.rate != "None":
            if isinstance(self.rate, str):
                self._rate: float = Q_(self.rate).to(self.mo.f_unit).magnitude
            elif isinstance(self.rate, Q_):
                self._rate: float = self.rate.to(self.mo.f_unit).magnitude
            elif isinstance(self.rate, int | float):
                self._rate: float = self.rate

        # if no reference reservoir is specified, default to the upstream
        # reservoir
        if self.ref_reservoirs == "None":
            self.ref_reservoirs = kwargs["source"]

        if isinstance(self.source, Source):
            self.isotopes = self.sink.isotopes
        else:
            self.isotopes = self.source.isotopes

        self.get_species(self.r1, self.r2)  #
        self.mo: Model = self.sp.mo  # the current model handle
        self.lof: list[Flux] = []  # list of fluxes in this connection
        # get a list of all reservoirs registered for this species
        self.lor: list[Species] = self.mo.lor

        if self.scale == "None":
            self.scale = 1.0

        if isinstance(self.scale, str):
            self.scale = Q_(self.scale)

        if isinstance(self.scale, Q_):  # test what type of Quantity we have
            if self.scale.check("[volume]/[time]"):  # flux
                self.scale = self.scale.to(self.mo.r_unit).magnitude
            # test if flux
            elif self.scale.check("[mass]/[time]") or self.scale.check(
                "[substance]/[time]"
            ):
                self.scale = self.scale.to(self.mo.f_unit).magnitude
            elif self.scale.check("[mass]/[volume]"):  # concentration
                self.scale = self.scale.to(self.mo.c_unit).magnitude
            else:
                Species2SpeciesError(
                    f"No conversion to model units for {self.scale} specified"
                )

        self.__set_name__()  # get name of connection
        if self.model.debug:
            print(f"{self.name} isotopes = {self.isotopes}")

        self.__register_with_parent__()  # register connection in namespace
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
        """Create connection name.

        The name is derived according to the following scheme:

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
            if isinstance(self.source.parent, Reservoir | Source | SourceProperties):
                so = self.source.parent.name
            else:
                so = self.source.name

            si = (
                self.sink.parent.name
                if isinstance(self.sink.parent, Reservoir)
                else self.sink.name
            )

            self.name = f"Conn_{so}_to_{si}_{self.source.sp.name}"

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
        """Update connection properties.

        This will delete existing processes
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
        """Set the species by r2.

        However, if we have backward fluxes the species depends on the r2
        """
        from esbmtk import Source

        self.r = r1 if isinstance(self.r1, Source) else r2
        self.sp = self.kwargs.get("species", self.r.sp)

    def __create_flux__(self) -> None:
        """Create flux object.

        Register with reservoir and global namespace
        """
        from esbmtk import Flux, GasReservoir, Sink, Source, Species

        # test if default arguments present
        d = 0 if self.delta == "None" else self.delta
        r = f"0 {self.sp.mo.f_unit}" if self.rate == "None" else self.rate
        num = [""]

        if isinstance(self.source, Source):
            isotopes = self.sink.isotopes
        else:
            isotopes = self.source.isotopes

        if self.model.debug:
            print(f"cf: {self.full_name}, isotopes = {self.isotopes}")
        if isotopes:
            num.append("_l")

        for e in num:
            self.fh = Flux(
                species=self.sp,  # SpeciesProperties handle
                delta=d,  # delta value of flux
                rate=r,  # flux value
                plot=self.plot,  # display this flux?
                register=self,  # is this part of a group?
                isotopes=isotopes,
                id=f"{self.id}{e}",
            )
            if self.model.debug:
                print(f"cf: created {self.fh.full_name}, isotopes = {self.isotopes}")

            # register flux with its reservoirs
            if isinstance(self.r1, Source):
                # add the flux name direction/pair
                if isinstance(self.r2, Species | GasReservoir):
                    self.r2.lio[self.fh] = self.influx
                    # add the handle to the list of fluxes
                    self.r2.lof.append(self.fh)

                    # register flux and element in the reservoir.
                    self.__register_species__(self.r2, self.r1.sp)
                    if self.model.debug:
                        print(
                            f"cf: registered {self.fh.full_name} with {self.r2.full_name}"
                        )

            elif isinstance(self.r2, Sink):
                # add the flux name direction/pair
                self.r1.lio[self.fh] = self.outflux
                # add flux to the upstream reservoir
                self.r1.lof.append(self.fh)
                # register flux and element in the reservoir.
                self.__register_species__(self.r1, self.r2.sp)
                if self.model.debug:
                    print(
                        f"cf: registered {self.fh.full_name} with {self.r1.full_name}"
                    )

            elif isinstance(self.r1, Sink):
                raise Species2SpeciesError(
                    "The Sink must be specified as a destination (i.e., as second argument"
                )

            elif isinstance(self.r2, Source):
                raise Species2SpeciesError(
                    "The Source must be specified as first argument"
                )

            else:  # add the flux name direction/pair
                self.r1.lio[self.fh] = self.outflux
                self.r2.lio[self.fh] = self.influx
                self.r1.lof.append(self.fh)  # add flux to the upstream reservoir
                self.r2.lof.append(self.fh)  # add flux to the downstream reservoir
                self.__register_species__(self.r1, self.r1.sp)
                self.__register_species__(self.r2, self.r2.sp)
                if self.model.debug:
                    print(
                        f"cf: registered {self.fh.full_name} with {self.r1.full_name}\n"
                        f"cf: registered {self.fh.full_name} with {self.r2.full_name}\n"
                    )

            self.lof.append(self.fh)

    def __register_species__(self, r, sp) -> None:
        """Add flux to the correct element dictionary."""
        if sp.eh in r.doe:  # test if element key is present in reservoir
            r.doe[sp.eh].append(self.fh)  # add flux handle to dictionary list
        else:  # add key and first list value
            r.doe[sp.eh] = [self.fh]

    def __set_process_type__(self) -> None:
        """Deduce flux type based on the provided flux properties.

        The method calls the appropriate method init routine
        """
        from esbmtk import (
            Sink,
            Source,
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
            if self.rate == "None":
                raise ConnectionError(
                    "fixed/regular connections require the 'rate' keyword instead of 'scale'"
                )

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
        for f in self.lof:
            if self.bypass == "source" and not isinstance(self.source, Source):
                self.source.lif.append(f)
                print(f"bypassing {f.full_name} in {self.source.full_name}")
            elif self.bypass == "sink" and not isinstance(self.sink, Sink):
                self.sink.lif.append(f)
                print(f"bypassing {f.full_name} in {self.sink.full_name}")

    def __scaleflux__(self) -> None:
        """Scale a flux relative to another flux."""
        from esbmtk import Flux

        if self.model.debug:
            print(f"sf: {self.full_name}, isotopes = {self.isotopes}")

        if isinstance(self.ref_flux, str):
            f = self.mo.flux_summary(filter_by=self.ref_flux, return_list=True)[0]
            self.ref_flux = f

        if not isinstance(self.ref_flux, Flux):
            raise ScaleFluxError(
                f"\n {self.ref_flux} must be flux or a an id-string. Check spelling\n"
            )

        self.ref_flux.serves_as_input = True

        if self.isotopes == "None":
            raise ScaleFluxError(f"{self.name}: You need to set the isotope keyword")

        if self.k_value != "None":
            self.scale = self.k_value
            print("\n Warning: use scale instead of k_value for scaleflux type\n")

    def __weathering__(self):
        """Initialize weathering function."""
        from esbmtk import init_weathering, register_return_values

        self.isotopes = self.sink.isotopes
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
        """Initialize gas exachange."""
        from esbmtk.processes import init_gas_exchange
        from esbmtk.utility_functions import register_return_values

        ec = init_gas_exchange(self)
        register_return_values(ec, self.sink)

    def __rateconstant__(self) -> None:
        """Add rate constant type process."""
        if self.ctype == "scale_with_concentration":
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
        index = kwargs.get("index", 0)
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
    def epsilon(self) -> float | int:
        """Epsilon property."""
        return self._epsilon

    @epsilon.setter
    def epsilon(self, a: float | int) -> None:
        """Epsilon Setter."""
        if self.update and a != "None":
            self.__delete_process__()
            self.__delete_flux__()
            self._epsilon = a
            self.__set_process_type__()  # derive flux type and create flux(es)

    # ---- rate  ----
    @property
    def rate(self) -> float | int:
        """Rate property."""
        return self._rate

    @rate.setter
    def rate(self, r: str) -> None:
        """Rate Setter."""
        from . import Q_

        if self.update and r != "None":
            self.__delete_process__()
            self.__delete_flux__()
            self._rate = Q_(r).to(self.model.f_unit).magnitude
            self.__create_flux__()  # Source/Sink/Fixed
            self.__set_process_type__()  # derive flux type and create flux(es)

    # ---- delta  ----
    @property
    def delta(self) -> float | int:
        """Delta property."""
        return self._delta

    @delta.setter
    def delta(self, d: float | int) -> None:
        """Delta Setter."""
        if self.update and d != "None":
            self.__delete_process__()
            self.__delete_flux__()
            self._delta = d
            self.kwargs["delta"] = d
            self.__create_flux__()  # Source/Sink/Fixed
            self.__set_process_type__()  # derive flux type and create flux(es)


class ConnectionProperties(esbmtkBase):
    """ConnectionProperties Class.

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
            Q_,
            Flux,
            GasReservoir,
            Model,
            Reservoir,
            Signal,
            SinkProperties,
            SourceProperties,
            Species,
            SpeciesProperties,
        )

        self.defaults: dict[str, Any] = {
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
        self.lrk: list = ["source", "sink", "ctype"]
        self.__initialize_keyword_variables__(kwargs)

        if self.register == "None":
            self.register = self.source.model

        # # self.source.lor is a  list with the object names in the group
        self.mo = self.sink.lor[0].mo
        self.model = self.mo
        self.source_name = self.source.full_name
        self.sink_name = self.sink.full_name
        self.loc: list = []  # list of connection objects

        self.name = f"ConnGrp_{self.source.name}_to_{self.sink.name}_{self.id}"
        # fixme this results in duplicate names in the model namespace.
        # probably related to the create connection function
        # if self.id != "None":
        #     self.name = f"{self.name}@{self.id}"

        self.base_name = self.name
        self.parent = self.register
        self.__register_with_parent__()
        self.__create_connections__()

    def add_connections(self, **kwargs) -> None:
        """Add connections to the connection group."""
        self.__initialize_keyword_variables__(kwargs)
        self.__create_connections__()

    def __create_connections__(self) -> None:
        """Create Species2Species connection."""
        from esbmtk import Reservoir, SinkProperties, SourceProperties

        self.connections: list = []
        if isinstance(self.ctype, str):
            if isinstance(self.source, Reservoir | SinkProperties | SourceProperties):
                if self.species == "None":
                    for s in self.source.lor:
                        self.connections.append(s.species)
                else:
                    for s in self.species:
                        self.connections.append(s)

        elif isinstance(self.ctype, dict):
            # find all sub reservoirs which have been specified by the ctype keyword
            for r, _t in self.ctype.items():
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
                "alpha": "None",
                "rate": "None",
                "scale": "None",
                "ctype": "None",
                "ref_reservoirs": "None",
                "ref_flux": "None",
                "bypass": "None",
                "signal": "None",
            }

            """loop over entries in defaults dict
            test if key in default dict is also specified as connection keyword
            test if rate in kwargs, if sp in rate dict """
            for key, _value in self.c_defaults[sp.name].items():
                if key in self.kwargs and isinstance(self.kwargs[key], dict):
                    if key in self.kwargs and sp in self.kwargs[key]:
                        self.c_defaults[sp.n][key] = self.kwargs[key][sp]
                elif key in self.kwargs and self.kwargs[key] != "None":
                    # if value was supplied, update defaults dict
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
                register=self.model,
            )
            self.loc.append(a)  # add connection to list of connections

    def info(self) -> None:
        """List all connections in this group."""
        print(f"Group Connect from {self.source.name} to {self.sink.name}\n")
        print("The following Species2Species are part of this group\n")
        print("You can query the details of each connection like this:\n")
        for c in self.loc:
            print(f"{c.name}: {self.name}.{c.name}.info()")

        print("")
