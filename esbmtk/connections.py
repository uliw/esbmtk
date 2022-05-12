"""

     esbmtk.connections

     Classes which handle the connections and fluxes between esbmtk objects
     like Reservoirs, Sources, and Sinks.

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

# from pint import UnitRegistry
from __future__ import annotations

# from numbers import Number
import typing as tp
import numpy as np
import copy as cp
import logging
from .utility_functions import map_units
from .processes import *
from .esbmtk import *
from esbmtk import Q_
from .utility_functions import check_for_quantity

np.set_printoptions(precision=4)


class Connect(esbmtkBase):
    """Two reservoirs connect to each other via at least one flux. This
     module creates the connecting flux and creates a connector object
     which stores all connection properties.

     For simple connections, the type flux type is derived implcitly
     from the specified parameters.  For complex connections, the flux
     type must be set explicitly. See the examples below:

     Parameters:
         - source: An object handle for a Source or Reservoir
         - sink: An object handle for a Sink or Reservoir
         - rate: A quantity (e.g., "1 mol/s"), optional
         - delta: The isotope ratio, optional
         - ref_reservoirs: Reservoir or flux reference
         - alpha: A fractionation factor, optional
         - id: A string wich will become part of the object name, it will override
           automatic name creation
         - plot: "yes" or "no", defaults to "yes"
         - signal: An object handle of signal, optional
         - pl: A list of process objects, optional
         - ctype: connection type, optional, this allows to scale a flux in response to other
           reservoirs and fluxes
         - bypass :str optional defaults to "None" see scale with flux
         - save_flux_data: bool, defaults to model setting, unles specified

    The connection name is derived automatically, see the documentation of
    __set_name__() for details

     Connection Types:
     -----------------
     Basic Connections (the advanced ones are below):

     - If both =rate= and =delta= are given, the flux is treated as a
        fixed flux with a given isotope ratio. This is usually the case for
        most source objects (they can still be affected by a signal, see
        above), but makes little sense for reservoirs and sinks.

     - If both the =rate= and =alpha= are given, the flux rate is fixed
       (subject to any signals), but the isotopic ratio of the output
       flux depends on the isotopic ratio of the upstream reservoir
       plus any isotopic fractionation specified by =alpha=. This is
       typically the case for fluxes which include an isotopic
       fractionation (i.e., pyrite burial). This combination is not
       particularly useful for source objects.

     - If the connection specifies only =delta= the flux is treated as a
       variable flux which is computed in such a way that the reservoir
       maintains steady state with respect to it's mass.

     - If the connection specifies only =rate= the flux is treated as a
       fixed flux which is computed in such a way that the reservoir
       maintains steady state with respect to it's isotope ratio.

     Examples of Basic Connections
     -----------------------------

     Connecting a Source to a Reservoir
     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

     Unless you use a Signal, a source typically provides a steady stream with a given isotope ratio (if used)

     Example::

        Connect(source =  Source,
                sink = downstrean reservoir,
                rate = "1 mol/s",
                delta = optional,
                signal = optional, see the signal documentation)

     Connecting a Reservoir to Sink or another Reservoir
     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

     Here we can distinguish between cases where we use fixed flux, or a flux that reacts to in some way to the
     upstream reservoir (see the Reservoir to Reservoir section for a more complete treatment):

     Fixed outflux, with no isotope fractionation

     Example::

          Connect(source =  upstream reservoir,
                sink = Sink,
                rate = "1 mol/s",)

     Fixed outflux, with isotope fractionation

     Example::

          Connect(source =  upstream reservoir,
                sink = Sink,
                alpha = -28,
                rate = "1 mol/s",)

     Advanced Connections
     --------------------

     You can aditionally define connection properties via the ctype
     keyword. This requires additional keyword parameters. The following values are
     recognized

     ctype = "scale_with_flux"
     ~~~~~~~~~~~~~~~~~~~~~~~~~

     This will scale a flux relative to another flux:

     Example::

         Connect(source =  upstream reservoir,
                sink = downstream reservoir,
                ctype = "scale_with_flux",
                ref_flux = flux handle,
                scale = a scaling factor, optional, defaults to 1
                bypass :str = "source"/"sink"
                )

    if bypass is set the flux will not affect the upstream or
    downstream reservoir. This is a more generalized approach than the
    virtual flux type. This is useful to describe reactions which
    split into two elements (i.e., deprotonation), or combine fluxes
    from two reservoirs (say S, and O) into a reservoir which records
    SO4.

    ctype = "virtual_flux"
    ~~~~~~~~~~~~~~~~~~~~~~~~~

    This will scale a flux relative to another flux, similar to
    scaleflux. However, in this case, the flux will only affect the
    sink, and not the source. This is useful to describe reactions which
    split into two elements (i.e., deprotonation).

    Example::

         Connect(source =  upstream reservoir,
                sink = downstream reservoir,
                ctype = "scale_with_flux",
                ref_flux = flux handle,
                scale = a scaling factor, optional, defaults to 1
                )

     ctype = "scale_with_mass" and "scale_with_concentration"
     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

     This will scale a flux relative to the mass or concentration of a reservoir

     Example::

         Connect(source =  upstream reservoir,
                sink = downstream reservoir,
                ctype = "scale_with_mass",
                ref_reservoirs = reservoir handle,
                scale = a scaling factor, optional, defaults to 1
    )

     ctype = "scale_relative_to_multiple_reservoirs"
     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

     This process scales the flux as a function one or more reservoirs
     or constants which describes the
     strength of relation between the reservoir concentrations and
     the flux scaling

     F = C1 * C2 * k

     where Ci denotes the concentrartion in one  or more reservoirs, k is one
     or more constants.

     Example::

         Connect(source =  upstream reservoir,
                sink = downstream reservoir,
                ctype = "scale_relative_to_multiple_reservoirs"
                ref_reservoirs = [r1, r2, k etc] # you must provide at least one
                scale = a scaling factor, optional, defaults to 1
    )

     scale is an overall scaling factor.


     ctype = "flux_balance"
     ~~~~~~~~~~~~~~~~~~~~~~

    This type can be used to express equilibration fluxes
    between two reservoirs. This connection type, takes three parameters:

    - =left= is a list which can contain constants and/or reservoirs. The
      list must contain at least one valid element. All elements in this
      list will be multiplied with each other. E.g. if we have a list
      with one constant and one reservoir, the reservoir concentration
      will be multiplied with the constant. If we have two reservoirs,
      the respective reservoir concentrations will be multiplied with
      each other.
    - =right= similar to =left= The final flux rate will be computed as
      the difference between =left= and =right=
    - =k_value= a constant which will be multiplied with the difference
      between =left=and =right=

     Example::

         Connect(source=R_CO2,         # target of flux
                 sink=R_HCO3,          # source of flux
                 rate="1 mol/s",       # flux rate
                 ctype="flux_balance", # connection type
                 scale=1,            # global scaling factor, optional
                 left=[K1, R_CO2],     # where K1 is a constant
                 right=[R_HCO3, R_Hplus])


     Useful methods in this class
     ----------------------------
     The following methods might prove useful

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

        from . import ureg, Q_

        # provide a dict of all known keywords and their type
        self.defaults: dict[str, list[any, tuple]] = {
            "id": ["", str],
            "source": ["None", (str, Source, Reservoir, GasReservoir)],
            "sink": ["None", (str, Sink, Reservoir, GasReservoir)],
            "delta": ["None", (int, float, str)],
            "rate": ["None", (str, int, float, Q_)],
            "pl": ["None", (list, str)],
            "alpha": ["None", (int, float, str)],
            "species": ["None", (Species, str)],
            "ctype": ["None", (str)],
            "ref_reservoirs": ["None", (Reservoir, GasReservoir, str, list)],
            "reservoir_ref": ["None", (GasReservoir, str)],
            "ref_flux": ["None", (Flux, str, list)],
            "ratio": ["None", (int, float, str)],
            "scale": [1, (int, float, Q_, str)],
            "ref_value": ["None", (str, int, float, Q_)],
            "k_value": ["None", (int, float, str, Q_)],
            "a_value": ["None", (int, float)],
            "b_value": ["None", (int, float)],
            "left": ["None", (list, int, float, Reservoir, GasReservoir)],
            "right": ["None", (list, int, float, Reservoir, GasReservoir)],
            "plot": ["yes", (str)],
            "groupname": [False, (bool)],
            "register": ["None", (str, Model, Connection, Connect, ConnectionGroup)],
            "signal": ["None", (Signal, str)],
            "bypass": ["None", (str, Reservoir, GasReservoir)],
            "isotopes": [False, (bool)],
            "solubility": ["None", (str, int, float)],
            "area": ["None", (str, int, float)],
            "ex": [1, (int, float)],
            "pco2_0": ["None", (str, Q_)],
            "piston_velocity": ["None", (str, int, float)],
            "function_ref": ["None", (str, callable)],
            "save_flux_data": ["None", (bool, str)],
        }

        # provide a list of absolutely required keywords
        self.lrk: list = ["source", "sink", "register", "id"]

        self.lop: list = list()
        self.lof: list = list()

        # validate and initialize instance variables
        self.__initialize_keyword_variables__(kwargs)

        # legacy names
        self.influx: int = 1
        self.outflux: int = -1
        self.mo = self.source.sp.mo
        self.model = self.mo
        self.sp = self.source.species
        self.p = 0  # the default process handle
        self.r1: (Process, Reservoir) = self.source
        self.r2: (Process, Reservoir) = self.sink
        self.parent = self.register
        
        if "pl" in kwargs:
            self.lop: list[Process] = self.pl
        else:
            self.lop: list[Process] = []

        if self.signal != "None":
            self.lop.append(self.signal)

        # if we have a signal, save flux data
        if self.signal != "None":
            self.save_flux_data = True
        # else if save_flux_data is unsepcified, use model default
        elif self.save_flux_data == "None":
            self.save_flux_data = self.source.sp.mo.save_flux_data

        if self.rate != "None":
            if isinstance(self.rate, str):
                self._rate: float = Q_(self.rate).to(self.mo.f_unit).magnitude
            elif isinstance(self.rate, Q_):
                self._rate: float = self.rate.to(self.mo.f_unit).magnitude
            elif isinstance(self.rate, (int, float)):
                self._rate : float = self.rate
                

        # if no reference reservoir is specified, default to the upstream
        # reservoir
        if self.ref_reservoirs == "None":
            self.ref_reservoirs = kwargs["source"]

        # decide if this connection needs isotope calculations
        if self.source.isotopes or self.sink.isotopes:
            self.isotopes = True

       
        self.get_species(self.r1, self.r2)  #
        self.mo: Model = self.sp.mo  # the current model handle
        self.lof: list[Flux] = []  # list of fluxes in this connection
        # get a list of all reservoirs registered for this species
        self.lor: list[Reservoir] = self.mo.lor

        # make sure scale is a number in model units

        if self.scale == "None":
            self.scale = 1.0

        if isinstance(self.scale, str):
            self.scale = Q_(self.scale)

        if isinstance(self.scale, Q_):
            # test what type of Quantity we have
            if self.scale.check(["volume]/[time"]):  # flux
                self.scale = self.scale.to(self.mo.r_unit)
            elif self.scale.check(["mass] / [time"]):  # flux
                self.scale = self.scale.to(self.mo.f_unit)
            elif self.scale.check("[mass]/[volume]"):  # concentration
                self.scale = self.scale.to(self.mo.c_unit)
            else:
                ValueError(f"No conversion to model units for {self.scale} specified")

        # if sink and source a regular, the name will be simply C_S_2_S
        # if we deal with ReservoirGroups we need to reflect this in the
        # connection name

        self.__set_name__()  # get name of connection
        self.__register_name_new__()  # register connection in namespace
        self.__create_flux__()  # Source/Sink/Regular
        self.__set_process_type__()  # derive flux type and create flux(es)

        # print(f"mo.reg = {self.mo.register}, slf reg = {self.register}")
        if self.mo.register == "local" and self.register == "None":
            self.register = self.mo

        self.source.loc.add(self)  # register connector with reservoir
        self.sink.loc.add(self)  # register connector with reservoir
        self.mo.loc.add(self)  # register connector with model

        # This should probably move to register fluxes
        self.__register_process__()

        logging.info(f"Created {self.full_name}")

    def __set_name__(self):
        """The connection name is derived according to the following scheme:

        if manual connection
            if sink and source species are equal
               name = C_source2sink_species name
            otherwise
               name = C_source.species_name2sink.species_name

        if parent == ConnectionGroup
           if sink and source species are equal
               name = species name
            otherwise
               name = source.species2sink.species

        if id is set and id contains the species name, id will be
        taken as as connection name, otherwise, append id to the name

        """

        # same species?
        if self.sink.species.name == self.source.species.name:
            self.name = f"{self.source.species.name}"
        else:
            self.name = f"{self.source.species.name}_to_{self.sink.species.name}"

        # Connection by itself
        if not isinstance(self.parent, ConnectionGroup):
            self.name = f"C_{self.source.name}_to_{self.sink.name}_{self.name}"

        # id set?
        if self.id != "None":
            if (self.source.species.name in self.id) or (
                self.sink.species.name in self.id
            ):
                self.name = f"{self.id}"
            else:
                self.name = f"{self.name}_{self.id}"

        # always overide name with id for manual connections
        if not isinstance(self.parent, ConnectionGroup):
            if self.id != "None":
                self.name = f"{self.id}"

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
        # print(f"r1 = {r1.n}, r2 = {r2.n}")
        if isinstance(self.r1, Source):
            self.r = r1
        else:  # in this case we do have an upstream reservoir
            self.r = r2

        # test if species was explicitly given
        if "species" in self.kwargs:  # this is a quick fix only
            self.sp = self.kwargs["species"]
        else:
            self.sp = self.r.sp  # get the parent species

    def __create_flux__(self) -> None:
        """Create flux object, and register with reservoir and global
        namespace"""

        # test if default arguments present
        if self.delta == "None":
            d = 0
        else:
            d = self.delta

        if self.rate == "None":
            r = f"0 {self.sp.mo.f_unit}"
            # self._rate = r
        else:
            r = self.rate

        # derive flux unit from species obbject
        funit = self.sp.mu + "/" + str(self.sp.mo.bu)  # xxx

        self.fh = Flux(
            species=self.sp,  # Species handle
            delta=d,  # delta value of flux
            rate=r,  # flux value
            plot=self.plot,  # display this flux?
            register=self,  # is this part of a group?
            isotopes=self.isotopes,
            id=self.id,
            save_flux_data=self.save_flux_data,
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
            raise NameError(
                "The Sink must be specified as a destination (i.e., as second argument"
            )

        elif isinstance(self.r2, Source):
            raise NameError("The Source must be specified as first argument")

        else:  # this is a regular connection
            # add the flux name direction/pair
            self.r1.lio[self.fh] = self.outflux
            # add the flux name direction/pair
            self.r2.lio[self.fh] = self.influx
            # add flux to the upstream reservoir
            self.r1.lof.append(self.fh)
            # add flux to the downstream reservoir
            self.r2.lof.append(self.fh)
            self.__register_species__(self.r1, self.r1.sp)
            self.__register_species__(self.r2, self.r2.sp)

        self.lof.append(self.fh)

    def __register_species__(self, r, sp) -> None:
        """Add flux to the correct element dictionary"""
        # test if element key is present in reservoir
        if sp.eh in r.doe:
            # add flux handle to dictionary list
            r.doe[sp.eh].append(self.fh)
        else:  # add key and first list value
            r.doe[sp.eh] = [self.fh]

    def __register_process__(self) -> None:
        """Register all flux related processes"""

        # first test if we have a signal in the list. If so,
        # remove signal and replace with process

        p_copy = cp.copy(self.lop)
        for p in p_copy:  # loop over process list if provided during init
            if isinstance(p, Signal):
                self.lop.remove(p)
                if p.ty == "addition":
                    # create AddSignal Process object
                    n = AddSignal(
                        name=p.n + "_addition_process",
                        reservoir=self.r,
                        flux=self.fh,
                        lt=p.data,
                        register=self.fh,
                    )
                    self.lop.insert(0, n)  # signals must come first
                    logging.debug(f"Inserting {n.n} in {self.name} for {self.r.n}")
                elif p.ty == "multiplication":
                    # create AddSignal Process object
                    n = MultiplySignal(
                        name=p.n + "_multiplication_process",
                        reservoir=self.r,
                        flux=self.fh,
                        lt=p.data,
                        register=self.fh,
                    )
                    self.lop.insert(len(self.lop), n)  # multiplaction should come last
                    logging.debug(f"Inserting {n.n} in {self.name} for {self.r.n}")
                else:
                    raise ValueError(f"Signal type {p.ty} is not defined")

        # ensure that processes are in the correct order
        self.__move_process_to_top_of_queue__(self.lop, MultiplySignal)
        self.__move_process_to_top_of_queue__(self.lop, AddSignal)
        self.__move_process_to_top_of_queue__(self.lop, ScaleFlux)
        self.__move_process_to_top_of_queue__(self.lop, VarDeltaOut)
        self.__move_process_to_end_of_queue__(self.lop, SaveFluxData)

        # nwo we can register everythig on lop
        i = 0
        for p in self.lop:
            if isinstance(p, (MultiplySignal)) and i > 0:
                p.__execute__ = p.__multiply_with_flux_fa__
                p.__get_process_args__ = p.__get_process_args_fa__
            if isinstance(p, (AddSignal)) and i > 0:
                print(f"Found Add signal")
                p.__execute__ = p.__add_with_fa__
                p.__get_process_args__ = p.__get_process_args_fa__

            p.__register__(self.r, self.fh)
            i += 1

    def __move_process_to_top_of_queue__(self, lop: list, ptype: any) -> None:
        """Return a copy of lop where ptype has been moved to the top of lop"""
        p_copy = cp.copy(lop)
        for p in p_copy:  # loop over process list if provided during init
            if isinstance(p, ptype):
                lop.remove(p)
                lop.insert(0, p)  # insert at top

    def __move_process_to_end_of_queue__(self, lop: list, ptype: any) -> None:
        """Return a copy of lop where ptype has been moved to the top of lop"""
        p_copy = cp.copy(lop)
        for p in p_copy:  # loop over process list if provided during init
            if isinstance(p, ptype):
                lop.remove(p)
                lop.append(p)  # insert at end

    def __set_process_type__(self) -> None:
        """Deduce flux type based on the provided flux properties. The method calls the
        appropriate method init routine
        """

        if isinstance(self.r1, Source):
            self.r = self.r2
        else:
            self.r = self.r1

        # if signal is provided but rate is omitted
        if self.signal != "None" and self.rate == "None":
            self._rate = "0 mmol/y"

        # if connection type is not set explicitly
        if self.ctype == "None" or self.ctype.casefold() == "regular":
            # set the fundamental flux type based on the flux arguments given
            if self.delta != "None" and self.rate != "None":
                # self.__vardeltaout__()
                pass  # if delta and rate are specified, do nothing
            # variable flux with fixed delta
            elif self.delta != "None":  # if delta is set
                self.__passivefluxfixeddelta__()
            elif self.rate != "None":  # if rate is set
                if self.delta != "None" and not isinstance(self.source, Sink):
                    self.__vardeltaout__()  # variable delta with fixed flux
            else:  # if neither are given -> default varflux type
                self._delta = 0
                self.__passiveflux__()

        elif self.ctype == "scale_to_input":
            self.__scale_to_input__()
        elif self.ctype == "flux_diff":
            self.__vardeltaout__()
            self.__flux_diff__()
        elif self.ctype == "scale_with_flux":
            self.__scaleflux__()
        elif self.ctype == "weathering":
            self.__rateconstant__()
        elif self.ctype == "virtual_flux":
            self.__virtual_flux__()
            # self.__vardeltaout__()
        elif self.ctype == "copy_flux":
            self.__scaleflux__()
            # self.__vardeltaout__()
        elif self.ctype == "scale_with_mass":
            self.__rateconstant__()
        elif self.ctype == "scale_with_concentration":
            self.__rateconstant__()
        elif self.ctype == "scale_relative_to_multiple_reservoirs":
            self.__rateconstant__()
        elif self.ctype == "flux_balance":
            self.__rateconstant__()
        elif self.ctype == "react_with":
            self.__reaction__()
        elif self.ctype == "monod_type_limit":
            # self.__vardeltaout__()
            self.__rateconstant__()
        elif self.ctype == "manual":
            pass
        else:
            print(f"Connection Type {self.ctype} is unknown")
            raise ValueError(f"Unknown connection type {self.ctype}")

        # Set optional flux processes
        if self.alpha != "None":
            self.__alpha__()  # Set optional flux processes
            # self.__vardeltaout__()

        # check if flux should bypass any reservoirs

        if self.bypass == "source" and not isinstance(self.source, Source):
            self.source.lof.remove(self.fh)
        elif self.bypass == "sink" and not isinstance(self.sink, Sink):
            self.sink.lof.remove(self.fh)
            # print(f"removing {self.fh.full_name} from {self.sink.full_name} lof")

        if self.save_flux_data:
            ph = SaveFluxData(
                name=f"{self.fh.full_name}_Pfd",
                flux=self.fh,
                register=self.fh,
            )
            self.lop.append(ph)

    def __passivefluxfixeddelta__(self) -> None:
        """Just a wrapper to keep the if statement manageable"""

        ph = PassiveFlux_fixed_delta(
            name="Pfd",
            reservoir=self.r,
            flux=self.fh,
            register=self.fh,
            delta=self.delta,
        )  # initialize a passive flux process object
        self.lop.append(ph)

    def __vardeltaout__(self) -> None:
        """Unlike a passive flux, this process sets the output flux from a
        reservoir to a fixed value, but the isotopic ratio of the
        output flux will be set equal to the isotopic ratio of the
        upstream reservoir.

        """

        ph = VarDeltaOut(
            name="Pvdo",
            reservoir=self.source,
            flux=self.fh,
            register=self.fh,
            rate=self.rate,
        )
        self.lop.append(ph)

    def __scaleflux__(self) -> None:
        """Scale a flux relative to another flux"""

        if not isinstance(self.ref_flux, Flux):
            raise ValueError("Scale reference must be a flux")

        if self.k_value != "None":
            self.scale = self.k_value
            print(f"\n Warning: use scale instead of k_value for scaleflux type\n")

        ph = ScaleFlux(
            name="PSF",
            source=self.source,
            reservoir=self.r,
            flux=self.fh,
            register=self.fh,
            scale=self.scale,
            ref_flux=self.ref_flux,
            delta=self.delta,
        )
        self.lop.append(ph)

        # if self.bypass == "source":
        #     self.source.lof.remove(self.fh)
        # elif self.bypass == "sink":
        #     self.sink.lof.remove(self.fh)
        #     print(f"removing {self.fh.full_name} from {self.sink.full_name} lof")
        # elif self.bypass == "None":
        #     pass
        # else:
        #     raise ValueError(f"bypass must be None/source/sink but not {self.bypass}")

    def __virtual_flux__(self) -> None:
        """Create a virtual flux. This is similar to __scaleflux__, however the new flux
        will only affect the sink, and not the source.

        """

        if self.k_value != "None":
            self.scale = self.k_value
            print(
                f"\n Warning: use scale instead of k_value for scale relative to multiple reservoirs\n"
            )

        if not isinstance(self.kwargs["ref_reservoirs"], Flux):
            raise ValueError("Scale reference must be a flux")

        ph = ScaleFlux(
            name="PSFV",
            reservoir=self.r,
            flux=self.fh,
            register=self.fh,
            scale=self.scale,
            ref_flux=self.ref_flux,
            delta=self.delta,
        )
        self.lop.append(ph)

        # this flux must not affect the source reservoir
        self.r.lof.remove(self.fh)

    def __flux_diff__(self) -> None:
        """Scale a flux relative to the difference between
        two fluxes

        """

        if self.k_value != "None":
            self.scale = self.k_value
            print(
                f"\n Warning: use scale instead of k_value for scale relative to multiple reservoirs\n"
            )

        if not isinstance(self.kwargs["ref_reservoirs"], list):
            raise ValueError("ref must be a list")

        ph = FluxDiff(
            name="PSF",
            reservoir=self.r,
            flux=self.fh,
            register=self.fh,
            scale=self.scale,
            ref_reservoirs=self.ref_reservoirs,
        )
        self.lop.append(ph)

    def __reaction__(self) -> None:
        """Just a wrapper to keep the if statement manageable"""

        if not isinstance(self.ref_reservoirs, (Flux)):
            raise ValueError("Scale reference must be a flux")

        if self.k_value != "None":
            self.scale = self.k_value
            print(f"\n Warning: use scale instead of k_value for reaction type\n")

        ph = Reaction(
            name="RF",
            reservoir=self.r,
            flux=self.fh,
            register=self.fh,
            scale=self.scale,
            ref_reservoirs=self.ref_reservoirs,
        )

        # we need to make sure to remove the flux referenced by
        # react_with is removed from the list of fluxes in this
        # reservoir. This should probably move in to the React Class
        self.r2.lof.remove(self.ref_reservoirs)
        self.lop.append(ph)

    def __passiveflux__(self) -> None:
        """Just a wrapper to keep the if statement manageable"""

        ph = PassiveFlux(
            name="_PF",
            reservoir=self.r,
            register=self.fh,
            flux=self.fh,
            scale=self.scale,
        )  # initialize a passive flux process object
        self.lop.append(ph)  # add this process to the process list

    def __scale_to_input__(self) -> None:
        """Just a wrapper to keep the if statement manageable"""

        ph = ScaleRelativeToInputFluxes(
            name="_SRTIF",
            reservoir=self.r,
            register=self.fh,
            flux=self.fh,
            scale=self.scale,
        )  # initialize a passive flux process object
        self.lop.append(ph)  # add this process to the process list

    def __alpha__(self) -> None:
        """Just a wrapper to keep the if statement manageable"""

        ph = Fractionation(
            name="_Pa",
            reservoir=self.r,
            flux=self.fh,
            register=self.fh,
            alpha=self.kwargs["alpha"],
        )
        self.lop.append(ph)  #

    def __rateconstant__(self) -> None:
        """Add rate constant type process"""

        from . import ureg, Q_

        # this process requires that we use the vardeltaout process
        if self.mo.m_type != "mass_only":
            # self.__vardeltaout__()
            pass

        if self.ctype == "scale_with_mass":
            if self.k_value != "None":
                self.scale = self.k_value
                print(
                    f"\n Warning: use scale instead of k_value for scale with mass type\n"
                )

            self.scale = map_units(self.scale, self.mo.m_unit)
            ph = ScaleRelativeToMass(
                name="_PkM",
                reservoir=self.ref_reservoirs,
                flux=self.fh,
                register=self.fh,
                scale=self.scale,
            )

        elif self.ctype == "scale_with_concentration":

            if self.k_value != "None":
                self.scale = self.k_value
                print(
                    f"\n Warning: use scale instead of k_value for scale with concentration type\n"
                )

            self.scale = map_units(
                self.scale, self.mo.c_unit, self.mo.f_unit, self.mo.r_unit
            )
            # print(
            #    f"Registering PKC with ref = {self.ref.full_name}, scale = {self.scale}"
            # )

            ph = ScaleRelativeToConcentration(
                name="PkC",
                reservoir=self.ref_reservoirs,
                flux=self.fh,
                register=self.fh,
                scale=self.scale,
            )
            # print(f"Process Name {ph.full_name}")

        elif self.ctype == "weathering":

            ph = weathering(
                name="Pw",
                source=self.source,
                reservoir=self.r,
                flux=self.fh,
                register=self.fh,
                scale=self.scale,
                reservoir_ref=self.reservoir_ref,
                ex=self.ex,
                pco2_0=self.pco2_0,
                f_0=self.rate,
                delta=self.delta,
            )

        elif self.ctype == "scale_relative_to_multiple_reservoirs":

            if self.k_value != "None":
                self.scale = self.k_value
                print(
                    f"\n Warning: use scale instead of k_value for scale relative to multiple reservoirs\n"
                )

            self.scale = map_units(
                self.scale, self.mo.c_unit, self.mo.f_unit, self.mo.r_unit
            )
            ph = ScaleRelative2otherReservoir(
                name="PkC",
                reservoir=self.source,
                ref_reservoirs=self.ref_reservoirs,
                flux=self.fh,
                register=self.fh,
                scale=self.scale,
            )

        elif self.ctype == "flux_balance":
            self.k_value = map_units(
                self.k_value, self.mo.c_unit, self.mo.f_unit, self.mo.r_unit
            )
            ph = Flux_Balance(
                name="_Pfb",
                reservoir=self.source,
                left=self.left,
                right=self.right,
                flux=self.fh,
                register=self.fh,
                k_value=self.k_value,
            )

        elif self.ctype == "monod_ctype_limit":
            self.ref_value = map_units(self.ref_value, self.mo.c_unit)
            ph = Monod(
                name="_PMonod",
                reservoir=self.ref_reservoirs,
                flux=self.fh,
                register=self.fh,
                ref_value=self.ref_value,
                a_value=self.a_value,
                b_value=self.b_value,
            )

        else:
            raise ValueError(
                f"This should not happen,and points to a keywords problem in {self.name}"
            )

        # print(f"adding {ph.name} to {self.name}")
        self.lop.append(ph)

    def info(self, **kwargs) -> None:
        """Show an overview of the object properties.
        Optional arguments are
        index  :int = 0 this will show data at the given index
        indent :int = 0 indentation

        """
        off: str = "  "
        if "index" not in kwargs:
            index = 0
        else:
            index = kwargs["index"]

        if "indent" not in kwargs:
            indent = 0
            ind = ""
        else:
            indent = kwargs["indent"]
            ind = " " * indent

        # print basic data bout this Connection
        print(f"{ind}{self.__str__(kwargs)}")

        print(f"{ind}Fluxes:")
        for f in sorted(self.lof):
            f.info(indent=indent, index=index)

    def __delete_process__(self) -> None:
        """Updates to the connection properties may change the connection type and thus
        the processes which are associated with this connection. We thus have to
        first delete the old processes, before we re-initialize the connection

        """

        # identify which processes we need to delete
        # unregister process from connection.lop, reservoir.lop, flux.lop, model.lmo
        # delete process from global name space if present

        lop = cp.copy(self.lop)

        for p in lop:
            for f in self.lof:
                if isinstance(f.register, ConnectionGroup):
                    # remove from Connection group list of model objects
                    self.register.lmo.remove(f)
                else:
                    self.r1.lop.remove(p)
                    self.fh.lop.remove(p)
                    self.lop.remove(p)
                    self.r1.mo.lmo.remove(p.n)
                    del p

    def __delete_flux__(self) -> None:
        """Updates to the connection properties may change the connection type and thus
        the processes which are associated with this connection. We thus have to
        first delete the old flux, before we re-initialize the connection

        """

        # identify which processes we need to delete
        # unregister process from connection.lop, reservoir.lop, flux.lop, model.lmo
        # delete process from global name space if present

        lof = cp.copy(self.lof)
        for f in lof:
            if isinstance(f.register, ConnectionGroup):
                # remove from Connection group list of model objects
                self.register.lmo.remove(f)
            else:
                self.r1.lof.remove(f)
                self.lof.remove(f)
                self.r1.mo.lmo.remove(f.n)
                del f

    # ---- Property definitions to allow for connection updates --------
    """ Changing the below properties requires that we delete all
    associated objects (processes), and determines the new flux type,
    and initialize/register these with the connection and model.
    We also have to update the keyword arguments as these are used
    for the log entry

    """

    # ---- alpha ----
    @property
    def alpha(self) -> tp.Union[float, int]:
        return self._alpha

    @alpha.setter
    def alpha(self, a: tp.Union[float, int]) -> None:

        if self.update and a != "None":
            self.__delete_process__()
            self.__delete_flux__()
            self._alpha = a
            self.__set_process_type__()  # derive flux type and create flux(es)
            self.__register_process__()

    # ---- rate  ----
    @property
    def rate(self) -> tp.Union[float, int]:
        return self._rate

    @rate.setter
    def rate(self, r: str) -> None:
        from . import ureg, Q_

        if self.update and r != "None":
            self.__delete_process__()
            self.__delete_flux__()
            self._rate = Q_(r).to(self.model.f_unit).magnitude
            self.__create_flux__()  # Source/Sink/Regular
            self.__set_process_type__()  # derive flux type and create flux(es)
            self.__register_process__()

    # ---- delta  ----
    @property
    def delta(self) -> tp.Union[float, int]:
        return self._delta

    @delta.setter
    def delta(self, d: tp.Union[float, int]) -> None:

        if self.update and r != "None":
            self.__delete_process__()
            self.__delete_flux__()
            self._delta = d
            self.kwargs["delta"] = d
            self.__create_flux__()  # Source/Sink/Regular
            self.__set_process_type__()  # derive flux type and create flux(es)
            self.__register_process__()


class Connection(Connect):
    """Alias for the Connect class"""


class ConnectionGroup(esbmtkBase):
    """ConnectionGroup

        Connect reservoir/sink/source groups when at least one of the
        arguments is a reservoirs_group object. This method will
        create regular connections for each matching species.

        Use the connection.update() method to fine tune connections
        after creation

    Example::

        ConnectionGroup(source =  upstream reservoir / upstream reservoir group
           sink = downstrean reservoir / downstream reservoirs_group
           delta = defaults to zero and has to be set manually
           alpha =  defaults to zero and has to be set manually
           rate = shared between all connections
           ref_reservoirs = shared between all connections
           ref_flux = shared between all connections
           species = list, optional, if present, only these species will be connected
           ctype = needs to be set for all connections. Use "Regular"
                   unless you require a specific connection type
           pl = [list]) process list. optional, shared between all connections
           id = optional identifier, passed on to individual connection
           plot = "yes/no" # defaults to yes, shared between all connections
           save_flux_data = True/False, use model default if not set
        )

        ConnectionGroup(
                  source=OM_Weathering,
                  sink=Ocean,
                  rate={DIC: f"{OM_w} Tmol/yr" ,
                        ALK: f"{0} Tmol/yr"},
                  ctype = {DIC: "Regular",
                           ALK: "Regular"},
    )


    """

    def __init__(self, **kwargs) -> None:

        self.defaults: Dict[str, any] = {
            "id": ["None", (str)],
            "source": ["None", (str, SourceGroup, Reservoir, ReservoirGroup)],
            "sink": ["None", (str, SinkGroup, Reservoir, ReservoirGroup)],
            "delta": ["None", (str, dict, tuple, int, float)],
            "rate": ["None", (Q_, str, dict, tuple, int, float)],
            "pl": ["None", (str, dict, tuple)],
            "signal": ["None", (str, Signal)],
            "alpha": ["None", (str, dict, tuple, int, float)],
            "species": ["None", (str, dict, tuple, Species)],
            "ctype": ["None", (str, dict, tuple)],
            "ref_reservoirs": ["None", (str, dict, tuple, Reservoir)],
            "ref_flux": ["None", (str, dict, tuple, Flux)],
            "plot": ["yes", (str, dict, tuple)],
            "scale": [1, (str, dict, tuple, int, float)],
            "bypass": ["None", (dict, tuple, str)],
            "register": ["None", (str, tuple, Model)],
            "save_flux_data": [False, (bool, tuple)],
        }

        # provide a list of absolutely required keywords
        self.lrk: list = ["source", "sink", "register"]
        self.__initialize_keyword_variables__(kwargs)

        if self.save_flux_data == "None":
            self.save_flux_data = self.register.save_flux_data
            self.kwargs.update({"save_flux_data": self.register.save_flux_data})

        # # self.source.lor is a  list with the object names in the group
        self.mo = self.sink.lor[0].mo
        self.model = self.mo
        self.loc: list = []  # list of connection objects

        self.name = f"CG_{self.source.name}_to_{self.sink.name}"
        self.base_name = self.name
        self.parent = self.register
        self.__register_name_new__()
        self.__create_connections__()

    def add_connections(self, **kwargs) -> None:
        """Add connections to the connection group"""

        self.__initialize_keyword_variables__(kwargs)
        self.__create_connections__()

    def __create_connections__(self) -> None:
        """Create Connections"""

        # find all sub reservoirs which have been specified by the ctype keyword
        self.connections: list = []
        for r, t in self.ctype.items():
            if t == "None":
                raise ValueError(
                    f"Connectiongroup {self.name} must specify 'ctype'. See help(Connectiongroup)"
                )
            self.connections.append(r)

        # now we need to create defaults for all connections
        self.cd: dict = {}  # connection dictionary
        # loop over speces
        for sp in self.connections:  # ["SO4", "H2S"]
            self.cd[sp.n] = {
                # "cid": self.id,
                "cid": "None",
                "plot": "yes",
                "delta": "None",
                "alpha": "None",
                "rate": "None",
                "scale": "None",
                "ctype": "None",
                "ref_reservoirs": "None",
                "ref_flux": "None",
                "bypass": "None",
            }

            # test defaults against actual keyword value
            for kcd, vcd in self.cd[sp.n].items():
                if kcd in self.kwargs:  # found entry like ctype
                    if r in self.kwargs[kcd]:  # {SO4: xxx}
                        # update the entry
                        self.cd[sp.n][kcd] = self.kwargs[kcd][sp]

            a = Connect(
                source=getattr(self.source, sp.n),
                sink=getattr(self.sink, sp.n),
                rate=self.cd[sp.n]["rate"],
                delta=self.cd[sp.n]["delta"],
                alpha=self.cd[sp.n]["alpha"],
                plot=self.cd[sp.n]["plot"],
                ctype=self.cd[sp.n]["ctype"],
                scale=self.cd[sp.n]["scale"],
                bypass=self.cd[sp.n]["bypass"],
                ref_reservoirs=self.cd[sp.n]["ref_reservoirs"],
                ref_flux=self.cd[sp.n]["ref_flux"],
                save_flux_data=self.save_flux_data,
                groupname=True,
                id=self.id,
                register=self,
            )

            ## add connection to list of connections
            self.loc.append(a)
            if self.mo.debug:
                print(
                    f"created connection with full name {a.full_name}, registered to {self.name} "
                    f"fn = {self.full_name}"
                )

    def info(self) -> None:
        """List all connections in this group"""

        print(f"Group Connection from {self.source.name} to {self.sink.name}\n")
        print("The following Connections are part of this group\n")
        print(f"You can query the details of each connection like this:\n")
        for c in self.loc:
            print(f"{c.name}: {self.name}.{c.name}.info()")

        print("")


class AirSeaExchange(esbmtkBase):
    """The class creates a connection between liquid reservoir (i.e., an
    ocean), and a gas reservoir (i.e., the atmosphere).

    Example :
    ~~~~~~~~

    AirSeaExchange(
        gas_reservoir= must be a gasreservoir
        liquid_reservoir = must be a reservoir
        solubility= as returned by the swc object
        area = Ocean.area, [m^2]
        piston_velocity = 4.8*365, [m/yr]
        id = str, optional
        water_vapor_pressure=Ocean.swc.p_H2O,
        ref_quantity = optional
        register = Model
        )

    In some cases the gas flux does not depend on the main reservoir species
    but on a derived quantity, e.g., [CO2aq]. Specify the ref_quantity
    keyword to point to a different species/calculated species.

    """

    def __init__(self, **kwargs) -> None:
        """initialize instance"""

        from .utility_functions import check_for_quantity

        self.defaults: dict[str, list[str, tuple]] = {
            "gas_reservoir": ["None", (str, GasReservoir)],
            "liquid_reservoir": ["None", (str, Reservoir)],
            "solubility": ["None", (str, float)],
            "piston_velocity": ["None", (str, Q_)],
            "area": [0.0, (float)],
            "id": ["None", (str)],
            "name": ["None", (str)],
            "water_vapor_pressure": [0, (int, float, np.float64)],
            "ref_species": ["None", (Reservoir, int, float, np.float64, np.ndarray)],
            "species": ["None", (Species, str)],
            "register": ["None", (str, Model)],
        }
        # provide a list of absolutely required keywords
        self.lrk: list[str] = [
            "gas_reservoir",
            "liquid_reservoir",
            "solubility",
            "piston_velocity",
            "area",
            "water_vapor_pressure",
            "species",
            "register",
        ]
        self.__initialize_keyword_variables__(kwargs)

        self.__misc_inits__()

        self.lof: list = []
        self.scale = self.area * self.piston_velocity

        # create connection and flux name
        self.name = f"C_{self.lr.register.name}_to_{self.gr.name}"
        self.parent = self.register
        self.__register_name_new__()

        # initalize a flux instance
        self.fh = Flux(
            species=self.species,  # Species handle
            delta=0,  # delta value of flux
            rate="0 mol/a",  # flux value
            register=self,  # register with this connection
            isotopes=self.isotopes,
            id=self.id,
        )
        # register flux with liquid reservoir
        self.lr.lof.append(self.fh)
        self.lr.lio[self.fh] = 1  # flux direction

        # register flux with gas reservoir
        self.gr.lof.append(self.fh)
        self.gr.lio[self.fh] = -1  # flux direction
        # register with connection
        self.lof.append(self.fh)

        if self.lr.register == "None":
            swc = self.lr.swc
        else:
            swc = self.lr.register.swc

        # initialize process instance
        ph = GasExchange(
            name="_PGex",
            gas=self.gr,  # gas reservoir
            liquid=self.lr,  # reservoir
            ref_species=self.ref_species,  # concentration
            flux=self.fh,  # flux handle
            register=self.fh,
            scale=self.scale,
            solubility=self.solubility,
            water_vapor_pressure=self.water_vapor_pressure,
            seawaterconstants=swc,
            isotopes=True,
        )

        # register process with reservoir
        ph.__register__(self.lr, self.fh)
        # register connector with liquid reservoirgroup
        # spr = getattr(self.lr, self.lr.name)
        self.lr.loc.add(self)
        # register connector with gas reservoir
        self.gr.loc.add(self)
        # register connector with model
        self.mo.loc.add(self)

    def __misc_inits__(self) -> None:
        """Bits and pices of house keeping"""

        # make sure piston velocity is in the right units
        self.piston_velocity = check_for_quantity(self.piston_velocity)
        self.piston_velocity = self.piston_velocity.to(
            f"meter/{self.liquid_reservoir.mo.t_unit}"
        ).magnitude

        # ref_species can point to vr_data fields which are of type
        # numpy array
        if isinstance(self.ref_species, np.ndarray):
            testv = self.ref_species[0]
        else:
            testv = self.ref_species

        if testv == "None":
            self.ref_species = self.species

        self.lr = self.liquid_reservoir
        self.gr = self.gas_reservoir

        if self.species.name in ["CO2", "DIC", "O2"]:
            pass
        else:
            raise ValueError(f"{self.species.name} not implemented yet")

        # decide if this connection needs isotope calculations
        if self.gas_reservoir.isotopes:
            self.isotopes = True
        else:
            self.isotopes = False

        self.mo = self.species.mo
        self.model = self.mo
