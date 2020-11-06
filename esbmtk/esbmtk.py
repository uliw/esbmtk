"""
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
from numbers import Number
from nptyping import *
from typing import *
from numpy import array, set_printoptions, arange, zeros, interp, mean
from pandas import DataFrame
from copy import deepcopy, copy
from time import process_time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import logging
import time
import builtins
set_printoptions(precision=4)
# from .utility_functions import *
from .base_class import esbmtkBase

class Model(esbmtkBase):
    """This is the class specify a new model.

    Example:

            esbmtkModel(name   =  "Test_Model",
                      start    = "0 yrs",    # optional: start time 
                      stop     = "1000 yrs", # end time
                      timestep = "2 yrs",    # as a string "2 yrs"
                      ref_time = "0 yrs",    # optional: time offset for plot
                      mass_unit = "mol/l",   #required
                      volume_unit = "mol/l", #required 
            )

    The 'ref_time' keyword will offset the time axis by the specified
    amount, when plotting the data, .i.e., the model time runs from to
    100, but you want to plot data as if where from 2000 to 2100, you would
    specify a value of 2000. This is for display purposes only, and does not affect
    the model. Care must be taken that any external data references the model
    time domain, and not the display time.
    
    All of the above keyword values are available as variables with 
    Model_Name.keyword

    The user facing methods of the model class are
       - Model_Name.describe()
       - Model_Name.save_data()
       - Model_Name.plot_data()
       - Model_Name.plot_reservoirs()
       - Model_Name.run()

    Optional, you can provide the element keyword which will setup a
    default set of Species for Carbon and Sulfur.  In this case, there
    is no need to define elements or species. The argument to this
    keyword are either "Carbon", or "Sulfur" or both as a list
    ["Carbon", "Sulfur"].

    """

    from typing import Dict
    from . import ureg, Q_
    def __init__(self, **kwargs: Dict[any, any]) -> None:
        """ initialize object"""

        from . import ureg, Q_

        # provide a dict of all known keywords and their type
        self.lkk: Dict[str, any] = {
            "name": str,
            "start": str,
            "stop": str,
            "timestep": str,
            "ref_time": str,
            "element": str,
            "mass_unit": str,
            "volume_unit": str,
        }

        # provide a list of absolutely required keywords
        self.lrk: list[str] = ["name", "stop", "timestep", "mass_unit", "volume_unit"]

        # list of default values if none provided
        self.lod: Dict[str, any] = {
            'start': "0 years",
            'ref_time': "0 years",
        }

        self.__initerrormessages__()
        self.bem.update({
            "ref_time": "a string",
            "timesetp": "a string",
            "element": "element name",
            "mass_unit": "a string",
            "volume_unit": "a string",
        })

        self.__validateandregister__(kwargs)  # initialize keyword values

        # empty list which will hold all reservoir references
        self.lor: list = []
        # empty list which will hold all connector references
        self.loc: list = []
        self.lel: list = []  # list which will hold all element references
        self.lsp: list = []  # list which will hold all species references
        self.lop: list = []  # list flux processe
        self.lmo: list = []  # list of all model objects
        self.olkk: list = [
        ]  # optional keywords for use in the connector class

        # Parse the strings which contain unit information and convert
        # into model base units For this we setup 3 variables which define
        self.t_unit = Q_(self.timestep).units # the time unit
        self.d_unit = Q_(self.stop).units # display time units
        self.m_unit = Q_(self.mass_unit) # the mass unit
        self.v_unit = Q_(self.volume_unit) # the volume unit
        self.c_unit = self.m_unit / self.v_unit # the concentration unit (mass/volume)
        self.f_unit = self.m_unit / self.t_unit # the flux unit (mass/time)
        ureg.define('Sverdrup = 1e6 * meter **3 / second = Sv = Sverdrups')

        # legacy variable names
        self.start =  Q_(self.start).to(self.t_unit).magnitude
        self.stop =  Q_(self.stop).to(self.t_unit).magnitude
        self.ref_time = Q_(self.ref_time).to(self.t_unit).magnitude

        self.bu = self.t_unit
        self.base_unit = self.t_unit
        self.dt =  Q_(self.timestep).magnitude
        self.tu = str(self.bu)  # needs to be a string
        self.n = self.name
        self.mo = self.name

        self.xl = f"Time [{self.bu}]"  # time axis label
        self.length = int(abs(self.stop - self.start))
        self.steps = int(abs(round(self.length / self.dt)))
        self.time = ((arange(self.steps) * self.dt) + self.start)
        
        if "element" in self.kwargs:
            if isinstance(self.kwargs["element"], list):
                element_list = self.kwargs["element"]
            else:
                element_list = [self.kwargs["element"]]

            for e in element_list:

                if e == "Carbon":
                    Carbon(model=self, name=self.mo + "_Carbon")
                elif e == "Sulfur":
                    Sulfur(model=self, name=self.mo + "_Sulfur")
                else:
                    raise ValueError(f"{e} not implemented yet")

    def describe(self) -> None:
        """ Describe Basic Model Parameters and log them
        
        """


        logging.info("---------- Model description start ----------")
        logging.info(f"Model Name = {self.n}")
        logging.info(f"Model time [{self.bu}], dt = {self.dt}")
        logging.info(
            f"start = {self.start}, stop={self.stop} [{self.tu}]")
        logging.info(f"Steps = {self.steps}\n")
        logging.info(f"  Species(s) in {self.n}:")

        for r in self.lor:
            r.describe()
            logging.info(" ")

            logging.info("---------- Model description end ------------\n")

            
    def list_species(self) -> None:
        """ List all species known to the model
        
        """
        for e in self.lel:
            print(f"Defined Species for {e.n}:")
            e.list_species()
            print("\n")

        print("Use the species class to add to this list")

    def save_data(self) -> None:
        """Save the model results to a CSV file. Each reservoir will have
        their own CSV file
        """
        for r in self.lor:
            r.write_data()

    def plot_data(self) -> None:
        """ 
        Loop over all reservoirs and either plot the data into a 
        window, or save it to a pdf
        """

        i = 0
        for r in self.lor:
            r.__plot__(i)
            i = i + 1

        plt.show()  # create the plot windows

    def plot_reservoirs(self) -> None:
        """Loop over all reservoirs and either plot the data into a window,
            or save it to a pdf
        """

        i = 0
        for r in self.lor:
            r.__plot_reservoirs__(i)
            i = i + 1

        plt.show()  # create the plot windows

    def run(self) -> None:
        """Loop over the time vector, and for each time step, calculate the
        fluxes for each reservoir
        """

       
        # this has nothing todo with self.time below!
        start: float = process_time()
        new: [NDArray, Float] = zeros(3) + 1

        i = self.execute(new, self.time, self.lor)

        duration: float = process_time() - start
        print(f"Execution took {duration} seconds")

    # some mumbo jumbbo to support numba optimization. Currently not working though
    @staticmethod # this provides significnt speedups
    def execute(new: [NDArray, Float], time: [NDArray, Float],
                lor: list) -> None:
        """ Moved this code into a separate function to enable numba optimization
        """

        i = 1  # some processes refer to the previous time step
        for t in time[0:-1]:  # loop over the time vector except the first
            # we first need to calculate all fluxes
            for r in lor:  # loop over all reservoirs
                for p in r.lop:  # loop over reservoir processes
                    p(r, i)  # update fluxes

            # and then update all reservoirs
            for r in lor:  # loop over all reservoirs
                flux_list = r.lof

                new[0] = new[1] = new[2] = 0
                for f in flux_list:  # do sum of fluxes in this reservoir
                    direction = r.lio[f.n]
                    new[0] = new[
                        0] + f.m[i] * direction  # current flux and direction
                    new[1] = new[
                        1] + f.l[i] * direction  # current flux and direction
                    new[2] = new[
                        2] + f.h[i] * direction  # current flux and direction

                #new = array([ms, ls, hs])
                new = new * r.mo.dt  # get flux / timestep
                new = new + r[i - 1]  # add to data from last time step
                new = new * (new > 0)  # set negative values to zero
                r[i] = new  # update reservoir data

            i = i + 1

    def __step_process__(self, r, i) -> None:
        """ For debugging. Provide reservoir and step number,
        """
        for p in r.lop:  # loop over reservoir processes
            print(f"{p.n}")
            p(r, i)  # update fluxes

    def __step_update_reservoir__(self, r, i) -> None:
        """ For debugging. Provide reservoir and step number,
        """
        flux_list = r.lof
        # new = sum_fluxes(flux_list,r,i) # integrate all fluxes in self.lof

        ms = ls = hs = 0
        for f in flux_list:  # do sum of fluxes in this reservoir
            direction = r.lio[f.n]
            ms = ms + f.m[i] * direction  # current flux and direction
            ls = ls + f.l[i] * direction  # current flux and direction
            hs = hs + f.h[i] * direction  # current flux and direction

        new = array([ms, ls, hs])
        new = new * r.mo.dt  # get flux / timestep
        new = new + r[i - 1]  # add to data from last time step
        new = new * (new > 0)  # set negative values to zero
        r[i] = new  # update reservoir data

class Element(esbmtkBase):
    """Each model, can have one or more elements.  This class sets
element specific properties
      
      Example:
        
            Element(name      = "S "           # the element name
                    model     = Test_model     # the model handle  
                    mass_unit =  "mol",        # base mass unit
                    li_label  =  "$^{32$S",    # Label of light isotope
                    hi_label  =  "$^{34}S",    # Label of heavy isotope
                    d_label   =  "$\delta^{34}$S",  # Label for delta value 
                    d_scale   =  "VCDT",       # Isotope scale
                    r         = 0.044162589,   # isotopic abundance ratio for element
                  
)  
      """

    # set element properties
    def __init__(self, **kwargs) -> any:
        """ Initialize all instance variables
        """

        # provide a dict of known keywords and types
        self.lkk = {
            "name": str,
            "model": Model,
            "mass_unit": str,
            "li_label": str,
            "hi_label": str,
            "d_label": str,
            "d_scale": str,
            "r": Number
        }

        # provide a list of absolutely required keywords
        self.lrk :list = ["name", "model", "mass_unit"]
        # list of default values if none provided
        self.lod = {
            'li_label': "NONE",
            'hi_label': "NONE",
            'd_label': "NONE",
            'd_scale': "NONE",
            'r': 1,
        }

        self.__initerrormessages__()
        self.__validateandregister__(kwargs)  # initialize keyword values

        # legacy name aliases
        self.n :str = self.name  # display name of species
        self.mu :str = self.mass_unit  # display name of mass unit
        self.ln :str = self.li_label  # display name of light isotope
        self.hn :str = self.hi_label  # display name of heavy isotope
        self.dn :str = self.d_label  # display string for delta
        self.ds :str = self.d_scale  # display string for delta scale
        self.mo :Model = self.model  # model handle
        self.lsp :list = [] # list of species for this element.
        self.mo.lel.append(self)

    def list_species(self) -> None:
        """ List all species which are predefined for this element
        
        """

        for e in self.lsp:
            print(e.n)

    def __lt__(self, other) -> None:  # this is needed for sorting with sorted()
        return self.n < other.n

class Species(esbmtkBase):
    """Each model, can have one or more species.  This class sets species
specific properties
      
      Example:
        
            Species(name = "SO4",
                    element = S,
)

    """

    # set species properties
    def __init__(self, **kwargs) -> None:
        """ Initialize all instance variables
            """
        
        # provide a list of all known keywords
        self.lkk :Dict[any,any] = {"name":str, "element":Element}

        # provide a list of absolutely required keywords
        self.lrk = ["name", "element"]

        # list of default values if none provided
        self.lod = {}

        self.__initerrormessages__()
        self.__validateandregister__(kwargs)  # initialize keyword values

        # legacy names
        self.n  = self.name        # display name of species
        self.mu = self.element.mu  # display name of mass unit
        self.ln = self.element.ln  # display name of light isotope
        self.hn = self.element.hn  # display name of heavy isotope
        self.dn = self.element.dn  # display string for delta
        self.ds = self.element.ds  # display string for delta scale
        self.r  = self.element.r   # ratio of isotope standard
        self.mo = self.element.mo  # model handle
        self.eh = self.element.n   # element name
        self.e  = self.element    # element handle

        #self.mo.lsp.append(self)   # register self on the list of model objects
        self.e.lsp.append(self) # register this species with the element 

    def __lt__(self, other) -> None:  # this is needed for sorting with sorted()
        return self.n < other.n

class Reservoir(esbmtkBase):
    """
      Tis object holds reservoir specific information. 

      Example:

              Reservoir(name = "IW_SO4",      # Name of reservoir
                        species = S,          # Species handle
                        delta = 20,           # initial delta - optional (defaults  to 0)
                        mass/concentration = "1 unit"  # species concentration or mass
                        volume = "1E5 l",      # reservoir volume (m^3) 
               )

      you must either give mass or concentration. The result will always be displayed as concentration

      You can access the reservoir data as
      - Name.m # mass
      - Name.d # delta
      - Name.c # concentration

    Useful methods include

      - Name.write_data() # dave data to file
      - Name.describe() # show data this takess an optional argument to show the nth dataset
      
    """
    def __init__(self, **kwargs) -> None:
        """ Initialize a reservoir.
        
        """

        from . import ureg, Q_

        # provide a dict of all known keywords and their type
        self.lkk: Dict[str, any] = {
            "name": str,
            "species": Species,
            "delta": Number,
            "concentration": str,
            "mass": str,
            "volume": str,
        }

        # provide a list of absolutely required keywords
        self.lrk: list = [
            "name", "species", "volume", ["mass", "concentration"]
        ]

        # list of default values if none provided
        self.lod: Dict[any, any] = {
            'delta': 0,
        }

        # validate and initialize instance variables
        self.__initerrormessages__()
        self.bem.update({
            "concentration": "string",
            "mass": "a  string",
            "concentration": "a string",
            "volume": "a string"
        })
        self.__validateandregister__(kwargs)

        # legacy names
        self.n: str = self.name  # name of reservoir
        self.sp: Species = self.species  # species handle
        self.mo: Model = self.species.mo  # model handle

        # convert units
        self.volume: Number = Q_(self.volume).to(self.mo.v_unit).magnitude
        self.v: Number = self.volume  # reservoir volume
        
        # This should probably be species specific?
        self.mu: str = self.sp.e.mass_unit  # massunit xxxx

        if "concentration" in kwargs:
            c = Q_(self.concentration)
            self.plt_units = c.units 
            self.concentration: Number = c.to(self.mo.c_unit).magnitude
            self.mass: Number = self.concentration * self.volume  # caculate mass
        elif "mass" in kwargs:
            m =  Q_(self.mass)
            self.plt_units = m.units 
            self.mass: Number = m.to(self.mo.m_unit).magnitude
            self.concentration = self.mass / self.volume
        else:
            raise ValueError("You need to specify mass or concentration")

        # save the unit which was provided by the user for display purposes
        
        
        self.lof: list[Flux] = []  #  flux references
        self.led: list[ExternalData] = []  # all external data references
        self.lio: dict[str, int] = {}  #  flux name:direction pairs
        self.lop: list[Process] = []  # list holding all processe references
        self.loe: list[Element] = []  # list of elements in thiis reservoir
        self.doe: Dict[Species, Flux] = {}  # species flux pairs

        # initialize mass vector
        self.m: [NDArray, Float[64]] = zeros(self.species.mo.steps) + self.mass
        # initialize concentration vector
        self.c: [NDArray, Float[64]] = self.m / self.v
        self.l: [NDArray, Float[64]] = zeros(self.mo.steps)
        self.h: [NDArray, Float[64]] = zeros(self.mo.steps)
        [self.l, self.h] = get_mass(self.m, self.delta,
                                    self.species.r)  # isotope mass
        self.d: [NDArray,
                 Float[64]] = get_delta(self.l, self.h,
                                        self.sp.r)  # delta of reservoir
        self.lm: str = f"{self.species.n} [{self.mu}/l]"  # left y-axis label
        self.ld: str = f"{self.species.dn} [{self.species.ds}]"  # right y-axis label
        self.xl: str = self.mo.xl  # set x-axis lable to model time

        self.legend_left = self.species.n
        self.legend_right = self.species.dn

        self.mo.lor.append(self)  # add this reservoir to the model

    def __call__(self) -> None:  # what to do when called as a function ()
        pass
        return self

    def __getitem__(self, i: int) -> NDArray[np.float64]:  
        """ Get flux data by index
        """

        return array([self.m[i], self.l[i], self.h[i]])

    def __setitem__(self, i: int,
                    value: float) -> None:  # howto write data by index
        self.m[i]: float = value[0]
        self.l[i]: float = value[1]
        self.h[i]: float = value[2]
        # update concentration and delta next. This is computationally inefficient
        # but the next time step may depend on on both variables.
        # update delta for this species
        #self.d = self.sp.getdelta(self.l, self.h)
        self.d[i]: float = get_delta(self.l[i], self.h[i], self.sp.r)
        self.c[i]: float = self.m[i] / self.v  # update concentration

    def log_description(self) -> None:

        o = 8 * " "
        logging.info(f"{o}{self.n}: Volume = {self.v:4E},\
            mass={self.m[1]},\
            concentration={self.c[1]}")
        logging.info(f"{o}    Initial d-value = {self.d[1]:.4f}")
        # loop over all reservoir objects
        o = 12 * " "
        if len(self.lop) > 0:
            logging.info(f"{o}Modifiers acting on fluxes in this reservoir:")
        for m in self.lop:
            m.describe(self)

    def write_data(self) -> None:
        """
            Write model data to CSV file. Each Reservoir gets its own file
            Files are named as 'Modelname_Reservoirname.csv'
            """
        # some short hands
        sn = self.sp.n  # species name
        sp = self.sp
        mo = self.sp.mo
       
        smu = f"{mo.m_unit:~P}"
        mtu = f"{mo.t_unit:~P}"
        fmu = f"{mo.f_unit:~P}"
        cmu = f"{mo.c_unit:~P}"
        
        sdn = self.sp.dn  # delta name
        sds = f"[{self.sp.ds}]"  # delta scale
        rn = self.n  # reservoir name
        mn = self.sp.mo.n  # model name
        fn = f"{mn}_{rn}.csv"  # file name

        # build the dataframe
        df: pd.dataframe = DataFrame()
        df[f"{self.n} {sn} [{smu}]"] = self.m    # mass
        df[f"{self.n} {sn} [{cmu}]"] = self.c    # concentration
        df[f"{self.n} {sp.ln}"] = self.l         # light isotope
        df[f"{self.n} {sp.hn} "] = self.h        # heavy isotope
        df[f"{self.n} {sdn} {sds}"] = self.d   # delta value

        for f in self.lof:  # Assemble the headers and data for the reservoir fluxes
            df[f"{f.n} {sn} [{fmu}]"] = f.m
            df[f"{f.n} {sn} [{sp.ln}]"] = f.l
            df[f"{f.n} {sn} [{sp.hn}]"] = f.h
            df[f"{f.n} {sn} {sdn} {sds}"] = f.d

        df.to_csv(fn)  # Write dataframe to file
        return df

    def __plot__(self, i: int) -> None:
        """ 
            Plot data from reservoirs and fluxes into a multiplot window
            """

        model = self.sp.mo
        species = self.sp
        obj = self
        #time = model.time + model.ref_time  # get the model time
        #xl = f"Time [{model.bu}]"

        size, geo = get_plot_layout(self)  # adjust layout
        filename = f"{model.n}_{self.n}.pdf"
        fn = 1  # counter for the figure number

        fig = plt.figure(i)  # Initialize a plot window
        fig.canvas.set_window_title(f"Reservoir Name: {self.n}")
        fig.set_size_inches(size)

        # plot reservoir data
        plot_object_data(geo, fn, self)

        # plot the fluxes assoiated with this reservoir
        for f in sorted(self.lof):  # plot flux data
            fn = fn + 1
            plot_object_data(geo, fn, f)

        fig.suptitle(f"Model: {model.n}, Reservoir: {self.n}\n", size=16)
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        fig.savefig(filename)

    def __plot_reservoirs__(self, i: int) -> None:
        """ 
            Plot only the  reservoirs data, and ignore the fluxes
            """

        model = self.sp.mo
        species = self.sp
        obj = self
        time = model.time + model.ref_time  # get the model time
        xl = f"Time [{model.bu}]"

        size = [5, 3]
        geo = [1, 1]
        filename = f"{model.n}_{self.n}.pdf"
        fn = 1  # counter for the figure number

        fig = plt.figure(i)  # Initialize a plot window
        fig.set_size_inches(size)

        # plt.legend()ot reservoir data
        plot_object_data(geo, fn, self.c, self.d, self, time)

        fig.tight_layout()
        # fig.subplots_adjust(top=0.88)
        fig.savefig(filename)

    def __lt__(self, other) -> None:
        """ This is needed for sorting with sorted()
            """
        return self.n < other.n

    def describe(self, i=0) -> None:
        """ Show an overview of the object properties"""
        list_fluxes(self, self.n, i)
        print("\n")
        show_data(self, self.n, i)

    def __list_processes__(self) -> None:
        """ List all processes associated with this reservoir"""
        for p in self.lop:
            print(f"{p.n}")

class Flux(esbmtkBase):
    """A class which defines a flux object. Flux objects contain
      information which links them to an species, describe things like
      the mass and time unit, and store data of the total flux rate at
      any given time step. Similarly, they store the flux of the light
      and heavy isotope flux, as well as the delta of the flux. This
      is typically handled through the Connect object. If you set it up manually
      
      Flux = (name = "Name"
              species = species_handle,
              delta = any number,
              rate  = "12 mol/s" # must be a string
      )

       You can access the flux data as
      - Name.m # mass
      - Name.d # delta
      - Name.c # concentration
      
      """

    def __init__(self, **kwargs: Dict[str, any]) -> None:
        """
          Initialize a flux. Arguments are the species name the flux rate
          (mol/year), the delta value and unit
          """

        from . import ureg, Q_
        
        # provide a dict of all known keywords and their type
        self.lkk: Dict[str, any] = {
            "name": str,
            "species": Species,
            "delta": Number,
            "rate": str,
        }

        # provide a list of absolutely required keywords
        self.lrk: list = ["name", "species", "rate"]

        # list of default values if none provided
        self.lod: Dict[any, any] = {'delta': 0}

        # initialize instance
        self.__initerrormessages__()
        self.bem.update({"rate": "a string"})
        self.__validateandregister__(kwargs)  # initialize keyword values

        # legacy names
        self.n: str = self.name  # name of flux
        self.sp: Species = self.species  # species name
        self.mo: Model = self.species.mo  # model name
        self.model: Model = self.species.mo  # model handle

        # model units
        self.plt_units =  Q_(self.rate).units
        self.mu :str = f"{self.species.mu}/{self.mo.tu}"
        
        # and convert flux into model units
        fluxrate :float =  Q_(self.rate).to(self.mo.f_unit).magnitude
        
        self.m: [NDArray, Float[64]
                 ] = zeros(self.model.steps) + fluxrate  # add the flux
        self.l: [NDArray, Float[64]] = zeros(self.model.steps)
        self.h: [NDArray, Float[64]] = zeros(self.model.steps)
        [self.l, self.h] = get_mass(self.m, self.delta, self.species.r)
        if self.delta == 0:
            self.d: [NDArray, Float[64]] = zeros(self.model.steps)
        else:
            self.d: [NDArray, Float[64]] = get_delta(self.l, self.h,
                                                     self.sp.r)  # update delta
        self.lm: str = f"{self.species.n} [{self.mu}]"  # left y-axis a label
        self.ld: str = f"{self.species.dn} [{self.species.ds}]"  # right y-axis a label
        self.legend_left = self.species.n
        self.legend_right = self.species.dn
        
        self.xl: str = self.model.xl  # se x-axis label equal to model time
        self.lop: list[Process] = []  # list of processes
        self.led: list[ExternalData] = []  # list of ext data
        #self.t = 0        # time dependent flux = 1, otherwise 0
        self.source: str = ""  # Name of reservoir which acts as flux source
        self.sink: str = ""  # Name of reservoir which acts as flux sink

    def __getitem__(
            self,
            i: int) -> NDArray[np.float64]:  # howto get data by index [i]

        return array([self.m[i], self.l[i], self.h[i]])

    def __setitem__(self, i: int,
                    value: float) -> None:  # howto write data by index
        self.m[i] = value[0]
        self.l[i] = value[1]
        self.h[i] = value[2]
        self.d[i] = get_delta(self.l[i], self.h[i], self.sp.r)  # update delta

    def __call__(self) -> None:  # what to do when called as a function ()
        pass
        return self

    def log_description(self, reservoir) -> None:

        o = 16 * " "
        logging.info(
            f"{o}{self.n}, Flux = {self.m[1]*self.reservoir.lio[self.n]}, delta = {self.d[1]:.4f}"
        )

        o = 20 * " "
        if len(self.lop) > 0:
            logging.info(f"{o}Associated Perturbations:")
            for p in self.lop:  # loop over all perturbations objects
                p.describe()

    def describe(self, i: int) -> None:
        """ Show an overview of the object properties"""
        show_data(self, self.n, i)

    def __lt__(self, other):  # this is needed for sorting with sorted()
        return self.n < other.n

    def plot(self) -> None:
        """Plot the flux data """

        fig, ax1 = plt.subplots()
        fig.set_size_inches(5, 4)  # Set figure size in inches
        fig.set_dpi(100)  # Set resolution in dots per inch

        ax1.plot(self.mo.time, self.m, c="C0")
        ax2 = ax1.twinx()  # get second y-axis
        ax2.plot(self.mo.time, self.d, c="C1", label=self.n)

        ax1.set_title(self.n)
        ax1.set_xlabel(f"Time [{self.mo.tu}]")  #
        ax1.set_ylabel(f"{self.sp.n} [{self.sp.mu}]")
        ax2.set_ylabel(f"{self.sp.dn} [{self.sp.ds}]")
        ax1.spines['top'].set_visible(False)  # remove unnecessary frame
        ax2.spines['top'].set_visible(False)  # remove unnecessary frame

        fig.tight_layout()
        plt.show()
        plt.savefig(self.n + ".pdf")

class SourceSink(esbmtkBase):
    """
    This is just a meta calls to setup a Source/Sink object. These are not 
    actual reservoirs, but we stil need to have them as objects
    Example:
    
           Sink(name = "Pyrite",species = SO4)

    where the first argument is a string, and the second is a reservoir handle
    """

    def __init__(self, **kwargs) -> None:


        # provide a dict of all known keywords and their type
        self.lkk: Dict[str, any] = {
            "name": str,
            "species": Species,
        }

        # provide a list of absolutely required keywords
        self.lrk: list[str] = ["name", "species"]
        # list of default values if none provided
        self.lod: Dict[str, any] = {}

        self.__initerrormessages__()
        self.__validateandregister__(kwargs)  # initialize keyword values

        # legacy names
        self.n = self.name
        self.sp = self.species
        self.u = self.species.mu + "/" + str(self.species.mo.bu)


class Sink(SourceSink):
    """
    This is just a wrapper to setup a Sink object
    Example:
    
           Sink(name = "Pyrite",species =SO4)

    where the first argument is a string, and the second is a species handle
    """


class Source(SourceSink):
    """
    This is just a wrapper to setup a Source object
    Example:
    
           Sink(name = "SO4_diffusion", species ="SO4")

    where the first argument is a string, and the second is a species handle
    """

class Signal(esbmtkBase):
    """We use a simple generator which will create a signal which is
      described by its startime (relative to the model time), it's
      size (as mass) and duration, or as duration and
      magnitude. Furthermore, we can presribe the signal shape
      (square, pyramid) and whether the signal will repeat. You
      can also specify whether the event will affect the delta value.

      The data in the signal class will simply be added to the data in
      a given flux. So this class cannot be used for scaling (can we
      add this functionality?)
  
      Example:

            Signal(name = "Name",
                   species = Species handle,
                   start = "0 yrs",     # optional
                   duration = "0 yrs",  #
                   delta = 0,           # optional
                   stype = "addition"   # optional, currently the only type
                   shape = "square"     # square, pyramid
                   mass/magnitude/filename  # give one
                  )

      Signals are cumulative, i.e., complex signals ar created by
      adding one signal to another (i.e., Snew = S1 + S2) 

      Signals are registered with a flux during flux creation,
      i.e., they are passed on the process list when calling the
      connector object.
    
      if the filename argument is used, you can provide a filename which
      contains the data to be used in csv format. The data will be
      interpolated to the model domain, and added to the already existing data.
      The external data need to be in the following format

        Time, Rate, delta value
        0,     10,   12

        i.e., the first row needs to be a header line


      This class has the following methods

        Signal.repeat()
        Signal.plot()
        Signal.describe()
    """

    def __init__(self, **kwargs) -> None:
        """ Parse and initialize variables
        
        """
        
        from . import ureg, Q_
        
        # provide a list of all known keywords and their type
        self.lkk: Dict[str, any] = {
            "name": str,
            "start": str,
            "duration": str,
            "species": Species,
            "delta": Number,
            "stype": str,
            "shape": str,
            "filename": str,
            "mass": str,
            "magnitude": Number
        }

        # provide a list of absolutely required keywords
        self.lrk: List[str] = [
            "name", "duration", "species", ["shape", "filename"],
            ["magnitude", "mass", "filename"]
        ]

        # list of default values if none provided
        self.lod: Dict[str, any] = {
            'start': "0 yrs",
            'stype': "addition",
            'shape': "external_data",
            'duration': "0 yrs",
            'delta': 0,
        }

        self.__initerrormessages__()
        self.bem.update({"data": "a string", "magnitude": Number})
        self.__validateandregister__(kwargs)  # initialize keyword values

        # list of signals we are based on.
        self.los: List[Signal] = []
        
        # convert units to model units
        self.st: Number = Q_(self.start).to(self.species.mo.t_unit).magnitude  # start time
        self.l: Number = Q_(self.duration).to(self.species.mo.t_unit).magnitude  # the duration

        if "mass" in self.kwargs:
            self.mass = Q_(self.mass).to_self.species.mo.m_unit.magnitude
        elif "magnitude" in self.kwargs:
            self.magnitude = Q_(self.magnitude).to_self.species.mo.f_unit.magnitude

        # legacy name definitions
        self.n: str = self.name  # the name of the this signal
        self.sp: Species = self.species  # the species
        self.mo: Model = self.species.mo  # the model handle
        self.ty: str = self.stype  # type of signal
        self.sh: str = self.shape  # shape the event
        self.d: float = self.delta  # delta value offset during the event
        self.kwd: Dict[str, any] = self.kwargs  # list of keywords

        # initialize signal data
        self.data = self.__init_signal_data__()
        self.data.n: str = self.name + "_data"  # update the name of the signal data
        # update isotope values
        self.data.li, self.data.hi = get_mass(self.data.m, self.data.d,
                                              self.sp.r)

    def __init_signal_data__(self) -> None:
        """ Create an empty flux and apply the shape
            """
        # create a dummy flux we can act up
        self.nf: Flux = Flux(name=self.n + "_data",
                             species=self.sp,
                             rate="0 mol/yr",
                             delta=0)

        # since the flux is zero, the delta value will be undefined. So we set it explicitly
        # this will avoid having additions with Nan values.
        self.nf.d[0:]: float = 0.0

        # find nearest index for start, and end point
        self.si: int = int(round(self.st / self.mo.dt))  # starting index
        self.ei: int = self.si + int(round(self.l / self.mo.dt))  # end index
        print(f"ei = {self.ei} l = {self.l}")
        # create slice of flux vector
        self.s_m: [NDArray, Float[64]] = array(self.nf.m[self.si:self.ei])
        # create slice of delta vector
        self.s_d: [NDArray, Float[64]] = array(self.nf.d[self.si:self.ei])

        if self.sh == "square":
            self.__square__(self.si, self.ei)

        elif self.sh == "pyramid":
            self.__pyramid__(self.si, self.ei)

        elif "filename" in self.kwargs:  # use an external data set
            self.__int_ext_data__(self.si, self.ei)

        else:
            raise ValueError(f"argument needs to be either square/pyramid, "
                             f"or an ExternalData object. "
                             f"shape = {self.sh} is not a valid Value")

        # now add the signal into the flux slice
        self.nf.m[self.si:self.ei] = self.s_m
        self.nf.d[self.si:self.ei] = self.s_d

        return self.nf

    def __square__(self, s, e) -> None:
        """ Create Square Signal """

        w: float = (e - s) * self.mo.dt  # get the base of the square

        if "mass" in self.kwd:
            h = self.mass / w  # get the height of the square
        elif "magnitude" in self.kwd:
            h = self.magnitude
        else:
            raise ValueError(
                "You must specify mass or magnitude of the signal")

        self.s_m: float = h  # add this to the section
        self.s_d: float = self.d  # add the delta offset

    def __pyramid__(self, s, e) -> None:
        """ Create pyramid type Signal """

        w: float = (s - 1) * self.mo.dt  # get the base of the pyramid

        if "mass" in self.kwd:
            h = 2 * self.mass / w  # get the height of the pyramid
            print("mass")
        elif "magnitude" in self.kwd:
            h = self.magnitude
        else:
            raise ValueError(
                "You must specify mass or magnitude of the signal")

        print(f"\n pyramid h = {h} \n")
        # create pyramid
        c: int = int(round((e - s) / 2))  # get the center index for the peak
        x: [NDArray, Float[64]] = array([0, c,
                                         e - s])  # setup the x coordinates
        y: [NDArray, Float[64]] = array([0, h, 0])  # setup the y coordinates
        d: [NDArray, Float[64]] = array([0, self.d,
                                         0])  # setup the d coordinates
        xi = arange(0, e - s)  # setup the points at which to interpolate
        h: [NDArray, Float[64]] = interp(xi, x, y)  # interpolate flux
        dy: [NDArray, Float[64]] = interp(xi, x, d)  # interpolate delta
        self.s_m: [NDArray,
                   Float[64]] = self.s_m + h  # add this to the section
        self.s_d: [NDArray, Float[64]] = self.s_d + dy  # ditto for delta

    def __int_ext_data__(self, s, e) -> None:
        """ Interpolate External data as a signal. Unlike the other signals,
        thiw will replace the values in the flux with those read from the
        external data source. The external data need to be in the following format

        Time [units], Rate [units], delta value [units]
        0,     10,   12

        i.e., the first row needs to be a header line
        
        """

        from . import ureg, Q_
        
        # read external dataset
        df = pd.read_csv(self.filename)

        # get unit information from each header
        xh = df.columns[0].split("[")[1].split("]")[0] 
        yh = df.columns[1].split("[")[1].split("]")[0]
        # zh = df.iloc[0,2].split("[")[1].split("]")[0]

        # create the associated quantities
        xq = Q_(xh)
        yq = Q_(yh)
        # zq = Q_(zh)

        # add these to the data we are are reading
        x = df.iloc[:, 0].to_numpy() * xq  
        y = df.iloc[:, 1].to_numpy() * yq
        d = df.iloc[:, 2].to_numpy() 

        # map into model units, and strip unit information
        x = x.to(self.mo.t_unit).magnitude
        y = y.to(self.mo.f_unit).magnitude

        self.st: float = x[0]  # set the start time
        l : float = int(x[-1] - x[0])  # calculate the length
        self.si: int = int(round(self.st / self.mo.dt))  # starting index
        self.ei: int = s + int(round(l / self.mo.dt))  # endf index

        self.s_m: [NDArray, Float[64]] = array(
            self.nf.m[self.si:self.ei])  # create slice of flux vector
        self.s_d: [NDArray, Float[64]] = array(
            self.nf.d[self.si:self.ei])  # create slice of delta vector

        xi = arange(0, e - s)  # setup the points at which to interpolate
        h: [NDArray, Float[64]] = interp(xi, x, y)  # interpolate flux
        dy: [NDArray, Float[64]] = interp(xi, x, d)  # interpolate delta
        #self.m :float = sum((s_m * h - s_m)) * model.dt  # calculate mass of excursion
        self.s_m: [NDArray,
                   Float[64]] = self.s_m + h  # add this to the section
        self.s_d: [NDArray, Float[64]] = self.s_d + dy  # ditto for delta
        print(f"length off s_m ={len(self.s_m)}")

    def __add__(self, other):
        """ allow the addition of two signals and return a new signal"""

        ns = deepcopy(self)

        # add the data of both fluxes
        ns.data.m: [NDArray, Float[64]] = self.data.m + other.data.m
        ns.data.d: [NDArray, Float[64]] = self.data.d + other.data.d
        ns.data.l: [NDArray, Float[64]]
        ns.data.h: [NDArray, Float[64]]

        [ns.data.l, ns.data.h] = get_mass(ns.data.m, ns.data.d, ns.data.sp.r)

        ns.n: str = self.n + "_and_" + other.n
        print(f"adding {self.n} to {other.n}, returning {ns.n}")
        ns.data.n: str = self.n + "_and_" + other.n + "_data"
        ns.st = min(self.st, other.st)
        ns.l = max(self.l, other.l)
        ns.sh = "compound"
        ns.los.append(self)
        ns.los.append(other)

        return ns

    def repeat(self, start, stop, offset, times) -> None:
        """ This method creates a new signal by repeating an existing signal.
        Example:
      
        new_signal = signal.repeat(start,   # start time of signal slice to be repeated
                                   stop,    # end time of signal slice to be repeated
                                   offset,  # offset between repetitions 
                                   times,   # number of time to repeat the slice
                              )
      """

        ns: Signal = deepcopy(self)
        ns.n: str = self.n + f"_repeated_{times}_times"
        ns.data.n: str = self.n + f"_repeated_{times}_times_data"
        start: int = int(start / self.mo.dt)  # convert from time to index
        stop: int = int(stop / self.mo.dt)
        offset: int = int(offset / self.mo.dt)
        ns.start: float = start
        ns.stop: float = stop
        ns.offset: float = stop - start + offset
        ns.times: float = times
        ns.ms: [NDArray, Float[64]
                ] = self.data.m[start:stop]  # get the data slice we are using
        ns.ds: [NDArray, Float[64]] = self.data.d[start:stop]

        diff = 0
        for i in range(times):
            start: int = start + ns.offset
            stop: int = stop + ns.offset
            if start > len(self.data.m):
                break
            elif stop > len(self.data.m):  # end index larger than data size
                diff: int = stop - len(self.data.m)  # difference
                stop: int = stop - diff  # new end index
                lds: int = len(ns.ds) - diff
            else:
                lds: int = len(ns.ds)

            ns.data.m[start:stop]: [NDArray, Float[64]
                                    ] = ns.data.m[start:stop] + ns.ms[0:lds]
            ns.data.d[start:stop]: [NDArray, Float[64]
                                    ] = ns.data.d[start:stop] + ns.ds[0:lds]

        # and recalculate li and hi
        ns.data.l: [NDArray, Float[64]]
        ns.data.h: [NDArray, Float[64]]
        [ns.data.l, ns.data.h] = get_mass(ns.data.m, ns.data.d, ns.data.sp.r)
        return ns

    def __register__(self, flux) -> None:
        """ Register this signal with a flux. This should probably be done
            through a process!  """

        self.fo: Flux = flux  # the flux handle
        self.sp: Species = flux.sp  # the species handle
        model: Model = flux.sp.mo  # the model handle add this process to the
        # list of processes
        flux.lop.append(self)

    def __call__(self) -> NDArray[np.float64]:
        """ what to do when called as a function ()"""

        return (array([self.fo.m, self.fo.l, self.fo.h,
                       self.fo.d]), self.fo.n, self)

    def plot(self) -> None:
        """
              Example:

                  Signal.plot()
            
            Plot the signal
            """
        self.data.plot()

    def describe(self) -> None:

        o = 24 * " "
        s = f"{o}{self.n}, Shape = {self.sh}:"
        logging.info(s)
        o = 24 * " "
        s = f"{o}Start = {self.st}, Mass = {self.m:4E}, Delta = {self.d}"
        logging.info(s)

    def __lt__(self, other):  # this is needed for sorting with sorted()
        return self.n < other.n

class ExternalData(esbmtkBase):
    """Instances of this class hold external X/Y data which can be associated with 
      a reservoir, e.g, to compare computed vs measured data, or with a perturbation
      where the data would be interpreted as control points.

      Example:

             ExternalData(name =  "Name"
                          file = "filename",
                          legend = "label",
                          model = model_handle)

      The data must exist as CSV file, where the first column contains
      the X-values, and the second column contains the Y-values.

      The x-values must be time and specify the time units. they will be mapped into the
      model time units.

      The y-values can be any data, but the user must take care that they match the model units
      defined in the model instance. So your data file mujst look like this

      Time [years], Data [any]
      1, 12
      2, 13
    
      The column headers are only used for the time conversion, and ignored by the
      default plotting methods, but they are available as self.xh,yh  

      The file must exist in the local working directory.

      Methods:
        - name.plot()
        - name.register(Reservoir) # associate the data with a reservoir
        - name.interpolate() # replaces input data with interpolated data across the model domain

      """

    def __init__(self, **kwargs: Dict[str, str]):

        from . import ureg, Q_
        
        # dict of all known keywords and their type
        self.lkk: Dict[str, any] = {
            "name": str,
            "filename": str,
            "legend": str,
            "model": Model,
        }
        
        # provide a list of absolutely required keywords
        self.lrk: list = ["name", "filename", "legend","model"]
        # list of default values if none provided
        self.lod: Dict[str, any] = {}

        # validate input and initialize instance variables
        self.__initerrormessages__()
        self.__validateandregister__(kwargs)  # initialize keyword values

        # legacy names
        self.n: str = self.name # string =  name of this instance
        self.fn: str = self.filename  # string = filename of data
        self.m :str = self.model.name # model handle

        self.df: pd.DataFrame = pd.read_csv(self.fn)  # read file
        logging.info(f"Read external data from {self.fn}")

        if len(self.df.columns) != 2:  # test if delta given
            raise ValueError("CSV file must have only two columns")
        
        self.xh: str = self.df.columns[0]  # get the column header
        self.yh: str = self.df.columns[1]  # get the column header

        # unit mapping
        xq = Q_(xh)
        x = df.iloc[:, 0].to_numpy() * xq
        
        self.x :[NDArray] = x.to(self.mo.t_unit).magnitude
        self.y :[NDArray] = self.df.to_numpy()[:, 1]
       
    def register(self, obj):
        """Register this dataset with a flux or reservoir. This will have the
          effect that the data will be printed together with the model
          results for this reservoir

          Example:

          ExternalData.register(Reservoir)

          """
        self.obj = obj  # reser handle we associate with
        obj.led.append(self)

    def interpolate(self) -> None:
        """Interpolate the input data with a resolution of dt across the model
        domain The first and last data point must coincide with the
        model start and end time. In other words, this method will not
        patch data at the end points.
        
        This will replace the original values of name.x and name.y. However
        the original data remains accessible as name.df


        """

        xi :[NDArray] = self.model.time
        
        if ((self.x[0] > xi[0]) or (self.x[-1] < xi[-1])):
            message = (f"\n Interpolation requires that the time domain"
                       f"is equal or greater than the model domain"
                       f"data t(0) = {self.x[0]}, tmax = {self.x[-1]}"
                       f"model t(0) = {xi[0]}, tmax = {xi[-1]}")
            
            raise ValueError(message)
        else:
            self.y :[NDArray] = interp(xi,self.x,self.y)
            self.x = xi 
        
        
    def plot(self) -> None:
        """ Plot the data and save a pdf

          Example:

                  ExternalData.plot()
          """

        fig, ax = plt.subplots() #
        ax.scatter(self.x,self.y)
        ax.set_label(self.legend)
        ax.set_xlabel(self.xh)
        ax.set_ylabel(self.yh)
        plt.show()
        plt.savefig(self.n + ".pdf")

from .connections import *
from .species_definitions import *
