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

def get_mass(m: float, d: float, r: float) -> [float, float]:
    """
    Calculate the isotope masses from bulk mass and delta value.
    Arguments are m = mass, d= delta value, r = abundance ratio 
    species
    """
    # from numpy import around
    # li = around((1000 * m) / ((d + 1000) * r + 1000),decimals=16)
    # hi = around(((d * m + 1000 * m) * r) / ((d + 1000) * r + 1000),decimals=16)
    li: float = (1000 * m) / ((d + 1000) * r + 1000)
    hi: float = ((d * m + 1000 * m) * r) / ((d + 1000) * r + 1000)
    return [li, hi]


def set_mass(m: float, d: float, r: float) -> [float, float, float]:
    """
    Calculate the isotope masses from bulk mass and delta value.
    Arguments are m = mass, d= delta value, r = abundance ratio 
    species. Unlike get_mass, this function returns the full array
    """
    from numpy import array

    l: float = (1000 * m) / ((d + 1000) * r + 1000)
    h: float = ((d * m + 1000 * m) * r) / ((d + 1000) * r + 1000)

    return array([m, l, h])

def get_delta(l, h, r):
    """
      Calculate the delta from the mass of light and heavy isotope
      Arguments are l and h which are the masses of the light and
      heavy isotopes respectively, r = abundance ratio of the respective
      element
    """
    # from numpy import around
    # d = around(1000 * (h / l - r) / r, 4)
    d = 1000 * (h / l - r) / r
    return d

def add_to (l, e):
    """
      add element e to list l, but check if the entry already exist. If so, throw
      exception. Otherwise add
    """

    if not (e in l):  # if not present, append element
        l.append(e)

def get_mag(unit, base_unit):
    """
      Compare the unit associated with the object obj (i.e., a flux, etc)
      with the base unit set for the species or model (base_unit)
      ms = magnitude string, s = scaling factor
    """

    # E.g., unit = mmol, and base_unit = mol -> ms = m, and thus s = 1E-3
    if len(base_unit) > len(unit):
        ms = base_unit.replace(unit, "")
    else:
        ms = unit.replace(base_unit, "")  # get the magnitude string of the species
        
        if ms == "":  # species unit and reservoir units are the same
            s = 1  # -> no scaling
        elif ms == "G":  # value is provided in mega
            s = 1E9  # thus they need to be scaled by 1e9
        elif ms == "M":  # value is provided in mega
            s = 1E6  # thus they need to be scaled by 1e6
        elif ms == "k":  # value is provided in kilo
            s = 1E3  # thus they need to be scaled by 1e3
        elif ms == "m":  # value is provided in milli
            s = 1E-3  # thus they need to be scaled by 1e-3
        elif ms == "u":  # value is provided in micro
            s = 1E-6  # thus they need to be scaled by 1e-6
        elif ms == "n":  # value is provided in nano
            s = 1E-9  # thus they need to be scaled by 1e-9
        elif ms == "p":  # value is print(replace. Tab to end.)ovided in pico
            s = 1E-12  # thus they need to be scaled by 1e-12
        elif ms == "f":  # value is print(replace. Tab to end.)ovided in femto
            s = 1E-15  # thus they need to be scaled by 1e-15
        else:  # unknown conversion
            s = 1  # -> no scaling
            raise ValueError(
                (f"magnitude = {ms}, unit = {unit} "
                 f"base_unit = {base_unit} ."
                 f"This case is not defined (yet?)")
            )
        
    if len(base_unit) > len(unit):
        s = 1 / s
        
    return s

def get_plot_layout(obj):
      """ Simple function which selects a row, column layout based on the number of
      objects to display.  The expected argument is a reservoir object which
      contains the list of fluxes in the reservoir

      """
      import logging

      noo = len(obj.lof) + 1  # number of objects in this reservoir
      logging.debug(f"{noo} subplots for {obj.n} required")

      if noo < 2:
          geo = [1, 1]  # one row, one column
          size = [5, 3]  # size in inches
      elif 1 < noo < 3:
          geo = [2, 1]  # two rows, one column
          size = [5, 6]  # size in inches
      elif 2 < noo < 5:
          geo = [2, 2]  # two rows, two columns
          size = [10, 6]  # size in inches
      elif 4 < noo < 7:
          geo = [2, 3]  # two rows, three columns
          size = [15, 6]  # size in inches
      elif 6 < noo < 10:
          geo = [3, 3]  # two rows, three columns
          size = [15, 9]  # size in inches
      else:
          print("plot geometry for more than 8 fluxes is not yet defined")
          print("Consider calling flux.plot individually on each flux in the reservoir")
          # print(f"Selected Geometry: rows = {geo[0]}, cols = {geo[1]}")

      return size, geo

def sum_fluxes(flux_list,reservoir,i):
    """This function takes a list of fluxes and calculates the sum for each
    flux element.  It will return a an array with [mass, li, hi] The
    summation is a bit more involved than I like, but I issues with summing
    small values, so here we doing is using fsum. For this step, we need to
    look up the reservoir specific flux direction which is stored as
    key-value pair in the reservoir.lio dictionary (flux name: direction)
    """
    from numpy import array
    
    ms  = 0
    ls  = 0
    hs  = 0

    for f in flux_list:  # do sum of fluxes in this reservoir
        direction = reservoir.lio[f.n]
        ms  = ms + f.m[i] * direction # current flux and direction
        ls  = ls + f.l[i] * direction # current flux and direction
        hs  = hs + f.h[i] * direction # current flux and direction

    # sum up the each array component individually
    new = array([ms, ls, hs])
    return new

def list_fluxes(self,name,i) -> None:
            """
            Echo all fluxes in the reservoir to the screen
            """
            print(f"\nList of fluxes in {self.n}:")
            
            for f in self.lof: # show the processes
                  direction = self.lio[f.n]
                  if direction == -1:
                        t1 = "From:"
                        t2 = "Outflux from"
                  else:
                        t1 = "To  :"   
                        t2 = "Influx to"

                  print(f"\t {t2} {self.n} via {f.n}")
                  
                  for p in f.lop:
                        p.describe()

            print(" ")
            for f in self.lof:
                  f.describe(i) # print out the flux data

def show_data(self,name,i) -> None:
    """ Print the first 4, and last 3 lines of the data for a given flux or reservoir object
    """
    
    # show the first 4 entries
    print(f"{name}:")
    for i in range(i,i+3):
        print(f"\t i = {i}, Mass = {self.m[i]:.2f}, LI = {self.l[i]:.2f}, HI = {self.h[i]:.2f}, delta = {self.d[i]:.2f}")
    
    print(".......................")

class esbmtkBase():
    """The esbmtk base class template. This class handles keyword
    arguments, name registration and other common tasks

    """
    from numpy import set_printoptions
    from typing import Dict

    set_printoptions(precision=4)

    def __init__(self) -> None:
        raise NotImplementedError

    def __validateandregister__(self, kwargs: Dict[str, any]) -> None:
        """Validate the user provided input key-value pairs. For this we need
        kwargs = dictionary with the user provided key-value pairs
        self.lkk = dictionary with allowed keys and type
        self.lrk = list of mandatory keywords
        self.lod = dictionary of default values for keys

        and register the instance variables and the instance in teh global name space
        """
        import builtins

        # validate input
        self.__validateinput__(kwargs)

        # register all key/value pairs as instance variables
        self.__registerkeys__()

        # register instance name in global name space
        setattr(builtins, self.name, self)

    def __validateinput__(self, kwargs: Dict[str, any]) -> None:
        """Validate the user provided input key-value pairs. For this we need
        kwargs = dictionary with the user provided key-value pairs
        self.lkk = dictionary with allowed keys and type
        self.lrk = list of mandatory keywords
        self.lod = dictionary of default values for keys

        """

        import logging
        import time
        from typing import Dict, List
        from esbmtk import Model

        self.kwargs = kwargs  # store the kwargs
        self.provided_kwargs = kwargs.copy()  # preserve a copy

        if not self.lkk:  #dictionary with allowed keys and type
            self.lkk: Dict[str, any] = {}
        if not self.lrk:  # list with mandatory keywords
            self.lrk: List[str] = []
        if not self.lod:  # dictionary of default values for keys
            self.lod: Dict[str, any] = []

        # check that mandatory keys are present
        # and that all keys are allowed
        self.__checkkeys__()

        # initialize missing parameters

        self.kwargs = self.__addmissingdefaults__(self.lod, kwargs)

        # check if key values are of correct type
        self.__checktypes__(self.lkk, self.kwargs)

        # this wont work since we don't know the model yet.
        # coulde be moved into a post init section?

        # register instance on the list of model objects
        #if not type(self) == Model:
        #    self.mo.lmo.append(self)  # register on the list of model objects

        # start log entry
        #logfile = self.name+".log"
        #logging.basicConfig(filename=logfile,
        #                    filemode='w',
        #                    format='%(message)s',
        #                    level=logging.INFO)
        #logging.info(f"{self.name} initialized on {time.ctime()}")

    def __checktypes__(self, av: Dict[any, any], pv: Dict[any, any]) -> None:
        """ this method will use the the dict key in the user provided
        key value data (pv) to look up the allowed data type for this key in av
        
        av = dictinory with the allowed input keys and their type
        pv = dictionary with the user provided key-value data
        """
        from numbers import Number
        from typing import Dict
        from esbmtk import Model, Element, Species, Flux, Reservoir, Signal, Process
        from esbmtk import ExternalData

        k: any
        v: any

        # provide more meaningful error messages

        # loop over provided keywords
        for k, v in pv.items():
            # check av if provided value v is of correct type
            if not isinstance(v, av[k]):
                print(f"Offending Key = {k}")
                raise TypeError(f"{v} must be {m[k]}, not {type(v)}")

    def __initerrormessages__(self):
        """ Init the list of known error messages"""
        self.bem: Dict[str, str] = {
            "Number": "a number",
            "Model": "a model handle (i.e. the name without quotation marks)",
            "Element":
            "an element handle (i.e. the name without quotation marks)",
            "Species":
            "a species handle (i.e. the name without quotation marks)",
            "Flux": "a flux handle (i.e. the name without quotation marks)",
            "Reservoir":
            "a reservoir handle (i.e. the name without quotation marks)",
            "Signal":
            "a signal handle (i.e. the name without quotation marks)",
            "Process":
            "a process handle (i.e. the name without quotation marks)",
            "Unit": "a string",
            "File": "a filename inb the local directory",
            "Legend": " a string",
            "Source": " a string",
            "Sink": " a string",
            "Ref": " a Flux reference",
            "Alpha": " a Number",
            "Delta": " a Number",
            "Scale": " a Number",
            "Ratio": " a Number",
            "number": "a number",
            "model": "a model handle (i.e. the name without quotation marks)",
            "element":
            "an element handle (i.e. the name without quotation marks)",
            "species":
            "a species handle (i.e. the name without quotation marks)",
            "flux": "a flux handle (i.e. the name without quotation marks)",
            "reservoir":
            "a reservoir handle (i.e. the name without quotation marks)",
            "signal":
            "a signal handle (i.e. the name without quotation marks)",
            "Process":
            "a process handle (i.e. the name without quotation marks)",
            "unit": "a string",
            "file": "a filename inb the local directory",
            "legend": " a string",
            "source": " a string",
            "sink": " a string",
            "ref": " a Flux reference",
            "alpha": " a Number",
            "delta": " a Number",
            "scale": "a Number",
            "ratio": "a Number",
            "concentration": "a Number",
            "pl": " a list with one or more process handles",
            "react_with": "a Flux handle",
            "data": "External Data Object",
            str: "a string with quotation marks",
        }

    def __registerkeys__(self) -> None:
        """ register the kwargs key/value pairs as instance variables
        and complain about uknown keywords"""
        k: any
        v: any

        for k, v in self.kwargs.items():
            setattr(self, k, v)

    def __checkkeys__(self) -> None:
        """ check if the mandatory keys are present"""
        from typing import List

        k: str
        v: any
        # test if the required keywords are given
        for k in self.lrk:  # loop over required keywords
            if isinstance(k, list):  # If keyword is a list
                s: int = 0  # loop over allowed substitutions
                for e in k:  # test how many matches are in this list
                    s = s + int(e in self.kwargs)
                if s != 1:  # if none, or more than one match, throw error
                    raise ValueError(
                        f"You need to specify exactly one from this list: {k}")

            else:  # keyword is not a list
                if k not in self.kwargs:
                    raise ValueError(f"You need to specify a value for {k}")

        tl: List[str] = []
        # get a list of all known keywords
        for k, v in self.lkk.items():
            tl.append(k)

        # test if we know all keys
        for k, v in self.kwargs.items():
            if k not in self.lkk:
                raise ValueError(
                    f"{k} is not a valid keyword. \n Try any of \n {tl}\n")

    def __addmissingdefaults__(self, lod: dict, kwargs: dict) -> dict:
        """
        test if the keys in lod exist in kwargs, otherwise add them with the default values
        in lod
        """
        new: dict = {}
        if len(self.lod) > 0:
            for k, v in lod.items():
                if k not in kwargs:
                    new.update({k: v})

        kwargs.update(new)
        return kwargs

    def __repr__(self):
        """ Print the basic parameters for this class when called via the print method
        """
        from esbmtk import esbmtkBase
        m: str = ""
        
        for k, v in self.provided_kwargs.items():
            if not isinstance({k}, esbmtkBase):
                m = m + f"{k} = {v}\n"
        return m

class Model(esbmtkBase):
    """ This is the class specify a new model.

    Example:

            esbmtkModel(name =  "Test_Model",
                      start = 0,          # optional: start time 
                      stop  = 1000,       # end time
                      time_unit = "yr",   # units for start/end time
                      dt = 2,
                      dt_unit = "yr",     # optional: units for dt
                      base_unit = "yr",   # optional: time unit for model
            )
           
    all of the above keyword values are available as Model_Name.keyword

    The user facing methods of the model class are
       - Model_Name.describe()
       - Model_Name.save_data()
       - Model_Name.plot_data()
       - Model_Name.plot_reservoirs()
       - Model_Name.run()
    
    """
    from typing import Dict
    from nptyping import NDArray, Float
    # from numba import jit, njit

    def __init__(self, **kwargs: Dict[any, any]) -> None:
        """ initialize object"""

        from numpy import arange, zeros, array
        from esbmtk import get_mag
        from typing import Dict
        from numbers import Number
        from nptyping import NDArray, Float

        # provide a dict of all known keywords and their type
        self.lkk: Dict[str, any] = {
            "name": str,
            "start": Number,
            "stop": Number,
            "time_unit": str,
            "dt": Number,
            "dt_unit": str,
            "base_unit": str
        }

        # provide a list of absolutely required keywords
        self.lrk: list[str] = ["name", "stop", "time_unit", "dt"]

        # list of default values if none provided
        self.lod: Dict[str, any] = {
            'start': 0,
            'dt_unit': kwargs['time_unit'],
            'base_unit': kwargs['time_unit']
        }

        self.__initerrormessages__()
        self.__validateandregister__(kwargs)  # initialize keyword values

        # empty list which will hold all reservoir references
        self.lor: list = []
        # empty list which will hold all connector references
        self.loc: list = []  
        self.lel: list = []  # list which will hold all element references
        self.lsp: list = []  # list which will hold all species references
        self.lop: list = []  # list flux processe
        self.lmo: list = []  # list of all model objects
        self.olkk : list = [] # optional keywords for use in the connector class

        # legacy name defintions
        self.bu = self.base_unit
        self.n = self.name
        self.mo = self.name

        self.xl = f"Time [{self.time_unit}]"  # time axis label
        self.mag_dt = get_mag(self.dt_unit,
                              self.base_unit)  # get magnitude rel to base unit
        print(f"mag_dt = {self.mag_dt}")
        self.dt = self.dt * self.mag_dt  # convert to base unit
        self.mag = get_mag(self.time_unit,
                           self.base_unit)  # get magnitude rel to base unit
        print(f"mag_dt = {self.mag_dt}, dt = {self.dt}")
        self.start = self.start * self.mag
        self.stop = self.stop * self.mag
        self.tu = self.time_unit  # Time units used to set start/end time
        self.length = int(abs(self.start - self.stop))
        self.steps = int(abs(round(self.length / self.dt)))
        self.time = ((arange(self.steps) * self.dt) + self.start) / self.mag

    def describe(self) -> None:
        """ Describe Basic Model Parameters and log them
        """

        import logging
        logging.info("---------- Model description start ----------")
        logging.info(f"Model Name = {self.n}")
        logging.info(f"Model time [{self.bu}], dt = {self.dt}")
        logging.info(
            f"start = {self.start}, stop={self.stop/self.mag} [{self.tu}]")
        logging.info(f"Steps = {self.steps}\n")
        logging.info(f"  Species(s) in {self.n}:")

        for r in self.lor:
            r.describe()
            logging.info(" ")

            logging.info("---------- Model description end ------------\n")

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

        import matplotlib.pyplot as plt
        i = 0
        for r in self.lor:
            r.__plot__(i)
            i = i + 1

        plt.show()  # create the plot windows

    def plot_reservoirs(self) -> None:
        """Loop over all reservoirs and either plot the data into a window,
            or save it to a pdf
        """

        import matplotlib.pyplot as plt
        i = 0
        for r in self.lor:
            r.__plot_reservoirs__(i)
            i = i + 1

        plt.show()  # create the plot windows

    def run(self) -> None:
        """Loop over the time vector, and for each time step, calculate the
        fluxes for each reservoir
        """

        from time import process_time
        from numpy import array, zeros
        from nptyping import NDArray, Float

        # this has nothing todo with self.time below!
        start: float = process_time()
        new: [NDArray, Float] = zeros(3) + 1
        
        i = self.execute(new, self.time, self.lor)
        
        duration: float = process_time() - start
        print(f"Execution took {duration} seconds")

    # some mumbo jumbbo to support numba optimization. Currently not working though
    @staticmethod
    def execute(new:[NDArray, Float] , time: [NDArray, Float], lor:list) -> None:
        """ Moved this code into a separate function to enable numba optimization
        """
        
        from nptyping import NDArray, Float
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
    def __init__(self, **kwargs) -> None:
        """ Initialize all instance variables
        """
        from numbers import Number
        from typing import Dict
        from esbmtk import Model
        import logging

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
        from esbmtk import Element
        from typing import Dict
        
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
        self.en = self.element.n   # element name
        self.e  = self. element    # element handle

        self.mo.lsp.append(self)   # register self on the list of model objects

    def __lt__(self, other) -> None:  # this is needed for sorting with sorted()
        return self.n < other.n

class Reservoir(esbmtkBase):
    """
      Tis object holds reservoir specific information. 

      Example:

              Reservoir(name = "IW_SO4",      # Name of reservoir
                        species = S,          # Species handle
                        delta = 20,           # initial delta - optional (defaults  to 0)
                        mass/concentration = 200,  # species concentration or mass
                        unit = "mmol",        # concentration unit
                        volume = 1E5,         # reservoir volume (m^3) 
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

    from nptyping import NDArray, Float
    import numpy as np

    def __init__(self, **kwargs) -> None:
        """ Initialize a reservoir. 
            """
        from numpy import zeros  # import numpy library
        from esbmtk import get_mass, get_mag, Species, ExternalData, Species, Process, Flux, Element
        from numbers import Number
        from typing import Dict, Tuple
        from nptyping import NDArray, Float, Number

        # provide a dict of all known keywords and their type
        self.lkk: Dict[str, any] = {
            "name": str,
            "species": Species,
            "delta": Number,
            "concentration": Number,
            "mass": Number,
            "unit": str,
            "volume": Number
        }

        # provide a list of absolutely required keywords
        self.lrk: list = [
            "name", "species", "unit", "volume", ["mass", "concentration"]
        ]

        # list of default values if none provided
        self.lod: Dict[any, any] = {'delta': 0, 'concentration': 0, 'mass': 0}

        # validate and initialize instance variables
        self.__initerrormessages__()
        self.bem.update({"concentration": "a number"})
        self.__validateandregister__(kwargs)

        # legacy names
        self.n: str = self.name  # name of reservoir
        self.mu: str = self.unit  # massunit
        self.sp: Species = self.species  # species handle
        self.mo: Model = self.species.mo  # model handle
        self.v: Number = self.volume  # reservoir volume
        if self.concentration == 0:
            if self.mass == None:
                raise ValueError("You need to specify mass or concentration")
            else:
                self.concentration = self.mass / self.volume

        self.c: Number = self.concentration  # concentration

        self.lof: list[Flux] = []  #  flux references
        self.led: list[ExternalData] = []  # all external data references
        self.lio: dict[str, int] = {}  #  flux name:direction pairs
        self.lop: list[Process] = []  # list holding all processe references
        self.loe: list[Element] = []  # list of elements in thiis reservoir
        self.doe: Dict[Species, Flux] = {}  # species flux pairs

        # get magnitude rel to species
        self.mag: float = get_mag(self.mu, self.species.mu)
        self.c: float = self.c * self.mag  # convert to base unit
        mass: float = self.c * self.v  # caculate mass
        # initialize mass vector
        self.m: [NDArray, Float[64]] = zeros(self.species.mo.steps) + mass
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

        self.mo.lor.append(self)  # add this reservoir to the model

    def __call__(self) -> None:  # what to do when called as a function ()
        pass
        return self

    def __getitem__(
            self,
            i: int) -> NDArray[np.float64]:  # howto get data by index [i]
        """ Get flux data by index
        """
        from numpy import array

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
        import logging
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
            Write model data \int_{}^{} d
        o csv file. Each Reservoir gets its own file
            Files are named as 'Modelname_Reservoirname.csv'
            """
        from pandas import DataFrame

        # some short hands
        sn = self.sp.n  # species name
        sp = self.sp
        # species mass units in the reservoir
        smu = f"[{self.sp.mu}]"
        mtu = f"[{self.sp.mo.bu}]"  # model time units
        fmu = f"[{self.sp.mu}/{self.sp.mo.bu}]"  # mass unit for the fluxes
        sdn = self.sp.dn  # delta name
        sds = f"[{self.sp.ds}]"  # delta scale
        rn = self.n  # reservoir name
        mn = self.sp.mo.n  # model name
        fn = f"{mn}_{rn}.csv"  # file name

        df: pd.dataframe = DataFrame()
        df[f"{self.n}_{sn}_{smu}"] = self.m
        df[f"{self.n}_{sp.ln}"] = self.l
        df[f"{self.n}_{sp.hn} "] = self.h
        df[f"{self.n}_{sdn} {sds}"] = self.d

        for f in self.lof:  # Assemble the headers and data for the reservoir fluxes
            df[f"{f.n}_{sn}_{fmu}"] = f.m
            df[f"{f.n}_{sn}_{sp.ln}"] = f.l
            df[f"{f.n}_{sn}_{sp.hn}"] = f.h
            df[f"{f.n}_{sn}_{sdn}, {sds}"] = f.d

        df.to_csv(fn)  # Write dataframe to file
        return df

    def __plot__(self, i: int) -> None:
        """ 
            Plot data from reservoirs and fluxes into a multiplot window
            """

        import matplotlib.pyplot as plt
        from esbmtk import get_plot_layout, plot_object_data

        model = self.sp.mo
        species = self.sp
        obj = self
        time = model.time  # get the model time
        xl = f"Time [{model.bu}]"

        size, geo = get_plot_layout(self)  # adjust layout
        filename = f"{model.n}_{self.n}.pdf"
        fn = 1  # counter for the figure number

        fig = plt.figure(i)  # Initialize a plot window
        fig.canvas.set_window_title(f"Reservoir Name: {self.n}")
        fig.set_size_inches(size)

        # plot reservoir data
        plot_object_data(geo, fn, self.c, self.d, self)

        # plot teh fluxes assoiated with this reservoir
        for f in sorted(self.lof):  # plot flux data
            fn = fn + 1
            plot_object_data(geo, fn, f.m, f.d, f)

        fig.suptitle(f"Model: {model.n}, Reservoir: {self.n}\n", size=16)
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        fig.savefig(filename)

    def __plot_reservoirs__(self, i: int) -> None:
        """ 
            Plot only the  reservoirs data, and ignore the fluxes
            """

        import matplotlib.pyplot as plt
        from esbmtk import get_plot_layout, plot_object_data

        model = self.sp.mo
        species = self.sp
        obj = self
        time = model.time  # get the model time
        xl = f"Time [{model.bu}]"

        size = [5, 3]
        geo = [1, 1]
        filename = f"{model.n}_{self.n}.pdf"
        fn = 1  # counter for the figure number

        fig = plt.figure(i)  # Initialize a plot window
        fig.set_size_inches(size)

        # plt.legend()ot reservoir data
        plot_object_data(geo, fn, self.c, self.d, self)

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
              rate  = flux rate,
              unit  = flux unit
      )

       You can access the flux data as
      - Name.m # mass
      - Name.d # delta
      - Name.c # concentration
      
      """
    from typing import Dict, Tuple
    from nptyping import NDArray, Float
    import numpy as np

    def __init__(self, **kwargs: Dict[str, any]) -> None:
        """
          Initialize a flux. Arguments are the species name the flux rate
          (mol/year), the delta value and unit
          """
        from numpy import zeros
        from esbmtk import get_mag, get_mass, Species, Model, Reservoir, Process, ExternalData
        from typing import Dict, Tuple
        from nptyping import NDArray, Float
        from numbers import Number
        import logging

        # provide a dict of all known keywords and their type
        self.lkk: Dict[str, any] = {
            "name": str,
            "species": Species,
            "delta": Number,
            "rate": Number,
            "unit": str,
        }

        # provide a list of absolutely required keywords
        self.lrk: list = ["name", "species", "unit", "rate"]

        # list of default values if none provided
        self.lod: Dict[any, any] = {'delta': 0}

        # initialize instance
        self.__initerrormessages__()
        self.__validateandregister__(kwargs)  # initialize keyword values

        # legacy names
        self.n: str = self.name  # name of flux
        self.sp: Species = self.species  # species name
        self.mo: Model = self.species.mo  # model name
        self.model: Model = self.species.mo  # model handle

        self.mu: str = self.unit  # mass unit used for this flux
        # unit is mass/time, so we need to first split the string
        mu, tu = self.unit.split("/")  # mu = mass unit, tu = time unit
        logging.debug(f"mu = {mu}, {self.species.mu}")
        logging.debug(f"tu = {tu}, model.bu = {self.model.bu}")
        mu_mag = get_mag(mu, self.species.mu)  # get magnitude rel to species
        tu_mag = get_mag(tu, self.model.bu)  # get timestep rel to model time
        logging.debug(f"scale factor time = {tu_mag}, mass = {mu_mag}")

        self.mag: float = mu_mag / tu_mag  # set scale factor accordingly

        fluxrate: float = self.rate * self.mag  # convert flux to model units
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
        self.xl: str = self.model.xl  # se x-axis label equal to model time
        self.lop: list[Process] = []  # list of processes
        self.led: list[ExternalData] = []  # list of ext data
        #self.t = 0        # time dependent flux = 1, otherwise 0
        self.source: str = ""  # Name of reservoir which acts as flux source
        self.sink: str = ""  # Name of reservoir which acts as flux sink

    def __getitem__(
            self,
            i: int) -> NDArray[np.float64]:  # howto get data by index [i]
        from numpy import array
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
        import logging
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

        import matplotlib.pyplot as plt

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

        from esbmtk import Species

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
        self.u = self.species.mu + "/" + self.species.mo.bu


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
                   start = 0,           # optional
                   duration = 0,        #
                   delta = 0,           # optional
                   stype = "addition"   # optional, currently the only type
                   shape = "square"     # square, pyramid
                   mass/magnitude/data  # give one
                  )

      Signals are cumulative, i.e., complex signals ar created by
      adding one signal to another (i.e., Snew = S1 + S2) 

      Signals are registered with a flux during flux creation,
      i.e., they are passed on the process list when calling the
      connector object.
    
      if the data argument is used, you can provide a filename which
      contains the data to be used in scv format. The data will be
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

    from nptyping import NDArray, Float
    import numpy as np

    def __init__(self, **kwargs) -> None:
        """ Parse and initialize variables
            """
        from numbers import Number
        from esbmtk import Species, Signal, Model
        from typing import Dict, List
        from esbmtk import ExternalData, get_mass

        # provide a list of all known keywords and their type
        self.lkk: Dict[str, any] = {
            "name": str,
            "start": Number,
            "duration": Number,
            "species": Species,
            "delta": Number,
            "stype": str,
            "shape": str,
            "filename": str,
            "mass": Number,
            "magnitude": Number
        }

        # provide a list of absolutely required keywords
        self.lrk: List[str] = [
            "name", "duration", "species", ["shape", "filename"],
            ["magnitude", "mass", "filename"]
        ]

        # list of default values if none provided
        self.lod: Dict[str, any] = {
            'start': 0,
            'stype': "addition",
            'shape': "external_data",
            'duration': 0,
            'delta': 0,
        }

        self.__initerrormessages__()
        self.bem.update({"data": "a string", "magnitude": Number})
        self.__validateandregister__(kwargs)  # initialize keyword values

        self.los: List[Signal] = []  # list of signals we are based on.

        # legacy name definitions
        self.n: str = self.name  # the name of the this signal
        self.sp: Species = self.species  # the species
        self.mo: Model = self.species.mo  # the model handle
        self.st: Number = self.start  # start time
        self.ty: str = self.stype  # type of signal
        self.l: Number = self.duration  # the duration
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
        from numpy import mean, array, interp, arange
        from esbmtk import Flux, get_mass
        from nptyping import NDArray, Float

        # create a dummy flux we can act up
        self.nf: Flux = Flux(name=self.n + "_data",
                             species=self.sp,
                             rate=0,
                             delta=0,
                             unit=self.sp.mu + "/" + self.mo.tu)

        # since the flux is zero, the delta value will be undefined. So we set it explicitly
        # this will avoid having additions with Nan values.
        self.nf.d[0:]: float = 0

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

        from numpy import mean, array, interp, arange
        from nptyping import NDArray, Float

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

        Time, Rate, delta value
        0,     10,   12

        i.e., the first row needs to be a header line
        
        """

        from numpy import mean, array, interp, arange
        import pandas as pd
        from nptyping import NDArray, Float

        # read external dataset
        df = pd.read_csv(self.filename)

        x = df.iloc[:, 0].to_numpy()
        y = df.iloc[:, 1].to_numpy()
        d = df.iloc[:, 2].to_numpy()

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

        from copy import deepcopy
        from nptyping import NDArray, Float

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

        from copy import deepcopy
        from esbmtk import Signal
        from nptyping import NDArray, Float
        import numpy as np

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
        from esbmtk import Flux, Species, model

        self.fo: Flux = flux  # the flux handle
        self.sp: Species = flux.sp  # the species handle
        model: Model = flux.sp.mo  # the model handle add this process to the
        # list of processes
        flux.lop.append(self)

    def __call__(self) -> NDArray[np.float64]:
        """ what to do when called as a function ()"""
        from numpy import array
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
        import logging
        o = 24 * " "
        s = f"{o}{self.n}, Shape = {self.sh}:"
        logging.info(s)
        o = 24 * " "
        s = f"{o}Start = {self.st}, Mass = {self.m:4E}, Delta = {self.d}"
        logging.info(s)

    def __lt__(self, other):  # this is needed for sorting with sorted()
        return self.n < other.n

class Vector:
      """
      The vector object simply contains a list of values, which can be used a input 
      to modify quantities like fluxes, or reservoir size. Their data structure is similar
      to fluxes, i.e., they have 4 fields, 3 of which are ignored. This is wasteful in terms
      of memory, but simplified the coding, since other classes do not need to be aware
      whether they have only one or 4 fields. So we can treat them like fluxes which are not
      associated with a reservoir. Which begs the question, whether we need them in 
      the first place?
      """ 

      def __int__(self, name, value) -> None:
          """
          Arguments are the name of the vector and the initial value. Vector values
          can subsequently be modified using the perturbation class.
          """

def plot_object_data(geo, fn, yl, yr, obj) -> None:
      """collection of commands which will plot and annotate a reservoir or flux
      object into an existing plot window. 
      """
      import matplotlib.pyplot as plt

      # geo = list with rows and cols
      # fn  = figure number
      # yl  = array with y values for the left side
      # yr  = array with y values for the right side
      # obj = object handle, i.e., reservoir or flux
      
      rows = geo[0]
      cols = geo[1]
      species = obj.sp
      model = obj.mo
      time = model.time

      ax1 = plt.subplot(rows, cols, fn, title=obj.n)  # start subplot

      cn = 0
      col  = f"C{cn}"
      ln1 = ax1.plot(time[1:-2], yl[1:-2] / obj.mag, color=col, label=obj.lm)        # plot left y-scale data
      ax1.set_xlabel(obj.xl)                    # set the x-axis label
      ax1.set_ylabel(obj.lm)   # the y labqel
      ax1.spines['top'].set_visible(False)  # remove unnecessary frame speciess

      cn = cn + 1
      col  = f"C{cn}"
      ax2 = ax1.twinx()                     # create a second y-axis
      ln2 = ax2.plot(time[1:-2], yr[1:-2], color=col, label=obj.ld)        # plof right y-scale data

      ax2.set_ylabel(obj.ld)  # species object delta label
      ax2.spines['top'].set_visible(False)  # remove unnecessary frame speciess

      for d in obj.led: # loop over external data objects if present
          if isinstance(d.x[0], str): # if string, something is off
              raise ValueError("No time axis in external data object {d.name}") 
          if isinstance(d.y[0],str) is False:  # mass or concentration data is present
              cn = cn + 1
              col  = f"C{cn}"
              leg  = f"{obj.lm} {d.legend}"
              ln3 = ax1.scatter(d.x, d.y, color=col, label = leg)
          if isinstance(d.d[0], str) is False:  # isotope data is present
              cn = cn + 1
              col  = f"C{cn}"
              leg  = f"{obj.ld} {d.legend}"
              ln3 = ax2.scatter(d.x, d.d, color=col, label = leg)

      # collect all labels and print them in one legend
      handler1, label1 = ax1.get_legend_handles_labels()
      handler2, label2 = ax2.get_legend_handles_labels()
      legend = ax2.legend(handler1+handler2, label1+label2, loc=0).set_zorder(6)
      # ax1.legend(frameon=False)

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
      the X-values, and the second column contains the Y-values. The first row should
      contain column headers, however, these are ignored by the
      default plotting methods, but they are available as self.xh,yh  

      The file must exist in the local working directory.

      Methods:
        - name.plot()
        - name.register(Reservoir) # associate the data with a reservoir
        - name.interpolate() # replaces input data with interpolated data across the model domain

      """

    from typing import Dict

    def __init__(self, **kwargs: Dict[str, str]):
        from esbmtk import Model, ExternalData
        import pandas as pd
        from typing import Dict
        from nptyping import Float, NDArray
        import logging

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
        # first column should be time
        # second colum should be data
        self.x :[NDArray] = self.df.to_numpy()[:, 0]
        self.y :[NDArray] = self.df.to_numpy()[:, 1]
        self.xh: str = self.df.columns[0]  # get the column header
        self.yh: str = self.df.columns[1]  # get the column header

        if len(self.df.columns) != 2:  # test if delta given
            raise ValueError("CSV file must have only two columns")

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
        from numpy import interp
        from nptyping import NDArray, Float

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
        import matplotlib.pyplot as plt
        import pandas as pd

        fig, ax = plt.subplots() #
        ax.scatter(self.x,self.y)
        ax.set_label(self.legend)
        ax.set_xlabel(self.xh)
        ax.set_ylabel(self.yh)
        plt.show()
        plt.savefig(self.n + ".pdf")

class Connect(esbmtkBase):
    """Name:

        Connect

    Description: Two reservoirs connect to each other via at least 1
    flux. This module creates the connecting flux and creates a
    connecctor object which stores all connection properties

    Connection properties include:
       - the direction of the flux (from A to B)
       - any processes which act on the flux, and whether these processes depend on the upstream, downstream or both reservoirs
       - the type of flux:
           - Fixed: both flux-rate and delta are given, allowed processes include signal and fractionation
           - Reservoir-Driven: delta and or flux rate depend on the reservoir data (upstream/downstream both)
               - if nothing assume upstream reservoir passive flux with var delta
               - if only flux it assume upstream reservoir with fixed flux and var delta
               - if only delta assume varflux with fixed delta
               - if both delta and flux are given print warning and suggest to use a static flux
               - if only alpha assume upstream var flux and fractionation process
               - Allowed processes: ALL

    Example:
    
    Connect(source =  upstream reservoir
	   sink = downstrean reservoir
           delta = optional
           alpha = optional
           rate = optional
           ref = optional
           species = optional
           type = optional
	   pl = [list]) process list. optional

    Currently reckonized flux properties: delta, rate, alpha, species
    """

    from typing import Dict

    def __init__(self, **kwargs):
        """ The init method of the connector obbjects performs sanity checks e.g.:
               - whether the reservoirs exist
               - correct flux properties (this will be handled by the process object)
               - whether the processes do exist (hmmh, that implies that the optional processes do get registered with the model)
               - creates the correct default processes
               - and connects the reservoirs

        Arguments:
           name = name of the connector object : string
           source   = upstream reservoir    : object handle
           sink  = downstream reservoir  : object handle
           fp   = connection_properties : dictionary {delta, rate, alpha, species, type}
           pl[optional]   = optional processes : list
        """
        import builtins
        from esbmtk import Source, Sink, Reservoir, Flux, Process
        from typing import Dict, List
        from numbers import Number

        # provide a dict of all known keywords and their type
        self.lkk: Dict[str, any] = {
            "name": str,
            "source": (Source, Reservoir),
            "sink": (Sink, Reservoir),
            "delta": Number,
            "rate": Number,
            "pl": list,
            "alpha": Number,
            "species": Species,
            "type": str,
            "ref": Flux,
            "react_with": Flux,
            "ratio": Number,
            "scale": Number,
            "kvalue": Number,
            "C0": Number
        }

        n = kwargs["source"].n + "_" + kwargs[
            "sink"].n + "_connector"  # set the name
        kwargs.update({"name": n})  # and add it to the kwargs

        # provide a list of absolutely required keywords
        self.lrk: list = ["name", "source", "sink"]

        # list of default values if none provided
        self.lod: Dict[any, any] = {}

        # validate and initialize instance variables
        self.__initerrormessages__()
        self.__validateandregister__(kwargs)

        if not 'pl' in kwargs:
            self.pl: list[Process] = []

        # legacy names
        self.influx: int = 1
        self.outflux: int = -1
        self.n = self.name
        self.mo = self.source.sp.mo

        self.p = 0  # the default process handle
        self.r1: (Process, Reservoir) = self.source
        self.r2: (Process, Reservoir) = self.sink

        self.get_species(self.r1, self.r2)  #
        self.mo: Model = self.sp.mo  # the current model handle
        self.lor: list[
            Reservoir] = self.mo.lor  # get a list of all reservoirs registered for this species

        self.mo.loc.append(self)  # register connector with model
        self.register_fluxes()  # Source/Sink/Regular
        self.__set_process_type__()  # derive flux type and create flux(es)
        self.register_process()  # This should probably move to register fluxes

    def get_species(self, r1, r2) -> None:
        """In most cases the species is set by r2. However, if we have
        backward fluxes the species depends on the r2

        """
        #print(f"r1 = {r1.n}, r2 = {r2.n}")
        if isinstance(self.r1, Source):
            self.r = r1
        else:  # in this case we do have an upstream reservoir
            self.r = r2

        # test if species was explicitly given
        if "species" in self.kwargs:  # this is a quick fix only
            self.sp = self.kwargs["species"]
        else:
            self.sp = self.r.sp  # get the parent species

    def register_fluxes(self) -> None:
        """Create flux object, and register with reservoir and global namespace

        """
        import builtins

        # test if default arguments present
        if "delta" in self.kwargs:
            d = self.kwargs["delta"]
        else:
            d = 0

        if "rate" in self.kwargs:
            r = self.kwargs["rate"]
        else:
            r = 1

        # flux name
        n = self.r1.n + '_to_' + self.r2.n  # flux name r1_to_r2

        # derive flux unit from species obbject
        funit = self.sp.mu + "/" + self.sp.mo.bu

        self.fh = Flux(
            name=n,  # flux name
            species=self.sp,  # Species handle
            delta=d,  # delta value of flux
            rate=r,  # flux value
            unit=funit,  # flux unit
        )

        # register flux with its reservoirs
        if isinstance(self.r1, Source):
            self.r2.lio[
                self.fh.n] = self.influx  # add the flux name direction/pair
            self.r2.lof.append(self.fh)  # add the handle to the list of fluxes
            self.__register_species__(
                self.r2,
                self.r1.sp)  # register flux and element in the reservoir.

        elif isinstance(self.r2, Sink):
            self.r1.lio[
                self.fh.n] = self.outflux  # add the flux name direction/pair
            self.r1.lof.append(self.fh)  # add flux to the upstream reservoir
            self.__register_species__(
                self.r1,
                self.r2.sp)  # register flux and element in the reservoir.

        elif isinstance(self.r1, Sink):
            raise NameError(
                "The Sink must be specified as a destination (i.e., as second argument"
            )

        elif isinstance(self.r2, Source):
            raise NameError("The Source must be specified as first argument")

        else:  # this is a regular connection
            self.r1.lio[
                self.fh.n] = self.outflux  # add the flux name direction/pair
            self.r2.lio[
                self.fh.n] = self.influx  # add the flux name direction/pair`
            self.r1.lof.append(self.fh)  # add flux to the upstream reservoir
            self.r2.lof.append(self.fh)  # add flux to the downstream reservoir
            self.__register_species__(self.r1, self.r1.sp)
            self.__register_species__(self.r2, self.r2.sp)

    def __register_species__(self, r, sp) -> None:
        """ Add flux to the correct element dictionary"""
        # test if element key is present in reservoir
        if sp.en in r.doe:
            # add flux handle to dictionary list
            r.doe[sp.en].append(self.fh)
        else:  # add key and first list value
            r.doe[sp.en] = [self.fh]

    def register_process(self) -> None:
        """ Register all flux related processes"""

        from copy import copy
        # first test if we have a signal in the list. If so,
        # remove signal and replace with process

        p_copy = copy(self.pl)
        for p in p_copy:
            if isinstance(p, Signal):
                print(f"removing Signal {p.n}")
                self.pl.remove(p)
                if p.ty == "addition":
                    # create AddSignal Process object
                    n = AddSignal(name=p.n + "_addition_process",
                                  reservoir=self.r,
                                  flux=self.fh,
                                  lt=p.data)
                    self.pl.append(n)
                    print(f"Adding new process {n.n}")
                else:
                    raise ValueError(f"Signal type {p.ty} is not defined")

        # nwo we can register everythig on pl
        for p in self.pl:
            print(f"Registering Process {p.n}")
            print(f"with reservoir {self.r.n} and flux {self.fh.n}")
            p.register(self.r, self.fh)

    def __set_process_type__(self) -> None:
        """ Deduce flux type based on the provided flux properties. The method returns the 
        flux handle, and the process handle(s).
        """
        from esbmtk import PassiveFlux, PassiveFlux_fixed_delta, VarDeltaOut, ScaleFlux, Fractionation
        from esbmtk import Source, Sink, Flux, RateConstant

        if isinstance(self.r1, Source):
            self.r = self.r2
        else:
            self.r = self.r1

        # set process name
        self.pn = self.r1.n + "_to_" + self.r2.n

        # set the flux type
        if "delta" in self.kwargs and "rate" in self.kwargs:
            pass  # static flux,
        elif "delta" in self.kwargs:
            self.__passivefluxfixeddelta__()  # variable flux with fixed delta
        elif "rate" in self.kwargs:
            self.__vardeltaout__()  # variable delta with fixed flux
        elif "scale" in self.kwargs:
            self.__scaleflux__()  # scaled variable flux with fixed delta
        elif "react_with" in self.kwargs:
            self.__reaction__()  # this flux will react with another flux
        elif "k" in self.kwargs:  # this flux uses a rate constant
            self.__rateconstant__()
        else:  # if neither are given -> default varflux type
            self.__passiveflux__()

        # Set optional flux processes
        if "alpha" in self.kwargs:  # isotope enrichment
            self.__alpha__()

        if "kvalue" in self.kwargs:
            self.__rateconstant__()  # flux depends on a rate constant

        if "type" in self.kwargs:
            if self.kwargs["type"] == "eq":  # equlibrium type connection
                self.__equilibrium__()

    def __passivefluxfixeddelta__(self) -> None:
        """ Just a wrapper to keep the if statement manageable
        """
        ph = PassiveFlux_fixed_delta(
            name=self.pn + "_Pfd",
            reservoir=self.r,
            flux=self.fh,
            delta=self.delta)  # initialize a passive flux process object
        self.pl.append(ph)

    def __vardeltaout__(self) -> None:
        """ Just a wrapper to keep the if statement manageable
        """
        ph = VarDeltaOut(name=self.pn + "_Pvdo",
                         reservoir=self.r,
                         flux=self.fh,
                         rate=self.kwargs["rate"])
        self.pl.append(ph)

    def __scaleflux__(self) -> None:
        """ Just a wrapper to keep the if statement manageable
        """
        if not isinstance(self.kwargs["ref"], Flux):
            raise ValueError("Scale reference must be a flux")

        ph = ScaleFlux(name=self.pn + "_PSF",
                       reservoir=self.r,
                       flux=self.fh,
                       scale=self.kwargs["scale"],
                       ref=self.kwargs["ref"])
        self.pl.append(ph)

    def __reaction__(self) -> None:
        """ Just a wrapper to keep the if statement manageable
        """
        if not isinstance(self.kwargs["react_with"], Flux):
            raise ValueError("Scale reference must be a flux")
        ph = Reaction(name=self.pn + "_RF",
                      reservoir=self.r,
                      flux=self.fh,
                      scale=self.kwargs["ratio"],
                      ref=self.kwargs["react_with"])
        # we need to make sure to remove the flux referenced by
        # react_with is removed from the list of fluxes in this
        # reservoir.
        self.r2.lof.remove(self.kwargs["react_with"])
        self.pl.append(ph)

    def __passiveflux__(self) -> None:
        """ Just a wrapper to keep the if statement manageable
        """
        ph = PassiveFlux(
            name=self.pn + "_PF", reservoir=self.r,
            flux=self.fh)  # initialize a passive flux process object
        self.pl.append(ph)  # add this process to the process list

    def __alpha__(self) -> None:
        """ Just a wrapper to keep the if statement manageable
        """
        ph = Fractionation(name=self.pn + "_Pa",
                           reservoir=self.r,
                           flux=self.fh,
                           alpha=self.kwargs["alpha"])
        self.pl.append(ph)  #

    def __rateconstant__(self) -> None:
        """ Add rate constant process"""
        if "rate" not in self.kwargs:
            raise ValueError(
                "The rate constant process requires that the flux rate is being set explicitly"
            )

        ph = RateConstant(name=self.pn + "_Pk",
                          reservoir=self.r,
                          flux=self.fh,
                          C0=self.C0,
                          kvalue=self.kvalue)
        self.pl.append(ph)

    def __equilibrium__(self) -> None:
        """ Just a wrapper to keep the if statement manageable
        """
        r1 = self.r1  # left side reservoir 1
        r2 = self.r2  # left side reservoir 2
        r3 = self.kwargs["r3"]  # right side reservoir 1
        r4 = self.kwargs["r4"]  # right side reservoir 1
        kf = self.kwargs["kf"]  # k-value for the forward reaction
        kb = self.kwargs["kr"]  # k-value for the backwards reaction
        n = self.kwargs["steps"]  # number of steps to equilibration

        # we need to setup 4 fluxes with two processes, or
        # better yet can we call the process once and it will
        # then set all fluxes?  do processes need to be bound
        # to reservoirs? -> yes!  but we add this process
        # simply to r1 and it will do the right thing.
        ph = Equilibrium(pn + "_eq", self.fh, r1, r2, r3, r4, kf, kb, n)

        self.pl.append(ph)  # add this process to the process list

class Process(esbmtkBase):
    """This class defines template for process which acts on one or more
     reservoir flux combinations. To use it, you need to create an
     subclass which defines the actual process implementation in their
     call method. See 'PassiveFlux as example'
    """

    from typing import Dict
    from esbmtk import Reservoir, Flux
    
    def __init__(self, **kwargs :Dict[str, any]) -> None:
        """
          Create a new process object with a given process type and options
          """

        self.__defaultnames__()      # default kwargs names
        self.__initerrormessages__() # default error messages
        # update self.bem dict if necessary
        self.__validateandregister__(kwargs)  # initialize keyword values
        self.__postinit__()          # do some housekeeping

    def __postinit__(self) -> None:
        """ Do some housekeeping for the process class
          """
        from typing import Dict, List
        from esbmtk import Model, Reservoir, Flux

        # legacy name aliases
        self.n: str = self.name  # display name of species
        self.r: Reservoir = self.reservoir
        self.f: Flux = self.flux
        self.m: Model = self.r.sp.mo  # the model handle

        # Create a list of fluxes wich texclude the flux this process
        # will be acting upon
        self.fws :List[Flux] = self.r.lof.copy()
        self.fws.remove(self.f)  # remove this handle

        self.rm0 :float = self.r.m[0]  # the initial reservoir mass
        self.direction :Dict[Flux,int] = self.r.lio[self.f.n]
        

    def __defaultnames__(self) -> None:
        """Set up the default names and dicts for the process class. This
          allows us to extend these values without modifying the entire init process"""

        from typing import Dict
        from esbmtk import Reservoir, Flux
        from numbers import Number

        # provide a dict of known keywords and types
        self.lkk: Dict[str, any] = {
            "name": str,
            "reservoir": Reservoir,
            "flux": Flux,
            "rate": Number,
            "delta": Number,
            "lt": Flux,
            "alpha": Number,
            "scale": Number,
            "ref": Flux,
        }

        # provide a list of absolutely required keywords
        self.lrk: list[str] = ["name"]

        # list of default values if none provided
        self.lod: Dict[str, any] = {}

        # default type hints
        self.scale :t
        self.delta :Number
        self.alpha :Number
        

    def register(self, reservoir :Reservoir, flux :Flux) -> None:
        """Register the flux/reservoir pair we are acting upon, and register
          the process with the reservoir
          """

        import builtins
        # register the reservoir flux combination we are acting on
        self.f :Flux = flux
        self.r :Reservoir = reservoir
        # add this process to the list of processes acting on this reservoir
        reservoir.lop.append(self)
        flux.lop.append(self)

    def describe(self) -> None:
        """Print basic data about this process """
        print(f"\t\tProcess: {self.n}", end="")
        for key, value in self.kwargs.items():
            print(f", {key} = {value}", end="")

        print("")

    def show_figure(self, x, y) -> None:
        """ Apply the current process to the vector x, and show the result as y.
          The resulting figure will be automatically saved.

          Example:
               process_name.show_figure(x,y)
          """
        pass

class LookupTable(Process):
     """This process replaces the flux-values with values from a static
lookup table

     Example:

     LookupTable("name", upstream_reservoir_handle, lt=flux-object)

     where the flux-object contains the mass, li, hi, and delta values
     which will replace the current flux values.

     """
     from typing import Dict
     from esbmtk import Reservoir, Flux
     
     def __call__(self, r: Reservoir, i: int) -> None:
          """Here we replace the flux value with the value from the flux object 
          which we use as a lookup-table

          """
          self.m[i] :float  = self.lt.m[i]
          self.d[i] :float  = self.lt.d[i]
          self.l[i] :float = self.lt.l[i]
          self.h[i] :float = self.lt.h[i]

class AddSignal(Process):
    """This process adds values to the current flux based on the values provided by the sifnal object.
    This class is typically invoked through the connector object

     Example:

     AddSignal(name = "name",
               reservoir = upstream_reservoir_handle,
               flux = flux_to_act_upon,
               lt = flux with lookup values)

     where - the upstream reservoir is the reservoir the process belongs too
             the flux is the flux to act upon
             lt= contains the flux object we lookup from

    """
    from typing import Dict

    def __init__(self, **kwargs: Dict[str, any]) -> None:
        """
        Create a new process object with a given process type and options
        """
        from esbmtk import Flux

        # get default names and update list for this Process
        self.__defaultnames__()  # default kwargs names
        self.lrk.extend(["lt", "flux", "reservoir"])  # new required keywords

        self.__validateandregister__(kwargs)  # initialize keyword values
        self.__postinit__()  # do some housekeeping

    def __call__(self, r, i) -> None:
        """Each process is associated with a flux (self.f). Here we replace
          the flux value with the value from the signal object which
          we use as a lookup-table (self.lt)

        """
        # add signal mass to flux mass
        self.f.m[i] = self.f.m[i] + self.lt.m[i]
        # add signal delta to flux delta
        self.f.d[i] = self.f.d[i] + self.lt.d[i]

        self.f.l[i], self.f.h[i] = get_mass(self.f.m[i], self.f.d[i], r.sp.r)
        # signals may have zero mass, but may have a delta offset. Thus, we do not know
        # the masses for the light and heavy isotope. As such we have to calculate the masses
        # after we add the signal to a flux

class PassiveFlux(Process):
     """This process sets the output flux from a reservoir to be equal to
     the sum of input fluxes, so that the reservoir concentration does
     not change. Furthermore, the isotopic ratio of the output flux
     will be set equal to the isotopic ratio of the reservoir The init
     and register methods are inherited from the process class. The
     overall result can be scaled, i.e., in order to create a split flow etc.
     Example:

     PassiveFlux(name = "name",
                 reservoir = upstream_reservoir_handle
                 flux = flux handle)

     """
     from typing import Dict
     from esbmtk import Reservoir, Flux

     def __init__(self, **kwargs :Dict[str,any]) -> None:
          """ Initialize this Process """
          
         
          # get default names and update list for this Process
          self.__defaultnames__()  # default kwargs names
          self.lrk.extend(["reservoir", "flux"]) # new required keywords
        
          self.__validateandregister__(kwargs)  # initialize keyword values
          self.__postinit__()  # do some housekeeping
     
     def __call__(self,reservoir :Reservoir, i :int) -> None:
          """Here we re-balance the flux. This code will be called by the 
          apply_flux_modifier method of a reservoir which itself is
          called by the model execute method"""

          fm :float = 0  # the new flux mass 
          for f in self.fws:  # do sum of fluxes in this reservoir
               fm = fm + f.m[i] * self.direction
          
          # get new reservoir mass
          rm :float = reservoir.m[i-1] + (fm * self.m.dt *  self.direction)

          # get the difference to the desired mass
          # fm = (rm - self.rm0) / self.m.dt  * reservoir.lio[self.f.n]
          fm = abs(self.rm0 -rm) / self.m.dt

          # set isotope mass according to the reservoir delta
          self.f[i] = set_mass(fm,reservoir.d[i-1],reservoir.sp.r)

class PassiveFlux_fixed_delta(Process):
     """This process sets the output flux from a reservoir to be equal to
     the sum of input fluxes, so that the reservoir concentration does
     not change. However, the isotopic ratio of the output flux is set
     at a fixed value. The init and register methods are inherited
     from the process class. The overall result can be scaled, i.e.,
     in order to create a split flow etc.  Example:

     PassiveFlux_fixed_delta(name = "name",
                             reservoir = upstream_reservoir_handle,
                             flux handle,
                             delta = delta offset)

     """
     from typing import Dict, List
     from esbmtk import Reservoir, Flux

     def __init__(self, **kwargs :Dict[str, any]) -> None:
          """ Initialize this Process """

          from esbmtk import Reservoir, Flux
          # get default names and update list for this Process
          self.__defaultnames__()  # default kwargs names
          self.lrk.extend(["reservoir","delta", "flux"]) # new required keywords
        
          self.__validateandregister__(kwargs)  # initialize keyword values
          self.__postinit__()  # do some housekeeping

          # legacy names
          self.f :Flux = self.flux

          print("\nn *** Warning, you selected the PassiveFlux_fixed_delta method ***\n ")
          print(" This is not a particularly phyiscal process is this really what you want?\n")
          print(self.__doc__)
     
     def __call__(self, reservoir :Reservoir, i :int) -> None:
          """Here we re-balance the flux. This code will be called by the
          apply_flux_modifier method of a reservoir which itself is
          called by the model execute method"""

          from numpy import array
          from esbmtk import Flux
          from typing import Dict, List
          
          r :float = reservoir.sp.r # the isotope reference value

          varflux :Flux = self.f 
          flux_list :List[Flux] = reservoir.lof.copy()
          flux_list.remove(varflux)  # remove this handle

          # sum up the remaining fluxes
          newflux :float = 0
          for f in flux_list:
               newflux = newflux + f.m[i-1] * reservoir.lio[f.n]

          # set isotope mass according to keyword value
          self.f[i] = array(set_mass(newflux, self.delta, r))

class VarDeltaOut(Process):
     """Unlike a passive flux, this process sets the output flux from a
     reservoir to a fixed value, but the isotopic ratio of the output
     flux will be set equal to the isotopic ratio of the reservoir The
     init and register methods are inherited from the process
     class. The overall result can be scaled, i.e., in order to create
     a split flow etc.  Example:

     VarDeltaOut(name = "name",
                 reservoir = upstream_reservoir_handle,
                 flux = flux handle
                 rate = rate)

     """
     from esbmtk import Reservoir, Flux
     from typing import Dict

     def __init__(self, **kwargs :Dict[str, any]) -> None:
          """ Initialize this Process """
          
          # get default names and update list for this Process
          self.__defaultnames__()  # default kwargs names
          self.lrk.extend(["reservoir", "rate"]) # new required keywords
        
          self.__validateandregister__(kwargs)  # initialize keyword values
          self.__postinit__()  # do some housekeeping
     
     def __call__(self, reservoir:Reservoir ,i :int) -> None:
          """Here we re-balance the flux. This code will be called by the
          apply_flux_modifier method of a reservoir which itself is
          called by the model execute method"""

          # set flux according to keyword value
          self.f[i] = set_mass(self.rate,reservoir.d[i-1], reservoir.sp.r)

class ScaleFlux(Process):
    """This process scales the mass of a flux (m,l,h) relative to another
     flux but does not affect delta. The scale factor "scale" and flux
     reference must be present when the object is being initalized

     Example:
          ScaleFlux(name = "Name",
                    reservoir = upstream_reservoir_handle,
                    scale = 1
                    ref = flux we use for scale)

     """
    from esbmtk import Reservoir, Flux
    from typing import Dict

    def __init__(self, **kwargs: Dict[str, any]) -> None:
        """ Initialize this Process """
        # get default names and update list for this Process
        self.__defaultnames__()  # default kwargs names
        self.lrk.extend(["reservoir", "flux", "scale",
                         "ref"])  # new required keywords

        self.__validateandregister__(kwargs)  # initialize keyword values
        self.__postinit__()  # do some housekeeping

    def __call__(self, reservoir: Reservoir, i: int) -> None:
        """Apply the scale factor. This is typically done through the the
          model execute method.
          Note that this will use the mass of the reference object, but that we will set the 
          delta according to the reservoir (or the flux?)
          """
        self.f[i] = self.ref[i] * self.scale
        self.f[i] = set_mass(self.f.m[i], reservoir.d[i - 1], reservoir.sp.r)

class Reaction(ScaleFlux):
     """This process approximates the effect of a chemical reaction between
     two fluxes which belong to a differents species (e.g., S, and O).
     The flux belonging to the upstream reservoir will simply be
     scaled relative to the flux it reacts with. The scaling is given
     by the ratio argument. So this function is equivalent to the
     ScaleFlux class. It is up to the connector class (or the user) to
     ensure that the reference flux is removed from the reservoir list
     of fluxes (.lof) which will be used to sum all fluxes in the
     reservoir.

     Example:
          Reaction("Name",upstream_reservoir_handle,{"scale":1,"ref":flux_handle})

     """

class Fractionation(Process):
     """This process offsets the isotopic ratio of the flux by a given
        delta value. In other words, we add a fractionation factor

     Example:
          Fractionation(name = "Name",
                        reservoir = upstream_reservoir_handle,
                        flux = flux handle
                        alpha = 12)

     """
     from esbmtk import Reservoir, Flux
     from typing import Dict

     def __init__(self, **kwargs :Dict[str, any]) -> None:
          """ Initialize this Process """
           # get default names and update list for this Process
          self.__defaultnames__()  # default kwargs names
          self.lrk.extend(["reservoir", "flux", "alpha"]) # new required keywords
        
          self.__validateandregister__(kwargs)  # initialize keyword values
          self.__postinit__()  # do some housekeeping
     
     
     def __call__(self,reservoir :Reservoir, i :int) -> None: 
        
          from esbmtk import get_mass

          self.f.d[i] = self.f.d[i] + self.alpha # set the new delta
          # recalculate masses based on new delta
          self.f.l[i], self.f.h[i] = get_mass(self.f.m[i],
                                              self.f.d[i],
                                              self.f.sp.r)
          return

class Equilibrium():
     """This class creates the connections and initializes the processes
     which describe equilibration reactions between different
     species. Each species must exist as separate reservoir

     """
     def __init__(self, name, reactants, products, ratios, kvalues) -> None:
          """This class creates the connections and initializes the processes
          which describe equilibration reactions between different
          species. Each species must exist as separate reservoir

          Example:
          [Ca2+] [HCO_3-] <--> [CaCO3] + [H2]

          Equilibrium("Name", [r1, r2], [p3, p4], [1,1], [k1,k2,k3])

          where:

          [r1,r2] is a list of the reactants
          [p1,p2] is a list of the reaction products
          [1,1,1] is a list of the stoichiometric ratios (i.e., r1:r2)
          [k1,k2,k3] is a list of the equilibrium constants

          """
          import builtins
          
          self.n = name         # process name
          self.lor = reactants  # list of reservoirs with the reactants
          self.lop = products   # list of reservoirs with the products
          self.los = ratios     # list of the stoichiometric ratios
          self.lok = kvalues    # list of the equilibrium constants

          setattr(builtins,name,self) # register this process in the global name space

          # initialize the connections between ri, pi
          i = 0
          for r in self.lor:
               p = self.lop[i]
               # create flux between r and p
               # create processes and add to reservoirs
               i = i + 1

     # this needs to move to a process type?
     def __call__(self,reservoir,i) -> None:
          """ we don't really need to know the reservoir handle, but we keep it here so that the call 
          to the process class remains consistent with the other processes
          """
          r1 = self.r1     # left side reservoir 1
          r2 = self.r2     # left side reservoir 2
          r3 = self.kwargs["r3"]     # right side reservoir 1
          r4 = self.kwargs["r4"]     # right side reservoir 1
          kf = self.kwargs["kf"]     # k-value for the forward reaction
          kb = self.kwargs["kr"]     # k-value for the backwards reaction
          n  = self.kwargs["steps"]  # number of steps to equilibration
          c1 = r1.m/r1.v    # get concentration
          c2 = r2.m/r2.v    # get concentration
          c3 = r3.m/r3.v    # get concentration
          c4 = r4.m/r4.v    # get concentration

         
          fw = (-kf * c1 * c2)/n  # calculate the forward fluxes
          fb = ( kb * c3 * c4)/n  # calculate the backward fluxes

          # Assign fluxes. How do we do that?

class RateConstant(Process):
    """This process scales the flux as a function of the upstream
     reservoir concentration C and a constant which describes the
     kvalue between the reservoir concentration and the flux scaling

     F = kvalue * (C/C0)

     where C denotes
     the concentration in the ustream reservoir, C0 denotes the baseline
     concentration and and m & k are constants
    

     Example:
          RateConstant(name = "Name",
                       reservoir= upstream_reservoir_handle,
                       flux = flux handle,
                       Kvalue =  0.00028,
                       C0 = 2 # reference_concentration
    )

    """
    from esbmtk import Reservoir, Flux
    from typing import Dict

    def __init__(self, **kwargs: Dict[str, any]) -> None:
        """ Initialize this Process """

        from esbmtk import Reservoir, Flux
        from numbers import Number

        # Note that self.lkk values also need to be added to the lkk
        # list of the connector object.

        # get default names and update list for this Process
        self.__defaultnames__()  # default kwargs names

        # update the allowed keywords
        self.lkk = {
            "kvalue": Number,
            "C0": Number,
            "name": str,
            "reservoir": Reservoir,
            "flux": Flux
        }

        # new required keywords
        self.lrk.extend(["reservoir", "kvalue", "C0"])

        # dict with default values if none provided
        # self.lod d

        self.__initerrormessages__()

        # add these terms to the known error messages
        self.bem.update({
            "kvalue": "a number",
            "reservoir": "Reservoir handle",
            "C0": "a number",
            "name": "a string value",
            "flux": "a flux handle",
        })

        # initialize keyword values
        self.__validateandregister__(kwargs)
        self.__postinit__()  # do some housekeeping

    def __call__(self, reservoir: Reservoir, i: int) -> None:
        """
          this will be called by the Model.run() method
          """
        scale: float = reservoir.c[i - 1] / self.C0 * self.kvalue
        self.f[i] = self.f[i] * scale

class Monod(Process):
     """This process scales the flux as a function of the upstream
     reservoir concentration using a Michaelis Menten type
     relationship

     F = a * F0 x C/(b+C)

     where F0 denotes the unscaled flux (i.e., at t=0), C denotes
     the concentration in the ustream reservoir, and a and b are
     constants.

     Example:
          Monod(name = "Name",
                reservoir upstream_reservoir_handle,
                a = ,
                b = )

     """
     from esbmtk import Reservoir, Flux
     from typing import Dict

     def __init__(self, **kwargs :Dict[str, any]) -> None:
          """ Initialize this Process """
           # get default names and update list for this Process
          self.__defaultnames__()  # default kwargs names
          self.lrk.extend(["reservoir", "a", "b"]) # new required keywords
        
          self.__validateandregister__(kwargs)  # initialize keyword values
          self.__postinit__()  # do some housekeeping
     
    
     def __call__(self, reservoir: Reservoir, i :int) -> None:
          """
          this willbe called by Model.execute apply_processes
          """
          from esbmtk import get_mass
          
          self.m[i] =set_mass(self.a * (self.f[0] * reservoir.c[i])/(self.b + reservoir.c[i]),
                              reservoir.d[i-1],
                              reservoir.sp.r)

class Hypsometry(esbmtkBase):
    """Sea level variations affect the size of the continental
shelves. This in turn affects a variety of biogeochemical
processes. As such, I here provide a class which handles sea-level
data, and provides shelf area estimates in return. The sealevel object
is instantiated with a signal object, which contains sealevel/time
data points along the modeling domain in meters relative to the modern
sealevel (positive is higher, negative is lower). These are converted
to shelf area, and other ocean area estimates (both in total area, as
well as percentage points. Optional arguments allow to specify the
depth of the shelf break (default = -250 mbsl), as well as the total
ocean area (3.61E14 m^2 in the modern ocean). The code will return
values for sealevel variations between -500 and + 500 meters.

    Example:
             Hypsometry(name = "Name"         # handle
                       data = s-handle       # data object handle
                       shelf_break = -250    # optional, defaults to -250
                       ocean_area  = 3.61E14 # optional, defaults to 3.61E14 m^2
                       satype = "static"     # default: static
                       omcutoff = -4500      # default -4500
    )

    if satype = static the shelf area will be computed relative to
    the modern shelf break. 

    if satype = dynamic, the shelf area is the interval between zero
    and 250 bsl, regardless of the modern shelf break

    if satype = burial, calculate the OM burial 

    The instance provides the following data

    self.carbon_flux = the normalized total carbon burial efficiency 
    self.cf_z    = carbon burial efficiency [4500 elements]
    self.cfn   = the normalized total carbon burial efficiency for SL = 0
    self.elevation = array  from -5000 to +500 [5500 elements]
    self.area = the normalized hypsographic area for self.elevation
    self.area_dz = the normalized hypsographic area per depth interval (1m)
    
    self.sa    = shelf area as function of sealevel and time in m^2
    self.sap   = shelf area in percent relative to z=0
    self.sbz0  = open ocean area up to the shelf break
    self.sa0   = shelf area for z=0

    Methods:

    Name.plot() will plot the current shelf area data
    Name.plotvsl() will plot the shelf area versus the sealevel data
    Name.save_data() will save Name.csv with the sealevel versus shelf area data

    """

    from esbmtk import ExternalData, Model
    from typing import Dict
    from nptyping import Float, NDArray

    def __init__(self, **kwargs: Dict[str, any]) -> None:
        """ Initialize this instance """

        # dict of all known keywords and their type
        self.lkk: Dict[str, any] = {
            "name": str,
            "data": ExternalData,
            "shelf_break": int,
            "ocean_area": float,
            "satype": str,
            "omcutoff": int,
        }

        # provide a list of absolutely required keywords
        self.lrk: list = ["name", "data"]

        # list of default values if none provided
        self.lod: Dict[str, any] = {
            "shelf_break": -250,
            "ocean_area": 3.61E14,
            "satype": "static",
            "omcutoff": -4500,
        }

        # provide a dictionary entry for a keyword specific error message
        # see esbmtkBase.__initerrormessages__()
        self.__initerrormessages__()
        self.bem.update({
            "ocean_area": "a number",
            "shelf_break": "an integer number",
            "omcutoff": "an integer number",
            "satype": "'static/dynamic/burial'",
        })

        self.__validateandregister__(kwargs)  # initialize keyword values

        # set variables
        self.sa: [NDArray] = self.data.model.time * 0
        self.time: [NDArray] = self.data.model.time
        self.bvsa: [NDArray] = []  # OM burial versus area

        # Test if data object is of correct resolution
        if (len(self.data.x) != len(self.data.model.time)):
            self.data.interpolate()
            print(f"\n Warning, sealevel data had to be interpolated")

        # set up hypsometry data
        self.__calc_area__()
        a = self.__carbon_burial_integral__(0)
        a = self.__S_fractionation__integral__(0)
        self.cfn: float = self.carbon_flux  # get the C flux for SL =0
        self.cf_zn = self.cf_dz / self.cfn  # normalize the data 
        #self.carbon_flux = self.carbon_flux / self.cfn

        # Calculate shelf area as function of the input data
        if self.satype == "static":
            self.__get_static_shelf_area__()
        elif self.satype == "dynamic":
            self.__get_dynamic_shelf_area__()
        elif self.satype == "burial":
            pass
            # print(self.__carbon_burial_integral__(0.0))
        else:
            raise ValueError(f"Wrong value for satype!"
)
        # calculate relative shelf area change in percent
        # self.sap :[NDArray] = self.sa / self.sa0

    def __calc_area__(self) -> None:
        """Calculate the normalized hypsometric area as a function of
        elevation after Bjerrum et al. 2006: modeling organic carbon
        burial doi:10.1029/2005GC001032 Note that Bjerrums equation
        assume that z in km, not meter We will also calculate teh
        first derivative, which is the area per depth interval """

        import numpy as np
        from nptyping import NDArray, Float, Int
        import matplotlib.pyplot as plt

        start: int = self.omcutoff - 500  # -5000 below m sl
        stop: int = 500  # 500m above sl

        # the current 0 elevatio is thus at i = 5000
        self.elevation: [NDArray, Int] = np.arange(start, stop, 1)

        # calculate the deep data. Note, z has to be in km
        z: [NDArray, Int] = self.elevation[self.elevation <= -1000] * 1E-3
        ps1: float = 0.020
        ps2: float = 0.103
        ps3: float = 0.219
        ps4: float = 1.016
        deep: [NDArray, Float] = (ps1 * z**3 + ps2 * z**2 + ps3 * z + ps4)

        # calculate the shallow data
        z: [NDArray, Int] = self.elevation[self.elevation > -1000] * 1E-3
        ps1: float = 0.307
        ps2: float = 0.624
        ps3: float = 0.430
        ps4: float = 0.99295  # manually adjusted so that both curves meet
        shallow: [NDArray, Float] = (ps1 * z**3 + ps2 * z**2 + ps3 * z + ps4)
        # combine both into single vector which contains the normalized area
        # from -5000 to 500 m sl with indices running fromm 0 to 5500
        self.area = np.append(deep, shallow)

        # calculate the first derivative which provides us with the area per depth
        # interval
        self.area_dz: [NDArray, Float] = self.area[1:] - self.area[:-1]

    def __carbon_burial_integral__(self, sl: float) -> float:
        """Calculate the normalized organic matter burial efficiency as
        integral from deep sea to shelf. We use an artifical deep sea
        carbon burial cutoff of -4500 meters This is done by
        calculating burial efficency beta as a function of z, and multiply
        it with the area at a given depth

        sum beta(z)* area/dz * np.exp(z / 700)

        """

        import numpy as np
        from nptyping import NDArray, Float, Int
       
        b1: float = 0.411
        b2: float = 0.15
        start: int = self.omcutoff
        stop: int = 0

        # adjust z values for sealevel. The resulting vector will alwasy have 4500
        # elements
        z: [NDArray, Int] = np.arange(start + sl, stop + sl, 1)

        #Parametrization of the organic matter burial efficiency. This
        #follows Bjerrum et al 2006, although some of his parameters had to be tweaked
        self.beta: [NDArray, Float] = b1 * np.exp(z * 1E-3) + b2

        # the first derivative of the hypsomentric data has beeen computed from
        # -5000 to + 500 msl. So we only consider a subset here
        # the current 0 mark is at i = 5000 see  __calc_area__
        #area_dz :[NDArray, Float] = self.area_dz[500+sl:5000+sl]
        # carbon flux as a function of z
        self.cf_z: [ NDArray, Float] = self.beta * np.exp(z/700)

        # carbon flux per depth interval
        self.cf_dz = self.cf_z * self.area_dz[500 + sl:5000 + sl]
        # integrate the carbon flux
        self.carbon_flux: float = np.sum(self.cf_dz)
        return self.carbon_flux

    def __S_fractionation__integral__(self,sl: float) -> float:
        """ Calculate the average fractionation rate as a function of the dept dependent 
        carbon burial flux
        """
        import numpy as np
        from nptyping import NDArray, Float, Int
             
        # we can establish an linear relationship between C-burial flux and s-sitope fractionation
        b1 :float = 1.93766662
        b2 :float = 6.16408015
        b3 :float = 8.38951494
        co = 10 # cutoff in permil

        # get S-fractionation as function of carbon flux at a given depth
        self.alpha_z :[NDArray, Float] = b1 * np.exp(b2*self.cf_z) + b3
        self.alpha_z = 80 -  self.alpha_z
        # the above equation will overpredict, once the carbon flux
        # exceeded 0.55. As such we cap alpha
        self.alpha_z = np.where(self.alpha_z < co, co, self.alpha_z)

        # get S-fractionation per depth interval
        self.alpha_dz :[NDArray, Float] = self.alpha_z * self.area_dz[500 + sl:5000 + sl]
        # integrate        
        self.alpha: float = np.sum(self.alpha_dz)

        return self.alpha
       
    def save_data(self):
        # save data to csv file
        import pandas as pd
        df = pd.DataFrame(data=[self.cf_z]).T
        df.to_csv(self.name + ".csv")

    def __get_static_shelf_area__(self) -> None:
        """Calculate shelf area as a function of sealevel z relative to the
        modern shelf break depth (z0). z = 0 is for the modern ocean. z < is
        for sealeves which are below the moden ocean, z>0 is for sealevels
        which are above the modern sealevel. z should be meters
        """

        # get ocean area up to the shelf break
        self.sbz0: float = self.__ocean_area__(self.shelf_break)
        self.sa0 = self.__ocean_area__(0) - self.sbz0

        # loop over all sealevel data points, and calculate the area
        # at a given sealevel
        i = 0
        for e in self.data.y:
            self.sa[i] = self.__ocean_area__(e) - self.sbz0
            i = i + 1

        self.sa = self.sa * self.ocean_area

    def __get_dynamic_shelf_area__(self) -> None:
        """Calculate shelf area as the aera bewteen 0 mbsl and the depth given
        in shelf-break relative to the sealevel at time(i). I.e., the
        shelf break depth is dynamic
        """

        # get ocean area up to the shelf break at t=0 this data is
        # used in the init routine to normalize the change relative to the starting
        # sealevel - is the useful? 
        self.sbz0: float = self.__ocean_area__(self.shelf_break)
        self.sa0 = self.__ocean_area__(0) - self.sbz0

        # loop over all sealevel data points, and calculate the area
        # at a given sealevel
        i: int = 0
        for e in self.data.y:
            # get new shelf break depth
            sbd: float = e - self.shelf_break
            # calc the areal difference
            self.sa[i] = self.__ocean_area__(sbd) - self.__ocean_area__(e)
            i = i + 1

        # scale to absolute area
        self.sa = self.sa * self.ocean_area

    def __ocean_area__(self, z: float) -> float:
        """This returns the normalized ocean area up to this depth interval
        It would be more elegant to query self.area_dz which gives the
        ocean area for a given interval. We keep this notation to remain compatibel with earlier
        version.
        """

        za: float = self.area[4500 - z]
        return za

class SomeClass(esbmtkBase):
    """
    A ESBMTK class template
    
    Example:
             ShelfArea(name = "Name"         # handle
                       data = s-handle       # data object handle
                       shelf_break = -250    # optional, defaults to -250
                       ocean_area  = 3.61E14 # optional, defaults to 3.61E14 m^2)

    The instance provides the following data
    
    Name.sa    = shelf area as function of sealevel and time in m^2
    Name.sap   = shelf area in percent relative to z=0
    Name.sbz0  = open ocean area up to the shelf break
    Name.sa0   = shelf area for z=0

    Methods:

    Name.plot() will plot the current shelf area data
    Name.plotvsl() will plot the shelf area versus the sealevel data 
"""

    from esbmtk import ExternalData, Model
    from typing import Dict
    from nptyping import Float, NDArray

    def __init__(self, **kwargs: Dict[str, any]) -> None:
        """ Initialize this instance """

        # dict of all known keywords and their type
        self.lkk: Dict[str, any] = {
            "name": str,
            "data": ExternalData,
            "shelf_break" :int,
            "ocean_area": float,
        }

        # provide a list of absolutely required keywords
        self.lrk: list = ["name", "data"]
        # list of default values if none provided
        self.lod: Dict[str, any] = {"shelf_break":-250,
                                    "ocean_area": 3.61E14}

        # provide a dictionary entry for a keyword specific error message
        # see esbmtkBase.__initerrormessages__()
        self.__initerrormessages__()
        self.bem.update({"ocean_area"  : "a number",
                         "shelf_break" : "an integer number"})
        
        self.__validateandregister__(kwargs)  # initialize keyword values
