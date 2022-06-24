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
from __future__ import annotations
import typing as tp
from pandas import DataFrame
import time
from time import process_time
from numba.typed import List
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
import os
import psutil
import collections as col
from . import ureg, Q_

from .esbmtk_base import esbmtkBase

from .utility_functions import (
    show_data,
    plot_geometry,
    get_plot_layout,
    find_matching_strings,
)
from .solver import (
    get_l_mass,
    execute,
    execute_e,
    get_delta_h,
)


if tp.TYPE_CHECKING:
    from .extended_classes import GasReservoir, ExternalData, DataField
    from .connections import Connection
    from .processes import Process


class Model(esbmtkBase):
    """This class is used to specify a new model

    Example:

          esbmtkModel(name   =  "Test_Model",
                      start    = "0 yrs",    # optional: start time
                      stop     = "10000 yrs", # end time
                      timestep = "2 yrs",    # as a string "2 yrs"
                      offset = "0 yrs",    # optional: time offset for plot
                      mass_unit = "mol",   #required
                      volume_unit = "l", #required
                      time_label = optional, defaults to "Time"
                      display_precision = optional, defaults to 0.01,
                      m_type = "mass_only/both"
                      plot_style = 'default', optional defaults to 'default'
                      number_of_datapoints = optional, see below
                      step_limit = optional, see below
                      register = 'local', see below
                      save_flux_data = False, see below
                      ideal_water = False
                      use_ode = False
                    )

    ref_time: will offset the time axis by the specified amount, when
                 plotting the data, .i.e., the model time runs from to
                 100, but you want to plot data as if where from 2000
                 to 2100, you would specify a value of 2000. This is
                 for display purposes only, and does not affect the
                 model. Care must be taken that any external data
                 references the model time domain, and not the display
                 time.

    display precision: affects the on-screen display of data. It is
                       also cutoff for the graphicak output. I.e., the
                       interval f the y-axis will not be smaller than
                       the display_precision.

    m_type: enables or disables isotope calculation for the entire
            model.  The default value is "Not set" in this case
            isotopes will only be calculaten for reservoirs which set
            the isotope keyword. 'mass_only' 'both' will override the
            reservoir settings

    register = local/None. If set to 'None' all objects are registered
               in the global namespace the default setting is local,
               i.e. all objects are registered in the model namespace.

    save_flux_data: Normally, flux data is not stored. Set this to True
               for debugging puposes. Note, Fluxes with signals are always
               stored. You can also enable this option for inidividual
               connections (fluxes).

    get_delta_values: Compute delta values as postprocessing step.

    All of the above keyword values are available as variables with
    Model_Name.keyword

    The user facing methods of the model class are
       - Model_Name.info()
       - Model_Name.save_data()
       - Model_Name.plot_data()
       - Model_Name.plot_reservoirs() takes an optional filename as argument
       - Model_Name.plot([sb.DIC, sb.TA]) plot any object in the list
       - Model_Name.save_state() Save the model state
       - Model_name.read_state() Initialize with a previous model state
       - Model_Name.run(), there are 2 optional arguments here, solver="hybrid"
         and solver = "numba". Both involve a 3 to 5 second overhead. The hybrid
         solver is compatible with all connection types, and about 3 times faster
         than the  regular solver. The numba solver is about 10 faster, but currently
         only supports a limited set of connection types.
       - Model_Name.list_species()
       - Model_name.flux_summary()
       - Model_Name.connection_summary()


    User facing variable are Model_Name.time which contains the time
    axis.

    Optional, you can provide the element keyword which will setup a
    default set of Species for Carbon and Sulfur. In this case, there
    is no need to define elements or species. The argument to this
    keyword are either "Carbon", or "Sulfur" or both as a list
    ["Carbon", "Sulfur"].


    Dealing with large datasets:
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    1) Limiting the size of the data which is being saved with save_data()

         number_of_datapoints = 100 will only write every nth data point to file.
                                where n = timesteps/ number_of_datapoints

         this defaults to 1000 until set explicitly.

    2) Reducing the memory footprint

    Models with a long runtime can easily exceed the available
    computer memory, as much if it is goobled up storing the model
    results. In this case, one can set the optional parameter

       step_limit = 1E6

    The above will limit the total number of iterations to 1E6, then
    save the data up to this point, and then restart the
    model. Subsequent results will be appended to the results.

    Caveat Emptor: If your model uses a signal instance, all signal
    data must fit into a single iteration set. At present, there is no
    support for signals which extend beyond the step_limit.

    In order to prevent the creation of massive datafiles, number_of_datapoints
    defaults to 1000. Modify as needed.

    """

    def __init__(self, **kwargs: dict[any, any]) -> None:
        """Init Sequence"""

        # from . import ureg, Q_

        self.defaults: dict[str, list[any, tuple]] = {
            "name": ["M", (str)],
            "start": ["0 yrs", (str, Q_)],
            "stop": ["None", (str, Q_)],
            "offset": ["0 yrs", (str, Q_)],
            "timestep": ["None", (str, Q_)],
            "element": ["None", (str, list)],
            "mass_unit": ["mol", (str, Q_)],
            "volume_unit": ["m**3", (str, Q_)],
            "concentration_unit": ["mol/kg", (str)],
            "time_label": ["Years", (str)],
            "display_precision": [0.01, (float)],
            "plot_style": ["default", (str)],
            "m_type": ["Not Set", (str)],
            "number_of_datapoints": [1000, (int)],
            "step_limit": [1e9, (int, float, str)],
            "register": ["local", (str)],
            "save_flux_data": [False, (bool)],
            "full_name": ["None", (str)],
            "parent": ["None", (str)],
            "isotopes": [False, (bool)],
            "debug": [False, (bool)],
            "ideal_water": [True, (bool)],
            "use_ode": [False, (bool)],
        }

        # provide a list of absolutely required keywords
        self.lrk: list[str] = [
            "name",
            "stop",
            "timestep",
            "mass_unit",
            "volume_unit",
            "concentration_unit",
        ]
        self.__initialize_keyword_variables__(kwargs)

        # self.__validateandregister__(kwargs)  # initialize keyword values

        # empty list which will hold all reservoir references
        self.lmo: list = []
        self.lmo2: list = []
        self.dmo: dict = {}  # dict of all model objects. useful for name lookups

        # start a log file
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        fn: str = f"{kwargs['name']}.log"
        logging.basicConfig(filename=fn, filemode="w", level=logging.WARN)
        self.__register_name_new__()

        self.lor: list = []
        self.lic: list = []  # list of all reserevoir type objects
        # empty list which will hold all connector references
        self.loc: set = set()  # set with connection handles
        self.lel: list = []  # list which will hold all element references
        self.led: list[ExternalData] = []  # all external data objects
        self.lsp: list = []  # list which will hold all species references
        self.lop: list = []  # set of flux processes
        self.lpc_f: list = []  # list of external functions affecting fluxes
        # list of external functions affecting virtual reservoirs
        self.lpc_r: list = []
        # list of virtual reservoirs
        self.lvr: list = []
        # optional keywords for use in the connector class
        self.olkk: list = []
        # list of objects which require a delayed initialize
        self.lto: list = []
        # list of datafield objects
        self.ldf: list = []
        # list of signals
        self.los: list = []
        self.first_start = True  # keep track of repeated solver calls
        self.lof: list = []  # list of fluxes

        # Parse the strings which contain unit information and convert
        # into model base units For this we setup 3 variables which define
        if not (
            self.concentration_unit == "mol/kg"
            or self.concentration_unit == "mol/l"
            or self.concentration_unit == "mol/liter"
        ):
            raise ValueError(
                f"{self.concentration_unit} must be either mol/l or mol/kg"
            )

        self.l_unit = ureg.meter  # the length unit
        self.t_unit = Q_(self.timestep).units  # the time unit
        self.d_unit = Q_(self.stop).units  # display time units
        self.m_unit = Q_(self.mass_unit).units  # the mass unit
        self.c_unit = Q_(self.concentration_unit).units  # the mass unit
        self.v_unit = Q_(self.volume_unit).units  # the volume unit
        # the concentration unit (mass/volume)

        self.f_unit = self.m_unit / self.t_unit  # the flux unit (mass/time)
        self.r_unit = self.v_unit / self.t_unit  # flux as volume/time
        # this is now defined in __init__.py
        # ureg.define('Sverdrup = 1e6 * meter **3 / second = Sv = Sverdrups')

        # legacy variable names

        self.start = self.ensure_q(self.start).to(self.t_unit).magnitude
        self.stop = self.ensure_q(self.stop).to(self.t_unit).magnitude
        self.offset = self.ensure_q(self.offset).to(self.t_unit).magnitude
        # self.start = self.start + self.offset
        # self.stop = self.stop + self.offset

        self.bu = self.t_unit
        self.base_unit = self.t_unit
        self.dt = Q_(self.timestep).magnitude
        self.tu = str(self.bu)  # needs to be a string
        self.n = self.name
        self.mo = self.name
        self.model = self
        self.plot_style: list = [self.plot_style]

        self.xl = f"Time [{self.bu}]"  # time axis label
        self.length = int(abs(self.stop - self.start))
        self.steps = int(abs(round(self.length / self.dt)))

        if self.steps < self.number_of_datapoints:
            self.number_of_datapoints = self.steps

        self.time = (np.arange(self.steps) * self.dt) + self.start
        self.timec = np.empty(0)
        self.state = 0

        # calculate stride
        self.stride = int(self.steps / self.number_of_datapoints)
        self.reset_stride = self.stride

        if self.step_limit == "None":
            self.number_of_solving_iterations: int = 0
        elif self.step_limit > self.steps:
            self.number_of_solving_iterations: int = 0
            self.step_limit = "None"
        else:
            self.step_limit = int(self.step_limit)
            self.number_of_solving_iterations = int(round(self.steps / self.step_limit))
            self.reset_stride = int(round(self.steps / self.number_of_datapoints))
            self.steps = self.step_limit
            self.time = (np.arange(self.steps) * self.dt) + self.start

        # set_printoptions(precision=self.display_precision)

        from esbmtk import species_definitions, hypsometry

        if "element" in self.kwargs:
            if isinstance(self.kwargs["element"], list):
                element_list = self.kwargs["element"]
            else:
                element_list = [self.kwargs["element"]]

            # register elements and species with model
            for e in element_list:
                # get function handle
                fh = getattr(species_definitions, e)
                fh(self)  # register element with model
                # get element handle
                eh = getattr(self, e)
                # register species with model
                eh.__register_species_with_model__()

        warranty = (
            f"\n"
            f"ESBMTK  Copyright (C) 2020  Ulrich G.Wortmann\n"
            f"This program comes with ABSOLUTELY NO WARRANTY\n"
            f"For details see the LICENSE file\n"
            f"This is free software, and you are welcome to redistribute it\n"
            f"under certain conditions; See the LICENSE file for details.\n"
        )
        print(warranty)

        # initialize the hypsometry class
        hypsometry(name="hyp", model=self, register=self)

    def info(self, **kwargs) -> None:
        """Show an overview of the object properties.
        Optional arguments are
        index  :int = 0 this will show data at the given index
        indent :int = 0 indentation

        """
        off: str = "  "
        # if "index" not in kwargs:
        #    index = 0
        # else:
        # index = kwargs["index"]

        if "indent" not in kwargs:
            indent = 0
            ind = ""
        else:
            indent = kwargs["indent"]
            ind = " " * indent

        # print basic data bout this object
        print(self)

        # list elements
        print("Currently defined elements and their species:")
        for e in self.lel:
            print(f"{ind}{e}")
            print(f"{off} Defined Species:")
            for s in e.lsp:
                print(f"{off}{off}{ind}{s.n}")

    def save_state(self) -> None:
        """Save model state. Similar to save data, but only saves the last 10
        time-steps

        """

        start: int = -10
        stop: int = -1
        stride: int = 1
        prefix: str = "state_"

        for r in self.lor:
            r.__write_data__(prefix, start, stop, stride, False, "state")

        for r in self.lvr:
            r.__write_data__(prefix, start, stop, stride, False, "state")

    def save_data(self, **kwargs) -> None:
        """Save the model results to a CSV file. Each reservoir will have
        their own CSV file

        Optional arguments:
        stride = int  # every nth element
        start = int   # start index
        stop = int    # end index
        append = True/False #

        """

        for k, v in kwargs.items():
            if not isinstance(v, int):
                print(f"{k} must be an integer number")
                raise ValueError(f"{k} must be an integer number")

        if "stride" in kwargs:
            stride = kwargs["stride"]
        else:
            stride = self.stride

        if "start" in kwargs:
            start = kwargs["start"]
        else:
            start = 0

        if "stop" in kwargs:
            stop = kwargs["stop"]
        else:
            stop = self.steps

        if "append" in kwargs:
            append = kwargs["append"]
        else:
            append = False

        prefix = ""

        print(f"start = {start}, stop = {stop}, stride={stride}, append ={append}")
        for r in self.lor:
            # print(f"R = {r.full_name}")
            # print(f"start = {start}, stop = {stop}, stride={stride}, append ={append}")
            r.__write_data__(prefix, start, stop, stride, append, "data")

        # save data fields
        # for r in self.ldf:
        #     r.__write_data__(prefix, start, stop, stride, append)
        print("Writing virtual reservoir data")
        for r in self.lvr:
            r.__write_data__(prefix, start, stop, stride, append, "data")
        print("done writing")

    def restart(self):
        """Restart the model with result of the last run.
        This is useful for long runs which otherwise would used
        to much memory

        """

        for r in self.lor:
            r.__reset_state__()
            for f in r.lof:
                f.__reset_state__()

        for r in self.lvr:
            r.__reset_state__()

        # print(f"len of time {len(self.time)}, stride = {self.stride}")
        # print(f"len of time with stride {len(self.time[0 : -2 : self.stride])}")
        self.timec = np.append(self.timec, self.time[0 : -2 : self.stride])
        t = int(round((self.stop - self.start) * self.dt))
        self.start = int(round((self.stop * self.dt)))
        self.stop = self.start + t
        self.time = (np.arange(self.steps) * self.dt) + self.start
        print(f"new start = {self.start}, new stop = {self.stop}")
        print(f"time[0] = {self.time[0]} time[-1] = {self.time[-1]}")
        # print(f"len of timec {len(self.timec)}")
        # self.time = (arange(self.steps) * self.dt) + self.start

    def read_state(self):
        """This will initialize the model with the result of a previous model
        run.  For this to work, you will need issue a
        Model.save_state() command at then end of a model run. This
        will create the necessary data files to initialize a
        subsequent model run.

        """

        from esbmtk import Reservoir, GasReservoir

        for r in self.lor:
            if isinstance(r, (Reservoir, GasReservoir)):
                # print(f" reading from {r.full_name}")
                r.__read_state__("state")

        for r in self.lvr:
            # print(f"lvr  reading from {r.full_name}")
            r.__read_state__("state")

    def merge_temp_results(self):
        """Replace the datafields which were used for an individual iteration
        with the data we saved from the previous iterations

        """

        self.time = self.timec
        for r in self.lor:
            r.__merge_temp_results__()
            for f in r.lof:
                f.__merge_temp_results__()

        for r in self.lvr:
            r.__merge_temp_results__()

    def plot(self, pl: list = [], **kwargs) -> None:
        """Plot all objects specified in pl

        M.plot([sb.PO4, sb.DIC],fn=test.pdf)

        fn is optional
        """

        if "fn" in kwargs:
            filename = kwargs["fn"]
        else:
            filename = f"{self.n}.pdf"

        noo: int = len(pl)
        size, geo = plot_geometry(noo)  # adjust layout
        fig, ax = plt.subplots(geo[0], geo[1])  # row, col
        axs = [[], []]

        """ The shape of the ax value of subplots depends on the figure
        geometry. So we need to ensure we are dealing with a 2-D array
        """
        if geo[0] == 1 and geo[1] == 1:  # row=1, col=1 only one window
            axs[0][0] = ax
        elif geo[0] > 1 and geo[1] == 1:  # mutiple rows, one column
            for i in range(geo[0]):
                axs[0].append(ax[i])
        elif geo[0] == 1 and geo[1] > 1:  # 1 row, multiple columns
            for i in range(geo[1]):
                axs[1].append(ax[i])
        else:
            axs = ax  # mutiple rows and mutiple columns

        # ste plot parameters
        plt.style.use(self.plot_style)
        fig.canvas.manager.set_window_title(f"{self.n} Reservoirs")
        fig.set_size_inches(size)

        i = 0  # loop over objects
        for c in range(geo[0]):  # rows
            for r in range(geo[1]):  # columns
                if i < noo:
                    pl[i].__plot__(self, axs[r][c])
                    axs[r][c].set_title(pl[i].full_name)
                    i = i + 1
                else:
                    axs[r][c].remove()

        fig.subplots_adjust(top=0.88)
        fig.tight_layout()
        plt.show(block=False)  # create the plot windows
        fig.savefig(filename)

    def run(self, **kwargs) -> None:
        """Loop over the time vector, and for each time step, calculate the
        fluxes for each reservoir
        """

        # this has nothing todo with self.time below!
        wts = time.time()
        start: float = process_time()
        # new: np.ndarray = np.zeros(4)

        # put direction dictionary into a list
        for r in self.lor:  # loop over reservoirs
            r.lodir = []
            for f in r.lof:  # loop over fluxes
                a = r.lio[f]
                r.lodir.append(a)

        # take care of objects which require a delayed init
        for o in self.lto:
            o.__delayed_init__()

        if "solver" not in kwargs:
            solver = "python"
        else:
            solver = kwargs["solver"]

        self.solver = solver
        if self.number_of_solving_iterations > 0:

            for i in range(self.number_of_solving_iterations):
                print(
                    f"\n Iteration {i+1} out of {self.number_of_solving_iterations}\n"
                )
                self.__run_solver__(solver)

                print(f"Restarting model")
                self.restart()

            print("Merge results")
            self.merge_temp_results()
            self.steps = self.number_of_datapoints
            # after merging, the model steps = number_of_datapoints
        # print("Saving data")
        # self.save_data(start=0, stop=self.number_of_datapoints, stride=1)
        # print("Done Saving")
        else:
            self.__run_solver__(solver, kwargs)

        # flag that the model has executed
        self.state = 1

        duration: float = process_time() - start
        wcd = time.time() - wts
        print(f"\n Execution took {duration:.2e} cpu seconds, wt = {wcd:.2e}\n")

        process = psutil.Process(os.getpid())
        print(f"This run used {process.memory_info().rss/1e9:.2f} Gbytes of memory \n")

    def get_delta_values(self):
        """Calculate the isotope ratios in the usual delta notation"""

        for r in self.lor:
            if r.isotopes:
                r.d = get_delta_h(r)

        # for vr in self.lvr:
        #     vr.d = get_delta_h(vr)

        # for f in self.lof:
        #     if f.save_flux_data:
        #         f.d = get_delta_h(f)

    def sub_sample_data(self):
        """Subsample the data. No need to save 100k lines of data You need to
        do this _after_ saving the state, but before plotting and
        saving the data

        """

        for r in self.lor:
            r.__sub_sample_data__()

        for vr in self.lvr:
            vr.__sub_sample_data__()

        for f in self.lof:
            f.__sub_sample_data__()

        stride = int(len(self.time) / self.number_of_datapoints)
        self.time = self.time[2:-2:stride]

    def __run_solver__(self, solver: str, kwargs: dict) -> None:
        from .ODEINT_Solver import run_solver

        if solver == "numba":
            execute_e(
                self,
                self.lop,
                self.lor,
                self.lpc_f,
                self.lpc_r,
            )
        elif solver == "odeint":
            run_solver(self)
        elif solver == "ode_uli":
            self.ode_uli(kwargs)
        elif solver == "python":
            execute(self.time, self.lop, self.lor, self.lpc_f, self.lpc_r)
        else:
            raise ValueError(
                f"Solver={solver} is unkknown, use 'python/numba/odeint/ode_uli'"
            )

    def ode_uli(self, kwargs):
        """Use the ode solver based on Uli's approach"""
        from esbmtk import Q_, write_equations_2, get_initial_conditions
        from scipy.integrate import odeint, solve_ivp
        import sys
        import pathlib as pl

        # build equation file
        R, icl, cpl, ipl = get_initial_conditions(self)
        self.R = R
        self.icl = icl
        self.cpl = cpl
        self.ipl = ipl

        # write equation system
        eqs_file = write_equations_2(self, R, icl, cpl, ipl)
        print(f"loc = {eqs_file.resolve()}")

        # ensure that cwd is in the load path. Required for windows
        cwd: pl.Path = pl.Path.cwd()
        sys.path.append(cwd)

        # import equation system
        from equations import setup_ode

        ode_system = setup_ode(self)  # create ode system instance
        self.ode_system = ode_system

        if "method" in kwargs:
            method = kwargs["method"]
        else:
            method = "RK23"

        if "stype" in kwargs:
            stype = kwargs["stype"]
        else:
            stype = "solve_ivp"

        if stype == "solve_ivp":
            results = solve_ivp(
                ode_system.eqs,
                (self.time[0], self.time[-1]),
                R,
                args=(self,),
                method=method,
                # t_eval=self.time,
                atol=1e-12,
                first_step=Q_("1 hour").to(self.t_unit).magnitude,
                # dense_output=True,
                # max_step=1,
            )

            # interpolate signals into the ode time domain
            # must be done before changing model time domain
            for s in self.los:
                s.data.m = np.interp(
                    results.t,
                    self.time,
                    s.data.m,
                )

            # interpolate external data into ode time domain
            # must be done before changing model time domain
            # for ed in self.led:
            #     ed.y = np.interp(results.t, ed.x, ed.y)

            # assign results to the esbmtk variables
            for i, r in enumerate(icl):
                r.c = results.y[i]
            self.time = results.t

            # interpolate intermediate results to match the model
            # time scale. This only applies to data in virtual
            # data fields
            for gf in self.lpc_r:
                # get cs instance handle
                cs = getattr(gf.register, "cs")
                for k, v in cs.vr_datafields.items():
                    od = getattr(cs, k)  # get ode data
                    od = np.interp(
                        self.time,
                        self.ode_system.t,
                        od[0 : self.ode_system.i],
                    )
                    setattr(cs, k, od)

        else:
            results = odeint(ode_system.eqs, R, t=self.time, args=(self,), tfirst=True)
            # assign results
            for i, r in enumerate(icl):
                r.c = results[:, i]

        self.results = results

    def __step_process__(self, r: Reservoir, i: int) -> None:
        """For debugging. Provide reservoir and step number,"""
        for p in r.lop:  # loop over reservoir processes
            print(f"{p.n}")
            p(r, i)  # update fluxes

    def __step_update_reservoir__(self, r: Reservoir, i: int) -> None:
        """For debugging. Provide reservoir and step number,"""
        flux_list = r.lof
        # new = sum_fluxes(flux_list,r,i) # integrate all fluxes in self.lof

        ms = ls = hs = 0
        for f in flux_list:  # do sum of fluxes in this reservoir
            direction = r.lio[f]
            ms = ms + f.m[i] * direction  # current flux and direction
            ls = ls + f.l[i] * direction  # current flux and direction
            hs = hs + f.h[i] * direction  # current flux and direction

        new = np.array([ms, ls, hs])
        new = new * r.mo.dt  # get flux / timestep
        new = new + r[i - 1]  # add to data from last time step
        # new = new * (new > 0)  # set negative values to zero
        r[i] = new  # update reservoir data

    def list_species(self):
        """List all  defined species."""
        for e in self.lel:
            print(f"{e.n}")
            e.list_species()

    def flux_summary(self, **kwargs: dict) -> tuple:
        """Show a summary of all model fluxes

        Optional parameters:

        filter_by :str = filter on flux name or part of flux name
                         words separated by blanks act as additional
                         conditions, i.e., all words must occur in a given name

        return_list: bool = False, if True return a list of fluxes matching the filter_by string.

        exclude:str = exclude all results matching this string

        Example:

              names = M.flux_summary(filter_by="POP A_sb", return_list=True)

        """

        fby = ""
        rl: list = []
        exclude: str = ""
        check_exlusion: bool = False

        if "exclude" in kwargs:
            exclude = kwargs["exclude"]
            check_exlusion = True

        if "filter_by" in kwargs:
            fby: list = kwargs["filter_by"].split(" ")

        if "filter" in kwargs:
            raise ValueError("use filter_by instead of filter")

        if "return_list" in kwargs:
            return_list = True
        else:
            return_list = False
            print(f"\n --- Flux Summary -- filtered by {fby}\n")

        for f in self.lof:  # loop over flux list

            if find_matching_strings(f.full_name, fby):
                if check_exlusion:
                    if exclude not in f.full_name:
                        rl.append(f)
                        if not return_list:
                            print(f"{f.full_name}")
                else:
                    rl.append(f)
                    if not return_list:
                        print(f"{f.full_name}")

        if not return_list:
            rl = None

        return rl

    def connection_summary(self, **kwargs: dict) -> None:
        """Show a summary of all connections

        Optional parameters:

        filter_by :str = filter on flux name or part of flux name
                         words separated by blanks act as additional conditions
                         i.e., all words must occur in a given name

        """

        if "filter_by" in kwargs:
            fby: list = kwargs["filter_by"].split(" ")
        else:
            fby: bool = False

        if "filter" in kwargs:
            raise ValueError("use filter_by instead of filter")

        print(f"fby = {fby}")
        self.cg_list: list = []
        # extract all connection groups. Note that loc contains all connections
        # i.e., not connection groups.
        for c in list(self.loc):
            # if "." in c.full_name:
            # if c.register not in self.cg_list and c.register != "None":
            if c not in self.cg_list and c.register != "None":
                self.cg_list.append(c)
            else:  # this is a regular connnection
                self.cg_list.append(c)

        print(f"\n --- Connection Group Summary -- filtered by {fby}\n")
        print(f"       run the following command to see more details:\n")

        # test if all words of the fby list occur in c.full_name. If yes,

        for c in self.cg_list:
            if not fby:
                print(f"{c.full_name}.info()")
            else:
                if find_matching_strings(c.full_name, fby):
                    print(f"{c.full_name}.info()")

        print("")

    def clear(self):
        """delete all model objects"""

        for o in self.lmo:
            print(f"deleting {o}")
            del __builtins__[o]


class Element(esbmtkBase):
    """Each model, can have one or more elements.  This class sets
    element specific properties

    Example::

            Element(name      = "S "           # the element name
                    model     = Test_model     # the model handle
                    mass_unit =  "mol",        # base mass unit
                    li_label  =  "$^{32$S",    # Label of light isotope
                    hi_label  =  "$^{34}S",    # Label of heavy isotope
                    d_label   =  r"$\delta^{34}$S",  # Label for delta value
                    d_scale   =  "VCDT",       # Isotope scale
                    r         = 0.044162589,   # isotopic abundance ratio for element
                  )

    """

    # set element properties
    def __init__(self, **kwargs) -> any:
        """Initialize all instance variables"""
        self.defaults: dict[str, list[any, tuple]] = {
            "name": ["M", (str)],
            "model": ["None", (str, Model)],
            "register": ["None", (str, Model)],
            "full_name": ["None", (str)],
            "li_label": ["None", (str)],
            "hi_label": ["None", (str)],
            "d_label": ["None", (str)],
            "d_scale": ["None", (str)],
            "r": [1, (float, int)],
            "mass_unit": ["None", (str, Q_)],
            "parent": ["None", (str, Model)],
        }

        # provide a list of absolutely required keywords
        # provide a list of absolutely required keywords
        self.lrk: list = ["name", "model", "mass_unit"]
        self.__initialize_keyword_variables__(kwargs)

        self.parent = self.model
        # self.__initerrormessages__()
        # self.__validateandregister__(kwargs)  # initialize keyword values

        # legacy name aliases
        self.n: str = self.name  # display name of species
        self.mo: Model = self.model  # model handle
        self.mu: str = self.mass_unit  # display name of mass unit
        self.ln: str = self.li_label  # display name of light isotope
        self.hn: str = self.hi_label  # display name of heavy isotope
        self.dn: str = self.d_label  # display string for delta
        self.ds: str = self.d_scale  # display string for delta scale
        self.lsp: list = []  # list of species for this element.
        self.mo.lel.append(self)

        if self.mo.register == "local" and self.register == "None":
            self.register = self.mo

        self.__register_name_new__()

    def list_species(self) -> None:
        """List all species which are predefined for this element"""

        for e in self.lsp:
            print(e.n)

    def __register_species_with_model__(self) -> None:
        """Bit of  hack, but makes model code more readable"""

        for s in self.lsp:
            setattr(self.model, s.name, s)


class Species(esbmtkBase):
    """Each model, can have one or more species.  This class sets species
    specific properties

          Example::

                Species(name = "SO4",
                        element = S,
    )

    """

    # set species properties
    def __init__(self, **kwargs) -> None:
        """Initialize all instance variables"""

        from esbmtk import GasReservoir

        # provide a list of all known keywords
        self.defaults: dict[any, any] = {
            "name": ["None", (str)],
            "element": ["None", (Element, str)],
            "display_as": [kwargs["name"], (str)],
            "m_weight": [0, (int, float, str)],
            "register": ["None", (Model, Element, Reservoir, GasReservoir)],
            "parent": ["None", (Model, Element, Reservoir, GasReservoir)],
        }

        # provide a list of absolutely required keywords
        self.lrk = ["name", "element"]
        self.__initialize_keyword_variables__(kwargs)
        self.parent = self.register

        if "display_as" not in kwargs:
            self.display_as = self.name

        # legacy names
        self.n = self.name  # display name of species
        self.mu = self.element.mu  # display name of mass unit
        self.ln = self.element.ln  # display name of light isotope
        self.hn = self.element.hn  # display name of heavy isotope
        self.dn = self.element.dn  # display string for delta
        self.ds = self.element.ds  # display string for delta scale
        self.r = self.element.r  # ratio of isotope standard
        self.mo = self.element.mo  # model handle
        self.eh = self.element.n  # element name
        self.e = self.element  # element handle
        self.dsa = self.display_as  # the display string.

        # self.mo.lsp.append(self)   # register self on the list of model objects
        self.e.lsp.append(self)  # register this species with the element

        if self.mo.register == "local" and self.register == "None":
            self.register = self.mo
        self.__register_name_new__()


class ReservoirBase(esbmtkBase):
    """Base class for all Reservoir objects"""

    def __init__(self, **kwargs) -> None:

        raise NotImplementedError(
            "ReservoirBase should never be used. Use the derived classes"
        )

    def __set_legacy_names__(self, kwargs) -> None:
        """
        Move the below out of the way
        """

        from esbmtk import get_box_geometry_parameters

        self.lof: list[Flux] = []  # flux references
        self.led: list[ExternalData] = []  # all external data references
        self.lio: dict[str, int] = {}  # flux name:direction pairs
        self.lop: list[Process] = []  # list holding all processe references
        self.loe: list[Element] = []  # list of elements in thiis reservoir
        self.doe: dict[Species, Flux] = {}  # species flux pairs
        self.loc: set[Connection] = set()  # set of connection objects
        self.ldf: list[DataField] = []  # list of datafield objects
        # list of processes which calculate reservoirs
        self.lpc: list[Process] = []

        # legacy names
        self.n: str = self.name  # name of reservoir
        # if "register" in self.kwargs:
        if self.register == "None":
            self.pt = self.name
        else:
            self.pt: str = f"{self.register.name}_{self.n}"
            self.groupname = self.register.name
            # self.full_name = f"{self.register.name}.{self.n}"
        # else:
        #   self.pt = self.name

        self.sp: Species = self.species  # species handle
        self.mo: Model = self.species.mo  # model handle
        self.model = self.mo
        self.rvalue = self.sp.r

        # right y-axis label
        self.ld: str = f"{self.species.dn} [{self.species.ds}]"
        self.xl: str = self.mo.xl  # set x-axis lable to model time

        if self.legend_left == "None":
            self.legend_left = self.species.dsa

        self.legend_right = f"{self.species.dn} [{self.species.ds}]"
        # legend_left is in __init__ !

        # decide whether we use isotopes
        if self.mo.m_type == "both":
            self.isotopes = True
        elif self.mo.m_type == "mass_only":
            self.isotopes = False

        if self.geometry != "None":
            get_box_geometry_parameters(self)

        if self.display_precision == 0:
            self.display_precision = self.mo.display_precision

        self.parent = self.register

    # setup a placeholder setitem function
    def __setitem__(self, i: int, value: float):
        return self.__set_data__(i, value)

    def __call__(self) -> None:  # what to do when called as a function ()
        pass
        return self

    def __getitem__(self, i: int) -> np.ndarray:
        """Get flux data by index"""

        return np.array([self.m[i], self.l[i], self.c[i]])

    def __set_with_isotopes__(self, i: int, value: float) -> None:
        """write data by index"""

        self.m[i]: float = value[0]
        # update concentration and delta next. This is computationally inefficient
        # but the next time step may depend on on both variables.
        self.c[i]: float = value[0] / self.v[i]  # update concentration
        self.l[i]: float = value[1]

    def __set_without_isotopes__(self, i: int, value: float) -> None:
        """write data by index"""

        self.m[i]: float = value[0]
        self.c[i]: float = self.m[i] / self.v[i]  # update concentration

    def __update_mass__() -> None:
        """Place holder function"""

        raise NotImplementedError("__update_mass__ is not yet implmented")

    def get_process_args(self):
        """Provide the data structure which needs to be passed to the numba solver"""

        print(f"Name = {self.full_name}")
        data = List(
            [
                self.m,  # 0
                self.l,  # 1
                self.c,  # 2
                self.v,  # 3
            ]
        )

        func_name: col.Callable = self.__update_mass__
        params = List([float(self.reservoir.species.element.r)])

        return func_name, data, params

    def __write_data__(
        self,
        prefix: str,
        start: int,
        stop: int,
        stride: int,
        append: bool,
        directory: str,
    ) -> None:
        """To be called by write_data and save_state"""

        from pathlib import Path

        p = Path(directory)
        p.mkdir(parents=True, exist_ok=True)

        # some short hands
        sn = self.sp.n  # species name
        sp = self.sp  # species handle
        mo = self.sp.mo  # model handle

        smu = f"{mo.m_unit:~P}"
        mtu = f"{mo.t_unit:~P}"
        fmu = f"{mo.f_unit:~P}"
        cmu = f"{mo.c_unit:~P}"

        # sdn = self.sp.dn  # delta name
        # sds = self.sp.ds  # delta scale
        rn = self.full_name  # reservoir name
        mn = self.sp.mo.n  # model name
        if self.sp.mo.register == "None":
            fn = f"{directory}/{prefix}{mn}_{rn}.csv"  # file name
        elif self.sp.mo.register == "local":
            fn = f"{directory}/{prefix}{rn}.csv"  # file name
        else:
            raise ValueError(
                f"Model register keyword must be 'None'/'local' not {self.sp.mo.register}"
            )

        # build the dataframe
        df: pd.dataframe = DataFrame()

        df[f"{rn} Time [{mtu}]"] = self.mo.time[start:stop:stride]  # time
        df[f"{rn} {sn} [{smu}]"] = self.m[start:stop:stride]  # mass
        df[f"{rn} {sp.ln} [{smu}]"] = self.l[start:stop:stride]  # light isotope
        df[f"{rn} {sn} [{cmu}]"] = self.c[start:stop:stride]  # concentration

        fullname: list = []

        for f in self.lof:  # Assemble the headers and data for the reservoir fluxes
            if f.full_name in fullname:
                raise ValueError(f"{f.full_name} is a double")
            fullname.append(f.full_name)

            if f.save_flux_data:
                df[f"{f.full_name} {sn} [{fmu}]"] = f.m[start:stop:stride]  # m
                df[f"{f.full_name} {sn} [{sp.ln}]"] = f.l[start:stop:stride]  # l
            else:
                df[f"{f.full_name} {sn} [{fmu}]"] = f.fa[0]  # m
                df[f"{f.full_name} {sn} [{sp.ln}]"] = f.fa[1]  # l

        file_path = Path(fn)
        if append:
            if file_path.exists():
                df.to_csv(file_path, header=False, mode="a", index=False)
            else:
                df.to_csv(file_path, header=True, mode="w", index=False)
        else:
            df.to_csv(file_path, header=True, mode="w", index=False)

        return df

    def __sub_sample_data__(self) -> None:
        """There is usually no need to keep more than a thousand data points
        so we subsample the results before saving, or processing them

        """
        stride = int(len(self.m) / self.mo.number_of_datapoints)

        # print(f"Reset data with {len(self.m)}, stride = {self.mo.reset_stride}")
        self.m = self.m[2:-2:stride]
        self.l = self.l[2:-2:stride]
        # self.h = self.h[2:-2:stride]
        # self.d = self.d[2:-2:stride]
        self.c = self.c[2:-2:stride]

    def __reset_state__(self) -> None:
        """Copy the result of the last computation back to the beginning
        so that a new run will start with these values

        save the current results into the temp fields

        """

        # print(f"Reset data with {len(self.m)}, stride = {self.mo.reset_stride}")
        self.mc = np.append(self.mc, self.m[0 : -2 : self.mo.reset_stride])
        # self.dc = np.append(self.dc, self.d[0 : -2 : self.mo.reset_stride])
        self.cc = np.append(self.cc, self.c[0 : -2 : self.mo.reset_stride])

        # copy last result into first field
        self.m[0] = self.m[-2]
        self.l[0] = self.l[-2]
        # self.h[0] = self.h[-2]
        # self.d[0] = self.d[-2]
        self.c[0] = self.c[-2]

    def __merge_temp_results__(self) -> None:
        """Once all iterations are done, replace the data fields
        with the saved values

        """

        self.m = self.mc
        self.c = self.cc
        # self.d = self.dc

    def __read_state__(self, directory: str) -> None:
        """read data from csv-file into a dataframe

        The CSV file must have the following columns

        Model Time     t
        Reservoir_Name m
        Reservoir_Name l
        Reservoir_Name h
        Reservoir_Name d
        Reservoir_Name c
        Flux_name m
        Flux_name l etc etc.

        """

        from .utility_functions import is_name_in_list, get_object_from_list

        read: set = set()
        curr: set = set()

        if self.sp.mo.register == "None":
            fn = f"{directory}/state_{self.mo.n}_{self.full_name}.csv"
        elif self.sp.mo.register == "local":
            fn = f"{directory}/state_{self.full_name}.csv"
        else:
            raise ValueError(
                f"Model register keyword must be 'None'/'local' not {self.sp.mo.register}"
            )

        logging.info(f"reading state for {self.full_name} from {fn}")

        if not os.path.exists(fn):
            raise FileNotFoundError(
                f"Flux {fn} does not exist in Reservoir {self.full_name}"
            )

        # get a set of all current fluxes
        for f in self.lof:
            curr.add(f.full_name)
            logging.debug(f"    Adding Flux {f.full_name} to list of fluxes to read")

        self.df: pd.DataFrame = pd.read_csv(fn)
        self.headers: list = list(self.df.columns.values)
        df = self.df
        headers = self.headers

        # the headers contain the object name for each data in the
        # reservoir or flux thus, we must reduce the list to unique
        # object names first. Note, we must preserve order
        header_list: list = []
        for x in headers:
            n = x.split(" ")[0]
            if n not in header_list:
                header_list.append(n)

        # loop over all columns
        col: int = 1  # we ignore the time column
        i: int = 0
        for n in header_list:
            name = n.split(" ")[0]
            logging.debug(f"Looking for {name}")
            # this finds the reservoir name
            if name == self.full_name:
                logging.debug(f"found reservoir data for {name}")
                col = self.__assign_reservoir_data__(self, df, col, True)
            # this loops over all fluxes in a reservoir
            elif is_name_in_list(name, self.lof):
                logging.debug(f"{name} is in {self.full_name}.lof")
                obj = get_object_from_list(name, self.lof)
                logging.debug(
                    f"found object {obj.full_name} adding flux data for {name}"
                )
                read.add(obj.full_name)
                col = self.__assign_flux_data__(obj, df, col, False)
                i += 1
            else:
                raise ValueError(f"Unable to find Flux {n} in {self.full_name}")

        # test if we missed any fluxes
        for f in list(curr.difference(read)):
            print(f"\n Warning: Did not find values for {f}\n in saved state")

    def __assign_flux_data__(
        self, obj: any, df: pd.DataFrame, col: int, res: bool
    ) -> int:
        """
        Assign the third last entry data to all values in flux

        parameters: df = dataframe
                    col = column number
                    res = true if reservoir

        """

        obj.fa[0] = df.iloc[0, col]
        obj.fa[1] = df.iloc[0, col + 1]
        # obj.fa[2] = df.iloc[0, col + 2]
        # obj.fa[3] = df.iloc[0, col + 3]
        col = col + 2

        return col

    def __assign_reservoir_data__(
        self, obj: any, df: pd.DataFrame, col: int, res: bool
    ) -> int:
        """
        Assign the third last entry data to all values in reservoir

        parameters: df = dataframe
                    col = column number
                    res = true if reservoir

        """

        obj.m[:] = df.iloc[-3, col]
        obj.l[:] = df.iloc[-3, col + 1]
        # obj.h[:] = df.iloc[-3, col + 2]
        # obj.d[:] = df.iloc[-3, col + 3]
        obj.c[:] = df.iloc[-3, col + 2]
        col = col + 3

        return col

    def __plot__(self, M: Model, ax) -> None:
        """Plot instructions.
        M: Model
        ax: matplotlib axes handle
        """

        from esbmtk import set_y_limits

        # convert time and data to display units
        x = (M.time * M.t_unit).to(M.d_unit).magnitude

        if self.display_as == "mass":
            y1 = (self.m * M.m_unit).to(self.plt_units).magnitude
            y1_label = f"{self.legend_left} [{self.plt_units:~P}]"
        elif self.display_as == "ppm":
            y1 = self.c * 1e6
            y1_label = "ppm"
        else:
            y1 = (self.c * M.c_unit).to(self.plt_units).magnitude
            y1_label = f"{self.legend_left} [{self.plt_units:~P}]"

        # test for plt_transform
        if self.plot_transform_c != "None":
            if callable(self.plot_transform_c):
                y1 = self.plot_transform_c(self.c)
            else:
                raise ValueError("Plot transform must be a function")

        # plot first axis
        ax.plot(x[1:-2], y1[1:-2], color="C0", label=y1_label)
        ax.set_xlabel(f"{M.time_label} [{M.d_unit:~P}]")
        ax.set_ylabel(f"{self.legend_left} [{M.c_unit:~P}]")

        # add any external data if present
        for i, d, in enumerate(self.led):
            time = (d.x * M.t_unit).to(M.d_unit).magnitude
            yd = d.y.to(self.plt_units).magnitude
            leg = f"{self.lm} {d.legend}"
            ax.scatter(time[1:-2], yd[1:-2], color=f"C{i+1}", label=leg)
        
        ax.spines["top"].set_visible(False)
        handler1, label1 = ax.get_legend_handles_labels()

        if self.isotopes:
            axt = ax.twinx()
            y2 = self.d  # no conversion for isotopes
            axt.plot(x[1:-2], y2[1:-2], color="C1", label=self.legend_right)
            axt.set_ylabel(self.data.ld)
            set_y_limits(axt, M)
            x.spines["top"].set_visible(False)
            # set combined legend
            handler2, label2 = axt.get_legend_handles_labels()
            legend = axt.legend(handler1 + handler2, label1 + label2, loc=0).set_zorder(
                6
            )
        else:
            ax.legend(handler1, label1)
            ax.spines["right"].set_visible(False)
            ax.yaxis.set_ticks_position("left")
            ax.xaxis.set_ticks_position("bottom")

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

        # print basic data bout this reservoir
        print(f"{ind}{self.__str__(kwargs)}")
        print(f"{ind}Data sample:")
        show_data(self, index=index, indent=indent)

        print(f"\n{ind}Connnections:")
        for p in sorted(self.loc):
            print(f"{off}{ind}{p.full_name}.info()")

        print(f"\n{ind}Fluxes:")

        # m = Q_("1 Sv").to("l/a").magnitude
        for i, f in enumerate(self.lof):
            print(f"{off}{ind}{f.full_name}: {self.lodir[i]*f.m[-2]:.2e}")

        print()
        print("Use the info method on any of the above connections")
        print("to see information on fluxes and processes")


class Reservoir(ReservoirBase):
    """This object holds reservoir specific information.

          Example::

                  Reservoir(name = "foo",      # Name of reservoir
                            species = S,          # Species handle
                            delta = 20,           # initial delta - optional (defaults  to 0)
                            mass/concentration = "1 unit"  # species concentration or mass
                            volume/geometry = "1E5 l",      # reservoir volume (m^3)
                            plot = "yes"/"no", defaults to yes
                            plot_transform_c = a function reference, optional (see below)
                            legend_left = str, optional, useful for plot transform
                            display_precision = number, optional, inherited from Model
                            register = optional, use to register with Reservoir Group
                            isotopes = True/False otherwise use Model.m_type
                            seawater_parameters= dict, optional
                            )

          You must either give mass or concentration.  The result will always be displayed
          as concentration though.

          You must provide either the volume or the geometry keyword. In the latter case
          provide a list where the first entry is the upper depth datum, the second entry is
          the lower depth datum, and the third entry is the area percentage. E.g., to specify
          the upper 200 meters of the entire ocean, you would write:

                 geometry=[0,-200,1]

          the corresponding ocean volume will then be calculated by the calc_volume method
          in this case the following instance variables will also be set:

                 self.volume in model units (usually liter)
                 self.are:a surface area in m^2 at the upper bounding surface
                 self.area_dz: area of seafloor which is intercepted by this box.
                 self.area_fraction: area of seafloor which is intercepted by this
                                    relative to the total ocean floor area

          Adding seawater_properties:
          ~~~~~~~~~~~~~~~~~~~~~~~~~~~
          If this optional parameter is specified, a SeaWaterConstants instance will be registered
          for this Reservoir as Reservoir.swc
          See the  SeaWaterConstants class for details how to specify the parameters, e.g.:
          seawater_parameters = {"temperature": 2, "pressure": 240, "salinity" : 35},

          Using a transform function
          ~~~~~~~~~~~~~~~~~~~~~~~~~~

          In some cases, it is useful to transform the reservoir
          concentration data before plotting it.  A good example is the H+
          concentration in water which is better displayed as pH.  We can
          do this by specifying a function to convert the reservoir
          concentration into pH units::

              def phc(c :float) -> float:
                  # Calculate concentration as pH. c can be a number or numpy array

                  import numpy as np

                  pH :float = -np.log10(c)
                  return pH

          this function can then be added to a reservoir as::

          hplus.plot_transform_c = phc

          You can modify the left legend to suit the transform via the legend_left keyword

          Note, at present the plot_transform_c function will only take one
          argument, which always defaults to the reservoir
          concentration. The function must return a single argument which
          will be interpreted as the transformed reservoir concentration.

    Accesing Reservoir Data:
    ~~~~~~~~~~~~~~~~~~~~~~~~

    You can access the reservoir data as:

    - Name.m # mass
    - Name.d # delta
    - Name.c # concentration

    Useful methods include:

    - Name.write_data() # save data to file
    - Name.info()   # info Reservoir
    """

    def __init__(self, **kwargs) -> None:
        """Initialize a reservoir."""

        from esbmtk import (
            SourceGroup,
            SinkGroup,
            ReservoirGroup,
            ConnectionGroup,
            SeawaterConstants,
            get_box_geometry_parameters,
        )

        # provide a dict of all known keywords and their type
        self.defaults: dict[str, list[any, tuple]] = {
            "name": ["None", (str)],
            "species": ["None", (str, Species)],
            "delta": ["None", (int, float, str)],
            "concentration": ["None", (str, Q_, float)],
            "mass": ["None", (str, Q_)],
            "volume": ["None", (str, Q_)],
            "geometry": ["None", (list, str)],
            "plot_transform_c": ["None", (any)],
            "legend_left": ["None", (str)],
            "plot": ["yes", (str)],
            "groupname": ["None", (str)],
            "rtype": ["regular", (str)],
            "function": ["None", (str, col.Callable)],
            "display_precision": [0.01, (int, float)],
            "register": [
                "None",
                (SourceGroup, SinkGroup, ReservoirGroup, ConnectionGroup, Model, str),
            ],
            "parent": [
                "None",
                (SourceGroup, SinkGroup, ReservoirGroup, ConnectionGroup, Model, str),
            ],
            "full_name": ["None", (str)],
            "seawater_parameters": ["None", (dict, str)],
            "isotopes": [False, (bool)],
            "ideal_water": ["None", (str, bool)],
            "has_cs1": [False, (bool)],
            "has_cs2": [False, (bool)],
        }

        # provide a list of absolutely required keywords
        self.lrk: list = [
            "name",
            "species",
            "register",
            ["volume", "geometry"],
            ["mass", "concentration"],
        ]

        # steps = kwargs["species"].mo.steps
        # self.m: np.ndarray =np.zeros(steps) + self.mass

        self.__initialize_keyword_variables__(kwargs)

        self.model = self.register
        self.parent = self.register

        self.__set_legacy_names__(kwargs)

        # geoemtry information
        if self.volume == "None":
            get_box_geometry_parameters(self)

        # convert units
        self.volume: tp.Union[int, float] = Q_(self.volume).to(self.mo.v_unit).magnitude

        # This should probably be species specific?
        self.mu: str = self.sp.e.mass_unit  # massunit xxxx

        self.ideal_water = self.mo.ideal_water
        if self.ideal_water:
            self.density = 1000
        else:
            if isinstance(self.parent, ReservoirGroup):
                self.swc = self.parent.swc
                self.density = self.swc.density
            else:
                if isinstance(self.seawater_parameters, str):
                    temperature = 25
                    salinity = 35
                    pressure = 1
                else:
                    if "temperature" in self.seawater_parameters:
                        temperature = self.seawater_parameters["temperature"]
                    else:
                        temperature = 25
                    if "salinity" in self.seawater_parameters:
                        salinity = self.seawater_parameters["salinity"]
                    else:
                        salinity = 35
                    if "pressure" in self.seawater_parameters:
                        pressure = self.seawater_parameters["pressure"]
                    else:
                        pressure = 1

                SeawaterConstants(
                    name="swc",
                    temperature=temperature,
                    pressure=pressure,
                    salinity=salinity,
                    register=self,
                )
                self.density = self.swc.density

        if self.mass == "None":
            if isinstance(self.concentration, (str, Q_)):
                c = Q_(self.concentration)
                self.plt_units = c.units
                self._concentration: tp.Union[int, float] = c.to(
                    self.mo.c_unit
                ).magnitude
            else:
                c = self.concentration
                self.plt_units = self.mo.c_unit
                self._concentration = c

            self.mass: tp.Union[int, float] = (
                self.concentration * self.volume * self.density / 1000
            )
            self.display_as = "concentration"
        elif self.concentration == "None":
            m = Q_(self.mass)
            self.plt_units = self.mo.m_unit
            self.mass: tp.Union[int, float] = m.to(self.mo.m_unit).magnitude
            self.concentration = self.mass / self.volume
            self.display_as = "mass"
        else:
            raise ValueError("You need to specify mass or concentration")

        self.state = 0

        # save the unit which was provided by the user for display purposes
        # left y-axis label
        self.lm: str = f"{self.species.n} [{self.mu}/l]"

        # initialize mass vector
        self.m: np.ndarray = np.zeros(self.species.mo.steps) + self.mass
        self.l: np.ndarray = np.zeros(self.mo.steps)
        self.v: np.ndarray = np.zeros(self.mo.steps) + self.volume  # reservoir volume

        if self.delta != "None":
            self.l = get_l_mass(self.m, self.delta, self.species.r)

        # create temporary memory if we use multiple solver iterations
        if self.mo.number_of_solving_iterations > 0:
            self.mc = np.empty(0)
            self.cc = np.empty(0)
            self.dc = np.empty(0)

        self.mo.lor.append(self)  # add this reservoir to the model
        self.mo.lic.append(self)  # reservoir type object list
        # register instance name in global name space
        if self.mo.register == "local" and self.register == "None":
            self.register = self.mo
        self.__register_name_new__()

        # decide which setitem functions to use
        if self.isotopes:
            self.__set_data__ = self.__set_with_isotopes__
        else:
            self.__set_data__ = self.__set_without_isotopes__

        # any auxilliary init - normally empty, but we use it here to extend the
        # reservoir class in virtual reservoirs
        self.__aux_inits__()

    @property
    def concentration(self) -> float:
        return self._concentration

    @property
    def delta(self) -> float:
        return self._delta

    @property
    def mass(self) -> float:
        return self._mass

    @concentration.setter
    def concentration(self, c) -> None:
        if self.update and c != "None":
            self._concentration: float = c.to(self.mo.c_unit).magnitude
            self.mass: float = (
                self._concentration * self.volume * self.density / 1000
            )  # caculate mass
            self.c = self.c * 0 + self._concentration
            self.m = self.m * 0 + self.mass

    @delta.setter
    def delta(self, d: float) -> None:
        if self.update and d != "None":
            self._delta: float = d
            self.isotopes = True
            self.l = get_l_mass(self.m, d, self.species.r)

    @mass.setter
    def mass(self, m: float) -> None:
        if self.update and m != "None":
            self._mass: float = m
            self.m = np.zeros(self.species.mo.steps) + m
            self.c = self.m / self.volume


class Flux(esbmtkBase):
    """A class which defines a flux object. Flux objects contain
    information which links them to an species, describe things like
    the mass and time unit, and store data of the total flux rate at
    any given time step. Similarly, they store the flux of the light
    and heavy isotope flux, as well as the delta of the flux. This
    is typically handled through the Connect object. If you set it up manually

    Flux = (name = "Name" # optional, defaults to _F
            species = species_handle,
            delta = any number,
            rate  = "12 mol/s" # must be a string
            display_precision = number, optional, inherited from Model
    )

     You can access the flux data as
    - Name.m # mass
    - Name.d # delta
    - Name.c # concentration

    """

    def __init__(self, **kwargs: dict[str, any]) -> None:
        """
        Initialize a flux. Arguments are the species name the flux rate
        (mol/year), the delta value and unit

        """

        from esbmtk import (
            Q_,
            AirSeaExchange,
            Reservoir,
            GasReservoir,
            Connect,
            Connection,
            Signal,
        )

        # provide a dict of all known keywords and their type
        self.defaults: dict[str, list[any, tuple]] = {
            "name": ["None", (str)],
            "species": ["None", (str, Species)],
            "delta": [0, (str, int, float)],
            "rate": ["None", (str, Q_, int, float)],
            "plot": ["yes", (str)],
            "display_precision": [0.01, (int, float)],
            "isotopes": [False, (bool)],
            "register": [
                "None",
                (
                    str,
                    Reservoir,
                    GasReservoir,
                    Connection,
                    Connect,
                    AirSeaExchange,
                    Signal,
                ),
            ],
            "save_flux_data": [False, (bool)],
            "id": ["None", (str)],
        }

        # provide a list of absolutely required keywords
        self.lrk: list = ["species", "rate", "register"]

        self.__initialize_keyword_variables__(kwargs)

        self.parent = self.register
        # if save_flux_data is unsepcified, use model default
        if self.save_flux_data == "None":
            self.save_flux_data = self.species.mo.save_flux_data

        # legacy names
        self.n: str = self.name  # name of flux
        self.sp: Species = self.species  # species name
        self.mo: Model = self.species.mo  # model name
        self.model: Model = self.species.mo  # model handle
        self.rvalue = self.sp.r

        if self.display_precision == 0:
            self.display_precision = self.mo.display_precision

        # model units
        self.plt_units = Q_(self.rate).units
        self.mu: str = f"{self.species.mu}/{self.mo.tu}"

        # and convert flux into model units
        if isinstance(self.rate, str):
            fluxrate: float = Q_(self.rate).to(self.mo.f_unit).magnitude
        elif isinstance(self.rate, Q_):
            fluxrate: float = self.rate.to(self.mo.f_unit).magnitude
        elif isinstance(self.rate, (int, float)):
            fluxrate: float = self.rate

        if self.delta:
            li = get_l_mass(fluxrate, self.delta, self.sp.r)
        else:
            li = 0
        self.fa: np.ndarray = np.array([fluxrate, li])

        # in case we want to keep the flux data
        if self.save_flux_data:
            self.m: np.ndarray = np.zeros(self.model.steps) + fluxrate  # add the flux
            self.l: np.ndarray = np.zeros(self.model.steps)

            if self.mo.number_of_solving_iterations > 0:
                self.mc = np.empty(0)
                self.dc = np.empty(0)

            if self.rate != 0:
                self.l = get_l_mass(self.m, self.delta, self.species.r)
                self.fa[1] = self.l[0]

        else:
            # setup dummy variables to keep existing numba data structures
            self.m = np.zeros(2)
            self.l = np.zeros(2)

            if self.rate != 0:
                self.fa[1] = get_l_mass(self.fa[0], self.delta, self.species.r)

        self.lm: str = f"{self.species.n} [{self.mu}]"  # left y-axis a label
        self.ld: str = f"{self.species.dn} [{self.species.ds}]"  # right y-axis a label

        self.legend_left: str = self.species.dsa
        self.legend_right: str = f"{self.species.dn} [{self.species.ds}]"

        self.xl: str = self.model.xl  # se x-axis label equal to model time
        self.lop: list[Process] = []  # list of processes
        self.lpc: list = []  # list of external functions
        self.led: list[ExternalData] = []  # list of ext data
        self.source: str = ""  # Name of reservoir which acts as flux source
        self.sink: str = ""  # Name of reservoir which acts as flux sink

        if self.name == "None":
            if isinstance(self.parent, (Connection, Connect)):
                self.name = "_F"
                self.n = self.name
            else:
                self.name = f"{self.id}_F"

        self.__register_name_new__()
        # print(f"f name set to {self.name}. fn =  {self.full_name}\n")
        self.mo.lof.append(self)  # register with model flux list

        # decide which setitem functions to use
        # decide whether we use isotopes
        if self.mo.m_type == "both":
            self.isotopes = True
        elif self.mo.m_type == "mass_only":
            self.isotopes = False

        if self.isotopes:
            self.__set_data__ = self.__set_with_isotopes__
            # self.__get_data__ = self.__get_with_isotopes__
        else:
            self.__set_data__ = self.__set_without_isotopes__
            # self.__get_data__ = self.__get_without_isotopes__

    # setup a placeholder setitem function
    def __setitem__(self, i: int, value: np.ndarray):
        return self.__set_data__(i, value)

    def __getitem__(self, i: int) -> np.ndarray:
        """Get data by index"""
        # return self.__get_data__(i)
        return self.fa

    def __set_with_isotopes__(self, i: int, value: np.ndarray) -> None:
        """Write data by index"""

        self.m[i] = value[0]
        self.l[i] = value[1]
        self.fa = value[0:4]

    def __set_without_isotopes__(self, i: int, value: np.ndarray) -> None:
        """Write data by index"""

        self.fa = [value[0], 0]
        self.m[i] = value[0]

    def __call__(self) -> None:  # what to do when called as a function ()
        pass
        return

    def __add__(self, other):
        """adding two fluxes works for the masses, but not for delta"""

        self.fa = self.fa + other.fa
        self.m = self.m + other.m
        self.l = self.l + other.l

    def __sub__(self, other):
        """substracting two fluxes works for the masses, but not for delta"""

        self.fa = self.fa - other.fa
        self.m = self.m - other.m
        self.l = self.l - other.l

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

        # print basic data bout this object
        print(f"{ind}{self.__str__(kwargs)}")
        print(f"{ind}Data sample:")
        show_data(self, index=index, indent=indent)

        if len(self.lop) > 0:
            print(f"\n{ind}Process(es) acting on this flux:")
            for p in self.lop:
                print(f"{off}{ind}{p.__repr__()}")

            print("")
            print(
                "Use help on the process name to get an explanation what this process does"
            )
            if self.register == "None":
                print(f"e.g., help({self.lop[0].n})")
            else:
                print(f"e.g., help({self.register.name}.{self.lop[0].name})")
        else:
            print("There are no processes for this flux")

    def __plot__(self, M: Model, ax) -> None:
        """Plot instructions.
        M: Model
        ax: matplotlib axes handle
        """

        from esbmtk import set_y_limits

        # convert time and data to display units
        x = (M.time * M.t_unit).to(M.d_unit).magnitude
        y1 = (self.m * M.m_unit).to(self.plt_units).magnitude

        # test for plt_transform
        if self.plot_transform_c != "None":
            if callable(self.plot_transform_c):
                y1 = self.plot_transform_c(self.c)
            else:
                raise ValueError("Plot transform must be a function")

        # plot first axis
        ax.plot(x[1:-2], y1[1:-2], color="C0", label=self.legend_left)
        ax.set_xlabel(f"{M.time_label} [{M.d_unit:~P}]")
        ax.set_ylabel(self.legend_left)
        ax.spines["top"].set_visible(False)
        handler1, label1 = ax.get_legend_handles_labels()

        # plot second axis
        if self.isotopes:
            axt = ax.twinx()
            y2 = self.d  # no conversion for isotopes
            ln2 = axt.plot(x[1:-2], y2[1:-2], color="C1", label=self.legend_right)
            axt.set_ylabel(self.data.ld)
            set_y_limits(axt, M)
            x.spines["top"].set_visible(False)
            # set combined legend
            handler2, label2 = axt.get_legend_handles_labels()
            legend = axt.legend(handler1 + handler2, label1 + label2, loc=0).set_zorder(
                6
            )
        else:
            ax.legend(handler1, label1)
            ax.spines["right"].set_visible(False)
            ax.yaxis.set_ticks_position("left")
            ax.xaxis.set_ticks_position("bottom")

    def __sub_sample_data__(self) -> None:
        """There is usually no need to keep more than a thousand data points
        so we subsample the results before saving, or processing them

        """

        if self.save_flux_data:
            stride = int(len(self.m) / self.mo.number_of_datapoints)

            self.m = self.m[2:-2:stride]
            self.l = self.m[2:-2:stride]

    def __reset_state__(self) -> None:
        """Copy the result of the last computation back to the beginning
        so that a new run will start with these values.

        Also, copy current results into temp field
        """

        if self.save_flux_data:
            self.mc = np.append(self.mc, self.m[0 : -2 : self.mo.reset_stride])
            # copy last element to first position
            self.m[0] = self.m[-2]
            self.l[0] = self.l[-2]

    def __merge_temp_results__(self) -> None:
        """Once all iterations are done, replace the data fields
        with the saved values

        """

        self.m = self.mc


class SourceSink(esbmtkBase):
    """
    This is a meta class to setup a Source/Sink objects. These are not
    actual reservoirs, but we stil need to have them as objects
    Example::

           Sink(name = "Pyrite",
               species = SO4,
               display_precision = number, optional, inherited from Model
               delta = number or str. optional defaults to "None"
           )

    where the first argument is a string, and the second is a reservoir handle

    """

    def __init__(self, **kwargs) -> None:

        from esbmtk import (
            SourceGroup,
            SinkGroup,
            ReservoirGroup,
            ConnectionGroup,
        )

        self.defaults: dict[str, list[any, tuple]] = {
            "name": ["None", (str)],
            "species": ["None", (str, Species)],
            "display_precision": [0.01, (int, float)],
            "register": [
                "None",
                (
                    SourceGroup,
                    SinkGroup,
                    ReservoirGroup,
                    ConnectionGroup,
                    Model,
                    str,
                ),
            ],
            "delta": ["None", (str, int, float)],
            "isotopes": [False, (bool)],
        }
        # provide a list of absolutely required keywords
        self.lrk: list[str] = ["name", "species", "register"]
        self.__initialize_keyword_variables__(kwargs)

        self.loc: set[Connection] = set()  # set of connection objects

        # legacy names
        # if self.register != "None":
        #    self.full_name = f"{self.name}.{self.register.name}"
        self.parent = self.register
        self.n = self.name
        self.sp = self.species
        self.mo = self.species.mo
        self.model = self.species.mo
        self.u = self.species.mu + "/" + str(self.species.mo.bu)
        self.lio: list = []
        self.mo.lic.append(self)  # add source to list of res type objects

        if self.mo.register == "local" and self.register == "None":
            self.register = self.mo
        else:
            if self.register == "None":
                self.pt = self.name
            else:
                self.pt: str = f"{self.register.name}_{self.n}"
                self.groupname = self.register.name

        if self.display_precision == 0:
            self.display_precision = self.mo.display_precision

        self.__register_name_new__()
        self.mo.lic.remove(self)

    @property
    def delta(self):
        return self._delta

    @delta.setter
    def delta(self, d):
        """Set/Update delta"""
        if d != "None":
            self._delta = d
            self.isotopes = True
            self.m = 1
            self.l = get_l_mass(self.m, d, self.species.r)
            self.c = self.l / (self.m - self.l)
            # self.provided_kwargs.update({"delta": d})


class Sink(SourceSink):
    """
    This is just a wrapper to setup a Sink object
    Example::

           Sink(name = "Pyrite",species =SO4)

    where the first argument is a string, and the second is a species handle
    """


class Source(SourceSink):
    """
    This is just a wrapper to setup a Source object
    Example::

           Source(name = "SO4_diffusion", species ="SO4")

    where the first argument is a string, and the second is a species handle
    """


# from .extended_classes import *
# from .connections import Connection, ConnectionGroup, Connect
# from .processes import *
# from .carbonate_chemistry import *
# from .sealevel import *
# from .solver import *
