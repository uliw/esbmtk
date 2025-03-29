"""esbmtk: A general purpose Earth Science box model toolkit.

Copyright (C), 2020 Ulrich G. Wortmann

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at
your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
import typing as tp
import warnings
from pathlib import Path
from time import process_time

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import psutil
from pandas import DataFrame
from scipy.integrate import solve_ivp

from esbmtk.ode_backend_2 import (
    build_eqs_matrix,
    get_initial_conditions,
    write_equations_3,
)

from . import Q_, ureg
from .esbmtk_base import esbmtkBase
from .utility_functions import (
    find_matching_strings,
    get_delta_from_concentration,
    get_delta_h,
    get_l_mass,
    plot_geometry,
)

# declare numpy types
NDArrayFloat = npt.NDArray[np.float64]

if tp.TYPE_CHECKING:
    from .connections import Species2Species
    from .extended_classes import DataField, ExternalData
    from .processes import Process


class ModelError(Exception):
    """Custom Error Class for Model-related errors."""

    def __init__(self, message):
        """Initialize Error Instance with formatted message."""
        message = f"\n\n{message}\n"
        super().__init__(message)


class SolverError(Exception):
    """Custom Error Class for solver-related errors."""

    def __init__(self, message):
        """Initialize Error Instance with formatted message."""
        message = f"\n\n{message}\n"
        super().__init__(message)


class ReservoirError(Exception):
    """Custom Error Class for reservoir-related errors."""

    def __init__(self, message):
        """Initialize Error Instance with formatted message."""
        message = f"\n\n{message}\n"
        super().__init__(message)


class FluxError(Exception):
    """Custom Error Class for flux-related errors."""

    def __init__(self, message):
        """Initialize Error Instance with formatted message."""
        message = f"\n\n{message}\n"
        super().__init__(message)


class ScaleError(Exception):
    """Custom Error Class for unit scale-related errors."""

    def __init__(self, message):
        """Initialize Error Instance with formatted message."""
        message = f"\n\n{message}\n"
        super().__init__(message)


class SpeciesError(Exception):
    """Custom Error Class for species-related errors."""

    def __init__(self, message):
        """Initialize Error Instance with formatted message."""
        message = f"\n\n{message}\n"
        super().__init__(message)


def deprecated_keyword(message):
    """Issue a deprecation warning with the provided message."""
    warnings.warn(message, DeprecationWarning, stacklevel=2)


class Model(esbmtkBase):
    """Earth Science Box Model Toolkit (ESBMTK) Model class.

    This class represents the main model framework for creating and running
    Earth science box models. It handles initialization of model parameters,
    management of reservoirs, fluxes, and species, and provides methods for
    running simulations and visualizing results.

    The user-facing methods of the model class are:

    - Model_Name.info() - Display model information
    - Model_Name.save_data() - Save model data to files
    - Model_Name.plot([sb.DIC, sb.TA]) - Plot specified objects
    - Model_Name.save_state() - Save current model state
    - Model_Name.read_state() - Initialize with a previous model state
    - Model_Name.run() - Run the model simulation
    - Model_Name.list_species() - List all defined species
    - Model_Name.flux_summary() - Display flux information
    - Model_Name.connection_summary() - Display connection information
    """

    def __init__(self, **kwargs: dict[str, any]) -> None:
        """Initialize a model instance.

        Parameters
        ----------
        **kwargs : dict
            A dictionary with key-value pairs for model configuration.

        Examples
        --------
        >>> esbmtkModel(
        ...     name="Test_Model",  # required
        ...     stop="10000 yrs",   # end time
        ...     max_timestep="1 yr",  # maximum time step
        ...     element=["Carbon", "Sulfur"]
        ... )

        Important Parameters
        -------------------
        name : str
            The model name, e.g., "M".
        mass_unit : str
            Base mass unit for the model, default is "mol".
        volume_unit : str
            Volume unit for the model, default is "liter".
        element : list or str
            One or more species names to include in the model.
        max_timestep : str
            Limit automatic step size increase (time resolution of the model).
            Optional, defaults to model duration/100.
        m_type : str
            Controls isotope calculation for the entire model.
            Options: "Not set" (default, isotopes calculated only for reservoirs
            with isotope keyword), "mass_only", or "both" (overrides reservoir settings).
        offset : str
            Offset the time axis by the specified amount when plotting data.
            For display purposes only, does not affect model calculations.
        display_precision : float
            Affects on-screen display of data and sets cutoff for graphical output.
        opt_k_carbonic : int
            See https://doi.org/10.5194/gmd-15-15-2022.
        opt_pH_scale : int
            pH scale setting: total=1, free=3.
        """
        from importlib.metadata import version

        from esbmtk.sealevel import hypsometry

        # Define default values for model parameters
        self.defaults: dict[str, list[any, tuple]] = {
            "start": ["0 yrs", (str, Q_)],
            "stop": ["None", (str, Q_)],
            "offset": ["0 yrs", (str, Q_)],  # deprecated
            "timestep": ["None", (str, Q_)],  # deprecated
            "max_timestep": ["None", (str, Q_)],
            "min_timestep": ["1 second", (str, Q_)],
            "element": ["None", (str, list)],
            "mass_unit": ["mol", (str)],
            "volume_unit": ["liter", (str)],
            "area_unit": ["m**2", (str)],
            "time_unit": ["year", (str)],
            "concentration_unit": ["mol/liter", (str)],
            "time_label": ["Years", (str)],
            "display_precision": [0.01, (float)],
            "plot_style": ["default", (str)],
            "m_type": ["Not Set", (str)],
            "step_limit": [1e9, (int, float, str)],
            "register": ["local", (str)],
            "save_flux_data": [False, (bool)],
            "full_name": ["None", (str)],
            "parent": ["None", (str)],
            "isotopes": [False, (bool)],
            "debug": [False, (bool)],
            "ideal_water": [True, (bool)],
            "use_ode": [True, (bool)],
            "debug_equations_file": [False, (bool)],
            "rtol": [1.0e-6, (float)],
            "bio_pump_functions": [0, (int)],  # custom/old
            "opt_k_carbonic": [15, (int)],
            "opt_pH_scale": [1, (int)],  # 1: total scale
            "opt_buffers_mode": [2, (int)],
        }

        # Define required keywords
        self.lrk: list[str] = [
            "stop",
            ["timestep", "max_timestep"],
        ]

        # Initialize keyword variables from provided arguments
        self.__initialize_keyword_variables__(kwargs)

        # Check for deprecated keywords
        if self.timestep != "None":
            self.max_timestep = self.timestep
            raise DeprecationWarning(
                "\ntimestep is deprecated, please replace with max_timestep\n"
            )

        # Set default model name
        self.name = "M"

        # Initialize model component containers
        self._initialize_model_containers()

        # Configure logging
        self._setup_logging()

        # Register with parent
        self.__register_with_parent__()

        # Set up unit definitions
        self._configure_units()

        # Process time parameters
        self._configure_time_parameters()

        # Create time arrays
        self._create_time_arrays()

        # Handle step limit
        self._handle_step_limit()

        # Register elements and species with model
        self._register_elements_and_species()

        # Display warranty information
        self._display_warranty(version)

        # Initialize the hypsometry class
        hypsometry(name="hyp", model=self, register=self)

    def _initialize_model_containers(self):
        """Initialize all model component containers."""
        # Model objects
        self.lmo: list = []  # List of all model objects
        self.lmo2: list = []  # Secondary list of model objects
        self.dmo: dict = {}  # Dict of all model objects (for name lookups)

        # Reservoirs and connections
        self.lor: list = []  # List of all reservoir type objects
        self.lic: list = []  # List of all reservoir type objects (internal)
        self.loc: set = set()  # Set of connection objects

        # Elements and species
        self.lel: list = []  # List of all element references
        self.lsp: list = []  # List of all species references

        # External data and signals
        self.led: list = []  # List of all external data objects
        self.los: list = []  # List of signal objects
        self.lvd: list = []  # List of vector data objects

        # Fluxes and processes
        self.lof: list = []  # List of flux objects
        self.lop: list = []  # List of flux processes
        self.lpc_f: list = []  # List of external functions affecting fluxes
        self.lpc_i: list = []  # List of external functions needed in ode_backend
        self.lpc_r: list = []  # List of external functions affecting virtual reservoirs
        self.lvr: list = []  # List of virtual reservoirs

        # Other model components
        self.ldf: list = []  # List of datafield objects
        self.lrg: list = []  # List of reservoir groups
        self.lto: list = []  # List of objects requiring delayed initialization
        self.olkk: list = []  # Optional keywords for use in connector class

        # Global parameters and constants
        self.gpt: tuple = ()  # Global parameter list
        self.toc: tuple = ()  # Global constants list
        self.gcc: int = 0  # Constants counter
        self.vpc: int = 0  # Parameter counter
        self.luf: dict = {}  # User functions and source

    def _setup_logging(self):
        """Configure model logging."""
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        log_filename: str = f"{self.name}.log"
        logging.basicConfig(filename=log_filename, filemode="w", level=logging.CRITICAL)

    def _configure_units(self):
        """Set up model units."""
        self.l_unit = ureg.meter  # Length unit
        self.t_unit = Q_(self.time_unit).units  # Time unit
        self.d_unit = Q_(self.stop).units  # Display time units
        self.m_unit = Q_(self.mass_unit).units  # Mass unit
        self.v_unit = Q_(self.volume_unit).units  # Volume unit
        self.a_unit = Q_(self.area_unit).units  # Area unit
        self.c_unit = Q_(self.concentration_unit).units  # Concentration unit
        self.f_unit = self.m_unit / self.t_unit  # Flux unit (mass/time)
        self.r_unit = self.v_unit / self.t_unit  # Flux as volume/time

    def _configure_time_parameters(self):
        """Process and configure time-related parameters."""
        # Process start and stop times
        self.start = self.ensure_q(self.start).to(self.t_unit).magnitude
        self.stop = self.ensure_q(self.stop).to(self.t_unit).magnitude

        # Handle deprecated timestep parameter
        if self.timestep != "None":
            self.max_timestep = self.ensure_q(self.timestep).to(self.t_unit).magnitude
            deprecated_keyword("timestep is deprecated. Please use max_timestep")
        else:
            self.max_timestep = (
                self.ensure_q(self.max_timestep).to(self.t_unit).magnitude
            )

        # Process remaining time parameters
        self.min_timestep = self.ensure_q(self.min_timestep).to(self.t_unit).magnitude
        self.dt = self.max_timestep
        self.offset = self.ensure_q(self.offset).to(self.t_unit).magnitude
        self.start = self.start + self.offset
        self.stop = self.stop + self.offset

        # Legacy variable names
        self.n = self.name
        self.mo = self.name
        self.model = self
        self.plot_style: list = [self.plot_style]

        # Configure time axis
        self.xl = f"Time [{self.t_unit}]"  # Time axis label
        self.length = int(abs(self.stop - self.start))
        self.steps = int(abs(round(self.length / self.dt)))
        self.number_of_datapoints = self.steps

    def _create_time_arrays(self):
        """Create time arrays for model simulation."""
        self.time_ode = np.linspace(
            self.start,
            self.stop - self.start,
            num=self.number_of_datapoints + 1,
        )
        self.time = self.time_ode
        self.timec = np.empty(0)
        self.state = 0

        # Set default stride
        self.stride = 1

    def _handle_step_limit(self):
        """Handle step limit configuration."""
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

    def _register_elements_and_species(self):
        """Register elements and species with the model."""
        from importlib import import_module

        if "element" in self.kwargs:
            if isinstance(self.kwargs["element"], list):
                element_list = self.kwargs["element"]
            else:
                element_list = [self.kwargs["element"]]

            # Process each element
            for element_name in element_list:
                # Get function handle from species_definitions
                element_handler = getattr(
                    import_module("esbmtk.species_definitions"), element_name
                )
                element_handler(self)  # Register element with model

                # Get element handle and register its species
                element_handle = getattr(self, element_name)
                element_handle.__register_species_with_model__()

    def _display_warranty(self, version_func):
        """Display warranty and citation information."""
        import datetime

        warranty_text = (
            f"\n"
            f"ESBMTK {version_func('esbmtk')}  \n Copyright (C) 2020 - "
            f"{datetime.date.today().year}  Ulrich G.Wortmann\n"
            f"This program comes with ABSOLUTELY NO WARRANTY\n"
            f"This is free software, and you are welcome to redistribute it\n"
            f"under certain conditions; See the LICENSE file for details.\n\n"
            f"If you use ESBMTK for your research, please cite:\n\n"
            f"Wortmann et al. 2025, https://doi.org/10.5194/gmd-18-1155-2025\n"
        )
        print(warranty_text)

    def info(self, **kwargs) -> None:
        """Display an overview of the model properties.

        Prints information about the model instance including defined elements
        and their associated species.

        Parameters
        ----------
        **kwargs : dict
            Optional keyword arguments.

        indent : int, default=0
            Number of spaces to use for indentation in the output.

        index : int, default=0
            Index to use when showing data samples (if applicable).

        Returns
        -------
        None
            This method prints to stdout but doesn't return a value.
        """
        # Handle indentation
        indent = kwargs.get("indent", 0)
        indentation = " " * indent
        offset = "  "  # Standard offset for nested items

        # Print basic model information
        print(self)

        # Display elements and their species
        print("Currently defined elements and their species:")
        for element in self.lel:
            print(f"{indentation}{element}")
            print(f"{offset} Defined SpeciesProperties:")

            # Display species for this element
            for species in element.lsp:
                print(f"{offset}{offset}{indentation}{species.n}")

    def save_state(self, directory: str = "state", prefix: str = "state_") -> None:
        """Save the current model state to files.

        Saves only the last time step of each reservoir to files in the specified directory.
        This is similar to save_data() but focuses on capturing the current state rather
        than the full time series.

        Parameters
        ----------
        directory : str, default="state"
            Directory where state files will be saved. Will be created if it doesn't exist
            and deleted if it already exists.

        prefix : str, default="state_"
            Prefix to add to all saved filenames.

        Returns
        -------
        None

        Raises
        ------
        FileExistsError
            If the directory exists and cannot be deleted.
        """
        from pathlib import Path

        from esbmtk.utility_functions import rmtree

        # Prepare directory
        target_path = Path.cwd() / directory

        # Check if directory exists and remove it if it does
        if target_path.exists():
            print(f"Found previous state directory, deleting {target_path}")
            rmtree(target_path)

            # Verify directory was deleted
            if target_path.exists():
                raise FileExistsError(
                    f"Failed to delete existing directory: {target_path}"
                )

        # Define slice parameters for the last state only
        start_idx = -2  # Second-to-last index (to avoid boundary effects)
        stop_idx = None  # No stop index means go to the end
        stride_idx = 1  # Use every value

        # Write data for each reservoir
        for reservoir in self.lor:
            reservoir.__write_data__(
                prefix=prefix,
                start=start_idx,
                stop=stop_idx,
                stride=stride_idx,
                append=False,
                directory=directory,
            )

    def save_data(self, directory: str = "./data") -> None:
        """Save all model results to CSV files.

        Creates a directory (or recreates if it exists) and saves the full time series
        of all model components to separate CSV files. Each reservoir, signal, and vector
        data object will have its own CSV file.

        Parameters
        ----------
        directory : str, default="./data"
            Directory where data files will be saved. Will be created if it doesn't exist
            and deleted if it already exists.

        Returns
        -------
        None

        Raises
        ------
        FileExistsError
            If the directory exists and cannot be deleted.
        """
        from pathlib import Path

        from esbmtk.utility_functions import rmtree

        # Prepare directory
        target_path = Path.cwd() / directory

        # Check if directory exists and remove it if it does
        if target_path.exists():
            print(f"Found previous data directory, deleting {target_path}")
            rmtree(target_path)

            # Verify directory was deleted
            if target_path.exists():
                raise FileExistsError(
                    f"Failed to delete existing directory: {target_path}"
                )

        # Define common parameters for data writing
        prefix = ""
        stride = self.stride
        start_idx = 0
        stop_idx = len(self.time)
        append = False

        # Save all regular reservoirs (excluding flux-only types)
        for reservoir in self.lor:
            if reservoir.rtype != "flux_only":
                reservoir.__write_data__(
                    prefix=prefix,
                    start=start_idx,
                    stop=stop_idx,
                    stride=stride,
                    append=append,
                    directory=directory,
                )

        # Save all signal objects
        for signal in self.los:
            signal.__write_data__(
                prefix=prefix,
                start=start_idx,
                stop=stop_idx,
                stride=stride,
                append=append,
                directory=directory,
            )

        # Save all vector data objects
        for vector_data in self.lvd:
            vector_data.__write_data__(
                prefix=prefix,
                start=start_idx,
                stop=stop_idx,
                stride=stride,
                append=append,
                directory=directory,
            )

    def read_data(self, directory: str = "./data") -> None:
        """Read model results from CSV files.

        Loads previously saved model data from CSV files in the specified directory.
        Updates the model's internal state with the loaded data.

        Parameters
        ----------
        directory : str, default="./data"
            Directory containing the saved model data files.

        Returns
        -------
        None
        """
        from esbmtk import GasReservoir, Species

        prefix = ""

        print(f"Reading data from {directory}")

        # Process each reservoir
        for reservoir in self.lor:
            # Only process Species and GasReservoir objects
            if isinstance(reservoir, Species | GasReservoir):
                # Read the state data
                reservoir.__read_state__(directory, prefix)

                # Calculate delta values for reservoirs with isotopes
                if reservoir.isotopes:
                    reservoir.d = get_delta_from_concentration(
                        reservoir.c, reservoir.l, reservoir.sp.r
                    )

    def read_state(self, directory="state"):
        """Initialize the model with the result of a previous.

        For this to work, you will need issue a
        Model.save_state() command at then end of a model run.  This
        will create the necessary data files to initialize a
        subsequent model run.
        """
        from pathlib import Path

        from esbmtk import GasReservoir, Species  # GasReservoir

        path = Path(directory).resolve()
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(
                f"The directory '{path}' does not exist or is not a directory."
            )

        for r in self.lor:
            if isinstance(r, Species | GasReservoir):
                r.__read_state__(directory)

        # update swc object
        for rg in self.lrg:
            if hasattr(rg, "swc"):
                rg.swc.update_parameters(pos=0)

    def plot(self, pl: list = None, **kwargs) -> tuple:
        """Plot model objects and save results to a file.

        Creates a figure with subplots for each provided model object and renders
        their data using the object's __plot__ method.

        Parameters
        ----------
        pl : list or object, default=None
            A list of ESBMTK instances (e.g., reservoirs) to plot.
            If a single object is provided, it will be converted to a list.
            If None, an empty list will be used.

        **kwargs : dict
            Optional plotting parameters:

            fn : str, default="{model_name}.pdf"
                Filename to save the plot.

            title : str, default=None
                Title for the plot window.

            no_show : bool, default=False
                If True, don't display or save the figure; instead return the
                plt, fig, and axes handles for manual customization.

            reverse_time : bool, default=False
                If True, reverse the time axis and adjust tick labels.

            blocking : bool, default=True
                If True, block execution until plot window is closed.

        Returns
        -------
        tuple or None
            If no_show=True, returns (plt, fig, axes), otherwise None.

        Examples
        --------
        Basic usage:

        >>> M.plot([sb.PO4, sb.DIC], fn='test.pdf')

        Advanced usage with customization:

        >>> from esbmtk import data_summaries
        >>> species_names = [M.DIC, M.TA, M.pH, M.CO3, M.zcc, M.zsat, M.zsnow, M.PO4]
        >>> box_names = [M.L_b, M.H_b, M.D_b]
        >>> pl = data_summaries(M, species_names, box_names, M.L_b.DIC)
        >>> pl += [M.CO2_At]
        >>> plt, fig, axs = M.plot(
        >>>     pl,
        >>>     fn="steady_state.pdf",
        >>>     title="ESBMTK Preindustrial Steady State",
        >>>     no_show=True,
        >>> )
        """
        # Ensure pl is a list
        if pl is None:
            pl = []
        if not isinstance(pl, list):
            pl = [pl]

        # Extract plot configuration from kwargs
        filename = kwargs.get("fn", f"{self.n}.pdf")
        blocking = kwargs.get("blocking", True)
        plot_title = kwargs.get("title", "None")
        reverse_time = kwargs.get("reverse_time", False)
        no_show = kwargs.get("no_show", False)

        # Determine layout based on number of plots
        num_plots = len(pl)
        size, geometry = plot_geometry(num_plots)
        row_count, col_count = geometry

        # Create figure and subplots
        fig, ax = plt.subplots(row_count, col_count)

        # Normalize axes structure based on subplot layout
        axs = self._normalize_axes_structure(ax, row_count, col_count)

        # Configure plot style and title
        plt.style.use(self.plot_style)
        window_title = plot_title if plot_title != "None" else f"{self.n} Species"
        fig.canvas.manager.set_window_title(window_title)
        fig.set_size_inches(size)

        # Plot each object in the appropriate subplot
        self._plot_objects_to_subplots(pl, axs, row_count, col_count, num_plots)

        # Adjust figure layout
        fig.subplots_adjust(top=0.88)

        # Handle time axis reversal if requested
        if reverse_time:
            self._reverse_time_axis(fig)

        # Return or display/save the figure
        if no_show:
            return plt, fig, fig.get_axes()
        else:
            fig.tight_layout()
            plt.show(block=blocking)
            fig.savefig(filename)
            return None

    def _normalize_axes_structure(self, ax, row_count: int, col_count: int) -> list:
        """Normalize the axes structure based on subplot layout.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or array of Axes
            The axes object(s) returned by plt.subplots()
        row_count : int
            Number of rows in the subplot grid
        col_count : int
            Number of columns in the subplot grid

        Returns
        -------
        list
            Normalized axes structure for consistent handling
        """
        if row_count == 1 and col_count == 1:
            # Single subplot
            return ax
        elif row_count > 1 and col_count == 1:
            # Multiple rows, one column
            return [ax[i] for i in range(row_count)]
        elif row_count == 1 and col_count > 1:
            # One row, multiple columns
            return [ax[i] for i in range(col_count)]
        else:
            # Multiple rows and columns
            return ax

    def _plot_objects_to_subplots(
        self, plot_objects: list, axes, row_count: int, col_count: int, num_plots: int
    ) -> None:
        """Plot objects to their respective subplots.

        Parameters
        ----------
        plot_objects : list
            List of objects to plot
        axes : matplotlib.axes.Axes or array of Axes
            The normalized axes structure
        row_count : int
            Number of rows in the subplot grid
        col_count : int
            Number of columns in the subplot grid
        num_plots : int
            Total number of objects to plot
        """
        plot_index = 0  # Index of current plot object

        for row in range(row_count):
            if col_count > 1:
                # Multi-column grid
                for col in range(col_count):
                    if plot_index < num_plots:
                        plot_objects[plot_index].__plot__(self, axes[row][col])
                        plot_index += 1
                    else:
                        # Remove unused subplots
                        axes[row][col].remove()
            elif row_count > 1:
                # Single column, multiple rows
                if plot_index < num_plots:
                    plot_objects[plot_index].__plot__(self, axes[row])
                    plot_index += 1
            else:
                # Single subplot
                if plot_index < num_plots:
                    plot_objects[plot_index].__plot__(self, axes)
                    plot_index += 1

    def _reverse_time_axis(self, fig) -> None:
        """Reverse the time axis for all subplots.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure containing the axes to modify
        """
        from matplotlib.ticker import FuncFormatter

        from esbmtk import Q_

        from .utility_functions import reverse_tick_labels_factory

        t_max = Q_(f"{self.time[-1]} {self.t_unit}").to(self.d_unit).magnitude
        for ax in fig.get_axes():
            ax.invert_xaxis()
            ax.xaxis.set_major_formatter(
                FuncFormatter(reverse_tick_labels_factory(t_max))
            )

    def run(self, **kwargs) -> None:
        """Run the model simulation.

        Executes the model simulation by solving the system of ordinary
        differential equations (ODEs) that describe the model dynamics.

        Parameters
        ----------
        **kwargs : dict
            Optional keyword arguments to control the simulation:

            solver : str, default="ode"
                The solver type to use. Currently only "ode" is supported.

            method : str, default="BDF"
                The integration method for the ODE solver.
                Options include "BDF" and "LSODA".

            stype : str, default="solve_ivp"
                The solver function to use. Currently only "solve_ivp" is supported.

        Returns
        -------
        None
            Results are stored in the model instance.

        Raises
        ------
        ModelError
            If an unsupported solver type is specified.
        SolverError
            If the solver fails to find a solution.

        Notes
        -----
        After running, performance metrics (CPU time, memory usage) are printed.
        """
        # Track execution time and resource usage
        wall_clock_start = time.time()
        cpu_start = process_time()
        # Run solver
        self._ode_solver(kwargs)

        # Mark model as executed
        self.state = 1

        # Calculate and display performance metrics
        cpu_duration = process_time() - cpu_start
        wall_clock_duration = time.time() - wall_clock_start

        print(
            f"\n Execution took {cpu_duration:.2f} CPU seconds, wall time = {wall_clock_duration:.2f} seconds\n"
        )

        # Get memory usage
        process = psutil.Process(os.getpid())
        memory_gb = process.memory_info().rss / 1e9
        print(f"This run used {memory_gb:.2f} GB of memory\n")

    def _write_temp_equations(self, cwd, R, icl, cpl, ipl):
        """Write temporary equations file and return the equation set.

        Creates a temporary Python module containing the model equations,
        imports it, and returns the equation set function.

        Parameters
        ----------
        cwd : str or Path
            Current working directory
        R : ndarray
            Initial conditions
        icl : list
            List of initial condition objects
        cpl : list
            List of constant parameters
        ipl : list
            List of initial parameters

        Returns
        -------
        function
            The equations function imported from the temporary module
        """
        import pathlib as pl

        # Set temporary directory to current working directory
        tempfile.tempdir = cwd

        # Create a temporary Python file
        with tempfile.NamedTemporaryFile(suffix=".py") as tmp_file:
            # Get path to temporary file
            equations_file_path = pl.Path(tmp_file.name)

            # Generate equations module
            equations_module_name = write_equations_3(
                self, R, icl, cpl, ipl, equations_file_path
            )

            # Import the equations module and get the equations function
            equations_set = __import__(equations_module_name).eqs

        return equations_set

    def _ode_solver(self, kwargs: dict):
        """Initialize and run the ODE solver.

        Sets up the system of ODEs, generates the equation file, and solves
        the system using scipy's solve_ivp.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments to control the solver behavior

        Raises
        ------
        SolverError
            If the solver fails to find a solution
        """
        # Get initial conditions and build equation matrices
        R, icl, cpl, ipl, atol = get_initial_conditions(self, self.rtol)

        # Store variables for later use
        self.R = R
        self.icl = icl
        self.cpl = cpl
        self.ipl = ipl

        # Build coefficient matrix
        self.CM, self.F = build_eqs_matrix(self)

        # Set up paths for equation files
        current_dir = Path.cwd()
        sys.path.append(str(current_dir))  # Required on Windows
        equation_filename = "equations.py"
        equation_file_path = current_dir / equation_filename

        # Handle equation file based on debug settings
        equations_set = self._handle_equation_file(
            equation_file_path, R, icl, cpl, ipl, current_dir
        )

        # Get solver configuration from kwargs
        method = kwargs.get("method", "BDF")

        # Initialize carbonate chemistry tables if not present
        self._initialize_carbonate_tables()

        # Run the ODE solver
        self._run_solve_ivp(R, equations_set, method, atol)

        # Process results
        self._process_solver_results()

    def _handle_equation_file(self, equation_file_path, R, icl, cpl, ipl, current_dir):
        """Handle equation file generation based on debug settings.

        Parameters
        ----------
        equation_file_path : Path
            Path to the equation file
        R, icl, cpl, ipl : various
            Parameters for equation generation
        current_dir : Path
            Current working directory

        Returns
        -------
        function
            The equations function
        """
        # If debugging equations is enabled
        if self.debug_equations_file:
            if equation_file_path.exists():
                warnings.warn(
                    "\n\n Warning re-using the equations file \n"
                    "\n type y to proceed. Any other key will delete the file and create a new one",
                    stacklevel=2,
                )
                user_input = input("type y/n: ")

                if user_input.lower() == "y":  # Use existing file
                    equation_module_name = equation_file_path.stem
                else:  # Create new file
                    equation_file_path.unlink()  # Delete old file
                    equation_module_name = write_equations_3(
                        self, R, icl, cpl, ipl, equation_file_path
                    )
            else:  # First run - create persistent file
                equation_module_name = write_equations_3(
                    self, R, icl, cpl, ipl, equation_file_path
                )

            # Import equations
            equations_set = __import__(equation_module_name).eqs
        else:
            # Use temporary file for equations
            if equation_file_path.exists():
                equation_file_path.unlink()

            equations_set = self._write_temp_equations(current_dir, R, icl, cpl, ipl)

        return equations_set

    def _initialize_carbonate_tables(self):
        """Initialize carbonate chemistry tables with default values if not present."""
        if not hasattr(self, "area_table"):
            self.area_table = 0
            self.area_dz_table = 0
            self.Csat_table = 0

    def _run_solve_ivp(self, R, equations_set, method, atol):
        """Run the solve_ivp ODE solver.

        Parameters
        ----------
        R : ndarray
            Initial conditions
        equations_set : function
            The ODE function
        method : str
            Integration method
        atol : float or ndarray
            Absolute tolerance
        """
        self.results = solve_ivp(
            equations_set,
            (self.time[0], self.time[-1]),
            R,
            args=(
                self,
                self.gpt,
                self.toc,  # Tuple of constants
                self.area_table,
                self.area_dz_table,
                self.Csat_table,
                self.CM,  # Coefficient matrix
                self.F,  # Flux vector
            ),
            method=method,
            atol=atol,
            rtol=self.rtol,
            t_eval=self.time_ode,
            first_step=self.min_timestep,
            max_step=self.dt,
            vectorized=False,  # Flux equations would need to be adjusted
        )

    def _process_solver_results(self):
        """Process the solver results and handle errors.

        Raises
        ------
        SolverError
            If the solver fails to find a solution
        """
        if self.results.status == 0:
            # Print solver statistics
            print(
                f"\nnfev={self.results.nfev}, njev={self.results.njev}, nlu={self.results.nlu}\n"
            )
            print(f"status={self.results.status}")
            print(f"message={self.results.message}\n")

            # Process data
            self.post_process_data(self.results)
        else:
            # Raise error with helpful message for failed solutions
            error_message = (
                "---------------------- Warning ------------------------\n"
                "No solution was obtained, check "
                "https://esbmtk.readthedocs.io/en/latest/manual/manual-6.html\n"
                "---------------------- Warning ------------------------\n"
            )
            raise SolverError(error_message)

    def get_delta_values(self) -> None:
        """Calculate reservoir masses and isotope delta values.

        Updates the mass (m) and delta (d) values for all reservoirs
        in the model that have isotopes enabled. For each reservoir,
        the mass is calculated from concentration and volume, and
        the delta value is calculated using the get_delta_h function.

        Parameters
        ----------
        None

        Returns
        -------
        None
            The method modifies reservoir objects in place.
        """
        for reservoir in self.lor:
            if reservoir.isotopes:
                # Update mass based on concentration and volume
                reservoir.m = reservoir.c * reservoir.volume

                # Calculate isotope delta values
                reservoir.d = get_delta_h(reservoir)

    def sub_sample_data(self) -> None:
        """Reduce data resolution by subsampling time series data.

        If the number of time points exceeds the desired number of data points,
        this method reduces the data resolution by taking every nth point
        (where n is the stride). This affects the time array and all data
        in reservoirs, virtual reservoirs, and fluxes.

        The method is mainly used to reduce memory usage and file sizes
        when saving model output.

        Parameters
        ----------
        None

        Returns
        -------
        None
            The method modifies model data in place.

        Notes
        -----
        Subsampling only occurs if the stride is greater than 1.
        The time series boundaries (first two and last two points) are excluded
        from subsampling to avoid boundary effects.
        """
        # Calculate stride based on current time array and desired number of points
        stride = int(len(self.time) / self.number_of_datapoints)

        # Only subsample if stride is greater than 1
        if stride > 1:
            # Subsample time array, excluding first two and last two points
            self.time = self.time[2:-2:stride]

            # Subsample all reservoir data
            for reservoir in self.lor:
                reservoir.__sub_sample_data__(stride)

            # Subsample all virtual reservoir data
            for virtual_reservoir in self.lvr:
                virtual_reservoir.__sub_sample_data__(stride)

            # Subsample all flux data
            for flux in self.lof:
                flux.__sub_sample_data__(stride)

    def post_process_data(self, results) -> None:
        """Process solver results and update model data structures.

        Takes the raw numerical results from the ODE solver and maps them back
        into the appropriate ESBMTK data structures (reservoirs, signals, fluxes).
        Also performs post-processing operations like interpolating signals,
        calculating derived quantities, and checking for pH stability.

        Parameters
        ----------
        results : scipy.integrate._ivp.ivp.OdeResult
            The results object returned by the ODE solver, containing solution
            time points (t) and state variables (y)

        Returns
        -------
        None
            The method updates model data structures in-place

        Notes
        -----
        The processing order is important:
        1. Interpolate signals and external data to match solver time points
        2. Map state variables to reservoir concentrations and masses
        3. Update time vector and flux data
        4. Perform specialized checks (pH stability) and calculations (carbonate chemistry)
        """
        # Step 1: Interpolate signals to match solver time domain
        self._interpolate_signals_to_solver_timepoints(results)

        # Step 2: Interpolate external data to match solver time domain
        self._interpolate_external_data_to_solver_timepoints(results)

        # Step 3: Map solver state variables to reservoir properties
        self._map_state_variables_to_reservoirs(results)

        # Step 4: Update model time vector to match solver time points
        self.time = results.t

        # Step 5: Update flux data to match solver time steps
        steps = len(results.t)  # Get number of solver steps
        self._update_flux_data(steps)

        # Step 6: Perform specialized post-processing
        self._perform_specialized_post_processing(results)

    def _interpolate_signals_to_solver_timepoints(self, results) -> None:
        """Interpolate signal data to match solver time points.

        Parameters
        ----------
        results : scipy.integrate._ivp.ivp.OdeResult
            The ODE solver results
        """
        for signal in self.los:
            # Interpolate mass data
            signal.signal_data.m = np.interp(results.t, self.time, signal.signal_data.m)

            # Interpolate isotope data if present
            if signal.isotopes:
                signal.signal_data.l = np.interp(
                    results.t, self.time, signal.signal_data.l
                )

    def _interpolate_external_data_to_solver_timepoints(self, results) -> None:
        """Interpolate external data to match solver time points.

        Parameters
        ----------
        results : scipy.integrate._ivp.ivp.OdeResult
            The ODE solver results
        """
        for external_data in self.led:
            external_data.y = np.interp(results.t, external_data.x, external_data.y)

    def _map_state_variables_to_reservoirs(self, results) -> None:
        """Map solver state variables to reservoir properties.

        Parameters
        ----------
        results : scipy.integrate._ivp.ivp.OdeResult
            The ODE solver results
        """
        state_index = 0

        for reservoir in self.icl:
            # Update reservoir concentration
            reservoir.c = results.y[state_index]

            # Update reservoir mass (assumes constant volume)
            # Note: This would need modification for variable volumes
            reservoir.m = results.y[state_index] * reservoir.volume

            # Move to next state variable
            state_index += 1

            # Process isotope data if present
            if reservoir.isotopes:
                reservoir.l = results.y[state_index]
                state_index += 1

                # Calculate delta values from concentrations
                reservoir.d = get_delta_from_concentration(
                    reservoir.c, reservoir.l, reservoir.sp.r
                )

    def _update_flux_data(self, steps: int) -> None:
        """Update flux data to match solver time steps.

        Parameters
        ----------
        steps : int
            Number of time steps in the solver results
        """
        for flux in self.lof:
            if flux.save_flux_data:
                # Trim flux mass data to match time steps
                flux.m = flux.m[0:steps]

                # Process isotope data if present
                if flux.isotopes:
                    flux.l = flux.l[0:steps]
                    flux.d = get_delta_h(flux)

    def _perform_specialized_post_processing(self, results) -> None:
        """Perform specialized post-processing tasks.

        Parameters
        ----------
        results : scipy.integrate._ivp.ivp.OdeResult
            The ODE solver results
        """
        from esbmtk import carbonate_system_1_pp

        for reservoir_group in self.lrg:
            # Check for pH stability if hydrogen ions are present
            if hasattr(reservoir_group, "Hplus"):
                self.test_d_pH(reservoir_group, results.t)

            # Calculate carbonate system parameters if needed
            if reservoir_group.has_cs1:
                carbonate_system_1_pp(reservoir_group)

    def test_d_pH(self, reservoir_group: Species, time_vector: NDArrayFloat) -> None:
        """Test for large changes in pH between time steps.

        Checks if the pH change between consecutive time steps exceeds 0.01 units,
        which could indicate numerical instability or unrealistic model behavior.
        Warnings are issued for any time steps where the threshold is exceeded.

        Parameters
        ----------
        reservoir_group : Species
            The reservoir group containing a Hplus species to be checked

        time_vector : NDArrayFloat
            Time vector as returned by the solver

        Returns
        -------
        None
            Issues warnings if large pH changes are detected

        Notes
        -----
        This is a crude test since the solver interpolates between integration steps,
        so it may not catch all problems. It only identifies pH changes that exceed
        0.01 units between the specific time points in the solution.

        The pH is calculated as -log10([H+]), where [H+] is the hydrogen ion concentration.
        """
        # Access the hydrogen ion concentration data
        hydrogen_ions = reservoir_group.Hplus

        # Calculate pH from hydrogen ion concentration and get differences between steps
        pH_values = -np.log10(hydrogen_ions.c)
        pH_changes = np.diff(pH_values)

        # Find time steps where pH change exceeds threshold
        pH_threshold = 0.01
        large_pH_changes = pH_changes > pH_threshold

        # If any large changes were found, issue warnings
        if np.any(large_pH_changes):
            for i, is_large_change in enumerate(large_pH_changes):
                if is_large_change:
                    warnings.warn(
                        f"\n\n{reservoir_group.full_name} delta pH = {pH_changes[i]:.2f} "
                        f"at t = {time_vector[i]:.2f} {self.t_unit:~P}\n",
                        stacklevel=2,
                    )

    def list_species(self) -> None:
        """Display all elements and species defined in the model.

        Prints a hierarchical list of all elements in the model and their
        associated species properties. This provides a quick overview of
        the chemical species available in the model simulation.

        Parameters
        ----------
        None

        Returns
        -------
        None
            This method prints to stdout but doesn't return a value.

        Examples
        --------
        >>> model.list_species()
        Currently defined elements and their species:
        Carbon
          Defined SpeciesProperties:
            DIC
            CO2
            HCO3
            CO3
        Sulfur
          Defined SpeciesProperties:
            SO4
            H2S
        """
        # Print header
        print("\nCurrently defined elements and their species:")

        # Iterate through each element
        for element in self.lel:
            # Display element name
            print(f"{element}")
            print("  Defined SpeciesProperties:")

            # Display all species for this element
            for species in element.lsp:
                print(f"    {species.n}")

    def flux_summary(self, **kwargs: dict) -> list | None:
        """Display or return a filtered summary of model fluxes.

        Creates a report of fluxes in the model, filtered by name patterns.
        Can either print the results to the console or return them as a list.

        Parameters
        ----------
        **kwargs : dict
            Optional keyword arguments:

            filter_by : str, default=""
                Filter fluxes by name or partial name. Multiple words separated
                by spaces act as additional conditions - all words must appear
                in the flux name.

            exclude : str, default=""
                Exclude any fluxes whose names contain this string.

            return_list : bool, default=False
                If True, return a list of flux objects instead of printing to console.

        Returns
        -------
        list or None
            If return_list=True, returns a list of flux objects matching the filters.
            If return_list=False, returns None (results are printed to console).

        Raises
        ------
        ModelError
            If the deprecated "filter" parameter is used instead of "filter_by".

        Examples
        --------
        # Display all fluxes containing "PO4" in their name
        >>> model.flux_summary(filter_by="PO4")

        # Get a list of fluxes containing both "POP" and "A_sb" in their names
        >>> fluxes = model.flux_summary(filter_by="POP A_sb", return_list=True)

        # Display fluxes containing "PO4" but not "H_sb"
        >>> model.flux_summary(filter_by="PO4", exclude="H_sb")
        """
        # Get filter parameters from kwargs with proper defaults
        filter_terms = (
            kwargs.get("filter_by", "").split() if "filter_by" in kwargs else []
        )
        exclude_term = kwargs.get("exclude", "")
        return_as_list = kwargs.get("return_list", False)

        # Check for deprecated parameter
        if "filter" in kwargs:
            raise ModelError("use filter_by instead of filter")

        # Print header if displaying results
        if not return_as_list:
            print(f"\n --- Flux Summary -- filtered by {filter_terms}\n")

        # Initialize result list
        matching_fluxes = []

        # Find all fluxes that match the filter criteria
        for flux in self.lof:
            # Check if flux name matches all filter terms and doesn't contain exclude term
            if find_matching_strings(flux.full_name, filter_terms) and (
                not exclude_term or exclude_term not in flux.full_name
            ):
                matching_fluxes.append(flux)

                # Print flux name if not returning a list
                if not return_as_list:
                    print(f"{flux.full_name}")

        # Return results based on the return_list parameter
        return matching_fluxes if return_as_list else None

    def connection_summary(self, **kwargs) -> None:
        """Display a summary of model connections.

        Prints information about all connections in the model or a filtered subset.
        For each connection, shows source and target, plus additional attributes
        if requested.

        Parameters
        ----------
        **kwargs : dict
            Optional keyword arguments:

            conname : str, default=None
                If provided, only show connections containing this substring in their name.

            list_all : bool, default=False
                If True, print all connection attributes including internal ones.

        Returns
        -------
        None
            This method prints to stdout but doesn't return a value.

        Examples
        --------
        >>> model.connection_summary()  # Show all connections

        >>> model.connection_summary(conname="DIC")  # Show only DIC connections

        >>> model.connection_summary(list_all=True)  # Show all connection details
        """
        # Extract configuration from kwargs
        show_all_attributes = kwargs.get("list_all", False)
        connection_name_filter = kwargs.get("conname")

        # Get filtered list of connections
        filtered_connections = self._filter_connections(connection_name_filter)

        # Exit early if no matching connections are found
        if not filtered_connections:
            self._report_no_connections(connection_name_filter)
            return

        # Display each connection
        print("")
        for connection in filtered_connections:
            self._display_connection_info(connection, show_all_attributes)

    def _filter_connections(self, name_filter: str = None) -> list:
        """Filter connections by name.

        Parameters
        ----------
        name_filter : str, default=None
            Substring to match in connection names

        Returns
        -------
        list
            Filtered list of connection objects
        """
        filtered_list = []

        for connection in self.loc:
            # If name filter is provided, only include connections with matching names
            if name_filter is not None:
                if name_filter in connection.n:  # Substring search
                    filtered_list.append(connection)
            else:
                # No filter - include all connections
                filtered_list.append(connection)

        return filtered_list

    def _report_no_connections(self, name_filter: str = None) -> None:
        """Report when no connections are found.

        Parameters
        ----------
        name_filter : str, default=None
            The filter string that was used (if any)
        """
        if name_filter is not None:
            print(f"No connections with name '{name_filter}' found")
        else:
            print("No connections found")

    def _display_connection_info(self, connection, show_all_attributes: bool) -> None:
        """Display information about a single connection.

        Parameters
        ----------
        connection : Connection
            The connection object to display
        show_all_attributes : bool
            Whether to show all attributes of the connection
        """
        # Get basic source and target info
        source = connection.source_name
        target = connection.target_name

        # Display connection header with appropriate format based on connection type
        if isinstance(connection, Species2Species):
            # For species-to-species connections, show the specific species
            source_species = f"{connection.source.sp.n}"
            target_species = f"{connection.target.sp.n}"
            print(
                f"Connection: {connection.id}: {source}.{source_species} -> {target}.{target_species}"
            )
        else:
            # For reservoir-to-reservoir connections
            print(f"Connection: {connection.id}: {source} -> {target}")

        # Display connection attributes
        self._display_connection_attributes(connection, show_all_attributes)

        # Add empty line after each connection for readability
        print("")

    def _display_connection_attributes(
        self, connection, show_all_attributes: bool
    ) -> None:
        """Display the attributes of a connection.

        Parameters
        ----------
        connection : Connection
            The connection object
        show_all_attributes : bool
            Whether to show all attributes
        """
        # If all attributes requested, show the entire __dict__
        if show_all_attributes:
            print(f"    {connection.__dict__}")
            return

        # Otherwise, show only selected attributes
        excluded_attributes = [
            "source",
            "target",
            "flux",
            "source_name",
            "target_name",
            *self.olkk,  # Optional keywords to exclude
        ]

        for attr_name, attr_value in connection.__dict__.items():
            # Skip private attributes and excluded ones
            if attr_name[0] != "_" and attr_name not in excluded_attributes:
                print(f"    {attr_name}: {attr_value}")

    def clear(self):
        """Delete all model objects."""
        for o in self.lmo:
            print(f"deleting {o}")
            del __builtins__[o]

    def __init_dimensionalities__(self, ureg):
        """No longer needed."""
        raise NotImplementedError()
        """Test the dimensionality of input data."""
        self.substance_per_volume_d = ureg("mol/liter").dimensionality
        self.substance_per_mass_d = ureg("mol/kg").dimensionality
        self.substance_d = ureg("mol").dimensionality
        self.mass_d = ureg("kg").dimensionality
        self.length_d = ureg("m").dimensionality
        self.flux_d = ureg("mol/s").dimensionality
        self.time_d = ureg("s").dimensionality


class ElementProperties(esbmtkBase):
    r"""Each model, can have one or more elements.

    This class sets element specific properties

    Example::

        ElementProperties(name      = "S "           # the element name
                model     = Test_model     # the model handle
                mass_unit =  "mol",        # base mass unit
                li_label  =  "$^{32$S",    # Label of light isotope
                hi_label  =  "$^{34}S",    # Label of heavy isotope
                d_label   =  r"$\delta^{34}$S",  # Label for delta value
                d_scale   =  "VCDT",       # Isotope scale
                r         = 0.044162589,   # isotopic abundance ratio for element
                reference = "https link or citation",
              )
    """

    # set element properties
    def __init__(self, **kwargs) -> any:
        """Initialize all instance variables.

        Defaults are as follows::

            self.defaults: dict[str, tp.List[any, tuple]] = {
               "name": ["M", (str)],
               "model": ["None", (str, Model)],
               "register": ["None", (str, Model)],
               "full_name": ["None", (str)],
               "li_label": ["None", (str)],
               "hi_label": ["None", (str)],
               "d_label": ["None", (str)],
               "d_scale": ["None", (str)],
               "r": [1, (float, int)],
               "mass_unit": ["mol", (str, Q_)],
               "parent": ["None", (str, Model)],
               "reference": ["None", (str)],

        }

        Required keywords: "name", "model", "mass_unit"
        """
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
            "mass_unit": ["", (str, Q_)],
            "parent": ["None", (str, Model)],
            "reference": ["None", (str)],
        }

        # list of absolutely required keywords
        self.lrk: list = ["name", "model", "mass_unit"]
        self.__initialize_keyword_variables__(kwargs)

        self.parent = self.model
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

        self.__register_with_parent__()

    def list_species(self) -> None:
        """List all species which are predefined for this element."""
        for e in self.lsp:
            print(e.n)

    def __register_species_with_model__(self) -> None:
        """Bit of hack, but makes model code more readable."""
        for s in self.lsp:
            setattr(self.model, s.name, s)


class SpeciesProperties(esbmtkBase):
    """Each model, can have one or more species.

    This class sets species specific properties

    Example::

        SpeciesProperties(name = "SO4",
                element = S,

    )

    Defaults::

        self.defaults: dict[any, any] = {
            "name": ["None", (str)],
            "element": ["None", (ElementProperties, str)],
            "display_as": [kwargs["name"], (str)],
            "m_weight": [0, (int, float, str)],
            "register": ["None", (Model, ElementProperties, Species, GasReservoir)],
            "parent": ["None", (Model, ElementProperties, Species, GasReservoir)],
            "flux_only": [False, (bool)],
            "logdata": [False, (bool)],
            "scale_to": ["None", (str)],
            "stype": ["concentration", (str)],
        }

    Required keywords: "name", "element"
    """

    # set species properties
    def __init__(self, **kwargs) -> None:
        """Initialize all instance variables."""
        from esbmtk import GasReservoir

        # provide a list of all known keywords
        self.defaults: dict[any, any] = {
            "name": ["None", (str)],
            "element": ["None", (ElementProperties, str)],
            "display_as": [kwargs["name"], (str)],
            "m_weight": [0, (int, float, str)],
            "register": [
                kwargs["element"],
                (Model, ElementProperties, Species, GasReservoir),
            ],
            "parent": ["None", (Model, ElementProperties, Species, GasReservoir)],
            "flux_only": [False, (bool)],
            "logdata": [False, (bool)],
            "scale_to": ["mmol", (str)],
            "stype": ["concentration", (str)],
        }

        # provide a list of absolutely required keywords
        self.lrk = ["name", "element"]
        self.__initialize_keyword_variables__(kwargs)
        self.parent = self.register

        if "display_as" not in kwargs:
            self.display_as = self.name

        # legacy names
        self.n = self.name  # display name of species
        self.mass_unit = self.element.mass_unit
        self.mu = self.mass_unit  # display name of mass unit
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
        self.__register_with_parent__()


class SpeciesBase(esbmtkBase):
    """Base class for all Species objects."""

    def __init__(self, **kwargs) -> None:
        """Instantiate Class."""
        raise NotImplementedError(
            "SpeciesBase should never be used. Use the derived classes"
        )

    def __set_legacy_names__(self, kwargs) -> None:
        """Move the below out of the way."""
        from esbmtk.sealevel import get_box_geometry_parameters

        self.atol: list[float] = [1.0, 1.0]  # tolerances
        self.lof: list[Flux] = []  # flux references
        self.led: list[ExternalData] = []  # all external data references
        self.lio: dict[str, int] = {}  # flux name:direction pairs
        self.lop: list[Process] = []  # list holding all processe references
        self.loe: list[ElementProperties] = []  # list of elements in thiis reservoir
        self.doe: dict[SpeciesProperties, Flux] = {}  # species flux pairs
        self.loc: set[Species2Species] = set()  # set of connection objects
        self.ldf: list[DataField] = []  # list of datafield objects
        # list of processes which calculate reservoirs
        self.lpc: list[Process] = []
        self.ef_results = False  # Species has external function results

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

        self.sp: SpeciesProperties = self.species  # species handle
        self.mo: Model = self.species.mo  # model handle
        self.model = self.mo
        self.rvalue = self.sp.r
        self.m_unit = self.model.m_unit
        self.v_unit = self.model.v_unit
        self.c_unit = self.model.c_unit

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

        if self.geometry != "None" and self.geometry_unset:
            get_box_geometry_parameters(self)

        if self.display_precision == 0:
            self.display_precision = self.mo.display_precision

        self.parent = self.register

    def __setitem__(self, i: int, value: float):
        """Create a placeholder setitem function."""
        return self.__set_data__(i, value)

    def __call__(self) -> None:  # what to do when called as a function ()
        """Return self when called as a function."""
        return self

    def __getitem__(self, i: int) -> NDArrayFloat:
        """Get flux data by index."""
        return np.array([self.m[i], self.l[i], self.c[i]])

    def __set_with_isotopes__(self, i: int, value: float) -> None:
        """Set values when isotope data is present.

        :param i: index
        :param value: array of [mass, li, hi, d]

        """
        self.m[i]: float = value[0]
        # update concentration and delta next. This is computationally inefficient
        # but the next time step may depend on on both variables.
        self.c[i]: float = value[0] / self.v[i]  # update concentration
        self.l[i]: float = value[1] / self.v[i]  # update concentration

    def __set_without_isotopes__(self, i: int, value: float) -> None:
        """Set values when no isotope data is present.

        :param i: index
        :param value: array of [mass]

        """
        self.m[i]: float = value[0]
        self.c[i]: float = self.m[i] / self.v[i]  # update concentration

    def __update_mass__() -> None:
        """Place holder function."""
        raise NotImplementedError("__update_mass__ is not yet implmented")

    def __write_data__(
        self,
        prefix: str,
        start: int,
        stop: int,
        stride: int,
        append: bool,
        directory: str,
    ) -> None:
        """Write data to file.

        This function is called by the write_data() and save_state() methods

        :param prefix:
        :param start:
        :param stop:
        :param stride:
        :param append:
        :param directory:
        """
        from pathlib import Path

        p = Path(directory)
        p.mkdir(parents=True, exist_ok=True)

        # some short hands
        sn = self.sp.n  # species name
        sp = self.sp  # species handle
        mo = self.sp.mo  # model handle

        # smu = f"{mo.m_unit:~P}"
        mtu = f"{mo.t_unit:~P}"
        # fmu = f"{mo.f_unit:~P}"
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
            raise SpeciesError(
                f"Model register keyword must be 'None'/'local' not {self.sp.mo.register}"
            )

        # build the dataframe
        df: pd.dataframe = DataFrame()

        df[f"{rn} Time [{mtu}]"] = self.mo.time[start:stop:stride]  # time
        # df[f"{rn} {sn} [{smu}]"] = self.m.to(self.mo.m_unit).magnitude[start:stop:stride]  # mass
        if self.isotopes:
            # print(f"rn = {rn}, sp = {sp.name}")
            df[f"{rn} {sp.ln} [{cmu}]"] = self.l[start:stop:stride]  # light isotope
        df[f"{rn} {sn} [{cmu}]"] = self.c[start:stop:stride]  # concentration

        file_path = Path(fn)
        if append and file_path.exists():
            df.to_csv(file_path, header=False, mode="a", index=False)
        else:
            df.to_csv(file_path, header=True, mode="w", index=False)
        return df

    def __sub_sample_data__(self, stride) -> None:
        """Subsample the results before saving processing."""
        self.m = self.m[2:-2:stride]
        self.l = self.l[2:-2:stride]
        self.c = self.c[2:-2:stride]

    def __reset_state__(self) -> None:
        """Copy the result of the last computation.

        beginning so that a new run will start with these values

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
        """Replace the data fields with saved values."""
        self.m = self.mc
        self.c = self.cc
        # self.d = self.dc

    def __read_state__(self, directory: str, prefix="state_") -> None:
        """Read data from csv-file into a dataframe.

        The CSV file must have the following columns
            - Model Time t
            - Species_Name m
            - Species_Name l
            - Species_Name h
            - Species_Name d
            - Species_Name c
            - Flux_name m
            - Flux_name l etc etc.
        """
        read: set = set()
        curr: set = set()

        if self.sp.mo.register == "None":
            fn = f"{directory}/{prefix}{self.mo.n}_{self.full_name}.csv"
        elif self.sp.mo.register == "local":
            fn = f"{directory}/{prefix}{self.full_name}.csv"
        else:
            raise SpeciesError(
                f"Model register keyword must be 'None'/'local' not {self.sp.mo.register}"
            )

        if not os.path.exists(fn):
            raise FileNotFoundError(
                f"Flux {fn} does not exist in Species {self.full_name}"
            )

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
        for n in header_list:
            name = n.split(" ")[0]
            logging.debug(f"Looking for {name}")
            # this finds the reservoir name
            if name == self.full_name:
                # logging.debug(f"found reservoir data for {name}")
                col = self.__assign_reservoir_data__(self, df, col, True)
            else:
                raise SpeciesError(f"Unable to find Flux {n} in {self.full_name}")

        # test if we missed any fluxes
        for f in list(curr.difference(read)):
            warnings.warn(
                f"\nDid not find values for {f}\n in saved state", stacklevel=2
            )

    def __assign_reservoir_data__(
        self, obj: any, df: pd.DataFrame, col: int, res: bool
    ) -> int:
        """Assign the third last entry data to all values in reservoir.

        :param obj: # Species
        :param df: pd.dataframe
        :param col: int # index into column position
        :param res: True # indicates whether obj is reservoir

        :returns: int # index into last column
        """
        if obj.isotopes:
            obj.l[:] = df.iloc[-1, col]  # get last row
            col += 1
            obj.c[:] = df.iloc[-1, col]
            col += 1
        else:
            # v = df.iloc[-1, col]
            obj.c[:] = df.iloc[-1, col]
            col += 1

        return col

    def get_plot_format(self):
        """Return concentrat data in plot units."""
        from pint import Unit

        if isinstance(self.plt_units, Q_):
            unit = f"{self.plt_units.units:~P}"
        elif isinstance(self.plt_units, Unit):
            unit = f"{self.plt_units:~P}"
        else:
            unit = f"{self.plt_units}"

        y1_label = f"{self.legend_left} [{unit}]"

        if self.display_as == "mass":
            y1 = (self.m * self.mo.m_unit).to(self.plt_units).magnitude
        elif self.display_as == "ppm":
            y1 = self.c * 1e6
            y1_label = "ppm"
        elif self.display_as == "length":
            y1 = (self.c * self.mo.l_unit).to(self.plt_units).magnitude
        else:
            y1 = (self.c * self.mo.c_unit).to(self.plt_units).magnitude

        # test for plt_transform
        if self.plot_transform_c != "None":
            if callable(self.plot_transform_c):
                y1 = self.plot_transform_c(self.c)
            else:
                raise SpeciesError("Plot transform must be a function")

        return y1, y1_label, unit

    def __plot__(self, M: Model, ax) -> None:
        """Plot Model data.

        :param M: Model
        :param ax: # graph axes handle
        """
        from esbmtk.utility_functions import set_y_limits

        # convert time and data to display units
        x = (M.time * M.t_unit).to(M.d_unit).magnitude

        y1, y1_label, _unit = self.get_plot_format()

        # plot first axis
        ax.plot(x[1:-2], y1[1:-2], color="C0", label=y1_label)
        ax.set_xlabel(f"{M.time_label} [{M.d_unit:~P}]")
        ax.set_ylabel(y1_label)

        # add any external data if present
        for i, d in enumerate(self.led):
            leg = f"{self.lm} {d.legend}"
            ax.scatter(d.x[1:-2], d.y[1:-2], color=f"C{i + 2}", label=leg)

        ax.spines["top"].set_visible(False)
        handler1, label1 = ax.get_legend_handles_labels()
        set_y_limits(ax, self)

        if self.isotopes:
            axt = ax.twinx()
            y2 = self.d  # no conversion for isotopes
            axt.plot(x[1:-2], y2[1:-2], color="C1", label=self.legend_right)
            axt.set_ylabel(self.ld)
            set_y_limits(axt, self)
            ax.spines["top"].set_visible(False)
            # set combined legend
            handler2, label2 = axt.get_legend_handles_labels()
            axt.legend(handler1 + handler2, label1 + label2, loc=0).set_zorder(6)
        else:
            ax.legend(handler1, label1)
            ax.spines["right"].set_visible(False)
            ax.yaxis.set_ticks_position("left")
            ax.xaxis.set_ticks_position("bottom")

        ax.set_title(self.full_name)

    def info(self, **kwargs) -> None:
        """Show an overview of the object properties.

        Optional arguments are:

        :param index: int = 0 # this will show data at the given index
        :param indent: int = 0 # print indentation
        """
        off: str = "  "
        # index = kwargs.get("index", 0)
        if "indent" not in kwargs:
            indent = 0
            ind = ""
        else:
            indent = kwargs["indent"]
            ind = " " * indent

        # print basic data bout this reservoir
        print(f"{ind}{self.__str__(kwargs)}")
        print(f"{ind}Data sample:")
        # show_data(self, index=index, indent=indent)

        print(f"\n{ind}Connnections:")
        for p in sorted(self.loc):
            print(f"{off}{ind}{p.full_name}.info()")

        print(f"\n{ind}Fluxes:")

        # m = Q_("1 Sv").to("l/a").magnitude
        for i, f in enumerate(self.lof):
            print(f"{off}{ind}{f.full_name}: {self.lodir[i] * f.m[-2]:.2e}")

        print()
        print("Use the info method on any of the above connections")
        print("to see information on fluxes and processes")


class Species(SpeciesBase):
    """Species specific information data fields.

    Example::

        Species(name = "foo",      # Name of reservoir
                  species = S,          # SpeciesProperties handle
                  delta = 20,           # initial delta - optional (defaults  to 0)
                  mass/concentration = "1 unit"  # species concentration or mass
                  volume/geometry = "1E5 l",      # reservoir volume (m^3)
                  plot = "yes"/"no", defaults to yes
                  plot_transform_c = a function reference, optional (see below)
                  legend_left = str, optional, useful for plot transform
                  display_precision = number, optional, inherited from Model
                  register = Model instance
                  isotopes = True/False otherwise use Model.m_type
                  seawater_parameters= dict, optional
                  )

    You must either give mass or concentration.  The result will
    always be displayed as concentration though.

    You must provide either the volume or the geometry keyword.  In
    the latter case provide a list where the first entry is the upper
    depth datum, the second entry is the lower depth datum, and the
    third entry is the total ocean area.  E.g., to specify the upper
    200 meters of the entire ocean, you would write:

    geometry=[0,-200,3.6e14]

    the corresponding ocean volume will then be calculated by the
    calc_volume method in this case the following instance variables
    will also be set:

    self.volume in model units (usually liter) self.are:a surface area
    in m^2 at the upper bounding surface self.sed_area: area of
    seafloor which is intercepted by this box.  self.area_fraction:
    area of seafloor which is intercepted by this relative to the
    total ocean floor area

    It is also possible to specify volume and area explicitly. In this
    case provide a dictionary like this::

        geometry = {"area": "1e14 m**2", # surface area
                    "volume": "3e16 m**3", # box volume
                   }

    Adding seawater_properties:
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~

    If this optional parameter is specified, a SeaWaterConstants instance will
    be registered for this Species as Species.swc See the
    SeaWaterConstants class for details how to specify the parameters,
    e.g.:

    .. code-block:: python

            seawater_parameters = {"temperature": 2,
                                   "pressure": 240,
                                   "salinity" : 35,
                                  }

    Using a transform function:
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~

    In some cases, it is useful to transform the reservoir
    concentration data before plotting it.  A good example is the H+
    concentration in water which is better displayed as pH.  We can do
    this by specifying a function to convert the reservoir
    concentration into pH units::

    .. code-block:: python

        def phc(c :float) -> float:
            # Calculate concentration as pH. c can be a number or numpy array
            import numpy as np
            pH :float = -np.log10(c)
            return pH

    this function can then be added to a reservoir as:

    hplus.plot_transform_c = phc

    You can modify the left legend to suit the transform via the
    legend_left keyword

    Note, at present the plot_transform_c function will only take one
    argument, which always defaults to the reservoir concentration.
    The function must return a single argument which will be
    interpreted as the transformed reservoir concentration.

    Accesing Species Data:
    ~~~~~~~~~~~~~~~~~~~~~~~~

    You can access the reservoir data as:

        - Name.m # mass

        - Name.d # delta

        - Name.c # concentration

    Useful methods include:

        - Name.write_data() # save data to file

        - Name.info() # info Species

    """

    def __init__(self, **kwargs) -> None:
        """Initialize a reservoir.

        Defaults::

            self.defaults: dict[str, tp.List[any, tuple]] = {
              "name": ["None", (str)],
              "species": ["None", (str, SpeciesProperties)],
              "delta": ["None", (int, float, str)],
              "concentration": ["None", (str, Q_, float)],
              "mass": ["None", (str, Q_)],
              "volume": ["None", (str, Q_)],
              "geometry": ["None", (list, dict, str)],
              "plot_transform_c": ["None", (any)],
              "legend_left": ["None", (str)],
              "plot": ["yes", (str)],
              "groupname": ["None", (str)],
              "rtype": ["regular", (str)],
              "function": ["None", (str, col.Callable)],
              "display_precision": [0.01, (int, float)],
              "register": [
                  "None",
                  (SourceProperties, SinkProperties, Reservoir, ConnectionProperties, Model, str),
              ],
              "parent": [
                  "None",
                  (SourceProperties, SinkProperties, Reservoir, ConnectionProperties, Model, str),
              ],
              "full_name": ["None", (str)],
              "seawater_parameters": ["None", (dict, str)],
              "isotopes": [False, (bool)],
              "ideal_water": ["None", (str, bool)],
              "has_cs1": [False, (bool)],
              "has_cs2": [False, (bool)],

        }

        Required Keywords::

            self.lrk: tp.List = [
              "name",
              "species",
              "register",
              ["volume", "geometry"],
              ["mass", "concentration"],

        ]
        """
        from esbmtk import (
            ConnectionProperties,
            Reservoir,
            SinkProperties,
            SourceProperties,
            phc,
        )
        from esbmtk.sealevel import get_box_geometry_parameters

        # provide a dict of all known keywords and their type
        self.defaults: dict[str, list[any, tuple]] = {
            "name": ["None", (str)],
            "species": ["None", (str, SpeciesProperties)],
            "delta": ["None", (int, float, str)],
            "concentration": ["None", (str, Q_, float)],
            "mass": ["None", (str, Q_)],
            "volume": ["None", (str, Q_)],
            "geometry": ["None", (list, dict, str)],
            "geometry_unset": [True, (bool)],
            "plot_transform_c": ["None", (any)],
            "legend_left": ["None", (str)],
            "plot": ["yes", (str)],
            "groupname": ["None", (str)],
            "rtype": ["regular", (str)],
            "function": ["None", (str, callable)],
            "display_precision": [0.01, (int, float)],
            "register": [
                "None",
                (
                    SourceProperties,
                    SinkProperties,
                    Reservoir,
                    ConnectionProperties,
                    Model,
                    str,
                ),
            ],
            "parent": [
                "None",
                (
                    SourceProperties,
                    SinkProperties,
                    Reservoir,
                    ConnectionProperties,
                    Model,
                    str,
                ),
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
            ["volume", "geometry"],
            ["mass", "concentration"],
        ]

        self.__initialize_keyword_variables__(kwargs)

        if isinstance(self.register, Model):
            self.model = self.register
        else:
            self.model = self.register.model
        self.parent = self.register
        self.c = np.zeros(len(self.model.time))
        self.l = np.zeros(len(self.model.time))
        self.m = np.zeros(len(self.model.time))
        self.__set_legacy_names__(kwargs)

        if self.delta != "None":
            self.isotopes = True

        # geoemtry information
        if self.volume == "None":
            get_box_geometry_parameters(self)
        else:
            self.volume = Q_(self.volume).to(self.mo.v_unit)

        # append reservoir volume to list of toc's
        self.model.toc = (*self.model.toc, self.volume.to(self.model.v_unit).magnitude)
        self.v_index = self.model.gcc
        self.model.gcc = self.model.gcc + 1
        self.c_unit = self.model.c_unit
        # This should probably be species specific?
        self.mu: str = self.sp.e.mass_unit  # massunit xxxx

        if self.sp.stype == "concentration":
            if self.mass == "None":
                if isinstance(self.concentration, str | Q_):
                    cc = Q_(self.concentration)
                    # concentration can be mol/kg or mol/l
                    _sm, sc = str(cc.units).split(" / ")  # get
                    _mm, mc = str(self.mo.c_unit).split(" / ")  # model
                    if mc == "liter" and sc == "kilogram":
                        cc = Q_(f"{cc.magnitude} {str(self.mo.c_unit)}")
                        warnings.warn(
                            "\nConvert mol/kg to mol/liter assuming density = 1\n",
                            stacklevel=2,
                        )
                    elif sc != mc:
                        raise ScaleError(
                            f"no transformation for {cc.units} to {self.mo.c_unit}"
                        )
                    self._concentration = cc.to(self.mo.c_unit)
                    self.plt_units = self.mo.c_unit
                else:
                    cc = self.concentration
                    self.plt_units = self.mo.c_unit
                    self._concentration = cc

                self.mass = (
                    self.concentration.to(self.mo.c_unit).magnitude
                    * self.volume.to(self.mo.v_unit).magnitude
                )
                self.mass = Q_(f"{self.mass} {self.mo.c_unit}")
                self.display_as = "concentration"

                # fixme: c should be dimensionless, not sure why this happens
                self.c = self.c.to(self.mo.c_unit).magnitude

                if self.species.scale_to != "None":
                    _c, m = str(self.mo.c_unit).split(" / ")
                    self.plt_units = Q_(f"{self.species.scale_to} / {m}")
            elif self.concentration == "None":
                m = Q_(self.mass)
                self.plt_units = self.mo.m_unit
                self.mass: int | float = m.to(self.mo.m_unit).magnitude
                self.concentration = self.massto(self.mo.m_unit) / self.volume.to(
                    self.mo.v_unit
                )
                self.display_as = "mass"
            else:
                raise SpeciesError("You need to specify mass or concentration")

        elif self.sp.stype == "length":
            self.plt_units = self.mo.l_unit
            self.c = (
                np.zeros(self.mo.number_of_datapoints + 1)
                + Q_(self.concentration).magnitude
            )
            self.display_as = "length"
        self.state = 0

        # save the unit which was provided by the user for display purposes
        # left y-axis label
        self.lm: str = f"{self.species.n} [{self.mu}/l]"

        # initialize mass vector
        if self.mass == "None":
            self.m: NDArrayFloat = np.zeros(self.mo.number_of_datapoints + 1)
        else:
            self.m: NDArrayFloat = np.zeros(self.species.mo.steps) + self.mass
        self.l: NDArrayFloat = np.zeros(self.mo.number_of_datapoints + 1)
        # self.c: NDArrayFloat = np.zeros(self.mo.steps)
        self.v: NDArrayFloat = (
            np.zeros(self.mo.number_of_datapoints + 1)
            + self.volume.to(self.mo.v_unit).magnitude
        )  # reservoir volume

        if self.delta != "None":
            self.l = get_l_mass(self.c, self.delta, self.species.r)

        # create temporary memory if we use multiple solver iterations
        if self.mo.number_of_solving_iterations > 0:
            self.mc = np.empty(0)
            self.cc = np.empty(0)
            self.dc = np.empty(0)

        self.mo.lor.append(self)  # add this reservoir to the model
        if self.rtype != "flux_only":
            self.mo.lic.append(self)  # reservoir type object list

        # register instance name in global name space
        if self.mo.register == "local" and self.register == "None":
            self.register = self.mo
        self.__register_with_parent__()

        # decide which setitem functions to use
        if self.isotopes:
            self.__set_data__ = self.__set_with_isotopes__
        else:
            self.__set_data__ = self.__set_without_isotopes__

        if self.species.name == "Hplus":
            self.plot_transform_c = phc
        # any auxilliary init - normally empty, but we use it here to extend the
        # reservoir class in virtual reservoirs
        self.__aux_inits__()

    @property
    def concentration(self) -> float:
        """Concentration Setter."""
        return self._concentration

    @property
    def delta(self) -> float:
        """Delta Setter."""
        return self._delta

    @property
    def mass(self) -> float:
        """Mass Setter."""
        return self._mass

    # @property
    # def volume(self) -> float:
    #     return self._volume

    # @volume.setter
    # def volume(self) -> None:
    #     self.volume = self._volume.to(self.register.v_unit)

    @concentration.setter
    def concentration(self, c) -> None:
        if self.update and c != "None":
            breakpoint()
            # this requires unit screening
            # then conversion into model units and mganitude
            # followed by updates to c and m
            self._concentration = c.to(self.mo.c_unit)
            self.mass = (
                self._concentration * self.volume * self.density / 1000
            )  # caculate mass
            self.c = self.c * 0 + self._concentration.magnitude
            self.m = self.m * 0 + self.mass

    @delta.setter
    def delta(self, d: float) -> None:
        if self.update and d != "None":
            self._delta: float = d
            self.isotopes = True
            self.l = get_l_mass(self.c, d, self.species.r)

    @mass.setter
    def mass(self, m: float) -> None:
        if self.update and m != "None":
            self._mass: float = m
            """ problem: m_unit can be mole, but data can be in liter * mole /kg
            this should not happen and results in an error converting to magnitide
            """
            self.m = np.zeros(self.species.mo.number_of_datapoints + 1) + m
            self.c = self.m / self.volume.to(self.mo.v_unit).magnitude


class Flux(esbmtkBase):
    """A class which defines a flux object.

    Flux objects contain
    information which links them to an species, describe things like
    the mass and time unit, and store data of the total flux rate at
    any given time step.  Similarly, they store the flux of the light
    and heavy isotope flux, as well as the delta of the flux.  This is
    typically handled through the Species2Species object.  If you set it up
    manually

    Example::

        Flux = (name = "Name" # optional, defaults to _F
             species = species_handle,
             delta = any number,
             rate  = "12 mol/s" # must be a string
             display_precision = number, optional, inherited from Model

    )

    You can access the flux data as

        - Name.m # mass
        - Name.d # delta
        - Name.c # same as Name.m since flux has no concentration
    """

    def __init__(self, **kwargs: dict[str, any]) -> None:
        """Initialize a flux.

        Arguments are the species name the flux
        rate (mol/year), the delta value and unit

        Example::

        Flux = (name = "Name" # optional, defaults to _F
             species = species_handle,
             delta = any number,
             rate  = "12 mol/s" # must be a string
             display_precision = number, optional, inherited from Model

        )

        You can access the flux data as:

        - Name.m # mass
        - Name.d # delta
        - Name.c # same as Name.m since flux has no concentration

        Defaults::

            self.defaults: dict[str, tp.List[any, tuple]] = {
              "name": ["None", (str)],
              "species": ["None", (str, SpeciesProperties)],
              "delta": [0, (str, int, float)],
              "rate": ["None", (str, Q_, int, float)],
              "plot": ["yes", (str)],
              "display_precision": [0.01, (int, float)],
              "isotopes": [False, (bool)],
              "register": [
                  "None",
                  (
                      str,
                      Species,
                      GasReservoir,
                      Species2Species,
                      Species2Species,
                      Signal,
                  ),
              ],
              "save_flux_data": [False, (bool)],
              "id": ["None", (str)],
              "ftype": ["None", (str)],

        }

        Required Keywords: "species", "rate", "register"
        """
        from esbmtk import (
            Q_,
            ExternalCode,
            GasReservoir,
            Signal,
            Species,
            Species2Species,
        )

        # provide a dict of all known keywords and their type
        self.defaults: dict[str, list[any, tuple]] = {
            "name": ["None", (str)],
            "species": ["None", (str, SpeciesProperties)],
            "delta": [0, (str, int, float)],
            "rate": ["None", (str, Q_, int, float)],
            "plot": ["yes", (str)],
            "display_precision": [0.01, (int, float)],
            "isotopes": [False, (bool)],
            "register": [
                "None",
                (
                    str,
                    Species,
                    GasReservoir,
                    Species2Species,
                    Species2Species,
                    Signal,
                ),
            ],
            "save_flux_data": [False, (bool)],
            "id": ["None", (str)],
            "ftype": ["None", (str)],
            "computed_by": ["None", (str, ExternalCode)],
            "serves_as_input": [False, (bool)],
        }

        # provide a list of absolutely required keywords
        self.lrk: list = ["species", "rate", "register"]
        self.__initialize_keyword_variables__(kwargs)
        self.parent = self.register

        # legacy names
        self.n: str = self.name  # name of flux
        self.sp: SpeciesProperties = self.species  # species name
        self.mo: Model = self.species.mo  # model name
        self.model: Model = self.species.mo  # model handle
        self.rvalue = self.sp.r

        if self.display_precision == 0:
            self.display_precision = self.mo.display_precision

        # model units
        self.plt_units = Q_(self.rate).units
        self.mu: str = f"{self.species.mu}/{self.mo.t_unit}"

        # and convert flux into model units
        if isinstance(self.rate, str):
            self.rate: float = Q_(self.rate).to(self.mo.f_unit).magnitude
        elif isinstance(self.rate, Q_):
            self.rate: float = self.rate.to(self.mo.f_unit).magnitude
        elif isinstance(self.rate, int | float):
            self.rate: float = self.rate

        li = get_l_mass(self.rate, self.delta, self.sp.r) if self.delta else 0
        self.fa: NDArrayFloat = np.asarray([self.rate, li])

        # in case we want to keep the flux data
        if self.save_flux_data:
            self.m: NDArrayFloat = (
                np.zeros(self.model.number_of_datapoints + 1) + self.rate
            )  # add the flux

            if self.isotopes:
                self.l: NDArrayFloat = np.zeros(self.model.number_of_datapoints + 1)
                if self.rate != 0:
                    self.l = get_l_mass(self.m, self.delta, self.species.r)
                    self.fa[1] = self.l[0]

            if self.mo.number_of_solving_iterations > 0:
                self.mc = np.empty(0)
                self.dc = np.empty(0)

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
            if isinstance(self.parent, (Species2Species)):
                self.name = f"_F{self.id}"
                self.n = self.name
            else:
                self.name = f"{self.id}_F"

        self.__register_with_parent__()
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
    def __setitem__(self, i: int, value: NDArrayFloat):
        """Setitem function."""
        return self.__set_data__(i, value)

    def __getitem__(self, i: int) -> NDArrayFloat:
        """Get data by index."""
        # return self.__get_data__(i)
        return self.fa

    def __set_with_isotopes__(self, i: int, value: NDArrayFloat) -> None:
        """Write data by index."""
        self.m[i] = value[0]
        self.l[i] = value[1]
        self.fa = value[:4]

    def __set_without_isotopes__(self, i: int, value: NDArrayFloat) -> None:
        """Write data by index."""
        self.fa = [value[0], 0]
        self.m[i] = value[0]

    # FIXME: this does nothing, do we still need it?
    # def __call__(self) -> None:  # what to do when called as a function ()
    #     """"""
    #     pass
    #     return

    def __add__(self, other):
        """Add two fluxes.

        FIXME: adding two fluxes works for the masses, but not for delta
        """
        self.fa = self.fa + other.fa
        self.m = self.m + other.m
        self.l = self.l + other.l

    def __sub__(self, other):
        """Substract two fluxes.

        FIXME: This works for the masses, but not for delta
        """
        self.fa = self.fa - other.fa
        self.m = self.m - other.m
        self.l = self.l - other.l

    def info(self, **kwargs) -> None:
        """Show an overview of the object properties.

        Optional arguments are:

        :param index: int = 0 this will show data at the given index
        :param indent: int = 0 indentation
        """
        # index = 0 if "index" not in kwargs else kwargs["index"]
        if "indent" not in kwargs:
            indent = 0
            ind = ""
        else:
            indent = kwargs["indent"]
            ind = " " * indent

        # print basic data bout this object
        print(f"{ind}{self.__str__(kwargs)}")
        print(f"{ind}Data sample:")
        # show_data(self, index=index, indent=indent)

        if len(self.lop) > 0:
            self._extracted_from_info_27(ind)
        else:
            print("There are no processes for this flux")

    # FIXME Rename this here and in `info`
    def _extracted_from_info_27(self, ind):
        print(f"\n{ind}Process(es) acting on this flux:")
        off: str = "  "
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

    def __plot__(self, M: Model, ax) -> None:
        """Plot instructions.

        :param M: Model
        :param ax: matplotlib axes handle
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
                raise FluxError("Plot transform must be a function")

        # plot first axis
        ax.plot(x[1:-2], y1[1:-2], color="C0", label=self.legend_left)
        ax.set_xlabel(f"{M.time_label} [{M.d_unit:~P}]")
        ax.set_ylabel(self.legend_left)
        ax.spines["top"].set_visible(False)
        handler1, label1 = ax.get_legend_handles_labels()

        # plot second axis
        if self.isotopes:
            axt = ax.twinx()
            # FIXME: y2 and ln2 are never used
            # y2 = self.d  # no conversion for isotopes
            # ln2 = axt.plot(x[1:-2], y2[1:-2], color="C1", label=self.legend_right)
            axt.set_ylabel(self.data.ld)
            set_y_limits(axt, M)
            ax.spines["top"].set_visible(False)
            # set combined legend
            _handler2, _label2 = axt.get_legend_handles_labels()
            # FIXME: legend is never used
            # legend = axt.legend(handler1 + handler2, label1 + label2, loc=0).set_zorder(
            #     6
            # )
        else:
            ax.legend(handler1, label1)
            ax.spines["right"].set_visible(False)
            ax.yaxis.set_ticks_position("left")
            ax.xaxis.set_ticks_position("bottom")
        ax.set_title(self.full_name)

    def __sub_sample_data__(self, stride) -> None:
        """Subsample the results before saving, or processing."""
        if self.save_flux_data:
            self.m = self.m[2:-2:stride]
            self.l = self.m[2:-2:stride]

    def __reset_state__(self) -> None:
        """Copy the result of the last computation."""
        if self.save_flux_data:
            self.mc = np.append(self.mc, self.m[0 : -2 : self.mo.reset_stride])
            # copy last element to first position
            self.m[0] = self.m[-2]
            self.l[0] = self.l[-2]

    def __merge_temp_results__(self) -> None:
        """Replace the data fields with saved values."""
        self.m = self.mc


class SourceSink(esbmtkBase):
    """Meta class to setup a Source/Sink objects.

    These are
    not actual reservoirs, but we stil need to have them as objects
    Example::

        Sink(name = "Pyrite",
            species = SO4,
            display_precision = number, optional, inherited from Model
            delta = number or str. optional defaults to "None"
            register = Model handle
        )
    """

    def __init__(self, **kwargs) -> None:
        """Initialize class instance.

        Defaults::

            self.defaults: dict[str, tp.List[any, tuple]] = {
               "name": ["None", (str)],
               "species": ["None", (str, SpeciesProperties)],
               "display_precision": [0.01, (int, float)],
               "register": [
                   "None",
                   (
                       SourceProperties,
                       SinkProperties,
                       Reservoir,
                       ConnectionProperties,
                       Model,
                       str,
                   ),
               ],
               "delta": ["None", (str, int, float)],
               "isotopes": [False, (bool)],

        Required Keywords: "name", "species"
        """
        from esbmtk import (
            ConnectionProperties,
            Reservoir,
            SinkProperties,
            SourceProperties,
        )

        self.defaults: dict[str, list[any, tuple]] = {
            "name": ["None", (str)],
            "species": ["None", (str, SpeciesProperties)],
            "display_precision": [0.01, (int, float)],
            "register": [
                "None",
                (
                    SourceProperties,
                    SinkProperties,
                    Reservoir,
                    ConnectionProperties,
                    Model,
                    str,
                ),
            ],
            "delta": ["None", (str, int, float)],
            "epsilon": ["None", (str, int, float)],
            "isotopes": [False, (bool)],
        }
        # provide a list of absolutely required keywords
        self.lrk: list[str] = ["name", "species"]
        self.__initialize_keyword_variables__(kwargs)

        if self.register == "None":  # use a sensible default
            self.register = self.species.element.register

        self.loc: set[Species2Species] = set()  # set of connection objects

        # legacy names
        # if self.register != "None":
        #    self.full_name = f"{self.name}.{self.register.name}"
        self.parent = self.register
        self.n = self.name
        self.sp = self.species
        self.mo = self.species.mo
        self.model = self.species.mo
        self.u = self.species.mu + "/" + str(self.species.mo.t_unit)
        self.lio: list = []
        self.m = 1  # set default mass and concentration values
        self.c = 1
        self.mo.lic.append(self)  # add source to list of res type objects

        if self.mo.register == "local" and self.register == "None":
            self.register = self.mo
        elif self.register == "None":
            self.pt = self.name
        else:
            self.pt: str = f"{self.register.name}_{self.n}"
            self.groupname = self.register.name

        if self.display_precision == 0:
            self.display_precision = self.mo.display_precision

        self.__register_with_parent__()
        self.mo.lic.remove(self)

    @property
    def delta(self):
        """Delta Setter."""
        return self._delta

    @delta.setter
    def delta(self, d):
        """Set/Update delta."""
        if d != "None":
            self._delta = d
            self.isotopes = True
            self.m = 1
            self.c = 1
            self.l = get_l_mass(self.c, d, self.species.r)
            # self.c = self.l / (self.m - self.l)
            # self.provided_kwargs.update({"delta": d})


class Sink(SourceSink):
    """Meta class to setup a Source/Sink objects.

    These are
    not actual reservoirs, but we stil need to have them as objects
    Example::

        Sink(name = "Pyrite",
            species = SO4,
            display_precision = number, optional, inherited from Model
            delta = number or str. optional defaults to "None"
            register = Model handle
        )
    """


class Source(SourceSink):
    """Meta class to setup a Source/Sink objects.

    These are
    not actual reservoirs, but we stil need to have them as objects
    Example::

        Ssource(name = "weathering",
            species = SO4,
            display_precision = number, optional, inherited from Model
            delta = number or str. optional defaults to "None"
            register = Model handle
        )
    """
