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
import psutil
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
    plot_geometry,
)

# declare numpy types
NDArrayFloat = npt.NDArray[np.float64]

if tp.TYPE_CHECKING:
    from .base_classes import Species

    # from .connections import Species2Species


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


class FluxNameError(Exception):
    """Custom Error Class for Flux lookup errors."""

    def __init__(self, message):
        """Initialize Error Instance with formatted message."""
        message = f"\n\n{message}\n"
        super().__init__(message)


def deprecated_keyword(message):
    """Issue a deprecation warning with the provided message."""
    warnings.warn(message, DeprecationWarning, stacklevel=2)


class Model(esbmtkBase):
    r"""Earth Science Box Model Toolkit (ESBMTK) Model class.

    This class represents the main model framework for creating and running
    Earth science box models. It handles initialization of model parameters,
    management of reservoirs, fluxes, and species, and provides methods for
    running simulations and visualizing results.

    The user-facing methods of the model class are:

    - Model_Name.info() - Display model information
    - Model_Name.save_data() - Save model data to files
    - Model_Name.plot([sb.DIC, sb.TA]) - Plot specified objects
    - Model_Name.save\_state() - Save current model state
    - Model_Name.read\_state() - Initialize with a previous model state
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
        debug: bool
            output debug information
        debug_equations_file: bool
            write a debug version of the equations file.
        """
        import io
        import sys
        import warnings
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
            "rtol": [1.0e-4, (float)],
            "bio_pump_functions": [0, (int)],  # custom/old
            "opt_k_carbonic": [15, (int)],
            "opt_pH_scale": [1, (int)],  # 1: total scale
            "opt_buffers_mode": [2, (int)],
            "display_steps": [1000, (int)],
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

        # collect all warnings so they can be printed at the end
        # Create a string IO to capture warnings
        self.warning_log = io.StringIO()

        # Keep a backup of the original function
        self.original_showwarning = warnings.showwarning

        # Initialize customized warnings collection:
        warnings.showwarning = self.capture_warnings

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

    # Define a custom warning handler that captures warnings
    def capture_warnings(
        self, message, category, filename, lineno, file=None, line=None
    ):
        """Custom warning handler that captures warnings."""
        self.warning_log.write(
            f"{category.__name__}: {message} (in {filename}, line {lineno})\n\n"
        )

    def _initialize_model_containers(self):
        """Initialize all model component containers."""
        # Model objects
        self.lmo: list = []  # List of all model objects
        self.lmo2: list = []  # Secondary list of model objects
        self.dmo: dict = {}  # Dict of all model objects (for name lookups)

        # Reservoirs and connections
        self.lor: list = []  # List of all reservoir type objects
        self.lic: list = []  # List reservoirs with initial conditions
        # self.lis: list = []  # List of sources with initial conditions
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
        self.display_time = np.linspace(
            self.start,
            self.stop - self.start,
            num=self.display_steps + 1,
        )
        self.time = self.time_ode
        self.timec = np.empty(0)
        self.executionstate = 0

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

    def save_state(self, directory: str = "state", prefix: str = "state") -> None:
        """Save the current model state to files.

        Saves only the last time step of each reservoir to files in the specified directory.
        This is similar to save_data() but focuses on capturing the current state rather
        than the full time series.

        Parameters
        ----------
        directory : str, default="state"
            Directory where state files will be saved. Will be created if it doesn't exist
            and deleted if it already exists.

        prefix : str, default="state"
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
        # ugly workaround because sphinx stumbles over the underscore when
        # we set it in the function signature
        prefix = f"{prefix}_"

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

        For this to work, you will first need to issue a
        `save_state` command at then end of a model run.  This
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
        self.executionstate = 1

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

        # printout any warnings
        # Now display all collected warnings at the end
        print("\n" + "=" * 80)
        print("WARNINGS COLLECTED DURING EXECUTION:")
        print("=" * 80)
        print(self.warning_log.getvalue() or "No warnings generated")
        print("\n" + "=" * 80)
        print("END WARNINGS COLLECTED DURING EXECUTION:")
        print("=" * 80)

        # Restore the original warning behavior
        warnings.showwarning = self.original_showwarning

    def _write_temp_equations(self, cwd, R, icl, cpl, ipl):
        """Write temporary equations file and return the equationsset.

        Creates a temporary Python module containing the model equations,
        imports it, and returns the equationsset function.

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
        from pathlib import Path

        # Set temporary directory to current working directory
        tempfile.tempdir = cwd

        # Create a temporary Python file
        with tempfile.NamedTemporaryFile(suffix=".py") as tmp_file:
            # Get path to temporary file
            equations_file_path = Path(tmp_file.name)

            # Generate equations module
            equations_module_name = write_equations_3(
                self, R, icl, cpl, ipl, equations_file_path
            )
            eqs = __import__(equations_module_name).eqs

        return eqs

    def _ode_solver(self, kwargs: dict):
        """Initialize and run the ODE solver.

        Sets up the system of ODEs, generates the equationsfile, and solves
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
        # Get initial conditions and build equationsmatrices
        self.R_names_dict, icl, cpl, ipl, atol = get_initial_conditions(self, self.rtol)
        self.R_names = list(self.R_names_dict.keys())

        # get initial concentrations for each reservoir
        R = np.array(list(self.R_names_dict.values()))

        # icl = dict[Species, list[int, int]] where reservoir
        #     indicates the reservoir handle, and the list contains the
        #     index into the reservoir data.  list[0] = concentration
        #     list[1] concentration of the light isotope.
        self.icl = icl
        # cpl = list of reservoirs that use function to evaluate
        #       reservoir data
        self.cpl = cpl
        #  ipl = list of static reservoirs that serve as input
        self.ipl = ipl

        # Build coefficient matrix
        self.CM, self.F, self.F_names = build_eqs_matrix(self)

        # Set up paths for equationsfiles
        current_dir = Path.cwd()
        sys.path.append(str(current_dir))  # Required on Windows
        equations_filename = "equations.py"
        coefficients_file = "eqs_coeff.npz"
        coeff_file_path = Path(f"{current_dir}/{coefficients_file}")
        equations_file_path = Path(f"{current_dir}/{equations_filename}")

        if self.debug_equations_file:
            np.savez(coeff_file_path, CM=self.CM, F=self.F)
        elif coeff_file_path.exists():
            coeff_file_path.unlink()

        # Handle equations file based on debug settings
        equations_set = self._handle_equations_file(
            equations_file_path, R, icl, cpl, ipl, current_dir
        )

        # Get solver configuration from kwargs
        method = kwargs.get("method", "LSODA")

        # Initialize carbonate chemistry tables if not present
        self._initialize_carbonate_tables()

        # Run the ODE solver
        self._run_solve_ivp(R, equations_set, method, atol)

        # Process results
        self._process_solver_results()

    def _handle_equations_file(
        self, equations_file_path, R, icl, cpl, ipl, current_dir
    ):
        """Handle equationsfile generation based on debug settings.

        Parameters
        ----------
        equations_file_path : Path
            Path to the equationsfile
        R, icl, cpl, ipl : various
            Parameters for equationsgeneration
        current_dir : Path
            Current working directory

        Returns
        -------
        function
            The equations function
        """
        import importlib

        # If debugging equations is enabled
        if self.debug_equations_file:
            if equations_file_path.exists():
                warnings.warn(
                    "\n\n Warning re-using the equations file \n"
                    "\n type r to reuse old file or n to create a new one",
                    stacklevel=2,
                )
                user_input = input("type r/n: ")

                if user_input.lower() == "r":  # Use existing file
                    equations_module_name = equations_file_path.stem
                    # Also load saved matrices if they exist
                    matrix_file = Path(
                        str(equations_file_path).replace(".py", "_matrices.npz")
                    )
                    if matrix_file.exists():
                        saved_data = np.load(matrix_file)
                        self.CM = saved_data["CM"]
                        self.F = saved_data["F"]
                        self.F_names = (
                            saved_data["F_names"].tolist()
                            if "F_names" in saved_data
                            else []
                        )
                    else:
                        print(
                            "Warning: Reusing equationsfile but matrix file not found. Results may be inconsistent."
                        )

                else:  # Create new file
                    equations_file_path.unlink()  # Delete old file
                    equations_module_name = write_equations_3(
                        self, R, icl, cpl, ipl, equations_file_path
                    )
                    # Save matrices for future reuse
                    matrix_file = Path(
                        str(equations_file_path).replace(".py", "_matrices.npz")
                    )
                    self.matrix_file = matrix_file
                    np.savez(
                        matrix_file,
                        CM=self.CM,
                        F=self.F,
                        F_names=np.array(self.F_names),
                    )

            else:  # First run - create persistent file
                equations_module_name = write_equations_3(
                    self, R, icl, cpl, ipl, equations_file_path
                )
            eqs = __import__(equations_module_name).eqs

        else:  # Use temporary file for equations
            if equations_file_path.exists():
                equations_file_path.unlink()

            eqs = self._write_temp_equations(current_dir, R, icl, cpl, ipl)
        return eqs  # module reference

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
        if self.debug:
            print(f"R: {R}")
            print(
                f"self.gpt shape: {np.shape(self.gpt) if hasattr(self.gpt, 'shape') else len(self.gpt)}"
            )
            print(
                f"self.toc shape: {np.shape(self.toc) if hasattr(self.toc, 'shape') else len(self.toc)}"
            )
            print(f"CM shape: {np.shape(self.CM)}")
            print(f"F shape: {np.shape(self.F)}")
            print(f"time_ode shape: {np.shape(self.time_ode)}")
            # Add hash values for large arrays to verify content
            print(f"CM hash: {hash(str(self.CM))}")
            print(f"F hash: {hash(str(self.F))}")
            print(f"time_ode hash: {hash(str(self.time_ode))}")

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
            # t_eval=self.time_ode,
            t_eval=self.display_time,
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
        # FIXME: Thios needs proper keyword parsing!
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

                if return_as_list:
                    if len(matching_fluxes) == 0:
                        raise FluxNameError(f"No flux {filter_terms} found. Typo?")
                else:
                    # Print flux name if not returning a list
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

            filter_by : str, default=None
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

        >>> model.connection_summary(filter_by="DIC")  # Show only DIC connections

        >>> model.connection_summary(list_all=True)  # Show all connection details
        """
        # Extract configuration from kwargs
        show_all_attributes = kwargs.get("list_all", False)
        connection_name_filter = kwargs.get("filter_by")

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
                if name_filter in connection.name:  # Substring search
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
        from esbmtk import Species2Species

        source = connection.source_name
        target = connection.sink_name

        # Display connection header with appropriate format based on connection type
        if isinstance(connection, Species2Species):
            # For species-to-species connections, show the specific species
            source_species = f"{connection.source.sp.n}"
            target_species = f"{connection.sink.sp.n}"
            print(
                f"Connection: {connection.full_name}: {source}.{source_species} -> {target}.{target_species}"
            )
        else:
            # For reservoir-to-reservoir connections
            print(f"Connection: {connection.full_name}: {source} -> {target}")

        # Display connection attributes
        # self._display_connection_attributes(connection, show_all_attributes)

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
