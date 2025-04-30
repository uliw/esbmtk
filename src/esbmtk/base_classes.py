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
import typing as tp
import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas import DataFrame

from . import Q_
from .esbmtk_base import esbmtkBase
from .model import Model
from .utility_functions import get_l_mass

# declare numpy types
NDArrayFloat = npt.NDArray[np.float64]

if tp.TYPE_CHECKING:
    from .connections import Species2Species
    from .extended_classes import DataField, ExternalData
    from .processes import Process


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


class ElementProperties(esbmtkBase):
    r"""Element-specific properties for models.

    Each model can have one or more elements. This class defines the properties
    specific to elements, such as name, mass unit, and isotope information.

    Parameters
    ----------
    name : str
        The element name, e.g., "S"
    model : Model
        The model handle
    mass_unit : str | Q\_
        Base mass unit, e.g., "mol"
    li_label : str, optional
        Label of light isotope, e.g., "$^{32}$S"
    hi_label : str, optional
        Label of heavy isotope, e.g., "$^{34}$S"
    d_label : str, optional
        Label for delta value, e.g., r"$\delta^{34}$S"
    d_scale : str, optional
        Isotope scale, e.g., "VCDT"
    r : float | int, optional
        Isotopic abundance ratio for element, defaults to 1
    reference : str, optional
        Reference for isotope values, e.g., URL or citation
    register : str | Model, optional
        Where to register this element, defaults to model
    parent : str | Model, optional
        Parent object (usually model), defaults to model
    full_name : str, optional
        Full name of the element

    Examples
    --------
    >>> ElementProperties(
    ...     name="S",
    ...     model=Test_model,
    ...     mass_unit="mol",
    ...     li_label="$^{32}$S",
    ...     hi_label="$^{34}$S",
    ...     d_label=r"$\delta^{34}$S",
    ...     d_scale="VCDT",
    ...     r=0.044162589,
    ...     reference="https link or citation"
    ... )
    """

    def __init__(self, **kwargs) -> None:
        """Initialize ElementProperties instance.

        Required keywords: "name", "model", "mass_unit"

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments for initialization

        Returns
        -------
        None
        """
        # Define defaults for all parameters
        self.defaults: dict[str, list[any, tuple]] = {
            "name": ["M", (str,)],
            "model": ["None", (str, Model)],
            "register": ["None", (str, Model)],
            "full_name": ["None", (str,)],
            "li_label": ["None", (str,)],
            "hi_label": ["None", (str,)],
            "d_label": ["None", (str,)],
            "d_scale": ["None", (str,)],
            "r": [1, (float, int)],
            "mass_unit": ["", (str, Q_)],
            "parent": ["None", (str, Model)],
            "reference": ["None", (str,)],
        }

        # List of absolutely required keywords
        self.required_keywords: list = ["name", "model", "mass_unit"]
        self.lrk = self.required_keywords  # Legacy alias for backward compatibility

        # Initialize variables from kwargs
        self.__initialize_keyword_variables__(kwargs)

        # Set parent to model
        self.parent = self.model

        # Initialize attributes and register with model
        self._initialize_legacy_names()
        self._register_with_model()
        self.__register_with_parent__()

    def _initialize_legacy_names(self) -> None:
        """Initialize legacy name aliases for backward compatibility.

        Returns
        -------
        None
        """
        # Legacy name aliases
        self.n: str = self.name  # display name of species
        self.mo: Model = self.model  # model handle
        self.mu: str = self.mass_unit  # display name of mass unit
        self.ln: str = self.li_label  # display name of light isotope
        self.hn: str = self.hi_label  # display name of heavy isotope
        self.dn: str = self.d_label  # display string for delta
        self.ds: str = self.d_scale  # display string for delta scale

        # More descriptive names
        self.light_isotope_label: str = self.li_label
        self.heavy_isotope_label: str = self.hi_label
        self.delta_label: str = self.d_label
        self.delta_scale: str = self.d_scale

        # List to store species for this element
        self.species_list: list = []
        self.lsp = self.species_list  # Legacy alias

    def _register_with_model(self) -> None:
        """Register this element with the model.

        Returns
        -------
        None

        Notes
        -----
        Adds this element to the model's element list and sets up registration.
        """
        # Add this element to the model's list of elements
        self.mo.lel.append(self)

        # Set registration based on model's registration setting
        if self.mo.register == "local" and self.register == "None":
            self.register = self.mo

    def list_species(self) -> None:
        """List all species which are predefined for this element.

        Returns
        -------
        None

        Notes
        -----
        Prints the names of all species associated with this element.
        """
        for species in self.species_list:
            print(species.n)

    def __register_species_with_model__(self) -> None:
        """Register all species with the model for easier access.

        This is a bit of a hack, but makes model code more readable by allowing
        direct access to species through model attributes.

        Returns
        -------
        None
        """
        for species in self.species_list:
            setattr(self.model, species.name, species)

    def add_species(self, species: SpeciesProperties) -> None:
        """Add a species to this element's species list.

        Parameters
        ----------
        species : SpeciesProperties
            Species to add to this element

        Returns
        -------
        None
        """
        self.species_list.append(species)
        # Optionally register with model
        setattr(self.model, species.name, species)


class SpeciesProperties(esbmtkBase):
    """Properties class for chemical species in a model.

    This class defines the properties specific to chemical species,
    such as their name, display format, and relationship to elements.

    Parameters
    ----------
    name : str
        Name of the species, e.g., "SO4"
    element : ElementProperties
        Handle to the element this species belongs to
    display_as : str, optional
        How to display the species, defaults to name if not provided
    m_weight : int | float | str, optional
        Molecular weight, defaults to 0
    register : Model | ElementProperties | Species | GasReservoir, optional
        Where to register this species, defaults to element
    parent : Model | ElementProperties | Species | GasReservoir, optional
        Parent object, defaults to register value
    flux_only : bool, optional
        Whether this species only exists in fluxes, not reservoirs, defaults to False
    logdata : bool, optional
        Whether to log data for this species, defaults to False
    scale_to : str, optional
        Unit to scale to for display, defaults to "mmol"
    stype : str, optional
        Species type, defaults to "concentration"

    Examples
    --------
    >>> SpeciesProperties(
    ...     name="SO4",
    ...     element=S,
    ... )

    Notes
    -----
    When a species is created, it's automatically registered with its element.
    """

    # Class constants
    DEFAULT_SCALE_TO = "mmol"
    DEFAULT_STYPE = "concentration"

    def __init__(self, **kwargs) -> None:
        """
        Initialize a SpeciesProperties instance.

        Required keywords: "name", "element"

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments for initialization

        Returns
        -------
        None

        Raises
        ------
        SpeciesError
            If required parameters are missing or invalid
        """
        from esbmtk import GasReservoir

        # Define defaults for all parameters
        # Use copied name value to avoid direct reference to kwargs
        name_default = kwargs.get("name", "None")
        element_default = kwargs.get("element", "None")

        self.defaults: dict[str, list[any, tuple]] = {
            "name": ["None", (str,)],
            "element": ["None", (ElementProperties, str)],
            "display_as": [name_default, (str,)],
            "m_weight": [0, (int, float, str)],
            "register": [
                element_default,
                (Model, ElementProperties, Species, GasReservoir),
            ],
            "parent": ["None", (Model, ElementProperties, Species, GasReservoir)],
            "flux_only": [False, (bool,)],
            "logdata": [False, (bool,)],
            "scale_to": [self.DEFAULT_SCALE_TO, (str,)],
            "stype": [self.DEFAULT_STYPE, (str,)],
        }

        # List of absolutely required keywords
        self.required_keywords = ["name", "element"]
        self.lrk = self.required_keywords  # Legacy alias

        # Initialize variables from kwargs
        self.__initialize_keyword_variables__(kwargs)

        # Set parent to register by default
        self.parent = self.register

        # Set display_as to name if not provided
        if "display_as" not in kwargs:
            self.display_as = self.name

        # Setup variable relationships and legacy names
        self._initialize_element_relationships()
        self._register_with_element()

        # Register with model if appropriate
        if self.model.register == "local" and self.register == "None":
            self.register = self.model

        # Register with parent
        self.__register_with_parent__()

    def _initialize_element_relationships(self) -> None:
        """
        Initialize attributes derived from the element.

        Sets up all properties inherited from the element and
        creates legacy name aliases for backward compatibility.

        Returns
        -------
        None
        """
        # Legacy names and main attributes
        self.n = self.name  # display name of species (legacy)
        self.mass_unit = self.element.mass_unit
        self.mu = self.mass_unit  # display name of mass unit (legacy)

        # Isotope-related attributes
        self.light_isotope_label = self.element.li_label
        self.heavy_isotope_label = self.element.hi_label
        self.delta_label = self.element.d_label
        self.delta_scale = self.element.d_scale
        self.r = self.element.r  # ratio of isotope standard

        # Legacy isotope attributes
        self.ln = self.element.ln  # display name of light isotope (legacy)
        self.hn = self.element.hn  # display name of heavy isotope (legacy)
        self.dn = self.element.dn  # display string for delta (legacy)
        self.ds = self.element.ds  # display string for delta scale (legacy)

        # Element-related attributes
        self.model = self.element.mo  # model handle
        self.mo = self.model  # Legacy alias for model
        self.element_name = self.element.n  # element name
        self.eh = self.element_name  # Legacy alias for element name
        self.element_handle = self.element  # element handle
        self.e = self.element_handle  # Legacy alias for element handle

        # Display-related attributes
        self.display_string = self.display_as  # the display string
        self.dsa = self.display_string  # Legacy alias

    def _register_with_element(self) -> None:
        """
        Register this species with its parent element.

        This adds the species to the element's list of species.

        Returns
        -------
        None
        """
        # Register this species with the element
        self.element_handle.lsp.append(self)

    @property
    def full_display_name(self) -> str:
        """
        Return the full display name of the species.

        Returns
        -------
        str
            Formatted display name including element and delta information if applicable
        """
        return f"{self.element_name}.{self.name}"


class SpeciesBase(esbmtkBase):
    """Base class for all Species objects.

    This class provides common functionality for all species-related classes,
    including data handling, visualization, and state management.

    Notes
    -----
    This is an abstract base class that should not be instantiated directly.
    Use the derived classes like Species instead.
    """

    def __init__(self, **kwargs) -> None:
        """Initialize a SpeciesBase instance.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments for initialization

        Raises
        ------
        NotImplementedError
            This class should not be instantiated directly
        """
        raise NotImplementedError(
            "SpeciesBase should never be used. Use the derived classes"
        )

    def _initialize_legacy_attributes(self, kwargs) -> None:
        """Initialize common attributes and legacy names.

        This method sets up attributes required by all species objects
        and maintains backward compatibility with legacy naming conventions.

        Parameters
        ----------
        kwargs : dict
            Initialization parameters

        Returns
        -------
        None
        """
        from esbmtk.sealevel import get_box_geometry_parameters

        # Initialize data structures with appropriate typing
        # Lists and collections for references
        # see scipy.solve_ivp docs for the meaning of this
        self.tolerances: list[float] = [1.0, 1.0]  # tolerances
        self.ignored_fluxes: list[Flux] = []
        self.flux_references: list[Flux] = []  # flux references
        self.set_area_warning = False
        self.external_data_references: list[
            ExternalData
        ] = []  # all external data references
        self.flux_direction_pairs: dict[str, int] = {}  # flux name:direction pairs
        self.process_references: list[
            Process
        ] = []  # list holding all process references
        self.element_references: list[
            ElementProperties
        ] = []  # list of elements in this reservoir
        self.species_flux_pairs: dict[
            SpeciesProperties, Flux
        ] = {}  # species flux pairs
        self.connection_objects: set[Species2Species] = (
            set()
        )  # set of connection objects
        self.datafield_objects: list[DataField] = []  # list of datafield objects
        self.calculated_processes: list[
            Process
        ] = []  # list of processes which calculate reservoirs

        # Legacy aliases for backward compatibility
        self.atol = self.tolerances
        self.lof = self.flux_references
        self.lif = self.ignored_fluxes
        self.led = self.external_data_references
        self.lio = self.flux_direction_pairs
        self.lop = self.process_references
        self.loe = self.element_references
        self.doe = self.species_flux_pairs
        self.loc = self.connection_objects
        self.ldf = self.datafield_objects
        self.lpc = self.calculated_processes

        # Flag for external function results
        self.has_external_function_results = False
        self.ef_results = self.has_external_function_results

        # Initialize core attributes with appropriate naming
        self._initialize_base_attributes()
        self._initialize_unit_attributes()
        self._initialize_display_attributes()

        # Configure geometry parameters if needed
        if self.geometry != "None" and self.geometry_unset:
            get_box_geometry_parameters(self)

        # Set display precision from model if not specified
        if self.display_precision == 0:
            self.display_precision = self.model.display_precision

        # Finalize parent relationship
        self.parent = self.register

    def _initialize_base_attributes(self) -> None:
        """Initialize core name and relationship attributes.

        Returns
        -------
        None
        """
        # Name and identification attributes
        self.n: str = self.name  # Legacy name alias

        # Set path name based on registration
        if self.register == "None":
            self.path_name = self.name
        else:
            self.path_name: str = f"{self.register.name}_{self.name}"
            self.groupname = self.register.name

        # Legacy alias
        self.pt = self.path_name

        # Species and model relationships
        self.species_properties: SpeciesProperties = self.species
        self.model: Model = self.species.model

        # Legacy aliases
        self.sp = self.species_properties
        self.mo = self.model

        # Isotope ratio value
        self.rvalue = self.species_properties.r

    def _initialize_unit_attributes(self) -> None:
        """Initialize unit-related attributes.

        Returns
        -------
        None
        """
        # Set up unit attributes from model
        self.m_unit = self.model.m_unit  # Mass unit
        self.v_unit = self.model.v_unit  # Volume unit
        self.c_unit = self.model.c_unit  # Concentration unit

    def _initialize_display_attributes(self) -> None:
        """Initialize display and plotting related attributes.

        Returns
        -------
        None
        """
        # Axis labels and legends
        self.right_axis_label: str = (
            f"{self.species.delta_label} [{self.species.delta_scale}]"
        )
        self.x_axis_label: str = self.model.xl  # Set x-axis label to model time

        # Legacy aliases
        self.ld = self.right_axis_label
        self.xl = self.x_axis_label

        # Set legend labels
        if self.legend_left == "None":
            self.legend_left = self.species.display_string

        self.legend_right = f"{self.species.delta_label} [{self.species.delta_scale}]"

        # Determine whether to use isotopes
        if self.model.m_type == "both":
            self.isotopes = True
        elif self.model.m_type == "mass_only":
            self.isotopes = False

    def __setitem__(self, i: int, value: float):
        """Set data values at the specified index.

        Parameters
        ----------
        i : int
            Index where to set the value
        value : float
            Value to set

        Returns
        -------
        any
            Result of the __set_data__ method
        """
        return self.__set_data__(i, value)

    def __call__(self) -> SpeciesBase:
        """Return self when called as a function.

        Returns
        -------
        SpeciesBase
            Self reference
        """
        return self

    def __getitem__(self, i: int) -> NDArrayFloat:
        """Get flux data by index.

        Parameters
        ----------
        i : int
            Index to get data from

        Returns
        -------
        NDArrayFloat
            Array containing [mass, light isotope, concentration] values
        """
        return np.array([self.m[i], self.l[i], self.c[i]])

    def __set_with_isotopes__(self, i: int, value: float) -> None:
        """Set values when isotope data is present.

        Parameters
        ----------
        i : int
            Index to set values at
        value : float
            Array of [mass, light isotope, heavy isotope, delta]

        Returns
        -------
        None
        """
        self.m[i]: float = value[0]
        # Update concentration and delta next. This is computationally inefficient
        # but the next time step may depend on both variables.
        self.c[i]: float = value[0] / self.v[i]  # Update concentration
        self.l[i]: float = value[1] / self.v[i]  # Update light isotope concentration

    def __set_without_isotopes__(self, i: int, value: float) -> None:
        """Set values when no isotope data is present.

        Parameters
        ----------
        i : int
            Index to set values at
        value : float
            Array of [mass]

        Returns
        -------
        None
        """
        self.m[i]: float = value[0]
        self.c[i]: float = self.m[i] / self.v[i]  # Update concentration

    def __update_mass__(self) -> None:
        """Update mass calculations.

        This function should be implemented in derived classes.

        Raises
        ------
        NotImplementedError
            This method must be implemented in derived classes
        """
        raise NotImplementedError("__update_mass__ is not yet implemented")

    def __write_data__(
        self,
        prefix: str,
        start: int,
        stop: int,
        stride: int,
        append: bool,
        directory: str,
    ) -> pd.DataFrame:
        """Write data to file.

        This function is called by the write_data() and save_state() methods.

        Parameters
        ----------
        prefix : str
            Prefix for the filename
        start : int
            Start index for data slice
        stop : int
            Stop index for data slice
        stride : int
            Stride for data slice
        append : bool
            Whether to append to existing file
        directory : str
            Directory to save file in

        Returns
        -------
        pd.DataFrame
            DataFrame containing the saved data

        Raises
        ------
        SpeciesError
            If the model register keyword is invalid
        """
        from pathlib import Path

        # Create directory if it doesn't exist
        p = Path(directory)
        p.mkdir(parents=True, exist_ok=True)

        # Prepare shortened variable names for readability
        species_name = self.species_properties.name
        species = self.species_properties
        model = self.species_properties.model

        # Get unit display strings
        model_time_unit = f"{model.t_unit:~P}"
        concentration_unit = f"{model.c_unit:~P}"

        # Determine filename based on registration
        reservoir_name = self.full_name
        model_name = self.species_properties.model.name

        if self.species_properties.model.register == "None":
            filename = f"{directory}/{prefix}{model_name}_{reservoir_name}.csv"
        elif self.species_properties.model.register == "local":
            filename = f"{directory}/{prefix}{reservoir_name}.csv"
        else:
            raise SpeciesError(
                f"Model register keyword must be 'None'/'local' not {self.species_properties.model.register}"
            )

        # Build the dataframe with time and data columns
        df: pd.DataFrame = DataFrame()
        df[f"{reservoir_name} Time [{model_time_unit}]"] = self.model.time[
            start:stop:stride
        ]

        # Add isotope data if available
        if self.isotopes:
            df[
                f"{reservoir_name} {species.light_isotope_label} [{concentration_unit}]"
            ] = self.l[start:stop:stride]

        # Add concentration data
        df[f"{reservoir_name} {species_name} [{concentration_unit}]"] = self.c[
            start:stop:stride
        ]

        # Write to file with appropriate mode
        file_path = Path(filename)
        if append and file_path.exists():
            df.to_csv(file_path, header=False, mode="a", index=False)
        else:
            df.to_csv(file_path, header=True, mode="w", index=False)

        return df

    def __sub_sample_data__(self, stride: int) -> None:
        """
        Subsample the results before saving or processing.

        Parameters
        ----------
        stride : int
            Step size for subsampling

        Returns
        -------
        None
        """
        self.m = self.m[2:-2:stride]
        self.l = self.l[2:-2:stride]
        self.c = self.c[2:-2:stride]

    def __reset_state__(self) -> None:
        """Copy the result of the last computation to the beginning.

        This allows a new run to start with the values from the previous run.
        Saves the current results into temporary fields.

        Returns
        -------
        None
        """
        # Append current results to temporary storage with specified stride
        self.mc = np.append(self.mc, self.m[0 : -2 : self.model.reset_stride])
        self.cc = np.append(self.cc, self.c[0 : -2 : self.model.reset_stride])

        # Copy last result into first position for next run
        self.m[0] = self.m[-2]
        self.l[0] = self.l[-2]
        self.c[0] = self.c[-2]

    def __merge_temp_results__(self) -> None:
        """Replace the data fields with saved values.

        Returns
        -------
        None
        """
        self.m = self.mc
        self.c = self.cc

    def __read_state__(self, directory: str, prefix="state_") -> None:
        """Read data from csv-file into a dataframe.

        Parameters
        ----------
        directory : str
            Directory containing the state file
        prefix : str, optional
            Prefix for the state file, defaults to "state_"

        Returns
        -------
        None

        Raises
        ------
        SpeciesError
            If the model register keyword is invalid
        FileNotFoundError
            If the state file doesn't exist

        Notes
        -----
        The CSV file must have the following columns:
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

        # Determine filename based on registration
        if self.species_properties.model.register == "None":
            filename = f"{directory}/{prefix}{self.model.name}_{self.full_name}.csv"
        elif self.species_properties.model.register == "local":
            filename = f"{directory}/{prefix}{self.full_name}.csv"
        else:
            raise SpeciesError(
                f"Model register keyword must be 'None'/'local' not {self.species_properties.model.register}"
            )

        # Check if file exists
        if not os.path.exists(filename):
            raise FileNotFoundError(
                f"Flux {filename} does not exist in Species {self.full_name}"
            )

        # Read CSV file into dataframe
        self.df: pd.DataFrame = pd.read_csv(filename)
        self.headers: list = list(self.df.columns.values)
        df = self.df
        headers = self.headers

        # Extract unique object names from headers while preserving order
        header_list: list = []
        for x in headers:
            n = x.split(" ")[0]
            if n not in header_list:
                header_list.append(n)

        # Process data from columns
        col: int = 1  # Skip the time column
        for n in header_list:
            name = n.split(" ")[0]
            logging.debug(f"Looking for {name}")

            # Find and assign reservoir data
            if name == self.full_name:
                col = self.__assign_reservoir_data__(self, df, col, True)
            else:
                raise SpeciesError(f"Unable to find Flux {n} in {self.full_name}")

        # Check for missing fluxes
        for f in list(curr.difference(read)):
            warnings.warn(
                f"\nDid not find values for {f}\n in saved state", stacklevel=2
            )

    def __assign_reservoir_data__(
        self, obj: any, df: pd.DataFrame, col: int, res: bool
    ) -> int:
        """Assign the data from dataframe to reservoir values.

        Parameters
        ----------
        obj : any
            Species object to assign data to
        df : pd.DataFrame
            DataFrame containing the data
        col : int
            Column index to start reading from
        res : bool
            Whether object is a reservoir

        Returns
        -------
        int
            Updated column index after reading

        Notes
        -----
        This assigns the last row of data to all values in the reservoir.
        """
        if obj.isotopes:
            # Assign light isotope values
            obj.l[:] = df.iloc[-1, col]
            col += 1

            # Assign concentration values
            obj.c[:] = df.iloc[-1, col]
            col += 1
        else:
            # Assign concentration values (no isotopes)
            obj.c[:] = df.iloc[-1, col]
            col += 1

        return col

    def get_plot_format(self) -> tuple:
        """Return concentration data in plot units.

        Returns
        -------
        tuple
            Tuple containing (y_values, y_label, unit)

        Raises
        ------
        SpeciesError
            If plot_transform_c is not a callable function
        """
        from pint import Unit

        # Determine unit string for display
        if isinstance(self.plt_units, Q_):
            unit = f"{self.plt_units.units:~P}"
        elif isinstance(self.plt_units, Unit):
            unit = f"{self.plt_units:~P}"
        else:
            unit = f"{self.plt_units}"

        # Create y-axis label
        y1_label = f"{self.legend_left} [{unit}]"

        # Calculate y values based on display type
        if self.display_as == "mass":
            y1 = (self.m * self.model.m_unit).to(self.plt_units).magnitude
        elif self.display_as == "ppm":
            y1 = self.c * 1e6
            y1_label = "ppm"
        elif self.display_as == "length":
            y1 = (self.c * self.model.l_unit).to(self.plt_units).magnitude
        else:
            y1 = (self.c * self.model.c_unit).to(self.plt_units).magnitude

        # Apply transform function if provided
        if self.plot_transform_c != "None":
            if callable(self.plot_transform_c):
                y1 = self.plot_transform_c(self.c)
            else:
                raise SpeciesError("Plot transform must be a function")

        return y1, y1_label, unit

    def __plot__(self, M: Model, ax) -> None:
        """Plot Model data.

        Parameters
        ----------
        M : Model
            Model containing data to plot
        ax : matplotlib.axes.Axes
            Axes to plot on

        Returns
        -------
        None
        """
        from esbmtk.utility_functions import set_y_limits

        # Convert time and data to display units
        x = (M.time * M.t_unit).to(M.d_unit).magnitude
        y1, y1_label, _unit = self.get_plot_format()

        # Plot first axis with concentration data
        ax.plot(x[1:-2], y1[1:-2], color="C0", label=y1_label)
        ax.set_xlabel(f"{M.time_label} [{M.d_unit:~P}]")
        ax.set_ylabel(y1_label)

        # Add any external data points if present
        for i, d in enumerate(self.external_data_references):
            legend = f"{self.lm} {d.legend}"
            ax.scatter(d.x[1:-2], d.y[1:-2], color=f"C{i + 2}", label=legend)

        # Remove top spine and get legend handles
        ax.spines["top"].set_visible(False)
        handler1, label1 = ax.get_legend_handles_labels()
        set_y_limits(ax, self)

        # Add isotope data on second y-axis if available
        if self.isotopes:
            # Create twin axis for isotope data
            axt = ax.twinx()
            y2 = self.d  # No conversion for isotopes
            axt.plot(x[1:-2], y2[1:-2], color="C1", label=self.legend_right)
            axt.set_ylabel(self.right_axis_label)
            set_y_limits(axt, self)
            ax.spines["top"].set_visible(False)

            # Create combined legend with both axes
            handler2, label2 = axt.get_legend_handles_labels()
            axt.legend(handler1 + handler2, label1 + label2, loc=0).set_zorder(6)
        else:
            # Single axis legend for concentration only
            ax.legend(handler1, label1)
            ax.spines["right"].set_visible(False)
            ax.yaxis.set_ticks_position("left")
            ax.xaxis.set_ticks_position("bottom")

        # Set title
        ax.set_title(self.full_name)

    def info(self, **kwargs) -> None:
        """Show an overview of the object properties.

        Parameters
        ----------
        **kwargs : dict
            Optional arguments:
                - index : int, default=0
                    Index to show data at
                - indent : int, default=0
                    Indentation level for display

        Returns
        -------
        None
        """
        # Set indentation
        indent = kwargs.get("indent", 0)
        ind = " " * indent if indent > 0 else ""
        offset = "  "  # Additional indentation for nested items

        # Print basic information
        print(f"{ind}{self.__str__(kwargs)}")
        print(f"{ind}Data sample:")

        # Print connections
        print(f"\n{ind}Connections:")
        for p in sorted(self.connection_objects):
            print(f"{offset}{ind}{p.full_name}.info()")

        # Print fluxes
        print(f"\n{ind}Fluxes:")
        for i, f in enumerate(self.flux_references):
            print(f"{offset}{ind}{f.full_name}: {self.lodir[i] * f.m[-2]:.2e}")

        # Print usage instructions
        print()
        print("Use the info method on any of the above connections")
        print("to see information on fluxes and processes")

    def flux_summary(self) -> None:
        """Display all flux names in self.lof."""
        print(f"Fluxes in {self.full_name}")
        for f in self.lof:
            print(f"    {f.full_name}")


class Species(SpeciesBase):
    r"""
    Species implementation for chemical species in a model.

    This class provides a concrete implementation of SpeciesBase
    for chemical species with properties like mass, concentration,
    and isotope information.

    Parameters
    ----------
    name : str
        Name of the reservoir
    species : SpeciesProperties
        Handle to the species properties
    delta : float | int | str, optional
        Initial delta value for isotope calculation
    concentration : str | Q\_ | float, optional
        Species concentration (must provide either this or mass)
    mass : str | Q\_, optional
        Species mass (must provide either this or concentration)
    volume : str | Q\_, optional
        Reservoir volume (must provide either this or geometry)
    geometry : list | dict | str, optional
        Geometry parameters for volume calculation
    plot : str, optional
        Whether to plot this species, defaults to "yes"
    plot_transform_c : callable, optional
        Function to transform concentration for plotting
    legend_left : str, optional
        Custom label for left y-axis
    display_precision : float | int, optional
        Decimal places for display, defaults to 0.01
    register : any, optional
        Where to register this species
    isotopes : bool, optional
        Whether to use isotope values
    seawater_parameters : dict | str, optional
        Parameters for seawater calculations

    Examples
    --------
    >>> Species(
    ...     name="foo",
    ...     species=S,
    ...     delta=20,
    ...     concentration="1 mmol/liter",
    ...     volume="1E5 liter",
    ... )

    Notes
    -----
    You must provide either mass or concentration, and either volume or geometry.
    For geometry, you can provide a list [upper_depth, lower_depth, total_area]
    or a dictionary with "area" and "volume" keys.
    """

    # Class constants
    DEFAULT_RTYPE = "regular"

    def __init__(self, **kwargs) -> None:
        """
        Initialize a Species instance.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments for initialization

        Returns
        -------
        None

        Raises
        ------
        SpeciesError
            If required parameters are missing or invalid
        ScaleError
            If unit conversion fails
        """
        from esbmtk import (
            ConnectionProperties,
            Reservoir,
            SinkProperties,
            SourceProperties,
            phc,
        )

        # Define defaults for all parameters
        self.defaults: dict[str, list[any, tuple]] = {
            "name": ["None", (str,)],
            "species": ["None", (str, SpeciesProperties)],
            "delta": ["None", (int, float, str)],
            "concentration": ["None", (str, Q_, float)],
            "mass": ["None", (str, Q_)],
            "volume": ["None", (str, Q_)],
            "geometry": ["None", (list, dict, str)],
            "geometry_unset": [True, (bool,)],
            "plot_transform_c": ["None", (any,)],
            "legend_left": ["None", (str,)],
            "plot": ["yes", (str,)],
            "groupname": ["None", (str,)],
            "rtype": [self.DEFAULT_RTYPE, (str,)],
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
            "full_name": ["None", (str,)],
            "seawater_parameters": ["None", (dict, str)],
            "isotopes": [False, (bool,)],
            "ideal_water": ["None", (str, bool)],
            "has_cs1": [False, (bool,)],
            "has_cs2": [False, (bool,)],
        }

        # List of absolutely required keywords
        self.required_keywords: list = [
            "name",
            "species",
            ["volume", "geometry"],
            ["mass", "concentration"],
        ]
        self.lrk = self.required_keywords  # Legacy alias

        # Initialize variables from kwargs
        self.__initialize_keyword_variables__(kwargs)

        # Set model and parent
        if isinstance(self.register, Model):
            self.model = self.register
        else:
            self.model = self.register.model
        self.parent = self.register

        # Initialize arrays
        self._initialize_arrays()

        # Initialize legacy attributes (replaced from __set_legacy_names__)
        self._initialize_legacy_attributes(kwargs)

        # Set isotopes flag if delta is provided
        if self.delta != "None":
            self.isotopes = True

        # Process geometry information
        self._initialize_geometry()

        # Process concentration/mass information
        self._initialize_concentration_and_mass()

        # Register with model and set up data functions
        self._register_with_model()

        # Set special transform for H+ if needed
        if self.species.name == "Hplus":
            self.plot_transform_c = phc

        # Any auxiliary init - normally empty, but used to extend the
        # reservoir class in virtual reservoirs
        self.__aux_inits__()

    def _initialize_arrays(self) -> None:
        """
        Initialize the data arrays for this species.

        Returns
        -------
        None
        """
        self.c = np.zeros(len(self.model.time))
        self.l = np.zeros(len(self.model.time))
        self.m = np.zeros(len(self.model.time))

    def _initialize_geometry(self) -> None:
        """
        Initialize geometry and volume information.

        Returns
        -------
        None
        """
        from esbmtk.sealevel import get_box_geometry_parameters

        # Handle geometry vs. volume information
        if self.volume == "None":
            get_box_geometry_parameters(self)
        else:
            self.volume = Q_(self.volume).to(self.model.v_unit)

        # Append reservoir volume to list of toc's
        self.model.toc = (*self.model.toc, self.volume.to(self.model.v_unit).magnitude)
        self.v_index = self.model.gcc
        self.model.gcc += 1
        self.c_unit = self.model.c_unit

        # This should probably be species specific
        self.mu: str = self.sp.e.mass_unit

        # Create volume array
        self.v: NDArrayFloat = (
            np.zeros(self.model.number_of_datapoints + 1)
            + self.volume.to(self.model.v_unit).magnitude
        )

    def _initialize_concentration_and_mass(self) -> None:
        """
        Initialize concentration and mass values based on input parameters.

        Returns
        -------
        None

        Raises
        ------
        SpeciesError
            If neither mass nor concentration is provided
        ScaleError
            If unit conversion fails
        """
        # Handle concentration vs mass setup based on species type
        if self.sp.stype == "concentration":
            self._setup_concentration_type()
        elif self.sp.stype == "length":
            self._setup_length_type()

        # Set state flag
        self.state = 0

        # Set left y-axis label for display
        self.lm: str = f"{self.species.n} [{self.mu}/l]"

        # Initialize mass, concentration, and isotope arrays
        self._initialize_data_arrays()

        # Create temporary memory if using multiple solver iterations
        if self.model.number_of_solving_iterations > 0:
            self.mc = np.empty(0)
            self.cc = np.empty(0)
            self.dc = np.empty(0)

    def _setup_concentration_type(self) -> None:
        """
        Set up species with concentration type.

        Returns
        -------
        None

        Raises
        ------
        SpeciesError
            If neither mass nor concentration is provided
        ScaleError
            If unit conversion fails
        """
        if self.mass == "None":
            self._setup_from_concentration()
        elif self.concentration == "None":
            self._setup_from_mass()
        else:
            raise SpeciesError(
                "You need to specify either mass or concentration, not both"
            )

    def _setup_from_concentration(self) -> None:
        """
        Set up species properties based on concentration input.

        Returns
        -------
        None

        Raises
        ------
        ScaleError
            If unit conversion fails
        """
        # Handle concentration input
        if isinstance(self.concentration, str | Q_):
            cc = Q_(self.concentration)
            # Concentration can be mol/kg or mol/l
            _sm, sc = str(cc.units).split(" / ")  # get unit parts
            _mm, mc = str(self.model.c_unit).split(" / ")  # model unit parts

            # Handle unit conversion
            if mc == "liter" and sc == "kilogram":
                # Convert mol/kg to mol/liter assuming density = 1
                cc = Q_(f"{cc.magnitude} {str(self.model.c_unit)}")
                warnings.warn(
                    "\nConvert mol/kg to mol/liter assuming density = 1\n",
                    stacklevel=2,
                )
            elif sc != mc:
                raise ScaleError(
                    f"No transformation available from {cc.units} to {self.model.c_unit}"
                )

            self._concentration = cc.to(self.model.c_unit)
            self.plt_units = self.model.c_unit
        else:
            # Direct concentration value
            self._concentration = self.concentration
            self.plt_units = self.model.c_unit

        # Calculate mass from concentration and volume
        concentration_magnitude = self._concentration.to(self.model.c_unit).magnitude
        volume_magnitude = self.volume.to(self.model.v_unit).magnitude
        mass_value = concentration_magnitude * volume_magnitude

        # Store calculated mass and set display properties
        self._mass = Q_(f"{mass_value} {self.model.c_unit}")
        self.display_as = "concentration"

        # Convert concentration to dimensionless magnitude (fix units)
        self.c = self.c * 0 + concentration_magnitude

        # Apply species-specific unit scaling if specified
        if self.species.scale_to != "None":
            _c, m = str(self.model.c_unit).split(" / ")
            self.plt_units = Q_(f"{self.species.scale_to} / {m}")

    def _setup_from_mass(self) -> None:
        """
        Set up species properties based on mass input.

        Returns
        -------
        None
        """
        # Convert mass to model units
        m = Q_(self.mass)
        self.plt_units = self.model.m_unit
        self._mass = m.to(self.model.m_unit).magnitude

        # Calculate concentration from mass and volume
        mass_magnitude = self._mass
        volume_magnitude = self.volume.to(self.model.v_unit).magnitude
        concentration_value = mass_magnitude / volume_magnitude

        # Store calculated concentration and set display properties
        self._concentration = concentration_value
        self.display_as = "mass"

    def _setup_length_type(self) -> None:
        """
        Set up species with length type.

        Returns
        -------
        None
        """
        self.plt_units = self.model.l_unit

        # Set concentration array with uniform value
        concentration_magnitude = Q_(self.concentration).magnitude
        self.c = np.zeros(self.model.number_of_datapoints + 1) + concentration_magnitude

        # Store properties
        self._concentration = concentration_magnitude
        self.display_as = "length"

    def _initialize_data_arrays(self) -> None:
        """
        Initialize mass, concentration, and isotope arrays.

        Returns
        -------
        None
        """
        # Initialize mass array
        if self.mass == "None":
            self.m = np.zeros(self.model.number_of_datapoints + 1)
        else:
            # Make sure we're using a numerical value
            if isinstance(self._mass, Q_):
                mass_value = self._mass.magnitude
            else:
                mass_value = self._mass

            self.m = np.zeros(self.species.model.steps) + mass_value

        # Initialize light isotope concentration array
        self.l = np.zeros(self.model.number_of_datapoints + 1)

        # Calculate isotope values if delta is provided
        if self.delta != "None":
            self.l = get_l_mass(self.c, self.delta, self.species.r)

    def _register_with_model(self) -> None:
        """
        Register this species with the model and set up data functions.

        Returns
        -------
        None
        """
        # Add to model's reservoir lists
        self.model.lor.append(self)
        if self.rtype != "flux_only":
            self.model.lic.append(self)

        # Register instance in global namespace
        if self.model.register == "local" and self.register == "None":
            self.register = self.model

        self.__register_with_parent__()

        # Select appropriate data handler based on isotope status
        if self.isotopes:
            self.__set_data__ = self.__set_with_isotopes__
        else:
            self.__set_data__ = self.__set_without_isotopes__

    def __aux_inits__(self) -> None:
        """
        Run any auxiliary initialization.

        This method is empty by default but can be overridden by subclasses
        to provide additional initialization steps.

        Returns
        -------
        None
        """
        pass

        """
        Get the mass value.

        Returns
        -------
        float
            The mass value
        """
        return self._mass

    def massto(self, unit):
        r"""
        Convert mass to specified unit.

        Parameters
        ----------
        unit : str | Q\_
            Unit to convert to

        Returns
        -------
        Q\_
            Converted mass
        """
        if isinstance(self._mass, Q_):
            return self._mass.to(unit)
        else:
            # Create a quantity with the model's mass unit and convert
            return Q_(f"{self._mass} {self.model.m_unit}").to(unit)


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
              "rate": ["None", (str, Q, int, float)],
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
            self.isotope_ratio = self.l / self.c


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

    def __init__(self, **kwargs) -> None:
        """Initialize Source Properties instance.

        This method extends the parent class initialization with source-specific functionality.
        """
        # Call the parent class's __init__ method to handle basic initialization
        super().__init__(**kwargs)

        # self.m = np.zeros(self.model.number_of_datapoints + 1)
        # self.c = np.zeros(self.model.number_of_datapoints + 1)

        if self.delta != "None":
            self.l = get_l_mass(self.c, self.delta, self.sp.r)
            self.isotopes = True

        # register with model
        # self.mo.lic.append(self)
        # # wee need these for the ode backend
        # self.rtype = "passive"
        # self.lof: list = []
        # self.atol = [1e-4, 1e-4]
