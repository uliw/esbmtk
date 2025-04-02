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

import time
import typing as tp

import numpy as np
import numpy.typing as npt

if tp.TYPE_CHECKING:
    from .esbmtk import SpeciesProperties

# declare numpy types
NDArrayFloat = npt.NDArray[np.float64]


class KeywordError(Exception):
    """Exception raised for errors in keyword arguments.

    Parameters
    ----------
    message : str
        Explanation of the error

    Examples
    --------
    >>> raise KeywordError("Invalid keyword 'xyz'")
    """

    def __init__(self, message):
        message = f"\n\n{message}\n"
        super().__init__(message)


class MissingKeywordError(Exception):
    """Exception raised when a required keyword argument is missing.

    Parameters
    ----------
    message : str
        Explanation of the error

    Examples
    --------
    >>> raise MissingKeywordError("'name' is a mandatory keyword")
    """

    def __init__(self, message):
        message = f"\n\n{message}\n"
        super().__init__(message)


class InputError(Exception):
    """Exception raised for errors in the input parameters.

    Parameters
    ----------
    message : str
        Explanation of the error

    Examples
    --------
    >>> raise InputError("Value must be positive")
    """

    def __init__(self, message):
        message = f"\n\n{message}\n"
        super().__init__(message)


class FluxSpecificationError(Exception):
    """Exception raised for errors in flux specifications.

    Parameters
    ----------
    message : str
        Explanation of the error

    Examples
    --------
    >>> raise FluxSpecificationError("Unknown flux units")
    """

    def __init__(self, message):
        message = f"\n\n{message}\n"
        super().__init__(message)


class SpeciesPropertiesMolweightError(Exception):
    """Exception raised when molecular weight is missing or invalid.

    Parameters
    ----------
    message : str
        Explanation of the error

    Examples
    --------
    >>> raise SpeciesPropertiesMolweightError("Missing molecular weight for C")
    """

    def __init__(self, message):
        message = f"\n\n{message}\n"
        super().__init__(message)


class InputParsing:
    """Provides various routines to parse and process keyword arguments.

    All derived classes need to declare the allowed keyword arguments,
    their default values and the type in the following format:

    defaults = {"key": [value, (allowed instances)]}

    The recommended sequence is to first set default values via
    __register_variable_names__() and then update with provided values
    using __update_dict_entries__(defaults, kwargs).

    Notes
    -----
    This class is not meant to be instantiated directly.
    """

    def __init__(self):
        raise NotImplementedError("InputParsing has no instance!")

    def __initialize_keyword_variables__(self, kwargs) -> None:
        """
        Check, register and update keyword variables.

        Parameters
        ----------
        kwargs : dict
            Dictionary of keyword arguments to process

        Returns
        -------
        None

        Examples
        --------
        >>> self.__initialize_keyword_variables__({"name": "test", "value": 10})
        """
        self.update = False
        self.__check_mandatory_keywords__(self.lrk, kwargs)
        self.__register_variable_names__(self.defaults, kwargs)
        self.__update_dict_entries__(self.defaults, kwargs)
        self.update = True

    def __check_mandatory_keywords__(self, lrk: list, kwargs: dict) -> None:
        """Verify that all required keywords are present in kwargs.

        Parameters
        ----------
        lrk : list
            List of required keywords or lists of alternative keywords
        kwargs : dict
            Dictionary of provided keyword arguments

        Raises
        ------
        MissingKeywordError
            If a required keyword is missing
        ValueError
            If exactly one of a list of alternative keywords is not provided
        TypeError
            If lrk is not a list or kwargs is not a dictionary

        Notes
        -----
        If an element of lrk is a list, it represents alternative keywords
        where exactly one must be provided.

        Examples
        --------
        >>> self.__check_mandatory_keywords__(["name", ["option1", "option2"]], {"name": "test", "option1": "value"})
        """
        if not lrk:
            return

        # Type checking using the newer isinstance syntax
        if not isinstance(lrk, list):
            raise TypeError(f"Required keywords list must be a list, not {type(lrk)}")

        if not isinstance(kwargs, dict):
            raise TypeError(f"Keywords must be a dictionary, not {type(kwargs)}")

        for key in lrk:
            if isinstance(key, list):
                # This is a list of alternative keywords - exactly one must be provided
                valid_keys = [k for k in key if k in kwargs and kwargs[k] != "None"]

                if len(valid_keys) == 0:
                    # Use context preservation for exceptions
                    try:
                        alternatives = ", ".join(key)
                        raise ValueError(
                            f"No valid alternatives found among: {alternatives}"
                        )
                    except ValueError as err:
                        raise MissingKeywordError(
                            f"At least one of these keywords must be provided: {key}"
                        ) from err
                elif len(valid_keys) > 1:
                    try:
                        choices = ", ".join(valid_keys)
                        raise ValueError(f"Multiple choices provided: {choices}")
                    except ValueError as err:
                        raise ValueError(
                            f"Only one of these keywords should be provided: {key}. "
                            f"Found: {valid_keys}"
                        ) from err
            elif key not in kwargs:
                raise MissingKeywordError(f"'{key}' is a mandatory keyword")
            elif kwargs[key] is None:
                raise MissingKeywordError(
                    f"'{key}' is a mandatory keyword and cannot be None"
                )

    def __register_variable_names__(
        self,
        defaults: dict[str, list[any, tuple]],
        kwargs: dict,
    ) -> None:
        """Register the key-value pairs as local instance variables.

        We register them with their actual variable name and as _variable_name
        to support setter and getter methods and avoid name conflicts.

        Parameters
        ----------
        defaults : dict
            Dictionary with default values and allowed types
        kwargs : dict
            Dictionary of keyword arguments

        Returns
        -------
        None

        Examples
        --------
        >>> defaults = {"name": ["default", (str,)], "value": [0, (int, float)]}
        >>> self.__register_variable_names__(defaults, {})
        """
        for key, value in defaults.items():
            setattr(self, f"_{key}", value[0])
            setattr(self, key, value[0])

        # save kwargs dict
        self.kwargs: dict = kwargs

    def __update_dict_entries__(
        self,
        defaults: dict[str, list[any, tuple]],
        kwargs: dict[str, any],
    ) -> None:
        """
        Validate and update instance attributes with provided keyword arguments.

        Parameters
        ----------
        defaults : dict
            Dictionary with format {"key": [default_value, (allowed_types)]}
        kwargs : dict
            Dictionary with format {"key": value}

        Raises
        ------
        KeywordError
            If a key in kwargs is not in defaults
        InputError
            If a value in kwargs is not of the expected type or fails validation
        ValueError
            If defaults dictionary is empty

        Notes
        -----
        This function assumes that all defaults have been registered with the
        instance via __register_variable_names__()

        Examples
        --------
        >>> defaults = {"name": ["default", (str,)], "value": [0, (int, float)]}
        >>> self.__update_dict_entries__(defaults, {"name": "test", "value": 10})
        """
        self.__validate_defaults_and_kwargs__(defaults, kwargs)

        if not kwargs:
            return  # Nothing to update

        for key, value in kwargs.items():
            self.__process_keyword__(defaults, key, value)

    def __validate_defaults_and_kwargs__(self, defaults, kwargs):
        """
        Validate that defaults dictionary is not empty.

        Parameters
        ----------
        defaults : dict
            Dictionary with default values
        kwargs : dict
            Dictionary of keyword arguments

        Raises
        ------
        ValueError
            If defaults dictionary is empty
        """
        if not defaults:
            raise ValueError("Defaults dictionary cannot be empty")

    def __process_keyword__(self, defaults, key, value):
        """
        Process a single keyword argument.

        Parameters
        ----------
        defaults : dict
            Dictionary with format {"key": [default_value, (allowed_types)]}
        key : str
            The keyword to process
        value : any
            The value to validate and set

        Raises
        ------
        KeywordError
            If key is not in defaults
        InputError
            If value is not of the expected type or fails validation
        """
        # Check if the key exists in defaults
        if key not in defaults:
            raise KeywordError(f"'{key}' is not a valid keyword")

        # Skip None values as they're handled elsewhere
        if value is None:  # or value == "None":
            return

        # Get the expected types
        expected_types = defaults[key][1]

        # Validate the value type
        self.__validate_value_type__(key, value, expected_types)

        # Perform additional validation based on type
        try:
            self._validate_value(key, value, expected_types)
        except Exception as err:
            raise InputError(f"Validation failed for '{key}': {str(err)}") from err

        # Update the values
        self.__update_attribute_values__(defaults, key, value)

    def __validate_value_type__(self, key, value, expected_types):
        """
        Validate that a value is of the expected type.

        Parameters
        ----------
        key : str
            The keyword being validated
        value : any
            The value to validate
        expected_types : type or tuple of types
            The expected types for the value

        Raises
        ------
        InputError
            If value is not of the expected type
        """
        if not isinstance(value, expected_types):
            try:
                actual_type = type(value).__name__
                if isinstance(expected_types, tuple):
                    expected_types_str = ", ".join(t.__name__ for t in expected_types)
                else:
                    expected_types_str = expected_types.__name__

                raise TypeError(
                    f"Value '{value}' has type {actual_type}, expected {expected_types_str}"
                )
            except TypeError as err:
                raise InputError(
                    f"'{value}' for '{key}' must be of type {expected_types_str}, "
                    f"not {actual_type}"
                ) from err

    def __update_attribute_values__(self, defaults, key, value):
        """
        Update the attribute values in both defaults dictionary and instance.

        Parameters
        ----------
        defaults : dict
            Dictionary with format {"key": [default_value, (allowed_types)]}
        key : str
            The keyword to update
        value : any
            The value to set
        """
        defaults[key][0] = value  # update defaults dictionary
        setattr(self, key, value)  # update instance variables
        setattr(self, f"_{key}", value)

    def _validate_value(self, key, value, expected_types):
        """
        Perform additional validation on values based on their types.

        Parameters
        ----------
        key : str
            The keyword being validated
        value : any
            The value to validate
        expected_types : type or tuple of types
            The expected types for the value

        Raises
        ------
        ValueError
            If the value fails validation

        Examples
        --------
        >>> self._validate_value("volume", 10.5, (int, float))
        """
        # String validations
        if isinstance(value, str):
            if key == "name" and (not value or value.isspace()):
                raise ValueError("Name cannot be empty or just whitespace")

        # Numeric validations
        elif isinstance(value, int | float):  # Using newer isinstance syntax
            # Example: values that should be positive
            if key in ("volume", "m_weight", "salinity", "temperature") and value <= 0:
                raise ValueError(f"'{key}' must be positive, got {value}")

            # Example: values that should be within a specific range
            if key == "ph" and (value < 0 or value > 14):
                raise ValueError(f"pH must be between 0 and 14, got {value}")

    def __register_with_parent__(self) -> None:
        """
        Register self as attribute of self.parent and set full name.

        If self.parent == "None", full_name = name.
        Otherwise, full_name = parent.full_name + "." + self.name

        Returns
        -------
        None

        Examples
        --------
        >>> self.__register_with_parent__()
        """
        if self.parent == "None":
            self.full_name = self.name
            reg = self
        else:
            self.full_name = f"{self.parent.full_name}.{self.name}"
            reg = self.parent.model
            if self.full_name in reg.lmo:
                print("\n -------------- Warning ------------- \n")
                print(f"\t {self.full_name} is a duplicate name in reg.lmo")
                print("\n ---------------------- ------------- \n")
            # register with model
            reg.lmo.append(self.full_name)
            reg.lmo2.append(self)
            reg.dmo.update({self.full_name: self})
            setattr(self.parent, self.name, self)
            self.kwargs["full_name"] = self.full_name
        self.reg_time = time.monotonic()

    def __test_and_resolve_duplicates__(self, name, lmo):
        """
        Test for duplicate instance names and resolve by appending _1, _2, etc.

        Parameters
        ----------
        name : str
            The instance name to check
        lmo : list
            List of existing instance names

        Returns
        -------
        tuple
            (resolved_name, updated_lmo_list)

        Examples
        --------
        >>> name, lmo = self.__test_and_resolve_duplicates__("test", ["existing"])
        >>> print(name)
        test
        """
        if name in lmo:
            print(f"\n Warning, {name} is a duplicate, trying with {name}_1\n")
            name = name + "_1"
            name, lmo = self.__test_and_resolve_duplicates__(name, lmo)
        else:
            lmo.append(name)

        return name, lmo


class esbmtkBase(InputParsing):
    """The esbmtk base class template.

    This class handles keyword arguments, name registration and
    other common tasks.

    Examples
    --------
    .. code-block:: python

            # Define required keywords in lrk list
            self.lrk: tp.List = ["name"]

            # Define allowed type per keyword in defaults dict
            self.defaults: dict[str, list[any, tuple]] = {
                "name": ["None", (str)],
                "model": ["None", (str, Model)],
                "salinity": [35, (int, float)],  # int or float
            }

            # Parse and register all keywords with the instance
            self.__initialize_keyword_variables__(kwargs)

            # Register the instance
            self.__register_with_parent__()
    """

    def __init__(self) -> None:
        raise NotImplementedError

    def __repr__(self, log=0) -> str:
        """Return string representation of the object.

        Parameters
        ----------
        log : int, default=0
            If 0 and object was just created (<1 second ago),
            returns empty string to suppress output

        Returns
        -------
        str
            String representation of the object

        Examples
        --------
        >>> repr(obj)
        'ClassName(name = "example", value = 42)'
        """
        from esbmtk import Q_

        m: str = ""

        # suppress output during object initialization
        tdiff = time.monotonic() - self.reg_time

        m = f"{self.__class__.__name__}(\n"
        for k, v in self.kwargs.items():
            if not isinstance({k}, esbmtkBase):
                # check if this is not another esbmtk object
                if "esbmtk" in str(type(v)):
                    m = f"{m}    {k} = {v.name},\n"
                elif isinstance(v, str | Q_):  # Using newer isinstance syntax
                    m = f"{m}    {k} = '{v}',\n"
                elif isinstance(v, list | np.ndarray):  # Using newer isinstance syntax
                    m = f"{m}    {k} = '{v[:3]}',\n"
                else:
                    m = f"{m}    {k} = {v},\n"

        m = "" if log == 0 and tdiff < 1 else f"{m})"
        return m

    def __str__(self, kwargs=None):
        """Return a string representation of the object with its key attributes.

        Parameters
        ----------
        kwargs : dict, optional
            Additional keyword arguments
            - indent : int
                Number of spaces to indent output
            - index : int
                Index to display for array values, default=-2

        Returns
        -------
        str
            Formatted string representation of the object

        Examples
        --------
        >>> str(obj)
        'example (ClassName)
          value = 42'
        """
        if kwargs is None:
            kwargs = {}
        from esbmtk import Q_

        m: str = ""
        off: str = "  "

        ind: str = kwargs["indent"] * " " if "indent" in kwargs else ""
        index = int(kwargs["index"]) if "index" in kwargs else -2
        m = f"{ind}{self.name} ({self.__class__.__name__})\n"
        for k, v in self.kwargs.items():
            if not isinstance({k}, esbmtkBase):
                # check if this is not another esbmtk object
                if "esbmtk" in str(type(v)):
                    pass
                elif isinstance(v, str) and k != "name" or isinstance(v, Q_):
                    m = f"{m}{ind}{off}{k} = {v}\n"
                elif isinstance(v, np.ndarray):
                    m = f"{m}{ind}{off}{k}[{index}] = {v[index]:.2e}\n"
                elif k != "name":
                    m = f"{m}{ind}{off}{k} = {v}\n"

        return m

    def __lt__(self, other) -> bool:  # Fixed return type annotation from None to bool
        """Compare if self is less than other for sorting with sorted().

        Parameters
        ----------
        other : esbmtkBase
            Object to compare with

        Returns
        -------
        bool
            True if self.n < other.n, False otherwise

        Examples
        --------
        >>> sorted([obj2, obj1])  # If obj1.n < obj2.n, returns [obj1, obj2]
        [obj1, obj2]
        """
        return self.n < other.n

    def __gt__(self, other) -> bool:  # Fixed return type annotation from None to bool
        """Compare if self is greater than other for sorting with sorted().

        Parameters
        ----------
        other : esbmtkBase
            Object to compare with

        Returns
        -------
        bool
            True if self.n > other.n, False otherwise

        Examples
        --------
        >>> sorted([obj1, obj2], reverse=True)  # If obj2.n > obj1.n, returns [obj2, obj1]
        [obj2, obj1]
        """
        return self.n > other.n

    def info(self, **kwargs) -> None:
        """Show an overview of the object properties.

        Parameters
        ----------
        **kwargs : dict, optional

        Additional keyword arguments
        ----------------------------
        * ``indent`` : int
            Number of spaces for indentation

        Returns
        -------
        None

        Examples
        --------
        .. code-block:: python

            obj.info(indent=2)
        """
        if "indent" not in kwargs:
            indent = 0
            ind = ""
        else:
            indent = kwargs["indent"]
            ind = " " * indent

        # print basic data about this object
        print(f"{ind}{self.__str__(kwargs)}")

    def __aux_inits__(self) -> None:
        """Auxiliary initialization code.

        This method is a placeholder for additional initialization steps
        that subclasses might implement.

        Returns
        -------
        None

        Examples
        --------
        >>> self.__aux_inits__()
        """
        pass

    def ensure_q(self, arg):
        """Ensure that a given input argument is a quantity object.

        Parameters
        ----------
        arg : str, Quantity, or numeric
            The argument to convert to a Quantity

        Returns
        -------
        Quantity
            The input argument as a Quantity object

        Raises
        ------
        InputError
            If the argument is None, empty, or cannot be converted to a Quantity

        Examples
        --------
        >>> self.ensure_q("10 kg")
        <Quantity(10, 'kilogram')>
        >>> self.ensure_q(existing_quantity)
        <Quantity(existing_quantity)>
        """
        from esbmtk import Q_

        # Check if argument is None or empty
        if arg is None:
            raise InputError("Cannot convert None to a Quantity")

        if isinstance(arg, str) and not arg.strip():
            raise InputError("Cannot convert empty string to a Quantity")

        # Convert based on type
        if isinstance(arg, Q_):
            return arg
        elif isinstance(arg, str):
            try:
                return Q_(arg)
            except Exception as err:
                raise InputError(
                    f"Failed to convert '{arg}' to a Quantity: {str(err)}"
                ) from err
        elif isinstance(arg, int | float):  # Using newer isinstance syntax
            # If only a number is provided with no units, raise an error
            raise InputError(
                f"Numeric value {arg} provided without units. "
                f"Please provide a string with units, e.g., '{arg} kg'"
            )
        else:
            raise InputError(
                f"Cannot convert type {type(arg)} to a Quantity. "
                f"Must be a string or Quantity."
            )

    def help(self) -> None:
        """
        Show all keywords, their default values and allowed types.

        Prints information about all available keywords and
        highlights which ones are mandatory.

        Returns
        -------
        None

        Examples
        --------
        >>> obj.help()
        """
        print(f"\n{self.full_name} has the following keywords:\n")
        for k, v in self.defaults_copy.items():
            print(f"{k} defaults to {v[0]}, allowed types = {v[1]}")
        print()
        print("The following keywords are mandatory:")
        for kw in self.lrk:
            print(f"{kw}")

    def set_flux(self, mass: str, time: str, substance: SpeciesProperties):
        """
        Convert a flux rate to model units (kg/s or mol/s).

        Parameters
        ----------
        mass : str
            Mass value with units, e.g., "12 Tmol" or "500 kg"
        time : str
            Time unit, e.g., "year", "day", "s"
        substance : SpeciesProperties
            Species properties object containing molecular weight information

        Returns
        -------
        Quantity
            Flux rate in model units (mol/time or g/time)

        Raises
        ------
        InputError
            If input parameters are None, empty, or of incorrect type
        FluxSpecificationError
            If unit conversion cannot be performed
        SpeciesPropertiesMolweightError
            If substance has no molecular weight defined

        Examples
        --------
        >>> M.set_flux("12 Tmol", "year", M.C)
        <Quantity(12, 'teramol / year')>

        Notes
        -----
        If model mass units are in mol, no mass unit conversion will be made.
        If model mass units are in kg, the flux will be converted accordingly.
        """
        # Validate all input parameters first
        self.__validate_flux_inputs__(mass, time, substance)

        # Validate substance properties
        self.__validate_substance_properties__(substance)

        # Convert the mass quantity
        r = self.__convert_mass_to_model_units__(mass, substance)

        # Apply the time unit and return the result
        return self.__apply_time_unit__(r, time)

    def __validate_flux_inputs__(self, mass, time, substance):
        """
        Validate the input parameters for set_flux.

        Parameters
        ----------
        mass : str or Quantity
            Mass value with units
        time : str
            Time unit
        substance : SpeciesProperties
            Species properties object

        Raises
        ------
        InputError
            If any input parameter is invalid
        """
        from esbmtk import Q_, ureg

        # Check for empty inputs
        if not mass:
            raise InputError("Mass parameter cannot be empty")
        if not time:
            raise InputError("Time parameter cannot be empty")
        if substance is None:
            raise InputError("Substance parameter cannot be None")

        # Type checks
        if not isinstance(mass, str | Q_):
            raise InputError(f"Mass must be a string or Quantity, not {type(mass)}")
        if not isinstance(time, str):
            raise InputError(f"Time must be a string, not {type(time)}")

        # Validate time units
        try:
            ureg(time)
        except Exception as err:
            raise InputError(f"Invalid time unit: '{time}'. Error: {str(err)}") from err

    def __validate_substance_properties__(self, substance):
        """
        Validate that substance has the required properties.

        Parameters
        ----------
        substance : SpeciesProperties
            Species properties object

        Raises
        ------
        SpeciesPropertiesMolweightError
            If substance properties are missing or invalid
        """
        # Check for m_weight attribute
        if not hasattr(substance, "m_weight"):
            raise SpeciesPropertiesMolweightError(
                f"Substance {getattr(substance, 'full_name', str(substance))} has no 'm_weight' attribute"
            )

        # Check for model unit attributes
        if not hasattr(substance, "mo") or not hasattr(substance.mo, "m_unit"):
            raise SpeciesPropertiesMolweightError(
                f"Substance {getattr(substance, 'full_name', str(substance))} has incomplete model unit definition"
            )

        # Check m_weight value
        if substance.m_weight <= 0:
            raise SpeciesPropertiesMolweightError(
                f"No molecular weight definition for {substance.full_name} (m_weight={substance.m_weight})"
            )

    def __convert_mass_to_model_units__(self, mass, substance):
        """
        Convert mass to model units using substance properties.

        Parameters
        ----------
        mass : str or Quantity
            Mass value with units
        substance : SpeciesProperties
            Species properties object

        Returns
        -------
        Quantity
            Mass converted to model units

        Raises
        ------
        FluxSpecificationError
            If unit conversion fails
        """
        from esbmtk import Q_, ureg

        try:
            mass = Q_(mass) if isinstance(mass, str) else mass
            g_per_mol = ureg("g/mol")

            if mass.is_compatible_with("g") or mass.is_compatible_with("mol"):
                return mass.to(
                    substance.mo.m_unit,  # target unit (mol)
                    "chemistry",  # context
                    mw=substance.m_weight * g_per_mol,  # g/mol
                )
            else:
                raise FluxSpecificationError(
                    f"No known conversion for {mass} (units: {mass.units}) and {substance.full_name}"
                )
        except Exception as err:
            if isinstance(err, FluxSpecificationError):
                raise
            else:
                raise FluxSpecificationError(
                    f"Failed to convert {mass} for {substance.full_name}: {str(err)}"
                ) from err

    def __apply_time_unit__(self, quantity, time_unit):
        """
        Apply time unit to a quantity to get a rate.

        Parameters
        ----------
        quantity : Quantity
            The quantity to convert to a rate
        time_unit : str
            The time unit to apply

        Returns
        -------
        Quantity
            Rate with the specified time unit

        Raises
        ------
        FluxSpecificationError
            If applying the time unit fails
        """
        from esbmtk import ureg

        try:
            return quantity / ureg(time_unit)
        except Exception as err:
            raise FluxSpecificationError(
                f"Failed to apply time unit '{time_unit}': {str(err)}"
            ) from err

    def __update_ode_constants__(self, value) -> int:
        """
        Add a value to the global parameter list and track its index.

        Parameters
        ----------
        value : any
            Value to add to the parameter list

        Returns
        -------
        int
            Index position of the value in the parameter list

        Raises
        ------
        AttributeError
            If the model attribute is missing or does not have required properties
        TypeError
            If model.toc is not a sequence that can be extended

        Examples
        --------
        >>> index = self.__update_ode_constants__(42.0)
        >>> print(index)
        5  # If this was the 6th constant added (zero-indexed)
        """
        # Check if model attribute exists
        if not hasattr(self, "model"):
            raise AttributeError(
                "Cannot update ODE constants: 'model' attribute is missing"
            )

        # Check if model has the required attributes
        if not hasattr(self.model, "toc"):
            raise AttributeError(
                "Model object is missing 'toc' attribute required for ODE constants"
            )

        if not hasattr(self.model, "gcc"):
            raise AttributeError(
                "Model object is missing 'gcc' attribute required for ODE constants"
            )

        # Add the value to the parameter list if it's not "None"
        if value != "None":
            # Validate that toc is a sequence that can be extended
            try:
                if not hasattr(self.model.toc, "__iter__"):
                    raise TypeError("Model.toc must be an iterable")
            except TypeError as err:
                raise TypeError(
                    f"Model.toc must be a sequence, not {type(self.model.toc)}"
                ) from err

            # Update the model's toc tuple with the new value
            self.model.toc = (*self.model.toc, value)

            # Get the current index
            index = self.model.gcc

            # Increment the counter
            self.model.gcc = self.model.gcc + 1
        else:
            index = 0

        return index

    def validate(self) -> bool:
        """
        Validate the object state after initialization.

        This method performs comprehensive validation of the object's attributes
        to ensure they are in a consistent and valid state.

        Returns
        -------
        bool
            True if the object is valid

        Raises
        ------
        ValueError
            If the object fails validation
        AttributeError
            If required attributes are missing

        Examples
        --------
        >>> obj = SomeClass(name="test", value=10)
        >>> obj.validate()  # Returns True if valid, raises exception if not
        True
        """
        # Check for required attributes
        required_attrs = ["name", "full_name"]
        for attr in required_attrs:
            try:
                if not hasattr(self, attr):
                    raise AttributeError(f"Required attribute '{attr}' is missing")

                value = getattr(self, attr)
                if value in (None, "None", ""):
                    raise ValueError(f"Required attribute '{attr}' cannot be empty")
            except (AttributeError, ValueError) as err:
                raise ValueError(
                    f"Required attribute '{attr}' is missing or empty"
                ) from err

        # Validate specific attributes based on their expected properties
        if hasattr(self, "volume") and hasattr(self, "model"):
            from esbmtk import Q_

            # Example: validate that volume has compatible units with the model
            try:
                if (
                    hasattr(self, "volume")
                    and hasattr(self, "model")
                    and isinstance(self.volume, Q_)
                    and hasattr(self.model, "v_unit")
                    and not self.volume.is_compatible_with(self.model.v_unit)
                ):
                    raise ValueError(
                        f"Volume units '{self.volume.units}' are incompatible with model units '{self.model.v_unit}'"
                    )
            except Exception as err:
                raise ValueError(
                    f"Volume units {self.volume.units} are not compatible "
                    f"with model volume units {self.model.v_unit}"
                ) from err
        # Add other validation rules specific to your application

        # If all validations pass, return True
        return True
