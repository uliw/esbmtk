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

import functools
import logging
import typing as tp

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

# from numba import njit

if tp.TYPE_CHECKING:
    from esbmtk import ExternalFunction, Model, Species

np.set_printoptions(precision=4)
# declare numpy types
NDArrayFloat = npt.NDArray[np.float64]


class ScaleError(Exception):
    """Custom Error Class."""

    def __init__(self, message):
        """Initialize Class."""
        message = f"\n\n{message}\n"
        super().__init__(message)


def rmtree(f) -> None:
    """Delete file or files if file is directory.

    :param f: pathlib path object

    """
    if f.is_file():
        f.unlink()
    else:
        for child in f.iterdir():
            rmtree(child)
        f.rmdir()


def phc(c: float) -> float:
    """Calculate concentration as pH.

    c can be a number or numpy array

    Parameters
    ----------
    c : float
        H+ concentration

    Returns
    -------
    float
        pH value

    """
    import numpy as np

    pH: float = -np.log10(c)
    return pH


def debug(func):
    """Print the function signature and return value."""

    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [f"{float(repr(a)):.2e}\n" for a in args]  # 1
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]  # 2
        signature = ", ".join(args_repr + kwargs_repr)  # 3
        print(f"Calling {func.__name__}\n({signature})")
        value = func(*args, **kwargs)
        value = [f"{v:.2e}" for v in value]
        print(f"\n{func.__name__!r} returned\n {value!r}\n")  # 4
        return value

    return wrapper_debug


def get_reservoir_reference(k: str, M: Model) -> tuple:
    """Get SpeciesProperties and Species handles.

    Parameters
    ----------
    k : str
        with the initial flux name, e.g., M_F_A_db_DIC
    M : Model
        Model handle

    Returns
    -------
    tuple
        Species2Species, SpeciesProperties

    Raises
    ------
    ValueError
        If reservoir_name is not of type ConnectionProperties or Species2Species

    """
    from esbmtk import (
        GasReservoir,
        Reservoir,
        Species,
        Species2Species,
        SpeciesProperties,
    )

    key_list = k[2:].split(".")  # get model, reservoir & species name

    if len(key_list) == 3:  # Reservoir
        _model_name, reservoir_name, species_name = key_list

        if hasattr(M, reservoir_name):
            obj = getattr(M, reservoir_name)
        else:
            raise ValueError(f"{reservoir_name} is not in the model")

        if k[0:2] == "R_":
            reservoir = obj
        elif k[0:2] == "F_":
            if isinstance(obj, Species | Reservoir):
                reservoir = obj
            elif isinstance(obj, Species2Species):
                reservoir = obj
                if isinstance(reservoir.source, GasReservoir):
                    species_name = obj.sink.name
                else:
                    species_name = obj.source.name
            else:
                raise ValueError(
                    f"{obj.name} must be Species2Species", f"not {type(obj)}"
                )

    elif len(key_list) == 2:  # (Gas)Species
        _model_name, reservoir_name = key_list
        reservoir = getattr(M, reservoir_name)
    else:
        raise ValueError("kl should look like this F_M.CO2_At")

    species = getattr(M, species_name)
    if not isinstance(species, SpeciesProperties):
        raise NotImplementedError()
    return reservoir, species


def register_new_flux(ec, model_object, dict_key, dict_value) -> list:
    """Register a new flux object with a Species2Species instance.

    Parameters
    ----------
    ec : ExternalCode object
    model_object : Species | Reservoir
        instance to register with
    dict_key : str
        E.g., "M.A_db.DIC"
    dict_value : str
        id value, e.g., "db_cs2"

    Returns
    -------
    list
        list of Flux instances

    """
    from .base_classes import Flux, Species
    from .extended_classes import Reservoir

    if isinstance(model_object, Reservoir):
        spn = dict_key.split(".")[-1]
        species_properties = getattr(model_object.mo, spn)
        species = getattr(
            model_object, species_properties.name
        )  # species to register flux with
    elif isinstance(model_object, Species):
        # in this case rg contains a species not a reservoirgroup
        species_properties = model_object.species
        rg = model_object.register  # get associated reservoir
        species = getattr(rg, species_properties.name)
    else:
        print(f" type(rg) = {type(model_object)}")
        raise NotImplementedError

    if not hasattr(species, "source"):
        species.source = species_properties.name

    ro = list()
    # this also registers the flux with M.lof
    num = ["", "_l"] if species.isotopes else [""]

    for e in num:
        if model_object.model.debug:
            print(f"ro= {model_object.full_name}, dv = {dict_value}, num = {num}")
        f = Flux(
            name=f"{dict_value}{e}",
            species=species_properties,
            rate=0,
            register=species,
            ftype=ec.ftype,
            isotopes=species.isotopes,
            id=f"{species.name}{e}",
        )
        ro.append(f)
        ec.lof.append(f)
        species.lof.append(f)  # register flux
        if model_object.model.debug:
            print(f"fname = {f.full_name}")

    return ro


def register_new_reservoir(r, sp, v):
    """Register a new reservoir."""
    from .base_classes import Species

    rt = Species(  # create new reservoir
        name=sp.name,
        species=sp,
        concentration=f"{v} mol/kg",
        register=r,
        volume=r.volume,
        rtype="computed",
    )
    r.lor.append(rt)
    return [rt]


def register_return_values(ef: ExternalFunction, rg) -> None:
    """Register the return values of an external function instance.

    Parameters
    ----------
    ec : ExternalFunction
        ExternalFunction Instance
    rg : Reservoir | Species
        The Resevoir or Reservoirgroup the external function is
        associated with

    Raises
    ------
    ValueError
        If the return value type is undefined

    Check the return values of external function instances,
    and create the necessary reservoirs, fluxes, or connections
    if they are missing.

    These fluxes are not associated with a connection Object
    so we register the source/sink relationship with the
    reservoir they belong to.

    This fails for GasReservoir since they can have a 1:many
    relatioship. The below is a terrible hack, it would be
    better to express this with several connection
    objects, rather than overloading the source attribute of the
    GasReservoir class.

    """
    from esbmtk import (
        Flux,
        Reservoir,
        Sink,
        SinkProperties,
        Source,
        SourceProperties,
        Species,
        Species2Species,
    )

    M = rg.mo
    # go through each entry in ec.return_values
    for line in ef.return_values:
        if isinstance(line, dict):
            dict_key = next(iter(line))  # get first key
            dict_value = line[dict_key]
            if dict_key[:2] == "F_":  # is flux
                """The following code needs to extract the connection
                object. Names must have the following scheme
                M.Conn_CO2_At_to_H_b_CO2_H_b._F
                """
                key_str = dict_key[2:].split(".")[1]
                if hasattr(M, key_str):
                    o = getattr(M, key_str)
                    if isinstance(o, Flux):
                        o: list = [o]
                    elif isinstance(o, Species2Species):
                        o: list = o.lof  # get flux handle
                    elif isinstance(
                        o,
                        Species
                        | Reservoir
                        | Sink
                        | SinkProperties
                        | Source
                        | SourceProperties,
                    ):
                        o: list = register_new_flux(ef, rg, dict_key[2:], dict_value)
                    else:
                        raise ValueError(f"No recipie for {type(o)}")
                else:
                    raise ValueError(f"{key_str} is not part of the Model definition")

            elif dict_key[:2] == "R_":  # is reservoir
                if dict_key[2:] in M.lor:
                    o: list = [getattr(M, dict_key)]
                else:
                    r, sp = get_reservoir_reference(dict_key, M)
                    o: list = register_new_reservoir(r, sp, dict_value)

                res = o[0]
                if res.rtype == "computed":  # register dummy flux
                    f = Flux(
                        species=sp,  # SpeciesProperties handle
                        rate=0,  # flux value
                        register=res,  # is this part of a group?
                        isotopes=res.isotopes,
                        id=f"{res.full_name}_F",
                        ftype="computed",
                    )
                res.lof.append(f)
                o = [f]

            elif dict_key[:2] == "C_":  # is connection
                raise NotImplementedError
            elif dict_key[:2] == "N_":  # is connection
                pass
            else:
                raise ValueError(f"{dict_key[0:2]} is not defined")

        elif isinstance(line, Species):
            dict_value.ef_results = True
            o = [dict_value]
        else:
            raise ValueError("This should not happen")

        ef.lro.extend(o)  # add to list of returned Objects


def summarize_results(M: Model) -> dict():
    """Summarize all model results.

    At t_max into a hirarchical
    dictionary, where values are accessed in the following way:

    results[basin_name][level_name][species_name]

    e.g., result["A"]["sb"]["O2"]

    """
    results = dict()

    for r in M.lor:
        species_name = r.name
        basin_name = r.register.name[:1]
        level_name = r.register.name[-2:]
        species_value = r.c[-1]

        if basin_name not in results:
            results[basin_name] = {level_name: {species_name: species_value}}
        elif level_name not in results[basin_name]:
            results[basin_name][level_name] = {species_name: species_value}
        elif species_name not in results[basin_name][level_name]:
            results[basin_name][level_name][species_name] = species_value

    return results


def find_matching_strings(s: str, fl: list[str]) -> bool:
    """Test if all elements of fl occur in s.

    Return True if yes,otherwise False
    """
    return all(f in s for f in fl)


# def insert_into_namespace(name, value, name_space=globals()):
#     name_space[name] = value


def add_to(my_list, e):
    """Add element e to list l, but check if the entry already exist.

    If so, throw exception. Otherwise add
    """
    if e not in my_list:  # if not present, append element
        my_list.append(e)


def get_plot_layout(obj):
    """Select a row, column layout.

    Based on the number of objects to display.  The expected argument
    is a reservoir object which contains the list of fluxes in the reservoir
    """
    noo = 1 + sum(f.plot == "yes" for f in obj.lof)
    for _ in obj.ldf:
        noo += 1

    # noo = len(obj.lof) + 1  # number of objects in this reservoir
    logging.debug(f"{noo} subplots for {obj.n} required")

    size, geo = plot_geometry(noo)

    return size, geo


def plot_geometry(noo: int) -> tuple():
    """Define plot geometry based on number of objects to plot."""
    if noo < 2:
        geometry = [1, 1]  # one row, one column
        size = [5, 3]  # width, height in inches
    elif 1 < noo < 3:
        geometry = [2, 1]  # two rows, one column
        size = [5, 6]  # width, height in inches
    elif 2 < noo < 5:
        geometry = [2, 2]  # two rows, two columns
        size = [10, 6]  # width, height in inches
    elif 4 < noo < 7:
        geometry = [3, 2]  # 3 rows, 2 columns
        size = [10, 9]  # width, height in inches
    elif 6 < noo < 10:
        geometry = [4, 2]  # 4 rows, 2 columns
        size = [10, 12]  # width, height in inches
    elif 9 < noo < 13:
        geometry = [5, 2]  # 5 rows, 2 columns
        size = [10, 15]  # width, height in inches
    elif 12 < noo < 16:
        geometry = [6, 2]  # 6 rows, 2 columns
        size = [10, 18]  # width, height in inches
    else:
        m = (
            "plot geometry for more than 15 fluxes is not yet defined"
            "Consider calling flux.plot individually on each flux in the reservoir"
        )
        raise ValueError(m)

    return size, geometry


def list_fluxes(self, name, i) -> None:
    """Echo all fluxes in the reservoir to the screen."""
    for f in self.lof:  # show the processes
        for p in f.lop:
            p.describe()

    for f in self.lof:
        f.describe(i)  # print out the flux data


def set_y_limits(ax: plt.Axes, obj: any) -> None:
    """Prevent the display or arbitrarily small differences."""
    bottom, top = ax.get_ylim()
    if (top - bottom) < obj.display_precision:
        top = bottom + obj.display_precision
        bottom -= obj.display_precision
        ax.set_ylim(bottom, top)


def is_name_in_list(n: str, my_list: list) -> bool:
    """Test if an object name is part of the object list."""
    return any(e.full_name == n for e in my_list)


def get_object_from_list(name: str, my_list: list) -> any:
    """Match a name to a list of objects.

    Return the object
    """
    match: bool = False
    for o in my_list:
        if o.full_name == name:
            r = o
            match = True

    if match:
        return r
    else:
        raise ValueError(f"Object = {o.full_name} has no matching flux {name}")


def sort_by_type(my_list: list, t: list, m: str) -> list:
    """Divide a list by type into new lists.

    This function will return a
    list and it is up to the calling code to unpack the list

    l is list with various object types
    t is a list which contains the object types used for sorting
    m is a string for the error function
    """
    # from numbers import Number

    lc = my_list.copy()
    rl = []

    for ot in t:  # loop over object types
        a = []
        for e in my_list:  # loop over list elements
            if isinstance(e, ot):
                a.append(e)  # add to temporary list
                lc.remove(e)  # remove this element

        rl.append(a)  # save the temporary list to rl

    # at this point, all elements of lc should have been processed
    # if not, lc contains element which are of a different type
    if lc:
        raise TypeError(m)

    return rl


def get_object_handle(res: list, M: Model):
    """Test if the key is a global reservoir handle.

    Or exists in the model namespace

    :param res: tp.List of strings, or reservoir handles
    :param M: Model handle
    """
    r_list: list = []

    if not isinstance(res, list):
        res = [res]
    for o in res:
        if o in M.dmo:  # is object known in global namespace
            r_list.append(M.dmo[o])
        elif o in M.__dict__:  # or does it exist in Model namespace
            r_list.append(getattr(M, o))
        else:
            print(f"{o} is not known for model {M.name}")
            raise ValueError(f"{o} is not known for model {M.name}")

    if len(r_list) == 1:
        r_list = r_list[0]

    return r_list


def split_key(k: str, M: any) -> any | any | str:
    """Split the string k with letters _to_.

    Test if optional id string is present
    """
    if "_to_" not in k:
        raise ValueError("Name must follow 'Source_to_Sink' format")

    source = k.split("_to_")[0]
    sinkandid = k.split("_to_")[1]
    if "@" in sinkandid:
        sink = sinkandid.split("@")[0]
        cid = sinkandid.split("@")[1]
    else:
        sink = sinkandid
        cid = ""

    sink = get_object_handle(sink, M)
    source = get_object_handle(source, M)

    return (source, sink, cid)


def make_dict(keys: list, values: list) -> dict:
    """Create a dictionary from a list and value, or from two lists."""
    d = {}

    if isinstance(values, list):
        if len(values) == len(keys):
            d: dict = dict(zip(keys, values, strict=False))
        else:
            raise ValueError("key and value list must be of equal length")
    else:
        values: list = [values] * len(keys)
        d: dict = dict(zip(keys, values, strict=False))

    return d


def initialize_reservoirs(M: Model, box_dict: dict) -> list(Species):
    """Initialize one or more reservoirs.

    Based
    on the data in box_dict (see the example below).

    Parameters
    ----------
    M : Model
        Model handle
    box_dict : dict
        dictionary with reservoir parameters

    Returns
    -------
    tp.List
        list of all Species objects in box_dict

    Raises
    ------
    ValueError
        If there are no Species objects in the dictionary


    Example::

        box_parameters = {  # name: [[ud, ld ap], T, P, S]
            # Atlantic Ocean
            "M.A_sb": {
                "g": [0, -100, A_ap],
                "T": 20,
                "P": 5,
                "S": 34.7,
                "c": {M.PO4: "2.1 mmol/kg",
                      M.DIC: "2.21 mmol/kg",
                      M.TA: "2.31 mmol/kg",
                      }
        species_list = initialize_reservoirs(M, box_parameters)

    Note: the first entry in the box_parameters dict must be a regular reservoir, not
    a Source or Sink.
    """
    from esbmtk import Reservoir, SinkProperties, SourceProperties, SpeciesProperties

    # loop over reservoir names
    for box_name, value in box_dict.items():
        match value.get("ty"):  # type is given
            case "Source":
                SourceProperties(
                    name=box_name,
                    species=value["sp"],
                    delta=value.get("d", {}),
                    isotopes=value.get("i", {}),
                    register=M,
                )
            case "Sink":
                SinkProperties(
                    name=box_name,
                    species=value["sp"],
                    delta=value.get("d", {}),
                    isotopes=value.get("i", {}),
                    register=M,
                )
            case _:  # default to creating a reservoir
                _rg = Reservoir(
                    name=box_name,
                    geometry=value["g"],
                    concentration=value.get("c", "0 mol/kg"),
                    delta=value.get("d", {}),
                    isotopes=value.get("i", {}),
                    plot=value.get("p", {}),
                    seawater_parameters={
                        "temperature": value["T"],
                        "pressure": value["P"],
                        "salinity": value["S"],
                    },
                    register=M,
                )
    first_key = list(box_dict.keys())[0]
    b_type = box_dict[first_key].get("ty")
    if b_type == "Source" or b_type == "Sink":
        raise ValueError(
            "\nThe first entry in the box_parameter dict must be a regular reservoir\n"
            "You likely started with a Source or Sink\n"
            "Check your arguments to initialize_reservoirs()\n"
        )

    species_dict = box_dict[first_key].get("c", "None")
    if isinstance(species_dict, dict):
        species_list = list(species_dict.keys())
        if not isinstance(species_list[0], SpeciesProperties):
            raise ValueError(
                f"{species_list[0]} must be of type SpeciesProperties",
                f"not {type(species_list[0])}",
            )
    else:
        raise ValueError("No species in species dict!")
    return species_list


def build_concentration_dicts(cd: dict, bg: dict) -> dict:
    """Build a dict which can be used by create_reservoirs.

    :param bg: dict where the box_names are dict keys.
    :param cd: dictionary

    with the following format::

        cd = {
             # species: [concentration, isotopes]
             PO4: [Q_("2.1 * umol/liter"), False],
             DIC: [Q_("2.1 mmol/liter"), False],
            }

    This function returns a new dict in the following format

    #  box_names: [concentrations, isotopes]
    d= {"bn": [{PO4: .., DIC: ..},{PO4:False, DIC:False}]}

    """
    if isinstance(bg, dict):
        box_names: list = bg.keys()
    elif isinstance(bg, list):
        box_names: list = bg
    elif isinstance(bg, str):
        box_names: list = [bg]
    else:
        raise ValueError("This should never happen")

    icd: dict = {}
    td1: dict = {}  # temp dictionary
    td2: dict = {}  # temp dictionary
    td3: dict = {}  # temp dictionary

    # create the dicts for concentration and isotopes
    for k, v in cd.items():
        td1[k] = v[0]
        td2[k] = v[1]
        td3[k] = v[2]

    # box_names: tp.List = bg.keys()
    for bn in box_names:  # loop over box names
        icd[bn] = [td1, td2, td3]

    return icd


def calc_volumes(bg: dict, M: any, h: any) -> list:
    """Calculate volume contained in a given depth interval.

    bg is a dictionary in the following format::

      bg={
          "hb": (0.1, 0, 200),
          "sb": (0.9, 0, 200),
         }

    where the key must be a valid box name, the first entry of the list denoted
    the areal extent in percent, the second number is upper depth limit, and last
    number is the lower depth limit.

    M must be a model handle
    h is the hypsometry handle

    The function returns a list with the corresponding volumes

    """
    # from esbmtk import hypsometry

    v: list = []  # list of volumes

    for v in bg.values():
        a = v[0]
        u = v[1]
        ll = v[2]

        v.append(h.volume(u, ll) * a)

    return v


def get_longest_dict_entry(d: dict) -> int:
    """Get length of each item in the connection dict."""
    l_length = 0  # length of  longest list
    p_length = 0  # length of single parameter
    nl = 0  # number of lists
    ll = []

    # we need to cover the case where we have two lists of different length
    # this happens if we have a long list of tuples with matched references,
    # as well as a list of species
    for v in d.values():
        if isinstance(v, list):
            nl = nl + 1
            if len(v) > l_length:
                l_length = len(v)
                ll.append(l_length)

        else:
            p_length = 1

    if nl > 1 and ll.count(l_length) != len(ll):
        raise ValueError("Mapping for multiple lists is not supported")

    if l_length > 0 and p_length == 0:
        case = 0  # Only lists present
    if l_length == 0 and p_length == 1:
        case = 1  # Only parameters present
    if l_length > 0 and p_length == 1:
        case = 2  # Lists and parameters present

    return case, l_length


def convert_to_lists(d: dict, my_list: int) -> dict:
    """Expand mixed dict entries.

    (i.e. list and single value) such that they are all lists
    of equal length
    """
    cd = d.copy()

    for k, v in cd.items():
        if not isinstance(v, list):
            p = [v for _ in range(my_list)]
            d[k] = p

    return d


def get_sub_key(d: dict, i: int) -> dict:
    """Take a dict which has where the value is a list.

    Return the key with the n-th value of that list
    """
    return {k: v[i] for k, v in d.items()}


def expand_dict(d: dict, mt: str = "1:1") -> int:
    """Determine dict structure.

    in case we have mutiple connections with mutiple species, the
    default action is to map connections to species (t = '1:1'). If
    you rather want to create mutiple connections (one for each
    species) in each connection set t = '1:N'

    """
    # loop over dict entries
    # ck = connection key
    # cd = connection dict

    r: dict = {}  # the dict we will return

    for ck, cd in d.items():  # loop over connections
        rd: dict = {}  # temp dict
        nd: dict = {}  # temp dict
        case, length = get_longest_dict_entry(cd)

        if isinstance(ck, tuple):
            # assume 1:1 mapping between tuple and connection parameters
            if mt == "1:1":
                # prep dictionaries
                if case == 0:
                    nd = cd
                elif case == 1:  # only parameters present. Expand for each tuple entry
                    length = len(ck)
                    nd = convert_to_lists(cd, length)
                elif case == 2:  # mixed list present, Expand list
                    nd = convert_to_lists(cd, length)

                # for each connection group in the tuple
                if length != len(ck):
                    message = (
                        f"The number of connection properties ({length})\n"
                        f"does not match the number of connection groups ({len(ck)})\n"
                        f"did you intend to do a 1:N mapping?"
                    )
                    raise ValueError(message)

                # map property dicts to connection group names
                i = 0
                for t in ck:
                    rd[t] = get_sub_key(nd, i)
                    i = i + 1

            elif mt == "1:N":  # apply each species to each connection
                if case == 0:
                    nd = cd
                elif case == 1:  # only parameters present. Expand for each tuple entry
                    length = len(ck)
                    nd = convert_to_lists(cd, length)
                elif case == 2:  # mixed list present, Expand list
                    nd = convert_to_lists(cd, length)

                for t in ck:  # apply the entire nd dict to all connections
                    rd[t] = nd
            else:
                raise ValueError(f"{mt} is not defined. must be '1:1' or '1:N'")

        else:
            if case in [0, 1]:  # only lists present, case 3
                nd = cd
            elif case == 2:  # list and parameters present case 4
                nd = convert_to_lists(cd, length)
            rd[ck] = nd

        # update the overall dict and move to the next entry
        # r |= rd
        r.update(rd)

    return r


def create_bulk_connections(ct: dict, M: Model, mt: int = "1:1") -> dict:
    """Create connections from a dictionary.

    The dict can have the following format:

    mt = mapping type. See below for explanation

    # na: names, tuple or str. If lists, all list elements share the same properties
    # sp: species list or species
    # ty: type, str
    # ra: rate, Quantity
    # sc: scale, Number
    # re: reference, optional
    # al: epsilon, optional
    # de: delta, optional
    # bp: bypass, see scale_with_flux
    # si: signal
    # mx: True, optional defaults to False. If set, it will create forward and backward fluxes (i.e. mixing)

    There are 6 different cases how to specify connections

    Case 1 One connection, one set of parameters
           ct1 = {"sb2hb": {"ty": "scale", 'ra'....}}

    Case 2 One connection, one set of instructions, one subset with multiple parameters
           This will be expanded to create connections for each species
           ct2 = {"sb2hb": {"ty": "scale", "sp": ["a", "b"]}}

    Case 3 One connection complete set of multiple characters. Similar to case 2,
           but now all parameters are given explicitly
           ct3 = {"sb2hb": {"ty": ["scale", "scale"], "sp": ["a", "b"]}}

    Case 4 Multiple connections, one set of parameters. This will create
           identical connection for "sb2hb" and  "ib2db"
           ct4 = {("sb2hb", "ib2db"): {"ty": "scale", 'ra': ...}}

    Case 5 Multiple connections, one subset of multiple set of parameters. This wil
          create a connection for species 'a' in sb2hb and with species 'b' in ib2db
           ct5 = {("sb2hb", "ib2db"): {"ty": "scale", "sp": ["a", "b"]}}

    Case 6 Multiple connections, complete set of parameters of multiple parameters
           Same as case 5, but now all parameters are specified explicitly
           ct6 = {("sb2hb", "ib2db"): {"ty": ["scale", "scale"], "sp": ["a", "b"]}}


    The default interpretation for cases 5 and 6 is that each list
    entry corresponds to connection. However, sometimes we want to
    create multiple connections for multiple entries. In this case
    provide the mt='1:N' parameter which will create a connection for
    each species in each connection group. See the below example.

    It is easy to shoot yourself in the foot. It is best to try the above first with
    some simple examples, e.g.,

    from esbmtk import expand_dict
    ct2 = {"sb2hb": {"ty": "scale", "sp": ["a", "b"]}}

    It is best to use the show_dict function to verify that your input
    dictionary produces the correct results!

    """
    from esbmtk.utility_functions import create_connection, expand_dict

    # expand dictionary into a well formed dict where each connection
    # has a fully formed entry
    c_ct = expand_dict(ct, mt=mt)

    # loop over dict entries and create the respective connections
    for k, v in c_ct.items():
        if isinstance(k, tuple):  # loop over names in tuple
            for c in k:
                create_connection(c, v, M)
        elif isinstance(k, str):
            create_connection(k, v, M)
        else:
            raise ValueError(f"{c} must be string or tuple")

    return c_ct


def create_connection(n: str, p: dict, M: Model) -> None:
    """Create a connection group.

    It is assumed that all rates are in liter/year or mol per year.  This may
    not be what you want or need.

    :param n: a connection key.  if the mix flag is given interpreted as mixing
              a connection between sb and db and thus create connections in both
              directions
    :param p: a dictionary holding the connection properties
    :param M: the model handle
    """
    from esbmtk import Q_, ConnectionProperties

    # get the reservoir handles by splitting the key
    source, sink, cid = split_key(n, M)
    # create default connections parameters and replace with values in
    # the parameter dict if present.
    los = list(p["sp"]) if isinstance(p["sp"], list) else [p["sp"]]
    ctype = p.get("ty", "None")
    scale = p.get("sc", 1)
    rate = Q_("0 mol/a") if "ra" not in p else p["ra"]
    ref_flux = p.get("re", "None")
    epsilon = p.get("ep", "None")
    # alpha = p.get("al", "None")
    delta = p.get("de", "None")
    cid = f"{cid}"
    bypass = p.get("bp", "None")
    # species = p.get("sp", "None")
    signal = p.get("si", "None")

    if isinstance(scale, Q_):
        if scale.check(["dimensionless"]):
            scale = scale.magnitude
        else:
            scale = scale.to(M.f_unit).magnitude

    # expand arguments
    ctype = make_dict(los, ctype)
    scale = make_dict(los, scale)  # get rate from dictionary
    rate = make_dict(los, rate)
    ref_flux = make_dict(los, ref_flux)
    epsilon = make_dict(los, epsilon)
    delta = make_dict(los, delta)
    bypass = make_dict(los, bypass)
    signal = make_dict(los, signal)

    name = f"{M.name}.ConnGrp_{source.name}_to_{sink.name}"
    if f"{name}" in M.lmo:  # Test if CG exists
        cg = getattr(M, name.split(".")[1])  # retriece CG object
        cg.add_connections(
            source=source,
            sink=sink,
            ctype=ctype,
            scale=scale,  # get rate from dictionary
            rate=rate,
            ref_flux=ref_flux,
            epsilon=epsilon,
            delta=delta,
            bypass=bypass,
            register=M,
            signal=signal,
            id=cid,  # get id from dictionary
        )
    else:  # Create New ConnectionProperties
        cg = ConnectionProperties(
            source=source,
            sink=sink,
            ctype=ctype,
            scale=scale,  # get rate from dictionary
            rate=rate,
            ref_flux=ref_flux,
            epsilon=epsilon,
            delta=delta,
            bypass=bypass,
            register=M,
            signal=signal,
            id=cid,  # get id from dictionary
        )


def get_name_only(o: any) -> any:
    """Test if item is an esbmtk type.

    If yes, extract the name
    """
    from esbmtk import Flux, Reservoir, Species, SpeciesProperties

    return (
        o.full_name
        if isinstance(o, Flux | Species | Reservoir | SpeciesProperties)
        else o
    )


def get_simple_list(my_list: list) -> list:
    """Return a list.

    Which only has the full name
    rather than all the object properties
    """
    return [get_name_only(e) for e in my_list]


def show_dict(d: dict, mt: str = "1:1") -> None:
    """Show dict entries in an organized manner."""
    from esbmtk import expand_dict, get_name_only, get_simple_list

    ct = expand_dict(d, mt)
    for _ck, cv in ct.items():
        for _pk, pv in cv.items():
            get_simple_list(pv) if isinstance(pv, list) else get_name_only(pv)


def find_matching_fluxes(my_list: list, filter_by: str, exclude: str) -> list:
    """Loop over all reservoirs in my_list.

    And extract the names of all fluxes
    which match the filter string. Return the list of names (not objects!)
    """
    lof: set = set()

    for r in my_list:
        for f in r.lof:
            if filter_by in f.full_name and exclude not in f.full_name:
                lof.add(f)

    return list(lof)


def reverse_key(key: str) -> str:
    """Reverse a connection key e.g., sb2db@POM becomes db2sb@POM."""
    left = key.split("@")
    left = left[0]
    rs = left.split("_to_")
    r1 = rs[0]
    r2 = rs[1]

    return f"{r2}_to_{r1}"


def reverse_tick_labels_factory(max_tick_value):
    """Reverse the label of the x-axis ticks but not the data.

    Usage:

    From matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(reverse_tick_labels_factory(x_max)))

    where xmax = max value on the x-axis
    """

    def formatter(x, pos):
        return f"{max_tick_value - x:.1f}"

    return formatter


def get_connection_keys(
    f_list: set,
    ref_id: str,
    target_id: str,
    inverse: bool,
    exclude: str,
) -> list[str]:
    """Extract connection keys from set of flux names.

    Replace ref_id with
    target_id so that the key can be used in create_bulk_connnections()

    :param f_list: a set with flux objects
    :param ref_id: string with the reference id
    :param target_id: string with the target_id
    :param inverse: Bool, optional, defaults to false

    :return cnc_l: a list of connection keys (str)

    The optional inverse parameter, can be used where in cases where the
    flux direction needs to be reversed, i.e., the returned key will not read
    sb2db@POM, but db2s@POM
    """
    cnc_l: list = []  # list of connection keys

    for f in f_list:
        # get connection and flux name
        fns = f.full_name.split(".")
        cnc = fns[1][3:]  # get key without leadinf C_
        if inverse:
            cnc = reverse_key(cnc)

        cnc.replace(ref_id, target_id)
        # create new cnc string
        cnc = f"{cnc}_to_{target_id}"
        cnc_l.append(cnc)

    return cnc_l


def gen_dict_entries(M: Model, **kwargs) -> tuple(tuple, list):
    """Find all fluxes that contain the reference string.

    Create a new
    Species2Species instance that connects the flux matching ref_id, with a flux
    matching target_id.  The function will a tuple containig the new connection
    keys that can be used by the create bulk_connection() function.  The second
    return value is a list containing the reference fluxes.

    The optional inverse parameter, can be used where in cases where the flux
    direction needs to be reversed, i.e., the returned key will not read
    sb_to_dbPOM, but db_to_sb@POM

    :param M: Model or list

    :param kwargs: keyword dictionary, known keys are ref_id, and raget_id, inverse

    :return f_list: tp.List of fluxes that match ref_id

    :return k_tuples: tuple of connection keys

    """
    from esbmtk import Model

    ref_id = kwargs["ref_id"]
    target_id = kwargs["target_id"]
    inverse = kwargs.get("inverse", False)
    exclude_str = kwargs.get("exclude", "None")

    # find matching fluxes
    if isinstance(M, Model):
        f_list: list = find_matching_fluxes(
            M.loc,
            filter_by=ref_id,
            exclude=exclude_str,
        )
    elif isinstance(M, list):
        f_list: list = find_matching_fluxes(
            M,
            filter_by=ref_id,
            exclude=exclude_str,
        )
    else:
        raise ValueError(f"gen_dict_entries: M must be list or Model, not {type(M)}")

    k_tuples: list = get_connection_keys(
        f_list,
        ref_id,
        target_id,
        inverse,
        exclude_str,
    )

    return tuple(k_tuples), f_list


def build_ct_dict(d: dict, p: dict) -> dict:
    """Build a connection dictionary.

    From a dict containing connection
    keys, and a dict containing connection properties. This is most
    useful for connections which a characterized by a fixed rate but
    apply to many species. E.g., mixing fluxes in a complex model etc.

    """
    # a = {k: {"sc": v} | p for k, v in d.items()}
    a = {}
    for k, v in d.items():
        a[k] = p.copy()
        a[k]["sc"] = v

    return a


def get_string_between_brackets(s: str) -> str:
    """Parse string and extract substring between square brackets."""
    s = s.split("[")
    if len(s) < 2:
        raise ValueError(f"Column header {s} must include units in square brackets")

    s = s[1]

    s = s.split("]")

    if len(s) < 2:
        raise ValueError(f"Column header {s} must include units in square brackets")

    return s[0]


def check_for_quantity(quantity, unit):
    r"""Check if keyword is quantity or string an convert as necessary.

    - If input is a string, convert string into a quantity
    - If input is a quantity, do nothing
    - if input is a number, convert to default quantity

    Parameters
    ----------
    quantity : str | quantity | float | int
        e.g., "12 m/s", or 12,
    unit : str
        desired unit for keyword, e.g., "m/s"

    Returns
    -------
    Q\_
        Returns a Quantity

    Raises
    ------
    ValueError
        if keywword is neither number, str or quantity

    """
    from esbmtk import Q_

    if isinstance(quantity, str):
        quantity = Q_(quantity)
    elif isinstance(quantity, float | int):
        quantity = Q_(f"{quantity} {unit}")
    elif not isinstance(quantity, Q_):
        raise ValueError("kw must be string, number or Quantity")

    return quantity


def map_units(obj: any, v: any, *args) -> float:
    """Parse v to see if it is a string.

    If yes, map to quantity.
    parse v to see if it is a quantity, if yes, map to model units
    and extract magnitude, assign mangitude to return value
    if not, assign value to return value

    :param obj: connection object
    :param v: input string/number/quantity
    :args: tp.List of model base units
    :returns: number

    :raises ScaleError: if input cannot be mapped to a model unit
    """
    from . import Q_

    m: float = 0
    match: bool = False

    # test if string, map to quantity if yes
    if isinstance(v, str):
        v = Q_(v)

    # test if we find a matching dimension, map if true
    if isinstance(v, Q_):
        for q in args:
            if v.dimensionality == q.dimensionality:
                m = v.to(q).magnitude
                match = True

        if not match:
            raise ScaleError(
                f"Missing base units for the scale in {obj.full_name}, this should not happen"
                f"complain to the program author"
            )

    else:  # no quantity, so it should be a number
        m = v

    if not isinstance(m, int | float):
        raise ValueError(f"m is {type(m)}, must be float, v={v}. Something is fishy")

    return m


def __find_flux__(reservoirs: list, full_name: str):
    """Find a Flux object based on its full_name.

    PRECONDITIONS: full_name must contain the full_name of the Flux

    Parameters
    ----------
        reservoirs: tp.List containing all reservoirs
        full_name: str specifying the full name of the flux (boxes.flux_name)

    """
    needed_flux = None
    for res in reservoirs:
        for flux in res.lof:
            if flux.full_name == full_name:
                needed_flux = flux
                break
        if needed_flux is not None:
            break
    if needed_flux is None:
        raise NameError(
            f"add_carbonate_system: Flux {full_name} cannot be found in any of the reservoirs in the Model!"
        )

    return needed_flux


def __checktypes__(allwed_keys: dict[any, any], provided_keys: dict[any, any]) -> None:
    """Look up the allowed data type for this key in allowed_keys.

    av = dictinory with the allowed input keys and their type
    pv = dictionary with the user provided key-value data
    """
    k: any
    v: any

    # loop over provided keywords
    for k, v in provided_keys.items():
        # check av if provided value v is of correct type
        if allwed_keys[k] != any and not isinstance(v, allwed_keys[k]):
            raise TypeError(
                f"{type(v)} is the wrong type for '{k}', should be '{allwed_keys[k]}'"
            )


def dict_alternatives(d: dict, e: str, a: str) -> any:
    """Use  dictionary =d=, an expression =e=, and an alternative expression =a=.

    Returns the value associated with
    either =a= or =e= in the dictionary =d=.

    :param d: A dictionary.
    :param e: The first expression to check.
    :param a: The alternative expression to check.

    :returns r: The value associated with either =a= or =e= in the dictionary =d=.

    :raises ValueError: If neither =a= nor =e= are found in the dictionary.
    """
    if a in d:
        r = d[a]
    elif e in d:
        r = d[e]
    else:
        raise ValueError(f"You must specify either {e} or {a} \nd = {d}")

    return r


def __checkkeys__(lrk: list, lkk: list, kwargs: dict) -> None:
    """Check if the mandatory keys are present.

    lrk = list of required keywords
    lkk = list of all known keywords
    kwargs = dictionary with key-value pairs

    """
    k: str
    # test if the required keywords are given
    for k in lrk:  # loop over required keywords
        if isinstance(k, list):  # If keyword is a list
            s: int = 0  # loop over allowed substitutions
            for e in k:  # test how many matches are in this list
                if (
                    e in kwargs
                    and not isinstance(e, np.ndarray | list)
                    and kwargs[e] != "None"
                ):
                    s = s + 1
            if s > 1:  # if more than one match
                raise ValueError(f"You need to specify exactly one from this list: {k}")

        elif k not in kwargs:
            raise ValueError(f"You need to specify a value for {k}")

    tl: list[str] = [k for k, v in lkk.items()]
    # test if we know all keys
    for k in kwargs:
        if k not in lkk:
            raise ValueError(f"{k} is not a valid keyword. \n Try any of \n {tl}\n")


def __addmissingdefaults__(lod: dict, kwargs: dict) -> dict:
    """Test if the keys in lod exist in kwargs.

    otherwise add them with the default values from lod
    """
    new: dict = {}
    if lod:
        for k, v in lod.items():
            if k not in kwargs:
                new[k] = v

    # kwargs |= new
    kwargs.update(new)
    return kwargs


def data_summaries(
    M: Model,
    species_names: list,
    box_names: list,
    register_with="None",
) -> list:
    r"""Group results by species and Reservoirs.

    :param M: model instance
    :param species_names: tp.List of species instances
    :param box_names: tp.List of Reservoir instances
    :param register_with: defaults to M

    :returns pl: a list of datafield instances to be plotted

    Note: because pl is a list, you need to expand it before
    using it in the plot method. Either use

    M.plot(pl)

    or
    M.plot([\*pl])

    If you call a plot with many subfigures, use pl like this:

    M.plot([o1, o2, \*pl])

    """
    from esbmtk import DataField, VectorData

    if register_with == "None":
        register_with = M

    pl = []
    for sp in species_names:
        data_list = []
        label_list = []
        t = ""
        for b in box_names:
            if isinstance(b, VectorData):
                # FIXME: Should a be b????
                data_list.append(a)
                label_list.append(a.full_name)
                t = b.name
            if hasattr(b, f"{sp.name}"):
                a = getattr(b, f"{sp.name}")  # this is a reservoir
                y1, y1_label, unit = a.get_plot_format()
                data_list.append(y1)
                label_list.append(f"{b.name}")
                t = sp.display_as if hasattr(sp, "display_as") else sp.name

        df = DataField(
            name=f"{sp.name}_df",
            register=register_with,
            x1_data=M.time,
            y1_data=data_list,
            y1_label=label_list,
            y1_legend=f"{sp.name} [{unit}]",
            x1_as_time=True,
            title=t,
        )
        pl.append(df)
        """ check if species has isotope data. This needs to be done
        after the above loop, so that the data is in its own datafield.
        """
        data_list = []
        label_list = []
        for b in box_names:
            if hasattr(b, f"{sp.name}"):
                a = getattr(b, f"{sp.name}")
                if a.isotopes:
                    unit = f"{a.species.element.d_label}"
                    data_list.append(a.d)
                    label_list.append(f"{b.name}")

        if len(data_list) > 0:
            df = DataField(
                name=f"d_{sp.name}_df",
                register=register_with,
                x1_data=M.time,
                y1_data=data_list,
                y1_label=label_list,
                y1_legend=f"{unit} [{sp.element.d_scale}]",
                x1_as_time=True,
                title=sp.element.d_label,
            )
            pl.append(df)

    return pl


# @njit(fastmath=True)
def get_l_mass(m: float, d: float, r: float) -> float:
    """Get mass of light isotope.

    :param m: mass or concentration
    :param d: delta value
    :param r: isotopic reference ratio

    return mass or concentration of the light isotopeb
    """
    return (1000.0 * m) / ((d + 1000.0) * r + 1000.0)


def get_delta_h(R) -> float:
    """Calculate the delta of a flux or reservoir.

    :param R: Species or Flux handle

    returns d as vector of delta values
    R.c = total concentration
    R.l = concentration of the light isotope
    """
    from esbmtk import Flux, GasReservoir, Species

    r = R.species.r  # reference ratio
    if isinstance(R, Species | GasReservoir):
        d = np.where(R.l > 0, 1e3 * ((R.c - R.l) / R.l - r) / r, 0)
    elif isinstance(R, Flux):
        d = np.where(R.l > 0, 1e3 * ((R.m - R.l) / R.l - r) / r, 0)
    else:
        raise ValueError(
            f"{R.full_name} must be of type Flux or Species, not {type(R)}"
        )

    return d


def get_delta_from_concentration(c, li, r):
    """Calculate the delta from the mass of light and heavy isotope.

    :param c: total mass/concentration
    :param l: light isotope mass/concentration
    :param r: reference ratio

    """
    h = c - li
    d = 1000 * (h / li - r) / r
    return d


def get_imass(m: float, d: float, r: float) -> [float, float]:
    """Calculate the isotope masses from bulk mass and delta value.

    Arguments are m = mass, d= delta value, r = abundance ratio
    species

    """
    li: float = (1000.0 * m) / ((d + 1000.0) * r + 1000.0)
    h: float = m - li
    return [li, h]


# @njit(fastmath=True)
def get_delta(li: NDArrayFloat, h: NDArrayFloat, r: float) -> NDArrayFloat:
    """Calculate the delta from the mass of light and heavy isotope.

    :param l: light isotope mass/concentration
    :param h: heavy isotope mass/concentration
    :param r: reference ratio

    :return : delta

    """
    return 1000 * (h / li - r) / r


# @njit(fastmath=True)
def get_new_ratio_from_alpha(
    ref_mass: float,  # reference mass
    ref_l: float,  # reference light istope
    a: float,  # fractionation factor
) -> [float, float]:
    """Calculate the effect of the istope fractionation factor alpha.

    For the ratio between the mass of the light isotope devided by the total mass

    Note that alpha needs to be given as fractional value, i.e., 1.07 rather
    than 70 (i.e., (alpha-1) * 1000
    """
    new_ratio = -ref_l / (a * ref_l - a * ref_mass - ref_l) if ref_mass > 0.0 else 0.0

    return new_ratio


def register_user_function(M: Model, lib_name: str, func_name: str | list) -> None:
    """Register user supplied library and function with the model.

    Parameters
    ----------
    M : Model
        Model handle
    lib_name : str
        name of python file that contains the function
    func_name : str | list
        Name of one or more user supplied function(s)

    """
    import pathlib as pl
    import sys

    if isinstance(func_name, str):
        func_name = [func_name]

    cwd: pl.Path = pl.Path.cwd()  # get the current working
    if cwd not in sys.path:
        sys.path.append(cwd)

    for f in func_name:
        M.luf[f] = [lib_name, cwd]


def create_reservoirs():
    """Raise Error on use."""
    raise NotImplementedError("create_reservoirs is no longer supported")
