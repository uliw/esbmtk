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
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import logging
import typing as tp
from esbmtk import Q_
import functools

if tp.TYPE_CHECKING:
    from esbmtk import Flux, Model, Connection, Connect

np.set_printoptions(precision=4)
# declare numpy types
NDArrayFloat = npt.NDArray[np.float64]


class ScaleError(Exception):
    def __init__(self, message):
        message = f"\n\n{message}\n"
        super().__init__(message)


def rmtree(f) -> None:
    """Delete file, of file is directorym delete all files in

    :param f: pathlib path object

    """
    if f.is_file():
        f.unlink()
    else:
        for child in f.iterdir():
            rmtree(child)
        f.rmdir()


def phc(c: float) -> float:
    # Calculate concentration as pH. c can be a number or numpy array
    import numpy as np

    pH: float = -np.log10(c)
    return pH


def debug(func):
    """Print the function signature and return value"""

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


def get_reservoir_reference(k, v, M):
    """parse dict key and value to determine whether we have a
    ReservoirGroup or just a (Gas) Reservoir and return the
    species

    :param v: e.g., "OM_remineralization
    :param k: e.g., "F_M.CO2_At" or "F_M.A_sb.PO4"

    :return reservoir: Reservoir/ReservoirGroup
    :return species: Species
    """

    key_list = k[2:].split(".")  # get model, reservoir & species name
    if len(key_list) == 3:  # ReservoirGroup
        model_name, reservoir_name, species_name = key_list
        reservoir = getattr(M, reservoir_name)
        if k[0:2] == "R_":
            species = getattr(reservoir.mo, species_name)
        elif k[0:2] == "F_":
            species = getattr(reservoir, species_name).species

    elif len(key_list) == 2:  # (Gas)Reservoir
        model_name, reservoir_name = key_list
        reservoir = getattr(M, reservoir_name)
        species = reservoir.species

    else:
        raise ValueError("kl should look like this F_M.CO2_At")

    return reservoir, species


def register_new_flux(r, sp, v, k, sink) -> list:
    """Create new flux and register with parent object"""
    from .esbmtk import Reservoir, Flux
    from .extended_classes import GasReservoir

    ro = []  # return objects
    if isinstance(r, (GasReservoir, Reservoir)):
        r.source = sink
        fn = f"{r.name}_2_{sink.name}_{v}_F"
        reg = r
    else:
        fn = f"{r.name}_{v}_F"
        reg = getattr(r, sp.name)
        reg.source = r

    f = Flux(
        name=fn,
        species=sp,
        rate=0,
        register=reg,
    )
    ro.append(f)
    reg.lof.append(f)  # register flux
    reg.ctype = "ignore"
    if reg.isotopes:
        f = Flux(
            name=f"{fn}_l",
            species=sp,
            rate=0,
            register=reg,
        )
        ro.append(f)
        # regular reservoirs will test if _l fluxes are needed or not.
        # if "exchange" in f.name:
        #     sink.lof.append(f)
    return ro


def register_new_reservoir(r, sp, v):
    """Register a new reservoir"""
    from .esbmtk import Reservoir

    rt = Reservoir(  # create new reservoir
        name=sp.name,
        species=sp,
        concentration=f"{v} mol/kg",
        register=r,
        volume=r.volume,
        rtype="computed",
    )
    r.lor.append(rt)
    return [rt]


def register_return_values(ec, sink) -> None:
    """Check the return values of external function instances,
    and create the necessary reservoirs or fluxes.

    These fluxes are not associated with a connection Object
    so we register the source/sink relationship with the
    reservoir they belong to.

    This fails for GasReservoirs since they can have a 1:many
    relatioship. The below is a terrible hack, it would be
    better to express this with several connection
    objects, rather than overloading the source attribute of the
    GasReservoir class.
    """
    from .esbmtk import Reservoir

    for line in ec.return_values:
        if isinstance(line, dict):
            k = next(iter(line))  # get first key
            v = line[k]
            r, sp = get_reservoir_reference(k, v, sink.mo)
            if k[:2] == "F_":  # check if flux
                o: list = register_new_flux(r, sp, v, k, sink)
            elif k[:2] == "R_":  # Create new reservoir
                o: list = register_new_reservoir(r, sp, v)
            else:
                raise ValueError(f"{k[0:2]} is not defined")

        elif isinstance(line, Reservoir):
            v.ef_results = True
            o = [v]
        # add to list of returned Objects
        ec.lro.append(o[0])
        if len(o) > 1:
            ec.lro.append(o[1])


def summarize_results(M: Model) -> dict():
    """Summarize all model results at t_max into a hirarchical
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
    """test if all elements of fl occur in s. Return True if yes,
    otherwise False

    """
    return all(f in s for f in fl)


def insert_into_namespace(name, value, name_space=globals()):
    name_space[name] = value


def add_to(l, e):
    """
    add element e to list l, but check if the entry already exist. If so, throw
    exception. Otherwise add
    """

    if e not in l:  # if not present, append element
        l.append(e)


def get_plot_layout(obj):
    """Simple function which selects a row, column layout based on the number of
    objects to display.  The expected argument is a reservoir object which
    contains the list of fluxes in the reservoir

    """

    noo = 1 + sum(f.plot == "yes" for f in obj.lof)
    for _ in obj.ldf:
        noo += 1

    # noo = len(obj.lof) + 1  # number of objects in this reservoir
    logging.debug(f"{noo} subplots for {obj.n} required")

    size, geo = plot_geometry(noo)

    return size, geo


def plot_geometry(noo: int) -> tuple():
    """Define plot geometry based on number of objects to plot"""

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
    elif 9 < noo < 13:
        geo = [4, 3]  # two rows, three columns
        size = [15, 12]  # size in inches
    elif 12 < noo < 16:
        geo = [5, 3]  # two rows, three columns
        size = [15, 15]  # size in inches
    else:
        m = (
            "plot geometry for more than 15 fluxes is not yet defined"
            "Consider calling flux.plot individually on each flux in the reservoir"
        )
        raise ValueError(m)

    return size, geo


def list_fluxes(self, name, i) -> None:
    """
    Echo all fluxes in the reservoir to the screen
    """
    for f in self.lof:  # show the processes
        direction = self.lio[f.n]
        if direction == -1:
            t1 = "From:"
            t2 = "Outflux from"
        else:
            t1 = "To  :"
            t2 = "Influx to"

        for p in f.lop:
            p.describe()

    for f in self.lof:
        f.describe(i)  # print out the flux data


def show_data(self, **kwargs) -> None:
    """Print the 3 lines of the data starting with index

    Optional arguments:

    index :int = 0 starting index
    indent :int = 0 indentation
    """

    off: str = "  "

    index = 0 if "index" not in kwargs else kwargs["index"]
    ind: str = kwargs["indent"] * " " if "indent" in kwargs else ""


def set_y_limits(ax: plt.Axes, obj: any) -> None:
    """Prevent the display or arbitrarily small differences"""

    bottom, top = ax.get_ylim()
    if (top - bottom) < obj.display_precision:
        top = bottom + obj.display_precision
        bottom -= obj.display_precision
        ax.set_ylim(bottom, top)


def is_name_in_list(n: str, l: list) -> bool:
    """Test if an object name is part of the object list"""

    return any(e.full_name == n for e in l)


def get_object_from_list(name: str, l: list) -> any:
    """Match a name to a list of objects. Return the object"""

    match: bool = False
    for o in l:
        if o.full_name == name:
            r = o
            match = True

    if match:
        return r
    else:
        raise ValueError(f"Object = {o.full_name} has no matching flux {name}")


def sort_by_type(l: list, t: list, m: str) -> list:
    """divide a list by type into new lists. This function will return a
    list and it is up to the calling code to unpack the list

    l is list with various object types
    t is a list which contains the object types used for sorting
    m is a string for the error function
    """

    # from numbers import Number

    lc = l.copy()
    rl = []

    for ot in t:  # loop over object types
        a = []
        for e in l:  # loop over list elements
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
    """Test if the key is a global reservoir handle
    or exists in the model namespace

    :param res: list of strings, or reservoir handles
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


def split_key(k: str, M: any) -> tp.Union[any, any, str]:
    """split the string k with letters _to_, and test if optional
    id string is present

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
    """Create a dictionary from a list and value, or from
    two lists

    """
    d = {}

    if isinstance(values, list):
        if len(values) == len(keys):
            d: dict = dict(zip(keys, values))
        else:
            raise ValueError("key and value list must be of equal length")
    else:
        values: list = [values] * len(keys)
        d: dict = dict(zip(keys, values))

    return d


def get_typed_list(data: list) -> list:
    tl = list()
    for x in data:
        tl.append(x)
    return tl


def create_reservoirs(bn: dict, ic: dict, M: any) -> dict:
    """boxes are defined by area and depth interval here we use an ordered
    dictionary to define the box geometries. The next column is temperature
    in deg C, followed by pressure in bar
    the geometry is [upper depth datum, lower depth datum, area percentage]

    bn = dictionary with box parameters
    bn: dict = {  # name: [[geometry], T, P]
                 "sb": {"g": [0, 200, 0.9], "T": 20, "P": 5},
                 "ib": {"g": [200, 1200, 1], "T": 10, "P": 100},
                }

    ic = dictionary with species default values. This is used to et up
         initial conditions. Here we use shortcut and use the same conditions
         in each box. If you need box specific initial conditions
         use the output of build_concentration_dicts as starting point

    ic: dict = { # species: concentration, Isotopes, delta, f_only
                   PO4: [Q_("2.1 * umol/liter"), False, 0, False],
                   DIC: [Q_("2.1 mmol/liter"), False, 0, False],
                   ALK: [Q_("2.43 mmol/liter"), False, 0, False],
               }

    M: Model object handle
    """

    from esbmtk import ReservoirGroup, build_concentration_dicts
    from esbmtk import SourceGroup, SinkGroup

    # parse for sources and sinks, create these and remove them from the list

    # loop over reservoir names
    for k, v in bn.items():
        # test key format
        if M.name in k:
            k = k.split(".")[1]

        if "ty" in v:  # type is given
            if v["ty"] == "Source":
                if "delta" in v:
                    SourceGroup(name=k, species=v["sp"], delta=v["delta"], register=M)
                else:
                    SourceGroup(name=k, species=v["sp"], register=M)
            elif v["ty"] == "Sink":
                SinkGroup(name=k, species=v["sp"], register=M)
            else:
                raise ValueError("'ty' must be either Source or Sink")

        else:  # create reservoirs
            icd: dict = build_concentration_dicts(ic, k)
            rg = ReservoirGroup(
                name=k,
                geometry=v["g"],
                concentration=icd[k][0],
                isotopes=icd[k][1],
                delta=icd[k][2],
                seawater_parameters={"temperature": v["T"], "pressure": v["P"]},
                register=M,
            )

    return icd


def build_concentration_dicts(cd: dict, bg: dict) -> dict:
    """Build a dict which can be used by create_reservoirs

    bg : dict where the box_names are dict keys.
    cd: dictionary with the following format:
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

    # box_names: list = bg.keys()
    for bn in box_names:  # loop over box names
        icd[bn] = [td1, td2, td3]

    return icd


def calc_volumes(bg: dict, M: any, h: any) -> list:
    """Calculate volume contained in a given depth interval
    bg is an ordered dictionary in the following format

    bg=  {
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
        l = v[2]

        v.append(h.volume(u, l) * a)

    return v


def get_longest_dict_entry(d: dict) -> int:
    """Get length of each item in the connection dict"""
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


def convert_to_lists(d: dict, l: int) -> dict:
    """expand mixed dict entries (i.e. list and single value) such
    that they are all lists of equal length

    """
    cd = d.copy()

    for k, v in cd.items():
        if not isinstance(v, list):
            p = [v for _ in range(l)]
            d[k] = p

    return d


def get_sub_key(d: dict, i: int) -> dict:
    """take a dict which has where the value is a list, and return the
    key with the n-th value of that list

    """

    return {k: v[i] for k, v in d.items()}


def expand_dict(d: dict, mt: str = "1:1") -> int:
    """Determine dict structure

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
    """Create connections from a dictionary. The dict can have the following keys
    following format:

    mt = mapping type. See below for explanation

    # na: names, tuple or str. If lists, all list elements share the same properties
    # sp: species list or species
    # ty: type, str
    # ra: rate, Quantity
    # sc: scale, Number
    # re: reference, optional
    # al: alpha, optional
    # de: delta, optional
    # bp: bypass, see scale_with_flux
    # si: signal
    # mx: True, optional defaults to False. If set, it will create forward
          and backward fluxes (i.e. mixing)

    There are 6 different cases how to specify connections

    Case 1 One connection, one set of parameters
           ct1 = {"sb2hb": {"ty": "scale", 'ra'....}}

    Case 2 One connection, one set of instructions, one subset with multiple parameters
           This will be expanded to create connections for each species
           ct2 = {"sb2hb": {"ty": "scale", "sp": ["a", "b"]}}

    Case 3 One connection complete set of multiple characters. Similar to case 2, but now
           all parameters are given explicitly
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

    from esbmtk import create_connection, expand_dict

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
    """called by create_bulk_connections in order to create a connection group
    It is assumed that all rates are in liter/year or mol per year.  This may
    not be what you want or need.

    :param n: a connection key.  if the mix flag is given interpreted as mixing
              a connection between sb and db and thus create connections in both
              directions
    :param p: a dictionary holding the connection properties
    :param M: the model handle
    """

    from esbmtk import ConnectionGroup, Q_

    # get the reservoir handles by splitting the key
    source, sink, cid = split_key(n, M)
    # create default connections parameters and replace with values in
    # the parameter dict if present.
    los = list(p["sp"]) if isinstance(p["sp"], list) else [p["sp"]]
    ctype = "None" if "ty" not in p else p["ty"]
    scale = 1 if "sc" not in p else p["sc"]
    rate = Q_("0 mol/a") if "ra" not in p else p["ra"]
    ref_flux = "None" if "re" not in p else p["re"]
    alpha = "None" if "al" not in p else p["al"]
    delta = "None" if "de" not in p else p["de"]
    cid = f"{cid}"
    bypass = "None" if "bp" not in p else p["bp"]
    species = "None" if "sp" not in p else p["sp"]
    signal = "None" if "si" not in p else p["si"]

    if isinstance(scale, Q_):
        scale = scale.to("l/a").magnitude

    # expand arguments
    ctype = make_dict(los, ctype)
    scale = make_dict(los, scale)  # get rate from dictionary
    rate = make_dict(los, rate)
    ref_flux = make_dict(los, ref_flux)
    alpha = make_dict(los, alpha)
    delta = make_dict(los, delta)
    bypass = make_dict(los, bypass)
    signal = make_dict(los, signal)

    # name of connectiongroup
    name = f"{M.name}.CG_{source.name}_to_{sink.name}"
    if f"{name}" in M.lmo:  # Test if CG exists
        # retriece CG object
        cg = getattr(M, name.split(".")[1])
        cg.add_connections(
            source=source,
            sink=sink,
            ctype=ctype,
            scale=scale,  # get rate from dictionary
            rate=rate,
            ref_flux=ref_flux,
            alpha=alpha,
            delta=delta,
            bypass=bypass,
            register=M,
            signal=signal,
            id=cid,  # get id from dictionary
        )
    else:  # Create New ConnectionGroup
        cg = ConnectionGroup(
            source=source,
            sink=sink,
            ctype=ctype,
            scale=scale,  # get rate from dictionary
            rate=rate,
            ref_flux=ref_flux,
            alpha=alpha,
            delta=delta,
            bypass=bypass,
            register=M,
            signal=signal,
            id=cid,  # get id from dictionary
        )


def get_name_only(o: any) -> any:
    """Test if item is an esbmtk type. If yes, extract the name"""

    from esbmtk import Flux, Reservoir, ReservoirGroup, Species

    return (
        o.full_name if isinstance(o, (Flux, Reservoir, ReservoirGroup, Species)) else o
    )


def get_simple_list(l: list) -> list:
    """return a list which only has the full name
    rather than all the object properties

    """

    return [get_name_only(e) for e in l]


def show_dict(d: dict, mt: str = "1:1") -> None:
    """show dict entries in an organized manner"""

    from esbmtk import expand_dict, get_simple_list, get_name_only

    ct = expand_dict(d, mt)
    for ck, cv in ct.items():
        for pk, pv in cv.items():
            x = get_simple_list(pv) if isinstance(pv, list) else get_name_only(pv)


def find_matching_fluxes(l: list, filter_by: str, exclude: str) -> list:
    """Loop over all reservoir in l, and extract the names of all fluxes
    which match the filter string. Return the list of names (not objects!)

    """

    lof: set = set()

    for r in l:
        for f in r.lof:
            if filter_by in f.full_name and exclude not in f.full_name:
                lof.add(f)

    return list(lof)


def reverse_key(key: str) -> str:
    """reverse a connection key e.g., sb2db@POM becomes db2sb@POM"""

    left = key.split("@")
    left = left[0]
    rs = left.split("_to_")
    r1 = rs[0]
    r2 = rs[1]

    return f"{r2}_to_{r1}"


def get_connection_keys(
    f_list: set,
    ref_id: str,
    target_id: str,
    inverse: bool,
    exclude: str,
) -> list[str]:
    """
    extract connection keys from set of flux names, replace ref_id with
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
    """Find all fluxes that contain the reference string, and create a new
    Connection instance that connects the flux matching ref_id, with a flux
    matching target_id.  The function will a tuple containig the new connection
    keys that can be used by the create bulk_connection() function.  The second
    return value is a list containing the reference fluxes.

    The optional inverse parameter, can be used where in cases where the flux
    direction needs to be reversed, i.e., the returned key will not read
    sb_to_dbPOM, but db_to_sb@POM

    :param M: Model or list
    :param **kwargs: keyword dictionary, known keys are ref_id, and raget_id,
        inverse

    :return f_list: List of fluxes that match ref_id
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
    """build a connection dictionary from a dict containing connection
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
    """Parse string and extract substring between square brackets"""

    s = s.split("[")
    if len(s) < 2:
        raise ValueError(f"Column header {s} must include units in square brackets")

    s = s[1]

    s = s.split("]")

    if len(s) < 2:
        raise ValueError(f"Column header {s} must include units in square brackets")

    return s[0]


def check_for_quantity(kw) -> Q_:
    """check if keyword is quantity or string an convert as necessary

    kw = str or Q_

    """

    from esbmtk import Q_

    if isinstance(kw, str):
        kw = Q_(kw)
    elif not isinstance(kw, Q_):
        raise ValueError("kw must be string or Quantity")

    return kw


def map_units(obj: any, v: any, *args) -> float:
    """parse v to see if it is a string. if yes, map to quantity.
    parse v to see if it is a quantity, if yes, map to model units
    and extract magnitude, assign mangitude to return value
    if not, assign value to return value

    :param obj: connection object 
    :param v: input string/number/quantity
    :args: list of model base units 
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

    if not isinstance(m, (int, float)):
        raise ValueError(f"m is {type(m)}, must be float, v={v}. Something is fishy")

    return m


def __find_flux__(reservoirs: list, full_name: str):
    """Helper function to find a Flux object based on its full_name in the reservoirs
    in the list of provided reservoirs.

    PRECONDITIONS: full_name must contain the full_name of the Flux

    Parameters:
        reservoirs: List containing all reservoirs
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


def __checktypes__(av: dict[any, any], pv: dict[any, any]) -> None:
    """this method will use the the dict key in the user provided
    key value data (pv) to look up the allowed data type for this key in av

    av = dictinory with the allowed input keys and their type
    pv = dictionary with the user provided key-value data
    """

    k: any
    v: any

    # loop over provided keywords
    for k, v in pv.items():
        # check av if provided value v is of correct type
        if av[k] != any and not isinstance(v, av[k]):
            raise TypeError(
                f"{type(v)} is the wrong type for '{k}', should be '{av[k]}'"
            )


def __checkkeys__(lrk: list, lkk: list, kwargs: dict) -> None:
    """check if the mandatory keys are present

    lrk = list of required keywords
    lkk = list of all known keywords
    kwargs = dictionary with key-value pairs

    """

    k: str
    v: any
    # test if the required keywords are given
    for k in lrk:  # loop over required keywords
        if isinstance(k, list):  # If keyword is a list
            s: int = 0  # loop over allowed substitutions
            for e in k:  # test how many matches are in this list
                if (
                    e in kwargs
                    and not isinstance(e, (np.ndarray, list))
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
    """
    test if the keys in lod exist in kwargs, otherwise add them with the default values
    from lod

    """

    new: dict = {}
    if lod:
        for k, v in lod.items():
            if k not in kwargs:
                new[k] = v

    # kwargs |= new
    kwargs.update(new)
    return kwargs


def data_summaries(M, species_names, box_names):
    from esbmtk import DataField

    pl = []
    for sp in species_names:
        data_list = []
        label_list = []
        for b in box_names:
            a = getattr(b, f"{sp.name}")
            if a.plot_transform_c == "None":
                data_list.append(a.c)
            else:
                data_list.append(a.plot_transform_c(a.c))
            label_list.append(a.full_name)

        df = DataField(
            name=f"{sp.name}_df",
            register=M.A_sb.O2,
            x1_data=M.time,
            y1_data=data_list,
            y1_label=label_list,
            y1_legend=f"{sp.element.name} [{sp.element.mass_unit}]",
            x1_as_time=True,
            title=f"{sp.name}",
        )
        pl.append(df)
        # check if species has isotope data
        data_list = []
        label_list = []
        for b in box_names:
            a = getattr(b, f"{sp.name}")
            if a.isotopes:
                data_list.append(a.d)
                label_list.append(a.full_name)

        if len(data_list) > 0:
            df = DataField(
                name=f"d_{sp.name}_df",
                register=M.A_sb.O2,
                x1_data=M.time,
                y1_data=data_list,
                y1_label=label_list,
                y1_legend=f"{sp.element.name} [{sp.element.d_scale}]",
                x1_as_time=True,
                title=f"d_{sp.name}",
            )
            pl.append(df)

    return pl