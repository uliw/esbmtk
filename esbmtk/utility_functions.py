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
import matplotlib.pyplot as plt
import logging
import typing as tp
from collections import OrderedDict
from esbmtk import Q_

if tp.TYPE_CHECKING:
    from esbmtk import Flux, Model, Connection, Connect

np.set_printoptions(precision=4)


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
    print(f"\nList of fluxes in {self.n}:")

    for f in self.lof:  # show the processes
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

    # show the first 4 entries
    # for i in range(index, index + 3):
    #     print(f"{off}{ind}i = {i}, Mass = {self.m[i]:.2e}, li= {self.l[i]:.2f}")


def set_y_limits(ax: plt.Axes, obj: any) -> None:
    """Prevent the display or arbitrarily small differences"""
    lower: float
    upper: float

    bottom, top = ax.get_ylim()
    if (top - bottom) < obj.display_precision:
        top = bottom + obj.display_precision
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


def get_object_handle(res: list | string | Reservoir | ReservoirGroup, M: Model):
    """Test if the key is a global reservoir handle
    or exists in the model namespace

    :param res: list of strings, or reservoir handles
    :param M: Model handle
    """

    r_list: list = []

    if not isinstance(res, list):
        res = [res]
    for o in res:
        # print(f"goh0: looking up {o} of {type(o)}")
        if o in M.dmo:  # is object known in global namespace
            r_list.append(M.dmo[o])
            # print(f"goh1: found {o} in dmo")
        elif o in M.__dict__:  # or does it exist in Model namespace
            r_list.append(getattr(M, o))
            # print(f"goh2: found {o} in __dict__\n")
        else:
            print(f"{o} is not known for model {M.name}")
            # raise ValueError(f"{o} is not known for model {M.name}")

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
            print(f"len values ={len(values)}, len keys ={len(keys)}")
            print(f"values = {values}")
            for k in keys:
                print(f"key = {k}")
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

    ic: dict = { # species: concentration, Isotopes
                   PO4: [Q_("2.1 * umol/liter"), False],
                   DIC: [Q_("2.1 mmol/liter"), False],
                   ALK: [Q_("2.43 mmol/liter"), False],
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

    icd: dict = OrderedDict()
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

        # print(f"ck = {ck}")
        # print(f"cd = {cd}")

        rd: dict = {}  # temp dict
        nd: dict = {}  # temp dict
        case, length = get_longest_dict_entry(cd)

        # print(f"length = {length}")

        if isinstance(ck, tuple):
            # assume 1:1 mapping between tuple and connection parameters
            if mt == "1:1":

                # prep dictionaries
                if case == 0:
                    nd = cd
                    # print("case 0")
                elif case == 1:  # only parameters present. Expand for each tuple entry
                    length = len(ck)
                    nd = convert_to_lists(cd, length)
                    # print("case 1")
                elif case == 2:  # mixed list present, Expand list
                    # print(f"case 2, length = {length}")
                    nd = convert_to_lists(cd, length)
                    # print(nd)

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
        r |= rd

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

    # print(f"cc00: {n}",flush=True)
    # get the reservoir handles by splitting the key
    source, sink, cid = split_key(n, M)
    # print(f"cc0: type of key {type(n)}", flush=True)
    # print(f"cc1: key = {n}, source = {source}, sink = {sink}", flush=True)

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
        print(f"{ck}")

        for pk, pv in cv.items():
            x = get_simple_list(pv) if isinstance(pv, list) else get_name_only(pv)
            print(f"     {pk} : {x}")


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
        # print()
        # print(f"gck0: flux = {f.full_name}, key = {cnc}")
        # print(f"gck1: ref_id = {ref_id}, target_id = {target_id}")
        if inverse:
            cnc = reverse_key(cnc)

        cnc.replace(ref_id, target_id)
        # create new cnc string
        cnc = f"{cnc}_to_{target_id}"
        # fix id vs connection Name
        # parts = cnc.split('_to_')
        # cnc = f"{parts[0]}@{parts[1]}"
        # print(f"gck3: cnc = {cnc}")
        cnc_l.append(cnc)

    return cnc_l


def gen_dict_entries(M: Model | list, **kwargs) -> tuple(tuple, list):
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

    return {k: {"sc": v} | p for k, v in d.items()}


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


def map_units(v: any, *args) -> float:
    """parse v to see if it is a string. if yes, map to quantity.
    parse v to see if it is a quantity, if yes, map to model units
    and extract magnitude, assign mangitude to return value
    if not, assign value to return value

    v : a keyword value number/string/quantity
    args: one or more quantities (units) see the Model class (e.g., f_unit)

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
            message = f"{v} is none of {print(*args)}"
            raise ValueError(message)

    else:  # no quantity, so it should be a number
        m = v

    if not isinstance(m, (int, float)):
        raise ValueError(f"m is {type(m)}, must be float, v={v}. Something is fishy")

    return m


def add_carbonate_system_1(rgs: list):
    """Creates a new carbonate system virtual reservoir for each
    reservoir in rgs. Note that rgs must be a list of reservoir groups.

    Required keywords:
        rgs: list = []  of Reservoir Group objects

    These new virtual reservoirs are registered to their respective Reservoir
    as 'cs'.

    The respective data fields are available as rgs.r.cs.xxx where xxx stands
    for a given key key in the  vr_datafields dictionary (i.e., H, CA, etc.)

    """

    from esbmtk import (
        ExternalCode,
        carbonate_system_1_ode,
        Reservoir,
    )

    # get object handle even if it defined in model namespace
    # rgs = get_object_handle(rgs)

    model = rgs[0].mo
    species = model.Carbon.CO2

    for rg in rgs:
        if rgs[0].mo.use_ode:
            if hasattr(rg, "DIC") and hasattr(rg, "TA"):
                ec = ExternalCode(
                    name="cs",
                    species=species,
                    function=carbonate_system_1_ode,
                    ftype="cs1",
                    vr_datafields={
                        "H": rg.swc.hplus,
                        "CA": rg.swc.ca,  # 1
                        "HCO3": rg.swc.hco3,  # 2
                        "CO3": rg.swc.co3,  # 3
                        "CO2aq": rg.swc.co2,  # 4
                    },
                    function_input_data=list(),
                    function_params=list(),
                    register=rg,
                    return_values={"Hplus": rg.swc.hplus},
                )
                for n, v in ec.return_values.items():
                    rt = Reservoir(
                        name=n,
                        species=getattr(model, n),
                        concentration=f"{v} mol/kg",
                        register=rg,
                        volume=rg.volume,
                        rtype="computed",
                    )
                    rg.lor.append(rt)
        elif hasattr(rg, "DIC") and hasattr(rg, "TA"):
            ec = ExternalCode(
                name="cs",
                species=species,
                function=calc_carbonates_1,
                ftype="cs1",
                vr_datafields={
                    "H": rg.swc.hplus,  # 0
                    "CA": rg.swc.ca,  # 1
                    "HCO3": rg.swc.hco3,  # 2
                    "CO3": rg.swc.co3,  # 3
                    "CO2aq": rg.swc.co2,  # 4
                },
                function_input_data=list([rg.DIC.c, rg.TA.c]),
                function_params=list(
                    [
                        rg.swc.K1,  # 0
                        rg.swc.K2,  # 1
                        rg.swc.KW,  # 2
                        rg.swc.KB,  # 3
                        rg.swc.boron,  # 4
                        rg.swc.ca2,  # 5
                        rg.swc.Ksp,  # 6
                        rg.swc.hplus,  #
                    ]
                ),
                # return_values="H CA HCO3 CO3 CO2aq".split(" "),
                register=rg,
            )
            rg.has_cs1 = True

        else:
            raise AttributeError(f"{rg.full_name} must have a TA and DIC reservoir")


def add_carbonate_system_2(**kwargs) -> None:
    """Creates a new carbonate system virtual reservoir
    which will compute carbon species, saturation, compensation,
    and snowline depth, and compute the associated carbonate burial fluxes

    Required keywords:
        r_sb: list of ReservoirGroup objects in the surface layer
        r_db: list of ReservoirGroup objects in the deep layer
        carbonate_export_fluxes: list of flux objects which must match the
                                 list of ReservoirGroup objects.
        zsat_min = depth of the upper boundary of the deep box
        z0 = upper depth limit for carbonate burial calculations
             typically zsat_min

    Optional Parameters:

        zsat = initial saturation depth (m)
        zcc = initial carbon compensation depth (m)
        zsnow = initial snowline depth (m)
        zsat0 = characteristic depth (m)
        Ksp0 = solubility product of calcite at air-water interface (mol^2/kg^2)
        kc = heterogeneous rate constant/mass transfer coefficient for calcite dissolution (kg m^-2 yr^-1)
        Ca2 = calcium ion concentration (mol/kg)
        pc = characteristic pressure (atm)
        pg = seawater density multiplied by gravity due to acceleration (atm/m)
        I = dissolvable CaCO3 inventory
        co3 = CO3 concentration (mol/kg)
        Ksp = solubility product of calcite at in situ sea water conditions (mol^2/kg^2)

    """

    from esbmtk import (
        ExternalCode,
        carbonate_system_2_ode,
        Reservoir,
    )

    # list of known keywords
    lkk: dict = {
        "r_db": list,  # list of deep reservoirs
        "r_sb": list,  # list of corresponding surface reservoirs
        "carbonate_export_fluxes": list,
        "AD": float,
        "zsat": int,
        "zsat_min": int,
        "zcc": int,
        "zsnow": int,
        "zsat0": int,
        "Ksp0": float,
        "kc": float,
        "Ca2": float,
        "pc": (float, int),
        "pg": (float, int),
        "I_caco3": (float, int),
        "alpha": float,
        "zmax": (float, int),
        "z0": (float, int),
        "Ksp": (float, int),
        # "BM": (float, int),
    }
    # provide a list of absolutely required keywords
    lrk: list[str] = ["r_db", "r_sb", "carbonate_export_fluxes", "zsat_min", "z0"]

    # we need the reference to the Model in order to set some
    # default values.

    reservoir = kwargs["r_db"][0]
    model = reservoir.mo
    # list of default values if none provided
    lod: dict = {
        "r_sb": [],  # empty list
        "zsat": -3715,  # m
        "zcc": -4750,  # m
        "zsnow": -5000,  # m
        "zsat0": -5078,  # m
        "Ksp0": reservoir.swc.Ksp0,  # mol^2/kg^2
        "kc": 8.84 * 1000,  # m/yr converted to kg/(m^2 yr)
        "AD": model.hyp.area_dz(-200, -6000),
        "alpha": 0.6,  # 0.928771302395292, #0.75,
        "pg": 0.103,  # pressure in atm/m
        "pc": 511,  # characteristic pressure after Boudreau 2010
        "I_caco3": 529,  # dissolveable CaCO3 in mol/m^2
        "zmax": -6000,  # max model depth
        "Ksp": reservoir.swc.Ksp,  # mol^2/kg^2
        # "BM": 0.0,
    }

    # make sure all mandatory keywords are present
    __checkkeys__(lrk, lkk, kwargs)

    # add default values for keys which were not specified
    kwargs = __addmissingdefaults__(lod, kwargs)

    # test that all keyword values are of the correct type
    __checktypes__(lkk, kwargs)

    # establish some shared parameters
    # depths_table = np.arange(0, 6001, 1)
    depths: np.ndarray = np.arange(0, 6002, 1, dtype=float)
    r_db = kwargs["r_db"]
    r_sb = kwargs["r_sb"]
    Ksp0 = kwargs["Ksp0"]
    ca2 = r_db[0].swc.ca2
    pg = kwargs["pg"]
    pc = kwargs["pc"]
    z0 = kwargs["z0"]
    Ksp = kwargs["Ksp"]

    # test if corresponding surface reservoirs have been defined
    if len(r_sb) == 0:
        raise ValueError(
            "Please update your call to add_carbonate_system_2and add the list of of corresponding surface reservoirs"
        )

    # C saturation(z) after Boudreau 2010
    Csat_table: np.ndarray = (Ksp0 / ca2) * np.exp((depths * pg) / pc)
    area_table = model.hyp.get_lookup_table(0, -6002)  # area in m^2(z)
    area_dz_table = model.hyp.get_lookup_table_area_dz(0, -6002) * -1  # area'
    AD = model.hyp.area_dz(z0, -6000)  # Total Ocean Area

    for i, rg in enumerate(r_db):  # Setup the virtual reservoirs
        if rg.mo.register == "local":
            species = rg.mo.Carbon.CO2
        else:
            species = __builtins__["CO2"]

        if r_db[0].mo.use_ode:
            ec = ExternalCode(
                name="cs",
                species=species,
                function="None",
                ftype="cs2",
                r_s=r_sb[i],  # source (RG) of CaCO3 flux,
                r_d=r_db[i],  # sink (RG) of CaCO3 flux,
                # datafield hold the results of the VR_no_set function
                # provide a default values which will be use to initialize
                # the respective datafield/
                vr_datafields={
                    "H": rg.swc.hplus,  # 0 H+
                    "CA": rg.swc.ca,  # 1 carbonate alkalinity
                    "HCO3": rg.swc.hco3,  # 2 HCO3
                    "CO3": rg.swc.co3,  # 3 CO3
                    "CO2aq": rg.swc.co2,  # 4 CO2aq
                    "zsat": abs(kwargs["zsat"]),  # 5 zsat
                    "zcc": abs(kwargs["zcc"]),  # 6 zcc
                    "zsnow": abs(kwargs["zsnow"]),  # 7 zsnow
                    "depth_area_table": area_table,
                    "area_dz_table": area_dz_table,
                    "Csat_table": Csat_table,
                    "Fburial": 0.0,
                    "Fburial_l": 0.0,
                    # "BM": 0.0,
                },
                ref_flux=kwargs["carbonate_export_fluxes"],
                function_input_data=list(),
                function_params=list(
                    [
                        rg.swc.K1,  # 0
                        rg.swc.K2,  # 1
                        rg.swc.KW,  # 2
                        rg.swc.KB,  # 3
                        rg.swc.boron,  # 4
                        Ksp0,  # 5
                        float(kwargs["kc"]),  # 6
                        float(rg.volume.to("liter").magnitude),  # 7
                        float(AD),  # 8
                        float(abs(kwargs["zsat0"])),  # 9
                        float(rg.swc.ca2),  # 10
                        rg.mo.dt,  # 11
                        float(kwargs["I_caco3"]),  # 12
                        float(kwargs["alpha"]),  # 13
                        float(abs(kwargs["zsat_min"])),  # 14
                        float(abs(kwargs["zmax"])),  # 15
                        float(abs(kwargs["z0"])),  # 16
                        Ksp,  # 17
                        rg.swc.hplus,  # 18
                        float(abs(kwargs["zsnow"])),  # 19
                        # float(abs(kwargs["BM"])),  # last known value for BM,
                        # 22325765737536.062,  # last known value for BM,
                    ]
                ),
                return_values={  # these must be known speces definitions
                    "Hplus": rg.swc.hplus,
                    "zsnow": float(abs(kwargs["zsnow"])),
                    # "BM": 22325765737536.062,
                },
                register=rg,
            )
            for n, v in ec.return_values.items():
                rt = Reservoir(
                    name=n,
                    species=getattr(model, n),
                    concentration=f"{v} mol/kg",
                    register=rg,
                    volume=rg.volume,
                    rtype="computed",
                )
                rg.lor.append(rt)

        else:
            ExternalCode(
                name="cs",
                species=species,
                function=calc_carbonates_2,
                ftype="cs2",
                # datafield hold the results of the VR_no_set function
                # provide a default values which will be use to initialize
                # the respective datafield/
                vr_datafields={
                    "H": rg.swc.hplus,  # 0 H+
                    "CA": rg.swc.ca,  # 1 carbonate alkalinity
                    "HCO3": rg.swc.hco3,  # 2 HCO3
                    "CO3": rg.swc.co3,  # 3 CO3
                    "CO2aq": rg.swc.co2,  # 4 CO2aq
                    "zsat": kwargs["zsat"],  # 5 zsat
                    "zcc": kwargs["zcc"],  # 6 zcc
                    "zsnow": kwargs["zsnow"],  # 7 zsnow
                    "Fburial": 0.0,  # 8 carbonate burial
                    "Fburial_l": 0.0,  # 9 carbonate burial 12C
                    "diss": 0.0,  # dissolution flux
                    "Bm": 0.0,
                },
                function_input_data=list(
                    [
                        rg.DIC.m,  # 0 DIC mass db
                        rg.DIC.l,  # 1 DIC light isotope mass db
                        rg.DIC.c,  # 2 DIC concentration db
                        rg.TA.m,  # 3 TA mass db
                        rg.TA.c,  # 4 TA concentration db
                        kwargs["carbonate_export_fluxes"][i].fa,  # 5
                        area_table,  # 6
                        area_dz_table,  # 7
                        Csat_table,  # 8
                        rg.DIC.v,  # 9 reservoir volume
                    ]
                ),
                function_params=list(
                    [
                        rg.swc.K1,  # 0
                        rg.swc.K2,  # 1
                        rg.swc.KW,  # 2
                        rg.swc.KB,  # 3
                        rg.swc.boron,  # 4
                        Ksp0,  # 5
                        float(kwargs["kc"]),  # 6
                        float(rg.volume.to("liter").magnitude),  # 7
                        float(AD),  # 8
                        float(abs(kwargs["zsat0"])),  # 9
                        float(rg.swc.ca2),  # 10
                        rg.mo.dt,  # 11
                        float(kwargs["I_caco3"]),  # 12
                        float(kwargs["alpha"]),  # 13
                        float(abs(kwargs["zsat_min"])),  # 14
                        float(abs(kwargs["zmax"])),  # 15
                        float(abs(kwargs["z0"])),  # 16
                        Ksp,  # 17
                        rg.swc.hplus,
                        float(abs(kwargs["zsnow"])),
                    ]
                ),
                register=rg,
            )
        rg.has_cs2 = True


def weathering_processes():

    pass


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
            # print(f"k={k}, v= {v}, av[k] = {av[k]}")
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
                    and not isinstance(e, (np.ndarray, np.float64, list))
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

    kwargs |= new
    return kwargs
