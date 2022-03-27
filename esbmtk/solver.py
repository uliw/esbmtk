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
from nptyping import *
from typing import *
from numpy import array, set_printoptions, arange, zeros, interp, mean
import time
from time import process_time
import numba
from numba.core import types
from numba import njit, prange
from numba.typed import List

import numpy as np

# import pandas as pd
# import mpmath

import time
import math


def get_l_mass(m: float, d: float, r: float) -> float:
    """
    Calculate the light isotope masses from bulk mass and delta value.
    Arguments are m = mass, d= delta value, r = abundance ratio
    """

    l: float = (1000.0 * m) / ((d + 1000.0) * r + 1000.0)
    return l


def get_imass(m: float, d: float, r: float) -> [float, float]:
    """
    Calculate the isotope masses from bulk mass and delta value.
    Arguments are m = mass, d= delta value, r = abundance ratio
    species

    """

    l: float = (1000.0 * m) / ((d + 1000.0) * r + 1000.0)
    h: float = m - l
    return [l, h]


# @njit()
def get_frac(m: float, l: float, a: float) -> [float, float]:
    """Calculate the effect of the istope fractionation factor alpha on
    the ratio between the light and heavy isotope.

    Note that alpha needs to be given as fractional value, i.e., 1.07 rather
    than 70 (i.e., (alpha-1) * 1000

    """

    if a > 1.1 or a < 0.9:
        raise ValueError("alpha needs to be given as fractional value not in permil")

    li: float = -l * m / (a * l - a * m - l)
    hi: float = m - li  # get the new heavy isotope value
    return li, hi


# @njit()
def get_flux_data(m: float, d: float, r: float) -> [NDArray, float]:
    """
    Calculate the isotope masses from bulk mass and delta value.
    Arguments are m = mass, d= delta value, r = abundance ratio
    species. Unlike get_mass, this function returns the full array

    """

    l: float = (1000.0 * m) / ((d + 1000.0) * r + 1000.0)
    h: float = m - l

    return np.array([m, l, h, d])


# @njit()
def get_delta(
    l: [NDArray, [Float64]], h: [NDArray, [Float64]], r: float
) -> [NDArray, [Float64]]:
    """Calculate the delta from the mass of light and heavy isotope
    Arguments are l and h which are the masses of the light and
    heavy isotopes respectively, r = abundance ratio of the
    respective element. Note that this equation can result in a
    siginificant loss of precision (on the order of 1E-13). I
    therefore round the results to numbers largers 1E12 (otherwise a
    delta of zero may no longer be zero)

    """

    return 1000 * (h / l - r) / r


def get_delta_i(l: float, h: float, r: float) -> float:
    """Calculate the delta from the mass of light and heavy isotope
    Arguments are l and h which are the masses of the light and
    heavy isotopes respectively, r = abundance ratio of the
    respective element. Note that this equation can result in a
    siginificant loss of precision (on the order of 1E-13). I
    therefore round the results to numbers largers 1E12 (otherwise a
    delta of zero may no longer be zero)

    """
    # d = 1000 * (h / l - r) / r
    if l > 0:
        d: float = 1e3 * (h / l - r) / r
    else:
        d = 0
    return d


def get_flux_delta(f) -> float:
    """Calculate the delta of flux f

    """

    m = f.fa[0]
    l = f.fa[1]
    h = m - l
    r = f.species.r
    d = 1e3 * (h / l - r) / r
    return d


def get_delta_h(h) -> float:
    """Calculate the delta of a flux or reserevoir from total mass
    and mass of light isotope.

    f = flux or reserevoir handle

    returns d a vector of delta values

    """

    r = h.species.r
    
    d = np.where(h.l > 0, 1e3 * ((h.m - h.l) / h.l - r) / r, 0)

    return d


def execute(
    new: [NDArray, Float64],
    time: [NDArray, Float64],
    lop: list,
    lor: list,
    lpc_f: list,
    lpc_r: list,
) -> None:
    """This is the original object oriented solver"""

    start: float = process_time()
    i = 1  # some processes refer to the previous time step
    for t in time[1:-1]:  # loop over the time vector except the first
        # we first need to calculate all fluxes
        # for r in lor:  # loop over all reservoirs
        #     for p in r.lop:  # loop over reservoir processes
        #         p(i)  # update fluxes

        for p in lop:  # loop over reservoir processes
            p(i)  # update fluxes

        for p in lpc_f:  # update all process based fluxes.
            p(i)

        for r in lor:  # loop over all reservoirs
            flux_list = r.lof

            m = l = 0
            for f in flux_list:  # do sum of fluxes in this reservoir
                direction = r.lio[f]
                m += f.fa[0] * direction  # current flux and direction
                l += f.fa[1] * direction  # current flux and direction

            r.m[i] = r.m[i - 1] + m * r.mo.dt
            r.l[i] = r.l[i - 1] + l * r.mo.dt
            r.c[i] = r.m[i] / r.v[i]

        # update reservoirs which do not depend on fluxes but on
        # functions
        for p in lpc_r:
            p(i)

        i = i + 1

    duration: float = process_time() - start
    print(f"\n Execution time {duration:.2e} cpu seconds\n")


def execute_e(model, new, lop, lor, lpc_f, lpc_r):
    """ """

    # numba.set_num_threads(2)

    # this has nothing todo with self.time below!

    dt: float = lor[0].mo.dt

    if model.first_start:
        start: float = process_time()
        (
            model.fn_vr,
            model.input_data,
            model.vr_data,
            model.vr_params,
            model.count,
        ) = build_vr_list(lpc_r)

        model.fn, model.da, model.pc = build_process_list(lor, lop)
        model.a, model.b, model.c, model.d, model.e = build_flux_lists_all(lor)
        """
        fn = typed list of functions (processes)
        da = process data
        pc = process parameters
        a = reservoir_list
        b = flux list
        c = direction list
        d = virtual reserevoir list
        e = r0 list ???
        """
        model.first_start = False

        duration: float = process_time() - start
        print(f"\n Setup time {duration} cpu seconds\n")

    print("Starting solver")

    wts = time.time()
    start: float = process_time()

    if model.count > 0:
        if len(model.lop) > 0:
            foo(
                model.fn_vr,
                model.input_data,
                model.vr_data,
                model.vr_params,
                model.fn,
                model.da,
                model.pc,
                model.a,  # reserevoir list
                model.b,  # flux list
                model.c,  # direction list
                model.d,  # r0 list
                model.e,
                model.time[:-1],
                model.dt,
            )
        else:
            foo_no_p(
                model.fn_vr,
                model.input_data,
                model.vr_data,
                model.vr_params,
                model.fn,
                model.a,  # reserevoir list
                model.b,  # flux list
                model.c,  # direction list
                model.d,  # r0 list
                model.e,
                model.time[:-1],
                model.dt,
            )
    else:
        foo_no_vr(
            model.fn,
            model.da,
            model.pc,
            model.a,
            model.b,
            model.c,
            model.d,
            model.e,
            model.time[:-1],
            model.dt,
        )

    duration: float = process_time() - start
    wcd = time.time() - wts
    print(f"\n Total solver time {duration} cpu seconds, wt = {wcd}\n")


# from numba import jit
@njit(parallel=False, fastmath=True, error_model="numpy")
def foo(fn_vr, input_data, vr_data, vr_params, fn, da, pc, a, b, c, d, e, maxt, dt):
    """
    fn = flux process list
    fn = typed list of functions (processes)
    da = process data
    pc = process parameters
    a = reservoir_list
    b = flux list
    c = direction list
    d = virtual reservoir list
    e = r0 list ???
    """

    i = 1
    for t in maxt:

        # loop over function (process) list and compute fluxes
        j = 0
        for f in enumerate(fn):
            fn[j](da[j], pc[j], i)
            j = j + 1

        # calculate the resulting reservoir concentrations
        # summarize_fluxes(a, b, c, d, e, i, dt)
        r_steps: int = len(b)
        # loop over reservoirs
        for j in range(r_steps):
            # for j, r in enumerate(b):  # this will catch the list for each reservoir

            # sum fluxes in each reservoir
            mass = li = 0.0
            f_steps = len(b[j])
            for u in range(f_steps):
                direction = c[j][u]
                mass += b[j][u][0] * direction  # mass
                li += b[j][u][1] * direction  # li

            # update masses
            a[j][0][i] = a[j][0][i - 1] + mass * dt  # mass
            a[j][1][i] = a[j][1][i - 1] + li * dt  # li
            a[j][2][i] = a[j][0][i] / a[j][3][i]  # c

        # # update reservoirs which do not depend on fluxes but on
        # # functions
        for j, f in enumerate(fn_vr):
            fn_vr[j](i, input_data[j], vr_data[j], vr_params[j])

        i = i + 1  # next time step


@njit(parallel=False, fastmath=True, error_model="numpy")
def foo_no_p(fn_vr, input_data, vr_data, vr_params, fn, a, b, c, d, e, maxt, dt):
    """
    fn = flux process list
    fn = typed list of functions (processes)
    da = process data
    pc = process parameters
    a = reservoir_list
    b = flux list
    c = direction list
    d = virtual reservoir list
    e = r0 list ???
    """

    i = 1
    for t in maxt:

        # calculate the resulting reservoir concentrations
        # summarize_fluxes(a, b, c, d, e, i, dt)
        r_steps: int = len(b)
        # loop over reservoirs
        for j in range(r_steps):
            # for j, r in enumerate(b):  # this will catch the list for each reservoir

            # sum fluxes in each reservoir
            mass = li = 0.0
            f_steps = len(b[j])
            for u in range(f_steps):
                direction = c[j][u]
                mass += b[j][u][0] * direction  # mass
                li += b[j][u][1] * direction  # li

            # update masses
            a[j][0][i] = a[j][0][i - 1] + mass * dt  # mass
            a[j][1][i] = a[j][1][i - 1] + li * dt  # li
            a[j][2][i] = a[j][0][i] / a[j][3][i]  # c

        # # update reservoirs which do not depend on fluxes but on
        # # functions
        for j, f in enumerate(fn_vr):
            fn_vr[j](i, input_data[j], vr_data[j], vr_params[j])

        i = i + 1  # next time step


@njit(parallel=False, fastmath=True, error_model="numpy")
def foo_no_vr(fn, da, pc, a, b, c, d, e, maxt, dt):
    """Same as foo but no virtual reservoirs present

    fn = flux process list
    fn = typed list of functions (processes)
    da = process data
    pc = process parameters
    a = reservoir_list
    b = flux list
    c = direction list
    d = virtual reserevoir list
    e = r0 list ???
    """

    i = 1
    for t in maxt:

        # loop over processes
        j = 0
        for f in enumerate(fn):
            fn[j](da[j], pc[j], i)
            j = j + 1

        # calculate the resulting reservoir concentrations
        # summarize_fluxes(a, b, c, d, e, i, dt)
        r_steps: int = len(b)
        # loop over reservoirs
        for j in range(r_steps):
            mass = li = 0.0
            f_steps = len(b[j])
            for u in range(f_steps):
                direction = c[j][u]
                mass += b[j][u][0] * direction  # mass
                li += b[j][u][1] * direction  # li

            # update masses
            a[j][0][i] = a[j][0][i - 1] + mass * dt  # mass
            a[j][1][i] = a[j][1][i - 1] + li * dt  # li
            a[j][2][i] = a[j][0][i] / a[j][3][i]  # c

        i = i + 1


@njit(parallel=False, fastmath=True, error_model="numpy")
def foo_no_vr_no_p(fn, a, b, c, d, e, maxt, dt):
    """Same as foo but no virtual reservoirs present

    fn = flux process list
    fn = typed list of functions (processes)
    da = process data
    pc = process parameters
    a = reservoir_list
    b = flux list
    c = direction list
    d = virtual reserevoir list
    e = r0 list ???
    """

    i = 1
    for t in maxt:

        # calculate the resulting reservoir concentrations
        # summarize_fluxes(a, b, c, d, e, i, dt)
        r_steps: int = len(b)
        # loop over reservoirs
        for j in range(r_steps):
            mass = li = 0.0
            f_steps = len(b[j])
            for u in range(f_steps):
                direction = c[j][u]
                mass += b[j][u][0] * direction  # mass
                li += b[j][u][1] * direction  # li

            # update masses
            a[j][0][i] = a[j][0][i - 1] + mass * dt  # mass
            a[j][1][i] = a[j][1][i - 1] + li * dt  # li
            a[j][2][i] = a[j][0][i] / a[j][3][i]  # c

        i = i + 1


def build_vr_list(lvr: list) -> tuple:
    """Build lists which contain all function references for
    virtual reservoirs as well aas their input values

    """

    fn = List()  # List() # list of functions
    input_data = List()  # reservoir data
    vr_data = List()  # flux data  flux.m flux.l, flux.h, flux.d
    vr_params = List()  # list of constants
    fn = numba.typed.List.empty_list(
        types.UniTuple(types.float64, 4)(
            types.int64,  # i
            types.ListType(types.float64[::1]),
            types.ListType(types.float64[::1]),
            types.ListType(types.float64),  # a3
        ).as_type()
    )

    count = 0
    for r in lvr:  # loop over reservoir processes

        func_name, in_d, vr_d, params = r.get_process_args()
        fn.append(func_name)
        input_data.append(in_d)
        vr_data.append(vr_d)
        vr_params.append(params)
        count = count + 1

    return fn, input_data, vr_data, vr_params, count


def build_flux_lists_all(lor, iso: bool = False) -> tuple:
    """flux_list :list [] contains all fluxes as
    [f.m, f.l, f.h, f.d], where each sublist relates to one reservoir

    i.e. per reservoir we have list [f1, f2, f3], where fi = [m, l]
    and m & l  = np.array()

    iso = False/True

    """

    r_list: list = List()
    v_list: list = List()
    r0_list: list = List()
    dir_list: list = List()
    rd_list: list = List()
    f_list: List = List()

    for r in lor:  # loop over all reservoirs
        if len(r.lof) > 0:
            rd_list = List([r.m, r.l, r.c, r.v])

            r_list.append(rd_list)
            v_list.append(float(r.volume))
            r0_list.append(float(r.sp.r))

            i = 0
            # add fluxes for each reservoir entry
            tf: list = List()  # temp list for flux data
            td: list = List()  # temp list for direction data

            # loop over all fluxes
            for f in r.lof:
                tf.append(f.fa)
                td.append(float(r.lodir[i]))
                i = i + 1

            f_list.append(tf)
            dir_list.append(td)

    return r_list, f_list, dir_list, v_list, r0_list


def build_process_list(lor: list, lop: list) -> tuple:
    from numba.typed import List
    import numba
    from numba.core import types

    fn = List()  # List() # list of functions
    da = List()  # data
    pc = List()  # list of constants

    print(f"Building Process List")

    tfn = numba.typed.List.empty_list(
        types.ListType(types.void)(  # return value
            types.ListType(types.float64[::1]),  # data array
            types.ListType(types.float64),  # parameter list
            types.int64,  # parameter 4
        ).as_type()
    )

    tda = List()  # temp list for data
    tpc = List()  # temp list for constants

    # note that types.List is differenfr from Types.ListType. Also
    # note that [::1]  declares C-style arrays see
    # https://numba.discourse.group/t/list-mistaken-as-list-when-creating-list-of-function-references/677/3

    for p in lop:  # loop over reservoir processes
        # print(f"working on {p.name}")
        func_name, data, proc_const = p.get_process_args()
        tfn.append(func_name)
        tda.append(data)
        tpc.append(proc_const)

    return tfn, tda, tpc
