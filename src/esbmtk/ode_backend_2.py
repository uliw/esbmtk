"""esbmtk: A general purpose Earth Science box model toolkit.

Copyright (C), 2020 Ulrich G.  Wortmann

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

import typing as tp

import numpy as np
import numpy.typing as npt

NDArrayFloat = npt.NDArray[np.float64]

if tp.TYPE_CHECKING:
    from esbmtk import (
        ExternalFunction,
        Flux,
        Model,
        Species,
        Species2Species,
    )


def build_eqs_matrix(M: Model) -> tuple[NDArrayFloat, NDArrayFloat]:
    """Build the coefficient matrix CM and Flux vector f.

    So that we can solve dC_dt = C dot flux_values
    """
    from esbmtk import GasReservoir, Species

    fi = 0
    F_names = []
    for f in M.lof:  # get flux index positions
        F_names.append(f.full_name)
        f.idx = fi  # set starting index
        fi = fi + 1

    # get number of reservoirs
    num_res = 0
    # M.r_list = []
    for r in M.lor:
        # M.r_list.append(r.full_name)
        num_res = num_res + 1
        if r.isotopes:
            # M.r_list.append(f"{r.full_name}_i")
            num_res = num_res + 1

    F = np.ones(fi)  # initialize flux value vector
    CM = np.zeros((num_res, fi), dtype=float)  # Initialize Coefficient matrix:

    """We have n reservoirs, without isotopes, this corresponds to
    n rows in the coefficinet matrix. However, if reservoirs have
    isotopes, we need an additional row to calculate the respective
    isotope concentration. Thus we have two counters, cr pointing to
    the current reservoir, and ri pointing to the respective row index
    in the coefficient matrix"""
    cr = 0
    ri = 0
    while ri < num_res:
        r = M.lor[cr]  # get current reservoir
        r.idx = ri  # record index position
        cr = cr + 1

        if r.rtype == "computed":
            # computed reservoirs are currently not set up by a Species2Species
            # connection, so we need to add a flux expression manually.
            if r.model.debug:
                print(f"bem1: {r.full_name} type = {r.rtype}")
            CM[ri, r.lof[0].idx] = 1
            if r.isotopes:
                CM[ri + 1, r.lof[1].idx] = 1

        else:  # regular reservoirs may have multiple fluxes
            if isinstance(r, Species):
                density = (
                    r.parent.swc.density / 1000 if hasattr(r.parent, "swc") else 1.0
                )
                mass = r.volume.to(r.model.v_unit).magnitude * density
            elif isinstance(r, GasReservoir):
                mass = r.v[0]
            else:
                raise ValueError(f"no definition for {type(r)}")

            """ Regular reservoirs, can have isotopes, If they do we need to treat them
            like a new reservoir, so they require their own line in the coefficient
            matrix. """
            fi = 0
            while fi < len(r.lof):  # loop over reservoir fluxes
                f = r.lof[fi]
                if f in r.lif:
                    sign = 0  # we ignore those fluxes
                else:
                    sign = -1 / mass if f.parent.source == r else 1 / mass

                if r.isotopes:  # add line for isotope calculations
                    CM[ri, r.lof[fi].idx] = sign
                    CM[ri + 1, r.lof[fi + 1].idx] = sign  # 2
                    fi = fi + 2
                else:
                    CM[ri, r.lof[fi].idx] = sign
                    fi = fi + 1

        ri = ri + 1  # increase count by 1, or two if isotopes
        if r.isotopes:
            ri = ri + 1

    return CM[:ri, :], F, F_names


def write_equations_3(
    M: Model,
    R: list[float],
    icl: dict,
    cpl: list,
    ipl: list,
    eqs_fn: str,
) -> str:
    """Write file that contains the ode-equations for the Model.

    :param Model: Model handle
    :param R: tp.List of floats with the initial conditions for each
              reservoir
    :param icl: dict of reservoirs that have actual fluxes
    :param cpl: tp.List of reservoirs that have no fluxes but are
        computed based on other reservoirs
    :param ipl: tp.List of reservoir that do not change in concentration
    :param fn: str, filename

    Returns: equationfile name
    """
    # construct header and static code:
    # add optional import statements
    h1 = ""
    h2 = '''# @njit(fastmath=True)
def eqs(t, R, M, gpt, toc, area_table, area_dz_table, Csat_table, CM, F):
    """Calculate dCdt for each reservoir.

    t = time vector
    R = initial conditions for each reservoir
    M = model handle. This is currently required for signals
    gpt = tuple of global constants (is this still used?)
    toc = is a tuple of containing  constants for each external function
    area_table = lookuptable used by carbonate_system_2
    area_dz_table = lookuptable used by carbonate_system_2
    Csat_table = lookuptable used by carbonate_system_2
    CM = Coefficient Matrix
    F = flux vector

    Returns: dCdt as numpy array

    Reservoir name lookup: M.R_names[15]
    Flux name lookup: M.F_names[2]
    Matrix Coefficients: M.CM[reservoir index, flux index]
    Table of constants: M.toc[idx]
    """
    '''
    ind2 = 4 * " "  # indention
    ind3 = 8 * " "  # indention
    h1 = '"""ESBMTk equationsfile."""'
    hi = ""

    # ensure there are no duplicates
    M.lpc_i = set(M.lpc_i)
    M.lpc_f = set(M.lpc_f)

    if len(M.lpc_f) > 0:
        hi += "from esbmtk import "
        for f in M.lpc_f:
            hi += f"{f}, "
        hi = f"{hi[:-2]}\n"  # strip comma and space

    if len(M.lpc_i) > 0:  # test if functions imports are required
        hi += f"from esbmtk.bio_pump_functions{M.bio_pump_functions} import "
        for f in M.lpc_i:
            hi += f"{f}, "
        hi = f"{hi[:-2]}\n"  # strip comma and space

    if M.luf:  # test if user defined function imports are required
        for function, args in M.luf.items():
            source = args[0]
            hi += f"from {source} import {function}\n"

    header = f"{h1}\n\n{h2}" if hi == "" else f"{h1}\n{hi}\n\n{h2}"

    rel = ""  # list of return values

    # write file
    with open(eqs_fn, "w", encoding="utf-8") as eqs:
        eqs.write(header)
        sep = (
            "# ---------------- write computed reservoir equations -------- #\n"
            + "    # that do not depend on fluxes"
        )
        eqs.write(f"{sep}\n")

        """  Computed reservoirs and computed fluxes are both stored in  M.lpc_r
        At this point, we only print the reservoirs.
        TODO: This should really be cleaned up
        """
        for r in M.lpc_r:  # All virtual reservoirs need to be in this list
            if r.ftype == "std":
                rel = write_ef(eqs, r, icl, rel, ind2, ind3, M.gpt)

        # write all fluxes in M.lof
        flist = []
        sep = "    # ---------------- write all flux equations ------------------- #"
        eqs.write(f"\n{sep}\n")
        fi = 0
        while fi < len(M.lof):
            # for flux in M.lof:  # loop over fluxes
            """This loop will only write regular flux equations, with the sole
            exception of fluxes belonging to ExternalCode objects that need to be
            in a given sequence" All other fluxes must be on the M.lpr_r list
            below.
            """
            # functions can return more than one flux, but we only need to
            # call the function once, so we check against flist first
            flux = M.lof[fi]
            if flux not in flist:
                flist.append(flux)  # add to list of fluxes already computed
                # include computed fluxes that need to be in sequence
                if flux.ftype == "in_sequence":
                    rel = write_ef(eqs, r, icl, rel, ind2, ind3, M.gpt)
                elif flux.ftype == "None":  # get flux expressions
                    rhs, rhs_out, rhs_debug = get_flux(flux, M, R, icl)
                    # write regular equation
                    if rhs_debug[0] != "":  # add debug if need be
                        eqs.write(f"{ind2}{rhs_debug[0]}")
                    if rhs_out[0]:
                        fn = f"F[{flux.idx}]"
                        eqs.write(f"{ind2}{fn} = {rhs[0]}\n")

                    # write isotope equations
                    if rhs_debug[1] != "":  # add debug if need beg
                        eqs.write(f"{ind2}{rhs_debug[1]}")
                    if rhs_out[1]:  # add line for isotopes
                        fn = f"F[{flux.idx + 1}]"
                        eqs.write(f"{ind2}{fn} = {rhs[1]}\n")
                        fi = fi + 1

            fi = fi + 1

        sep = (
            "    # ---------------- write computed reservoir equations -------- #\n"
            + "# that do depend on fluxes"
        )
        eqs.write(f"\n{sep}\n")

        for r in M.lpc_r:  # All virtual reservoirs need to be in this list
            if r.ftype == "computed":  #
                rel = write_ef(eqs, r, icl, rel, ind2, ind3, M.gpt)

        sep = "    # ---------------- write input only reservoir equations -------- #"
        eqs.write(f"\n{sep}\n")
        for r in ipl:
            rname = r.full_name.replace(".", "_")
            eqs.write(f"{ind2}{rname} = 0.0")

        sep = "    # ---------------- calculate concentration change ------------ #"
        eqs.write(f"\n{sep}\n")
        eqs.write(f"{ind2}return CM.dot(F)\n")

    return eqs_fn.stem


# ------------------------ define processes ------------------------- #
def write_ef(
    eqs,
    ef: Species | ExternalFunction,
    icl: dict,
    rel: str,
    ind2: str,
    ind3: str,
    gpt: tuple,
) -> str:
    """Write external function call code.

    :param eqs: equation file handle
    :param ef: external_function handle
    :param icl: dict of reservoirs that have actual fluxes
    :param rel: string with reservoir names returned by setup_ode
    :param ind2: indent 2 times
    :param ind3: indent 3 times
    :param gpt: tuple with global paramaters

    :returns: rel: modied string of reservoir names
    """
    from esbmtk import Flux

    """ get list of return values. This is a bit of hack, since we imply
    anything that its not a flux, is a reservoir with no other inputs.
    Although, legacy fluxes do not come with a separate flux for isotopes,
    whereas newly created fluxes do.
    """
    rv = ind2
    for _i, o in enumerate(ef.lro):
        if isinstance(o, Flux):
            v = f"F[{o.idx}], "
        elif isinstance(o, str):
            if o != "None":
                raise ValueError(f"type(o) = {type(o)}, o = {o}")
        else:
            raise ValueError(f"type(o) = {type(o)}, o = {o}")

        rv += v

    rv = rv[:-2]  # strip the last comma
    a = ""

    # this can probably be simplified similar to the old parse_return_values()
    for d in ef.function_input_data:
        a += parse_esbmtk_input_data_types(d, ef, ind3, icl)

    if ef.function_params == "None":
        eqs.write(f"{rv} = {ef.fname}(\n{a}{ind2})\n\n")
    else:
        s = f"gpt[{ef.param_start}]"
        eqs.write(f"    # id: {ef.name}\n")
        eqs.write(f"{rv} = {ef.fname}(\n{a}{ind3}{s},\n{ind2})\n\n")

    rel += f"{ind3}{rv},\n"

    return rel


def parse_esbmtk_input_data_types(d: any, r: Species, ind: str, icl: dict) -> str:
    """Parse esbmtk data types.

    That are provided as arguments
    to external function objects, and convert them into a suitable string
    format that can be used in the ode equation file
    """
    from esbmtk import (
        Q_,
        Flux,
        GasReservoir,
        Reservoir,
        SeawaterConstants,
        Source,
        Species,
    )

    if isinstance(d, str):
        sr = getattr(r.register, d)
        a = f"{ind}{get_ic(sr, icl)},\n"
    elif isinstance(d, Species | GasReservoir | Source):
        a = f"{ind}({get_ic(d, icl, d.isotopes)}),\n"
    elif isinstance(d, Reservoir):
        a = f"{ind}{d.full_name},\n"
    elif isinstance(d, Flux):
        sr = f"F[{d.idx}]"
        a = f"{ind}{sr},\n"
    elif isinstance(d, SeawaterConstants):
        a = f"{ind}{d.full_name},\n"
    elif isinstance(d, (float | int | np.ndarray)):
        a = f"{ind}{d},\n"
    elif isinstance(d, dict):
        # get key and reservoiur handle
        sr = getattr(r.register, next(iter(d)))
        a = f"{ind}{get_ic(sr, icl)},\n"
    elif isinstance(d, Q_ | float):
        a = f"{ind}{d.magnitude},\n"
    elif isinstance(d, list):  # loop over list elements
        a = f"{ind}{d},\n"
        a = f"{ind}npa(["
        for e in d:
            a += f"{parse_esbmtk_input_data_types(e, r, '', icl)[0:-2]},"
        a += "]),\n"
    elif isinstance(d, str):
        a = d
    else:
        raise ValueError(
            f"\n r = {r.full_name}, d ={d}\n\n{d} is of type {type(d)}\n"
            f"I have no recpie for this type in parse_esbmtk_input_data_types\n"
        )

    return a


def get_flux(flux: Flux, M: Model, R: list[float], icl: dict) -> tuple[str, str, list]:
    """Create formula expressions that calculates the flux F.

    Return the equation expression as string

    :param flux: The flux object for which we create the equation
    :param M: The current model object
    :param R: The list of initial conditions for each reservoir
    :param icl: dict of reservoirs that have actual fluxes

    :returns: A tuple where the first string is the equation for the
              total flux, and the second string is the equation for
              the flux of the light isotope
    """
    ind2 = 8 * " "  # indentation
    ind3 = 12 * " "  # indentation
    rhs = ["", ""]
    rhs_debug = ["", ""]
    rhs_out = [False, False]
    c = flux.parent  # shorthand for the connection object
    cfn = flux.parent.full_name  # shorthand for the connection object name

    """flux parent would typically be a connection object. However for computed
    reservoirs, the flux is registed with a the computed reservoir.
    """
    from esbmtk import Species, Species2Species

    if isinstance(c, Species2Species):
        if c.ctype.casefold() == "regular" or c.ctype.casefold() == "fixed":
            rhs, rhs_out, rhs_debug = get_regular_flux_eq(
                flux, c, icl, ind2, ind3, M.CM, M.toc
            )
        elif c.ctype == "scale_with_concentration" or c.ctype == "scale_with_mass":
            rhs, rhs_out, rhs_debug = get_scale_with_concentration_eq(
                flux, c, cfn, icl, ind2, ind3, M.CM, M.toc
            )
        elif c.ctype == "scale_with_flux":
            rhs, rhs_out, rhs_debug = get_scale_with_flux_eq(
                flux, c, cfn, icl, ind2, ind3, M.CM, M.toc
            )
        elif c.ctype == "ignore" or c.ctype == "gasexchange" or c.ctype == "weathering":
            pass
        else:
            raise ValueError(
                f"Species2Species type {c.ctype} for {c.full_name} is not implemented"
            )
    elif isinstance(c, Species):
        rhs = f"{0}  # {c.full_name}"  # The flux will simply be zero
    else:
        raise ValueError(f"No definition for \n{c.full_name} type = {type(c)}\n")

    return rhs, rhs_out, rhs_debug


def get_ic(r: Species, icl: dict, isotopes=False) -> str:
    """Get initial condition in a reservoir.

    If the reservoir is icl,
    return index expression into R.c. If the reservoir is not in the
    index, return the Species concentration a t=0

    In both cases return these a string

    If isotopes == True, return the pointer to the light isotope
    concentration

    :param r: A reservoir handle
    :param icl: icl = dict[Species, list[int, int]] where reservoir
        indicates the reservoir handle, and the list contains the
        index into the reservoir data.  list[0] = concentration
        list[1] concentration of the light isotope.

    :raises ValueError: get_ic: can't find {r.full_name} in list of
        initial conditions

    :returns: the string s which is the full_name of the reservoir
              concentration or isotope concentration
    """
    from esbmtk import Sink, Source

    if r in icl:
        s1 = f"R[{icl[r][0]}]"
        if isotopes:
            s1 += f", R[{icl[r][1]}]"

    elif isinstance(r, Source | Sink):
        # FIXME: this should really point to a variable, maybe add to
        # gpt?
        s1 = f"{r.c}"
        if isotopes:
            # FIXME: this should really point to a variable
            s1 += f", {r.l}"
    else:
        raise ValueError(
            f"get_ic: can't find {r.full_name} in list of initial conditions"
        )

    return s1


def get_regular_flux_eq(
    flux: Flux,  # flux instance
    c: Species2Species,  # connection instance
    icl: dict,  # list of initial conditions
    ind2,  # indentation
    ind3,  # indentation
    CM: NDArrayFloat,  # coefficient matrix
    toc: tuple,  # tuple of constants
) -> tuple[str, str, list]:
    """Create a string containing the equation for a regular connection.

    Regular = fixed rate

    :param flux: flux instance
    :param c: connection object
    :param icl: dict of reservoirs that have actual fluxes
    :param ind2: indent 2 times
    :param ind3: indent 3 times

    :returns: two strings, where the first describes the equation for
              the total flux, and the second describes the rate for
              the light isotope
    """
    """FIXME: write 3 blocks
    1) debug information: connection name, constants, RHS
    2) constants
    3) RHS

    ditto for isotopes
    """
    from esbmtk import Source

    rhs, rhs_l = ["", ""]
    debug_rhs = ["", ""]
    rhs_out = [False, False]
    if flux.serves_as_input or c.signal != "None":
        # Needs full expression with all constants, so that we can
        # reference the expression elsewhere.
        rhs = toc[c.r_index]  # constant flux -> rhs = c
        if flux.isotopes:
            rhs_l, rhs_out[1], debug_rhs[1] = isotopes_regular_flux(
                rhs, c, icl, ind3, ind2, CM, toc, flux
            )
        rhs, rhs_l = check_signal_2(rhs, rhs_l, c)  # check if we hav to add a signal
        rhs_out[0] = True
    else:
        # get constants and place them on matrix
        # FIXME: moving this to the matrix, creates problems in the
        # isotope module. add second one?
        # CM[:, flux.idx] = CM[:, flux.idx] * toc[c.r_index]
        rhs = toc[c.r_index]
        rhs_out[0] = True
        if flux.isotopes:
            rhs_l, rhs_out[1], debug_rhs[1] = isotopes_regular_flux(
                rhs, c, icl, ind3, ind2, CM, toc, flux
            )

    if c.mo.debug_equations_file:  # and output:
        if isinstance(c.source, Source):
            m1 = "constant flux from a source\n"
            m2 = "\n"
        else:
            m1 = "    constants =  CM[c.source.idx, flux.idx] * toc[c.r_index]\n"
            m2 = "    constants = CM[{c.source.idx}:, {flux.idx}] * toc[{c.r_index}]\n"

        debug_rhs[0] = (
            f'"""\n'
            f"    {c.ctype}: {c.name}, id={c.id}\n"
            f"{m1}"
            f"{m2}"
            f"    rhs   = None\n"
            f'    """\n'
        )

    return [rhs, rhs_l], rhs_out, debug_rhs


def get_scale_with_concentration_eq(
    flux: Flux,  # flux instance
    c: Species2Species,  # connection instance
    cfn: str,  # full name of the connection instance
    icl: dict,  # list of initial conditions
    ind2: str,  # whitespace
    ind3: str,  # whitespace
    CM: NDArrayFloat,  # coefficient matrix
    toc: tuple,  # tuple of constants
) -> tuple[str, str, list]:
    """Create equation string for flux.

    The flux will scale with the
    concentration in the upstream reservoir

    Example: M1_ConnGrp_D_b_to_L_b_TA_thc__F =
    M1.ConnGrp_D_b_to_L_b.TA_thc.scale * R[5]

    :param flux: Flux object
    :param c: connection instance
    :param cfn: full name of the connection instance
    :param icl: dict[Species, list[int, int]] where reservoir
        indicates the reservoir handle, and the list contains the
        index into the reservoir data.  list[0] = concentration
        list[1] concentration of the light isotope.

    :returns: two strings with the respective equations for the change
              in the total reservoir concentration and the
              concentration of the light isotope
    """
    rhs, rhs_l = ["", ""]
    debug_rhs = ["", ""]
    rhs_out = [True, False]  # we always have a right hand side
    s_c = get_ic(c.ref_reservoirs, icl)  # get index to concentration`
    rhs = f"{s_c}"  # source concentration

    if flux.serves_as_input or c.signal != "None":
        # place all constants into the rhs so we can re-use the expression
        rhs = f"toc[{c.s_index}] * {s_c}"  # (scale * c)
        if flux.isotopes:
            rhs_l, rhs_out[1], debug_rhs[1] = isotopes_scale_concentration(
                rhs, c, icl, ind3, ind2, CM, toc, flux
            )
        rhs, rhs_l = check_signal_2(rhs, rhs_l, c)
    else:  # place all constants on matrix
        # FIXME: moving this to the matrix, creates problems in the
        # isotope module. Create a second version?
        # CM[:, flux.idx] = CM[:, flux.idx] * toc[c.s_index]
        # rhs = f"{s_c}"  # flux changes with concentration in source
        rhs = f"toc[{c.s_index}] * {s_c}"  # (scale * c)
        if flux.isotopes:
            rhs_l, rhs_out[1], debug_rhs[1] = isotopes_scale_concentration(
                rhs, c, icl, ind3, ind2, CM, toc, flux
            )

    if c.mo.debug_equations_file:
        debug_rhs[0] = (
            f'\n    """\n'
            f"    {c.ctype}: {c.name}, id = {c.id}\n"
            f"    {flux.full_name} = CM[{c.source.idx}, {flux.idx}] * toc[{c.s_index}] * {c.ref_reservoirs.full_name}.\n"
            f"    {flux.full_name} = {CM[c.source.idx, flux.idx]:.2e} * {toc[c.s_index]:.2e} * {s_c}\n"
            f'    """\n'
        )

    return [rhs, rhs_l], rhs_out, debug_rhs


def get_scale_with_flux_eq(
    flux: Flux,  # flux instance
    c: Species2Species,  # connection instance
    cfn: str,  # full name of the connection instance
    icl: dict,  # list of initial conditions
    ind2: str,  # indentation
    ind3: str,  # indentation
    CM: NDArrayFloat,  # coefficient matrix
    toc: tuple,  # tuple of constants
) -> tuple(str, str):
    """Equation defining a flux.

    The flux will scale relative to the magnitude of another
    flux. If isotopes are used, use the isotope ratio of the upstream reservoir.

    :param flux: Flux object
    :param c: connection instance
    :param cfn: full name of the connection instance
    :param icl: dict[Species, list[int, int]] where reservoir
        indicates the reservoir handle, and the list contains the
        index into the reservoir data.  list[0] = concentration
        list[1] concentration of the light isotope.

    :returns: two strings with the respective equations for the change
              in the total reservoir concentration and the
              concentration of the light isotope
    """
    from esbmtk import Sink, Source

    rhs, rhs_l = ["", ""]
    debug_rhs = ["", ""]
    rhs_out = [True, False]

    if flux.serves_as_input or c.signal != "None":
        """The flux for the light isotope will computed as follows:
        We will use the mass of the flux for scaling, but we
        isotope ratio of the source.
        """
        rhs = f"toc[{c.s_index}] * F[{c.ref_flux.idx}]"
        if c.source.isotopes and c.sink.isotopes:
            if flux.isotopes:
                rhs_l, rhs_out[1], debug_rhs[1] = isotopes_scale_flux(
                    rhs, c, icl, ind3, ind2, CM, toc, flux
                )
        else:
            rhs, rhs_l = check_signal_2(rhs, rhs_l, c)
    else:
        # FIXME: moving this to the matrix, creates problems in the
        # isotope module
        # CM[:, flux.idx] = CM[:, flux.idx] * toc[c.s_index]
        rhs = f"F[{c.ref_flux.idx}] * toc[{c.s_index}]"
        if flux.isotopes:
            rhs_l, rhs_out[1], debug_rhs[1] = isotopes_scale_flux(
                rhs, c, icl, ind3, ind2, CM, toc, flux
            )

    if c.mo.debug_equations_file:
        if isinstance(c.source, Source | Sink):
            debug_rhs[0] = (
                f'\n    """\n'
                f"    {c.ctype}: {c.name}, id = {c.id}\n"
                f"    {flux.full_name} = toc[{c.s_index}] * {c.ref_flux.full_name}\n"
                f"    {flux.full_name} = {toc[c.s_index]:.2e} * F[{c.ref_flux.idx}]\n"
                f'    """\n'
            )
        else:
            debug_rhs[0] = (
                f'\n    """\n'
                f"    {c.ctype}: {c.name}, id = {c.id}\n"
                f"    {flux.full_name} = CM[{c.source.idx}, {flux.idx}] * toc[{c.s_index}] * {c.ref_flux.full_name}\n"
                f"    {flux.full_name} = {CM[c.source.idx, flux.idx]:.2e} * {toc[c.s_index]:.2e} * F[{c.ref_flux.idx}]\n"
                f'    """\n'
            )

    return [rhs, rhs_l], rhs_out, debug_rhs


def isotopes_regular_flux(
    f_m: str,  # flux expression
    c: Species2Species,  # connection object
    icl: dict,  # initial conditions
    ind3: str,
    ind2: str,
    CM: NDArrayFloat,  # matrix
    toc: NDArrayFloat,  # table of constants
    flux: Flux,  #
) -> str:
    """Test if the connection involves any isotope effects.

    :param f_m: string with the flux name
    :param c: connection object
    :param icl: dict of reservoirs that have actual fluxes
    :param ind2: indent 2 times
    :param ind3: indent 3 times

    :returns equation string:
    """
    """Calculate the flux of the light isotope 
    Where f_m = flux mass/t, s_c = source concentration
    s_l = source light isotope, a = fractionation factor alpha,
    r = isotopic reference ratio
         
    If delta is given: fl = f_m * 1000 / (r * (d + 1000) + 1000)
    If epsilon is given: fl = f_m * s_l / (a * sc + sl - a * sl)
    If neither: fl = f_m * s_l

    If flux mass equals source concentration, this is a scale
    with concentration connection, and f_l = s_l * scale where scale is
    part of the constants on the equations matrix. Otherwise, the flux
    isotope ratio is similar to the source isotope ratio:
    fl = fm * sl/sc * scale. Note that scale is dropped in the below equations
    since it is moved to the equations matrix.

    FIXME: move constants in the matrix?
    """
    r: float = c.source.species.r  # isotope reference value
    s: str = get_ic(c.source, icl, True)  # R[0], R[1] reservoir concentrations
    s_c, s_l = s.replace(" ", "").split(",")  # total c, light isotope c
    debug_str = ""
    rhs_out = True

    # light isotope flux with no effects
    # CM[:, flux.idx + 1] = CM[:, flux.idx + 1] * toc[c.r_index]
    if c.delta != "None":
        equation_string = f"{f_m} * 1000 / ({r} * ({c.delta} + 1000) + 1000)"
        ds1 = "{f_m} * 1000 / (r * (c.delta + 1000) + 1000)"
    elif c.epsilon != "None":
        a = c.epsilon / 1000 + 1  # convert to alpha notation
        if f_m == "":  # if f_m is not provided.
            breakpoint()
            equation_string = f"{s_l} / ({a} * {s_c} + {s_l} - {a} * {s_l})"
            ds1 = (
                f"{c.source.full_name}.l"
                f" / (a * {c.source.full_name}.c + {c.source.full_name}.l"
                f" - a * {c.source.full_name}.l)"
            )
        else:
            equation_string = f"{f_m} * {s_l} / ({a} * {s_c} + {s_l} - {a} * {s_l})"
            ds1 = (
                f"{c.source.full_name}.l * {flux.full_name}"
                f" / (a * {c.source.full_name}.c + {c.source.full_name}.l"
                f" - a * {c.source.full_name}.l)"
            )
    else:
        equation_string = f"toc[{c.r_index}]"
        ds1 = f"{c.source.full_name}.l * {flux.full_name}"

    if c.mo.debug_equations_file:
        debug_str = (
            f'"""\n'
            f"    isotope equations for {c.full_name}:{c.ctype}\n"
            f"    constants = CM[c.source.idx:, flux.idx + 1] = CM[:, flux.idx + 1] * toc[c.r_index]\n"
            f"    constants = CM[c.source.idx:, {flux.idx + 1}] = CM[:, {flux.idx + 1}] * {toc[c.r_index]}\n"
            f"    rhs_l = {ds1}\n"
            f"    rhs_l = {equation_string}\n"
            f'    """\n'
        )

    return equation_string, rhs_out, debug_str


def isotopes_scale_concentration(
    f_m: str,  # flux expression
    c: Species2Species,  # connection object
    icl: dict,  # initial conditions
    ind3: str,
    ind2: str,
    CM: NDArrayFloat,  # matrix
    toc: NDArrayFloat,  # table of constants
    flux: Flux,  #
) -> str:
    """Test if the connection involves any isotope effects.

    :param f_m: string with the flux name
    :param c: connection object
    :param icl: dict of reservoirs that have actual fluxes
    :param ind2: indent 2 times
    :param ind3: indent 3 times

    :returns equation string:
    """
    """Calculate the flux of the light isotope (f_l).

    f_m = flux mass/t Note: this already has the scale factor!
    s_c = source concentration
    s_l = source concentration of light isotope
    r_index references the rate keyword in a given connection instance
    s_index references the scale keyword in a given connection instance
    FIXME: unify rate & scale?

    If delta is given: fl = f_m * 1000 / (r * (d + 1000) + 1000)
    If epsilon is given: fl = f_m * sl * / (e * sc + sl - a * sl)
    If neither: fl = f_m * s_l / s_c
    
    FIXME: move constants on the matrix?
    """
    r: float = c.source.species.r  # isotope reference value
    s: str = get_ic(c.source, icl, True)  # R[0], R[1] reservoir concentrations
    s_c, s_l = s.replace(" ", "").split(",")  # total c, light isotope c
    debug_str = ""
    rhs_out = True

    # light isotope flux with no effects
    # CM[:, flux.idx + 1] = CM[:, flux.idx + 1] * toc[c.s_index]
    if c.delta != "None":
        equation_string = f"{f_m} * 1000 / ({r} * ({c.delta} + 1000) + 1000)"
        ds1 = f"{c.source.full_name}.c * {toc[c.s_index]} * 1000 / (r * (c.delta + 1000) + 1000)"
    elif c.epsilon != "None":
        a = c.epsilon / 1000 + 1  # convert to alpha notation
        equation_string = f"{f_m} * {s_l} / ({a} * {s_c} + {s_l} - {a} * {s_l})"
        ds1 = (
            f"toc[c.s_index] * {c.source.full_name}.l * {flux.full_name}.c"
            f"/ (a * {c.source.full_name}.c + {c.source.full_name}.l"
            f"- a * {c.source.full_name}.l)"
        )
    else:
        if f_m == s_c:
            equation_string = f"{s_l} * toc[{c.s_index}]"
            ds1 = f"{c.source.full_name}.c"
        else:
            # equation_string = f"{f_m} * {s_l} / {s_c}"
            equation_string = f"{s_l} * toc[{c.s_index}]"
            ds1 = f"{flux.full_name} * toc[{c.s_index}] * {c.source.full_name}.l / {c.source.full_name}.c"

    if c.mo.debug_equations_file:
        debug_str = (
            f'"""\n'
            f"    isotope equations for {c.full_name}:{c.ctype}\n"
            f"    Flux = {flux.full_name}, d = {c.delta}, e = {c.epsilon}\n"
            f"    constants = CM[s.source.idx:, flux.idx + 1] = CM[:, flux.idx + 1] * toc[c.s_index]\n"
            f"    constants = CM[{c.source.idx + 1}, {flux.idx + 1}] ="
            f"    {CM[c.source.idx + 1, flux.idx + 1]:.2e} * {toc[c.s_index]:.2e}\n"
            f"    rhs_l = {ds1}\n"
            f"    rhs_l = {equation_string}\n"
            f'    """\n'
        )

    return equation_string, rhs_out, debug_str


def isotopes_scale_flux(
    f_m: str,  # flux expression including scale
    c: Species2Species,  # connection object
    icl: dict,  # initial conditions
    ind3: str,
    ind2: str,
    CM: NDArrayFloat,  # matrix
    toc: NDArrayFloat,  # table of constants
    flux: Flux,  #
) -> str:
    """Test if the connection involves any isotope effects.

    :param f_m: string with the flux name
    :param c: connection object
    :param icl: dict of reservoirs that have actual fluxes
    :param ind2: indent 2 times
    :param ind3: indent 3 times

    :returns equation string:
    """
    """Calculate the flux of the light isotope (f_l).

        If delta is given: fl = fm * 1000 / (r * (d + 1000) + 1000)
        If epsilon is given: fl = sl * fm / (e * sc + sl - a * sl)

        If neither: If flux mass equals source concentration, this is a scale
        with concentration connection, and f_l = s_l * scale where scale is
        part of the constants on the equations matrix. Otherwise, the flux
        isotope ratio is similar to the source isotope ratio:
        fl = fm * sl/sc * scale. Note that scale is dropped in the below equations
        since it is moved to the equations matrix.

        Where fl = flux light isotope, fm = flux mass/t, s_c = source concentration
        s_l = source concentration of light isotope
        r_index references the rate keyword in a given connection instance
        s_index references the scale keyword in a given connection instance
        FIXME: unify rate & scale?
    """
    from esbmtk import Source

    r: float = c.source.species.r  # isotope reference value
    s: str = get_ic(c.source, icl, True)  # R[0], R[1] reservoir concentrations
    s_c, s_l = s.replace(" ", "").split(",")  # total c, light isotope c
    debug_str = ""
    rhs_out = True
    debug_expression = ""
    scale = f"toc[{c.s_index}]"
    f_m = f"F[{c.ref_flux.idx}]"  # The ref flux is already scaled

    if c.delta != "None":
        # equation_string = f"{s_c} * 1000 / ({r} * ({c.delta} + 1000) + 1000)"
        equation_string = f"F[{flux.idx}] * 1000 / ({r} * ({c.delta} + 1000) + 1000)"
        ds1 = f"{scale} * {flux.full_name}.c * 1000 / (r * (c.delta + 1000) + 1000)"
    elif c.epsilon != "None":
        a = c.epsilon / 1000 + 1  # convert to alpha notation
        # note f_m is already scaled!
        equation_string = (
            f"{scale} * {f_m} * {s_l} / ({a} * {s_c} + {s_l} - {a} * {s_l})"
        )
        ds1 = (
            f"{scale} * {c.source.full_name}.l * {flux.full_name}.c"
            f"/ (a * {c.source.full_name}.c + {c.source.full_name}.l"
            f"- a * {c.source.full_name}.l)"
        )
    else:
        if f_m == s_c:
            equation_string = f"{s_l}"
            ds1 = f"{c.source.full_name}.c"
            debug_expression = "Flux from Reservoir = [flux.idx + 1] * c \n"
        elif isinstance(c.source, Source):
            # source isotope ratios are fixed, so we can place it on the coefficient matrix
            CM[:, flux.idx + 1] = CM[:, flux.idx + 1] * scale * c.source.isotope_ratio
            equation_string = f"{f_m}"
            ds1 = f"scale * {flux.full_name} * {c.source.isotope_ratio}"
            debug_expression = (
                f"Flux from Source = [flux.idx + 1] * scale * c.source.isotope_ratio\n"
                f"Flux from Source = {[flux.idx + 1]} * {scale} * {c.source.isotope_ratio}\n"
            )
        else:
            equation_string = f"{scale} * {f_m} * {s_l} / {s_c}"
            ds1 = f"{scale} * {flux.full_name} * {c.source.full_name}.l / {c.source.full_name}.c"

    if c.mo.debug_equations_file:
        # FIXME: this should reflect the above expressions using the same pattern as in
        # scale with concentration.
        debug_str = (
            f'"""\n'
            f"    isotope equations for {c.full_name}:{c.ctype}\n"
            f"    Flux = {flux.full_name}, d = {c.delta}, e = {c.epsilon}\n"
            f"    {debug_expression}"
            f"    rhs_l = {ds1}\n"
            f"    rhs_l = {equation_string}\n"
            f'    """\n'
        )

    return equation_string, rhs_out, debug_str


def check_signal_2(rhs: str, rhs_l: str, c: Species2Species) -> (str, str):
    """Test if connection is affected by a signal.

    :param ex: equation string
    :param c: connection object

    :returns: (modified) equation string
    """
    if c.signal != "None":  # get signal type
        if c.signal.stype == "addition":
            sign = "+"
        elif c.signal.stype == "multiplication":
            sign = "*"
        else:
            raise ValueError(f"stype={c.signal.stype} not implemented")

        rhs = f"{rhs} {sign} {c.signal.full_name}(t)[0]  # Signal"
        if rhs_l != "":
            rhs_l = f"{rhs_l} {sign} {c.signal.full_name}(t)[1]  # Signal"

    return rhs, rhs_l


def get_initial_conditions(
    M: Model,
    rtol: float,
    atol_d: float = 1e-7,
) -> tuple[dict[str, float], dict, list, list, NDArrayFloat]:
    """Get list of initial conditions.

    This list needs to match the number of equations.

    :param Model: The model handle
    :param rtol: relative tolerance for BDF solver.
    :param atol_d: default value for atol if c = 0

    :return: lic = dict of initial conditions
    :return: icl = dict[Species, list[int, int]] where reservoir
             indicates the reservoir handle, and the list contains the
             index into the reservoir data.  list[0] = concentration
             list[1] concentration of the light isotope.
    :return: cpl = list of reservoirs that use function to evaluate
             reservoir data
    :return: ipl = list of static reservoirs that serve as input
    :return: rtol = array of tolerence values for ode solver

        We need to consider 3 types of reservoirs:

        1) Species that change as a result of physical fluxes i.e.
        r.lof > 0.  These require a flux statements and a reservoir
        equation.

        2) Species that do not have active fluxes but are computed
        as a tracer, i.e.. HCO3.  These only require a reservoir
        equation

        3) Species that do not change but are used as input.  Those
        should not happen in a well formed model, but we cannot
        exclude the possibility.  In this case, there is no flux
        equation, and we state that dR/dt = 0

        get_ic() will look up the index position of the
        reservoir_handle on icl, and then use this index to retrieve
        the correspinding value in R

        Isotopes are handled by adding a second entry
    """
    import numpy as np

    lic = {}  # list of initial conditions
    atol: list = []  # list of tolerances for ode solver
    # dict that contains the reservoir_handle as key and the index positions
    # for c and l as a list
    icl: dict[Species, list[int, int]] = {}
    cpl: list = []  # list of reservoirs that are computed
    ipl: list = []  # list of static reservoirs that serve as input

    # collect all reservoirs that have initial conditions
    # if r.rtype != "flux_only":
    i: int = 0
    for r in M.lic:
        if len(r.lof) >= 0 or r.rtype == "computed" or r.rtype == "passive":
            lic[r.full_name] = r.c[0]
            tol = rtol * abs(r.c[0]) / 10 if r.c[0] > 0 else atol_d
            atol.append(tol)
            r.atol[0] = tol
            if r.isotopes:
                tol = rtol * abs(r.l[0]) / 10 if r.l[0] > 0 else atol_d
                atol.append(tol)
                r.atol[1] = tol
                lic[f"{r.full_name}_l"] = r.l[0]
                icl[r] = [i, i + 1]
                i += 2
            else:
                icl[r] = [i, i]
                i += 1
                # if r.name == "O2_At":
            if M.debug_equations_file:
                print(f"r = {r.full_name}, r.c[0] = {r.c[0]:.2e}, rtol = {tol:.2e}")

    return lic, icl, cpl, ipl, np.array(atol)
