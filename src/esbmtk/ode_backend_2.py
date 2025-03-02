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
    from esbmtk import ExternalFunction, Flux, Model, Species, Species2Species


def build_eqs_matrix(M: Model) -> tuple[NDArrayFloat, NDArrayFloat]:
    """Build the coefficient matrix CM and Flux vector f.

    So that we can solve dC_dt = C dot flux_values
    """
    from esbmtk import GasReservoir, Species

    fi = 0
    for f in M.lof:  # get flux index positions
        f.idx = fi  # set starting index
        fi = fi + 1

    # get number of reservoirs
    num_res = 0
    for r in M.lor:
        num_res = num_res + 1
        if r.isotopes:
            num_res = num_res + 1

    F = np.ones(fi)  # initialize flux value vector
    CM = np.zeros((num_res, fi), dtype=float)  # Initialize Coefficient matrix:

    """We have n reservoirs, without isotopes, this corresponds to
    n rows in the coefficinet matrix. However, if reservoirs have
    isotopes, we need an additional rao to calculate the respective
    isotope concentration. Thus we have to two counter, cr pointing to
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
                sign = -1 / mass if f.parent.source == r else 1 / mass

                if r.isotopes:  # add equation for isotopes
                    CM[ri, r.lof[fi].idx] = sign
                    CM[ri + 1, r.lof[fi + 1].idx] = sign  # 2
                    fi = fi + 2
                else:
                    CM[ri, r.lof[fi].idx] = sign
                    fi = fi + 1

        ri = ri + 1  # increase count by 1, or two is isotopes
        if r.isotopes:
            ri = ri + 1

    return CM[:ri, :], F


def write_equations_3(
    M: Model,
    R: list[float],
    icl: dict,
    cpl: list,
    ipl: list,
    eqs_fn: str,
) -> tuple:
    """Write file that contains the ode-equations for the Model.

    Returns the list R that contains the initial condition for each
    reservoir

    :param Model: Model handle
    :param R: tp.List of floats with the initial conditions for each
              reservoir
    :param icl: dict of reservoirs that have actual fluxes
    :param cpl: tp.List of reservoirs that have no fluxes but are
        computed based on other reservoirs
    :param ipl: tp.List of reservoir that do not change in concentration
    :param fn: str, filename
    """
    # construct header and static code:
    # add optional import statements
    h1 = """

    """

    h2 = """# @njit(fastmath=True)
def eqs(t, R, M, gpt, toc, area_table, area_dz_table, Csat_table, CM, F):
    '''Calculate dCdt for each reservoir.

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
   
    Reservoir and flux name lookup: M.lor[idx] and M.lof[idx]
    '''
    """
    ind2 = 4 * " "  # indention
    ind3 = 8 * " "  # indention
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

    header = f"{h1}\n{hi}\n{h2}"

    rel = ""  # list of return values
    # """
    # write file
    with open(eqs_fn, "w", encoding="utf-8") as eqs:
        eqs.write(header)
        sep = (
            "    # ---------------- write computed reservoir equations -------- #\n"
            + "    # that do not depend on fluxes"
        )
        eqs.write(f"\n{sep}\n")

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
                elif flux.ftype == "None":
                    ex, exl = get_flux(flux, M, R, icl)  # get flux expressions
                    if ex != "":
                        fn = f"F[{flux.idx}]"
                        eqs.write(f"{ind2}{fn} = {ex}\n")
                    if exl != "":  # add line for isotopes
                        fn = f"F[{flux.idx + 1}]"
                        eqs.write(f"{ind2}{fn} =  {exl}\n")
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
            # v = f"dCdt_{o.full_name.replace('.', '_')}, "

        rv += v
        # if ef.lro[i].isotopes:
        #     # rv += f"{v[:-2]}_l, "
        #     rv += f"F[{o.idx + 1}], "

    rv = rv[:-2]  # strip the last comma
    a = ""

    # if ef.fname == "carbonate_system_2_ode":
    # if ef.fname == "weathering":

    # this can probably be simplified similar to the old parse_return_values()
    for d in ef.function_input_data:
        a += parse_esbmtk_input_data_types(d, ef, ind3, icl)

    if ef.function_params == "None":
        eqs.write(f"{rv} = {ef.fname}(\n{a}{ind2})\n\n")
    else:
        s = f"gpt[{ef.param_start}]"
        eqs.write(f"{rv} = {ef.fname}(\n{a}{ind3}{s},\n{ind2})\n\n")

    rel += f"{ind3}{rv},\n"

    return rel


def parse_esbmtk_input_data_types(d: any, r: Species, ind: str, icl: dict) -> str:
    """Parse esbmtk data types.

    That are provided as arguments
    to external function objects, and convert them into a suitable string
    format that can be used in the ode equation file
    """
    from esbmtk import Q_, Flux, GasReservoir, Reservoir, SeawaterConstants, Species

    if isinstance(d, str):
        sr = getattr(r.register, d)
        a = f"{ind}{get_ic(sr, icl)},\n"
    elif isinstance(d, Species | GasReservoir):
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
            f"\n r = {r.full_name}, d ={d}\n\n{d} is of type {type(d)}\n",
        )

    return a


def get_flux(flux: Flux, M: Model, R: list[float], icl: dict) -> tuple(str, str):
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

    ex = ""  # expression string
    exl = ""  # expression string for light isotope

    c = flux.parent  # shorthand for the connection object
    cfn = flux.parent.full_name  # shorthand for the connection object name

    """flux parent would typically be a connection object. However for computed
    reservoirs, the flux is registed with a the computed reservoir.
    """
    from esbmtk import Species, Species2Species

    if isinstance(c, Species2Species):
        if c.ctype.casefold() == "regular" or c.ctype.casefold() == "fixed":
            ex, exl = get_regular_flux_eq(flux, c, icl, ind2, ind3, M.CM, M.toc)
        elif c.ctype == "scale_with_concentration" or c.ctype == "scale_with_mass":
            ex, exl = get_scale_with_concentration_eq(
                flux, c, cfn, icl, ind2, ind3, M.CM, M.toc
            )
        elif c.ctype == "scale_with_flux":
            ex, exl = get_scale_with_flux_eq(flux, c, cfn, icl, ind2, ind3, M.CM, M.toc)
        elif c.ctype == "ignore" or c.ctype == "gasexchange":
            pass
        else:
            raise ValueError(
                f"Species2Species type {c.ctype} for {c.full_name} is not implemented"
            )
    elif isinstance(c, Species):
        ex = f"{0}  # {c.full_name}"  # The flux will simply be zero
    else:
        raise ValueError(f"No definition for \n{c.full_name} type = {type(c)}\n")

    return ex, exl


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
        s1 = f"{r.full_name}.c"
        if isotopes:
            s1 += f", {r.full_name}.l"
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
) -> tuple:
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
    ex = ""
    exl = ""
    output = True
    if flux.serves_as_input or c.signal != "None":
        ex = f"toc[{c.r_index}]"
        exl = check_isotope_effects(ex, c, icl, ind3, ind2)
        ex, exl = check_signal_2(ex, exl, c)  # check if we hav to add a signal
    else:
        output = False
        CM[:, flux.idx] = CM[:, flux.idx] * toc[c.r_index]
        if flux.isotopes:
            CM[:, flux.idx + 1] = CM[:, flux.idx + 1] * toc[c.r_index]
            exl = check_isotope_effects(ex, c, icl, ind3, ind2)

    if c.mo.debug_equations_file and output:
        ex = ex + f"  # {flux.full_name} = {toc[c.r_index]:.2e}"

    return ex, exl


def get_scale_with_concentration_eq(
    flux: Flux,  # flux instance
    c: Species2Species,  # connection instance
    cfn: str,  # full name of the connection instance
    icl: dict,  # list of initial conditions
    ind2: str,  # whitespace
    ind3: str,  # whitespace
    CM: NDArrayFloat,  # coefficient matrix
    toc: tuple,  # tuple of constants
) -> tuple(str, str):
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
    s_c = get_ic(c.ref_reservoirs, icl)  # get index to concentration`
    exl = ""
    ex = ""
    if flux.serves_as_input or c.signal != "None":
        ex = f"toc[{c.s_index}] * {s_c}"
        exl = check_isotope_effects(ex, c, icl, ind3, ind2)
        ex, exl = check_signal_2(ex, exl, c)
    else:
        CM[:, flux.idx] = CM[:, flux.idx] * toc[c.s_index]
        ex = f"{s_c}"
        if flux.isotopes:
            CM[:, flux.idx + 1] = CM[:, flux.idx + 1] * toc[c.s_index]
            exl = check_isotope_effects(ex, c, icl, ind3, ind2)

    if c.mo.debug_equations_file:
        ex = (
            ex
            + f"  # {flux.full_name} = {toc[c.s_index]:.2e} * {c.ref_reservoirs.full_name}"
        )
    return ex, exl


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
    # get the reference flux name
    exl = ""
    fn = f"F[{c.ref_flux.idx}]"
    if flux.serves_as_input or c.signal != "None":
        ex = f"toc[{c.s_index}] * {fn}"
        """The flux for the light isotope will computed as follows:
        We will use the mass of the flux we or scaling, but that we will set the
        delta|epsilon according to isotope ratio of the reference flux
        """
        exl = ""
        if c.source.isotopes and c.sink.isotopes:
            exl = check_isotope_effects(ex, c, icl, ind3, ind2)

        else:
            ex, exl = check_signal_2(ex, exl, c)

    else:
        CM[:, flux.idx] = CM[:, flux.idx] * toc[c.s_index]
        ex = fn
        if flux.isotopes:
            CM[:, flux.idx + 1] = CM[:, flux.idx + 1] * toc[c.s_index]
            exl = check_isotope_effects(ex, c, icl, ind3, ind2)

    if c.mo.debug_equations_file:
        ex = (
            ex + f"  # {flux.full_name} = {toc[c.s_index]:.2e} * {c.ref_flux.full_name}"
        )

    return ex, exl


def check_isotope_effects(
    f_m: str,
    c: Species2Species,
    icl: dict,
    ind3: str,
    ind2: str,
) -> str:
    """Test if the connection involves any isotope effects.

    :param f_m: string with the flux name
    :param c: connection object
    :param icl: dict of reservoirs that have actual fluxes
    :param ind2: indent 2 times
    :param ind3: indent 3 times

    :returns eq: equation string
    """
    if c.isotopes:
        r = c.source.species.r
        s = get_ic(c.source, icl, True)
        s_c, s_l = s.replace(" ", "").split(",")
        """ Calculate the flux of the light isotope (f_l) as a function of the isotope
        ratios in the source reservoir soncentrations (s_c, s_l), and epsilon (a) as
        f_l = f_m * 1000/(r * (d + 1000) + 1000)

        Note that the scale has already been applaied to f_m in the calling function.
        """
        if c.delta != "None":
            d = c.delta
            eq = f"{f_m} * 1000 / ({r} * ({d} + 1000) + 1000)"
        elif c.epsilon != "None":
            a = c.epsilon / 1000 + 1
            eq = f"{s_l} * {f_m} / ({a} * {s_c} + {s_l} - {a} * {s_l})"
        else:
            eq = f"{f_m} * {s_l} / {s_c}"
    else:
        eq = ""

    return eq


def check_signal_2(ex: str, exl: str, c: Species2Species) -> (str, str):
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

        ex += f" {sign} {c.signal.full_name}(t)[0]  # Signal"
        if c.source.isotopes:
            exl += f" {sign} {c.signal.full_name}(t)[1]  # Signal"
        else:
            exl += ""

    return ex, exl


def get_initial_conditions(
    M: Model,
    rtol: float,
    atol_d: float = 1e-7,
) -> tuple[list, dict, list, list, NDArrayFloat]:
    """Get list of initial conditions.

    This list needs to match the number of equations.

    :param Model: The model handle
    :param rtol: relative tolerance for BDF solver.
    :param atol_d: default value for atol if c = 0

    :return: R = list of initial conditions as floats
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

    R = []  # list of initial conditions
    atol: list = []  # list of tolerances for ode solver
    # dict that contains the reservoir_handle as key and the index positions
    # for c and l as a list
    icl: dict[Species, list[int, int]] = {}
    cpl: list = []  # list of reservoirs that are computed
    ipl: list = []  # list of static reservoirs that serve as input

    i: int = 0
    for r in M.lic:
        # collect all reservoirs that have initial conditions
        # if r.rtype != "flux_only":
        if len(r.lof) > 0 or r.rtype == "computed" or r.rtype == "passive":
            R.append(r.c[0])  # add initial condition
            if r.c[0] > 0:
                # compute tol such that tol < rtol * abs(y)
                tol = rtol * abs(r.c[0]) / 10
                atol.append(tol)
                r.atol[0] = tol
            else:
                atol.append(atol_d)
                r.atol[0] = atol_d

            if r.isotopes:
                if r.l[0] > 0:
                    # compute tol such that tol < rtol * abs(y)
                    tol = rtol * abs(r.l[0]) / 10
                    atol.append(tol)
                    r.atol[1] = tol
                else:
                    atol.append(atol_d)
                    r.atol[1] = atol_d

                R.append(r.l[0])  # add initial condition for l
                icl[r] = [i, i + 1]
                i += 2
            else:
                icl[r] = [i, i]
                i += 1
            # if r.name == "O2_At":
    return R, icl, cpl, ipl, np.array(atol)
