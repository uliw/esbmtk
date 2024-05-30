"""
esbmtk: A general purpose Earth Science box model toolkit Copyright
(C), 2020 Ulrich G.  Wortmann

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
import numpy as np
import numpy.typing as npt
import typing as tp

NDArrayFloat = npt.NDArray[np.float64]

if tp.TYPE_CHECKING:
    from esbmtk import Flux, Species, Model, Species2Species
    from esbmtk import ExternalFunction


def get_initial_conditions(
    M: Model,
    rtol: float,
    atol_d: float = 1e-7,
) -> tuple[list, dict, list, list, NDArrayFloat]:
    """Get list of initial conditions.  This list needs to match the
    number of equations.

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
    atol: tp.List = []  # list of tolerances for ode solver
    # dict that contains the reservoir_handle as key and the index positions
    # for c and l as a list
    icl: dict[Species, List[int, int]] = {}
    cpl: tp.List = []  # list of reservoirs that are computed
    ipl: tp.List = []  # list of static reservoirs that serve as input

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
            #     breakpoint()

    return R, icl, cpl, ipl, np.array(atol)


def write_reservoir_equations(eqs, M: Model, rel: str, ind2: str, ind3: str) -> str:
    """Loop over reservoirs and their fluxes to build the reservoir
    equation

    :param eqs: equation file handle
    :param rel: string with reservoir names used in return function.
        Note that these are the reervoir names as used by the
        equations and not the reservoir names used by esbmtk. E.g.,
        M1.R1.O2 will be M1_R1_O2,
    :param ind2: string with indentation offset
    :param ind3: string with indentation offset

    :returns: rel = updated list of reservoirs names
    """

    for r in M.lor:  # loop over reservoirs
        """
        Write equations for each reservoir and create unique variable
        names.  Species are typiclally called M.rg.r so we replace
        all dots with underscore -> M_rg_r
        """

        if r.rtype != "flux_only":
            name = f'dCdt_{r.full_name.replace(".", "_")}'
            fex = ""
            # v_val = f"{r.volume.to(r.v_unit).magnitude}"
            v_val = f"toc[{r.v_index}]"

            # add all fluxes
            for flux in r.lof:  # check if in or outflux
                if flux.parent.source == r:
                    sign = "-"
                else:
                    sign = "+"

                fname = f'{flux.full_name.replace(".", "_")}'
                fex = f"{fex}{ind3}{sign} {fname}\n"

            if len(r.lof) > 0:  # avoid reservoirs without active fluxes
                if r.ef_results:
                    eqs.write(f"{ind2}{name} += (\n{fex}{ind2})/{v_val}\n\n")
                else:
                    eqs.write(f"{ind2}{name} = (\n{fex}{ind2})/{v_val}\n\n")

                rel = f"{rel}{ind3}{name},\n"

    return rel


def write_reservoir_equations_with_isotopes(
    eqs,
    M: Model,
    rel: str,  # string with reservoir names returned by setup_ode
    ind2: str,  # indent 2 times
    ind3: str,  # indent 3 times
) -> str:  # updated list of reservoirs
    """Loop over reservoirs and their fluxes to build the reservoir equation"""

    for r in M.lor:  # loop over reservoirs
        if r.rtype != "flux_only":
            # v_val = f"{r.volume.to(r.v_unit).magnitude}"
            v_val = f"toc[{r.v_index}]"
            # Write equations for each reservoir
            # create unique variable names. Species are typiclally called
            # M.rg.r so we replace all dots with underscore
            if r.isotopes:
                name = f'dCdt_{r.full_name.replace(".", "_")}_l'
                fex = ""
                # add all fluxes
                for flux in r.lof:  # check if in or outflux
                    if flux.parent.source == r:
                        sign = "-"
                    else:
                        # elif flux.parent.sink == r:
                        sign = "+"

                    fname = f'{flux.full_name.replace(".", "_")}_l'
                    fex = f"{fex}{ind3}{sign} {fname}\n"

                # avoid reservoirs without active fluxes
                if len(r.lof) > 0:
                    if r.ef_results:
                        eqs.write(f"{ind2}{name} += (\n{fex}{ind2})/{v_val}\n\n")
                    else:
                        eqs.write(f"{ind2}{name} = (\n{fex}{ind2})/{v_val}\n\n")

                    rel = f"{rel}{ind3}{name},\n"

    return rel


def write_equations_2(
    M: Model,
    R: tp.List[float],
    icl: dict,
    cpl: tp.List,
    ipl: tp.List,
    fn: str,
) -> tuple:
    """Write file that contains the ode-equations for the Model
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
    from esbmtk import Species


    # construct header and static code:
    # add optional import statements
    h1 = """from __future__ import annotations
from numpy import array as npa
from numba import njit
"""

    h2 = """# @njit(fastmath=True)
def eqs(t, R, M, gpt, toc, area_table, area_dz_table, Csat_table) -> list:
        '''Auto generated esbmtk equations do not edit
        '''
"""
    ind2 = 8 * " "  # indention
    ind3 = 12 * " "  # indention
    hi = ""

    # ensure there are no duplicates
    M.lpc_i = set(M.lpc_i)
    M.lpc_f = set(M.lpc_f)

    if len(M.lpc_f) > 0:
        hi += f"from esbmtk import "
        for f in M.lpc_f:
            hi += f"{f} ,"
        hi = f"{hi[:-2]}\n"  # strip comma and space

    if len(M.lpc_i) > 0:  # test if functions imports are required
        hi += f"from esbmtk.bio_pump_functions{M.bio_pump_functions} import "
        for f in M.lpc_i:
            hi += f"{f} ,"
        hi = f"{hi[:-2]}\n"  # strip comma and space

    if M.luf:  # test if user defined function imports are required
        for function, args in M.luf.items():
            source = args[0]
            hi += f"from {source} import {function}\n"

    header = f"{h1}{hi}\n{h2}"

    rel = ""  # list of return values
    # """
    # write file
    with open(fn, "w", encoding="utf-8") as eqs:
        eqs.write(header)
        sep = (
            "# ---------------- write computed reservoir equations -------- #\n"
            + "# that do not depend on fluxes"
        )
        eqs.write(f"\n{sep}\n")

        for r in M.lpc_r:  # All virtual reservoirs need to be in this list
            if r.ftype == "std":
                # rel = write_cs_1(eqs, r, icl, rel, ind2, ind3)
                rel = write_ef(eqs, r, icl, rel, ind2, ind3, M.gpt)
            elif r.ftype == "needs_flux":
                pass
            else:
                raise ValueError(f"{r.ftype} is undefined")

        flist = []
        sep = "# ---------------- write all flux equations ------------------- #"
        eqs.write(f"\n{sep}\n")
        for flux in M.lof:  # loop over fluxes
            if flux.register.ctype == "ignore" or flux.ftype == "computed":
                continue  # skip
            # fluxes belong to at least 2 reservoirs, so we need to avoid duplication
            # we cannot use a set, since we need to preserve order
            if flux not in flist:
                flist.append(flux)  # add to list of fluxes already computed
                if isinstance(flux.parent, Species):
                    continue  # skip computed fluxes

                ex, exl = get_flux(flux, M, R, icl)  # get flux expressions
                fn = flux.full_name.replace(".", "_")
                # all others types that have separate expressions/isotope
                eqs.write(f"{ind2}{fn} = {ex}\n")
                if flux.parent.isotopes:  # add line for isotopes
                    eqs.write(f"{ind2}{fn}_l =  {exl}\n")
                    if flux.full_name == "M.CG_A_sb_to_A_ib.DIC_._F":
                        print(f"wrote = {fn}_l  {exl}")

        sep = (
            "# ---------------- write computed reservoir equations -------- #\n"
            + "# that do depend on fluxes"
        )

        eqs.write(f"\n{sep}\n")

        for r in M.lpc_r:  # All virtual reservoirs need to be in this list
            if r.ftype == "std":
                pass  # see above
            elif r.ftype == "needs_flux":  #
                rel = write_ef(eqs, r, icl, rel, ind2, ind3, M.gpt)
            else:
                raise ValueError(f"{r.ftype} is undefined")

        sep = "# ---------------- write input only reservoir equations -------- #"
        eqs.write(f"\n{sep}\n")
        for r in ipl:
            rname = r.full_name.replace(".", "_")
            eqs.write(f"{ind2}{rname} = 0.0")

        sep = "# ---------------- write regular reservoir equations ------------ #"
        eqs.write(f"\n{sep}\n")

        rel = write_reservoir_equations(eqs, M, rel, ind2, ind3)

        sep = "# ---------------- write isotope reservoir equations ------------ #"
        eqs.write(f"\n{sep}\n")

        rel = write_reservoir_equations_with_isotopes(eqs, M, rel, ind2, ind3)

        sep = "# ---------------- bits and pieces --------------------------- #"
        eqs.write(f"\n{sep}\n" f"{ind2}return [\n")
        # Write all initial conditions that are recorded in icl
        for k, v in icl.items():
            eqs.write(f"{ind3}dCdt_{k.full_name.replace('.', '_')},  # {v[0]}\n")
            if k.isotopes:
                eqs.write(f"{ind3}dCdt_{k.full_name.replace('.', '_')}_l,  # {v[1]}\n")

        eqs.write(f"{ind2}]\n")

    return fn


def get_flux(flux: Flux, M: Model, R: tp.List[float], icl: dict) -> tuple(str, str):
    """Create formula expressions that calcultes the flux F.  Return
    the equation expression as string

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

    if c.ctype.casefold() == "regular":
        ex, exl = get_regular_flux_eq(flux, c, icl, ind2, ind3)

    elif c.ctype == "scale_with_concentration":
        ex, exl = get_scale_with_concentration_eq(flux, c, cfn, icl, ind2, ind3)

    elif c.ctype == "scale_with_mass":
        ex, exl = get_scale_with_concentration_eq(flux, c, cfn, icl, ind2, ind3)

    elif c.ctype == "scale_with_flux":
        ex, exl = get_scale_with_flux_eq(flux, c, cfn, icl, ind2, ind3)

    elif c.ctype == "ignore":
        pass  # do nothing

    else:
        raise ValueError(
            f"Species2Species type {c.ctype} for {c.full_name} is not implmented"
        )

    return ex, exl


def check_signal_2(ex: str, exl: str, c: Species2Species) -> (str, str):
    """Test if connection is affected by a signal

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


def get_ic(r: Species, icl: dict, isotopes=False) -> str:
    """Get initial condition in a reservoir.  If the reservoir is icl,
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
    from esbmtk import Source, Sink

    if r in icl:
        s1 = f"R[{icl[r][0]}]"
        if isotopes:
            s1 += f", R[{icl[r][1]}]"

    elif isinstance(r, (Source, Sink)):
        s1 = f"{r.full_name}.c"
        if isotopes:
            s1 += f", {r.full_name}.l"
    else:
        raise ValueError(
            f"get_ic: can't find {r.full_name} in list of initial conditions"
        )

    return s1


def parse_esbmtk_input_data_types(d: any, r: Species, ind: str, icl: dict) -> str:
    """Parse esbmtk data types that are provided as arguments
    to external function objects, and convert them into a suitable string
    format that can be used in the ode equation file
    """
    from esbmtk import Flux, Species, Reservoir, SeawaterConstants, Q_
    from esbmtk import GasReservoir

    if isinstance(d, str):
        sr = getattr(r.register, d)
        a = f"{ind}{get_ic(sr, icl)},\n"
    elif isinstance(d, Species):
        a = f"{ind}({get_ic(d, icl,d.isotopes)}),\n"
    elif isinstance(d, GasReservoir):
        # print(f" {d.full_name} isotopes {d.isotopes}")
        a = f"{ind}({get_ic(d, icl,d.isotopes)}),\n"
    elif isinstance(d, Reservoir):
        a = f"{ind}{d.full_name},\n"
    elif isinstance(d, Flux):
        sr = d.full_name.replace(".", "_")
        a = f"{ind}{sr},\n"
    elif isinstance(d, SeawaterConstants):
        a = f"{ind}{d.full_name},\n"
    elif isinstance(d, (float, int, np.ndarray)):
        a = f"{ind}{d},\n"
    elif isinstance(d, dict):
        # get key and reservoiur handle
        sr = getattr(r.register, next(iter(d)))
        a = f"{ind}{get_ic(sr, icl)},\n"
    elif isinstance(d, Q_):
        a = f"{ind}{d.magnitude},\n"
    elif isinstance(d, float):
        a = f"{ind}{d.magnitude},\n"
    elif isinstance(d, list):  # loo pover list elements
        a = f"{ind}{d},\n"
        a = f"{ind}npa(["
        for e in d:
            a += f"{parse_esbmtk_input_data_types(e, r,'',icl)[0:-2]},"
        a += "]),\n"
    else:
        raise ValueError(
            f"\n r = {r.full_name}, d ={d}\n" f"\n{d} is of type {type(d)}\n",
        )

    return a


def parse_function_params(params, ind) -> str:
    """Parse function_parameters and convert them into a suitable string
    format that can be used in the ode equation file
    """
    a = ""
    for p in params:
        if isinstance(p, str):
            a += f"{ind}{p},\n"
        else:
            a += f"{ind}{p},\n"
    return a


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
    """Write external function call code

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
    for i, o in enumerate(ef.lro):
        if isinstance(o, Flux):
            v = f"{o.full_name.replace('.', '_')}, "
        else:
            v = f"dCdt_{o.full_name.replace('.', '_')}, "

        rv += v
        if ef.lro[i].isotopes:
            rv += f"{v[:-2]}_l, "

    rv = rv[:-2]  # strip the last comma
    a = ""

    # if ef.fname == "carbonate_system_2_ode":
    # if ef.fname == "weathering":
    #     breakpoint()
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


def get_regular_flux_eq(
    flux: Flux,  # flux instance
    c: Species2Species,  # connection instance
    icl: dict,  # list of initial conditions
    ind2,  # indentation
    ind3,  # indentation
) -> tuple:
    """Create a string containing the equation for a regular (aka
    fixed rate) connection

    :param flux: flux instance
    :param c: connection object
    :param icl: dict of reservoirs that have actual fluxes
    :param ind2: indent 2 times
    :param ind3: indent 3 times

    :returns: two strings, where the first describes the equation for
              the total flux, and the second describes the rate for
              the light isotope
    """
    ex = f"toc[{c.r_index}]"
    # ex = f"{flux.full_name}.rate"  # get flux rate string
    exl = check_isotope_effects(ex, c, icl, ind3, ind2)
    ex, exl = check_signal_2(ex, exl, c)  # check if we hav to add a signal

    return ex, exl


def check_isotope_effects(
    f_m: str,
    c: Species2Species,
    icl: dict,
    ind3: str,
    ind2: str,
) -> str:
    """Test if the connection involves any isotope effects

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


def get_scale_with_concentration_eq(
    flux: Flux,  # flux instance
    c: Species2Species,  # connection instance
    cfn: str,  # full name of the connection instance
    icl: dict,  # list of initial conditions
    ind2: str,  # whitespace
    ind3: str,  # whitespace
) -> tuple(str, str):
    """Create equation string defining a flux that scales with the
    concentration in the upstream reservoir

    Example: M1_CG_D_b_to_L_b_TA_thc__F =
    M1.CG_D_b_to_L_b.TA_thc.scale * R[5]

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
    s_c = get_ic(c.ref_reservoirs, icl)  # get index to concentration
    ex = f"toc[{c.s_index}] * {s_c}"
    exl = check_isotope_effects(ex, c, icl, ind3, ind2)
    ex, exl = check_signal_2(ex, exl, c)
    return ex, exl


def get_scale_with_flux_eq(
    flux: Flux,  # flux instance
    c: Species2Species,  # connection instance
    cfn: str,  # full name of the connection instance
    icl: dict,  # list of initial conditions
    ind2: str,  # indentation
    ind3: str,  # indentation
) -> tuple(str, str):
    """Equation defining a flux that scales with strength of another
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
    p = flux.parent.ref_flux.parent
    fn = f"{p.full_name.replace('.', '_')}__F"
    # get the equation string for the flux
    ex = f"toc[{c.s_index}] * {fn}"
    """ The flux for the light isotope will computed as follows:
    We will use the mass of the flux we or scaling, but that we will set the
    delta|epsilon according to isotope ratio of the reference flux
    """
    if c.source.isotopes and c.sink.isotopes:
        exl = check_isotope_effects(ex, c, icl, ind3, ind2)

    else:
        exl = ""
    ex, exl = check_signal_2(ex, exl, c)
    return ex, exl
