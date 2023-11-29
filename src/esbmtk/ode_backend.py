from __future__ import annotations
import numpy as np
import numpy.typing as npt

# from numba.typed import List
from esbmtk import Flux, Reservoir, Model, Connection, Connect
from esbmtk import AirSeaExchange

# declare numpy types
NDArrayFloat = npt.NDArray[np.float64]


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
    :return: icl = dict[Reservoir, list[int, int]] where reservoir
             indicates the reservoir handle, and the list contains the
             index into the reservoir data.  list[0] = concentration
             list[1] concentration of the light isotope.
    :return: cpl = list of reservoirs that use function to evaluate
             reservoir data
    :return: ipl = list of static reservoirs that serve as input
    :return: rtol = array of tolerence values for ode solver

        We need to consider 3 types of reservoirs:

        1) Reservoirs that change as a result of physical fluxes i.e.
        r.lof > 0.  These require a flux statements and a reservoir
        equation.

        2) Reservoirs that do not have active fluxes but are computed
        as a tracer, i.e.. HCO3.  These only require a reservoir
        equation

        3) Reservoirs that do not change but are used as input.  Those
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
    icl: dict[Reservoir, list[int, int]] = {}
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
        names.  Reservoirs are typiclally called M.rg.r so we replace
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
                    # elif flux.parent.sink == r:
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
            # create unique variable names. Reservoirs are typiclally called
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
    M: Model, R: list[float], icl: dict, cpl: list, ipl: list
) -> tuple:
    """Write file that contains the ode-equations for the Model
    Returns the list R that contains the initial condition for each
    reservoir

    :param Model: Model handle
    :param R: list of floats with the initial conditions for each
              reservoir
    :param icl: dict of reservoirs that have actual fluxes
    :param cpl: list of reservoirs that have no fluxes but are
        computed based on other reservoirs
    :param ipl: list of reservoir that do not change in concentration
    """
    # from esbmtk import Model, ReservoirGroup
    import pathlib as pl

    # get pathlib object
    fn: str = "equations.py"  # file name
    cwd: pl.Path = pl.Path.cwd()  # get the current working directory
    fqfn: pl.Path = pl.Path(f"{cwd}/{fn}")  # fully qualified file name

    # construct header and static code:
    # add optional import statements
    h1 = """from __future__ import annotations
from numpy import array as npa
from numba import njit
from esbmtk import gas_exchange_ode, gas_exchange_ode_with_isotopes\n\n"""
    
    h2 = """# @njit(fastmath=True)
def eqs(t, R, M, gpt, toc, area_table, area_dz_table, Csat_table) -> list:
        '''Auto generated esbmtk equations do not edit
        '''
"""
    ind2 = 8 * " "  # indention
    ind3 = 12 * " "  # indention
    hi = ""
    if len(M.lpc_f) > 0:
        hi = f"from esbmtk.bio_pump_functions{M.bio_pump_functions} import "
        for f in set(M.lpc_f):
            hi += f"{f} ,"
        hi = hi[:-2] # strip comma and space
            
    header = f"{h1}\n{hi}\n{h2}"

    rel = ""  # list of return values
    # """
    # write file
    with open(fqfn, "w", encoding="utf-8") as eqs:
        eqs.write(header)
        sep = (
            "# ---------------- write computed reservoir equations -------- #\n"
            + "# that do not depend on fluxes"
        )
        eqs.write(f"\n{sep}\n")

        for r in M.lpc_r:  # All virtual reservoirs need to be in this list
            if r.ftype == "cs1":
                # rel = write_cs_1(eqs, r, icl, rel, ind2, ind3)
                rel = write_ef(eqs, r, icl, rel, ind2, ind3, M.gpt)
            elif r.ftype != "cs2":
                raise ValueError(f"{r.ftype} is undefined")

        flist = []
        sep = "# ---------------- write all flux equations ------------------- #"
        eqs.write(f"\n{sep}\n")
        for flux in M.lof:  # loop over fluxes
            if flux.register.ctype == "ignore":
                continue  # skip
            # fluxes belong to at least 2 reservoirs, so we need to avoid duplication
            # we cannot use a set, since we need to preserve order
            if flux not in flist:
                flist.append(flux)  # add to list of fluxes already computed

                if isinstance(flux.parent, Reservoir):
                    continue  # skip computed fluxes

                ex, exl = get_flux(flux, M, R, icl)  # get flux expressions
                fn = flux.full_name.replace(".", "_")

                # check for types that return isotope data as well
                if isinstance(flux.parent, AirSeaExchange):
                    if flux.parent.isotopes:  # add F_l if necessary
                        fn = f"{fn}, {fn}_l"
                    eqs.write(f"{ind2}{fn} = {ex}\n")

                else:  # all others types
                    eqs.write(f"{ind2}{fn} = {ex}\n")
                    if flux.parent.isotopes:  # add line for isotopes
                        eqs.write(f"{ind2}{fn}_l = {exl}\n")

        sep = (
            "# ---------------- write computed reservoir equations -------- #\n"
            + "# that do depend on fluxes"
        )

        eqs.write(f"\n{sep}\n")

        for r in M.lpc_r:  # All virtual reservoirs need to be in this list
            if r.ftype == "cs1":
                pass  # see above
            elif r.ftype == "cs2":  #
                # rel = write_cs_2(eqs, r, icl, rel, ind2, ind3)
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

        # rel = write_reservoir_equations(eqs, M, rel, ind2, ind3)
        rel = write_reservoir_equations(eqs, M, rel, ind2, ind3)

        sep = "# ---------------- write isotope reservoir equations ------------ #"
        eqs.write(f"\n{sep}\n")

        # rel = write_reservoir_equations(eqs, M, rel, ind2, ind3)
        rel = write_reservoir_equations_with_isotopes(eqs, M, rel, ind2, ind3)

        sep = "# ---------------- bits and pieces --------------------------- #"
        eqs.write(f"\n{sep}\n" f"{ind2}return [\n")
        # Write all initial conditions that are recorded in icl
        for k, v in icl.items():
            eqs.write(f"{ind3}dCdt_{k.full_name.replace('.', '_')},  # {v[0]}\n")
            if k.isotopes:
                eqs.write(f"{ind3}dCdt_{k.full_name.replace('.', '_')}_l,  # {v[1]}\n")

        eqs.write(f"{ind2}]\n")

    return fqfn


def get_flux(flux: Flux, M: Model, R: list[float], icl: dict) -> tuple(str, str):
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

    elif c.ctype == "weathering":
        ex, exl = get_weathering_eq(flux, c, cfn, icl, ind2, ind3)

    elif c.ctype == "gas_exchange":  # Gasexchange
        if c.isotopes:
            ex = get_gas_exchange_w_isotopes_eq(flux, c, cfn, icl, ind2, ind3)
        else:
            ex = get_gas_exchange_eq(flux, c, cfn, icl, ind2, ind3)

    elif c.ctype == "ignore":
        pass  # do nothing

    else:
        raise ValueError(
            f"Connection type {c.ctype} for {c.full_name} is not implmented"
        )

    return ex, exl


def check_signal_2(ex: str, exl: str, c: Connection) -> (str, str):
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


def get_ic(r: Reservoir, icl: dict, isotopes=False) -> str:
    """Get initial condition in a reservoir.  If the reservoir is icl,
    return index expression into R.c. If the reservoir is not in the
    index, return the Reservoir concentration a t=0

    In both cases return these a string

    If isotopes == True, return the pointer to the light isotope
    concentration

    :param r: A reservoir handle
    :param icl: icl = dict[Reservoir, list[int, int]] where reservoir
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


def parse_esbmtk_return_data_types():
    pass


def parse_esbmtk_return_data_types_old(
    line: any, r: Reservoir, ind: str, icl: dict
) -> str:
    """Parse esbmtk data types that are provided as arguments
    to external function objects, and convert them into a suitable string
    format that can be used in the ode equation file
    """
    from esbmtk import GasReservoir, ReservoirGroup
    from .utility_functions import get_reservoir_reference

    # convert to object handle if need be
    if isinstance(line, str):
        o = getattr(r.register, line)
    elif isinstance(line, dict):
        k = next(iter(line))  # get first key
        v = line[k]
        # get reservoir and species handle
        r, sp = get_reservoir_reference(k, v, r.mo)
        if k[:2] == "F_":  # this is a flux
            if isinstance(r, (GasReservoir, Reservoir)):
                sr = f"{r.register.name}.{r.name}.{v}_F".replace(".", "_")
                #  sr = f"M.{r.register.name}_F".replace(".", "_")
                o = r
            elif isinstance(r, ReservoirGroup):
                sr = f"{r.full_name}.{sp.name}.{r.name}.{line[k]}_F".replace(".", "_")
                o = r
            else:
                raise ValueError(f"r = {type(r)}")

        elif k[:2] == "R_":  # this is reservoir or reservoirgroup
            if isinstance(r, (GasReservoir, Reservoir)):
                sr = f"{r.full_name}".replace(".", "_")
                o = r
            elif isinstance(r, ReservoirGroup):
                sr = f"{r.full_name}.{sp.name}".replace(".", "_")
                o = r
            else:
                raise ValueError(f"r = {type(r)}")

    elif isinstance(line, Flux):
        o = line
        sr = o.full_name.replace(".", "_")
    else:  # argument is a regular reservoir
        o = line
        sr = f'dCdt_{o.full_name.replace(".", "_")}'

    if isinstance(o, (Reservoir, GasReservoir, Flux)):
        isotopes = o.isotopes
    else:
        isotopes = getattr(o, sp.name).isotopes
    if isotopes:
        sr = f"{sr}, {sr}_l"

    return sr


def parse_esbmtk_input_data_types(d: any, r: Reservoir, ind: str, icl: dict) -> str:
    """Parse esbmtk data types that are provided as arguments
    to external function objects, and convert them into a suitable string
    format that can be used in the ode equation file
    """
    from esbmtk import Flux, Reservoir, ReservoirGroup, SeawaterConstants, Q_
    from esbmtk import GasReservoir

    if isinstance(d, str):
        sr = getattr(r.register, d)
        a = f"{ind}{get_ic(sr, icl)},\n"
    elif isinstance(d, Reservoir):
        a = f"{ind}{get_ic(d, icl,d.isotopes)},\n"
    elif isinstance(d, GasReservoir):
        a = f"{ind}{get_ic(d, icl,d.isotopes)},\n"
    elif isinstance(d, ReservoirGroup):
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
    r: Reservoir,
    icl: dict,
    rel: str,
    ind2: str,
    ind3: str,
    gpt: tuple,
) -> str:
    """Write external function call code

    :param eqs: equation file handle
    :param r: reservoir_handle
    :param icl: dict of reservoirs that have actual fluxes
    :param rel: string with reservoir names returned by setup_ode
    :param ind2: indent 2 times
    :param ind3: indent 3 times
    :param gpt: tuple with global paramaters

    :returns: rel: modied string of reservoir names
    """

    # get list of return values. This is a bit of hack, since we imply
    # anything that its not a flux, is a reservoir with no other inputs
    rv = ind2
    for o in r.lro:
        if isinstance(o, Flux):
            rv += f"{o.full_name.replace('.', '_')}, "
        else:
            rv += f"dCdt_{o.full_name.replace('.', '_')}, "

    rv = rv[:-2]

    a = ""
    # this can probably be simplified simiar to the old parse_return_values()
    for d in r.function_input_data:
        a += parse_esbmtk_input_data_types(d, r, ind3, icl)

    if r.function_params == "None":
        eqs.write(f"{rv} = {r.fname}(\n{a}{ind2})\n\n")
    else:
        s = f"gpt[{r.param_start}]"
        eqs.write(f"{rv} = {r.fname}(\n{a}{ind3}{s},\n{ind2})\n\n")
        # params = parse_function_params(r.function_params, ind3)
        # eqs.write(f"{rv} = {r.fname}(\n{a}{ind3}(\n{params}{ind3}),\n{ind2})\n\n")
    rel += f"{ind3}{rv},\n"

    return rel


def get_regular_flux_eq(
    flux: Flux,  # flux instance
    c: Connection,  # connection instance
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
    c: Connection,
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
        ratios in the source reservoir soncentrations (s_c, s_l), and alpha (a) as
        f_l = f_m * 1000/(r * (d + 1000) + 1000)

        Note that the scale has already been applaied to f_m in the calling function.
        """
        if c.delta != "None":
            d = c.delta
            eq = f"{f_m} * 1000 / ({r} * ({d} + 1000) + 1000)"
        elif c.alpha != "None":
            a = c.alpha / 1000 + 1
            eq = f"{s_l} * {f_m} / ({a} * {s_c} + {s_l} - {a} * {s_l})"
        else:
            eq = f"{f_m} * {s_l} / {s_c}"
    else:
        eq = ""

    return eq


def get_scale_with_concentration_eq(
    flux: Flux,  # flux instance
    c: Connection,  # connection instance
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
    :param icl: dict[Reservoir, list[int, int]] where reservoir
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
    c: Connection,  # connection instance
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
    :param icl: dict[Reservoir, list[int, int]] where reservoir
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
    delta|alpha according to isotope ratio of the reference flux
    """
    exl = check_isotope_effects(ex, c, icl, ind3, ind2)
    ex, exl = check_signal_2(ex, exl, c)
    return ex, exl


def get_weathering_eq(
    flux: Flux,  # flux instance
    c: Connection,  # connection instance
    cfn: str,  # full name of the connection instance
    icl: dict,  # list of initial conditions
    ind2: str,
    ind3: str,
) -> tuple(str, str):
    """Equation defining a flux that scales with the concentration of pcO2

    F = F_0 (pcO2/pCO2_0) * c, see Zeebe, 2012, doi:10.5194/gmd-5-149-2012

    Example::

     M1_C_Fw_to_L_b_DIC_Ca_W__F = (
            M1.C_Fw_to_L_b_DIC_Ca_W.rate
            * M1.C_Fw_to_L_b_DIC_Ca_W.scale
            * (R[10]/M1.C_Fw_to_L_b_DIC_Ca_W.pco2_0)
            **  M1.C_Fw_to_L_b_DIC_Ca_W.ex
        )

    :param flux: Flux object
    :param c: connection instance
    :param cfn: full name of the connection instance
    :param icl: dict[Reservoir, list[int, int]] where reservoir
        indicates the reservoir handle, and the list contains the
        index into the reservoir data.  list[0] = concentration
        list[1] concentration of the light isotope.

    :returns: two strings with the respective equations for the change
              in the total reservoir concentration and the
              concentration of the light isotope

    """

    from esbmtk import Source, Sink

    s_c = get_ic(c.reservoir_ref, icl)
    ex = (
        f"(\n{ind3}{cfn}.rate\n"
        f"{ind3}* {cfn}.scale\n"
        f"{ind3}* ({s_c}/{cfn}.pco2_0)\n"
        f"{ind3}**  {cfn}.ex\n"
        f"{ind2})"
    )

    if c.isotopes:
        # get isotope ratio of source
        if isinstance(c.source, (Source, Sink)):
            s_c = f"{c.source.full_name}.c"
            s_l = f"{c.source.full_name}.l"
        else:
            s_c, s_l = get_ic(c.source, icl, True).replace(" ", "").split(",")

        fn = flux.full_name.replace(".", "_")
        exl = f"{fn} * {s_l} / {s_c}"
        exl = f"{exl}  # weathering + isotopes"
    else:
        exl = ""

    ex, exl = check_signal_2(ex, exl, c)

    return ex, exl


def get_gas_exchange_eq(
    flux: Flux,  # flux instance
    c: Connect,  # connection instance
    cfn: str,  # full name of the connection instance
    icl: dict,  # list of initial conditions
    ind2: str,
    ind3: str,
) -> str:
    """Equation defining a flux that scales with the concentration of
    of the gas in the atmosphere versus the concentration of the gas
    in solution.  See Zeebe, 2012, doi:10.5194/gmd-5-149-2012

    ..Example::

        M1_C_H_b_to_CO2_At_gex_hb_F = gas_exchange_ode(
             M1.C_H_b_to_CO2_At.scale,
             R[10],  # pco2 in atmosphere
             M1.C_H_b_to_CO2_At.water_vapor_pressure,
             M1.C_H_b_to_CO2_At.solubility,
             M1.H_b.CO2aq, # [co2]aq
         )  # gas_exchange

    :param flux: Flux object
    :param c: connection instance
    :param cfn: full name of the connection instance
    :param icl: dict[Reservoir, list[int, int]] where reservoir
        indicates the reservoir handle, and the list contains the
        index into the reservoir data.  list[0] = concentration
        list[1] concentration of the light isotope.

    :returns: string with the gas exchange equation
    """

    # get co2_aq reference
    gas_sp = f"{get_ic(c.gas_reservoir, icl)}"
    ref_sp = f"{get_ic(c.ref_species, icl)}"

    ex = (
        f"gas_exchange_ode(\n"
        f"{ind3}toc[{c.s_index}],\n"
        f"{ind3}{gas_sp},\n"
        f"{ind3}toc[{c.vp_index}],\n"
        f"{ind3}toc[{c.solubility_index}],\n"
        f"{ind3}{ref_sp},\n"
        f"{ind2})"
    )
    ex, exl = check_signal_2(ex, "", c)
    ex = ex + "  # gas_exchange\n"

    return ex


def get_gas_exchange_w_isotopes_eq(
    flux: Flux,  # flux instance
    c: Connection,  # connection instance
    cfn: str,  # full name of the connection instance
    icl: dict,  # list of initial conditions
    ind2: str,
    ind3: str,
) -> tuple(str, str):
    """Equation defining a flux that scales with the concentration of
    of the gas in the atmosphere versus the concentration of the gas
    in solution.  See Zeebe, 2012, doi:10.5194/gmd-5-149-2012

    ..Example::

        M1_C_H_b_to_CO2_At_gex_hb_F, M1_C_H_b_to_CO2_At_gex_hb_F_l  = gas_exchange_ode(
           M1.C_H_b_to_CO2_At.scale, # surface area in m^2
           R[10],  # gas c in atmosphere
           R[11],  # gas c_l in atmosphere
           R[12],  # c of main species, e.g., DIC
           R[13],  # c_l of main species, e.g., DIC_12
           M1.C_H_b_to_CO2_At.water_vapor_pressure,
           M1.C_H_b_to_CO2_At.solubility,
           M1_H_b_CO2aq, # gas concentration in liquid, e.g., [co2]aq
           a_db,  # fractionation factor between dissolved CO2aq and HCO3-
           a_gb,  # fractionation between CO2g HCO3-
        )  # gas_exchange

    :param flux: Flux object
    :param c: connection instance
    :param cfn: full name of the connection instance
    :param icl: dict[Reservoir, list[int, int]] where reservoir
        indicates the reservoir handle, and the list contains the
        index into the reservoir data.  list[0] = concentration
        list[1] concentration of the light isotope.

    :returns: string with the gas exchange equation for the light
              isotope
    """

    # get isotope data
    pco2, pco2_l = get_ic(c.gas_reservoir, icl, True).replace(" ", "").split(",")
    dic, dic_l = get_ic(c.liquid_reservoir, icl, True).replace(" ", "").split(",")

    a_db = f"{c.liquid_reservoir.parent.full_name}.swc.a_db"
    a_dg = f"{c.liquid_reservoir.parent.full_name}.swc.a_dg"
    a_u = f"{c.liquid_reservoir.parent.full_name}.swc.a_u"

    # get co2_aq reference
    lrn = f"{c.liquid_reservoir.full_name}"
    sp = lrn.split(".")[-1]
    # test if we refer to CO2 or other gas species
    if sp == "DIC":
        # here we reference the dyanamically calculated CO2aq from cs1,
        # and not the vector field of cs1
        refsp = f"dCdt_{c.liquid_reservoir.parent.full_name}.CO2aq".replace(".", "_")
        # refsp = f"{c.liquid_reservoir.parent.full_name}.CO2aq"
    else:
        raise ValueError(f"Species{sp} has no isotope definition for gas exchange")

    ex = (
        f"gas_exchange_ode_with_isotopes(\n"
        f"{ind3}toc[{c.s_index}],\n"  # surface area in m^2
        f"{ind3}{pco2},\n"  # gas c in atmosphere
        f"{ind3}{pco2_l},\n"  # gas c_l in atmosphere
        f"{ind3}{dic},\n"  # c of reference species, e.g., DIC
        f"{ind3}{dic_l},\n"  # c_l reference species, e.g., DIC_12
        f"{ind3}toc[{c.vp_index}],\n"  # water_vapor_pressure,\n"
        f"{ind3}toc[{c.solubility_index}],\n"  # solubility
        f"{ind3}{refsp},\n"  # gas concentration in liquid, e.g., [co2]aq
        f"{ind3}{a_db},\n"  # fractionation factor between dissolved CO2aq and HCO3-
        f"{ind3}{a_dg},\n"  # fractionation between CO2aq and CO2g
        f"{ind3}{a_u},\n"  # kinetic fractionation during gas exchange
        f"{ind2})"
    )
    ex = ex + "  # gas_exchange\n"

    return ex
