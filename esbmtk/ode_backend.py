from __future__ import annotations
import typing as tp

if tp.TYPE_CHECKING:
    from esbmtk import Flux, Reservoir, Model, Connection, Connect, numpy


def get_initial_conditions(
    M: Model,
    rtol: float,
    atol_d: float = 1e-6,
) -> tuple[list, dict, list, list, numpy.array]:
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

    from esbmtk import ReservoirGroup

    for r in M.lor:  # loop over reservoirs
        """
        Write equations for each reservoir and create unique variable
        names.  Reservoirs are typiclally called M.rg.r so we replace
        all dots with underscore -> M_rg_r
        """

        name = f'{r.full_name.replace(".", "_")}'
        fex = ""

        # add all fluxes
        for flux in r.lof:  # check if in or outflux
            if flux.parent.source == r:
                sign = "-"
            elif flux.parent.sink == r:
                sign = "+"

            fname = f'{flux.full_name.replace(".", "_")}'
            fex = f"{fex}{ind3}{sign} {fname}\n"

        if (  # check if reservoir requires carbonate burial fluxes
            isinstance(r.parent, ReservoirGroup)
            and r.parent.has_cs2
            and r.species.name in ["DIC", "TA"]
        ):
            fn = f"{r.full_name}.burial".replace(".", "_")
            fex = f"{fex}{ind3}- {fn}\n"

        if len(r.lof) > 0:  # avoid reservoirs without active fluxes
            eqs.write(f"{ind2}{name} = (\n{fex}{ind2})/{r.full_name}.volume\n\n")
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

    from esbmtk import ReservoirGroup

    for r in M.lor:  # loop over reservoirs
        # Write equations for each reservoir
        # create unique variable names. Reservoirs are typiclally called
        # M.rg.r so we replace all dots with underscore
        if r.isotopes:
            name = f'{r.full_name.replace(".", "_")}_l'
            fex = ""
            # add all fluxes
            for flux in r.lof:  # check if in or outflux
                if flux.parent.source == r:
                    sign = "-"
                elif flux.parent.sink == r:
                    sign = "+"

                fname = f'{flux.full_name.replace(".", "_")}_l'
                fex = f"{fex}{ind3}{sign} {fname}\n"

            # check if reservoir requires carbonate burial fluxes
            if (
                isinstance(r.parent, ReservoirGroup)
                and r.parent.has_cs2
                and r.species.name == "DIC"
            ):
                fn = f"{r.full_name}.burial".replace(".", "_")
                fn = f"{fn}_l"
                fex = f"{fex}{ind3}- {fn}\n"

            # avoid reservoirs without active fluxes
            if len(r.lof) > 0:
                # print(f"Adding: {name} to rel")
                eqs.write(f"{ind2}{name} = (\n{fex}{ind2})/{r.full_name}.volume\n\n")
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
    header = """from __future__ import annotations\n\n
import typing as tp

if tp.TYPE_CHECKING:
    from esbmtk import Model


class setup_ode():
    '''Class stub to enable state in the equation system passed to ODEINT
    '''

    from esbmtk import Model, ReservoirGroup

    def __init__(self, M: Model)->None:
        ''' Use this method to initialize all variables that require the state
            t-1
        '''
        import numpy as np

        self.i = 0
        self.t = []
        self.last_t = 0

    def eqs(self, t, R: list, M: Model) -> list:
        '''Auto generated esbmtk equations do not edit
        '''

        from esbmtk import carbonate_system_1_ode, carbonate_system_2_ode
        from esbmtk import gas_exchange_ode, gas_exchange_ode_with_isotopes

        max_i = len(M.time)-1

        # flux equations
"""

    from esbmtk import AirSeaExchange

    # """
    # write file
    with open(fqfn, "w", encoding="utf-8") as eqs:
        rel = ""  # list of return values
        # ind1 = 4 * " "
        ind2 = 8 * " "  # indention
        ind3 = 12 * " "  # indention
        eqs.write(header)
        eqs.write(f"{ind2}{M.name} = M\n")
        sep = (
            "# ---------------- write computed reservoir equations -------- #\n"
            + "# that do not depend on fluxes"
        )
        eqs.write(f"\n{sep}\n")

        for r in M.lpc_r:  # All virtual reservoirs need to be in this list
            if r.ftype == "cs1":
                rel = write_cs_1(eqs, r, icl, rel, ind2, ind3)
            elif r.ftype != "cs2":
                raise ValueError(f"{r.ftype} is undefined")

        flist = []
        sep = "# ---------------- write all flux equations ------------------- #"
        eqs.write(f"\n{sep}\n")
        for flux in M.lof:  # loop over fluxes
            # fluxes belong to at least 2 reservoirs, so we need to avoid duplication
            # we cannot use a set, since we need to preserve order
            if flux not in flist:
                flist.append(flux)  # add to list of fluxes already computed
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

                if flux.save_flux_data:
                    eqs.write(f"{ind2}{flux.full_name}.m[self.i] = {fn}\n")
                    if flux.parent.isotopes:
                        eqs.write(f"{ind2}{flux.full_name}.l[self.i] = {fn}_l\n")

        sep = (
            "# ---------------- write computed reservoir equations -------- #\n"
            + "# that do depend on fluxes"
        )

        eqs.write(f"\n{sep}\n")

        for r in M.lpc_r:  # All virtual reservoirs need to be in this list
            if r.ftype == "cs1":
                pass  # see above
            elif r.ftype == "cs2":  #
                rel = write_cs_2(eqs, r, icl, rel, ind2, ind3)
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
        eqs.write(
            f"\n{sep}\n"
            f"{ind2}self.i += 1\n"
            f"{ind2}self.t.extend([t])\n"
            f"{ind2}self.last_t = t\n"
            f"{ind2}return [\n"
        )
        # Write all initial conditions that are recorded in icl
        for k, v in icl.items():
            eqs.write(f"{ind3}{k.full_name.replace('.', '_')},  # {v[0]}\n")
            if k.isotopes:
                eqs.write(f"{ind3}{k.full_name.replace('.', '_')}_l,  # {v[1]}\n")

        eqs.write(f"{ind2}]\n")
        # if len(R) != len(rel.split(",")) - 1:
        #     raise ValueError(
        #         f"number of initial conditions ({len(R)})"
        #         f"does not match number of return values"
        #         f"({len(rel.split(','))-1}')\n\n"
        #     )
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
    else:
        raise ValueError(
            f"Connection type {c.ctype} for {c.full_name} is not implmented"
        )

    return ex, exl


def check_signal_2(ex: str, exl: str, c: Connection | Connect) -> (str, str):
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
    :param isotopes: whether to return the total mass or the mass of
        the light isotope

    :raises ValueError: get_ic: can't find {r.full_name} in list of
        initial conditions

    :returns: the string s which is the full_name of the reservoir
              concentration or isotope concentration
    """
    from esbmtk import Source, Sink

    s = ""

    if r in icl:
        s = f"R[{icl[r][1]}]" if isotopes else f"R[{icl[r][0]}]"
    elif isinstance(r, (Source, Sink)):
        s = f"{r.full_name}.c"
        if r.isotopes:
            s = f"{r.full_name}.l"
    else:
        #
        raise ValueError(
            f"get_ic: can't find {r.full_name} in list of initial conditions"
        )

    return s


# ------------------------ define processes ------------------------- #


def write_cs_1(eqs, r: Reservoir, icl: dict, rel: str, ind2: str, ind3: str) -> str:
    """Write the python code that defines carbonate system 1, add the
    name of the reservoir carbonate system to the string of reservoir
    names in rel

    :param eqs: equation file handle
    :param r: reservoir_handle
    :param icl: dict of reservoirs that have actual fluxes
    :param rel: string with reservoir names returned by setup_ode
    :param ind2: indent 2 times
    :param ind3: indent 3 times

    :returns: rel: modied string of reservoir names
    """
    fname = f"{r.parent.full_name}.Hplus".replace(".", "_")
    cname = f"{r.parent.full_name}.CO2aq".replace(".", "_")

    eqs.write(
        f"{ind2}{fname}, {cname} = carbonate_system_1_ode(\n"
        f"{ind3}{r.parent.full_name},\n"
        f"{ind3}{get_ic(r.parent.DIC, icl)},\n"
        f"{ind3}{get_ic(r.parent.TA, icl)},\n"
        f"{ind3}{get_ic(r.parent.Hplus, icl)},\n"
        f"{ind3}self.i,\n"
        f"{ind3}max_i)  # cs1\n"
    )
    rel += f"{ind3}{fname},\n"

    return rel


def write_cs_2(eqs, r: Reservoir, icl: dict, rel: str, ind2: str, ind3: str) -> str:
    """Write the python code that defines carbonate system 2, add the
    name of the reservoir carbonate system to the string of reservoir
    names in rel

    :param eqs: equation file handle
    :param r: reservoir_handle
    :param icl: dict of reservoirs that have actual fluxes
    :param rel: string with reservoir names returned by setup_ode
    :param ind2: indent 2 times
    :param ind3: indent 3 times

    :returns: rel: modied string of reservoir names
    """

    influx = r.parent.cs.ref_flux[0].full_name.replace(".", "_")

    # get DIC reservoir of the surface box
    sb_DIC = getattr(r.r_s, "DIC")

    if r.register.DIC.isotopes:
        source_m = get_ic(sb_DIC, icl)
        source_l = get_ic(sb_DIC, icl, isotopes=True)
    else:
        # print(f"sb_DIC = {sb_DIC.full_name}")
        source_m = get_ic(sb_DIC, icl)
        source_l = 1

    fn_dic = f"{r.register.DIC.full_name}.burial".replace(".", "_")
    fn_dic_l = f"{fn_dic}_l"
    fn_ta = f"{r.register.TA.full_name}.burial".replace(".", "_")

    fname = f"{r.parent.full_name}.Hplus".replace(".", "_")
    zname = f"{r.parent.full_name}.zsnow".replace(".", "_")
    zmax = r.parent.cs.function_params[15]
    eqs.write(
        f"{ind2}{fn_dic}, {fn_dic_l}, {fname}, {zname} = carbonate_system_2_ode(\n"
        f"{ind3}t,  # 1: current time\n"
        f"{ind3}{r.parent.full_name},  # 2: Reservoirgroup handle\n"
        f"{ind3}{influx},  # 3: CaCO3 export flux\n"
        f"{ind3}{get_ic(r.parent.DIC, icl)},  # 4: DIC in the deep box\n"
        f"{ind3}{get_ic(r.parent.TA, icl)},  # 5: TA in the deep box\n"
        f"{ind3}{source_m},  # 6: DIC in the surface box\n"
        f"{ind3}{source_l},  # 7: DIC of the light isotope in the surface box\n"
        f"{ind3}{get_ic(r.parent.Hplus, icl)},  # 8: H+ in the deep box at t-1\n"
        f"{ind3}{get_ic(r.parent.zsnow, icl)},  # 9: zsnow in mbsl at t-1\n"
        f"{ind3}self.i,  # 10: current index\n"
        f"{ind3}max_i,   # 11: max is of vr data fields\n"
        f"{ind3}self.last_t, # 12: t at t-1\n"
        f"{ind2})  # cs2\n"
        # calculate the TA dissolution flux from DIc diss flux
        f"{ind2}{fn_ta} = {fn_dic} * 2  # cs2\n"
        f"{ind2} # Limit zsnow >= zmax\n"
        f"{ind2}if {get_ic(r.parent.zsnow, icl)} > {zmax}:"
        f" {get_ic(r.parent.zsnow, icl)} = {zmax}\n"
    )
    # add Hplus to the list of return values
    rel += f"{ind3}{fname},\n"
    rel = f"{rel}{ind3}{zname},\n"

    return rel


def get_regular_flux_eq(
    flux: Flux,  # flux instance
    c: Connect | Connection,  # connection instance
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
    ex = f"{flux.full_name}.rate"  # get flux rate string
    exl = check_isotope_effects(ex, c, icl, ind3, ind2)
    ex, exl = check_signal_2(ex, exl, c)  # check if we hav to add a signal

    return ex, exl


def check_isotope_effects(
    f_m: str,
    c: Connection | Connect,
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
        s_c = get_ic(c.source, icl)
        s_l = get_ic(c.source, icl, isotopes=True)
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
    c: Connect | Connection,  # connection instance
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
    s_c = get_ic(c.source, icl)  # get index to concentration
    ex = f"{cfn}.scale * {s_c}"  # {c.id} scale with conc in {c.source.full_name}"
    exl = check_isotope_effects(ex, c, icl, ind3, ind2)
    ex, exl = check_signal_2(ex, exl, c)
    return ex, exl


def get_scale_with_flux_eq(
    flux: Flux,  # flux instance
    c: Connect | Connection,  # connection instance
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
    ex = f"{cfn}.scale * {fn}"
    """ The flux for the light isotope will computed as follows:
    We will use the mass of the flux we or scaling, but that we will set the
    delta|alpha according to isotope ratio of the reference flux
    """
    exl = check_isotope_effects(ex, c, icl, ind3, ind2)
    ex, exl = check_signal_2(ex, exl, c)
    return ex, exl


def get_weathering_eq(
    flux: Flux,  # flux instance
    c: Connect | Connection,  # connection instance
    cfn: str,  # full name of the connection instance
    icl: dict,  # list of initial conditions
    ind2: str,
    ind3: str,
) -> tuple(str, str):
    """Equation defining a flux that scales with the concentration of pcO2

    F = F_0 (pcO2/pCO2_0) * c, see Zeebe, 2012, doi:10.5194/gmd-5-149-2012

    Example:

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
            s_c = get_ic(c.source, icl)  # get index to concentration
            s_l = get_ic(c.source, icl, isotopes=True)  # get index to concentration l

        fn = flux.full_name.replace(".", "_")
        exl = f"{fn} * {s_l} / {s_c}"
        exl = f"{exl}  # weathering + isotopes"

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
             M1_H_b_CO2aq, # [co2]aq
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
    lrn = f"{c.liquid_reservoir.full_name}"
    sp = lrn.split(".")[-1]
    # test if we refer to CO2 or other gas species
    if sp == "DIC":
        refsp = f"{c.liquid_reservoir.parent.full_name}.CO2aq".replace(".", "_")
    elif sp == "O2":
        s_c = get_ic(c.liquid_reservoir, icl)
        refsp = f"{s_c}"
    else:
        raise ValueError(f"Species{sp} has not definition for gex")

    # get atmosphere pcO2 reference
    pco2 = get_ic(c.gas_reservoir, icl)
    ex = (
        f"gas_exchange_ode(\n"
        f"{ind3}{cfn}.scale,\n"
        f"{ind3}{pco2},\n"
        f"{ind3}{cfn}.water_vapor_pressure,\n"
        f"{ind3}{cfn}.solubility,\n"
        f"{ind3}{refsp},\n"
        f"{ind2})"
    )
    ex, exl = check_signal_2(ex, "", c)
    ex = ex + "  # gas_exchange\n"

    return ex


def get_gas_exchange_w_isotopes_eq(
    flux: Flux,  # flux instance
    c: Connect | Connection,  # connection instance
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
    pco2 = get_ic(c.gas_reservoir, icl)
    pco2_l = get_ic(c.gas_reservoir, icl, isotopes=True)
    dic = get_ic(c.liquid_reservoir, icl)
    dic_l = get_ic(c.liquid_reservoir, icl, isotopes=True)

    a_db = f"{c.liquid_reservoir.parent.full_name}.swc.a_db"
    a_dg = f"{c.liquid_reservoir.parent.full_name}.swc.a_dg"
    a_u = f"{c.liquid_reservoir.parent.full_name}.swc.a_u"

    # get co2_aq reference
    lrn = f"{c.liquid_reservoir.full_name}"
    sp = lrn.split(".")[-1]
    # test if we refer to CO2 or other gas species
    if sp == "DIC":
        refsp = f"{c.liquid_reservoir.parent.full_name}.CO2aq".replace(".", "_")
    else:
        raise ValueError(f"Species{sp} has no isotope definition for gas exchange")

    ex = (
        f"gas_exchange_ode_with_isotopes(\n"
        f"{ind3}{cfn}.scale,\n"  # surface area in m^2
        f"{ind3}{pco2},\n"  # gas c in atmosphere
        f"{ind3}{pco2_l},\n"  # gas c_l in atmosphere
        f"{ind3}{dic},\n"  # c of reference species, e.g., DIC
        f"{ind3}{dic_l},\n"  # c_l reference species, e.g., DIC_12
        f"{ind3}{cfn}.water_vapor_pressure,\n"
        f"{ind3}{cfn}.solubility,\n"
        f"{ind3}{refsp},\n"  # gas concentration in liquid, e.g., [co2]aq
        f"{ind3}{a_db},\n"  # fractionation factor between dissolved CO2aq and HCO3-
        f"{ind3}{a_dg},\n"  # fractionation between CO2aq and CO2g
        f"{ind3}{a_u},\n"  # kinetic fractionation during gas exchange
        f"{ind2})"
    )
    ex = ex + "  # gas_exchange\n"

    return ex