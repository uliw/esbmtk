from __future__ import annotations
import typing as tp

if tp.TYPE_CHECKING:
    from esbmtk import Flux, Reservoir, Model, Connection, Connect


def get_initial_conditions(M: Model) -> tuple[list, dict, list, list]:
    """get list of initial conditions. THis list needs to match
    the number of equations. We need to consider 3 types reservoirs:

    1) Reservoirs that change as a result of physical fluxes
       i.e. r.lof > 0. These require a flux statements and a
       reservoir equation.

    2) Reservoirs that do not have active fluxes but are computed
       as a tracer, i.e.. HCO3. These only require a reservoir
       equation

    3) Reservoirs that do not change but are used as input. Those
       should not happen in a well formed model, but we cannot
       exclude the possibility. In this case, there is no flux
       equation, and we state that dR/dt = 0

    get_ic() will look up the index position of the reservoir_handle on
    icl, and then use this index to retrieve the correspinding value in R

    Isotopes are handled by adding a second entry

    """

    R = []  # list of initial conditions
    # dict that contains the reservoir_handle as key and the index positions
    # for c and l as a list
    icl: dict[Reservoir, list[int, int]] = {}
    cpl: list = []  # list of reservoirs that are computed
    ipl: list = []  # list of static reservoirs that serve as input

    i: int = 0
    for r in M.lic:
        # collect all reservoirs that have initial conditions
        if len(r.lof) > 0 or r.rtype == "computed" or r.rtype == "passive":
            if r.isotopes:
                R.append(r.c[0])  # add initial condition
                R.append(r.l[0])  # add initial condition for l
                icl.update({r: [i, i + 1]})  # record index position
                i += 2
            else:
                R.append(r.c[0])  # add initial condition for c
                icl.update({r: [i, i]})  # record index position
                i += 1

    return R, icl, cpl, ipl


def write_reservoir_equations(
    eqs,
    M: Model,
    rel: str,  # string with reservoir names used in return function
    ind2: str,  # indent 2 times
    ind3: str,  # indent 3 times
) -> str:  # updated list of reservoirs
    """Loop over reservoirs and their fluxes to build the reservoir equation"""

    from esbmtk import ReservoirGroup

    for r in M.lor:  # loop over reservoirs
        # Write equations for each reservoir
        # create unique variable names. Reservoirs are typiclally called
        # M.rg.r so we replace all dots with underscore

        name = f'{r.full_name.replace(".", "_")}'
        fex = ""

        # add all fluxes
        for flux in r.lof:  # check if in or outflux
            if flux.parent.source == r:
                sign = "-"
            elif flux.parent.sink == r:
                sign = "+"

            fname = f'{flux.full_name.replace(".", "_")}'
            fex = fex + f"{ind3}{sign} {fname}\n"

        # check if reservoir requires carbonate burial fluxes
        if isinstance(r.parent, ReservoirGroup):
            if r.parent.has_cs2:
                if r.species.name == "DIC" or r.species.name == "TA":
                    fn = f"{r.full_name}.burial".replace(".", "_")
                    fex = f"{fex}{ind3}- {fn}\n"

        # avoid reservoirs without active fluxes
        if len(r.lof) > 0:
            eqs.write(f"{ind2}{name} = (\n{fex}{ind2})/{r.full_name}.volume\n\n")
            rel = rel + f"{ind3}{name},\n"

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
                fex = fex + f"{ind3}{sign} {fname}\n"

            # check if reservoir requires carbonate burial fluxes
            if isinstance(r.parent, ReservoirGroup):
                if r.parent.has_cs2:
                    if r.species.name == "DIC":
                        fn = f"{r.full_name}.burial".replace(".", "_")
                        fn = f"{fn}_l"
                        fex = f"{fex}{ind3}- {fn}\n"

            # avoid reservoirs without active fluxes
            if len(r.lof) > 0:
                print(f"Adding: {name} to rel")
                eqs.write(f"{ind2}{name} = (\n{fex}{ind2})/{r.full_name}.volume\n\n")
                rel = rel + f"{ind3}{name},\n"

    return rel


def write_equations_2(
    M: Model, R: list[float], icl: dict, cpl: list, ipl: list
) -> tuple:
    """Write file that contains the ode-equations for M
    Returns the list R that contains the initial condition
    for each reservoir

    icl: dict of reservoirs that have actual fluxes
    cpl: list of reservoirs that hjave no fluxes but are computed based on other R's
    ipl: list of reservoir that do not change in concentration

    """
    from esbmtk import Model, ReservoirGroup
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

        sep = "# ---------------- write computed reservoir equations -------- #\n"
        sep = sep + "# that do not depend on fluxes"

        eqs.write(f"\n{sep}\n")

        for r in M.lpc_r:  # All virtual reservoirs need to be in this list

            if r.ftype == "cs1":
                rel = write_cs_1(eqs, r, icl, rel, ind2, ind3)
            elif r.ftype == "cs2":
                pass  # see below
            else:
                raise ValueError(f"{r.ftype} is undefined")

        flist = list()

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

                    else:
                        eqs.write(f"{ind2}{fn} = {ex}\n")

                else:  # all others types
                    eqs.write(f"{ind2}{fn} = {ex}\n")
                    if flux.parent.isotopes:  # add line for isotopes
                        eqs.write(f"{ind2}{fn}_l = {exl}\n")

        sep = "# ---------------- write computed reservoir equations -------- #\n"
        sep = sep + "# that do depend on fluxes"
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
    """Create formula expressions that describes the flux f
    return expression ex as string
    """

    ind2 = 8 * " "  # indentation
    ind3 = 12 * " "  # indentation

    ex = ""  # expression string
    exl = ""  # expression string for light isotope
    c = flux.parent  # shorthand for the connection object
    cfn = flux.parent.full_name  # shorthand for the connection object name

    if c.ctype.casefold() == "regular":
        ex, exl = get_regular_flux(flux, c, icl, ind2, ind3)

    elif c.ctype == "scale_with_concentration":
        ex, exl = get_scale_with_concentration(flux, c, cfn, icl)

    elif c.ctype == "scale_with_mass":
        ex, exl = get_scale_with_concentration(flux, c, cfn, icl)

    elif c.ctype == "scale_with_flux":
        ex, exl = get_scale_with_flux(flux, c, cfn, icl)

    elif c.ctype == "weathering":
        ex, exl = get_weathering(flux, c, cfn, icl, ind2, ind3)

    elif c.ctype == "gas_exchange":  # Gasexchange
        if c.isotopes:
            ex = get_gas_exchange_w_isotopes(flux, c, cfn, icl, ind2, ind3)
        else:
            ex = get_gas_exchange(flux, c, cfn, icl, ind2, ind3)
    else:
        raise ValueError(
            f"Connection type {c.ctype} for {c.full_name} is not implmented"
        )

    return ex, exl


def check_signal_2(ex: str, c: tp.union(Connection, Connect)) -> str:
    """Test if connection requires a signal"""

    sign = ""
    if c.signal != "None":
        # get signal type
        if c.signal.stype == "addition":
            sign = "+"
        elif c.signal.stype == "scale":
            sign = "*"
        else:
            raise ValueError(f"stype={c.signal.stype} not implemented")

        ex = ex + f" {sign} {c.signal.full_name}(t)[0]  # Signal"

    return ex


def get_ic(r: Reservoir, icl: dict, isotopes=False) -> str:
    """Get initial condition. If r in icl, return index
    expression into R. If r is not in index, return
    Reservoir concentration a t=0

    In both cases return these a string

    If the isotopes varibale is set to True, return the pointter to
    the light isotope reser instead

    icl: dict[Reservoir, [c, l]]
    """

    from esbmtk import Source, Sink

    s = ""

    # print(f"r = {r.full_name}")

    if r in icl:
        s = f"R[{icl[r][0]}]"
        if isotopes:
            s = f"R[{icl[r][1]}]"

    # sources&sinks are static, so not in list of initial conditions
    elif isinstance(r, (Source, Sink)):
        s = f"{r.full_name}.c"
        if isotopes:
            s = f"{r.full_name}.l"
    else:
        #
        raise ValueError(
            f"get_ic: can't find {r.full_name} in list of initial conditions"
        )

    return s


# ------------------------ define processes ------------------------- #


def write_cs_1(eqs, r: Reservoir, icl: dict, rel: str, ind2: str, ind3: str) -> list:
    """Write the python code that defines carbonate system 1"""

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
    rel = rel + f"{ind3}{fname},\n"

    return rel


def write_cs_2(eqs, r: Reservoir, icl: dict, rel: str, ind2: str, ind3: str) -> list:
    """Write the python code that defines carbonate system 2"""

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
    rel = rel + f"{ind3}{fname},\n"
    rel = rel + f"{ind3}{zname},\n"

    return rel


def get_regular_flux(
    flux: Flux,  # flux instance
    c: Connect,  # connection instance
    icl: dict,  # list of initial conditions
    ind2,  # indentation
    ind3,  # indentation
) -> tuple:
    """Equation defining a fixed rate flux
    Example:

    ex =  M1.C_Fw_to_CO2_At_DIC_volcanic_flux._F.rate
    exl = M1.C_Fw_to_CO2_At_DIC_volcanic_flux._F.rate

    """

    ex = f"{flux.full_name}.rate"
    ex = check_signal_2(ex, c)
    ex = ex + "  # fixed rate"

    if c.isotopes:
        # r, d and a are typically only a few decimal places so we use them
        # directly
        if c.delta != "None":
            r = c.source.species.r
            d = c.delta
            exl = (
                f"(\n"
                f"{ind3}{flux.full_name}.rate * 1e3\n"
                f"{ind3}/({r} * ({d} + 1000) + 1000)\n"
                f"{ind2})"
            )
            exl = exl + "  # fixed rate with delta"

        elif c.alpha != "None":
            if c.alpha > 1.1 or c.alpha < 0.9:
                raise ValueError(
                    "alpha needs to be given as fractional value not in permil"
                )
            r = c.source.species.r
            a = c.alpha
            s_c = get_ic(c.source, icl)
            s_l = get_ic(c.source, icl, isotopes=True)

            exl = (
                f"(\n"
                f"{ind3} - {s_l} * {s_c}\n"
                f"{ind3}/({a} * {s_l} - {a} * {s_c} - {s_l})"
                f"{ind2})"
            )
            exl = exl + "  # fixed rate with alpha"
        else:
            raise ValueError(
                "A regular flux in an isotope system must specify"
                "either delta, or alpha. Otherwise use a connection"
                "that depends on the upstream reservoir delta, or"
                "set the isotope flag to False"
            )
        # this will probably not work
        exl = check_signal_2(exl, c)

    else:
        exl = ""

    return ex, exl


def get_scale_with_concentration(
    flux: Flux,  # flux instance
    c: Connect,  # connection instance
    cfn: str,  # full name of the connection instance
    icl: dict,  # list of initial conditions
) -> tuple(str, str):
    """Equation defining a flux that scales with the concentration in the upstream
    reservoir

    Example:

    M1_CG_D_b_to_L_b_TA_thc__F = M1.CG_D_b_to_L_b.TA_thc.scale * R[5]
    """

    s_c = get_ic(c.source, icl)  # get index to concentration
    ex = f"{cfn}.scale * {s_c}"  # {c.id} scale with conc in {c.source.full_name}"
    ex = check_signal_2(ex, c)
    ex = ex + "  # scale with concentration"

    if c.isotopes:
        # get c of light isotope in source
        s_l = get_ic(c.source, icl, isotopes=True)  # R[x]
        exl = f"{cfn}.scale * {s_l}"
        exl = check_signal_2(exl, c)
        exl = exl + "  # scale with concentration"
    else:
        exl = ""

    return ex, exl


def get_scale_with_mass(
    flux: Flux,  # flux instance
    c: Connect,  # connection instance
    cfn: str,  # full name of the connection instance
    icl: dict,  # list of initial conditions
) -> tuple(str, str):
    """Equation defining a flux that scales with the concentration in the upstream
    reservoir

    Example:

    M1_CG_D_b_to_L_b_TA_thc__F = M1.CG_D_b_to_L_b.TA_thc.scale * R[5]
    """

    s_c = get_ic(c.source, icl)  # get index to concentration
    ex = f"{cfn}.scale * {c.source.full_name}.volume * {s_c}"
    ex = check_signal_2(ex, c)
    ex = ex + "  # scale with mass"
    if c.isotopes:
        s_l = get_ic(c.source, icl, isotopes=True)  # get index to concentration
        exl = f"{cfn}.scale * {c.source.full_name}.volume * {s_l}"
        exl = check_signal_2(exl, c)
        exl = exl + "  # scale with mass"
    else:
        exl = ""

    return ex, exl


def get_scale_with_flux(
    flux: Flux,  # flux instance
    c: Connect,  # connection instance
    cfn: str,  # full name of the connection instance
    icl: dict,  # list of initial conditions
) -> tuple(str, str):
    """Equation defining a flux that scales with strength of another flux
    reservoir

    Example:

    M1_C_Fw_to_L_b_TA_wca_ta__F = M1.C_Fw_to_L_b_TA_wca_ta.scale * M1_C_Fw_to_L_b_DIC_Ca_W__F
    """

    p = flux.parent.ref_flux.parent
    fn = f"{p.full_name.replace('.', '_')}__F"
    ex = f"{cfn}.scale * {fn}"
    ex = check_signal_2(ex, c)
    ex = ex + "  # scale with flux"

    """ The flux for the light isotope will computed as follows:
    We will use the mass of the flux we or scaling, but that we will set the
    delta according to reservoir this flux derives from
    """

    if c.isotopes:
        s_c = get_ic(c.source, icl)
        s_l = get_ic(c.source, icl, isotopes=True)
        exl = f"(\n" f"{cfn}.scale\n" f"* {fn}\n" f"* {s_l} / {s_c}\n" f")\n"
        # exl = check_signal_2(exl, c)
        exl = exl + "  # scale with flux"
    else:
        exl = ""

    return ex, exl


def get_weathering(
    flux: Flux,  # flux instance
    c: Connect,  # connection instance
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
    ex = check_signal_2(ex, c)
    ex = ex + "  # weathering"

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
        exl = exl + "  # weathering + isotopes"
    else:
        exl = ""

    return ex, exl


def get_gas_exchange(
    flux: Flux,  # flux instance
    c: Connect,  # connection instance
    cfn: str,  # full name of the connection instance
    icl: dict,  # list of initial conditions
    ind2: str,
    ind3: str,
) -> tuple(str, str):
    """Equation defining a flux that scales with the concentration of
    of the gas in the atmosphere versus the concentration of the gas
    in solution. See Zeebe, 2012, doi:10.5194/gmd-5-149-2012

    M1_C_H_b_to_CO2_At_gex_hb_F = gas_exchange_ode(
            M1.C_H_b_to_CO2_At.scale,
            R[10],  # pco2 in atmosphere
            M1.C_H_b_to_CO2_At.water_vapor_pressure,
            M1.C_H_b_to_CO2_At.solubility,
            M1_H_b_CO2aq, # [co2]aq
            )  # gas_exchange

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
    ex = check_signal_2(ex, c)
    ex = ex + "  # gas_exchange\n"

    return ex


def get_gas_exchange_w_isotopes(
    flux: Flux,  # flux instance
    c: Connect,  # connection instance
    cfn: str,  # full name of the connection instance
    icl: dict,  # list of initial conditions
    ind2: str,
    ind3: str,
) -> tuple(str, str):
    """Equation defining a flux that scales with the concentration of
    of the gas in the atmosphere versus the concentration of the gas
    in solution. See Zeebe, 2012, doi:10.5194/gmd-5-149-2012

    M1_C_H_b_to_CO2_At_gex_hb_F, M1_C_H_b_to_CO2_At_gex_hb_F_l  = gas_exchange_ode(
            M1.C_H_b_to_CO2_At.scale, # surface area in m^2
            R[10],  # gas c in atmosphere
            R[11],  # gas c_l in atmosphere
            R[12],  # c of reference species, e.g., DIC
            R[13],  # c_l reference species, e.g., DIC_12
            M1.C_H_b_to_CO2_At.water_vapor_pressure,
            M1.C_H_b_to_CO2_At.solubility,
            M1_H_b_CO2aq, # gas concentration in liquid, e.g., [co2]aq
            a_db,  # fractionation factor between dissolved CO2aq and HCO3-
            a_gb,  # fractionation between CO2g HCO3-
            )  # gas_exchange

     source_l = get_ic(sb_DIC, icl, isotopes=True)
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
    elif sp == "O2":
        s_c = get_ic(c.liquid_reservoir, icl)
        refsp = f"{s_c}"
    else:
        raise ValueError(f"Species{sp} has not definition for gex")

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
