
def execute_e(model, lop, lor, lpc_f, lpc_r):
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
        d = virtual reservoir list
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
                model.a,  # reservoir list
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
                model.a,  # reservoir list
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
# @njit(parallel=False, fastmath=True, error_model="numpy")
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
        for _ in enumerate(fn):
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


# @njit(parallel=False, fastmath=True, error_model="numpy")
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


# @njit(parallel=False, fastmath=True, error_model="numpy")
def foo_no_vr(fn, da, pc, a, b, c, d, e, maxt, dt):
    """Same as foo but no virtual reservoirs present

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
        # loop over processes
        j = 0
        for _ in enumerate(fn):
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


# @njit(parallel=False, fastmath=True, error_model="numpy")
def foo_no_vr_no_p(fn, a, b, c, d, e, maxt, dt):
    """Same as foo but no virtual reservoirs present

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
    # calculate the resulting reservoir concentrations
    # summarize_fluxes(a, b, c, d, e, i, dt)
    r_steps: int = len(b)
    for _ in maxt:
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

    print("Building Process List")

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


def apply_fractionation(
    f_m: str, c: Connection | Connect, icl: dict, ind3: str, ind2: str
) -> str:
    """If the connection involves isotope fractionation, add equation
    string that calculates the fractionation effect.

    :param f_m: string with the flux name
    :param c: connection object
    :param icl: dict of reservoirs that have actual fluxes
    :param ind2: indent 2 times
    :param ind3: indent 3 times

    :raises ValueError: if alpha is not between 0.9 and 1.1

    :returns eq: string with the fractionation equation

    Calculate the flux of the light isotope (f_l) as a function of the isotope
    ratios in the source reservoir soncentrations (s_c, s_l), and alpha (a) as

    f_l = s_l * f_m/ (a * s_c + s_l - a * s_l)
    """

    a = c.alpha / 1000 + 1
    s_c = get_ic(c.source, icl)
    s_l = get_ic(c.source, icl, isotopes=True)

    if int(c.alpha) != 0:
        eq = f"(\n{ind3} {s_l} * {f_m}\n{ind3} / ({a} * {s_c} + {s_l} - {a} * {s_l}))"
    else:
        eq = f"(\n{ind3} {f_m} * {s_l} / {s_c})"
    return eq


def apply_delta(
    f_m: str, c: Connection | Connect, icl: dict, ind3: str, ind2: str
) -> str:
    """If the connection involves a fixed deltae offset add equation
    string according to the delta value

    :param f_m: string with the flux name
    :param c: connection object
    :param icl: dict of reservoirs that have actual fluxes
    :param ind2: indent 2 times
    :param ind3: indent 3 times

    :raises ValueError: if alpha is not between 0.9 and 1.1

    :returns eq: string with the fractionation equation

    Calculate the flux of the light isotope (f_l) as a function of the isotope
    ratios in the source reservoir soncentrations (s_c, s_l), and alpha (a) as

    f_l = f_m * 1000/(r * (d + 1000) + 1000)
    """
    r = c.source.species.r
    d = c.delta
    exl = f"(\n" f"{ind3}{f_m} * 1000\n" f"{ind3} / ({r} * ({d} + 1000) + 1000))\n"
    return exl


# @njit(parallel=False, fastmath=True, error_model="numpy")
def calc_carbonates_1(i: int, input_data: List, vr_data: List, params: List) -> None:
    """Calculates and returns the carbonate concentrations and saturation state
     at the ith time-step of the model.

    The function assumes that vr_data will be in the following order:
        [H+, CA, HCO3, CO3, CO2(aq), omega]

    LIMITATIONS:
    - This in used in conjunction with ExternalCode objects!
    - Assumes all concentrations are in mol/kg
    - Assumes your Model is in mol/kg ! Otherwise, DIC and TA updating will not
    be correct.

    Calculations are based off equations from:
    Boudreau et al., 2010, https://doi.org/10.1029/2009GB003654
    Follows, 2006, doi:10.1016/j.ocemod.2005.05.004

    See add_carbonate_system_1 in utility_functions.py on how to call this function

    Author: M. Niazi & T. Tsan, 2021
    """

    k1 = params[0]  # K1
    k2 = params[1]  # K2
    KW = params[2]  # KW
    KB = params[3]  # KB
    boron = params[4]  # boron
    ca2 = params[5]  # Ca2+
    ksp = params[6]  # Ksp

    dic: float = input_data[0][i - 1]
    ta: float = input_data[1][i - 1]
    hplus: float = vr_data[0][i - 1]

    # calculates carbonate alkalinity (ca) based on H+ concentration from the
    # previous time-step
    oh: float = KW / hplus
    boh4: float = boron * KB / (hplus + KB)
    fg: float = hplus - oh - boh4
    ca: float = ta + fg

    # hplus
    gamm: float = dic / ca
    dummy: float = (1 - gamm) * (1 - gamm) * k1 * k1 - 4 * k1 * k2 * (1 - (2 * gamm))

    hplus: float = 0.5 * ((gamm - 1) * k1 + (dummy**0.5))
    # hco3 and co3
    """ Since CA = [hco3] + 2[co3], can the below expression can be simplified
    """
    # co3: float = dic / (1 + (hplus / k2) + ((hplus ** 2) / (k1 * k2)))
    hco3: float = dic / (1 + (hplus / k1) + (k2 / hplus))
    co3: float = (ca - hco3) / 2
    # co2 (aq)
    """DIC = hco3 + co3 + co2 + H2CO3 The last term is however rather
    small, so it may be ok to simply write co2aq = dic - hco3 + co3.
    Let's test this once we have a case where pco2 is calculated from co2aq
    """
    #  co2aq: float = dic / (1 + (k1 / hplus) + (k1 * k2 / (hplus ** 2)))
    co2aq: float = dic - hco3 - co3
    # omega: float = ca2 * co3 / ksp

    vr_data[0][i] = hplus
    vr_data[1][i] = ca
    vr_data[2][i] = hco3
    vr_data[3][i] = co3
    vr_data[4][i] = co2aq
    # vr_data[5][i] = oh
    # vr_data[5][i] = boh4
    # vr_data[5][i] = omega


# @njit(fastmath=True, error_model="numpy")
def calc_carbonates_2(i: int, input_data: List, vr_data: List, params: List) -> None:
    """Calculates and returns the carbonate concentrations and carbonate compensation
    depth (zcc) at the ith time-step of the model.

    The function assumes that vr_data will be in the following order:
        [H+, CA, HCO3, CO3, CO2(aq), zsat, zcc, zsnow, Fburial,
        B, BNS, BDS_under, BDS_resp, BDS, BCC, BPDC, BD,omega]

    LIMITATIONS:
    - This in used in conjunction with ExternalCode objects!
    - Assumes all concentrations are in mol/kg
    - Assumes your Model is in mol/kg ! Otherwise, DIC and TA updating will not
    be correct.

    Calculations are based off equations from:
    Boudreau et al., 2010, https://doi.org/10.1029/2009GB003654
    Follows, 2006, doi:10.1016/j.ocemod.2005.05.004

    See add_carbonate_system_2 in utility_functions.py on how to call this function.
    The input data is a follows

        reservoir DIC.m,  # 0
        reservoir DIC.l,  # 1
        reservoir DIC.c,  # 2
        reservoir TA.m,  # 3 TA mass
        reservoir.TA.c,  # 4 TA concentration
        Export_flux.fa,  # 5
        area_table,  # 6
        area_dz_table,  # 7
        Csat_table,  # 8
        reservoir.DIC.v,  # 9 reservoir volume

    Author: M. Niazi & T. Tsan, 2021
    """

    # Parameters
    k1 = params[0]
    k2 = params[1]
    KW = params[2]
    KB = params[3]
    boron = params[4]
    ksp0 = params[5]
    kc = params[6]
    AD = params[8]
    zsat0 = int(abs(params[9]))
    ca2 = params[10]
    dt = params[11]
    I_caco3 = params[12]
    alpha = params[13]
    zsat_min = int(abs(params[14]))
    zmax = int(abs(params[15]))
    z0 = int(abs(params[16]))

    # Data
    dic: float = input_data[2][i - 1]  # DIC concentration [mol/kg]
    ta: float = input_data[4][i - 1]  # TA concentration [mol/kg]
    Bm: float = input_data[5][0]  # Carbonate Export Flux [mol/yr]
    B12: float = input_data[5][1]  # Carbonate Export Flux light isotope
    v: float = input_data[9][i - 1]  # volume
    # lookup tables
    depth_area_table: np.ndarray = input_data[6]  # depth look-up table
    area_dz_table: np.ndarray = input_data[7]  # area_dz table
    Csat_table: np.ndarray = input_data[8]  # Csat table

    # vr_data
    hplus: float = vr_data[0][i - 1]  # H+ concentration [mol/kg]
    zsnow = vr_data[7][i - 1]  # previous zsnow

    # calc carbonate alkalinity based t-1
    oh: float = KW / hplus
    boh4: float = boron * KB / (hplus + KB)
    fg: float = hplus - oh - boh4
    ca: float = ta + fg

    # calculate carbon speciation
    # The following equations are after Follows et al. 2006
    gamm: float = dic / ca
    dummy: float = (1 - gamm) * (1 - gamm) * k1 * k1 - 4 * k1 * k2 * (1 - (2 * gamm))

    hplus: float = 0.5 * ((gamm - 1) * k1 + (dummy**0.5))
    # co3: float = dic / (1 + (hplus / k2) + ((hplus ** 2) / (k1 * k2)))
    hco3: float = dic / (1 + (hplus / k1) + (k2 / hplus))
    co3: float = (ca - hco3) / 2
    # DIC = hco3 + co3 + co2 + H2CO3 The last term is however rather
    # small, so it may be ok to simply write co2aq = dic - hco3 + co3.
    co2aq: float = dic - co3 - hco3
    # co2aq: float = dic / (1 + (k1 / hplus) + (k1 * k2 / (hplus ** 2)))
    # omega: float = (ca2 * co3) / ksp

    # ---------- compute critical depth intervals eq after  Boudreau (2010)
    # all depths will be positive to facilitate the use of lookup_tables

    # prevent co3 from becoming zero
    if co3 <= 0:
        co3 = 1e-16

    zsat = int(max((zsat0 * np.log(ca2 * co3 / ksp0)), zsat_min))  # eq2
    if zsat < zsat_min:
        zsat = int(zsat_min)

    zcc = int(zsat0 * np.log(Bm * ca2 / (ksp0 * AD * kc) + ca2 * co3 / ksp0))  # eq3

    # ---- Get fractional areas
    B_AD = Bm / AD

    if zcc > zmax:
        zcc = int(zmax)
    if zcc < zsat_min:
        zcc = zsat_min

    A_z0_zsat = depth_area_table[z0] - depth_area_table[zsat]
    A_zsat_zcc = depth_area_table[zsat] - depth_area_table[zcc]
    A_zcc_zmax = depth_area_table[zcc] - depth_area_table[zmax]

    # ------------------------Calculate Burial Fluxes------------------------------------
    # BCC = (A(zcc, zmax) / AD) * B, eq 7
    BCC = A_zcc_zmax * B_AD

    # BNS = alpha_RD * ((A(z0, zsat) * B) / AD) eq 8
    BNS = alpha * A_z0_zsat * B_AD

    # BDS_under = kc int(zcc,zsat) area' Csat(z,t) - [CO3](t) dz, eq 9a
    diff_co3 = Csat_table[zsat:zcc] - co3
    area_p = area_dz_table[zsat:zcc]

    BDS_under = kc * area_p.dot(diff_co3)
    BDS_resp = alpha * (A_zsat_zcc * B_AD - BDS_under)
    BDS = BDS_under + BDS_resp

    # BPDC =  kc int(zsnow,zcc) area' Csat(z,t) - [CO3](t) dz, eq 10
    if zcc < zsnow:  # zcc cannot
        if zsnow > zmax:  # zsnow cannot exceed ocean depth
            zsnow = zmax

        diff = Csat_table[zcc : int(zsnow)] - co3
        area_p = area_dz_table[zcc : int(zsnow)]

        BPDC = kc * area_p.dot(diff)
        zold = BPDC / (area_dz_table[int(zsnow)] * I_caco3) * dt
        # eq 4 dzsnow/dt = Bpdc(t) / (a'(zsnow(t)) * ICaCO3
        zsnow = zsnow - BPDC / (area_dz_table[int(zsnow)] * I_caco3) * dt

    else:  # zcc > zsnow
        # there is no carbonate below zsnow, so BPDC = 0
        zsnow = zcc
        BPDC = 0

    # BD & F_burial
    BD: float = BDS + BCC + BNS + BPDC
    Fburial = Bm - BD
    Fburial_l = Fburial * input_data[1][i - 1] / input_data[0][i - 1]
    diss = (Bm - Fburial) * dt  # dissolution flux
    diss12 = (B12 - Fburial_l) * dt  # dissolution flux light isotope

    # # print("{Fburial}.format(")
    # print(Bm)
    # print(Fburial)
    # print(diss)
    # print()
    # # print('df ={:.2e}\n'.format(diss/dt))

    """ Now that the fluxes are known we need to update the reservoirs.
    The concentration in the in the DIC (and TA) of this box are
    DIC.m[i] + Export Flux - Burial Flux, where the isotope ratio
    the Export flux is determined by the overlying box, and the isotope ratio
    of the burial flux is determined by the isotope ratio of this box
    """

    # Update DIC in the deep box
    input_data[0][i] = input_data[0][i] + diss  # DIC
    input_data[1][i] = input_data[1][i] + diss12  # 12C
    input_data[2][i] = input_data[0][i] / v  # DIC concentration

    # Update TA in deep box
    input_data[3][i] = input_data[3][i] + 2 * diss  # TA
    input_data[4][i] = input_data[3][i] / v  # TA concentration

    # copy results into datafields
    vr_data[0][i] = hplus  # 0
    vr_data[1][i] = ca  # 1
    vr_data[2][i] = hco3  # 2
    vr_data[3][i] = co3  # 3
    vr_data[4][i] = co2aq  # 4
    vr_data[5][i] = zsat  # 5
    vr_data[6][i] = zcc  # 6
    vr_data[7][i] = zsnow  # 7
    vr_data[8][i] = Fburial  # 8
    vr_data[9][i] = Fburial_l  # 9
    vr_data[10][i] = diss / dt  # 9
    vr_data[11][i] = Bm  # 9
