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
