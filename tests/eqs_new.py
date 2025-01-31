from esbmtk import gas_exchange


# @njit(fastmath=True)
def eqs(t, R, M, gpt, toc, area_table, area_dz_table, Csat_table, C, F):
    """Calculate dCdt for each reservoir.

    t = time vector
    R = initial conditions for each reservoir
    M = model handle. This is currently required for signals
    gpt = tuple of global constants (is this still used?)
    toc = is a tuple of containing  constants for each external function
    area_table = lookuptable used by carbonate_system_2
    area_dz_table = lookuptable used by carbonate_system_2
    Csat_table = lookuptable used by carbonate_system_2
    C = Coefficient Matrix
    F = flux vector

    Returns: dCdt as numpy array

    Reservoir and flux name lookup: M.lor[idx] and M.lof[idx]
    """

    # ---------------- write computed reservoir equations -------- #
    # that do not depend on fluxes
    F[0], F[1] = gas_exchange(
        (R[0], R[1]),
        (R[2], R[3]),
        (R[2], R[3]),
        gpt[0],
    )

    # ---------------- write all flux equations ------------------- #

    # ---------------- write computed reservoir equations -------- #
    # that do depend on fluxes

    # ---------------- write input only reservoir equations -------- #

    # ---------------- calculate concentration change ------------ #
    return C.dot(F)
