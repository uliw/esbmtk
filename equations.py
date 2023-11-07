from __future__ import annotations


from numpy import array as npa
from esbmtk import carbonate_system_1_ode, carbonate_system_2_ode
from esbmtk import gas_exchange_ode, gas_exchange_ode_with_isotopes

class setup_ode():
    '''Class stub to enable state in the equation system passed to ODEINT
    '''
    

    def __init__(self, M: Model)->None:
        ''' Use this method to initialize all variables that require the state
            t-1
        '''

    def eqs(self, t, R: list, M: Model, area_table, area_dz_table, Csat_table) -> list:
        '''Auto generated esbmtk equations do not edit
        '''

 


# ---------------- write computed reservoir equations -------- #
# that do not depend on fluxes

# ---------------- write all flux equations ------------------- #
        M_C_weathering_to_sb_PO4_river__F = M.C_weathering_to_sb_PO4_river._F.rate + M.CR(t)[0]  # Signal
        M_downwelling_PO4__F = M.downwelling_PO4.scale * R[0]
        M_upwelling_PO4__F = M.upwelling_PO4.scale * R[1]
        M_C_sb_to_db_PO4_primary_production__F = M.C_sb_to_db_PO4_primary_production.scale * R[0]
        M_C_db_to_burial_PO4_burial__F = M.C_db_to_burial_PO4_burial.scale * M_C_sb_to_db_PO4_primary_production__F

# ---------------- write computed reservoir equations -------- #
# that do depend on fluxes

# ---------------- write input only reservoir equations -------- #

# ---------------- write regular reservoir equations ------------ #
        dCdt_M_sb = (
            + M_C_weathering_to_sb_PO4_river__F
            - M_downwelling_PO4__F
            + M_upwelling_PO4__F
            - M_C_sb_to_db_PO4_primary_production__F
        )/2.9999999999999996e+19

        dCdt_M_db = (
            + M_downwelling_PO4__F
            - M_upwelling_PO4__F
            + M_C_sb_to_db_PO4_primary_production__F
            - M_C_db_to_burial_PO4_burial__F
        )/9.999999999999999e+20


# ---------------- write isotope reservoir equations ------------ #

# ---------------- bits and pieces --------------------------- #
        return [
            dCdt_M_sb,  # 0
            dCdt_M_db,  # 1
        ]
