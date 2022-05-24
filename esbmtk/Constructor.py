from esbmtk import Q_, Model, Reservoir, Connect, Source, Sink
from sympy import Symbol
import numpy as np
from typing import Union
from Equation_Terms import EQ_Terms


class Constructor:
    '''constructs the equation from the terms
    in res_flux dictionary'''

    def __init__(self, terms:EQ_Terms):
        self.terms = terms
        # tuple of resevoir ids (also res_flux keys)
        self.var_ids = tuple(terms.res_flux.keys())
        #holds the equations
        self.eqs = []
        self.ivp = []
        vars = []
        for var_id in self.var_ids:
            #reservoir id
            res_id = terms.res_flux[var_id]
            # list of variables
            vars.append(self.terms.var_dict[res_id[2]])
            # volume of the given reservoir
            vol = res_id[3].volume
            # creates the IVP
            self.ivp.append(res_id[3].concentration)

            # this is the equation
            eq = (sum(res_id[0])-sum(res_id[1]))/vol
            self.eqs.append(eq)
        # turns the above list of variables into a tuple
        self.vars = tuple(vars)