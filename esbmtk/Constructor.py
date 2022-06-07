"""
 esbmtk: A general purpose Earth Science box model toolkit
 Copyright (C), 2022 Ruben Navasardyan

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import annotations
from esbmtk import EQ_Terms


class Constructor:
    """constructs the equation from the terms
    in res_flux dictionary"""

    def __init__(self, terms: EQ_Terms):
        """Initializes the 'inftrastructure' needed for
        system creation and evaluation such as:
        - variables
        - order of equation
        - volume data in order
        - ivp"""

        self.terms: EQ_Terms = terms
        # tuple of resevoir ids (also res_flux keys)
        self.var_ids = tuple(terms.res_flux.keys())
        # holds the equations
        self.res_flux: dict = terms.res_flux
        self.res_vol_dict = {}
        self.ivp = []
        vars = []
        for res_id in self.var_ids:
            # reservoir id
            res_info = self.res_flux[res_id]
            # list of variables
            vars.append(res_info[2])
            # volume of the given reservoir
            vol = res_info[3].volume
            self.res_vol_dict[res_id] = vol
            # creates the IVP
            self.ivp.append(res_info[3].concentration)
        # turns the above list of variables into a tuple
        self.vars = tuple(vars)

    def sum_vol(self, inp_dict: dict):
        """evaluates the system of the form
        dx/dt = (influx - outflux)/volume"""

        out_list = []
        repeated_func_dict = {}  # holds values of functions that are
        # present in multiple equations (to avoid recalculation)
        for var in self.vars:
            equation_total = 0
            res_id: int = int(var[2:])
            res_info: tuple = self.res_flux[res_id]

            for term in res_info[0]:
                """adding terms to the equation"""
                if isinstance(term, tuple):
                    """if the term appears in one equation"""
                    func, func_variable = term[0], term[1]
                    function_val: float = func(inp_dict[func_variable])
                    equation_total += function_val

                elif isinstance(term, list):
                    """if the term appears in many equations
                    check if its value has already been calculated
                    and stored in a dict, if not do it"""
                    func, func_variable = term[0], term[1]
                    func_id: int = id(func)
                    if func_id in repeated_func_dict.keys():
                        equation_total += repeated_func_dict[func_id]
                    else:
                        function_val: float = func(inp_dict[func_variable])
                        repeated_func_dict[func_id] = function_val
                        equation_total += function_val
                else:
                    """if the term is a constant"""
                    equation_total += term

            for term in res_info[1]:
                """subtracting terms from the equation"""
                if isinstance(term, tuple):
                    """if the term appears in one equation"""
                    func, func_variable = term[0], term[1]
                    function_val: float = func(inp_dict[func_variable])
                    equation_total -= function_val

                elif isinstance(term, list):
                    """if the term appears in many equations
                    check if its value has already been calculated
                    and stored in a dict, if not do it"""
                    func, func_variable = term[0], term[1]
                    func_id: int = id(func)
                    if func_id in repeated_func_dict.keys():
                        equation_total -= repeated_func_dict[func_id]
                    else:
                        function_val: float = func(inp_dict[func_variable])
                        repeated_func_dict[func_id] = function_val
                        equation_total -= function_val
                else:
                    """if the term is a constant"""
                    equation_total -= term

            vol: float = self.res_vol_dict[res_id]
            out_list.append(equation_total / vol)
        return out_list
