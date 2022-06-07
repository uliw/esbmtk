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
from esbmtk import Reservoir, Connect
from typing import Union


def decorate_con_w_signal(flux_func):
    def inner(*args):
        """decorator to add both
        the original connection terms
        and the associated signal terms"""
        flux_func(*args)
        eq_terms = args[0]
        connection = args[1]
        if connection.signal != "None":
            f_interp = lambda t: connection.signal(t)[0]
            eq_terms.res_flux_filler((f_interp, "t"), connection)

    return inner


class EQ_Terms:
    """terms of ode equations"""

    def __init__(self, M):
        self.res_flux = {}  # dict that holds variables, reservoirse
        # and in/out-flux (keys are int [res ids])
        self.M = M

        for res in self.M.lor:
            # influx, outflux, concentration variable for reservoir, reservoir
            var_str = f"x_{id(res)}"
            self.res_flux[id(res)] = ([], [], var_str, res)

        for connection in M.loc:
            # I need to further split these if blocks into a separate function
            if (
                connection.ctype == "scale_with_concentration"
                or connection.ctype == "regular"
            ):
                if connection.rate != "None":
                    self.scale_w_concentration_or_regular(connection)
                else:
                    # if the rate is not given, it assumes a variable rate
                    self.variable_rate(connection)

            elif connection.ctype == "scale_with_mass":
                self.scale_with_mass(connection)

            elif connection.ctype == "scale_with_flux":
                if isinstance(M.burial.ref_flux.parent, type(connection)):
                    self.scale_with_flux(connection)
                else:
                    raise TypeError("Please input proper ref_flux for scale_with_flux")
            else:
                raise TypeError("Please check all connection types inputted")

    # the methods have the "factor" parameter that is there in order
    # to be able to add terms to the equations that are some already existing
    # terms in that equation from other connections but scaled by a factor
    # the default is 1

    # if the equation term is scaled off of another term, then the term
    # generating methods the factor_scaling_connection is set to be equal the
    # the method that is generating this scaled term otherwise,
    # factor_scaling_connection parameter is set at None by default

    @decorate_con_w_signal
    def scale_w_concentration_or_regular(
        self,
        connection: Connect,
        factor: float = 1,
        factor_scaling_connection: Union[None, Connect] = None,
    ):
        """this method is used when we work with regular or
        scale_with_concentration connections

        no variables involved"""
        if factor_scaling_connection is None:
            self.res_flux_filler(connection.rate * factor, connection)
        else:
            self.res_flux_filler(connection.rate * factor, factor_scaling_connection)

    def scale_with_flux(self, connection: Connect):
        """given a reference flux, we calculate the
        new flux as a percentage of that other flux
        we don't need a factor parameter here as this method
        itself is using the factor parameters from other methods"""

        original_flux = connection.ref_flux.parent
        factor = connection.scale

        if (
            original_flux.ctype == "scale_with_concentration"
            or original_flux.ctype == "regular"
        ):
            if original_flux.rate != "None":
                self.scale_w_concentration_or_regular(original_flux, factor, connection)
            else:
                # if the rate is not given, it assumes a variable rate
                self.variable_rate(original_flux, factor, connection)

        elif original_flux.ctype == "scale_with_mass":
            self.scale_with_mass(original_flux, factor, connection)

        else:
            raise TypeError("Please input proper ref_flux connection type")

    @decorate_con_w_signal
    def scale_with_mass(
        self,
        connection: Connect,
        factor: float = 1,
        factor_scaling_connection: Union[None, Connect] = None,
    ):
        """the variable for this equation term is the
        same as for the source variable and the coeff
        is the product of the scale (1/tau) multipled
        by the volume of the source reservoir"""
        coeff: float = connection.source.volume * connection.scale
        var_str: str = self.res_flux[id(connection.source)][2]
        exec(
            f"global func; func = lambda variable, factor={factor}: {coeff}*variable*factor"
        )

        if factor_scaling_connection is None:
            self.res_flux_filler((func, var_str), connection)
        else:
            self.res_flux_filler((func, var_str), factor_scaling_connection)

    @decorate_con_w_signal
    def variable_rate(
        self,
        connection: Connect,
        factor: Union[float, int] = 1,
        factor_scaling_connection: Union[None, Connect] = None,
    ):
        """if the connection has a specified rate then
        use this one to get the units matching"""

        scale_mag: Union[int, float] = connection.scale
        var_str: str = self.res_flux[id(connection.source)][2]
        exec(
            f"global func; func = lambda variable, factor={factor}: {scale_mag}*variable*factor"
        )

        if factor_scaling_connection is None:
            self.res_flux_filler((func, var_str), connection)
        else:
            self.res_flux_filler((func, var_str), factor_scaling_connection)

    def res_flux_filler(self, input, connection: Connect):
        """Fills the influx and outflux lists in the res_flux dictionary
        filling the res_flux dict depending on the source and sink of the flux

        - if both source and sink are reservoirs the input (which is the
          equation term in form of a Sympy object) will be filled into both
          the influx and outflux lists of corresponding reservoirs

        - if only the source is a reservoir, then the input will be
          filled into the source list of the corresponding reservoir

        - if only the sink is a reservoir, then the input will be filled into
          the sink list of the corresponding reservoir"""

        if isinstance(connection.source, Reservoir) and isinstance(
            connection.sink, Reservoir
        ):

            if isinstance(input, tuple):
                """If a function appears in
                many equations, to denote that,
                we use a list and not a tuple"""
                input = list(input)

            # outflowing and inflow lists
            self.res_flux[id(connection.source)][1].append(
                input
            )  # outflowing is the first list
            self.res_flux[id(connection.sink)][0].append(
                input
            )  # inflowing is the second list

        elif isinstance(
            connection.source, Reservoir
        ):  # and not isinstance(connection.sink, Reservoir)
            # outflowing list
            self.res_flux[id(connection.source)][1].append(input)

        else:
            # inflowing list
            self.res_flux[id(connection.sink)][0].append(input)

        return self.res_flux
