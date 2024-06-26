"""
     esbmtk: A general purpose Earth Science box model toolkit
     Copyright (C), 2020 Ulrich G. Wortmann

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

import esbmtk

from esbmtk import (
    Model,
    Reservoir,
    ConnectionProperties,
    SourceProperties,
    SinkProperties,
    Q_,
)

#def define_model(F_w: str, tau: Q_, thc: Q_, F_b: Q_) -> tuple[float, float, float, float, Model]:
def define_model(F_w: str, tau: Q_, thc: Q_, F_b: Q_) -> tuple[float, float]:
    """
    Runs the ESBMTK model with the given parameters and returns the final oxygen concentrations.

    :param F_w: Weathering flux in Gmol/year
    :type F_w: str
    :param tau: Residence time in years
    :type tau: Q_
    :param thc: THC in Sverdrup (Sv)
    :type thc: Q_
    :param F_b: Burial flux in fraction
    :type F_b: Q_
    :return: Final oxygen concentrations in S_b and D_b
    :rtype: tuple[float, float]
    """
    
    # Basic model parameters
    M = Model(
        stop="6 Myr",
        timestep="1 kyr",
        element=["Phosphor", "Oxygen"],
    )

    # Parameters

    SourceProperties(
        name="weathering",
        species=[M.PO4],
    )

    SinkProperties(
        name="burial",
        species=[M.PO4],
    )

    # Reservoir Definitions
    Reservoir(
        name="S_b",
        volume="3E16 m**3",
        concentration={M.PO4: "0 umol/l", M.O2: "300 umol/l"},  # Initial O2 set to 300 umol/l
    )

    Reservoir(
        name="D_b",
        volume="100E16 m**3",
        concentration={M.PO4: "0 umol/l", M.O2: "100 umol/l"},  # Initial O2 set to 100 umol/l
    )

    # Connection Properties (Fluxes)
    ConnectionProperties(
        source=M.weathering,
        sink=M.S_b,
        rate=str(F_w),  # Convert F_w to string
        id="river",
        ctype="regular",
    )

    ConnectionProperties(
        source=M.S_b,
        sink=M.D_b,
        ctype="scale_with_concentration",
        scale=thc,
        id="downwelling",
        species=[M.O2, M.PO4]
    )

    ConnectionProperties(
        source=M.D_b,
        sink=M.S_b,
        ctype="scale_with_concentration",
        scale=thc,
        id="upwelling",
        species=[M.O2, M.PO4]
    )

    ConnectionProperties(
        source=M.S_b,
        sink=M.D_b,
        ctype="scale_with_concentration",
        scale=M.S_b.volume / tau,
        id="primary_production",
        species=[M.PO4],
    )

    ConnectionProperties(
        source=M.D_b,
        sink=M.burial,
        ctype="scale_with_flux",
        ref_flux=M.flux_summary(filter_by="primary_production", return_list=True)[0],
        scale=F_b,
        id="burial",
        species=[M.PO4],
    )

    ConnectionProperties(
        source=M.D_b,
        sink=M.S_b,
        ctype="scale_with_flux",
        ref_flux=M.flux_summary(filter_by="primary_production", return_list=True)[0],
        scale=(1 - F_b) * 138,
        id="O2upwelling",
        species=[M.O2],
    )
    M.read_state()
    M.run()
    
    S_b_O2 = M.S_b.O2.c[-1] * 1E6
    D_b_O2 = M.D_b.O2.c[-1] * 1E6
    S_b_PO4 = M.S_b.PO4.c[-1] * 1E6
    D_b_PO4 = M.D_b.PO4.c[-1] * 1E6

    return S_b_O2, D_b_O2, S_b_PO4, D_b_PO4, M
