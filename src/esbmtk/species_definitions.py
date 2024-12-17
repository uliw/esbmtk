"""
esbmtk: A general purpose Earth Science box model toolkit Copyright
(C), 2020 Ulrich G.  Wortmann

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at
your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import annotations
import typing as tp
import numpy as np
import numpy.typing as npt
from .esbmtk import ElementProperties, SpeciesProperties

# declare numpy types
NDArrayFloat = npt.NDArray[np.float64]
np.set_printoptions(precision=4)
if tp.TYPE_CHECKING:
    from .esbmtk import Model


def Carbon(model):
    """Some often used definitions"""

    eh = ElementProperties(
        name="Carbon",  # ElementProperties Name
        model=model,  # Model handle
        mass_unit="mol",  # base mass unit
        li_label="C^{12}$S",  # Name of light isotope
        hi_label="C^{13}$S",  # Name of heavy isotope
        d_label=r"$\delta^{13}$C",  # Name of isotope delta
        d_scale="mUr VPDB",  # Isotope scale. End of plot labels
        r=0.0112372,  # VPDB C13/C12 ratio
        reference="https://www-pub.iaea.org/MTCD/publications/PDF/te_825_prn.pdf",
        register=model,
    )

    # add species
    SpeciesProperties(
        name="CO2", element=eh, display_as=r"CO$_2$", register=eh, m_weight=44.0095
    )  # Name & element handle
    SpeciesProperties(name="DIC", element=eh, register=eh, display_as="DIC")
    SpeciesProperties(name="OM", element=eh, register=eh)
    # Particulate OM
    SpeciesProperties(name="POM", element=eh, register=eh, flux_only=True)
    SpeciesProperties(name="PIC", element=eh, register=eh, flux_only=True)
    SpeciesProperties(name="DOC", element=eh, register=eh)
    SpeciesProperties(
        name="CaCO3", element=eh, display_as=r"CaCO$_3$", register=eh, m_weight=100.0869
    )
    SpeciesProperties(
        name="HCO3", element=eh, display_as=r"HCO$_3^-$", register=eh, m_weight=61.01684
    )
    SpeciesProperties(
        name="CO3", element=eh, display_as="CO$_3^{2-}$", register=eh, m_weight=60.0089
    )
    SpeciesProperties(name="D14C", element=eh, display_as="$\Delta^{14}$C", register=eh)
    SpeciesProperties(name="C", element=eh, register=eh, m_weight=12.0107)
    SpeciesProperties(name="CO2aq", element=eh, register=eh, scale_to="umol")
    SpeciesProperties(name="ALK", element=eh, register=eh)  # Alkalinity
    SpeciesProperties(name="CALK", element=eh, register=eh)  # Carbonate Alkalinity
    SpeciesProperties(name="CA", element=eh, register=eh)
    SpeciesProperties(name="TALK", element=eh, register=eh)  # Total Alkalinity
    SpeciesProperties(name="TA", element=eh, register=eh)
    SpeciesProperties(name="PALK", element=eh, register=eh)  # Practical Alkalinity
    SpeciesProperties(name="PA", element=eh, register=eh)


def Sulfur(model):
    eh = ElementProperties(
        name="Sulfur",
        model=model,  # model handle
        mass_unit="mol",  # base mass unit
        li_label="$^{32}$S",  # Name of light isotope
        hi_label="$^{34}$S",  # Name of heavy isotope
        d_label=r"$\delta^{34}$S",  # Name of isotope delta
        d_scale="mUr VCDT",  # Isotope scale. End of plot labels
        r=0.044162589,  # isotopic abundance ratio for species
        register=model,
    )

    # add species
    SpeciesProperties(
        name="SO4",
        element=eh,
        display_as=r"SO$_{4}^{2-}$",
        register=eh,
        m_weight=96.0626,
    )
    SpeciesProperties(
        name="SO3", element=eh, display_as=r"SO$_{3}$", register=eh, m_weight=80.0632
    )
    SpeciesProperties(
        name="SO2", element=eh, display_as=r"SO$_{2$}", register=eh, m_weight=64.0638
    )
    SpeciesProperties(
        name="HS", element=eh, display_as=r"HS$^-$", register=eh, m_weight=33.07294
    )
    SpeciesProperties(
        name="H2S", element=eh, display_as=r"H$_{2}$S", register=eh, m_weight=34.08088
    )
    SpeciesProperties(name="FeS", element=eh, register=eh, m_weight=87.91)
    SpeciesProperties(
        name="FeS2", element=eh, display_as=r"FeS$_{2}$", register=eh, m_weight=119.975
    )
    SpeciesProperties(name="S0", element=eh, register=eh, m_weight=32.065)
    SpeciesProperties(name="S", element=eh, register=eh, m_weight=32.065)
    SpeciesProperties(
        name="S2minus", element=eh, display_as=r"S$^{2-}$", register=eh, m_weight=64.13
    )


def Hydrogen(model):
    eh = ElementProperties(
        name="Hydrogen",
        model=model,  # model handle
        mass_unit="milli",  # base mass unit
        li_label="$^{1$}H",  # Name of light isotope
        hi_label="$^{2}$H",  # Name of heavy isotope
        d_label=r"$\delta^{2}$D",  # Name of isotope delta
        d_scale="mUr VSMOV",  # Isotope scale. End of plot labels  # needs verification
        r=155.601e-6,
        reference="After Groenig, 2000, Tab 40.1",
        register=model,
    )

    # add species
    SpeciesProperties(
        name="Hplus", element=eh, display_as=r"$H^+$", register=eh, logdata=True
    )  # Name & element handle
    SpeciesProperties(
        name="H2O", element=eh, display_as=r"H$_{2}$O", register=eh
    )  # Name & element handle
    SpeciesProperties(
        name="H", element=eh, display_as=r"$H^+$", register=eh
    )  # Name & element handle
    SpeciesProperties(
        name="pH", element=eh, display_as=r"$pH$", register=eh
    )  # Name & element handle


def Oxygen(model: Model) -> None:
    """Common Properties of Oxygen

    Parameters
    ----------
    model : Model
        Model instance

    """
    eh = ElementProperties(
        name="Oxygen",
        model=model,  # model handle
        mass_unit="mol",  # base mass unit
        li_label="$^{16$}O",  # Name of light isotope
        hi_label="$^{18}$O",  # Name of heavy isotope
        d_label=r"$\delta^{18}$O",  # Name of isotope delta
        d_scale="mUr VSMOV",  # Isotope scale. End of plot labels  # needs verification
        r=2005.201e-6,
        reference="https://nucleus.iaea.org/rpst/documents/vsmow_slap.pdf",
        register=model,
    )

    # add species
    SpeciesProperties(name="O", element=eh, display_as="O")
    # Name & element handle
    SpeciesProperties(
        name="O2",
        element=eh,
        display_as=r"O$_{2}$",
        scale_to="umol",
    )  # Name & element handle
    SpeciesProperties(
        name="OH", element=eh, display_as=r"OH$^{-}$"
    )  # Name & element handle


def Phosphor(model):
    eh = ElementProperties(
        name="Phosphor",
        model=model,  # model handle
        mass_unit="mol",  # base mass unit
        li_label="LI",  # Name of light isotope
        hi_label="HL",  # Name of heavy isotope
        d_label="D",  # Name of isotope delta
        d_scale="DS",  # Isotope scale. End of plot labels  # needs verification
        r=1,  # isotopic abundance ratio for species # needs verification
        register=model,
    )

    # add species
    SpeciesProperties(
        name="PO4",
        element=eh,
        display_as=r"PO$_{4}$",
        register=eh,
        m_weight=94.971362,
        scale_to="umol",
    )  # Name & element handle
    SpeciesProperties(
        name="P",
        element=eh,
        display_as=r"P",
        register=eh,
        m_weight=30.973762,
        scale_to="umol",
    )  # Name & element handle


def Nitrogen(model):
    eh = ElementProperties(
        name="Nitrogen",
        model=model,  # model handle
        mass_unit="mol",  # base mass unit
        li_label=r"$^{15$}N",  # Name of light isotope
        hi_label=r"$^{14$}N",  # Name of heavy isotope
        d_label=r"$\delta^{15}$N",  # Name of isotope delta
        d_scale="mUr Air",  # Isotope scale. End of plot labels  # needs verification
        r=3676.5e-6,  # isotopic abundance ratio for species # needs verification
        register=model,
    )

    # add species
    SpeciesProperties(name="N", element=eh, display_as=r"N")
    SpeciesProperties(
        name="N2", element=eh, display_as=r"N$_{2}$"
    )  # Name & element handle
    SpeciesProperties(
        name="Nox", element=eh, display_as=r"Nox"
    )  # Name & element handle
    SpeciesProperties(
        name="NH4", element=eh, display_as=r"NH$_{4}^{+}$"
    )  # Name & element handle
    SpeciesProperties(
        name="NH3", element=eh, display_as=r"NH$_{3}$"
    )  # Name & element handle


def Boron(model):
    eh = ElementProperties(
        name="Boron",
        model=model,  # model handle
        mass_unit="mmol",  # base mass unit
        li_label=r"$^{11$}B",  # Name of light isotope
        hi_label=r"$^{10$}B",  # Name of heavy isotope
        d_label=r"$\delta{11}B",  # Name of isotope delta
        d_scale="mUr SRM951",  # Isotope scale. End of plot labels  # needs verification
        r=0.26888,  # isotopic abundance ratio for species # needs verification
        register=model,
    )

    # add species
    SpeciesProperties(name="B", element=eh, display_as=r"B")  # Name & element handle
    SpeciesProperties(name="BOH", element=eh, display_as=r"B")  # Name & element handle
    SpeciesProperties(name="BOH3", element=eh, display_as=r"B(OH)$_{3}$")  # Boric Acid
    SpeciesProperties(name="BOH4", element=eh, display_as=r"B(OH)$_{4}^{-}$")  # Borate


def misc_variables(model):
    eh = ElementProperties(
        name="misc_variables",
        model=model,  # model handle
        mass_unit="m",  # base mass unit
        register=model,
    )
    # add species
    SpeciesProperties(name="zsnow", element=eh, display_as="zsnow", stype="length")
    SpeciesProperties(name="zcc", element=eh, display_as="zcc", stype="length")
    SpeciesProperties(name="zsat", element=eh, display_as="zcc", stype="length")
    SpeciesProperties(name="Fburial", element=eh, display_as="Fburial")
    SpeciesProperties(name="Fdiss", element=eh, display_as="Fdiss")
    # Name & element handle
