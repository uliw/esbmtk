- [ESBMTK - An Earth-sciences box modeling toolkit](#org88ac966)
- [News](#org828deb0)
- [Contributing](#orgbc7ac65)
- [Installation](#orgf7672da)
- [Documentation](#org9cc1016)
- [Todo](#org273f938)
- [License](#org172d8fa)


<a id="org88ac966"></a>

# ESBMTK - An Earth-sciences box modeling toolkit

ESBMTK is a python library that aims to simplify typical box modeling projects in the Earth-Sciences. The general focus is to make box modeling more approachable for classroom teaching. At present the code is under active development and the API changes frequently.


<a id="org828deb0"></a>

# News

-   Sept. 18<sup>th</sup>, 2023 v 0.10.0.0 This is a breaking change. Remineralization and photosynthesis must be implemented via functions, rather than transport connections. CS1 and CS2 are retired, and replaced by photosynthesis, organic-matter remineralization and carbonate-dissolution functions.

-   July 28<sup>th</sup>, 2023, v 0.9.0.1 The ODEPACk backend is now fully functional, and the basic API is more or less stable.

-   Nov. 11<sup>th</sup> 2022, v 0.9.0.0 Moved to odepack based backend. Removed now defunct code. The odepack backend does not yet support isotope calculations.

-   0.8.0.0
    -   Cleanup of naming scheme which is now strictly hierarchical.
    -   Bulk connection dictionaries now have to be specified as `source_to_sink` instead of `source2sink`.
    -   The connection naming scheme has been revamped. Please see `esbmtk.connect.__set_name__()` documentation for details.
    -   Model concentration units must now match 'mole/liter' or 'mol/kg'. Concentrations can still be specified as `mmol/l` or `mmol/kg`, but model output will be in mole/liter or kg. At present, the model does not provide for the automatic conversion of mol/l to mol/kg. Thus you must specify units in a consistent way.
    -   The SeawaterConstants class now always returns values as mol/kg solution. Caveat Emptor.
    -   The SeawaterConstants class no longer accepts the 'model' keyword
    -   All of his will break existing models.
    -   Models assume by default that they deal with ideal water, i.e., where the density equals one. To work with seawater, you must set `ideal_water=False`. In that case, you should also set the `concentration_unit` keyword to `'mol/kg'` (solution).
    -   Several classes now require the "register" keyword. You may need to fix your code accordingly

-   The flux and connection summary methods can be filtered by more than one keyword. Provide a filter string in the following format `"keyword_1 keyword_2` and it will only return results that match both keywords.
-   Removed the dependency on the nptyping and number libraries

-   0.7.3.9 Moved to setuptools build system. Lost of code fixes wrt isotope calculations, minor fixes in the carbonate module.

-   March 2<sup>nd</sup> 0.7.3.4 `Flux_summary` now supports an `exclude` keyword. Hot fixed an error in the gas exchange code, which affected the total mass of atmosphere calculations. For the time being, the mass of the atmosphere is treated as constant.

-   0.7.3.0 Flux data is no longer kept by default. This results in huge memory savings. esbmtk now requires python 3.9 or higher, and also depends on `os` and `psutil`. the scale with flux process now uses the `ref_flux` keyword instead of `ref_reservoirs`. Models must adapt their scripts accordingly. esbmtk objects no longer provide delta values by default. Rather they need to be calculated in the post-processing step via `M.get_delta_values()`. The `f_0` keyword in the weathering connection is now called `rate`. Using the old keyword will result in a unit error.

-   January 8<sup>th</sup> 0.7.2.2 Fixed several isotope calculation regressions. Added 31 Unit tests.

Older releases are mentioned in the Release History.


<a id="orgbc7ac65"></a>

# Contributing

Don't be shy. Contributing is as easy as finding bugs by using the code, or maybe you want to add a new process code? If you have plenty of time to spare, ESMBTK could use a solver for stiff problems, or a graphical interface ;-) See the todo section for ideas.


<a id="orgf7672da"></a>

# Installation

ESBMTK relies on the following python versions and libraries

-   python >= 3.9
-   matplotlib
-   numpy
-   pandas
-   typing
-   pint

If you work with conda, it is recommended to install the above via conda. If you work with pip, the installer should install these libraries automatically. ESBMTK itself can be installed with pip

-   pip install esbmtk


<a id="org9cc1016"></a>

# Documentation

The documentation is woefully out of date, at present, your best bet is the code itself

The API documentation is available at <https://uliw.github.io/esbmtk/esbmtk/index.html>


<a id="org273f938"></a>

# Todo

-   expand the documentation
-   provide more examples
-   do more testing


<a id="org172d8fa"></a>

# License

ESBMTK: A general purpose Earth Science box model toolkit Copyright (C), 2020 Ulrich G. Wortmann

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.