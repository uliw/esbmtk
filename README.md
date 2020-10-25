
# Table of Contents

1.  [ESBMTK - An Earth-sciences box modeling toolkit](#orgf878683)
2.  [News](#org6ed9f2a)
3.  [Installation](#org4a81af2)
4.  [Documentation](#org8eef6b5)
5.  [License](#orgdb1fa68)


<a id="orgf878683"></a>

# ESBMTK - An Earth-sciences box modeling toolkit

ESBMTK is python library which aims to simplify typical box modeling
projects the in Earth-Sciences. The focus of this project is to make
box modeling more approachable for classroom teaching. So performance
and scalability are not great.

At present, it will calculate masses/concentrations in reservoirs and
fluxes including isotope ratios. It provides a variety of classes
which allow the creation and manipulation of input signals, and the
generation of graphical result. There is however no support for
chemical reactions (including equilibrium reactions).


<a id="org6ed9f2a"></a>

# News

-   Oct. 25<sup>th</sup>, Initial release on github.


<a id="org4a81af2"></a>

# Installation

ESBMTK relies on the following python versions and libraries

-   python > 3.5
-   matplotlib
-   numpy
-   pandas
-   logging
-   nptyping

please install the above with pip or conda etc. 

At present, there is no ready made installation routine. Rather
download the `esbmtk.py` library into your local working library using
one of the following commands

-   <https://github.com/uliw/esbmtk/archive/main.zip>
-   <https://github.com/uliw/esbmtk.git>
-   git@github.com:uliw/esbmtk.git


<a id="org8eef6b5"></a>

# Documentation

See the documentation folder, [[<esbmtk::Documentation/ESBMTK-Quick-Start_Guide.org::c394>][specifically the quickstart guide] in the Documentation folder


<a id="orgdb1fa68"></a>

# License

ESBMTK: A general purpose Earth Science box model toolkit
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

