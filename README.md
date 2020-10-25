
# Table of Contents

1.  [ESBMTK - An Earth-sciences box modeling toolkit](#org9cc91bb)
2.  [News](#orgf36df6e)
3.  [Installation](#orge18d954)
4.  [Documentation](#orgb271d0f)
5.  [License](#org68a68bd)


<a id="org9cc91bb"></a>

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


<a id="orgf36df6e"></a>

# News

-   Oct. 25<sup>th</sup>, Initial release on github.


<a id="orge18d954"></a>

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


<a id="orgb271d0f"></a>

# Documentation

See the documentation folder, [specifically the quickstart guide.](esbmtk::Documentation/ESBMTK-Quick-Start_Guide.org::c394)


<a id="org68a68bd"></a>

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

